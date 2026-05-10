"""
ROS (Rest-of-Season) Projection Blending
=========================================

Moved from ``daily_ros/ros_projections.py`` to
``value_determination/pipelines/ros.py``.

Blends pre-season predictions with actual current-season performance
using Bayesian shrinkage on *base components* (not derived stats).

Architecture:
    Batters  — blend 8 base components (K%, BB%, HBP%, ISO, BABIP, HR/FB,
               GB%, LD%), then recompose AVG/SLG/OBP/wOBA/wRC+ via
               ``stat_composition.compose_from_df``.
    Pitchers — blend 8 base components (K%, BB%, HBP%, BABIP, HR/FB, GB%,
               FB%, LD%), then reconstruct FIP/ERA/SIERA/K-rates.
    Fielding — blend preseason sc_total_runs/150 with actual Fld runs.
    Baserunning — blend preseason BsR projections with actual BsR.

Park factors:
    Actual stats contain park effects; preseason predictions are park-neutral.
    Before blending, we neutralize actual rate stats so both sides are
    park-neutral.  Park factors are reapplied downstream in WAR calculation.

Data source:
    Uses ``_with_statcast`` CSV variants for richer data (xwOBA, sprint speed,
    barrel rate, Statcast fielding/baserunning columns).
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path

from value_determination.config import Config, logger, CURRENT_YEAR
from core.park_factors import get_park_factor, EXCLUDED_STATS
from core.stat_composition import compose_from_df
from core.marcel_projections import (
    BATTER_BASE_COMPONENTS,
    PITCHER_MARCEL_RATE_STATS,
    _reconstruct_fip,
    _reconstruct_siera,
    _derive_bf_per_ip,
    _FIP_CONSTANT,
    _ERA_FIP_STAB_TBF,
)

# =============================================================================
# Shrinkage parameters
# =============================================================================
BATTER_PRIOR_PA = 400    # PA for pre-season prior to equal actual weight
PITCHER_PRIOR_IP = 80    # IP for pre-season prior to equal actual weight

# Fielding/baserunning priors (in games — ~half season for equal weight)
FIELDING_PRIOR_G = 80
BASERUNNING_PRIOR_PA = 400

# Full-season baselines (for remaining-fraction calculation)
FULL_SEASON_PA = 650
FULL_SEASON_SP_IP = Config.WAR.DEFAULT_SP_IP   # 180
FULL_SEASON_RP_IP = Config.WAR.DEFAULT_RP_IP   # 70
TOTAL_SEASON_GAMES = 162

MLB_API_BASE = 'https://statsapi.mlb.com/api/v1'


# =============================================================================
# Team games played (for ROS remaining-fraction)
# =============================================================================

def fetch_team_games_played(season: int = None) -> dict:
    """Fetch games played per team from MLB Stats API standings.

    Uses the team_info.csv from the roster scraper to map MLB team IDs
    to the abbreviations used throughout the pipeline.

    Returns:
        Dict mapping team abbreviation (e.g. 'NYY') → games_played (int).
        Returns empty dict on failure (callers fall back to PA/IP-based).
    """
    if season is None:
        season = CURRENT_YEAR

    try:
        resp = requests.get(
            f'{MLB_API_BASE}/standings',
            params={'leagueId': '103,104', 'season': str(season)},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # team_id → games_played
        id_to_gp = {}
        for rec in data.get('records', []):
            for t in rec.get('teamRecords', []):
                tid = t.get('team', {}).get('id')
                gp = t.get('gamesPlayed', 0)
                if tid is not None:
                    id_to_gp[tid] = gp

        # Map team_id → abbreviation via team_info.csv
        team_info_path = Config.Paths.DATA_DIR / 'active_roster' / 'team_info.csv'
        if not team_info_path.exists():
            logger.warning(f"team_info.csv not found at {team_info_path}")
            return {}

        team_info = pd.read_csv(team_info_path)
        abbrev_map = dict(zip(team_info['team_id'], team_info['abbreviation']))

        # Also build full-name → abbreviation for FanGraphs-style team codes
        result = {}
        for tid, gp in id_to_gp.items():
            abbrev = abbrev_map.get(tid)
            if abbrev:
                result[abbrev] = gp

        # Add common FanGraphs abbreviation aliases
        fg_aliases = {
            'CWS': 'CHW', 'CHW': 'CWS',
            'AZ': 'ARI', 'ARI': 'AZ',
            'SDP': 'SD', 'SD': 'SDP',
            'SFG': 'SF', 'SF': 'SFG',
            'TBR': 'TB', 'TB': 'TBR',
            'KCR': 'KC', 'KC': 'KCR',
            'WSN': 'WSH', 'WSH': 'WSN',
        }
        for existing, alias in fg_aliases.items():
            if existing in result and alias not in result:
                result[alias] = result[existing]

        logger.info(
            f"Fetched team games played for {len(result)} teams "
            f"(season={season}, range={min(result.values())}-{max(result.values())} GP)"
        )
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch team games played: {e} — will fall back to PA/IP-based remaining")
        return {}

# =============================================================================
# Stats that get park-neutralized before blending
# =============================================================================
# Batter components affected by park: ISO, BABIP (excluded by park_factors),
# plus derived rate stats.  K%, BB%, HBP%, HR/FB, GB%, LD% are pitcher/batter
# matchup dependent, not park-dependent, so they stay un-adjusted.
BATTER_PARK_ADJUSTABLE = ['ISO']  # BABIP excluded by park_factors module
PITCHER_PARK_ADJUSTABLE = []       # pitcher rates are all sequencing/skill — not park-adjusted


def _safe_float(value):
    """Convert a value to float, handling percentage strings and NaN."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        is_pct = '%' in value
        cleaned = value.strip().rstrip('%').strip()
        try:
            v = float(cleaned)
            return v / 100.0 if is_pct else v
        except ValueError:
            return None
    return float(value)


def load_current_season_actuals(current_year=None):
    """Load actual batting and pitching stats for the current season.

    Uses the ``_with_statcast`` CSV variants for richer data (Statcast
    fielding, baserunning, xwOBA, sprint speed).

    Reads from the historic CSV files (which Phase 1 merge keeps updated).
    Falls back to ``current_year - 1`` if the target year has no data yet.

    Returns:
        Tuple (batting_df, pitching_df, actual_year)
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    hist_dir = Config.Paths.HISTORIC_MLB_DIR

    batting = pd.read_csv(
        hist_dir / 'mlb_batting_data_1950_2025_with_statcast.csv', low_memory=False,
    )
    pitching = pd.read_csv(
        hist_dir / 'mlb_pitching_data_1950_2025_with_statcast.csv', low_memory=False,
    )

    # Find current-season rows; fall back to prior year if absent
    actual_year = current_year
    bat_current = batting[batting['Season'] == current_year]
    if bat_current.empty:
        actual_year = current_year - 1
        bat_current = batting[batting['Season'] == actual_year]
        logger.warning(
            f"No {current_year} batting data — falling back to {actual_year}. "
            "Run Phase 1 (merge) first to ingest current-season stats."
        )

    pit_current = pitching[pitching['Season'] == actual_year]

    # Ensure IDfg is numeric for matching
    bat_current = bat_current.copy()
    pit_current = pit_current.copy()
    bat_current['IDfg'] = pd.to_numeric(bat_current['IDfg'], errors='coerce')
    pit_current['IDfg'] = pd.to_numeric(pit_current['IDfg'], errors='coerce')

    # Derive HBP% from counting stats (CSV has HBP as count, not rate)
    for df_actual, pa_col in [(bat_current, 'PA'), (pit_current, 'TBF')]:
        if 'HBP' in df_actual.columns and pa_col in df_actual.columns:
            pa_vals = pd.to_numeric(df_actual[pa_col], errors='coerce').fillna(0)
            hbp_vals = pd.to_numeric(df_actual['HBP'], errors='coerce').fillna(0)
            df_actual['HBP%'] = np.where(pa_vals > 0, hbp_vals / pa_vals, 0.0)

    logger.info(
        f"Loaded {actual_year} actuals (with_statcast): "
        f"{len(bat_current)} batters, {len(pit_current)} pitchers"
    )
    return bat_current, pit_current, actual_year


# ─────────────────────────────────────────────────────────────────────────
# Batter blending
# ─────────────────────────────────────────────────────────────────────────

def _team_remaining_fraction(team_abbrev, team_games_map):
    """Compute remaining fraction of the season based on team games played.

    Returns (162 - games_played) / 162.  If team not found, returns 1.0
    (full season — conservative fallback).
    """
    if not team_games_map or not team_abbrev:
        return 1.0
    gp = team_games_map.get(team_abbrev)
    if gp is None:
        return 1.0
    return max(0.0, (TOTAL_SEASON_GAMES - gp) / TOTAL_SEASON_GAMES)


def blend_batter_projections(preseason_df, actual_batting, current_year=None,
                             team_games_map=None, player_team_map=None):
    """Blend current-year batter base components with pre-season projections.

    Blends the 8 Marcel base components (K%, BB%, HBP%, ISO, BABIP, HR/FB,
    GB%, LD%) using Bayesian shrinkage, then recomposes derived stats
    (AVG, SLG, OBP, wOBA, wRC+, HR, 2B, 3B) via ``compose_from_df``.

    Also blends fielding (Fld) and baserunning (BsR) values when available
    in the actuals data.

    Park factors:
        Actual ISO is park-neutralized before blending (preseason predictions
        are already park-neutral).  BABIP is excluded from park adjustment
        per ``park_factors.EXCLUDED_STATS``.

    Only rows for ``current_year`` are modified; future years are untouched.

    Args:
        team_games_map: Dict mapping team abbreviation → games played this season.
        player_team_map: Dict mapping IDfg → team abbreviation.

    Returns:
        Tuple (blended_df, war_proration_info)
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    use_team_remaining = bool(team_games_map and player_team_map)

    df = preseason_df.copy()
    war_proration = {}

    # Index actual data by IDfg
    actual_lookup = {}
    for _, row in actual_batting.iterrows():
        idfg = row['IDfg']
        if pd.notna(idfg):
            actual_lookup[int(idfg)] = row

    blended_count = 0
    current_mask = df['Year'] == current_year
    # Track which current-year rows were blended for batch recomposition
    blended_indices = []

    for idx in df.index[current_mask]:
        pred_row = df.loc[idx]
        idfg = int(pred_row['IDfg'])
        actual = actual_lookup.get(idfg)

        # Compute remaining fraction from team games
        if use_team_remaining:
            team = player_team_map.get(idfg)
            remaining_frac = _team_remaining_fraction(team, team_games_map)
        else:
            remaining_frac = 1.0

        if actual is None:
            war_proration[idfg] = {'actual_war': 0.0, 'remaining_fraction': remaining_frac}
            continue

        actual_pa = _safe_float(actual.get('PA', 0)) or 0.0
        actual_war = _safe_float(actual.get('WAR', 0)) or 0.0

        if actual_pa < 10:
            war_proration[idfg] = {'actual_war': 0.0, 'remaining_fraction': remaining_frac}
            continue

        # Get player's team for park neutralization
        player_team = (player_team_map or {}).get(idfg) or actual.get('Team', '')
        pf = get_park_factor(player_team)

        # Bayesian shrinkage weight
        w = actual_pa / (actual_pa + BATTER_PRIOR_PA)

        # Blend each base component
        for stat in BATTER_BASE_COMPONENTS:
            if stat in actual.index and stat in df.columns:
                act_val = _safe_float(actual[stat])
                pre_val = _safe_float(pred_row[stat])
                if act_val is not None and pre_val is not None:
                    # Park-neutralize actual ISO before blending
                    if stat == 'ISO' and pf != 1.0:
                        act_val = act_val / pf
                    df.at[idx, stat] = w * act_val + (1 - w) * pre_val

        blended_indices.append(idx)

        # Fall back to PA-based remaining if team data unavailable
        if not use_team_remaining:
            remaining_frac = max(0.0, 1.0 - actual_pa / FULL_SEASON_PA)

        war_proration[idfg] = {
            'actual_war': actual_war,
            'remaining_fraction': remaining_frac,
        }
        blended_count += 1

    # ── Recompose derived stats from blended components ──────────────────
    if blended_indices:
        current_rows = df.loc[blended_indices].copy()
        # Ensure HBP% column exists (it's a base component but may need deriving)
        if 'HBP%' not in current_rows.columns and 'HBP' in current_rows.columns:
            pa_vals = pd.to_numeric(current_rows.get('PA', 650), errors='coerce').fillna(650)
            current_rows['HBP%'] = pd.to_numeric(current_rows['HBP'], errors='coerce').fillna(0) / pa_vals

        recomposed = compose_from_df(current_rows)

        # Overwrite derived stats in the main DataFrame
        derived_cols = ['AVG', 'SLG', 'OBP', 'wOBA', 'wRC+', 'FB%',
                        'HR', '2B', '3B', '1B', 'H']
        for col in derived_cols:
            if col in recomposed.columns:
                df.loc[blended_indices, col] = recomposed[col].values

    logger.info(
        f"Blended {blended_count} current-year batter projections "
        f"(8 base components → recompose, prior_PA={BATTER_PRIOR_PA})"
    )
    return df, war_proration


# ─────────────────────────────────────────────────────────────────────────
# Pitcher blending
# ─────────────────────────────────────────────────────────────────────────

def blend_pitcher_projections(preseason_sp, preseason_rp, actual_pitching,
                              current_year=None, team_games_map=None,
                              player_team_map=None):
    """Blend current-year pitcher base components with pre-season projections.

    Blends the 8 pitcher Marcel base components (K%, BB%, HBP%, BABIP,
    HR/FB, GB%, FB%, LD%) using Bayesian shrinkage, then reconstructs
    derived stats (FIP, ERA, SIERA, K/9, BB/9, HR/9) from components.

    Args:
        team_games_map: Dict mapping team abbreviation → games played this season.
        player_team_map: Dict mapping IDfg → team abbreviation.

    Returns:
        Tuple (blended_sp, blended_rp, war_proration_info)
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    use_team_remaining = bool(team_games_map and player_team_map)

    sp_df = preseason_sp.copy()
    rp_df = preseason_rp.copy()
    war_proration = {}

    # Index actual data by IDfg
    actual_lookup = {}
    for _, row in actual_pitching.iterrows():
        idfg = row['IDfg']
        if pd.notna(idfg):
            actual_lookup[int(idfg)] = row

    blended_count = 0
    for df, role, full_ip in [
        (sp_df, 'SP', FULL_SEASON_SP_IP),
        (rp_df, 'RP', FULL_SEASON_RP_IP),
    ]:
        current_mask = df['Year'] == current_year
        blended_indices = []

        for idx in df.index[current_mask]:
            pred_row = df.loc[idx]
            idfg = int(pred_row['IDfg'])
            actual = actual_lookup.get(idfg)

            # Compute remaining fraction from team games
            if use_team_remaining:
                team = player_team_map.get(idfg)
                remaining_frac = _team_remaining_fraction(team, team_games_map)
            else:
                remaining_frac = 1.0

            if actual is None:
                war_proration[idfg] = {
                    'actual_war': 0.0, 'remaining_fraction': remaining_frac,
                }
                continue

            actual_ip = _safe_float(actual.get('IP', 0)) or 0.0
            actual_war = _safe_float(actual.get('WAR', 0)) or 0.0

            if actual_ip < 5:
                war_proration[idfg] = {
                    'actual_war': 0.0, 'remaining_fraction': remaining_frac,
                }
                continue

            w = actual_ip / (actual_ip + PITCHER_PRIOR_IP)

            # Blend each base component
            for stat in PITCHER_MARCEL_RATE_STATS:
                if stat in actual.index and stat in df.columns:
                    act_val = _safe_float(actual[stat])
                    pre_val = _safe_float(pred_row[stat])
                    if act_val is not None and pre_val is not None:
                        df.at[idx, stat] = w * act_val + (1 - w) * pre_val

            blended_indices.append(idx)

            # Fall back to IP-based remaining if team data unavailable
            if not use_team_remaining:
                remaining_frac = max(0.0, 1.0 - actual_ip / full_ip)

            war_proration[idfg] = {
                'actual_war': actual_war,
                'remaining_fraction': remaining_frac,
            }
            blended_count += 1

        # ── Reconstruct derived pitcher stats from blended components ────
        if blended_indices:
            for idx in blended_indices:
                row = df.loc[idx]
                k_pct   = _safe_float(row.get('K%', 0.22)) or 0.22
                bb_pct  = _safe_float(row.get('BB%', 0.08)) or 0.08
                hbp_pct = _safe_float(row.get('HBP%', 0.01)) or 0.01
                babip   = _safe_float(row.get('BABIP', 0.29)) or 0.29
                hr_fb   = _safe_float(row.get('HR/FB', 0.11)) or 0.11
                gb_pct  = _safe_float(row.get('GB%', 0.43)) or 0.43
                fb_pct  = _safe_float(row.get('FB%', 0.35)) or 0.35
                ld_pct  = _safe_float(row.get('LD%', 0.22)) or 0.22

                # Normalize batted ball rates
                bb_total = gb_pct + fb_pct + ld_pct
                if bb_total > 0 and abs(bb_total - 1.0) > 0.01:
                    gb_pct /= bb_total
                    fb_pct /= bb_total
                    ld_pct /= bb_total
                    df.at[idx, 'GB%'] = gb_pct
                    df.at[idx, 'FB%'] = fb_pct
                    df.at[idx, 'LD%'] = ld_pct

                bip_rate = max(1.0 - k_pct - bb_pct - hbp_pct, 0.20)
                hr_pct = hr_fb * fb_pct * bip_rate
                bf_per_ip = _derive_bf_per_ip(k_pct, bb_pct, hbp_pct, hr_pct, babip)

                # FIP
                fip = _reconstruct_fip(k_pct, bb_pct, hbp_pct, hr_fb, fb_pct, bf_per_ip)
                df.at[idx, 'FIP'] = fip

                # ERA = FIP + career ERA-FIP gap (already in preseason row)
                pre_era = _safe_float(row.get('ERA'))
                pre_fip = _safe_float(row.get('FIP'))
                if pre_era is not None and pre_fip is not None:
                    era_fip_gap = pre_era - pre_fip
                else:
                    era_fip_gap = 0.0
                df.at[idx, 'ERA'] = np.clip(fip + era_fip_gap, 0.5, 10.0)

                # SIERA
                df.at[idx, 'SIERA'] = _reconstruct_siera(k_pct, bb_pct, gb_pct)

                # K/9, BB/9, HR/9
                df.at[idx, 'K/9'] = np.clip(k_pct * bf_per_ip * 9.0 / 3.0, 0.0, 20.0)
                df.at[idx, 'BB/9'] = np.clip(bb_pct * bf_per_ip * 9.0 / 3.0, 0.0, 10.0)
                df.at[idx, 'HR/9'] = np.clip(hr_pct * bf_per_ip * 9.0 / 3.0, 0.0, 5.0)

    logger.info(
        f"Blended {blended_count} current-year pitcher projections "
        f"(8 base components → reconstruct FIP/ERA/SIERA, prior_IP={PITCHER_PRIOR_IP})"
    )
    return sp_df, rp_df, war_proration


# ─────────────────────────────────────────────────────────────────────────
# Playing-time reduction to remaining season
# ─────────────────────────────────────────────────────────────────────────

# Counting stats that scale with playing time (batters)
BATTER_COUNTING_COLS = [
    'PA', 'G', 'HR', '2B', '3B', '1B', 'H', 'AB', 'K',
    'BB_count', 'HBP_count', 'RBI', 'R', 'HBP', 'BB',
    'SB', 'CS', 'SF',
]
# Counting stats that scale with playing time (pitchers)
PITCHER_COUNTING_COLS = [
    'IP', 'GS', 'G', 'W', 'L', 'SO', 'H', 'HR',
    'BB', 'HBP', 'ER', 'R', 'TBF',
]


def reduce_to_remaining_season(df, war_proration, current_year=None,
                               player_type='batter'):
    """Scale current-year playing time and counting stats to remaining season.

    If a batter was projected for 150 G / 650 PA and the team has played 10
    games, the projection is reduced to ~140 G / ~610 PA and all counting
    stats are scaled proportionally.  Rate stats are untouched.

    For pitchers, IP / GS / G and counting stats are reduced the same way.

    Sets ``playing_time_reduced=True`` in each player's war_proration entry
    so that :func:`prorate_current_year_war` uses direct addition
    (``actual + projected_ROS``) instead of re-multiplying by
    ``remaining_fraction``.

    Args:
        df: Prediction DataFrame (must have IDfg, Year columns).
        war_proration: Dict mapping IDfg → {actual_war, remaining_fraction}.
        current_year: Season to reduce.
        player_type: ``'batter'`` or ``'pitcher'``.

    Returns:
        DataFrame with reduced current-year counting stats.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    counting_cols = (BATTER_COUNTING_COLS if player_type == 'batter'
                     else PITCHER_COUNTING_COLS)

    df = df.copy()
    reduced = 0

    # Ensure counting columns are float so fractional values can be stored
    for col in counting_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    current_mask = df['Year'] == current_year
    for idx in df.index[current_mask]:
        try:
            idfg = int(df.at[idx, 'IDfg'])
        except (ValueError, TypeError):
            continue

        info = war_proration.get(idfg)
        if info is None:
            continue

        remaining = info['remaining_fraction']
        if remaining >= 1.0:
            continue  # full season — nothing to reduce

        for col in counting_cols:
            if col in df.columns:
                val = df.at[idx, col]
                if pd.notna(val):
                    df.at[idx, col] = val * remaining

        info['playing_time_reduced'] = True
        reduced += 1

    logger.info(
        f"Reduced {reduced} current-year {player_type} projections "
        f"to remaining season (counting stats × remaining_fraction)"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────
# Partial-season WAR proration
# ─────────────────────────────────────────────────────────────────────────

def prorate_current_year_war(df, war_proration, current_year=None):
    """Replace current-year WAR with ROS projected WAR.

    If playing time was already reduced to remaining season via
    :func:`reduce_to_remaining_season`, the projected WAR is already
    ROS-scale and we simply use it::

        new_WAR = projected_ros_war

    Otherwise (full-season projection), the old formula applies::

        new_WAR = projected_full_war × remaining_fraction

    Future years (Year > current_year) are untouched.

    Args:
        df: DataFrame with IDfg, Year, WAR columns (post WAR calculation).
        war_proration: Dict mapping IDfg → {actual_war, remaining_fraction,
            playing_time_reduced (optional)}.
        current_year: Season to prorate (default CURRENT_YEAR).

    Returns:
        DataFrame with prorated current-year WAR.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    df = df.copy()
    prorated = 0

    current_mask = df['Year'] == current_year
    for idx in df.index[current_mask]:
        raw_idfg = df.at[idx, 'IDfg']
        try:
            idfg = int(raw_idfg)
        except (ValueError, TypeError):
            continue

        info = war_proration.get(idfg)
        if info is None:
            continue

        projected_war = df.at[idx, 'WAR']

        if info.get('playing_time_reduced', False):
            # Playing time already reduced → projected WAR is ROS-scale
            prorated_war = projected_war
        else:
            # Full-season projection → scale down
            prorated_war = projected_war * info['remaining_fraction']

        df.at[idx, 'WAR'] = prorated_war
        prorated += 1

    logger.info(
        f"Prorated {prorated} current-year WAR values "
        f"(actual + projected ROS)"
    )
    return df


def prorate_current_year_salary(df, war_proration, current_year=None):
    """Prorate current-year salary to reflect remaining season only.

    Current-year contract_value should only include remaining salary owed,
    not salary already paid. We scale it by the same remaining_fraction
    used for WAR proration.

    Formula::
        new_contract_value = full_year_contract × remaining_fraction

    Future years (Year > current_year) are untouched.

    Args:
        df: DataFrame with IDfg, Year, contract_value columns.
        war_proration: Dict mapping IDfg → {remaining_fraction, ...}.
        current_year: Season to prorate (default CURRENT_YEAR).

    Returns:
        DataFrame with prorated current-year contract_value.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    df = df.copy()
    prorated = 0

    # Use lowercase 'contract_value' to match value_calculator.py output
    if 'contract_value' not in df.columns:
        logger.warning("contract_value column not found; skipping salary proration")
        return df

    current_mask = df['Year'] == current_year
    for idx in df.index[current_mask]:
        raw_idfg = df.at[idx, 'IDfg']
        try:
            idfg = int(raw_idfg)
        except (ValueError, TypeError):
            continue

        info = war_proration.get(idfg)
        if info is None:
            continue

        # Get remaining fraction from war_proration
        remaining_frac = info.get('remaining_fraction', 1.0)
        if remaining_frac >= 1.0:
            # Full season already played or remaining_fraction not set
            continue

        original_contract = df.at[idx, 'contract_value']
        if pd.isna(original_contract) or original_contract == 0:
            continue

        # Prorate to remaining season
        prorated_contract = original_contract * remaining_frac
        df.at[idx, 'contract_value'] = prorated_contract
        prorated += 1

    logger.info(
        f"Prorated {prorated} current-year contract_value entries "
        f"to remaining season"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────
# Fielding blending
# ─────────────────────────────────────────────────────────────────────────

def blend_fielding_projections(fielding_df, actual_batting, current_year=None):
    """Blend preseason fielding projections with actual Fld runs from FanGraphs.

    The FanGraphs batting CSV includes a ``Fld`` column (total fielding runs)
    which serves as the current-season actual.  We blend this with the
    preseason ``sc_total_runs/150`` projection using games-based shrinkage.

    The fielding_df (per-150 rates) is converted to seasonal runs for blending,
    then normalized back.

    Args:
        fielding_df: Preseason fielding prediction DataFrame with
            columns IDfg, Year, sc_total_runs/150, and other sc_* runs.
        actual_batting: Current-season FanGraphs batting DataFrame
            (must include IDfg, G, Fld columns).
        current_year: Season to blend (default CURRENT_YEAR).

    Returns:
        Updated fielding_df with blended current-year values.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    df = fielding_df.copy()
    current_mask = df['Year'] == current_year
    if not current_mask.any():
        return df

    # Build actual lookup: IDfg → (Fld_runs, games_played)
    actual_lookup = {}
    for _, row in actual_batting.iterrows():
        idfg = row.get('IDfg')
        if pd.isna(idfg):
            continue
        fld = _safe_float(row.get('Fld'))
        games = _safe_float(row.get('G', 0)) or 0.0
        if fld is not None and games >= 5:
            actual_lookup[int(idfg)] = (fld, games)

    blended_count = 0
    for idx in df.index[current_mask]:
        idfg = int(df.at[idx, 'IDfg'])
        actual_data = actual_lookup.get(idfg)
        if actual_data is None:
            continue

        actual_fld, games = actual_data

        # Convert actual Fld (seasonal runs in G games) to per-150 rate
        if games > 0:
            actual_per_150 = actual_fld * (150.0 / games)
        else:
            continue

        pre_per_150 = _safe_float(df.at[idx, 'sc_total_runs/150']) or 0.0

        # Bayesian shrinkage: w = G / (G + FIELDING_PRIOR_G)
        w = games / (games + FIELDING_PRIOR_G)
        blended_per_150 = w * actual_per_150 + (1 - w) * pre_per_150

        df.at[idx, 'sc_total_runs/150'] = blended_per_150
        blended_count += 1

    logger.info(
        f"Blended {blended_count} current-year fielding projections "
        f"(Fld per-150 via Bayesian shrinkage, prior_G={FIELDING_PRIOR_G})"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────
# Baserunning blending
# ─────────────────────────────────────────────────────────────────────────

def blend_baserunning_projections(baserunning_df, actual_batting, current_year=None):
    """Blend preseason baserunning projections with actual BsR from FanGraphs.

    The FanGraphs batting CSV includes a ``BsR`` column (baserunning runs)
    which serves as the current-season actual.  We blend this with the
    preseason ``sc_baserunning_runner_runs_tot_rate`` projection.

    Args:
        baserunning_df: Preseason baserunning prediction DataFrame with
            columns IDfg, Year, sc_baserunning_runner_runs_tot_rate.
        actual_batting: Current-season FanGraphs batting DataFrame
            (must include IDfg, PA, BsR columns).
        current_year: Season to blend (default CURRENT_YEAR).

    Returns:
        Updated baserunning_df with blended current-year values.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    df = baserunning_df.copy()
    current_mask = df['Year'] == current_year
    if not current_mask.any():
        return df

    # Build actual lookup: IDfg → (BsR_runs, PA)
    actual_lookup = {}
    for _, row in actual_batting.iterrows():
        idfg = row.get('IDfg')
        if pd.isna(idfg):
            continue
        bsr = _safe_float(row.get('BsR'))
        pa = _safe_float(row.get('PA', 0)) or 0.0
        if bsr is not None and pa >= 10:
            actual_lookup[int(idfg)] = (bsr, pa)

    blended_count = 0
    for idx in df.index[current_mask]:
        idfg = int(df.at[idx, 'IDfg'])
        actual_data = actual_lookup.get(idfg)
        if actual_data is None:
            continue

        actual_bsr, pa = actual_data

        # Convert actual BsR (seasonal runs in PA) to per-650 rate
        if pa > 0:
            actual_rate = actual_bsr * (FULL_SEASON_PA / pa)
        else:
            continue

        pre_rate = _safe_float(df.at[idx, 'sc_baserunning_runner_runs_tot_rate']) or 0.0

        # Bayesian shrinkage: w = PA / (PA + BASERUNNING_PRIOR_PA)
        w = pa / (pa + BASERUNNING_PRIOR_PA)
        blended_rate = w * actual_rate + (1 - w) * pre_rate

        df.at[idx, 'sc_baserunning_runner_runs_tot_rate'] = blended_rate
        blended_count += 1

    logger.info(
        f"Blended {blended_count} current-year baserunning projections "
        f"(BsR per-650 via Bayesian shrinkage, prior_PA={BASERUNNING_PRIOR_PA})"
    )
    return df
