"""
ROS (Rest-of-Season) Projection Blending
=========================================

Blends pre-season predictions with actual current-season performance
using Bayesian shrinkage.

Principle:
    ROS_rate = w * actual + (1 - w) * prior
    w = sample_size / (sample_size + prior_strength)

As actual sample grows, the weight shifts from pre-season projection
toward actual performance.  The prior_strength parameter controls how
quickly the shift occurs (higher = more conservative).

Note on park factors:
    Actual stats contain park effects baked in (they are real performance).
    Pre-season predictions may be park-neutral (depending on config toggle).
    The blending mixes these directly; park factor reapplication in the
    WAR pipeline runs afterward.  The resulting error is second-order
    (~2% at mid-season) and within projection noise.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path

from value_determination.config import Config, logger, CURRENT_YEAR

# =============================================================================
# Shrinkage parameters
# =============================================================================
BATTER_PRIOR_PA = 400    # PA for pre-season prior to equal actual weight
PITCHER_PRIOR_IP = 80    # IP for pre-season prior to equal actual weight

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

# Rate stats to blend (must exist in both prediction and actual data)
BATTER_RATE_STATS = ['wOBA', 'BB%', 'K%', 'AVG', 'OBP', 'SLG']
PITCHER_RATE_STATS = [
    'FIP', 'ERA', 'K%', 'BB%', 'BABIP', 'SIERA',
    'HR/FB', 'GB%', 'FB%', 'K/9', 'BB/9', 'HR/9',
]


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

    Reads from the historic CSV files (which Phase 1 merge keeps updated).
    Falls back to ``current_year - 1`` if the target year has no data yet.

    Returns:
        Tuple (batting_df, pitching_df, actual_year)
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    hist_dir = Config.Paths.HISTORIC_MLB_DIR

    batting = pd.read_csv(
        hist_dir / 'mlb_batting_data_1950_2025.csv', low_memory=False,
    )
    pitching = pd.read_csv(
        hist_dir / 'mlb_pitching_data_1950_2025.csv', low_memory=False,
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

    logger.info(
        f"Loaded {actual_year} actuals: "
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
    """Blend current-year batter rate stats with pre-season projections.

    Only rows for ``current_year`` are modified; future years are untouched.

    Args:
        team_games_map: Dict mapping team abbreviation → games played this season.
            When provided, remaining_fraction is based on team remaining games
            instead of individual PA accumulation.
        player_team_map: Dict mapping IDfg → team abbreviation.

    Returns:
        Tuple (blended_df, war_proration_info)

        war_proration_info maps IDfg → {actual_war, remaining_fraction}
        for use by :func:`prorate_current_year_war`.
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

    for idx in df.index[current_mask]:
        pred_row = df.loc[idx]
        idfg = int(pred_row['IDfg'])
        actual = actual_lookup.get(idfg)

        # Compute remaining fraction from team games
        if use_team_remaining:
            team = player_team_map.get(idfg)
            remaining_frac = _team_remaining_fraction(team, team_games_map)
        else:
            remaining_frac = 1.0  # will be overwritten below for PA-based fallback

        if actual is None:
            # No current-season data → keep pre-season as-is
            war_proration[idfg] = {'actual_war': 0.0, 'remaining_fraction': remaining_frac}
            continue

        actual_pa = _safe_float(actual.get('PA', 0)) or 0.0
        actual_war = _safe_float(actual.get('WAR', 0)) or 0.0

        if actual_pa < 10:
            war_proration[idfg] = {'actual_war': 0.0, 'remaining_fraction': remaining_frac}
            continue

        # Bayesian shrinkage weight
        w = actual_pa / (actual_pa + BATTER_PRIOR_PA)

        for stat in BATTER_RATE_STATS:
            if stat in actual.index and stat in df.columns:
                act_val = _safe_float(actual[stat])
                pre_val = _safe_float(pred_row[stat])
                if act_val is not None and pre_val is not None:
                    df.at[idx, stat] = w * act_val + (1 - w) * pre_val

        # Fall back to PA-based remaining if team data unavailable
        if not use_team_remaining:
            remaining_frac = max(0.0, 1.0 - actual_pa / FULL_SEASON_PA)

        war_proration[idfg] = {
            'actual_war': actual_war,
            'remaining_fraction': remaining_frac,
        }
        blended_count += 1

    logger.info(
        f"Blended {blended_count} current-year batter projections "
        f"with actual data (prior_PA={BATTER_PRIOR_PA})"
    )
    return df, war_proration


# ─────────────────────────────────────────────────────────────────────────
# Pitcher blending
# ─────────────────────────────────────────────────────────────────────────

def blend_pitcher_projections(preseason_sp, preseason_rp, actual_pitching,
                              current_year=None, team_games_map=None,
                              player_team_map=None):
    """Blend current-year pitcher rate stats with pre-season projections.

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

            for stat in PITCHER_RATE_STATS:
                if stat in actual.index and stat in df.columns:
                    act_val = _safe_float(actual[stat])
                    pre_val = _safe_float(pred_row[stat])
                    if act_val is not None and pre_val is not None:
                        df.at[idx, stat] = w * act_val + (1 - w) * pre_val

            # Fall back to IP-based remaining if team data unavailable
            if not use_team_remaining:
                remaining_frac = max(0.0, 1.0 - actual_ip / full_ip)

            war_proration[idfg] = {
                'actual_war': actual_war,
                'remaining_fraction': remaining_frac,
            }
            blended_count += 1

    logger.info(
        f"Blended {blended_count} current-year pitcher projections "
        f"with actual data (prior_IP={PITCHER_PRIOR_IP})"
    )
    return sp_df, rp_df, war_proration


# ─────────────────────────────────────────────────────────────────────────
# Partial-season WAR proration
# ─────────────────────────────────────────────────────────────────────────

def prorate_current_year_war(df, war_proration, current_year=None):
    """Replace current-year WAR with actual + (projected_full × remaining).

    For each player in ``war_proration``:
        new_WAR = actual_war + projected_full_season_war × remaining_fraction

    Future years (Year > current_year) are untouched.

    Args:
        df: DataFrame with IDfg, Year, WAR columns (post WAR calculation).
        war_proration: Dict mapping IDfg → {actual_war, remaining_fraction}.
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

        projected_full = df.at[idx, 'WAR']
        prorated_war = info['actual_war'] + projected_full * info['remaining_fraction']
        df.at[idx, 'WAR'] = prorated_war
        prorated += 1

    logger.info(
        f"Prorated {prorated} current-year WAR values "
        f"(actual + projected × remaining)"
    )
    return df
