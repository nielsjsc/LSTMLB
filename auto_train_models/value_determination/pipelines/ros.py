"""
ROS (Rest-of-Season) Projection Blending
=========================================

Blends pre-season Marcel projections with actual current-season performance
using Marcel-consistent weighting on *base components* (not derived stats).

Blending Philosophy
-------------------
Pre-season Marcel projects Year N using a 3/4/5-weighted average of years
N-3, N-2, N-1 (total weight = 12).  Once the current season begins, the
partial year N sample should earn weight proportional to its completeness
within that same 12-unit Marcel framework:

    current_season_weight = 5 × (actual_volume / full_season_volume)
    historical_weight     = 12 − current_season_weight

    blend_w   = current_season_weight / 12        # 0 → 5/12 over the season
    prior_w   = historical_weight     / 12        # 12/12 → 7/12 over the season

    blended = blend_w × actual + prior_w × preseason_projection

This preserves the 5:4:3 ratio among the three prior seasons (they are all
embedded in the preseason projection and shrink together as current-season
data accumulates).  At a full season of PA/IP the current year earns exactly
the Marcel weight-5 slot.  At the start of the season it contributes nothing
and the preseason projection is used unchanged.

Key consequence: by September, the blend is roughly 40% current season /
60% prior three seasons — not 60% current season as the old Bayesian
shrinkage formula produced.

Architecture
------------
Batters  — NO LONGER BLENDED HERE. marcel_batter_projections() (see
           core/marcel_projections.py) now re-runs with cutoff_year set to
           the current in-progress season and emits the current year
           directly at year_offset=0 from a freshly recomputed year1_base
           (Marcel weighted average + MiLB-prior blend, both re-derived
           from up-to-date in-season data). blend_batter_projections() in
           this file is kept only to compute war_proration (banked actual
           WAR + remaining-season fraction) for
           prorate_current_year_war/prorate_current_year_salary — it no
           longer touches rate stats. This avoids having two independently-
           weighted blends of the same current-season data (this module's
           fixed 12-unit formula vs. Marcel's PA/career-PA-weighted
           formula) disagree with each other, which previously caused the
           current-year projection to diverge sharply from next year's.
Pitchers — still blended here: 8 base components (K%, BB%, HBP%, BABIP,
           HR/FB, GB%, FB%, LD%), then reconstruct FIP/ERA/SIERA/K-rates.
           (Not in scope for this fix — batters only, for now.)
Fielding — blend preseason sc_total_runs/150 with actual Fld runs using
           Marcel regression-consistent weighting.
Baserunning — blend preseason BsR with actual BsR using Marcel regression-
           consistent weighting.

Park factors
------------
Actual stats contain park effects; preseason predictions are park-neutral.
Before blending, actual rate stats are neutralized so both sides are in the
same space.  Park factors are reapplied downstream in WAR calculation.

Data source
-----------
Uses ``_with_statcast`` CSV variants for richer data (xwOBA, sprint speed,
barrel rate, Statcast fielding/baserunning columns).
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path

from value_determination.config import Config, logger, CURRENT_YEAR
from value_determination.calculate_war import load_player_orgs
from core.park_factors import get_park_factor, EXCLUDED_STATS
from core.stat_composition import compose_from_df, compose_wrc_plus
from core.marcel_projections import (
    BATTER_BASE_COMPONENTS,
    PITCHER_MARCEL_RATE_STATS,
    _reconstruct_fip,
    _reconstruct_siera,
    _derive_bf_per_ip,
    _FIP_CONSTANT,
    _load_aging_curves,
    _get_smoothed_aging_delta,
    _ERA_FIP_STAB_TBF,
    BATTER_MULTIVARIATE_EQUATIONS,
    PITCHER_SP_MULTIVARIATE_EQUATIONS,
    PITCHER_RP_MULTIVARIATE_EQUATIONS,
    SEASON_WEIGHTS,           # [5, 4, 3]
    FIELDING_REGRESSION_INNINGS,
    BASERUNNING_REGRESSION_GAMES,
)
from configs.batter_config import BatterConfig

# --- PATCH: helpers for call-up fixes ---
POS_TO_GROUP = {
    'LF':'outfield','CF':'outfield','RF':'outfield',
    '1B':'infield','2B':'infield','3B':'infield','SS':'infield',
    'C':'catcher','DH':'infield','OF':'outfield','UT':'infield'
}
def _resolve_name_from_actuals(idfg, actual_df=None, fallback="Unknown"):
    if actual_df is None or actual_df.empty or 'IDfg' not in actual_df.columns:
        return None
    try:
        m = actual_df[actual_df['IDfg'].astype(int)==int(idfg)]
        if not m.empty:
            # try Name, player_name, etc
            for col in ('Name','PlayerName','player_name'):
                if col in m.columns and pd.notna(m.iloc[0].get(col)):
                    return str(m.iloc[0][col])
    except Exception:
        pass
    return None

from configs.pitcher_sp_config import PitcherSPConfig

# =============================================================================
# Marcel weighting constants
# =============================================================================

# Total Marcel weight across all three prior seasons (5 + 4 + 3)
MARCEL_TOTAL_WEIGHT = float(sum(SEASON_WEIGHTS))   # 12.0

# The weight a *full* current season earns in the Marcel framework.
# A complete season becomes the new Y-1 and gets weight 5.
MARCEL_CURRENT_FULL_WEIGHT = float(SEASON_WEIGHTS[0])   # 5.0

# Full-season volume baselines (denominators for current-season weight)
FULL_SEASON_PA    = 650
FULL_SEASON_SP_IP = Config.WAR.DEFAULT_SP_IP    # 180
FULL_SEASON_RP_IP = Config.WAR.DEFAULT_RP_IP    # 70

# Minimum volume before we blend at all (too noisy below these thresholds)
MIN_BATTER_PA  = 30     # ~10 games
MIN_PITCHER_IP = 10     # ~3 starts / ~8 relief appearances
MIN_FIELDING_G = 10
MIN_BASERUNNING_PA = 30

TOTAL_SEASON_GAMES = 162

MLB_API_BASE = 'https://statsapi.mlb.com/api/v1'


def _marcel_blend_weight(actual_volume: float, full_season_volume: float) -> float:
    """Return the Marcel-consistent blend weight for the current season.

    Scales linearly from 0 (no data) to MARCEL_CURRENT_FULL_WEIGHT / MARCEL_TOTAL_WEIGHT
    (full season), keeping the prior-seasons share at the complementary fraction.

    Args:
        actual_volume:      Current season PA, IP, or games accumulated so far.
        full_season_volume: Full-season baseline (650 PA, 180 IP, etc.)

    Returns:
        w in [0, MARCEL_CURRENT_FULL_WEIGHT / MARCEL_TOTAL_WEIGHT]
        Blend formula: blended = w * actual + (1 - w) * preseason_projection
    """
    season_fraction = np.clip(actual_volume / full_season_volume, 0.0, 1.0)
    current_weight  = MARCEL_CURRENT_FULL_WEIGHT * season_fraction
    return current_weight / MARCEL_TOTAL_WEIGHT


# =============================================================================
# Team games played  (for ROS remaining-fraction)
# =============================================================================

def fetch_team_games_played(season: int = None) -> dict:
    """Fetch games played per team from the MLB Stats API standings endpoint.

    Uses team_info.csv from the roster scraper to map MLB team IDs to the
    FanGraphs-style abbreviations used throughout the pipeline.

    Returns:
        Dict mapping team abbreviation (e.g. 'NYY') → games_played (int).
        Returns empty dict on failure; callers fall back to PA/IP-based
        remaining-fraction.
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

        id_to_gp = {}
        for rec in data.get('records', []):
            for t in rec.get('teamRecords', []):
                tid = t.get('team', {}).get('id')
                gp  = t.get('gamesPlayed', 0)
                if tid is not None:
                    id_to_gp[tid] = gp

        team_info_path = Config.Paths.DATA_DIR / 'active_roster' / 'team_info.csv'
        if not team_info_path.exists():
            logger.warning(f"team_info.csv not found at {team_info_path}")
            return {}

        team_info  = pd.read_csv(team_info_path)
        abbrev_map = dict(zip(team_info['team_id'], team_info['abbreviation']))

        result = {}
        for tid, gp in id_to_gp.items():
            abbrev = abbrev_map.get(tid)
            if abbrev:
                result[abbrev] = gp

        # FanGraphs ↔ MLB Stats API abbreviation aliases
        fg_aliases = {
            'CWS': 'CHW', 'CHW': 'CWS',
            'AZ':  'ARI', 'ARI': 'AZ',
            'SDP': 'SD',  'SD':  'SDP',
            'SFG': 'SF',  'SF':  'SFG',
            'TBR': 'TB',  'TB':  'TBR',
            'KCR': 'KC',  'KC':  'KCR',
            'WSN': 'WSH', 'WSH': 'WSN',
        }
        for existing, alias in fg_aliases.items():
            if existing in result and alias not in result:
                result[alias] = result[existing]

        logger.info(
            f"Fetched team games played for {len(result)} teams "
            f"(season={season}, range={min(result.values())}–{max(result.values())} GP)"
        )
        return result

    except Exception as e:
        logger.warning(
            f"Failed to fetch team games played: {e} — "
            "will fall back to PA/IP-based remaining"
        )
        return {}


def _team_remaining_fraction(team_abbrev: str, team_games_map: dict) -> float:
    """Return the fraction of the season remaining for a team.

    Returns (162 − games_played) / 162.  Falls back to 1.0 if the team
    is not in the map.
    """
    if not team_games_map or not team_abbrev:
        return 1.0
    gp = team_games_map.get(team_abbrev)
    if gp is None:
        return 1.0
    return max(0.0, (TOTAL_SEASON_GAMES - gp) / TOTAL_SEASON_GAMES)


# =============================================================================
# Helpers
# =============================================================================

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


def _overlay_current_year_rows(
    raw_df: pd.DataFrame,
    blended_current_year_df: pd.DataFrame,
    current_year: int,
) -> pd.DataFrame:
    """Overlay ROS-blended current-year rows onto a historical data frame.

    Used by build_ros_blended_history_snapshots to replace the current-year
    rows in the historical CSV with the blended versions so that the next
    Marcel pass treats them as the most recent season.
    """
    if raw_df.empty or blended_current_year_df.empty:
        return raw_df.copy()

    result  = raw_df.copy()
    overlay = blended_current_year_df.copy()

    # Normalise Season / Year column names
    for df_ in (result, overlay):
        if 'Season' not in df_.columns and 'Year' in df_.columns:
            df_['Season'] = df_['Year']
        if 'Year' not in df_.columns and 'Season' in df_.columns:
            df_['Year'] = df_['Season']

    # Coerce any array-like or non-scalar cells in the overlay to scalars.
    def _coerce_to_scalar(x):
        if pd.isna(x):
            return x
        try:
            import numpy as _np
            if isinstance(x, (list, tuple, _np.ndarray)):
                try:
                    arr = _np.asarray(x, dtype=float)
                    return float(_np.nanmean(arr)) if arr.size else pd.NA
                except Exception:
                    return str(x[0]) if len(x) else pd.NA
        except Exception:
            pass
        try:
            import pyarrow as _pa
            if isinstance(x, (_pa.Array, _pa.ChunkedArray)):
                lst = x.to_pylist()
                import numpy as _np
                try:
                    arr = _np.asarray(lst, dtype=float)
                    return float(_np.nanmean(arr)) if arr.size else pd.NA
                except Exception:
                    return str(lst[0]) if lst else pd.NA
        except Exception:
            pass
        return x

    for _col in list(overlay.columns):
        try:
            overlay[_col] = overlay[_col].apply(_coerce_to_scalar)
        except Exception:
            pass

    # Add any new columns the overlay introduces.
    # Preserve object/string dtype for new columns so overlay updates do not
    # fail when the overlay contains string values such as Role.
    for col in overlay.columns:
        if col not in result.columns:
            if pd.api.types.is_string_dtype(overlay[col]) or overlay[col].dtype == object:
                result[col] = pd.Series([pd.NA] * len(result), dtype='object')
            else:
                result[col] = np.nan

    key_col = 'Season' if 'Season' in result.columns else 'Year'
    if 'IDfg' not in result.columns or key_col not in result.columns:
        return result

    # Upcast int columns to float to avoid LossySetitemError
    for _c in result.select_dtypes(include=['int64', 'Int64']).columns:
        try:
            result[_c] = result[_c].astype(float)
        except Exception:
            pass

    idx_cols = ['IDfg', key_col]
    if 'Pos' in result.columns and 'Pos' in overlay.columns:
        idx_cols.append('Pos')
    elif 'Role' in result.columns and 'Role' in overlay.columns:
        idx_cols.append('Role')

    # Drop duplicates in overlay to be safe
    overlay = overlay.drop_duplicates(subset=idx_cols, keep='last')

    # If result still has duplicates in the index, drop them
    if result.duplicated(subset=idx_cols).any():
        result = result.drop_duplicates(subset=idx_cols, keep='last')

    result  = result.set_index(idx_cols)
    overlay = overlay.set_index(idx_cols)
    result.update(overlay)
    return result.reset_index()


# =============================================================================
# History snapshot builder
# =============================================================================

def build_ros_blended_history_snapshots(
    preseason_batter_df: pd.DataFrame,
    preseason_sp_df: pd.DataFrame,
    preseason_rp_df: pd.DataFrame,
    preseason_fielding_df: pd.DataFrame | None = None,
    current_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ROS-blended history snapshots for the next Marcel prediction pass.

    Replaces the current-year rows in the historical batter/pitcher CSVs with
    Marcel-blended rows.  The blended row carries the player's *actual* PA/IP
    so that the next Marcel run weights it by real sample size, not a phantom
    full-season volume.

    Returns:
        (blended_batter_history, blended_pitcher_history, blended_fielding_history)
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    batter_history_path  = Path(Config.Paths.ROOT_DIR / 'auto_train_models' / BatterConfig.DATA_FILE)
    pitcher_history_path = Path(Config.Paths.ROOT_DIR / 'auto_train_models' / PitcherSPConfig.DATA_FILE)

    raw_batter_history  = pd.read_csv(batter_history_path,  low_memory=False)
    raw_pitcher_history = pd.read_csv(pitcher_history_path, low_memory=False)

    actual_batting, actual_pitching, actual_year = load_current_season_actuals(current_year)
    if actual_year != current_year:
        logger.info(
            f"ROS history snapshot: using {actual_year} actuals "
            f"while building {current_year} overlays"
        )

    team_games_map = fetch_team_games_played(season=current_year)
    org_data = load_player_orgs()
    org_data['IDfg'] = pd.to_numeric(org_data['IDfg'], errors='coerce')
    org_valid = org_data.dropna(subset=['IDfg'])
    player_team_map = dict(zip(
        org_valid['IDfg'].astype(int),
        org_valid['Team'],
    ))

    blended_batter_df, _ = blend_batter_projections(
        preseason_batter_df, actual_batting,
        current_year=current_year,
        team_games_map=team_games_map,
        player_team_map=player_team_map,
    )

    # ── Fielding blending ──────────────────────────────────────────────
    blended_fielding_df = pd.DataFrame()
    raw_fielding_history = pd.DataFrame()
    if preseason_fielding_df is not None and not preseason_fielding_df.empty:
        from configs.defense_infield_config import DefenseInfieldConfig
        from core.data_processing import calculate_rate_stats as _calc_rate
        fielding_history_path = Path(Config.Paths.ROOT_DIR / 'auto_train_models' / DefenseInfieldConfig.DATA_FILE)
        raw_fielding_history = pd.read_csv(fielding_history_path, low_memory=False)

        # Fill missing Statcast sub-metrics with 0.0 (league average) so
        # rookies missing specific micro-metrics aren't dropped by Marcel.
        for col in ['sc_range_runs', 'sc_arm_runs', 'sc_dp_runs',
                     'sc_framing_runs', 'sc_throwing_runs', 'sc_blocking_runs']:
            if col in raw_fielding_history.columns:
                raw_fielding_history[col] = raw_fielding_history[col].fillna(0.0)

        # Compute rate stats (sc_total_runs/150 etc.) so overlaid rows have
        # the per-150 columns Marcel expects, not just raw totals.
        raw_fielding_history = _calc_rate(raw_fielding_history)

        # Blend existing preseason rows with actual performance
        blended_fielding_df = blend_fielding_projections(
            preseason_fielding_df, actual_batting, current_year=current_year
        )
        # Derive baselines for missing players (e.g. rookies)
        current_year_ids = actual_batting['IDfg'].dropna().astype(int).unique()
        missing_fielding_df = derive_missing_fielding_baseline(
            preseason_fielding_df, current_year_ids, actual_batting, current_year
        )
        if not missing_fielding_df.empty:
            blended_fielding_df = pd.concat(
                [blended_fielding_df, missing_fielding_df], ignore_index=True
            )

    # Ensure SP/RP frames are populated (some pipelines write a combined CSV)
    if (preseason_sp_df is None or preseason_sp_df.empty) and \
       (preseason_rp_df is None or preseason_rp_df.empty):
        try:
            combined_path = (
                Path(Config.Paths.ROOT_DIR) / 'data' / 'generated' /
                'pipeline' / 'preseason' / 'pitcher_predictions.csv'
            )
            if combined_path.exists():
                combined = pd.read_csv(combined_path, low_memory=False)
                if 'Role' in combined.columns:
                    preseason_sp_df = combined[combined['Role'] == 'SP'].copy()
                    preseason_rp_df = combined[combined['Role'] == 'RP'].copy()
                else:
                    preseason_sp_df = pd.DataFrame(columns=combined.columns)
                    preseason_rp_df = pd.DataFrame(columns=combined.columns)
        except Exception:
            pass

    def _use_next_year_if_missing(df):
        if df is None or df.empty:
            return df
        if 'Year' in df.columns and not (df['Year'] == current_year).any():
            ny = current_year + 1
            if (df['Year'] == ny).any():
                tmp = df[df['Year'] == ny].copy()
                tmp['Year'] = current_year
                return tmp
        return df

    sp_for_blend = _use_next_year_if_missing(
        preseason_sp_df.copy() if preseason_sp_df is not None else pd.DataFrame()
    )
    rp_for_blend = _use_next_year_if_missing(
        preseason_rp_df.copy() if preseason_rp_df is not None else pd.DataFrame()
    )

    blended_sp_df, blended_rp_df, _ = blend_pitcher_projections(
        sp_for_blend, rp_for_blend, actual_pitching,
        current_year=current_year,
        team_games_map=team_games_map,
        player_team_map=player_team_map,
    )

    batter_overlay = blended_batter_df[blended_batter_df['Year'] == current_year].copy()

    # ── Fielding overlay ───────────────────────────────────────────────
    fielding_overlay = pd.DataFrame()
    if not blended_fielding_df.empty:
        fielding_overlay = blended_fielding_df[
            blended_fielding_df['Year'] == current_year
        ].copy()
        # Raw history uses 'Season' not 'Year'; rename so
        # _overlay_current_year_rows can match on the same column name.
        if 'Year' in fielding_overlay.columns and 'Season' not in fielding_overlay.columns:
            fielding_overlay = fielding_overlay.rename(columns={'Year': 'Season'})

    # The overlay rows must carry actual PA/IP (not 650/180) so that the next
    # Marcel run weights them by real sample size when the snapshot is used as
    # Y-1 history.  Stamp actual_pa onto each overlay batter row.
    actual_pa_map = {
        int(r['IDfg']): _safe_float(r.get('PA', 0)) or 0.0
        for _, r in actual_batting.iterrows()
        if pd.notna(r.get('IDfg'))
    }
    if 'PA' in batter_overlay.columns:
        batter_overlay['PA'] = batter_overlay['IDfg'].map(
            lambda idfg: actual_pa_map.get(int(idfg), 0.0)
            if pd.notna(idfg) else 0.0
        )

    actual_g_map = {
        int(r['IDfg']): _safe_float(r.get('G', 0)) or 0.0
        for _, r in actual_batting.iterrows()
        if pd.notna(r.get('IDfg'))
    }
    if not fielding_overlay.empty:
        # Stamp actual Games so Marcel sees real sample size
        fielding_overlay['G'] = fielding_overlay['IDfg'].map(
            lambda idfg: actual_g_map.get(int(idfg), 0.0)
            if pd.notna(idfg) else 0.0
        )

    actual_ip_map = {
        int(r['IDfg']): _safe_float(r.get('IP', 0)) or 0.0
        for _, r in actual_pitching.iterrows()
        if pd.notna(r.get('IDfg'))
    }
    pitcher_overlay = pd.concat([
        blended_sp_df[blended_sp_df['Year'] == current_year].copy(),
        blended_rp_df[blended_rp_df['Year'] == current_year].copy(),
    ], ignore_index=True)
    if 'IP' in pitcher_overlay.columns:
        pitcher_overlay['IP'] = pitcher_overlay['IDfg'].map(
            lambda idfg: actual_ip_map.get(int(idfg), 0.0)
            if pd.notna(idfg) else 0.0
        )

    return (
        _overlay_current_year_rows(raw_batter_history, batter_overlay, current_year),
        _overlay_current_year_rows(raw_pitcher_history, pitcher_overlay, current_year),
        _overlay_current_year_rows(raw_fielding_history, fielding_overlay, current_year)
            if not fielding_overlay.empty else pd.DataFrame(),
    )


# =============================================================================
# Current-season data loaders
# =============================================================================

def _load_current_year_batter_xstats(current_year: int) -> pd.DataFrame:
    """Load current-year batter x-stats from the Statcast expected-stats file."""
    statcast_data_dir = Config.Paths.ROOT_DIR / 'data' / 'statcast'
    expected_file = (
        statcast_data_dir /
        f'statcast_batter_expected_stats_{current_year}_{current_year}.csv'
    )
    if not expected_file.exists():
        return pd.DataFrame()

    statcast_expected = pd.read_csv(expected_file, low_memory=False)

    if 'year' in statcast_expected.columns:
        statcast_expected = statcast_expected.rename(columns={'year': 'Year'})
    if 'Year' not in statcast_expected.columns:
        statcast_expected['Year'] = current_year

    # Map MLBAM IDs → FanGraphs IDs if IDfg is absent
    if 'IDfg' not in statcast_expected.columns:
        mlb_id_col = next(
            (c for c in ('player_id', 'mlbam_id', 'playerid')
             if c in statcast_expected.columns),
            None,
        )
        if mlb_id_col is not None:
            statcast_expected[mlb_id_col] = pd.to_numeric(
                statcast_expected[mlb_id_col], errors='coerce'
            )
            try:
                crosswalk_path = Config.Paths.CROSSWALK_FILE
                if crosswalk_path.exists():
                    cross = pd.read_csv(crosswalk_path, low_memory=False)
                    mlbam_col = next(
                        (c for c in cross.columns
                         if c.lower() in ('mlbam_id', 'player_id', 'mlbamid')),
                        None,
                    )
                    fg_col = next(
                        (c for c in cross.columns
                         if c.lower() in ('fg_id', 'fgid')),
                        None,
                    )
                    if mlbam_col and fg_col:
                        cross[mlbam_col] = pd.to_numeric(cross[mlbam_col], errors='coerce')
                        cross[fg_col]    = pd.to_numeric(cross[fg_col],    errors='coerce')
                        cross = cross.dropna(subset=[mlbam_col, fg_col])
                        mapping = dict(zip(
                            cross[mlbam_col].astype(int),
                            cross[fg_col].astype(int),
                        ))
                        statcast_expected['IDfg'] = statcast_expected[mlb_id_col].map(mapping)
            except Exception:
                pass

    def _find_col(df, token):
        token = token.lower()
        return next(
            (c for c in df.columns if token == c.lower() or token in c.lower()),
            None,
        )

    rename_map = {}
    for internal, token in [('sc_est_ba', 'est_ba'), ('sc_est_slg', 'est_slg'), ('sc_est_woba', 'est_woba')]:
        col = _find_col(statcast_expected, token)
        if col and col not in (internal, 'xBA', 'xSLG', 'xwOBA'):
            rename_map[col] = internal
    if rename_map:
        statcast_expected = statcast_expected.rename(columns=rename_map)

    for sc_col, x_col in [('sc_est_ba', 'xBA'), ('sc_est_slg', 'xSLG'), ('sc_est_woba', 'xwOBA')]:
        if sc_col in statcast_expected.columns and x_col not in statcast_expected.columns:
            statcast_expected[x_col] = statcast_expected[sc_col]

    keep = ['IDfg', 'Year']
    for col in statcast_expected.columns:
        cl = col.lower()
        if (cl.startswith('x') or 'expected' in cl or cl.startswith('sc_est')
                or cl.startswith('est_') or 'est_' in cl):
            keep.append(col)

    statcast_expected = statcast_expected[[c for c in keep if c in statcast_expected.columns]]
    statcast_expected['IDfg'] = pd.to_numeric(statcast_expected['IDfg'], errors='coerce')
    statcast_expected = statcast_expected.dropna(subset=['IDfg'])
    statcast_expected['IDfg'] = statcast_expected['IDfg'].astype(int)
    return statcast_expected


def load_current_season_actuals(current_year=None):
    """Load actual batting and pitching stats for the current season.

    Prefers current-season per-year files; falls back to the full historic
    CSV.  Falls back to current_year − 1 if no current-year data exists yet.

    Returns:
        Tuple (batting_df, pitching_df, actual_year)
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    hist_dir          = Config.Paths.HISTORIC_MLB_DIR
    current_season_dir = Config.Paths.DATA_DIR / 'current_season'

    def _read_actual_source(filename: str) -> pd.DataFrame:
        if 'batting' in filename:
            current_name = f"mlb_batting_data_{current_year}_{current_year}.csv"
        elif 'pitching' in filename:
            current_name = f"mlb_pitching_data_{current_year}_{current_year}.csv"
        elif 'fielding' in filename:
            current_name = f"mlb_fielding_data_{current_year}_{current_year}.csv"
        else:
            current_name = filename
        current_path = current_season_dir / current_name
        if current_path.exists():
            return pd.read_csv(current_path, low_memory=False)
        return pd.read_csv(hist_dir / filename, low_memory=False)

    batting  = _read_actual_source('mlb_batting_data_1950_2025_with_statcast.csv')
    pitching = _read_actual_source('mlb_pitching_data_1950_2025_with_statcast.csv')

    actual_year  = current_year
    bat_current  = batting[batting['Season'] == current_year]
    if bat_current.empty:
        actual_year = current_year - 1
        bat_current = batting[batting['Season'] == actual_year]
        logger.warning(
            f"No {current_year} batting data — falling back to {actual_year}. "
            "Run Phase 1 (merge) first to ingest current-season stats."
        )

    pit_current = pitching[pitching['Season'] == actual_year]

    bat_current = bat_current.copy()
    pit_current = pit_current.copy()
    bat_current['IDfg'] = pd.to_numeric(bat_current['IDfg'], errors='coerce')
    pit_current['IDfg'] = pd.to_numeric(pit_current['IDfg'], errors='coerce')
    bat_current['Year'] = bat_current['Season']

    xstats = _load_current_year_batter_xstats(actual_year)
    if not xstats.empty:
        bat_current = bat_current.merge(xstats, on=['IDfg', 'Year'], how='left')

    # Derive HBP% from counting stats
    for df_actual, pa_col in [(bat_current, 'PA'), (pit_current, 'TBF')]:
        if 'HBP' in df_actual.columns and pa_col in df_actual.columns:
            pa_vals  = pd.to_numeric(df_actual[pa_col], errors='coerce').fillna(0)
            hbp_vals = pd.to_numeric(df_actual['HBP'],  errors='coerce').fillna(0)
            df_actual['HBP%'] = np.where(pa_vals > 0, hbp_vals / pa_vals, 0.0)

    logger.info(
        f"Loaded {actual_year} actuals (with_statcast): "
        f"{len(bat_current)} batters, {len(pit_current)} pitchers"
    )
    return bat_current, pit_current, actual_year


# =============================================================================
# Batter blending
# =============================================================================

# UNUSED as of the batter-blend removal (see blend_batter_projections) —
# these fed the per-feature blend multiplier that no longer runs, since
# Marcel now recomputes the full multivariate equation (including these
# features) from in-season data directly, rather than blending a separate
# post-hoc adjustment on top of a stale preseason number. Left in place
# only in case something outside this module still imports them; safe to
# delete once confirmed unused repo-wide.
_BATTER_AUX_WEIGHT_MULTIPLIERS = {
    # Highly sticky — stabilize in ~50–100 PA; use current season more
    'Contact%':  2.0,
    'O-Swing%':  2.0,
    'Z-Contact%': 2.0,
    'O-Contact%': 2.0,
    'Swing%':    2.0,
    'Pull%':     1.5,
    'Oppo%':     1.5,
    'F-Strike%': 1.5,
    # Moderate stabilization
    'Hard%':     1.2,
    'HardHit%':  1.2,
    'Barrel%':   1.2,
    'IFFB%':     1.0,
    'FB%':       1.0,
    # Noisy — stabilize slowly; stay conservative
    'sc_ev50':   0.8,
    'EV':        0.8,
    'Spd':       0.8,
    'xSLG':      0.8,
    'OBP':       0.8,
}

# Pre-compute the set of batter aux features once at import time
_BATTER_AUX_FEATURES: set = {
    feat
    for eq in BATTER_MULTIVARIATE_EQUATIONS.values()
    for feat in eq
    if not feat.startswith('_') and feat not in BATTER_BASE_COMPONENTS
}

# Pitcher aux features (same idea, computed once)
_PITCHER_AUX_FEATURES: set = {
    feat
    for eq_dict in (
        PITCHER_SP_MULTIVARIATE_EQUATIONS,
        PITCHER_RP_MULTIVARIATE_EQUATIONS,
    )
    for eq in eq_dict.values()
    for feat in eq
    if not feat.startswith('_') and feat not in PITCHER_MARCEL_RATE_STATS
}

_PITCHER_DERIVED_STATS = frozenset({
    'ERA', 'FIP', 'SIERA', 'K/9', 'BB/9', 'HR/9', 'WHIP',
    'G', 'GS', 'IP', 'TBF', 'H', 'R', 'ER', 'BB', 'K', 'HR',
    'HBP', 'WP', 'BK', 'W', 'L', 'SV', 'BS', 'Pit', 'Str', 'CStr%',
})


def blend_batter_projections(
    preseason_df,
    actual_batting,
    current_year=None,
    team_games_map=None,
    player_team_map=None,
):
    """Compute WAR-proration info for the current batter season.

    DEPRECATED (rate-stat blending removed): this function used to also
    overwrite each current-year row's base components (K%, BB%, HBP%, ISO,
    BABIP, HR/FB, GB%, LD%) by blending actual current-season stats with the
    *pre-season* Marcel projection, using a fixed 12-unit Marcel-weight
    formula. That is no longer done here.

    Why: the pre-season projection this used to blend against was computed
    once, before the season started (cutoff_year = last completed season),
    and for players with little/no prior MLB history it was built mostly or
    entirely from the MiLB-translated prior. marcel_batter_projections() is
    now re-run with cutoff_year = the CURRENT season (raw_df already carries
    that season's stats-to-date), and emits the current year directly at
    year_offset=0 from a freshly recomputed year1_base — which re-derives
    the Marcel weighted average AND the MiLB-prior blend weight
    (mlb_weight = career_pa / (career_pa + 400)) using up-to-date career PA.
    That is a strictly better-informed number than this function's blend
    ever was, and — critically — it's now built from the *same* year1_base
    that year_offset=1 (next year) also aggregates from. Blending here on
    top of it would double-count the current season's PA through two
    different, inconsistent formulas, which is exactly what caused the
    current-year projection to diverge sharply from next year's.

    preseason_df (the batter_predictions.csv rows for `current_year`) should
    now be treated as already correct and passed through unmodified.

    What THIS function still does: compute `war_proration`, i.e. each
    player's actual banked WAR-to-date and their remaining-season fraction,
    for use by prorate_current_year_war / prorate_current_year_salary. That
    is a "how much season is left" calculation, unrelated to which rate-stat
    projection is correct, and is unaffected by this change.

    Args:
        preseason_df:    Batter projection DataFrame (Year == current_year rows
                         are read, not modified, to look up actual PA/WAR).
        actual_batting:  Current-season FanGraphs batting DataFrame.
        current_year:    Season to prorate (default CURRENT_YEAR).
        team_games_map:  Dict mapping team abbreviation → games played this season.
        player_team_map: Dict mapping IDfg → team abbreviation.

    Returns:
        Tuple (preseason_df unchanged, war_proration_info)
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    use_team_remaining = bool(team_games_map and player_team_map)

    df            = preseason_df  # no longer copied/mutated — passthrough
    war_proration = {}

    actual_lookup = {
        int(row['IDfg']): row
        for _, row in actual_batting.iterrows()
        if pd.notna(row['IDfg'])
    }

    current_mask = df['Year'] == current_year
    prorated     = 0

    for idx in df.index[current_mask]:
        idfg   = int(df.at[idx, 'IDfg'])
        actual = actual_lookup.get(idfg)

        if use_team_remaining:
            team           = player_team_map.get(idfg)
            remaining_frac = _team_remaining_fraction(team, team_games_map)
        else:
            remaining_frac = 1.0

        if actual is None:
            war_proration[idfg] = {
                'actual_war': 0.0,
                'remaining_fraction': remaining_frac,
            }
            continue

        actual_pa  = _safe_float(actual.get('PA', 0)) or 0.0
        actual_war = _safe_float(actual.get('WAR', 0)) or 0.0

        if actual_pa < MIN_BATTER_PA:
            war_proration[idfg] = {
                'actual_war': 0.0,
                'remaining_fraction': remaining_frac,
            }
            continue

        if not use_team_remaining:
            remaining_frac = max(0.0, 1.0 - actual_pa / FULL_SEASON_PA)

        war_proration[idfg] = {
            'actual_war':        actual_war,
            'remaining_fraction': remaining_frac,
        }
        prorated += 1

    logger.info(
        f"Computed WAR proration for {prorated} current-year batters "
        f"(rate-stat blending removed — Marcel year1_base is now the sole "
        f"source of current-year rate stats; min_PA={MIN_BATTER_PA})"
    )
    return df, war_proration


# =============================================================================
# Pitcher blending
# =============================================================================

def blend_pitcher_projections(
    preseason_sp,
    preseason_rp,
    actual_pitching,
    current_year=None,
    team_games_map=None,
    player_team_map=None,
):
    """Blend current-year pitcher base components with pre-season projections.

    Uses Marcel-consistent weighting with the same philosophy as batter
    blending: the current partial IP earns weight proportional to its share
    of a full SP/RP season, capped at the Marcel weight-5 slot.

    Blends 8 base components (K%, BB%, HBP%, BABIP, HR/FB, GB%, FB%, LD%).
    After blending, reconstructs FIP, ERA, SIERA, K/9, BB/9, HR/9 from
    the blended components.  ERA-FIP gap is read from the preseason row and
    preserved (not blended directly).

    Args:
        preseason_sp:    SP preseason projection DataFrame.
        preseason_rp:    RP preseason projection DataFrame.
        actual_pitching: Current-season FanGraphs pitching DataFrame.
        current_year:    Season to blend (default CURRENT_YEAR).
        team_games_map:  Dict mapping team abbreviation → games played.
        player_team_map: Dict mapping IDfg → team abbreviation.

    Returns:
        Tuple (blended_sp_df, blended_rp_df, war_proration_info)
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    use_team_remaining = bool(team_games_map and player_team_map)

    sp_df = preseason_sp.copy()
    rp_df = preseason_rp.copy()
    war_proration = {}

    # Copy next-year rows into current_year slot if needed
    def _map_next_year(df):
        if df is None or df.empty:
            return df
        if 'Year' in df.columns and not (df['Year'] == current_year).any():
            ny = current_year + 1
            if (df['Year'] == ny).any():
                tmp = df[df['Year'] == ny].copy()
                tmp['Year'] = current_year
                return tmp
        return df

    sp_df = _map_next_year(sp_df)
    rp_df = _map_next_year(rp_df)

    actual_lookup = {
        int(row['IDfg']): row
        for _, row in actual_pitching.iterrows()
        if pd.notna(row['IDfg'])
    }

    blended_count = 0

    for df, role, full_ip in [
        (sp_df, 'SP', FULL_SEASON_SP_IP),
        (rp_df, 'RP', FULL_SEASON_RP_IP),
    ]:
        current_mask    = df['Year'] == current_year
        blended_indices = []

        for idx in df.index[current_mask]:
            pred_row = df.loc[idx]
            idfg     = int(pred_row['IDfg'])
            actual   = actual_lookup.get(idfg)

            if use_team_remaining:
                team           = player_team_map.get(idfg)
                remaining_frac = _team_remaining_fraction(team, team_games_map)
            else:
                remaining_frac = 1.0

            if actual is None:
                war_proration[idfg] = {
                    'actual_war': 0.0,
                    'remaining_fraction': remaining_frac,
                }
                continue

            actual_ip  = _safe_float(actual.get('IP', 0)) or 0.0
            actual_war = _safe_float(actual.get('WAR', 0)) or 0.0

            if actual_ip < MIN_PITCHER_IP:
                war_proration[idfg] = {
                    'actual_war': 0.0,
                    'remaining_fraction': remaining_frac,
                }
                continue

            # ── Marcel-consistent blend weight ───────────────────────────
            w = _marcel_blend_weight(actual_ip, full_ip)

            # ── Blend 8 base components ──────────────────────────────────
            for stat in PITCHER_MARCEL_RATE_STATS:
                if stat not in actual.index or stat not in df.columns:
                    continue
                act_val = _safe_float(actual[stat])
                pre_val = _safe_float(pred_row[stat])
                if act_val is None or pre_val is None:
                    continue
                df.at[idx, stat] = w * act_val + (1.0 - w) * pre_val

            # ── Blend auxiliary Phase 2b features ───────────────────────
            p_aux = _PITCHER_AUX_FEATURES - _PITCHER_DERIVED_STATS
            for feat in p_aux:
                if feat not in actual.index or feat not in df.columns:
                    continue
                act_val = _safe_float(actual[feat])
                pre_val = _safe_float(pred_row.get(feat))
                if act_val is None or pre_val is None:
                    continue
                df.at[idx, feat] = w * act_val + (1.0 - w) * pre_val

            blended_indices.append(idx)

            if not use_team_remaining:
                remaining_frac = max(0.0, 1.0 - actual_ip / full_ip)

            war_proration[idfg] = {
                'actual_war':        actual_war,
                'remaining_fraction': remaining_frac,
            }
            blended_count += 1

        # ── Reconstruct derived pitcher stats from blended components ────
        if blended_indices:
            for idx in blended_indices:
                row = df.loc[idx]

                k_pct   = _safe_float(row.get('K%',    0.22)) or 0.22
                bb_pct  = _safe_float(row.get('BB%',   0.08)) or 0.08
                hbp_pct = _safe_float(row.get('HBP%',  0.01)) or 0.01
                babip   = _safe_float(row.get('BABIP',  0.29)) or 0.29
                hr_fb   = _safe_float(row.get('HR/FB',  0.11)) or 0.11
                gb_pct  = _safe_float(row.get('GB%',    0.43)) or 0.43
                fb_pct  = _safe_float(row.get('FB%',    0.35)) or 0.35
                ld_pct  = _safe_float(row.get('LD%',    0.22)) or 0.22

                # Normalize batted-ball rates to sum to 1.0
                bb_total = gb_pct + fb_pct + ld_pct
                if bb_total > 0 and abs(bb_total - 1.0) > 0.01:
                    gb_pct /= bb_total
                    fb_pct /= bb_total
                    ld_pct /= bb_total
                    df.at[idx, 'GB%'] = gb_pct
                    df.at[idx, 'FB%'] = fb_pct
                    df.at[idx, 'LD%'] = ld_pct

                bip_rate  = max(1.0 - k_pct - bb_pct - hbp_pct, 0.20)
                hr_pct    = hr_fb * fb_pct * bip_rate
                bf_per_ip = _derive_bf_per_ip(k_pct, bb_pct, hbp_pct, hr_pct, babip)

                fip = _reconstruct_fip(k_pct, bb_pct, hbp_pct, hr_fb, fb_pct, bf_per_ip)
                df.at[idx, 'FIP'] = fip

                # Preserve the preseason ERA-FIP gap (pitcher-specific tendency)
                pre_era = _safe_float(row.get('ERA'))
                pre_fip = _safe_float(row.get('FIP'))
                era_fip_gap = (pre_era - pre_fip) if (pre_era is not None and pre_fip is not None) else 0.0
                df.at[idx, 'ERA'] = float(np.clip(fip + era_fip_gap, 0.5, 10.0))

                df.at[idx, 'SIERA'] = _reconstruct_siera(k_pct, bb_pct, gb_pct)
                df.at[idx, 'K/9']   = float(np.clip(k_pct   * bf_per_ip * 9.0 / 3.0, 0.0, 20.0))
                df.at[idx, 'BB/9']  = float(np.clip(bb_pct  * bf_per_ip * 9.0 / 3.0, 0.0, 10.0))
                df.at[idx, 'HR/9']  = float(np.clip(hr_pct  * bf_per_ip * 9.0 / 3.0, 0.0,  5.0))

    logger.info(
        f"Blended {blended_count} current-year pitcher projections "
        f"(Marcel-weighted 8 base components → reconstruct FIP/ERA/SIERA; "
        f"min_IP={MIN_PITCHER_IP})"
    )
    return sp_df, rp_df, war_proration


# =============================================================================
# Playing-time reduction to remaining season
# =============================================================================

# Counting stats that scale with playing time
BATTER_COUNTING_COLS = [
    'PA', 'G', 'HR', '2B', '3B', '1B', 'H', 'AB', 'K',
    'BB_count', 'HBP_count', 'RBI', 'R', 'HBP', 'BB',
    'SB', 'CS', 'SF',
]
PITCHER_COUNTING_COLS = [
    'IP', 'GS', 'G', 'W', 'L', 'SO', 'H', 'HR',
    'BB', 'HBP', 'ER', 'R', 'TBF',
]


# =============================================================================
# Next-year de-aging
# =============================================================================
# marcel_batter_projections() / marcel_pitcher_projections() build the
# (cutoff_year + 1) row as:
#
#     next_year_row[stat] = year1_base[stat] + aging_delta(stat, age_next_year)
#
# where year1_base IS the current-year talent-level estimate, before that
# one year of aging gets added. For a mid-season call-up, Round 1 (preseason)
# never ran for them — they didn't exist in the league yet — so they have no
# current-year row. But Round 2 (updated, cutoff = current_year) DOES compute
# their next-year row once they've debuted, using their real current-year
# stats as history. Reversing that one aging step recovers year1_base
# directly, using the same aging curves Marcel used to build it in the first
# place. The result is methodologically identical to every other player's
# preseason row — just computed a year late.
#
# This deliberately does NOT touch players who are also missing from the
# next-year projection (e.g. someone with only a handful of PA who hasn't
# cleared Round 2's own inclusion threshold yet). Those are rare and
# self-resolving: once they clear the threshold on a later day's run, this
# picks them up automatically.
# =============================================================================
 
_AGING_CURVES_CACHE = None
 
 
def _aging_curves():
    global _AGING_CURVES_CACHE
    if _AGING_CURVES_CACHE is None:
        _AGING_CURVES_CACHE = _load_aging_curves()
    return _AGING_CURVES_CACHE
 
 
def derive_missing_batter_baselines(
    batter_data: pd.DataFrame,
    current_year: int,
    actual_batting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build current-year batter baseline rows from each player's own
    (current_year + 1) Marcel row, for players who have a next-year
    projection but no current-year one.
 
    Returns a DataFrame of new rows (possibly empty), same columns as
    batter_data, ready to pd.concat onto it BEFORE blend_batter_projections()
    runs.
    """
    next_year = current_year + 1
    empty = batter_data.iloc[0:0].copy()
 
    if 'Year' not in batter_data.columns or 'IDfg' not in batter_data.columns:
        return empty
 
    cur_ids = set(
        batter_data.loc[batter_data['Year'] == current_year, 'IDfg']
        .dropna().astype(int)
    )
    next_df = batter_data[batter_data['Year'] == next_year].copy()
    if next_df.empty:
        return empty
 
    next_df['IDfg'] = next_df['IDfg'].astype(int)
    missing = next_df[~next_df['IDfg'].isin(cur_ids)].copy()
    if missing.empty:
        return empty
 
    curves = _aging_curves().get('batting', {})
 
    new_rows = []
    for _, row in missing.iterrows():
        row = row.copy()
        age = _safe_float(row.get('Age'))
 
        for stat in BATTER_BASE_COMPONENTS:
            val = _safe_float(row.get(stat))
            if val is None:
                continue
            stat_curves = curves.get(stat, {})
            delta = (
                _get_smoothed_aging_delta(stat_curves, int(age))
                if age is not None else 0.0
            )
            row[stat] = val - delta
 
        row['Year'] = current_year
        if 'prediction_year' in row.index:
            row['prediction_year'] = current_year
        if age is not None:
            row['Age'] = age - 1
 
        new_rows.append(row)
 
    new_df = pd.DataFrame(new_rows).reset_index(drop=True)
 
    # Renormalize batted-ball rates the same way Marcel does before composing
    if 'GB%' in new_df.columns and 'LD%' in new_df.columns:
        gb = pd.to_numeric(new_df['GB%'], errors='coerce').fillna(0.0)
        ld = pd.to_numeric(new_df['LD%'], errors='coerce').fillna(0.0)
        over = (gb + ld) >= 1.0
        if over.any():
            total = (gb + ld).clip(lower=1e-9)
            new_df.loc[over, 'GB%'] = (gb / total * 0.90)[over]
            new_df.loc[over, 'LD%'] = (ld / total * 0.90)[over]
 
    recomposed = compose_from_df(new_df)
    for col in ['AVG', 'SLG', 'OBP', 'wOBA', 'wRC+', 'FB%', 'HR', '2B', '3B', '1B', 'H']:
        if col in recomposed.columns:
            new_df[col] = recomposed[col].values
 
    logger.info(
        f"derive_missing_batter_baselines: derived {len(new_df)} current-year "
        f"({current_year}) baselines from next-year ({next_year}) Marcel "
        f"projections"
    )
    return new_df
 
 
def derive_missing_pitcher_baselines(
    pitcher_df: pd.DataFrame,
    current_year: int,
    role: str,
    actual_pitching: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Same idea as derive_missing_batter_baselines(), for pitchers.
 
    Args:
        pitcher_df: sp_data or rp_data, pre-filtered to one role — call
                    once per role, mirroring how blend_pitcher_projections()
                    handles SP/RP separately.
        role:       'SP' or 'RP', used only for logging.
    """
    next_year = current_year + 1
    empty = pitcher_df.iloc[0:0].copy()
 
    if 'Year' not in pitcher_df.columns or 'IDfg' not in pitcher_df.columns:
        return empty
 
    cur_ids = set(
        pitcher_df.loc[pitcher_df['Year'] == current_year, 'IDfg']
        .dropna().astype(int)
    )
    next_df = pitcher_df[pitcher_df['Year'] == next_year].copy()
    if next_df.empty:
        return empty
 
    next_df['IDfg'] = next_df['IDfg'].astype(int)
    missing = next_df[~next_df['IDfg'].isin(cur_ids)].copy()
    if missing.empty:
        return empty
 
    curves = _aging_curves().get('pitching', {})
 
    new_rows = []
    for _, row in missing.iterrows():
        row = row.copy()
        age = _safe_float(row.get('Age'))
 
        # Capture this pitcher's own ERA-FIP gap before we touch ERA/FIP —
        # Marcel treats it as a fixed career tendency, not age-dependent,
        # and it's already baked into the next-year row.
        pre_era = _safe_float(row.get('ERA'))
        pre_fip = _safe_float(row.get('FIP'))
        era_fip_gap = (pre_era - pre_fip) if (pre_era is not None and pre_fip is not None) else 0.0
 
        for stat in PITCHER_MARCEL_RATE_STATS:
            val = _safe_float(row.get(stat))
            if val is None:
                continue
            stat_curves = curves.get(stat, {})
            delta = (
                _get_smoothed_aging_delta(stat_curves, int(age))
                if age is not None else 0.0
            )
            row[stat] = val - delta
 
        row['Year'] = current_year
        if 'prediction_year' in row.index:
            row['prediction_year'] = current_year
        if age is not None:
            row['Age'] = age - 1
 
        # Reconstruct FIP/ERA/SIERA/K9/BB9/HR9 — same formulas
        # blend_pitcher_projections() uses after blending.
        k_pct   = _safe_float(row.get('K%',    0.22)) or 0.22
        bb_pct  = _safe_float(row.get('BB%',   0.08)) or 0.08
        hbp_pct = _safe_float(row.get('HBP%',  0.01)) or 0.01
        babip   = _safe_float(row.get('BABIP', 0.29)) or 0.29
        hr_fb   = _safe_float(row.get('HR/FB', 0.11)) or 0.11
        gb_pct  = _safe_float(row.get('GB%',   0.43)) or 0.43
        fb_pct  = _safe_float(row.get('FB%',   0.35)) or 0.35
        ld_pct  = _safe_float(row.get('LD%',   0.22)) or 0.22
 
        bb_total = gb_pct + fb_pct + ld_pct
        if bb_total > 0 and abs(bb_total - 1.0) > 0.01:
            gb_pct /= bb_total
            fb_pct /= bb_total
            ld_pct /= bb_total
            row['GB%'], row['FB%'], row['LD%'] = gb_pct, fb_pct, ld_pct
 
        bip_rate  = max(1.0 - k_pct - bb_pct - hbp_pct, 0.20)
        hr_pct    = hr_fb * fb_pct * bip_rate
        bf_per_ip = _derive_bf_per_ip(k_pct, bb_pct, hbp_pct, hr_pct, babip)
 
        fip = _reconstruct_fip(k_pct, bb_pct, hbp_pct, hr_fb, fb_pct, bf_per_ip)
        row['FIP']   = fip
        row['ERA']   = float(np.clip(fip + era_fip_gap, 0.5, 10.0))
        row['SIERA'] = _reconstruct_siera(k_pct, bb_pct, gb_pct)
        row['K/9']   = float(np.clip(k_pct  * bf_per_ip * 9.0 / 3.0, 0.0, 20.0))
        row['BB/9']  = float(np.clip(bb_pct * bf_per_ip * 9.0 / 3.0, 0.0, 10.0))
        row['HR/9']  = float(np.clip(hr_pct * bf_per_ip * 9.0 / 3.0, 0.0,  5.0))
 
        new_rows.append(row)
 
    new_df = pd.DataFrame(new_rows).reset_index(drop=True)
    logger.info(
        f"derive_missing_pitcher_baselines: derived {len(new_df)} current-year "
        f"({current_year}) {role} baselines from next-year ({next_year}) "
        f"Marcel projections"
    )
    return new_df

def reduce_to_remaining_season(
    df,
    war_proration,
    current_year=None,
    player_type='batter',
):
    """Scale current-year counting stats to the remaining season.

    Multiplies all counting stats for the current year by remaining_fraction
    so the projection represents what the player is expected to accumulate
    from today through the end of the season, not the full year.

    Rate stats (AVG, ERA, K%, etc.) are left untouched — they represent
    expected performance level, not volume.

    Sets playing_time_reduced=True in each player's war_proration entry so
    that prorate_current_year_war uses direct addition (actual + projected_ROS)
    rather than re-multiplying by remaining_fraction.

    Args:
        df:            Prediction DataFrame (must have IDfg, Year columns).
        war_proration: Dict mapping IDfg → {actual_war, remaining_fraction}.
        current_year:  Season to reduce.
        player_type:   'batter' or 'pitcher'.

    Returns:
        DataFrame with reduced current-year counting stats.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    counting_cols = (
        BATTER_COUNTING_COLS if player_type == 'batter' else PITCHER_COUNTING_COLS
    )

    df      = df.copy()
    reduced = 0

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
            continue

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


# =============================================================================
# WAR and salary proration
# =============================================================================

def prorate_current_year_war(df, war_proration, current_year=None):
    """Replace current-year WAR with ROS projected WAR.

    If playing time was already reduced via reduce_to_remaining_season, the
    projected WAR is already ROS-scale and is used directly::

        new_WAR = projected_ros_war

    Otherwise (full-season projection)::

        new_WAR = projected_full_war × remaining_fraction

    Future years (Year > current_year) are untouched.

    Args:
        df:            DataFrame with IDfg, Year, WAR columns.
        war_proration: Dict mapping IDfg → {actual_war, remaining_fraction,
                       playing_time_reduced (optional)}.
        current_year:  Season to prorate (default CURRENT_YEAR).

    Returns:
        DataFrame with prorated current-year WAR.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    df       = df.copy()
    prorated = 0

    current_mask = df['Year'] == current_year
    for idx in df.index[current_mask]:
        try:
            idfg = int(df.at[idx, 'IDfg'])
        except (ValueError, TypeError):
            continue

        info = war_proration.get(idfg)
        if info is None:
            continue

        projected_war = df.at[idx, 'WAR']

        if info.get('playing_time_reduced', False):
            prorated_war = projected_war
        else:
            prorated_war = projected_war * info['remaining_fraction']

        df.at[idx, 'WAR'] = prorated_war
        prorated += 1

    logger.info(
        f"Prorated {prorated} current-year WAR values "
        f"(actual + projected ROS)"
    )
    return df


def prorate_current_year_salary(df, war_proration, current_year=None):
    """Prorate current-year contract_value to reflect only remaining salary.

    Formula::
        new_contract_value = full_year_contract × remaining_fraction

    Future years are untouched.

    Args:
        df:            DataFrame with IDfg, Year, contract_value columns.
        war_proration: Dict mapping IDfg → {remaining_fraction, ...}.
        current_year:  Season to prorate (default CURRENT_YEAR).

    Returns:
        DataFrame with prorated current-year contract_value.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    df       = df.copy()
    prorated = 0

    if 'contract_value' not in df.columns:
        logger.warning("contract_value column not found; skipping salary proration")
        return df

    current_mask = df['Year'] == current_year
    for idx in df.index[current_mask]:
        try:
            idfg = int(df.at[idx, 'IDfg'])
        except (ValueError, TypeError):
            continue

        info = war_proration.get(idfg)
        if info is None:
            continue

        remaining_frac = info.get('remaining_fraction', 1.0)
        if remaining_frac >= 1.0:
            continue

        original_contract = df.at[idx, 'contract_value']
        if pd.isna(original_contract) or original_contract == 0:
            continue

        df.at[idx, 'contract_value'] = original_contract * remaining_frac
        prorated += 1

    logger.info(
        f"Prorated {prorated} current-year contract_value entries "
        f"to remaining season"
    )
    return df


# =============================================================================
# Fielding blending
# =============================================================================

def blend_fielding_projections(fielding_df, actual_batting, current_year=None):
    """Blend preseason fielding projections with actual Fld runs from FanGraphs.

    Uses Marcel-consistent weighting calibrated to fielding's stabilization
    properties.  Fielding is very noisy — the Marcel engine itself uses
    550–1100 innings of regression (vs ~400 PA for batting).  The blend
    weight is scaled to respect that: a full season of fielding (~1350 innings
    for a regular) earns the Marcel weight-5 slot, so early-season data has
    very little influence.

    The FanGraphs batting CSV ``Fld`` column (total fielding runs for the
    season) is converted to a per-150-game rate before blending with the
    preseason per-150 projection.

    Args:
        fielding_df:    Preseason fielding prediction DataFrame with columns
                        IDfg, Year, sc_total_runs/150.
        actual_batting: Current-season FanGraphs batting DataFrame
                        (must include IDfg, G, Fld).
        current_year:   Season to blend (default CURRENT_YEAR).

    Returns:
        Updated fielding_df with blended current-year sc_total_runs/150 values.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    df           = fielding_df.copy()
    current_mask = df['Year'] == current_year
    if not current_mask.any():
        return df

    # Fielding stabilization: ~1350 innings for a full-time player.
    # We use the outfield regression constant (550 inn) as the reference
    # denominator so that a full-season outfielder earns roughly the Marcel
    # weight-5 slot — the most conservative group.  Infield and catcher are
    # even noisier (up to 1100 inn regression) so this is a reasonable middle.
    # In practice, the weight grows slowly: at 500 inn (~81 games) w ≈ 0.16.
    FULL_SEASON_FIELDING_INN = 1350.0

    actual_lookup = {}
    for _, row in actual_batting.iterrows():
        idfg  = row.get('IDfg')
        if pd.isna(idfg):
            continue
        fld   = _safe_float(row.get('Fld'))
        games = _safe_float(row.get('G', 0)) or 0.0
        if fld is not None and games >= MIN_FIELDING_G:
            actual_lookup[int(idfg)] = (fld, games)

    blended_count = 0
    for idx in df.index[current_mask]:
        idfg        = int(df.at[idx, 'IDfg'])
        actual_data = actual_lookup.get(idfg)
        if actual_data is None:
            continue

        actual_fld, games = actual_data
        if games <= 0:
            continue

        # Approximate innings from games (regular ≈ ~8.5 inn/game in the field)
        approx_inn = games * 8.5

        # Marcel-consistent weight for fielding
        w = _marcel_blend_weight(approx_inn, FULL_SEASON_FIELDING_INN)

        # Convert actual Fld (seasonal total in 'games' games) → per-150 rate
        actual_per_150 = actual_fld * (150.0 / games)
        pre_per_150    = _safe_float(df.at[idx, 'sc_total_runs/150']) or 0.0

        df.at[idx, 'sc_total_runs/150'] = w * actual_per_150 + (1.0 - w) * pre_per_150
        blended_count += 1

    logger.info(
        f"Blended {blended_count} current-year fielding projections "
        f"(Marcel-weighted Fld per-150; min_G={MIN_FIELDING_G})"
    )
    return df


# =============================================================================
# Fielding baseline derivation (mid-season call-ups / signings)
# =============================================================================
# blend_fielding_projections() above only ever *blends* actuals into rows
# that already exist for current_year — it has no way to create a row for
# a player who has none. Batters and pitchers get a current-year row
# derived from their (current_year + 1) Marcel projection via
# derive_missing_batter_baselines() / derive_missing_pitcher_baselines(),
# but fielding_data never got the same treatment. A mid-season call-up
# (Round 1 predates their debut) ends up with ZERO fielding_data rows for
# ANY year, so calculate_defensive_value()'s
#     player_fielding = fielding_data[(IDfg == id) & (Year == year)]
# is empty for every year, weighted_fld is permanently 0.0, and Def
# collapses onto the (also-frozen) positional adjustment alone — the same
# dead number on every projection row, with no aging curve to move it.
#
# Unlike the batter/pitcher case, there's no fielding aging curve to
# reverse. But we don't need one: a call-up's real current-season
# defensive performance (already confirmed live-updated in the actuals
# data) is a better current-year estimate than a model guess would be
# anyway. So this seeds the current-year row directly from actual
# sc_total_runs/150 when available, and only falls back to carrying the
# (current_year + 1) rate forward unchanged if they haven't yet cleared
# MIN_FIELDING_G games in the field.
# =============================================================================


def derive_missing_fielding_baseline(
    fielding_df: pd.DataFrame,
    current_year_ids,
    actual_batting: pd.DataFrame,
    current_year: int = None,
    actual_fielding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build current-year fielding baseline rows for players with no
    preseason fielding row for current_year.

    FIXED: now also handles players who have *no* fielding projection at all
    (no current_year AND no next_year row) by creating a row directly from
    actual Fld/G. This covers rookies who didn't clear MIN_POSITION_INNINGS=50
    for any single position.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    empty = fielding_df.iloc[0:0].copy()
    if 'Year' not in fielding_df.columns or 'IDfg' not in fielding_df.columns:
        return empty

    next_year = current_year + 1
    id_set    = {int(i) for i in current_year_ids if pd.notna(i)}

    cur_ids = set(
        fielding_df.loc[fielding_df['Year'] == current_year, 'IDfg']
        .dropna().astype(int)
    )
    # All IDs that ever appear in fielding file (any year)
    all_fielding_ids = set(fielding_df['IDfg'].dropna().astype(int).unique())

    next_df = fielding_df[fielding_df['Year'] == next_year].copy()
    if not next_df.empty:
        next_df['IDfg'] = next_df['IDfg'].astype(int)
        missing = next_df[
            next_df['IDfg'].isin(id_set) & ~next_df['IDfg'].isin(cur_ids)
        ].copy()
    else:
        missing = empty.copy()

    # Real current-season defensive rate, keyed by IDfg
    actual_lookup = {}
    name_lookup = {}
    age_lookup = {}
    for _, row in actual_batting.iterrows():
        idfg = row.get('IDfg')
        if pd.isna(idfg):
            continue
        idfg = int(idfg)
        fld   = _safe_float(row.get('Fld'))
        games = _safe_float(row.get('G', 0)) or 0.0
        if fld is not None and games >= MIN_FIELDING_G:
            actual_lookup[idfg] = fld * (150.0 / games)
        # store name/age for fallback row creation
        if pd.notna(row.get('Name')):
            name_lookup[idfg] = str(row.get('Name'))
        if pd.notna(row.get('Age')):
            age_lookup[idfg] = _safe_float(row.get('Age'))

    # Pos lookup from actual_fielding if provided (most innings position)
    pos_lookup = {}
    group_lookup = {}
    if actual_fielding is not None and not actual_fielding.empty and 'IDfg' in actual_fielding.columns:
        try:
            af = actual_fielding.copy()
            af['IDfg'] = pd.to_numeric(af['IDfg'], errors='coerce')
            af = af.dropna(subset=['IDfg'])
            af['IDfg'] = af['IDfg'].astype(int)
            # Expect columns Pos, Inn or INN
            inn_col = next((c for c in ('Inn','INN','InnOuts') if c in af.columns), None)
            pos_col = 'Pos' if 'Pos' in af.columns else None
            if pos_col and inn_col:
                for pid, sub in af.groupby('IDfg'):
                    if pid not in id_set:
                        continue
                    # pick pos with max innings
                    sub = sub.sort_values(inn_col, ascending=False)
                    primary = sub.iloc[0][pos_col]
                    pos_lookup[pid] = str(primary)
                    group_lookup[pid] = POS_TO_GROUP.get(str(primary), 'infield')
            elif pos_col:
                for pid, sub in af.groupby('IDfg'):
                    pos_lookup[pid] = str(sub.iloc[0][pos_col])
                    group_lookup[pid] = POS_TO_GROUP.get(str(sub.iloc[0][pos_col]), 'infield')
        except Exception:
            pass

    new_rows = []
    derived_from_actual = 0
    handled_ids = set()

    for _, row in missing.iterrows():
        row  = row.copy()
        idfg = int(row['IDfg'])
        handled_ids.add(idfg)

        if idfg in actual_lookup:
            row['sc_total_runs/150'] = actual_lookup[idfg]
            derived_from_actual += 1

        # FIX name if Unknown
        if str(row.get('Name','')).lower().startswith('unknown') and idfg in name_lookup:
            row['Name'] = name_lookup[idfg]
        if pd.isna(row.get('Age')) and idfg in age_lookup:
            row['Age'] = age_lookup[idfg]

        # FIX Pos if we have better info from actual fielding
        if idfg in pos_lookup:
            row['Pos'] = pos_lookup[idfg]
            row['Position_Group'] = group_lookup.get(idfg, row.get('Position_Group','infield'))

        row['Year'] = current_year
        if 'prediction_year' in row.index:
            row['prediction_year'] = current_year

        # FIX: store as dict to keep list homogeneous (Series + dict mix breaks pd.DataFrame)
        new_rows.append(row.to_dict() if hasattr(row, 'to_dict') else dict(row))

    # --- NEW: truly missing players (no fielding row for current or next year) ---
    truly_missing_ids = id_set - cur_ids - handled_ids
    # include all missing players, even if they have no actual fielding yet (e.g. new rookies)
    # they will get a 0.0 baseline.

    for idfg in truly_missing_ids:
        # build minimal row from empty template
        base_row = {}
        # copy dtypes from empty if possible
        for col in fielding_df.columns:
            base_row[col] = pd.NA

        base_row['IDfg'] = idfg
        base_row['Year'] = current_year
        base_row['sc_total_runs/150'] = actual_lookup.get(idfg, 0.0)
        base_row['Name'] = name_lookup.get(idfg, f"Unknown ({idfg})")
        base_row['Age'] = age_lookup.get(idfg, 26)

        pos = pos_lookup.get(idfg, 'LF')
        base_row['Pos'] = pos
        base_row['Position_Group'] = group_lookup.get(idfg, POS_TO_GROUP.get(pos, 'outfield'))

        # fill other stat cols with 0 so they don't break downstream
        for c in ('sc_range_runs/150','sc_arm_runs/150','sc_dp_runs/150',
                  'sc_framing_runs/150','sc_throwing_runs/150','sc_blocking_runs/150'):
            if c in fielding_df.columns:
                base_row[c] = 0.0

        new_rows.append(base_row)
        derived_from_actual += 1

    if not new_rows:
        return empty

    new_df = pd.DataFrame(new_rows).reset_index(drop=True)
    # Ensure IDfg int
    if 'IDfg' in new_df.columns:
        new_df['IDfg'] = pd.to_numeric(new_df['IDfg'], errors='coerce').astype('Int64')

    logger.info(
        f"derive_missing_fielding_baseline: derived {len(new_df)} current-year "
        f"({current_year}) fielding baselines ({derived_from_actual} from actual "
        f"defensive stats, {len(new_df) - derived_from_actual} carried forward "
        f"from next-year projection; min_G={MIN_FIELDING_G}; truly_missing={len(truly_missing_ids)})"
    )
    return new_df


# =============================================================================
# Baserunning blending
# =============================================================================

def blend_baserunning_projections(baserunning_df, actual_batting, current_year=None):
    """Blend preseason baserunning projections with actual BsR from FanGraphs.

    Uses Marcel-consistent weighting calibrated to baserunning's stabilization
    properties.  Marcel regresses baserunning toward 0 with ~100 games of
    regression (BASERUNNING_REGRESSION_GAMES), meaning a full season (~150 G)
    earns roughly the weight-5 slot.  The blend weight respects this: at
    50 games w ≈ 0.22, at 100 games w ≈ 0.31, at 150 games w ≈ 0.38.

    The FanGraphs batting CSV ``BsR`` column (seasonal baserunning runs) is
    converted to a per-650-PA rate before blending.

    Args:
        baserunning_df: Preseason baserunning prediction DataFrame with columns
                        IDfg, Year, sc_baserunning_runner_runs_tot_rate.
        actual_batting: Current-season FanGraphs batting DataFrame
                        (must include IDfg, PA, BsR).
        current_year:   Season to blend (default CURRENT_YEAR).

    Returns:
        Updated baserunning_df with blended current-year rate values.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    df           = baserunning_df.copy()
    current_mask = df['Year'] == current_year
    if not current_mask.any():
        return df

    # Full-season baserunning volume (PA equivalent of ~150 games)
    # Marcel baserunning regression = 100 games; full season ≈ 150 games.
    # We use 150 G * (650 PA / 162 G) ≈ 600 PA as the full-season PA target,
    # which aligns with FULL_SEASON_PA and keeps the math consistent.
    FULL_SEASON_BR_PA = float(FULL_SEASON_PA)   # 650

    actual_lookup = {}
    for _, row in actual_batting.iterrows():
        idfg = row.get('IDfg')
        if pd.isna(idfg):
            continue
        bsr = _safe_float(row.get('BsR'))
        pa  = _safe_float(row.get('PA', 0)) or 0.0
        if bsr is not None and pa >= MIN_BASERUNNING_PA:
            actual_lookup[int(idfg)] = (bsr, pa)

    blended_count = 0
    for idx in df.index[current_mask]:
        idfg        = int(df.at[idx, 'IDfg'])
        actual_data = actual_lookup.get(idfg)
        if actual_data is None:
            continue

        actual_bsr, pa = actual_data
        if pa <= 0:
            continue

        # Marcel-consistent weight for baserunning
        w = _marcel_blend_weight(pa, FULL_SEASON_BR_PA)

        # Convert actual BsR (seasonal total in 'pa' PA) → per-650 rate
        actual_rate = actual_bsr * (FULL_SEASON_PA / pa)
        pre_rate    = _safe_float(df.at[idx, 'sc_baserunning_runner_runs_tot_rate']) or 0.0

        df.at[idx, 'sc_baserunning_runner_runs_tot_rate'] = (
            w * actual_rate + (1.0 - w) * pre_rate
        )
        blended_count += 1

    logger.info(
        f"Blended {blended_count} current-year baserunning projections "
        f"(Marcel-weighted BsR per-650; min_PA={MIN_BASERUNNING_PA})"
    )
    return df