"""
Unified Component-Based Marcel Projection Engine
================================================

Projects MLB batting, pitching, fielding, and baserunning components forward
up to 15 years using:
  1. 3-year recency- and volume-weighted averages (5/4/3 weights)
  2. Sample-size-scaled Statcast multivariate Year-1 equations
  3. Rookie/FV talent priors for low-experience MLB players (<1 full season)
  4. Empirically derived, bounded aging curves
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# AGING CURVE MANAGEMENT
# ============================================================================

# Path to the empirically-derived aging curves JSON
_AGING_CURVES_PATH = Path(__file__).parent.parent / 'analysis' / 'marcel_aging_curves.json'

# Cached curves (loaded once)
_cached_curves: Optional[dict] = None


def _load_aging_curves() -> dict:
    """Load and cache the aging curves from JSON."""
    global _cached_curves
    if _cached_curves is None:
        with open(_AGING_CURVES_PATH, 'r') as f:
            _cached_curves = json.load(f)
    return _cached_curves

def _safe_resolve_player_name(player_id: int, player_names: pd.DataFrame, raw_df: pd.DataFrame | None = None) -> str:
    """Resolve name with fallback to raw_df to avoid Unknown for debutants."""
    try:
        if player_names is not None and not player_names.empty and 'IDfg' in player_names.columns:
            m = player_names[player_names['IDfg'] == player_id]
            if not m.empty and 'Name' in m.columns and pd.notna(m.iloc[0]['Name']):
                return str(m.iloc[0]['Name'])
    except Exception:
        pass
    try:
        if raw_df is not None and not raw_df.empty and 'IDfg' in raw_df.columns and 'Name' in raw_df.columns:
            m2 = raw_df[raw_df['IDfg'] == player_id]
            if not m2.empty and pd.notna(m2.iloc[0]['Name']):
                return str(m2.iloc[0]['Name'])
    except Exception:
        pass
    return f"Unknown ({player_id})"



def _get_smoothed_aging_delta(
    raw_deltas: Dict[str, float],
    age: int,
    window: int = 3,
) -> float:
    """
    Get a smoothed aging delta for a given age.

    Uses a ±window moving average of the raw delta-method values.
    For ages outside the observed range, extrapolates from the nearest
    boundary value.

    Args:
        raw_deltas: {str(age): delta} from the derived curves
        age: The age to query (the age in the *later* year of the pair)
        window: Half-width of the smoothing window

    Returns:
        Smoothed per-year delta at this age
    """
    if not raw_deltas:
        return 0.0

    ages_int = sorted(int(a) for a in raw_deltas.keys())
    min_age, max_age = ages_int[0], ages_int[-1]

    if age < min_age:
        # Before observed range: use the youngest available delta
        return raw_deltas[str(min_age)]
    if age > max_age:
        # After observed range: use the oldest available delta
        return raw_deltas[str(max_age)]

    # Smoothing window
    neighbors = []
    for a in range(age - window, age + window + 1):
        if str(a) in raw_deltas:
            neighbors.append(raw_deltas[str(a)])

    if not neighbors:
        return 0.0

    return float(np.mean(neighbors))


# Minimum per-year aging floor for defensive stats (runs/150).
# These are ceiling values on the delta — if the empirical (smoothed) delta is
# above this threshold, the floor is used instead.  Defence declines faster
# than offensive production; the floor corrects for noisy empirical curves
# that can show near-zero decline in the 30s due to small-sample artefacts.
# Values are calibrated to consensus sabermetric priors:
#   - ~0.5 runs/yr decline through late-20s
#   - ~1.0-1.5 runs/yr through early-30s
#   - ~2.0+ runs/yr from mid-30s onward
FIELDING_AGING_FLOOR = {
    # sc_total_runs/150 — the primary total defensive value
    'sc_total_runs/150': {
        29: -0.5, 30: -0.75, 31: -1.0, 32: -1.25, 33: -1.5,
        34: -1.75, 35: -2.0, 36: -2.25, 37: -2.5, 38: -2.75,
        39: -3.0, 40: -3.25, 41: -3.5, 42: -3.75,
    },
    # Range runs — the biggest component of defensive decline
    'sc_range_runs/150': {
        29: -0.4, 30: -0.6, 31: -0.8, 32: -1.0, 33: -1.2,
        34: -1.4, 35: -1.6, 36: -1.8, 37: -2.0, 38: -2.2,
        39: -2.4, 40: -2.6, 41: -2.8, 42: -3.0,
    },
    # Arm / throwing / framing — decline slower
    'sc_arm_runs/150': {
        32: -0.3, 33: -0.4, 34: -0.5, 35: -0.6, 36: -0.7,
        37: -0.8, 38: -0.9, 39: -1.0, 40: -1.1, 41: -1.2, 42: -1.3,
    },
    'sc_dp_runs/150': {
        30: -0.2, 31: -0.3, 32: -0.4, 33: -0.5, 34: -0.6,
        35: -0.7, 36: -0.8, 37: -0.9, 38: -1.0, 39: -1.1,
        40: -1.2, 41: -1.3, 42: -1.4,
    },
    'sc_framing_runs/150': {
        32: -0.5, 33: -0.75, 34: -1.0, 35: -1.5, 36: -2.0,
        37: -2.5, 38: -3.0, 39: -3.5, 40: -4.0, 41: -4.5, 42: -5.0,
    },
    'sc_throwing_runs/150': {
        32: -0.2, 33: -0.3, 34: -0.4, 35: -0.5, 36: -0.6,
        37: -0.7, 38: -0.8, 39: -0.9, 40: -1.0, 41: -1.1, 42: -1.2,
    },
    'sc_blocking_runs/150': {
        32: -0.2, 33: -0.3, 34: -0.4, 35: -0.5, 36: -0.6,
        37: -0.7, 38: -0.8, 39: -0.9, 40: -1.0, 41: -1.1, 42: -1.2,
    },
    # Traditional stat equivalents (reuse total curve)
    'UZR/150': {
        29: -0.5, 30: -0.75, 31: -1.0, 32: -1.25, 33: -1.5,
        34: -1.75, 35: -2.0, 36: -2.25, 37: -2.5, 38: -2.75,
        39: -3.0, 40: -3.25, 41: -3.5, 42: -3.75,
    },
    'RngR/150': {
        29: -0.4, 30: -0.6, 31: -0.8, 32: -1.0, 33: -1.2,
        34: -1.4, 35: -1.6, 36: -1.8, 37: -2.0, 38: -2.2,
        39: -2.4, 40: -2.6, 41: -2.8, 42: -3.0,
    },
    'ARM/150': {
        32: -0.3, 33: -0.4, 34: -0.5, 35: -0.6, 36: -0.7,
        37: -0.8, 38: -0.9, 39: -1.0, 40: -1.1, 41: -1.2, 42: -1.3,
    },
    'DPR/150': {
        30: -0.2, 31: -0.3, 32: -0.4, 33: -0.5, 34: -0.6,
        35: -0.7, 36: -0.8, 37: -0.9, 38: -1.0, 39: -1.1,
        40: -1.2, 41: -1.3, 42: -1.4,
    },
    'DRS/150': {
        29: -0.5, 30: -0.75, 31: -1.0, 32: -1.25, 33: -1.5,
        34: -1.75, 35: -2.0, 36: -2.25, 37: -2.5, 38: -2.75,
        39: -3.0, 40: -3.25, 41: -3.5, 42: -3.75,
    },
}


# Maximum per-year aging growth for defensive stats (runs/150), applied to
# ages BEFORE peak (~27). These are ceilings on the delta — if the empirical
# (smoothed) delta is above this threshold, the cap is used instead.
#
# This mirrors FIELDING_AGING_FLOOR but for the opposite failure mode: the
# delta-method aging curve is derived from very few paired player-seasons at
# ages 20-24 (almost nobody gets meaningful defensive innings that young),
# and the players who do and who stick around long enough to have a next
# season are a survivorship-biased sample of good prospects who kept
# improving and playing more. That selection effect gets baked into the
# curve as an oversized "aging" delta. Left unbounded, it compounds across
# every year of a multi-year projection (see marcel_fielding_projections),
# producing unrealistic gains for players who debut very young — e.g. a
# 20-year-old debutant with a mediocre small-sample FRV projecting to
# near-Gold-Glove defense by their mid-20s.
#
# Values are calibrated to be modest relative to the decline-side floor:
# real defensive development from a rookie-quality debut to peak shouldn't
# plausibly sum to more than ~3-4 runs/150 total across the whole climb.
FIELDING_AGING_GROWTH_CAP: Dict[str, Dict[int, float]] = {
    'sc_total_runs/150': {
        20: 0.20, 21: 0.20, 22: 0.15, 23: 0.15, 24: 0.10, 25: 0.05, 26: 0.00,
    },
    'sc_range_runs/150': {
        20: 0.15, 21: 0.15, 22: 0.12, 23: 0.10, 24: 0.08, 25: 0.05, 26: 0.00,
    },
    'sc_arm_runs/150': {
        20: 0.10, 21: 0.10, 22: 0.08, 23: 0.05, 24: 0.05, 25: 0.00, 26: 0.00,
    },
    'sc_dp_runs/150': {
        20: 0.08, 21: 0.08, 22: 0.05, 23: 0.05, 24: 0.00, 25: 0.00, 26: 0.00,
    },
    'sc_framing_runs/150': {
        20: 0.15, 21: 0.15, 22: 0.12, 23: 0.10, 24: 0.08, 25: 0.05, 26: 0.00,
    },
    'sc_throwing_runs/150': {
        20: 0.08, 21: 0.08, 22: 0.05, 23: 0.05, 24: 0.00, 25: 0.00, 26: 0.00,
    },
    'sc_blocking_runs/150': {
        20: 0.08, 21: 0.08, 22: 0.05, 23: 0.05, 24: 0.00, 25: 0.00, 26: 0.00,
    },
    # Traditional stat equivalents (reuse total curve caps)
    'UZR/150': {
        20: 0.20, 21: 0.20, 22: 0.15, 23: 0.15, 24: 0.10, 25: 0.05, 26: 0.00,
    },
    'RngR/150': {
        20: 0.15, 21: 0.15, 22: 0.12, 23: 0.10, 24: 0.08, 25: 0.05, 26: 0.00,
    },
    'ARM/150': {
        20: 0.10, 21: 0.10, 22: 0.08, 23: 0.05, 24: 0.05, 25: 0.00, 26: 0.00,
    },
    'DPR/150': {
        20: 0.08, 21: 0.08, 22: 0.05, 23: 0.05, 24: 0.00, 25: 0.00, 26: 0.00,
    },
    'DRS/150': {
        20: 0.20, 21: 0.20, 22: 0.15, 23: 0.15, 24: 0.10, 25: 0.05, 26: 0.00,
    },
}


def _get_defensive_aging_delta(
    raw_deltas: Dict[str, float],
    age: int,
    stat: str,
    window: int = 3,
) -> float:
    """
    Get an aging delta for a defensive stat, bounded on both sides of peak.

    Same smoothing logic as _get_smoothed_aging_delta, but:
      - enforces a MINIMUM per-year decline after peak ages using
        FIELDING_AGING_FLOOR (if the smoothed empirical delta is above/
        less-negative-than the floor, the floor is used instead).
      - enforces a MAXIMUM per-year growth before peak ages using
        FIELDING_AGING_GROWTH_CAP (if the smoothed empirical delta is
        above the cap, the cap is used instead).

    Both bounds exist for the same reason: the empirical delta-method curve
    is noisiest — and most prone to selection-bias artefacts — at the
    extreme ends of the observed age range, and an unconstrained per-year
    delta compounds across every year of a multi-year projection.
    """
    smoothed = _get_smoothed_aging_delta(raw_deltas, age, window=window)

    floor_map = FIELDING_AGING_FLOOR.get(stat, {})
    if floor_map:
        max_floor_age = max(floor_map.keys())
        if age > max_floor_age:
            last_floor = floor_map[max_floor_age]
            # Continue accelerating decline at 0.25 runs/yr beyond last entry
            floor = last_floor - 0.25 * (age - max_floor_age)
            return min(smoothed, floor)
        elif age in floor_map:
            return min(smoothed, floor_map[age])

    cap_map = FIELDING_AGING_GROWTH_CAP.get(stat, {})
    if cap_map and age in cap_map:
        return min(smoothed, cap_map[age])

    return smoothed


# ============================================================================
# MARCEL FIELDING PROJECTIONS
# ============================================================================

# Season weighting: most recent season gets weight 5, then 4, then 3, etc.
# These are relative weights for the weighted average.
SEASON_WEIGHTS = [5, 4, 3]

# Regression toward league average (0 for defense). This is how many
# "innings worth of league average" we add to the denominator.
# Lower = less regression (trust the data more).
# Higher = more regression (pull toward 0 more).
#
# Values derived from empirical YoY stabilization analysis:
#   stab_inn = avg_inn * (1 - r) / r, where r = year-to-year correlation
# Infield statcast metrics are much noisier (r≈0.42) than outfield (r≈0.56)
# or catcher (r≈0.51), so they need substantially more regression.
FIELDING_REGRESSION_INNINGS = {
    'outfield': {
        'sc_total_runs/150': 550,    # stab=535, r=0.555
        'sc_range_runs/150': 550,    # stab=532, r=0.557
        'sc_arm_runs/150': 2000,     # stab=2032, r=0.247 (very noisy)
    },
    'infield': {
        'sc_total_runs/150': 1100,   # stab=1114, r=0.416
        'sc_range_runs/150': 1150,   # stab=1152, r=0.408
        'sc_arm_runs/150': 2000,     # stab=inf, r=-0.16 (pure noise)
        'sc_dp_runs/150': 2000,      # stab=3672, r=0.178 (near noise)
    },
    'catcher': {
        'sc_total_runs/150': 600,    # stab=616, r=0.505
        'sc_framing_runs/150': 700,  # stab=699, r=0.473
        'sc_throwing_runs/150': 1050, # stab=1055, r=0.373
        'sc_blocking_runs/150': 550,  # stab=557, r=0.526
    },
}

# Reliability multipliers applied AFTER the Marcel weighted average.
# Even with proper regression innings, full-season players (~1300 inn) have
# so much data that the regression is negligible (<7%).  These multipliers
# provide the additional shrinkage needed.
#
# Calibrated via out-of-sample grid search: for each player-year, compute
# Marcel base through year N, multiply by α, and find α that minimises RMSE
# predicting actual year N+1 fielding (2016-2025 statcast data, N≥1200 pairs).
FIELDING_RELIABILITY = {
    'outfield': {
        'sc_total_runs/150': 0.85,
        'sc_range_runs/150': 0.85,
        'sc_arm_runs/150': 0.75,
    },
    'infield': {
        'sc_total_runs/150': 0.85,
        'sc_range_runs/150': 0.85,
        'sc_arm_runs/150': 0.25,
        'sc_dp_runs/150': 0.60,
    },
    'catcher': {
        'sc_total_runs/150': 0.85,
        'sc_framing_runs/150': 0.85,
        'sc_throwing_runs/150': 0.75,
        'sc_blocking_runs/150': 0.75,
    },
}

SMALL_SAMPLE_FULL_SEASON_INN = {
    'outfield': 1300.0,
    'infield': 1300.0,
    'catcher': 1000.0,
}


def _small_sample_reliability_scale(total_actual_inn: float, group_name: str) -> float:
    """With normalized [1.0, 0.8, 0.6] Fisher-information weights in
    `_compute_marcel_weighted_average`, single-season small samples are already
    shrunk strictly by `Inn / (Inn + reg_inn)` without 5x inflation."""
    return 1.0 if total_actual_inn > 0 else 0.0


BASERUNNING_REGRESSION_GAMES = {
    'sc_baserunning_runner_runs_tot_rate': 100,
    'SB_rate': 80,
    'CS_rate': 80,
}


# ============================================================================
# ERA-AWARE CONFIGURATIONS (pre-Statcast vs Statcast)
# ============================================================================

# Statcast fielding data begins 2016; Marcel needs ~3 seasons → cutoff >= 2018
STATCAST_MIN_CUTOFF = 2018

# --- Statcast-era fielding groups (current behavior) ---
FIELDING_GROUPS_STATCAST = {
    'outfield': {'positions': ['LF', 'CF', 'RF'],
                 'stats': ['sc_total_runs/150', 'sc_range_runs/150', 'sc_arm_runs/150']},
    'infield':  {'positions': ['1B', '2B', '3B', 'SS'],
                 'stats': ['sc_total_runs/150', 'sc_range_runs/150', 'sc_arm_runs/150', 'sc_dp_runs/150']},
    'catcher':  {'positions': ['C'],
                 'stats': ['sc_total_runs/150', 'sc_framing_runs/150', 'sc_throwing_runs/150', 'sc_blocking_runs/150']},
}

# --- Traditional (pre-Statcast) fielding groups: UZR/DRS components ---
FIELDING_GROUPS_TRADITIONAL = {
    'outfield': {'positions': ['LF', 'CF', 'RF'],
                 'stats': ['UZR/150', 'RngR/150', 'ARM/150']},
    'infield':  {'positions': ['1B', '2B', '3B', 'SS'],
                 'stats': ['UZR/150', 'RngR/150', 'ARM/150', 'DPR/150']},
    'catcher':  {'positions': ['C'],
                 'stats': ['DRS/150']},
}

FIELDING_REGRESSION_INNINGS_TRADITIONAL = {
    'UZR/150': 600,
    'RngR/150': 600,
    'ARM/150': 800,
    'DPR/150': 800,
    'DRS/150': 600,
}

# Map traditional stats → standardized output columns (WAR reads sc_total_runs/150)
_TRAD_FIELDING_TO_OUTPUT = {
    'UZR/150':  'sc_total_runs/150',
    'RngR/150': 'sc_range_runs/150',
    'ARM/150':  'sc_arm_runs/150',
    'DPR/150':  'sc_dp_runs/150',
    'DRS/150':  'sc_total_runs/150',
}

# Map traditional stats → aging curve keys (reuse statcast curves)
_TRAD_FIELDING_TO_CURVE = {
    'UZR/150':  'sc_total_runs/150',
    'RngR/150': 'sc_range_runs/150',
    'ARM/150':  'sc_arm_runs/150',
    'DPR/150':  'sc_dp_runs/150',
    'DRS/150':  'sc_total_runs/150',
}

# --- Baserunning era configs ---
BASERUNNING_STATS_STATCAST = ['sc_baserunning_runner_runs_tot_rate', 'SB_rate', 'CS_rate']
BASERUNNING_STATS_TRADITIONAL = ['BsR_rate', 'SB_rate', 'CS_rate']

BASERUNNING_REGRESSION_GAMES_TRADITIONAL = {
    'BsR_rate': 100,
    'SB_rate': 80,
    'CS_rate': 80,
}

# Map traditional baserunning → standardized output column
_TRAD_BR_TO_OUTPUT = {
    'BsR_rate': 'sc_baserunning_runner_runs_tot_rate',
}

# Map traditional baserunning → aging curve key
_TRAD_BR_TO_CURVE = {
    'BsR_rate': 'sc_baserunning_runner_runs_tot_rate',
}


def _compute_marcel_weighted_average(
    seasons_data: List[Dict],
    stats: List[str],
    volume_col: str,
    regression_amounts: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute the Marcel weighted average for a player.

    Takes up to 3 recent seasons, applies 5/4/3 weighting (by recency)
    multiplied by sample size (innings or games), then regresses toward 0.

    Args:
        seasons_data: List of dicts, newest first, each with stat values and volume
        stats: List of stat column names to project
        volume_col: Name of the volume column ('Inn' or 'G')
        regression_amounts: Dict of {stat: regression_volume} — how much league-average
                           volume to add for regression

    Returns:
        Dict of {stat: projected_value}
    """
    if not seasons_data:
        return {s: 0.0 for s in stats}

    result = {}
    for stat in stats:
        weighted_sum = 0.0
        total_weight = 0.0

        for i, season in enumerate(seasons_data[:len(SEASON_WEIGHTS)]):
            val = season.get(stat, 0.0)
            vol = season.get(volume_col, 0.0)
            if pd.isna(val) or pd.isna(vol) or vol <= 0:
                continue

            # Normalize [5, 4, 3] -> [1.0, 0.8, 0.6] so sample volume stays in true innings/games
            recency_weight = SEASON_WEIGHTS[i] / float(SEASON_WEIGHTS[0])
            weight = recency_weight * vol
            weighted_sum += val * weight
            total_weight += weight

        # Regression toward 0 (league average for fielding/baserunning)
        reg_vol = regression_amounts.get(stat, 600)
        # The regression volume acts as if we add reg_vol innings/games of 0.0
        total_weight += reg_vol

        if total_weight > 0:
            result[stat] = weighted_sum / total_weight
        else:
            result[stat] = 0.0

    return result


def marcel_fielding_projections(
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    position_group_map: Dict[str, str],
    input_features_map: Dict[str, List[str]],
    future_years: int = 15,
    cutoff_year: int = 2025,
    roster_ids: Optional[Set[int]] = None,
    position_profiles: Optional[Dict[int, Dict[str, float]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Generate Marcel-style fielding projections.

    Replaces the LSTM-based prediction with a simpler, more robust method:
    1. Weighted average of last 3 seasons (5/4/3, sample-size weighted)
    2. Regression toward 0 (league average for defense)
    3. Per-stat aging curve applied year-by-year

    Output format matches the LSTM pipeline exactly.

    Args:
        raw_df: Historical fielding data with per-150 stats computed
        player_names: DataFrame with IDfg and Name columns
        position_group_map: Maps position (e.g. 'RF') to group ('outfield')
        input_features_map: Maps group to list of features
        future_years: Number of years to project
        cutoff_year: Last year of actual data
        roster_ids: Optional set of IDfg values for active roster players
        position_profiles: Dict mapping IDfg -> {pos: fraction}

    Returns:
        DataFrame matching the fielding_predictions.csv format
    """
    MIN_POSITION_INNINGS = 50

    curves = _load_aging_curves()
    fielding_curves = curves.get('fielding', {})

    # Select era-appropriate stat configurations
    use_statcast = cutoff_year >= STATCAST_MIN_CUTOFF
    if use_statcast:
        groups = FIELDING_GROUPS_STATCAST
        regression_by_group = FIELDING_REGRESSION_INNINGS   # per-group dict
        output_col_map = None
        curve_key_map = None
    else:
        groups = FIELDING_GROUPS_TRADITIONAL
        regression_by_group = None  # flat dict for traditional era
        regression_flat = FIELDING_REGRESSION_INNINGS_TRADITIONAL
        output_col_map = _TRAD_FIELDING_TO_OUTPUT
        curve_key_map = _TRAD_FIELDING_TO_CURVE
        logger.info(f"Marcel fielding: using traditional stats (cutoff {cutoff_year} < {STATCAST_MIN_CUTOFF})")

    # All stat columns that appear in output (NaN for non-applicable ones)
    all_stat_cols = [
        'sc_total_runs/150', 'sc_range_runs/150', 'sc_arm_runs/150',
        'sc_dp_runs/150', 'sc_framing_runs/150', 'sc_throwing_runs/150',
        'sc_blocking_runs/150',
    ]

    all_predictions = []

    for group_name, group_info in groups.items():
        positions = group_info['positions']
        stats = group_info['stats']
        group_curves = fielding_curves.get(group_name, {})

        # Select regression amounts for this position group
        if regression_by_group is not None:
            regression = regression_by_group.get(group_name, {})
        else:
            regression = regression_flat

        group_df = raw_df[raw_df['Pos'].isin(positions)].copy()

        # Build player-position pairs (same logic as LSTM pipeline)
        player_position_pairs = []
        seen_pairs = set()

        if position_profiles:
            for player_id, profile in position_profiles.items():
                for pos, fraction in profile.items():
                    if pos in positions and (player_id, pos) not in seen_pairs:
                        player_position_pairs.append((player_id, pos))
                        seen_pairs.add((player_id, pos))

        # Also include cutoff-year qualifiers
        players_current = group_df[
            (group_df['Season'] == cutoff_year) &
            (group_df['Inn'] >= MIN_POSITION_INNINGS) &
            (group_df['Pos'].isin(positions))
        ][['IDfg', 'Pos', 'Inn']].copy()

        if not players_current.empty:
            for player_id in players_current['IDfg'].unique():
                if player_id not in (position_profiles or {}):
                    player_rows = players_current[players_current['IDfg'] == player_id]
                    primary_pos = player_rows.loc[player_rows['Inn'].idxmax(), 'Pos']
                    if (player_id, primary_pos) not in seen_pairs:
                        player_position_pairs.append((player_id, primary_pos))
                        seen_pairs.add((player_id, primary_pos))

        # Roster recovery
        if roster_ids is not None:
            current_ids = {pid for pid, _ in player_position_pairs}
            missing = roster_ids - current_ids
            if missing:
                recovery = group_df[
                    (group_df['IDfg'].isin(missing)) &
                    (group_df['Season'] <= cutoff_year) &
                    (group_df['Inn'] >= MIN_POSITION_INNINGS) &
                    (group_df['Pos'].isin(positions))
                ]
                if not recovery.empty:
                    for pid in recovery['IDfg'].unique():
                        pr = recovery[recovery['IDfg'] == pid].sort_values(['Season', 'Inn'])
                        primary = pr.iloc[-1]['Pos']
                        if (pid, primary) not in seen_pairs:
                            player_position_pairs.append((pid, primary))
                            seen_pairs.add((pid, primary))

        logger.info(f"Marcel {group_name}: {len(player_position_pairs)} player-position pairs")

        for player_id, position in player_position_pairs:
            # Get historical data at this position (within the group)
            player_hist = group_df[
                (group_df['IDfg'] == player_id) &
                (group_df['Season'] <= cutoff_year) &
                (group_df['Pos'] == position)
            ].copy()
            player_hist = player_hist.dropna(subset=stats)

            if player_hist.empty:
                continue

            # FIXED: resolve name with raw_df fallback to avoid Unknown for debutants
            player_name = _safe_resolve_player_name(player_id, player_names, raw_df if 'raw_df' in locals() else (group_df if 'group_df' in locals() else None))

            # Get last known age (default 27.0 if missing)
            player_hist = player_hist.sort_values('Season')
            _valid_ages = pd.to_numeric(player_hist['Age'], errors='coerce').dropna()
            last_age = float(_valid_ages.iloc[-1]) if not _valid_ages.empty else 27.0
            last_season = player_hist['Season'].iloc[-1]

            # Build recent seasons list (newest first)
            recent = player_hist.tail(len(SEASON_WEIGHTS)).iloc[::-1]
            seasons_data = []
            for _, row in recent.iterrows():
                s = {'Inn': row['Inn']}
                for stat in stats:
                    s[stat] = row[stat]
                seasons_data.append(s)

            # Compute Marcel base projection
            base = _compute_marcel_weighted_average(
                seasons_data, stats, 'Inn', regression
            )

            # Apply reliability shrinkage to the base.
            # Even with proper regression innings, full-season players have
            # enough data to overwhelm the regression.  The reliability
            # multiplier provides the empirically-calibrated additional
            # shrinkage needed to minimise out-of-sample prediction error.
            #
            # On top of that flat multiplier, apply extra shrinkage scaled
            # to the player's ACTUAL total innings — this is what corrects
            # for small, partial-season samples (e.g. a rookie's first
            # month) that the recency-weighted Marcel average alone doesn't
            # regress hard enough. See _small_sample_reliability_scale.
            reliability = FIELDING_RELIABILITY.get(group_name, {})
            total_actual_inn = sum(
                s.get('Inn', 0.0) for s in seasons_data if pd.notna(s.get('Inn'))
            )
            small_sample_scale = _small_sample_reliability_scale(total_actual_inn, group_name)
            for stat in stats:
                r_key = curve_key_map[stat] if curve_key_map and stat in curve_key_map else stat
                alpha = reliability.get(r_key, 1.0)
                base[stat] *= alpha * small_sample_scale

            # Project forward with aging
            for year_offset in range(1, future_years + 1):
                proj_year = cutoff_year + year_offset
                proj_age = last_age + (cutoff_year - last_season) + year_offset

                # Apply cumulative aging from year 1 through year_offset
                prediction = {}
                for stat in stats:
                    aging_total = 0.0
                    # Map traditional stat to the corresponding statcast aging curve key
                    curve_key = curve_key_map[stat] if curve_key_map and stat in curve_key_map else stat
                    stat_curves = group_curves.get(curve_key, {})
                    for y in range(1, year_offset + 1):
                        age_at_y = last_age + (cutoff_year - last_season) + y
                        aging_total += _get_defensive_aging_delta(
                            stat_curves, int(age_at_y), stat
                        )
                    prediction[stat] = base[stat] + aging_total

                # Remap traditional stats to standardised output columns
                if output_col_map:
                    remapped = {}
                    for stat_key, val in prediction.items():
                        out_col = output_col_map.get(stat_key, stat_key)
                        remapped[out_col] = val
                    prediction = remapped

                # Build output row
                row = {
                    'Name': player_name,
                    'Age': proj_age,
                    'Year': proj_year,
                    'IDfg': player_id,
                    'Pos': position,
                    'Position_Group': group_name,
                }
                for col in all_stat_cols:
                    row[col] = prediction.get(col, np.nan)

                all_predictions.append(row)

    if not all_predictions:
        logger.warning("No Marcel fielding predictions generated")
        return None

    result_df = pd.DataFrame(all_predictions)
    result_df = result_df.sort_values(['Name', 'Pos', 'Year'])
    return result_df


# ============================================================================
# MARCEL BASERUNNING PROJECTIONS
# ============================================================================

def marcel_baserunning_projections(
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    input_features: List[str],
    future_years: int = 15,
    cutoff_year: int = 2025,
    roster_ids: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    """
    Generate Marcel-style baserunning projections.

    Args:
        raw_df: Historical batting data with rate stats computed
        player_names: DataFrame with IDfg and Name columns
        input_features: List of baserunning features (e.g. ['Age', 'sc_baserunning_...', 'SB_rate', 'CS_rate'])
        future_years: Number of years to project
        cutoff_year: Last year of actual data
        roster_ids: Optional set of IDfg values for active roster players

    Returns:
        DataFrame matching the baserunning_predictions.csv format
    """
    curves = _load_aging_curves()
    br_curves = curves.get('baserunning', {})

    # Select era-appropriate stats and regression
    use_statcast = cutoff_year >= STATCAST_MIN_CUTOFF
    if use_statcast:
        stats = [f for f in BASERUNNING_STATS_STATCAST if f != 'Age']
        regression = BASERUNNING_REGRESSION_GAMES
        output_col_map = None
        curve_key_map = None
    else:
        stats = [f for f in BASERUNNING_STATS_TRADITIONAL if f != 'Age']
        regression = BASERUNNING_REGRESSION_GAMES_TRADITIONAL
        output_col_map = _TRAD_BR_TO_OUTPUT
        curve_key_map = _TRAD_BR_TO_CURVE
        logger.info(f"Marcel baserunning: using traditional stats (cutoff {cutoff_year} < {STATCAST_MIN_CUTOFF})")

    # Non-negative stats
    non_negative = {'SB_rate', 'CS_rate'}

    # Get current players
    current_players = set(raw_df[raw_df['Season'] == cutoff_year]['IDfg'].unique())

    # Roster recovery
    if roster_ids is not None:
        missing = roster_ids - current_players
        if missing:
            historical = set(raw_df[
                (raw_df['IDfg'].isin(missing)) &
                (raw_df['Season'] <= cutoff_year)
            ]['IDfg'].unique())
            current_players = current_players | historical
            if historical:
                logger.info(f"Marcel baserunning: recovered {len(historical)} roster players from history")

    logger.info(f"Marcel baserunning: {len(current_players)} players")

    all_predictions = []

    for player_id in current_players:
        player_hist = raw_df[
            (raw_df['IDfg'] == player_id) &
            (raw_df['Season'] <= cutoff_year)
        ].copy()
        player_hist = player_hist.dropna(subset=stats)

        if player_hist.empty:
            continue

        # FIXED: resolve name with fallback
        _raw_for_name = raw_df if 'raw_df' in locals() else None
        player_name = _safe_resolve_player_name(player_id, player_names, _raw_for_name)

        player_hist = player_hist.sort_values('Season')
        _valid_br_ages = pd.to_numeric(player_hist['Age'], errors='coerce').dropna()
        last_age = float(_valid_br_ages.iloc[-1]) if not _valid_br_ages.empty else 27.0
        last_season = player_hist['Season'].iloc[-1]

        # Build recent seasons (newest first)
        recent = player_hist.tail(len(SEASON_WEIGHTS)).iloc[::-1]
        seasons_data = []
        for _, row in recent.iterrows():
            s = {'G': row.get('G', 150)}
            for stat in stats:
                s[stat] = row[stat]
            seasons_data.append(s)

        # Compute Marcel base
        base = _compute_marcel_weighted_average(
            seasons_data, stats, 'G', regression
        )

        # Project forward with aging
        for year_offset in range(1, future_years + 1):
            proj_year = cutoff_year + year_offset
            proj_age = last_age + (cutoff_year - last_season) + year_offset

            prediction = {'Name': player_name, 'IDfg': player_id,
                          'Year': proj_year, 'Age': proj_age}

            for stat in stats:
                curve_key = curve_key_map[stat] if curve_key_map and stat in curve_key_map else stat
                stat_curves = br_curves.get(curve_key, {})
                aging_total = 0.0
                for y in range(1, year_offset + 1):
                    age_at_y = last_age + (cutoff_year - last_season) + y
                    aging_total += _get_smoothed_aging_delta(stat_curves, int(age_at_y))

                val = base[stat] + aging_total
                if stat in non_negative:
                    val = max(0.0, val)

                # Map to standardised output column name
                out_col = output_col_map.get(stat, stat) if output_col_map else stat
                prediction[out_col] = val

            all_predictions.append(prediction)

    if not all_predictions:
        logger.warning("No Marcel baserunning predictions generated")
        return None

    result_df = pd.DataFrame(all_predictions)
    result_df = result_df.sort_values(['Year', 'SB_rate' if 'SB_rate' in result_df.columns else 'Name'],
                                       ascending=[True, False])
    return result_df


# ============================================================================
# MARCEL BATTER PROJECTIONS  — Component-Based Architecture
# ============================================================================
#
# Instead of projecting rate stats (AVG, OBP, SLG, wOBA) directly, we project
# 8 base components and derive everything else via composition formulas.
#
# Base components: K%, BB%, HBP%, ISO, BABIP, HR/FB, GB%, LD%
#
# Flow:
#   1. Marcel weighted average of base components (3-yr 5/4/3 × PA)
#   2. Multivariate adjustment via Phase 2b regression equations
#      (replaces the old single-stat regression-to-mean step)
#   3. Aging curves applied year-by-year for Years 2–15
#   4. Composition: AVG, SLG, OBP, wOBA, wRC+, HR, 2B, 3B from components
# ============================================================================

from core.stat_composition import compose_all

# The 8 base components that Marcel directly projects for batters.
BATTER_BASE_COMPONENTS = ['K%', 'BB%', 'HBP%', 'ISO', 'BABIP', 'HR/FB', 'GB%', 'LD%']

# Output counting stats (derived from composition, not from career profile)
BATTER_COUNTING_STATS = ['HR', '2B', '3B', 'RBI', 'R', 'HBP']

# Legacy alias so downstream code that imports BATTER_MARCEL_RATE_STATS still works
BATTER_MARCEL_RATE_STATS = BATTER_BASE_COMPONENTS

# Regression toward prior for base components (in true Fisher-information PA units,
# paired with recency decay weights [1.0, 0.8, 0.6] preserving 5:4:3 season ratios).
BATTER_REGRESSION_PA = {
    'K%':    65,     # ~60 PA to stabilize (very sticky)
    'BB%':   140,    # ~120-150 PA to stabilize
    'HBP%':  300,    # very noisy, heavy regression
    'ISO':   180,    # ~160-200 PA to stabilize
    'BABIP': 420,    # ~800+ BIP to stabilize (very noisy)
    'HR/FB': 320,    # noisy, moderate-heavy regression
    'GB%':   70,     # ~60 PA to stabilize (very sticky)
    'LD%':   260,    # moderate noise
}

# League-average priors for base components (approximately 2020-2024 average)
BATTER_LEAGUE_AVG = {
    'K%':    0.224,
    'BB%':   0.083,
    'HBP%':  0.012,
    'ISO':   0.154,
    'BABIP': 0.292,
    'HR/FB': 0.127,
    'GB%':   0.430,
    'LD%':   0.213,
}

# Physical bounds for base components
BATTER_BOUNDS = {
    'K%':    (0.05,  0.40),
    'BB%':   (0.02,  0.25),
    'HBP%':  (0.001, 0.04),
    'ISO':   (0.020, 0.400),
    'BABIP': (0.200, 0.380),
    'HR/FB': (0.030, 0.350),
    'GB%':   (0.20,  0.65),
    'LD%':   (0.12,  0.28),
}

# PA threshold for full multivariate blend weight.  Below this, the R²-based
# multivariate weight is scaled down linearly so that small-sample seasons
# (e.g. 70-PA rookie debuts) don't override Marcel's properly-regressed
# estimate.  At 400 PA the multivariate equation gets its full R² weight;
# at 70 PA only 17.5% of R² is applied.
MV_BLEND_STABILIZATION_PA = 400

# ---------------------------------------------------------------------------
# Multivariate regression equations (Phase 2b exhaustive brute-force search)
#
# Each entry maps a base-component target to a dict of
#   {feature_name: coefficient, ..., '_intercept': value, '_r2': cv_r2}.
#
# The equation for Year N+1's target is:
#   target_{N+1} = intercept + Σ(coef_i × feature_i_N)
#
# Features are the player's most recent season's values (after x-stat sub).
# Features that are unavailable or NaN for a player fall back to league avg.
# ---------------------------------------------------------------------------
BATTER_MULTIVARIATE_EQUATIONS = {
    'K%': {
        'K%': 0.502673, 'Contact%': -0.272776, 'BB%': 0.104932,
        '_intercept': 0.309488, '_r2': 0.659,
    },
    'BB%': {
        'BB%': 0.229113, 'O-Swing%': -0.197499, 'Barrel%': 0.087445,
        'ISO': 0.063516,
        '_intercept': 0.110775, '_r2': 0.516,
    },
    'HBP%': {
        'HBP%': 0.085738, 'OBP': 0.028392, 'Pull%': 0.011814,
        'EV': -0.000460, 'K%': 0.010916,
        '_intercept': 0.034203, '_r2': 0.102,
    },
    'ISO': {
        'ISO': 0.166567, 'Hard%': 0.130315, 'sc_ev50': 0.009700,
        'HardHit%': -0.122753, 'GB%': -0.121401, 'Pull%': 0.083149,
        '_intercept': -0.801955, '_r2': 0.393,
    },
    'BABIP': {
        'BABIP': 0.110756, 'FB%': -0.113637, 'Oppo%': 0.108819,
        'Spd': 0.002997, 'F-Strike%': 0.086343, 'xSLG': 0.088158,
        '_intercept': 0.175094, '_r2': 0.212,
    },
    'HR/FB': {
        'HR/FB': 0.156716, 'sc_ev50': 0.011446, 'Hard%': 0.166396,
        'HardHit%': -0.156014, 'Contact%': -0.146888, 'BB%': 0.117759,
        '_intercept': -0.920288, '_r2': 0.452,
    },
    'GB%': {
        'GB%': 0.409550, 'FB%': -0.241323, 'Pull%': -0.125978,
        '_intercept': 0.387990, '_r2': 0.538,
    },
    'LD%': {
        'LD%': 0.170445, 'IFFB%': -0.119454, 'sc_ev50': -0.002912,
        'Oppo%': 0.060649, 'ISO': 0.052306,
        '_intercept': 0.446693, '_r2': 0.182,
    },
}

# League-average fallbacks for auxiliary features used in the equations.
# These are used when a player's historical data lacks a feature (e.g. pre-Statcast).
_AUX_FEATURE_LEAGUE_AVG = {
    'Contact%': 0.776,
    'O-Swing%': 0.300,
    'Barrel%':  0.068,
    'Hard%':    0.352,
    'HardHit%': 0.352,
    'sc_ev50':  89.0,
    'Pull%':    0.400,
    'Oppo%':    0.220,
    'Spd':      5.0,
    'F-Strike%': 0.600,
    'xSLG':     0.402,
    'FB%':      0.350,
    'IFFB%':    0.100,
    'EV':       88.5,
    'OBP':      0.314,
}

# Default triple share: 3B / (2B + 3B), league average ~7.8%
DEFAULT_TRIPLE_SHARE = 0.078


def _compute_marcel_weighted_average_toward_league(
    seasons_data: List[Dict],
    stats: List[str],
    volume_col: str,
    regression_amounts: Dict[str, float],
    league_avg: Dict[str, float],
) -> Dict[str, float]:
    """
    Conjugate Bayesian weighted average toward `league_avg` (player prior P_0).

    Uses normalized recency decay weights `[1.0, 0.8, 0.6]` (exact `5:4:3` ratio)
    so effective sample volume stays in true `PA`/`IP` Fisher-information units
    rather than inflating single-season call-ups by `5x` against `regression_amounts`.
    """
    if not seasons_data:
        return {s: league_avg.get(s, 0.0) for s in stats}

    top_w = float(SEASON_WEIGHTS[0])
    result = {}
    for stat in stats:
        weighted_sum = 0.0
        total_weight = 0.0

        for i, season in enumerate(seasons_data[:len(SEASON_WEIGHTS)]):
            val = season.get(stat, np.nan)
            vol = season.get(volume_col, 0.0)
            if pd.isna(val) or pd.isna(vol) or vol <= 0:
                continue

            decay_weight = float(SEASON_WEIGHTS[i]) / top_w  # 1.0, 0.8, 0.6
            weight = decay_weight * float(vol)
            weighted_sum += float(val) * weight
            total_weight += weight

        reg_vol = float(regression_amounts.get(stat, 250))
        prior_val = float(league_avg.get(stat, 0.0))
        weighted_sum += prior_val * reg_vol
        total_weight += reg_vol

        if total_weight > 0:
            result[stat] = weighted_sum / total_weight
        else:
            result[stat] = prior_val

    return result


def _apply_multivariate_equations(
    recent_features: Dict[str, float],
    marcel_base: Dict[str, float],
    recent_pa: float = 650.0,
) -> Dict[str, float]:
    """
    Apply Phase 2b multivariate regression equations to produce Year 1 projection.

    For each base component, two estimates are blended:
      - marcel_base: 3-year weighted average with regression to league mean
      - multivariate: intercept + Σ(coef × feature) from the player's most
        recent season

    The blend is weighted by each equation's R², scaled by the sample size
    of the most recent season (recent_pa).  This prevents small-sample
    seasons from overriding Marcel's properly-regressed estimate.

    Blend formula:
        pa_reliability = min(1, recent_pa / MV_BLEND_STABILIZATION_PA)
        effective_r2   = r2 × pa_reliability
        projected      = effective_r2 × multivariate + (1 − effective_r2) × marcel_base

    At full PA (≥400), behavior is identical to the original R²-only blend.
    At 70 PA, a K% equation with R²=0.66 gets only 0.66×0.175 = 0.115
    effective weight instead of the full 0.66.
    """
    # Scale R² blend weight by sample size of the most recent season
    pa_reliability = min(1.0, max(0.0, recent_pa) / MV_BLEND_STABILIZATION_PA)

    result = {}
    for component, equation in BATTER_MULTIVARIATE_EQUATIONS.items():
        intercept = equation['_intercept']
        r2 = equation['_r2']
        effective_r2 = r2 * pa_reliability

        # Compute multivariate prediction from most recent season features
        mv_pred = intercept
        for feat, coef in equation.items():
            if feat.startswith('_'):
                continue
            # Get feature value: check recent season, then Marcel base, then league avg
            val = recent_features.get(feat, np.nan)
            if pd.isna(val):
                val = marcel_base.get(feat, _AUX_FEATURE_LEAGUE_AVG.get(feat, 0.0))
            mv_pred += coef * val

        # Blend: effective-r2-weighted mix of multivariate and Marcel base
        mb = marcel_base.get(component, BATTER_LEAGUE_AVG.get(component, 0.0))
        result[component] = effective_r2 * mv_pred + (1.0 - effective_r2) * mb

        # Clip to physical bounds
        if component in BATTER_BOUNDS:
            lo, hi = BATTER_BOUNDS[component]
            result[component] = float(np.clip(result[component], lo, hi))

    return result


def _build_batter_career_profile(
    player_hist: pd.DataFrame,
    n_recent: int = 3,
) -> Optional[Dict]:
    """
    Build a PA-weighted career counting profile for a single batter.

    Returns triple_share and RBI/R rates for composition.
    """
    recent = player_hist.tail(n_recent)
    if len(recent) == 0 or 'PA' not in recent.columns:
        return None

    pa = recent['PA'].values.astype(float)
    total_pa = pa.sum()
    if total_pa == 0:
        return None
    weights = pa / total_pa

    # Triple share: 3B / (2B + 3B)
    triple_share = DEFAULT_TRIPLE_SHARE
    if '2B' in recent.columns and '3B' in recent.columns:
        d = np.nan_to_num(recent['2B'].values.astype(float), nan=0.0)
        t = np.nan_to_num(recent['3B'].values.astype(float), nan=0.0)
        total_2b = np.average(d, weights=weights)
        total_3b = np.average(t, weights=weights)
        if (total_2b + total_3b) > 1.0:
            triple_share = total_3b / (total_2b + total_3b)

    # RBI and R rates (per 150 games) for output — these can't be composed
    # from the 8 components, so we keep career profile ratios.
    rbi_rate = 0.0
    r_rate = 0.0
    if 'RBI' in recent.columns:
        rbi_rate = float(np.average(
            np.nan_to_num(recent['RBI'].values.astype(float), nan=0.0), weights=weights
        ))
    if 'R' in recent.columns:
        r_rate = float(np.average(
            np.nan_to_num(recent['R'].values.astype(float), nan=0.0), weights=weights
        ))

    # Base wOBA for RBI/R scaling
    base_woba = 0.312
    if 'wOBA' in recent.columns:
        woba_vals = np.nan_to_num(recent['wOBA'].values.astype(float), nan=0.0)
        base_woba = max(0.15, float(np.average(woba_vals, weights=weights)))

    career_pa = float(player_hist['PA'].sum())

    return {
        'triple_share': triple_share,
        'rbi_rate': rbi_rate,
        'r_rate': r_rate,
        'base_woba': base_woba,
        'career_pa': career_pa,
    }


# ---------------------------------------------------------------------------
# xStat-informed shrinkage for BABIP / ISO
#
# Marcel's BABIP and ISO base components are pure outcome stats, regressed
# only toward the *league* mean (BATTER_REGRESSION_PA). That regression
# can't distinguish a hot BABIP/ISO season backed by real quality of
# contact from one that just found grass on weak contact — both get
# regressed by the same amount for the same PA.
#
# Before a season's BABIP/ISO enters the Marcel weighted average, shrink
# the *actual* value toward an xStat-implied value, with the shrink
# weight set by that season's PA (small samples lean harder on the
# expected value; an established full season barely moves). This runs
# upstream of BATTER_REGRESSION_PA, not instead of it — the league-mean
# regression step still applies afterward to whatever comes out of this.
#
#   expected_iso:   xSLG - xBA (the direct expected-stat analog of
#                   ISO = SLG - AVG).
#   expected_babip: rebuild the BABIP formula using xBA-implied hits in
#                   place of actual hits, handling HR/strikeouts the same
#                   way the real BABIP formula does:
#                     (xBA*AB - HR) / (AB - SO - HR + SF)
#
# NOTE: this function assumes xBA/xSLG arriving here have already been park-
# neutralized (see predict_models.py's park_neutral_features), same as the
# actual BABIP/ISO they're blended with. If that upstream neutralization
# ever stops covering xBA/xSLG/xwOBA, this shrinkage will blend a
# park-adjusted actual value with a raw expected value and quietly bias
# extreme-park (Coors/T-Mobile-type) players.
# ---------------------------------------------------------------------------
XSTAT_SHRINKAGE_PA = 200  # PA at which actual/expected are weighted evenly


def _xstat_expected_babip_iso(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Derive xStat-implied BABIP and ISO for a single player-season row.

    Returns (expected_babip, expected_iso). Either may be None if the
    columns needed to derive it aren't available for that season (e.g.
    pre-Statcast years) — the caller falls back to the actual value in
    that case.
    """
    expected_babip = None
    expected_iso = None

    x_ba = row.get('xBA', np.nan)
    x_slg = row.get('xSLG', np.nan)

    if pd.notna(x_slg) and pd.notna(x_ba):
        expected_iso = float(x_slg - x_ba)

    ab = row.get('AB', np.nan)
    so = row.get('SO', np.nan)
    hr = row.get('HR', np.nan)
    sf = row.get('SF', np.nan)
    if pd.notna(x_ba) and pd.notna(ab) and pd.notna(so) and pd.notna(hr) and ab > 0:
        sf_val = sf if pd.notna(sf) else 0.0
        x_hits = x_ba * ab
        denom = ab - so - hr + sf_val
        if denom > 0:
            expected_babip = float((x_hits - hr) / denom)

    return expected_babip, expected_iso


def _apply_xstat_shrinkage_to_babip_iso(hist_for_marcel: pd.DataFrame) -> pd.DataFrame:
    """Shrink each season's actual BABIP/ISO toward its xStat-implied value.

    Weight is PA-based (XSTAT_SHRINKAGE_PA). Falls back to the actual
    value untouched when xBA/xSLG (or the counting stats needed to derive
    expected BABIP) aren't available for that season.
    """
    out = hist_for_marcel.copy()
    for idx, row in out.iterrows():
        pa = row.get('PA', np.nan)
        if pd.isna(pa) or pa <= 0:
            continue
        w = pa / (pa + XSTAT_SHRINKAGE_PA)

        expected_babip, expected_iso = _xstat_expected_babip_iso(row)

        actual_babip = row.get('BABIP', np.nan)
        if expected_babip is not None and pd.notna(actual_babip):
            out.at[idx, 'BABIP'] = w * actual_babip + (1 - w) * expected_babip

        actual_iso = row.get('ISO', np.nan)
        if expected_iso is not None and pd.notna(actual_iso):
            out.at[idx, 'ISO'] = w * actual_iso + (1 - w) * expected_iso

    return out


def marcel_batter_projections(
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    future_years: int = 15,
    cutoff_year: int = 2025,
    roster_ids: Optional[Set[int]] = None,
    use_xstats: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Generate component-based Marcel batter projections.

    Projects 8 base components (K%, BB%, HBP%, ISO, BABIP, HR/FB, GB%, LD%)
    using the Marcel method with multivariate regression equations, then
    derives all display stats via stat_composition.compose_all().

    Flow:
      1. Marcel weighted average of base components (3-yr 5/4/3 × PA)
      2. Year 1: Multivariate adjustment (Phase 2b equations) replaces
         the old single-stat regression-to-mean step
      3. Years 2–15: Apply empirical aging curves to base components
      4. Composition: derive AVG, SLG, OBP, wOBA, wRC+, HR, 2B, 3B

    When use_xstats=True, xwOBA/xBA/xSLG are used in feature extraction for
    the multivariate equations (more predictive of future performance).

    Output format matches batter_predictions.csv.
    """
    from configs.batter_config import BatterConfig

    curves = _load_aging_curves()
    batting_curves = curves.get('batting', {})
    min_pa_current = getattr(BatterConfig, 'MIN_PA_CURRENT', 70)

    # Get current-year qualifying batters
    current_players = raw_df[
        (raw_df['Season'] == cutoff_year) &
        (raw_df['PA'] >= min_pa_current)
    ]['IDfg'].unique()
    current_ids = set(current_players)

    # Roster recovery: add MLB-debuted roster players from history (even tiny stints >= 1 PA)
    if roster_ids is not None:
        missing = roster_ids - current_ids
        if missing:
            historical = set(raw_df[
                (raw_df['IDfg'].isin(missing)) &
                (raw_df['Season'] <= cutoff_year) &
                (raw_df['PA'] >= 1)
            ]['IDfg'].unique())
            current_ids = current_ids | historical
            if historical:
                logger.info(f"Marcel batting: recovered {len(historical)} roster players from history")

    logger.info(f"Marcel batting: {len(current_ids)} MLB-debuted players")

    # All features needed by the multivariate equations
    all_eq_features = set()
    for eq in BATTER_MULTIVARIATE_EQUATIONS.values():
        for feat in eq:
            if not feat.startswith('_'):
                all_eq_features.add(feat)

    all_predictions = []
    components = list(BATTER_BASE_COMPONENTS)

    # ---- Pedigree-Conditioned Prior P_0(theta | FV, CareerPA) ----
    # Low-sample players shrink directly toward their unconditional AAA/waiver
    # replacement prior (~81 wRC+) unless backed by recent prospect FV pedigree,
    # transitioning smoothly toward BATTER_LEAGUE_AVG (~100 wRC+) as career PA
    # reaches full MLB regular volume (600 PA). Computing P_0 BEFORE the Marcel
    # weighted average eliminates survivorship-biased double regression.
    ROOKIE_STABILIZATION_PA = 600.0
    WAIVER_BATTER_PRIOR = {
        'K%': 0.272, 'BB%': 0.068, 'HBP%': 0.011, 'ISO': 0.114,
        'BABIP': 0.281, 'HR/FB': 0.095, 'GB%': 0.448, 'LD%': 0.202,
    }

    def _batter_fv_component_prior(fv: Optional[float]) -> Dict[str, float]:
        if fv is None or fv < 40.0:
            return WAIVER_BATTER_PRIOR
        # Smooth interpolation from FV 40 (~87 wRC+) to FV 70 (~128 wRC+)
        t = float(np.clip((fv - 40.0) / 30.0, 0.0, 1.0))
        return {
            'K%':    0.260 - 0.068 * t,   # 26.0% -> 19.2%
            'BB%':   0.072 + 0.034 * t,   #  7.2% -> 10.6%
            'HBP%':  0.012,
            'ISO':   0.124 + 0.086 * t,   #  .124 -> .210
            'BABIP': 0.284 + 0.024 * t,   #  .284 -> .308
            'HR/FB': 0.102 + 0.058 * t,   # 10.2% -> 16.0%
            'GB%':   0.442 - 0.032 * t,
            'LD%':   0.204 + 0.014 * t,
        }

    try:
        from core.id_crosswalk import load_recent_prospect_info
        from core.name_utils import name_key as _nk_fn
        _raw_for_name_map = raw_df if 'raw_df' in locals() else None
        name_to_fg = {
            _nk_fn(_safe_resolve_player_name(int(pid), player_names, _raw_for_name_map)): int(pid)
            for pid in current_ids
        }
        prospect_lookup = load_recent_prospect_info(cutoff_year, recency_years=3, name_to_fg_fallback=name_to_fg)
    except Exception as exc:
        logger.warning(f"Could not load prospect info for batter priors: {exc}")
        prospect_lookup = {}

    profiles_to_project = []
    rookie_adjusted = 0

    for player_id in current_ids:
        all_player_mlb = raw_df[
            (raw_df['IDfg'] == player_id) &
            (raw_df['Season'] <= cutoff_year) &
            (raw_df['PA'] >= 1)
        ]
        if all_player_mlb.empty:
            continue

        total_career_pa = float(all_player_mlb['PA'].fillna(0).sum())

        player_hist = all_player_mlb[all_player_mlb['PA'] >= 50].copy()
        if player_hist.empty:
            player_hist = all_player_mlb.copy()

        _raw_for_name = raw_df if 'raw_df' in locals() else None
        player_name = _safe_resolve_player_name(player_id, player_names, _raw_for_name)

        player_hist = player_hist.sort_values('Season')
        _valid_br_ages = pd.to_numeric(player_hist['Age'], errors='coerce').dropna()
        last_age = float(_valid_br_ages.iloc[-1]) if not _valid_br_ages.empty else 27.0
        last_season = player_hist['Season'].iloc[-1]

        # Build player-specific Bayesian prior P_0(theta | FV, total_career_pa)
        if total_career_pa < ROOKIE_STABILIZATION_PA:
            surv_w = float(np.clip(total_career_pa / ROOKIE_STABILIZATION_PA, 0.0, 1.0))
            p_info = prospect_lookup.get(int(player_id))
            # Unranked players age >= 26 without prospect pedigree stay anchored to waiver prior
            fv_val = p_info['fv'] if p_info else None
            base_prior = _batter_fv_component_prior(fv_val)
            player_prior = {
                comp: surv_w * BATTER_LEAGUE_AVG[comp] + (1.0 - surv_w) * base_prior[comp]
                for comp in components
            }
            rookie_adjusted += 1
        else:
            player_prior = BATTER_LEAGUE_AVG

        # Derive HBP% if not present
        if 'HBP%' not in player_hist.columns:
            if 'HBP' in player_hist.columns and 'PA' in player_hist.columns:
                player_hist['HBP%'] = player_hist['HBP'] / player_hist['PA'].replace(0, np.nan)
            else:
                player_hist['HBP%'] = player_prior['HBP%']

        # x-stat substitution for the multivariate equation features
        hist_for_marcel = player_hist.copy()
        if use_xstats:
            for real_col, x_col in [('wOBA', 'xwOBA'), ('AVG', 'xBA'), ('SLG', 'xSLG')]:
                if x_col in hist_for_marcel.columns:
                    mask = hist_for_marcel[x_col].notna()
                    hist_for_marcel.loc[mask, real_col] = hist_for_marcel.loc[mask, x_col]

            hist_for_marcel = _apply_xstat_shrinkage_to_babip_iso(hist_for_marcel)

        # ---- Extract most recent season features for multivariate equations ----
        most_recent = hist_for_marcel.iloc[-1]
        recent_features = {}
        for feat in all_eq_features:
            val = most_recent.get(feat, np.nan)
            if pd.notna(val):
                recent_features[feat] = float(val)

        # ---- Build recent seasons list for Marcel conjugate posterior ----
        recent_seasons = hist_for_marcel.tail(len(SEASON_WEIGHTS)).iloc[::-1]
        seasons_data = []
        for _, row in recent_seasons.iterrows():
            s = {'PA': row.get('PA', 650)}
            for stat in components:
                s[stat] = row.get(stat, np.nan)
            seasons_data.append(s)

        # Single conjugate posterior shrinking directly toward player_prior P_0
        marcel_base = _compute_marcel_weighted_average_toward_league(
            seasons_data, components, 'PA', BATTER_REGRESSION_PA, player_prior
        )

        # Apply sample-reliability-scaled multivariate equations → Year 1 base
        recent_pa = float(most_recent.get('PA', 650))
        year1_base = _apply_multivariate_equations(recent_features, marcel_base, recent_pa=recent_pa)

        # Build career profile (for triple_share and RBI/R rates)
        career_profile = _build_batter_career_profile(hist_for_marcel)

        triple_share = DEFAULT_TRIPLE_SHARE
        rbi_rate = 75.0
        r_rate = 75.0
        base_woba = BATTER_LEAGUE_AVG.get('ISO', 0.154) + 0.248
        if career_profile is not None:
            triple_share = career_profile['triple_share']
            rbi_rate = career_profile['rbi_rate']
            r_rate = career_profile['r_rate']
            base_woba = career_profile['base_woba']

        profiles_to_project.append({
            'player_id': player_id,
            'player_name': player_name,
            'last_age': last_age,
            'last_season': last_season,
            'year1_base': year1_base,
            'triple_share': triple_share,
            'rbi_rate': rbi_rate,
            'r_rate': r_rate,
            'base_woba': base_woba,
            'career_pa': total_career_pa,
        })

    logger.info(f"Applied conjugate FV/waiver priors to {rookie_adjusted} batters with <{int(ROOKIE_STABILIZATION_PA)} career PA")

    # ---- Project forward for all MLB-debuted profiles ----
    for profile in profiles_to_project:
        player_id = profile['player_id']
        player_name = profile['player_name']
        last_age = profile['last_age']
        last_season = profile['last_season']
        year1_base = profile['year1_base']
        triple_share = profile['triple_share']
        rbi_rate = profile['rbi_rate']
        r_rate = profile['r_rate']
        base_woba = profile['base_woba']

        # year_offset=0 is the CURRENT season (cutoff_year itself). It is
        # emitted directly from year1_base with zero aging applied, since
        # year1_base already reflects the player's in-season-to-date
        # performance (via the PA-weighted Marcel average + MiLB-prior
        # blend above) — this IS the current-season projection. Previously
        # this row didn't exist here at all; a separate module (ros.py)
        # patched a *stale* preseason projection with actual current-season
        # stats using a different, inconsistent weighting formula, which is
        # what caused the current-year number to diverge from year_offset=1
        # (next year) even though both nominally reflect the same season's
        # data. Emitting year_offset=0 here makes this function the single
        # source of truth for every year, including the current one, so
        # cutoff_year and cutoff_year+1 can only differ by one year of
        # aging — never by a blending-formula mismatch.
        #
        # NOTE: this requires cutoff_year to be set to the CURRENT
        # (in-progress) season, with raw_df already containing that
        # season's stats-to-date as the most recent row per player — not
        # the last *completed* season. See predict_models.py.
        for year_offset in range(0, future_years + 1):
            proj_year = cutoff_year + year_offset
            proj_age = last_age + (cutoff_year - last_season) + year_offset

            # Start from multivariate Year 1 base, apply cumulative aging
            # (aging_total is 0.0 at year_offset=0 by construction, since
            # range(1, 1) is empty — this year IS year1_base, unmodified)
            projected = {}
            for stat in components:
                stat_curves = batting_curves.get(stat, {})
                aging_total = 0.0
                for y in range(1, year_offset + 1):
                    age_at_y = last_age + (cutoff_year - last_season) + y
                    aging_total += _get_smoothed_aging_delta(stat_curves, int(age_at_y))

                val = year1_base[stat] + aging_total

                # Apply physical bounds
                if stat in BATTER_BOUNDS:
                    lo, hi = BATTER_BOUNDS[stat]
                    val = float(np.clip(val, lo, hi))
                projected[stat] = val

            # Normalize batted ball rates so GB% + LD% + FB% ≈ 1.0
            gb = projected['GB%']
            ld = projected['LD%']
            fb_raw = 1.0 - gb - ld
            if fb_raw < 0.10:
                total = gb + ld
                if total > 0:
                    projected['GB%'] = gb / total * 0.90
                    projected['LD%'] = ld / total * 0.90

            # ---- Compose all derived stats ----
            composed = compose_all(
                k_pct=projected['K%'],
                bb_pct=projected['BB%'],
                hbp_pct=projected['HBP%'],
                iso=projected['ISO'],
                babip=projected['BABIP'],
                hr_fb=projected['HR/FB'],
                gb_pct=projected['GB%'],
                ld_pct=projected['LD%'],
                triple_share=triple_share,
            )

            # Scale RBI and R using wOBA ratio from career profile
            proj_woba = composed['wOBA']
            if base_woba > 0.15:
                woba_ratio = np.clip(proj_woba / base_woba, 0.50, 1.50)
            else:
                woba_ratio = 1.0
            rbi = max(0.0, rbi_rate * woba_ratio)
            r_count = max(0.0, r_rate * woba_ratio)

            # Build output row
            row = {
                'Name': player_name,
                'IDfg': player_id,
                'Year': proj_year,
                'Age': proj_age,
                'PA': 650,
                # Base components
                'K%':    projected['K%'],
                'BB%':   projected['BB%'],
                'HBP%':  projected['HBP%'],
                'ISO':   projected['ISO'],
                'BABIP': projected['BABIP'],
                'HR/FB': projected['HR/FB'],
                'GB%':   projected['GB%'],
                'LD%':   projected['LD%'],
                # Composed rate stats
                'AVG':   composed['AVG'],
                'OBP':   composed['OBP'],
                'SLG':   composed['SLG'],
                'wOBA':  composed['wOBA'],
                'wRC+':  composed['wRC+'],
                'FB%':   composed['FB%'],
                # Composed counting stats (per 150 games)
                'HR':    composed['HR'],
                '2B':    composed['2B'],
                '3B':    composed['3B'],
                '1B':    composed['1B'],
                'H':     composed['H'],
                # RBI and R from career profile scaling
                'RBI':   rbi,
                'R':     r_count,
                'HBP':   composed['HBP'],
            }
            all_predictions.append(row)

    if not all_predictions:
        logger.warning("No Marcel batter predictions generated")
        return None

    result_df = pd.DataFrame(all_predictions)

    # Standard output column order (backward-compatible + new columns)
    output_cols = [
        'Name', 'IDfg', 'Year', 'Age', 'PA',
        'BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA',   # legacy rate stats
        'HR', '2B', '3B', 'RBI', 'R', 'HBP',          # counting stats
        'ISO', 'BABIP', 'HR/FB', 'GB%', 'LD%', 'FB%', # base components
        'HBP%', 'wRC+', '1B', 'H',                    # new columns
    ]
    for col in output_cols:
        if col not in result_df.columns:
            result_df[col] = 0.0
    result_df = result_df[[c for c in output_cols if c in result_df.columns]]
    result_df = result_df.sort_values(['Name', 'Year'])

    logger.info(
        f"Marcel batting (component-based): {len(result_df)} projections for "
        f"{result_df['Name'].nunique()} batters "
        f"(avg wOBA={result_df['wOBA'].mean():.3f}, avg HR={result_df['HR'].mean():.1f})"
    )
    return result_df


# ============================================================================
# MARCEL PITCHER PROJECTIONS
# ============================================================================

# Component rate stats that Marcel directly projects for pitchers.
# FIP, ERA, SIERA, K/9, BB/9, HR/9 are all reconstructed from these.
PITCHER_MARCEL_RATE_STATS = ['K%', 'BB%', 'HBP%', 'BABIP', 'HR/FB', 'GB%', 'FB%', 'LD%']

# Regression toward prior for pitchers (in true Fisher-information IP units,
# paired with recency decay weights [1.0, 0.8, 0.6] preserving 5:4:3 season ratios).
PITCHER_REGRESSION_IP = {
    'K%':    22,     # ~16-20 IP to stabilize (very sticky)
    'BB%':   40,     # ~35-40 IP to stabilize
    'HBP%':  90,     # noisy
    'BABIP': 170,    # ~180+ IP to stabilize (very noisy)
    'HR/FB': 150,    # slow stabilization, heavy regression
    'GB%':   22,     # ~16-20 IP to stabilize (very sticky)
    'FB%':   22,     # ~16-20 IP to stabilize (very sticky)
    'LD%':   80,     # ~60-80 IP to stabilize
}

# League-average priors for pitcher rate stats (approximately 2020-2024 average)
PITCHER_LEAGUE_AVG = {
    'K%':   0.222,
    'BB%':  0.082,
    'HBP%': 0.010,
    'BABIP': 0.293,
    'HR/FB': 0.115,
    'GB%':  0.430,
    'FB%':  0.350,
    'LD%':  0.220,
}

# Physical bounds for pitcher rate stats
PITCHER_BOUNDS = {
    'K%':    (0.05,  0.45),
    'BB%':   (0.02,  0.20),
    'HBP%':  (0.001, 0.04),
    'BABIP': (0.220, 0.360),
    'HR/FB': (0.03,  0.25),
    'GB%':   (0.20,  0.65),
    'FB%':   (0.15,  0.55),
    'LD%':   (0.15,  0.28),
}

# ---------------------------------------------------------------------------
# Pitcher multivariate regression equations (Phase 2b exhaustive brute-force)
#
# Separate equations for SP and RP because reliever skill profiles stabilize
# differently and different features matter (e.g. Stuff+ is more important
# for RP where pure stuff dominates short outings).
#
# Same R²-weighted blending as batters:
#   projected = r2 × multivariate + (1 − r2) × marcel_base
# ---------------------------------------------------------------------------
PITCHER_SP_MULTIVARIATE_EQUATIONS = {
    'K%': {
        'K%': 0.152914, 'Contact%': -0.093733, 'Stuff+': 0.001819,
        'Z-Contact%': -0.080183, 'O-Contact%': -0.084590, 'FBv': 0.004091,
        '_intercept': -0.180372, '_r2': 0.566,
    },
    'BB%': {
        'BB%': 0.028227, 'F-Strike%': -0.038367, 'Location+': -0.001389,
        'Zone%': -0.038033, 'Stuff+': -0.000831, 'FBv': 0.002500, 'ERA': -0.001798,
        '_intercept': 0.109696, '_r2': 0.235,
    },
    'HBP%': {
        'HBP%': 0.005164, 'Location+': 0.000583, 'ch_avg_speed': 0.000335,
        'Pitching+': -0.000810, 'Stuff+': 0.000719, 'FB%': -0.006424,
        '_intercept': -0.065534, '_r2': 0.089,
    },
    'BABIP': {
        'BABIP': 0.026115, 'FB%': -0.090702, 'Pitching+': -0.000751,
        'ch_avg_speed': -0.001349, 'ff_avg_speed': 0.000729,
        '_intercept': 0.439166, '_r2': 0.121,
    },
    'HR/FB': {
        'HR/FB': 0.029479, 'O-Contact%': -0.034033, 'K%': -0.038835,
        '_intercept': 0.151505, '_r2': 0.007,
    },
    'GB%': {
        'GB%': 0.303467, 'FB%': -0.280852, 'IFFB%': -0.096955,
        'Swing%': -0.059745, 'ch_avg_speed': 0.002482,
        '_intercept': 0.220851, '_r2': 0.579,
    },
    'FB%': {
        'FB%': 0.319543, 'GB%': -0.306019, 'IFFB%': 0.104389,
        'Swing%': 0.068787, 'F-Strike%': 0.050964,
        '_intercept': 0.315306, '_r2': 0.607,
    },
    'LD%': {
        'LD%': 0.029444, 'Pitching+': -0.000419, 'ch_avg_speed': -0.001050,
        'FB%': -0.037707,
        '_intercept': 0.341562, '_r2': 0.043,
    },
}

PITCHER_RP_MULTIVARIATE_EQUATIONS = {
    'K%': {
        'K%': 0.192442, 'Contact%': -0.154788, 'Z-Contact%': -0.088205,
        'FBv': 0.004157, 'Stuff+': 0.000960, 'GB%': -0.098306,
        '_intercept': -0.067144, '_r2': 0.399,
    },
    'BB%': {
        'BB%': 0.082734, 'F-Strike%': -0.076240, 'Zone%': -0.073591,
        'Swing%': -0.074733, 'O-Contact%': -0.047265, 'ff_avg_speed': 0.001840,
        'FB%': 0.039804,
        '_intercept': 0.034978, '_r2': 0.291,
    },
    'HBP%': {
        'HBP%': 0.010419, 'Swing%': -0.017130, 'Z-Swing%': -0.017871,
        'O-Swing%': -0.012569, 'Stuff+': 0.000272, 'Pitching+': -0.000219,
        '_intercept': 0.030153, '_r2': 0.084,
    },
    'BABIP': {
        'BABIP': -0.006000, 'FB%': -0.086430, 'Stuff+': -0.000555,
        '_intercept': 0.379093, '_r2': 0.053,
    },
    'HR/FB': {
        'HR/FB': 0.006041, 'ff_avg_spin': -0.000031, 'Stuff+': -0.000573,
        'ERA': -0.003671,
        '_intercept': 0.252398, '_r2': 0.007,
    },
    'GB%': {
        'GB%': 0.394343, 'FB%': -0.280045, 'IFFB%': -0.089278,
        'K%': -0.105318, 'Z-Swing%': -0.092269,
        '_intercept': 0.460009, '_r2': 0.571,
    },
    'FB%': {
        'FB%': 0.291781, 'GB%': -0.361050, 'ff_avg_spin': 0.000059,
        'IFFB%': 0.092249, 'ERA': -0.005357,
        '_intercept': 0.301656, '_r2': 0.568,
    },
    'LD%': {
        'LD%': 0.043393, 'Pitching+': -0.000876, 'Location+': 0.001064,
        'K%': 0.050035,
        '_intercept': 0.159601, '_r2': 0.038,
    },
}

# Column mapping: Phase 2b feature names → actual pitching CSV column names
_PITCHER_STATCAST_COL_MAP = {
    'FBv':          'vFA (sc)',
    'ch_avg_speed': 'sc_ch_avg_speed',
    'ff_avg_speed': 'sc_ff_avg_speed',
    'ff_avg_spin':  'sc_ff_avg_spin',
}

# League-average fallbacks for auxiliary pitcher features
_PITCHER_AUX_FEATURE_LEAGUE_AVG = {
    'Contact%':     0.776,
    'Z-Contact%':   0.828,
    'O-Contact%':   0.654,
    'Stuff+':       100.0,
    'Location+':    100.0,
    'Pitching+':    100.0,
    'FBv':          93.5,
    'F-Strike%':    0.607,
    'Zone%':        0.450,
    'Swing%':       0.465,
    'Z-Swing%':     0.685,
    'O-Swing%':     0.300,
    'IFFB%':        0.100,
    'ERA':          4.20,
    'ch_avg_speed':  84.0,
    'ff_avg_speed':  93.5,
    'ff_avg_spin':  2250.0,
}


def _apply_pitcher_multivariate_equations(
    recent_features: Dict[str, float],
    marcel_base: Dict[str, float],
    role: str,
    recent_ip: float = 150.0,
) -> Dict[str, float]:
    """
    Apply Phase 2b multivariate regression equations for a pitcher.

    Scales R² by sample-size reliability (min(1, recent_ip / stab_ip)) so
    small-sample stints (e.g. 15 IP of high Stuff+) cannot override Marcel's
    regressed estimate.
    """
    equations = PITCHER_SP_MULTIVARIATE_EQUATIONS if role == 'SP' else PITCHER_RP_MULTIVARIATE_EQUATIONS
    # Pitch-physics features (Stuff+, Location+, FBv, Contact%) stabilize rapidly (~15-30 IP)
    stab_ip = 30.0 if role == 'SP' else 15.0
    ip_reliability = min(1.0, max(0.0, float(recent_ip)) / stab_ip)

    result = {}
    for component, equation in equations.items():
        intercept = equation['_intercept']
        r2 = equation['_r2']
        effective_r2 = r2 * ip_reliability

        # Compute multivariate prediction
        mv_pred = intercept
        for feat, coef in equation.items():
            if feat.startswith('_'):
                continue
            val = recent_features.get(feat, np.nan)
            if pd.isna(val):
                val = marcel_base.get(feat, _PITCHER_AUX_FEATURE_LEAGUE_AVG.get(feat, 0.0))
            mv_pred += coef * val

        # Sample-scaled R² blend
        mb = marcel_base.get(component, PITCHER_LEAGUE_AVG.get(component, 0.0))
        result[component] = effective_r2 * mv_pred + (1.0 - effective_r2) * mb

        # Clip to physical bounds
        if component in PITCHER_BOUNDS:
            lo, hi = PITCHER_BOUNDS[component]
            result[component] = float(np.clip(result[component], lo, hi))

    return result


# FIP constant and BF/IP fallback (synced with pitcher_prediction.py)
_FIP_CONSTANT = 3.15
_BF_PER_IP_FALLBACK = 4.25

_cached_primary_batter_ids_by_cutoff: Dict[int, Set[int]] = {}


def _get_primary_batter_ids(cutoff_year: int) -> Set[int]:
    """Return IDfg set of players with >= 50 career MLB PA through cutoff_year."""
    if cutoff_year in _cached_primary_batter_ids_by_cutoff:
        return _cached_primary_batter_ids_by_cutoff[cutoff_year]

    data_dir = Path(__file__).resolve().parents[2] / 'data' / 'historic_mlb'
    cand_files = [
        data_dir / 'mlb_batting_data_1950_2025_with_statcast.csv',
        data_dir / 'mlb_batting_data_1950_2025.csv',
    ]
    b_ids: Set[int] = set()
    for p in cand_files:
        if p.exists():
            try:
                bdf = pd.read_csv(p, usecols=['IDfg', 'Season', 'PA'], low_memory=False)
                bdf['IDfg'] = pd.to_numeric(bdf['IDfg'], errors='coerce')
                bdf['Season'] = pd.to_numeric(bdf['Season'], errors='coerce')
                bdf['PA'] = pd.to_numeric(bdf['PA'], errors='coerce').fillna(0.0)
                sub = bdf[(bdf['IDfg'].notna()) & (bdf['Season'] <= cutoff_year)]
                pa_by_fg = sub.groupby('IDfg')['PA'].sum()
                b_ids = {int(fg) for fg, tot in pa_by_fg.items() if tot >= 50.0}
                break
            except Exception:
                continue
    _cached_primary_batter_ids_by_cutoff[cutoff_year] = b_ids
    return b_ids


def _apply_xstat_shrinkage_to_pitcher_components(player_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Shrink noisy seasonal pitcher `HR/FB` and `BABIP` toward expected run-prevention
    anchors (`xERA`, `SIERA`, `xFIP`) weighted by IP sample reliability.

    Strips out small-sample 0-HR or BABIP flukes (e.g. 25 IP of 3.50 FIP backed
    by 5.20 xERA / 4.30 SIERA) while preserving genuine multi-season contact skill.
    """
    out = player_hist.copy()
    for idx, row in out.iterrows():
        ip = row.get('IP', np.nan)
        if pd.isna(ip) or float(ip) <= 0:
            continue
        ip_f = float(ip)

        xera = row.get('xERA', np.nan)
        siera = row.get('SIERA', np.nan)
        fip = row.get('FIP', np.nan)
        era = row.get('ERA', np.nan)

        target_est = np.nan
        if pd.notna(xera) and pd.notna(siera):
            target_est = 0.55 * float(xera) + 0.45 * float(siera)
        elif pd.notna(xera):
            target_est = float(xera)
        elif pd.notna(siera):
            target_est = float(siera)

        # 1. HR/FB shrinkage (w_hr = IP / (IP + 110))
        hr_fb = row.get('HR/FB', np.nan)
        if pd.notna(hr_fb):
            w_hr = ip_f / (ip_f + 110.0)
            if pd.notna(xera) and pd.notna(siera):
                # Contact quality difference (xERA - SIERA) informs barrel/HR suppression
                hr_fb_target = float(np.clip(0.115 + 0.014 * (float(xera) - float(siera)), 0.075, 0.165))
            else:
                hr_fb_target = 0.115
            out.at[idx, 'HR/FB'] = w_hr * float(hr_fb) + (1.0 - w_hr) * hr_fb_target

        # 2. BABIP shrinkage (w_babip = IP / (IP + 120))
        babip = row.get('BABIP', np.nan)
        if pd.notna(babip) and pd.notna(target_est):
            w_babip = ip_f / (ip_f + 120.0)
            ref_run = float(fip) if pd.notna(fip) else (float(era) if pd.notna(era) else target_est)
            babip_target = float(np.clip(0.293 + 0.012 * np.clip(target_est - ref_run, -1.5, 1.5), 0.255, 0.325))
            out.at[idx, 'BABIP'] = w_babip * float(babip) + (1.0 - w_babip) * babip_target

    return out


def _project_reliever_gmli(player_hist: pd.DataFrame, siera_y1: float) -> float:
    """
    Project Year-1 reliever Leverage Index (gmLI) from trailing IP-weighted gmLI
    regressed toward a skill-conditioned baseline.
    """
    skill_gmli = float(np.clip(1.0 + (3.95 - float(siera_y1)) * 0.26, 0.72, 1.35))
    if 'gmLI' not in player_hist.columns:
        return skill_gmli

    recent = player_hist.tail(len(SEASON_WEIGHTS)).iloc[::-1]
    top_w = float(SEASON_WEIGHTS[0])
    w_sum = 0.0
    tot_w = 0.0
    for i, (_, row) in enumerate(recent.iterrows()):
        gmli = row.get('gmLI', np.nan)
        ip = row.get('IP', 0.0)
        if pd.isna(gmli) or pd.isna(ip) or float(ip) <= 0:
            continue
        decay = float(SEASON_WEIGHTS[i]) / top_w
        wt = decay * float(ip)
        w_sum += float(gmli) * wt
        tot_w += wt

    reg_ip = 35.0
    proj = (w_sum + skill_gmli * reg_ip) / (tot_w + reg_ip)
    return float(np.clip(proj, 0.55, 2.00))


# ERA-FIP stabilization TBF (synced with pitcher_prediction.py)
_ERA_FIP_STAB_TBF = 2000

# SIERA reconstruction coefficients (synced with pitcher_prediction.py)
_SIERA_INTERCEPT = 6.8905
_SIERA_COEFS = {
    'K%':     -16.9845,
    'BB%':     +4.4756,
    'GB%':     -0.5257,
    'K%^2':    +3.9039,
    'BB%^2':  +11.9800,
    'GB%^2':   -6.4369,
    'K%*BB%':  -6.3522,
    'K%*GB%': +12.0693,
    'BB%*GB%':+13.9631,
}


def _normalize_batted_ball(gb: float, fb: float, ld: float) -> Tuple[float, float, float]:
    """Normalize GB%+FB%+LD% to sum to 1.0."""
    total = gb + fb + ld
    if total <= 0:
        return 0.43, 0.35, 0.22
    return gb / total, fb / total, ld / total


def _derive_bf_per_ip(k_pct: float, bb_pct: float, hbp_pct: float,
                      hr_pct: float, babip: float) -> float:
    """Derive BF/IP from component rates and BABIP."""
    bip_rate = 1.0 - k_pct - bb_pct - hbp_pct - hr_pct
    bip_rate = max(bip_rate, 0.20)
    out_rate = k_pct + bip_rate * (1.0 - babip)
    out_rate = np.clip(out_rate, 0.45, 0.85)
    return 3.0 / out_rate


def _reconstruct_fip(k_pct: float, bb_pct: float, hbp_pct: float,
                     hr_fb: float, fb_pct: float, bf_per_ip: float) -> float:
    """Reconstruct FIP from component rates."""
    bip_rate = max(1.0 - k_pct - bb_pct - hbp_pct, 0.20)
    hr_pct = hr_fb * fb_pct * bip_rate
    component_sum = 13.0 * hr_pct + 3.0 * (bb_pct + hbp_pct) - 2.0 * k_pct
    return np.clip(bf_per_ip * component_sum + _FIP_CONSTANT, 0.5, 10.0)


def _reconstruct_siera(k_pct: float, bb_pct: float, gb_pct: float) -> float:
    """Reconstruct SIERA from K%, BB%, GB%."""
    return np.clip(
        _SIERA_INTERCEPT
        + _SIERA_COEFS['K%']     * k_pct
        + _SIERA_COEFS['BB%']    * bb_pct
        + _SIERA_COEFS['GB%']    * gb_pct
        + _SIERA_COEFS['K%^2']   * k_pct ** 2
        + _SIERA_COEFS['BB%^2']  * bb_pct ** 2
        + _SIERA_COEFS['GB%^2']  * gb_pct ** 2
        + _SIERA_COEFS['K%*BB%'] * k_pct * bb_pct
        + _SIERA_COEFS['K%*GB%'] * k_pct * gb_pct
        + _SIERA_COEFS['BB%*GB%']* bb_pct * gb_pct,
        1.0, 8.0
    )


def _compute_career_era_fip_gap_marcel(
    player_hist: pd.DataFrame,
) -> float:
    """
    Compute regressed career ERA-FIP gap for a single pitcher.

    Uses IP-weighted career ERA-FIP gap, regressed toward 0 via
    James-Stein shrinkage with n0 = ERA_FIP_STAB_TBF.
    """
    required = {'ERA', 'FIP', 'IP'}
    if not required.issubset(set(player_hist.columns)):
        return 0.0

    valid = player_hist.dropna(subset=['ERA', 'FIP', 'IP'])
    valid = valid[valid['IP'] > 0]
    if len(valid) == 0:
        return 0.0

    total_ip = valid['IP'].sum()
    if total_ip <= 0:
        return 0.0

    weighted_era = (valid['ERA'] * valid['IP']).sum() / total_ip
    weighted_fip = (valid['FIP'] * valid['IP']).sum() / total_ip
    raw_gap = weighted_era - weighted_fip

    # Estimate career TBF from IP
    if 'TBF' in valid.columns and valid['TBF'].notna().any():
        career_tbf = valid['TBF'].sum()
    else:
        career_tbf = total_ip * _BF_PER_IP_FALLBACK

    signal_fraction = career_tbf / (career_tbf + _ERA_FIP_STAB_TBF)
    return raw_gap * signal_fraction


def marcel_pitcher_projections(
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    future_years: int = 15,
    cutoff_year: int = 2025,
    roster_ids: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    """
    Generate Marcel-style pitcher projections for both SP and RP.

    Projects component rate stats (K%, BB%, HBP%, BABIP, HR/FB, GB%, FB%, LD%)
    using the component-based Marcel method:
      1. Weighted average of last 3 seasons (5/4/3 × IP) with regression
      2. Multivariate adjustment via Phase 2b equations (role-specific SP/RP)
         R²-weighted blend of multivariate prediction and Marcel base
      3. Apply empirical aging curves year-by-year

    Then reconstructs:
      - FIP from K%, BB%, HBP%, HR/FB, FB% components
      - ERA = FIP + regressed career ERA-FIP gap
      - SIERA from K%, BB%, GB% (quadratic model)
      - K/9, BB/9, HR/9 from per-TBF rates × BF/IP

    SP/RP role is determined by GS rate (≥0.8 = SP) in the most recent season.

    Output format matches the LSTM pitcher_predictions.csv exactly.

    Args:
        raw_df: Historical pitching data with rate stats computed
        player_names: DataFrame with IDfg and Name columns
        future_years: Number of years to project
        cutoff_year: Last year of actual data
        roster_ids: Optional set of IDfg values for active roster players

    Returns:
        DataFrame matching the pitcher_predictions.csv format
    """
    from configs.pitcher_sp_config import PitcherSPConfig
    from configs.pitcher_rp_config import PitcherRPConfig

    curves = _load_aging_curves()
    pitching_curves = curves.get('pitching', {})

    sp_min_ip = getattr(PitcherSPConfig, 'MIN_IP_CURRENT', 25)
    rp_min_ip = getattr(PitcherRPConfig, 'MIN_IP_CURRENT', 15)

    stats = list(PITCHER_MARCEL_RATE_STATS)

    # Identify SP vs RP by GS rate in the most recent season
    pitchers_current = raw_df[raw_df['Season'] == cutoff_year].copy()
    pitchers_prev = raw_df[raw_df['Season'] == cutoff_year - 1].copy()

    # Build {player_id: role} mapping
    player_roles: Dict[int, str] = {}
    qualifying_ids: Set[int] = set()

    if not pitchers_current.empty and 'GS' in pitchers_current.columns and 'G' in pitchers_current.columns:
        for pid in pitchers_current['IDfg'].unique():
            pdata = pitchers_current[pitchers_current['IDfg'] == pid]
            total_g = pdata['G'].sum()
            total_gs = pdata['GS'].sum()
            total_ip = pdata['IP'].sum() if 'IP' in pdata.columns else 0

            if total_g == 0:
                continue

            gs_rate = total_gs / total_g

            if gs_rate >= 0.8 and total_ip >= sp_min_ip:
                player_roles[pid] = 'SP'
                qualifying_ids.add(pid)
            elif gs_rate < 0.8 and total_ip >= rp_min_ip:
                player_roles[pid] = 'RP'
                qualifying_ids.add(pid)

    # Recover from previous year if not in current
    if not pitchers_prev.empty and 'GS' in pitchers_prev.columns:
        for pid in pitchers_prev['IDfg'].unique():
            if pid in player_roles:
                continue
            pdata = pitchers_prev[pitchers_prev['IDfg'] == pid]
            total_g = pdata['G'].sum()
            total_gs = pdata['GS'].sum()
            total_ip = pdata['IP'].sum() if 'IP' in pdata.columns else 0

            if total_g == 0:
                continue
            gs_rate = total_gs / total_g

            if gs_rate >= 0.8 and total_ip >= sp_min_ip:
                player_roles[pid] = 'SP'
                qualifying_ids.add(pid)
            elif gs_rate < 0.8 and total_ip >= rp_min_ip:
                player_roles[pid] = 'RP'
                qualifying_ids.add(pid)

    # Roster recovery (include MLB-debuted roster pitchers with >= 1 IP,
    # excluding position players who only threw mop-up blowout innings!)
    primary_batter_ids = _get_primary_batter_ids(cutoff_year)
    if roster_ids is not None:
        missing = roster_ids - qualifying_ids
        if missing:
            for pid in missing:
                if pid in player_roles:
                    continue
                phist = raw_df[
                    (raw_df['IDfg'] == pid) &
                    (raw_df['Season'] <= cutoff_year) &
                    (raw_df['IP'] >= 1.0)
                ]
                if phist.empty:
                    continue
                if int(pid) in primary_batter_ids:
                    max_ip_szn = float(phist['IP'].max())
                    max_gs_szn = float(phist['GS'].max()) if 'GS' in phist.columns else 0.0
                    if max_ip_szn < 15.0 and max_gs_szn < 1.0:
                        continue  # Position-player mop-up appearance, not a two-way pitcher
                last = phist.sort_values('Season').iloc[-1]
                if last['G'] > 0:
                    gs_rate = last['GS'] / last['G']
                    player_roles[pid] = 'SP' if gs_rate >= 0.8 else 'RP'
                    qualifying_ids.add(pid)

    # Also drop any primary batter who slipped into qualifying_ids via <15 IP mop-up stints
    mopup_ids = [
        pid for pid in list(player_roles.keys())
        if int(pid) in primary_batter_ids and (
            float(raw_df[(raw_df['IDfg'] == pid) & (raw_df['Season'] <= cutoff_year)]['IP'].max() or 0.0) < 15.0
            and float(raw_df[(raw_df['IDfg'] == pid) & (raw_df['Season'] <= cutoff_year)]['GS'].max() or 0.0) < 1.0
        )
    ]
    for pid in mopup_ids:
        player_roles.pop(pid, None)
        qualifying_ids.discard(pid)
    if mopup_ids:
        logger.info(f"Marcel pitching: excluded {len(mopup_ids)} position-player mop-up pitchers")

    logger.info(f"Marcel pitching: {len(player_roles)} pitchers "
                f"({sum(1 for r in player_roles.values() if r == 'SP')} SP, "
                f"{sum(1 for r in player_roles.values() if r == 'RP')} RP)")

    try:
        from core.id_crosswalk import load_recent_prospect_info
        pitcher_prospect_lookup = load_recent_prospect_info(cutoff_year + 1, recency_years=3)
    except Exception as exc:
        logger.warning(f"Could not load prospect info for pitcher priors: {exc}")
        pitcher_prospect_lookup = {}

    def _pitcher_fv_component_prior(fv: Optional[float], p_role: str) -> Dict[str, float]:
        if fv is None or fv < 40.0:
            # Unconditional AAA/waiver replacement pitcher priors (~4.68 SP FIP, ~4.36 RP FIP)
            if p_role == 'SP':
                return {
                    'K%': 0.185, 'BB%': 0.094, 'HBP%': 0.011, 'BABIP': 0.298,
                    'HR/FB': 0.130, 'GB%': 0.420, 'FB%': 0.360, 'LD%': 0.220,
                }
            return {
                'K%': 0.212, 'BB%': 0.100, 'HBP%': 0.012, 'BABIP': 0.295,
                'HR/FB': 0.124, 'GB%': 0.425, 'FB%': 0.355, 'LD%': 0.220,
            }
        t = float(np.clip((fv - 40.0) / 30.0, 0.0, 1.0))
        if p_role == 'SP':
            return {
                'K%':    0.198 + 0.085 * t,   # 19.8% -> 28.3%
                'BB%':   0.092 - 0.024 * t,   #  9.2% ->  6.8%
                'HBP%':  0.010,
                'BABIP': 0.296 - 0.011 * t,
                'HR/FB': 0.125 - 0.020 * t,
                'GB%':   0.428 + 0.027 * t,
                'FB%':   0.352 - 0.027 * t,
                'LD%':   0.220,
            }
        return {
            'K%':    0.225 + 0.095 * t,   # 22.5% -> 32.0%
            'BB%':   0.096 - 0.022 * t,
            'HBP%':  0.011,
            'BABIP': 0.293 - 0.013 * t,
            'HR/FB': 0.120 - 0.017 * t,
            'GB%':   0.432 + 0.022 * t,
            'FB%':   0.348 - 0.022 * t,
            'LD%':   0.220,
        }

    all_predictions = []
    rookie_pitchers_adjusted = 0

    # Collect all features used by SP and RP multivariate equations
    all_pitcher_eq_features = set()
    for eq_dict in (PITCHER_SP_MULTIVARIATE_EQUATIONS, PITCHER_RP_MULTIVARIATE_EQUATIONS):
        for eq in eq_dict.values():
            for feat in eq:
                if not feat.startswith('_'):
                    all_pitcher_eq_features.add(feat)

    for player_id, role in player_roles.items():
        min_ip_season = 20 if role == 'SP' else 10

        all_pit_mlb = raw_df[
            (raw_df['IDfg'] == player_id) &
            (raw_df['Season'] <= cutoff_year) &
            (raw_df['IP'] >= 1.0)
        ]
        if all_pit_mlb.empty:
            continue

        career_ip = float(all_pit_mlb['IP'].fillna(0).sum())
        player_hist = all_pit_mlb[all_pit_mlb['IP'] >= min_ip_season].copy()
        if player_hist.empty:
            player_hist = all_pit_mlb.copy()

        _raw_for_name = raw_df if 'raw_df' in locals() else None
        player_name = _safe_resolve_player_name(player_id, player_names, _raw_for_name)

        player_hist = player_hist.sort_values('Season')
        _valid_br_ages = pd.to_numeric(player_hist['Age'], errors='coerce').dropna()
        last_age = float(_valid_br_ages.iloc[-1]) if not _valid_br_ages.empty else 27.0
        last_season = player_hist['Season'].iloc[-1]

        # Build player-specific Bayesian prior P_0(theta | FV, Role, career_ip)
        rookie_stab_ip = 150.0 if role == 'SP' else 70.0
        if career_ip < rookie_stab_ip:
            surv_w = float(np.clip(career_ip / rookie_stab_ip, 0.0, 1.0))
            p_info = pitcher_prospect_lookup.get(int(player_id))
            fv_val = p_info['fv'] if p_info else None
            base_prior = _pitcher_fv_component_prior(fv_val, role)
            player_prior = {
                st: surv_w * PITCHER_LEAGUE_AVG[st] + (1.0 - surv_w) * base_prior[st]
                for st in stats
            }
            rookie_pitchers_adjusted += 1
        else:
            player_prior = PITCHER_LEAGUE_AVG

        # Derive HBP% if not present
        if 'HBP%' not in player_hist.columns:
            if 'HBP' in player_hist.columns and 'TBF' in player_hist.columns:
                player_hist['HBP%'] = player_hist['HBP'] / player_hist['TBF'].replace(0, np.nan)
            elif 'HBP' in player_hist.columns and 'IP' in player_hist.columns:
                tbf_est = player_hist['IP'] * _BF_PER_IP_FALLBACK
                player_hist['HBP%'] = player_hist['HBP'] / tbf_est.replace(0, np.nan)
            else:
                player_hist['HBP%'] = player_prior['HBP%']

        # Shrink noisy seasonal HR/FB and BABIP toward xERA/SIERA expected contact anchors
        hist_for_marcel = _apply_xstat_shrinkage_to_pitcher_components(player_hist)

        # ---- Extract most recent season features for multivariate equations ----
        most_recent = hist_for_marcel.iloc[-1]
        recent_features = {}
        for feat in all_pitcher_eq_features:
            col_name = _PITCHER_STATCAST_COL_MAP.get(feat, feat)
            val = most_recent.get(col_name, np.nan)
            if pd.isna(val) and feat == 'FBv':
                val = most_recent.get('FAv', np.nan)
            if pd.notna(val):
                recent_features[feat] = float(val)

        # Build recent seasons list (newest first)
        recent = hist_for_marcel.tail(len(SEASON_WEIGHTS)).iloc[::-1]
        seasons_data = []
        for _, row in recent.iterrows():
            s = {'IP': row.get('IP', 100)}
            for stat in stats:
                s[stat] = row.get(stat, np.nan)
            seasons_data.append(s)

        # Single conjugate posterior shrinking directly toward player_prior P_0
        marcel_base = _compute_marcel_weighted_average_toward_league(
            seasons_data, stats, 'IP', PITCHER_REGRESSION_IP, player_prior
        )

        # Apply multivariate equations → Year 1 base (role-specific, fast physics stabilization)
        recent_ip = float(most_recent.get('IP', 100.0))
        year1_base = _apply_pitcher_multivariate_equations(
            recent_features, marcel_base, role, recent_ip=recent_ip
        )

        # Project reliever leverage index (gmLI) from trailing usage + projected SIERA
        gb1, fb1, ld1 = _normalize_batted_ball(year1_base['GB%'], year1_base['FB%'], year1_base['LD%'])
        siera_y1 = float(_reconstruct_siera(year1_base['K%'], year1_base['BB%'], gb1))
        gmli_y1 = _project_reliever_gmli(player_hist, siera_y1) if role == 'RP' else 1.0

        # Career ERA-FIP gap
        era_fip_gap = _compute_career_era_fip_gap_marcel(player_hist)

        # Project forward with aging (year_offset=0 emits cutoff_year directly, matching batters)
        for year_offset in range(0, future_years + 1):
            proj_year = cutoff_year + year_offset
            proj_age = last_age + (cutoff_year - last_season) + year_offset

            # Apply cumulative aging to each stat from year1_base
            projected = {}
            for stat in stats:
                stat_curves = pitching_curves.get(stat, {})
                aging_total = 0.0
                for y in range(1, year_offset + 1):
                    age_at_y = last_age + (cutoff_year - last_season) + y
                    aging_total += _get_smoothed_aging_delta(stat_curves, int(age_at_y))

                val = year1_base[stat] + aging_total

                # Apply physical bounds
                if stat in PITCHER_BOUNDS:
                    lo, hi = PITCHER_BOUNDS[stat]
                    val = np.clip(val, lo, hi)

                projected[stat] = val

            # Normalize batted ball rates
            gb, fb, ld = _normalize_batted_ball(
                projected['GB%'], projected['FB%'], projected['LD%']
            )
            projected['GB%'] = gb
            projected['FB%'] = fb
            projected['LD%'] = ld

            # Derive HR% and BF/IP
            bip_rate = max(1.0 - projected['K%'] - projected['BB%'] - projected['HBP%'], 0.20)
            hr_pct = projected['HR/FB'] * projected['FB%'] * bip_rate
            bf_per_ip = _derive_bf_per_ip(
                projected['K%'], projected['BB%'], projected['HBP%'],
                hr_pct, projected['BABIP']
            )

            # Reconstruct FIP
            fip = _reconstruct_fip(
                projected['K%'], projected['BB%'], projected['HBP%'],
                projected['HR/FB'], projected['FB%'], bf_per_ip
            )

            # ERA = FIP + regressed ERA-FIP gap
            era = np.clip(fip + era_fip_gap, 0.5, 10.0)

            # Reconstruct SIERA
            siera = _reconstruct_siera(projected['K%'], projected['BB%'], projected['GB%'])

            # Decay reliever gmLI gently toward 1.08 across future seasons (role turnover)
            if role == 'RP':
                decay_steps = max(0, year_offset - 1)
                proj_gmli = float(np.clip(1.08 + (gmli_y1 - 1.08) * (0.88 ** decay_steps), 0.55, 2.00))
            else:
                proj_gmli = 1.0

            # Derive per-9 rates
            k_per_9 = projected['K%'] * bf_per_ip * 9.0
            bb_per_9 = projected['BB%'] * bf_per_ip * 9.0
            hr_per_9 = hr_pct * bf_per_ip * 9.0

            # Build output row
            row = {
                'Name': player_name,
                'Year': proj_year,
                'Age': proj_age,
                'Role': role,
                'IDfg': player_id,
                'K%': projected['K%'],
                'BB%': projected['BB%'],
                'HBP%': projected['HBP%'],
                'BABIP': projected['BABIP'],
                'FIP': fip,
                'ERA': era,
                'GB%': projected['GB%'],
                'FB%': projected['FB%'],
                'HR/FB': projected['HR/FB'],
                'SIERA': siera,
                'LD%': projected['LD%'],
                'K/9': k_per_9,
                'BB/9': bb_per_9,
                'HR/9': hr_per_9,
                'gmLI': round(proj_gmli, 3),
            }

            all_predictions.append(row)

    if not all_predictions:
        logger.warning("No Marcel pitcher predictions generated")
        return None

    result_df = pd.DataFrame(all_predictions)
    result_df = result_df.sort_values(['Name', 'Year'])

    logger.info(f"Marcel pitching: generated {len(result_df)} projections for "
                f"{result_df['Name'].nunique()} pitchers")
    return result_df