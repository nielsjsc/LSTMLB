"""
Marcel-style projection engine for fielding and baserunning.

Instead of autoregressive LSTM predictions, this uses the classic Marcel method:
1. Weighted average of recent seasons (5/4/3 weighting, sample-size weighted)
2. Regress toward zero (league average for fielding/baserunning)
3. Apply empirically-derived aging curves
4. Project forward year-by-year with aging adjustments

This produces more realistic projections for defensive metrics and baserunning,
which are too noisy for LSTM models to learn meaningful signal from limited data.
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


def _get_defensive_aging_delta(
    raw_deltas: Dict[str, float],
    age: int,
    stat: str,
    window: int = 3,
) -> float:
    """
    Get an aging delta for a defensive stat with a minimum decline floor.

    Same smoothing logic as _get_smoothed_aging_delta, but enforces a minimum
    per-year decline after peak ages using FIELDING_AGING_FLOOR.  If the
    smoothed empirical delta is *above* (less negative than) the floor for
    this age, the floor is used instead.
    """
    smoothed = _get_smoothed_aging_delta(raw_deltas, age, window=window)

    floor_map = FIELDING_AGING_FLOOR.get(stat, {})
    if not floor_map:
        return smoothed

    # For ages beyond the floor table, extrapolate from the last entry
    max_floor_age = max(floor_map.keys())
    if age > max_floor_age:
        last_floor = floor_map[max_floor_age]
        # Continue accelerating decline at 0.25 runs/yr beyond last entry
        floor = last_floor - 0.25 * (age - max_floor_age)
    elif age in floor_map:
        floor = floor_map[age]
    else:
        return smoothed  # Age is below the floor range — no constraint

    return min(smoothed, floor)


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
FIELDING_REGRESSION_INNINGS = {
    'sc_total_runs/150': 600,
    'sc_range_runs/150': 600,
    'sc_arm_runs/150': 800,
    'sc_dp_runs/150': 800,
    'sc_framing_runs/150': 600,
    'sc_throwing_runs/150': 1000,
    'sc_blocking_runs/150': 1000,
}

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

            recency_weight = SEASON_WEIGHTS[i]
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
        regression = FIELDING_REGRESSION_INNINGS
        output_col_map = None
        curve_key_map = None
    else:
        groups = FIELDING_GROUPS_TRADITIONAL
        regression = FIELDING_REGRESSION_INNINGS_TRADITIONAL
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

            # Get player name
            name_match = player_names[player_names['IDfg'] == player_id]
            player_name = name_match['Name'].iloc[0] if not name_match.empty else f"Unknown ({player_id})"

            # Get last known age
            player_hist = player_hist.sort_values('Season')
            last_age = player_hist['Age'].iloc[-1]
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

        # Player name
        name_match = player_names[player_names['IDfg'] == player_id]
        player_name = name_match['Name'].iloc[0] if not name_match.empty else f"Unknown ({player_id})"

        player_hist = player_hist.sort_values('Season')
        last_age = player_hist['Age'].iloc[-1]
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
# MARCEL BATTER PROJECTIONS
# ============================================================================

# Rate stats that Marcel directly projects for batters.
# Counting stats (HR, 2B, 3B, RBI, R, HBP) are derived from projected wOBA
# via career profile ratios — exactly as the LSTM pipeline does in Mode A.
BATTER_MARCEL_RATE_STATS = ['BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA']

# Counting stats derived from wOBA × career profile
BATTER_COUNTING_STATS = ['HR', '2B', '3B', 'RBI', 'R', 'HBP']

# Regression toward league average for batters (in PA-equivalents).
# Higher values → more regression (pull toward league avg).
BATTER_REGRESSION_PA = {
    'BB%':  400,    # ~340 PA to stabilize
    'K%':   150,    # ~60 PA to stabilize (very sticky)
    'AVG':  800,    # ~910 PA to stabilize (noisy)
    'OBP':  500,    # ~460 PA to stabilize
    'SLG':  500,    # ~320 PA to stabilize
    'wOBA': 400,    # ~310 PA to stabilize
}

# League-average priors for batting rate stats (approximately 2020-2024 average)
BATTER_LEAGUE_AVG = {
    'BB%':  0.083,
    'K%':   0.224,
    'AVG':  0.248,
    'OBP':  0.314,
    'SLG':  0.402,
    'wOBA': 0.312,
}

# Physical bounds for batter rate stats
BATTER_BOUNDS = {
    'BB%':  (0.02,  0.25),
    'K%':   (0.05,  0.40),
    'AVG':  (0.150, 0.380),
    'OBP':  (0.200, 0.500),
    'SLG':  (0.200, 0.850),
    'wOBA': (0.200, 0.500),
}

# 2025 FanGraphs wOBA linear weights (synced with batter_prediction.py)
_WOBA_WEIGHTS = {
    'wBB': 0.691,
    'wHBP': 0.722,
    'w1B': 0.882,
    'w2B': 1.252,
    'w3B': 1.584,
    'wHR': 2.037,
}


def _compute_marcel_weighted_average_toward_league(
    seasons_data: List[Dict],
    stats: List[str],
    volume_col: str,
    regression_amounts: Dict[str, float],
    league_avg: Dict[str, float],
) -> Dict[str, float]:
    """
    Marcel weighted average with regression toward league average (not zero).

    For batting/pitching, unlike fielding/baserunning (which regress to 0),
    rate stats regress toward the league mean.

    Args:
        seasons_data: List of dicts (newest first), each with stat values and volume
        stats: Stat names to project
        volume_col: Volume column name ('PA' for batters, 'IP' for pitchers)
        regression_amounts: {stat: regression_volume} for regression sizing
        league_avg: {stat: league_average_value}

    Returns:
        Dict of {stat: projected_value}
    """
    if not seasons_data:
        return {s: league_avg.get(s, 0.0) for s in stats}

    result = {}
    for stat in stats:
        weighted_sum = 0.0
        total_weight = 0.0

        for i, season in enumerate(seasons_data[:len(SEASON_WEIGHTS)]):
            val = season.get(stat, np.nan)
            vol = season.get(volume_col, 0.0)
            if pd.isna(val) or pd.isna(vol) or vol <= 0:
                continue

            recency_weight = SEASON_WEIGHTS[i]
            weight = recency_weight * vol
            weighted_sum += val * weight
            total_weight += weight

        # Regression: add league-average volume (flat, not scaled by season weight)
        reg_vol = regression_amounts.get(stat, 400)
        lg_avg = league_avg.get(stat, 0.0)
        weighted_sum += lg_avg * reg_vol
        total_weight += reg_vol

        if total_weight > 0:
            result[stat] = weighted_sum / total_weight
        else:
            result[stat] = lg_avg

    return result


def _build_batter_career_profile(
    player_hist: pd.DataFrame,
    n_recent: int = 3,
) -> Optional[Dict]:
    """
    Build a PA-weighted career counting profile for a single batter.

    Returns profile dict with base_woba and per-150 counting stat rates,
    matching the format expected by the counting derivation step.
    """
    counting_available = [s for s in BATTER_COUNTING_STATS if s in player_hist.columns]

    recent = player_hist.tail(n_recent)
    if len(recent) == 0 or 'PA' not in recent.columns:
        return None

    pa = recent['PA'].values.astype(float)
    total_pa = pa.sum()
    if total_pa == 0:
        return None
    weights = pa / total_pa

    # Baseline wOBA
    if 'wOBA' not in recent.columns:
        return None
    woba_vals = np.nan_to_num(recent['wOBA'].values.astype(float), nan=0.0)
    base_woba = float(np.average(woba_vals, weights=weights))
    if base_woba < 0.15:
        return None

    # Per-150 counting rates
    base_counts = {}
    for stat in counting_available:
        vals = np.nan_to_num(recent[stat].values.astype(float), nan=0.0)
        base_counts[stat] = float(np.average(vals, weights=weights))

    career_pa = float(player_hist['PA'].sum())

    return {
        'base_woba': base_woba,
        'base_counts': base_counts,
        'career_pa': career_pa,
    }


def _derive_counting_from_woba(
    projected_woba: float,
    career_profile: Dict,
    pa_full: float = 1500.0,
) -> Dict[str, float]:
    """
    Derive counting stats from projected wOBA × career profile ratio.

    Matches the LSTM pipeline's Mode A (CALCULATE_COMPONENTS_FROM_WOBA):
      ratio = projected_wOBA / career_wOBA
      derived = career_count_per150 × ratio
      blend = min(career_PA / pa_full, 1.0)
      final = derived  (pure derivation for Marcel; no model fallback)
    """
    base_woba = career_profile['base_woba']
    if base_woba < 0.15:
        return {}

    ratio = np.clip(projected_woba / base_woba, 0.50, 1.50)

    counts = {}
    for stat, career_rate in career_profile['base_counts'].items():
        counts[stat] = max(0.0, career_rate * ratio)

    return counts


def _reconstruct_rate_stats(
    bb_pct: float,
    avg: float,
    counting: Dict[str, float],
    pa: float = 650.0,
) -> Dict[str, float]:
    """
    Reconstruct OBP, SLG, and wOBA from counting components.

    Ensures rate stat consistency after counting derivation.
    """
    bb = bb_pct * pa
    hbp = counting.get('HBP', pa * 0.01)
    sf = pa * 0.007
    ab = pa - bb - hbp - sf
    if ab <= 0:
        return {}

    h = avg * ab
    hr = counting.get('HR', 0.0)
    doubles = counting.get('2B', 0.0)
    triples = counting.get('3B', 0.0)
    singles = max(0.0, h - doubles - triples - hr)

    # OBP
    obp_den = ab + bb + hbp + sf
    obp = np.clip((h + bb + hbp) / obp_den, 0, 1) if obp_den > 0 else 0.314

    # SLG
    slg_num = singles + 2.0 * doubles + 3.0 * triples + 4.0 * hr
    slg = np.clip(slg_num / ab, 0, 4) if ab > 0 else 0.402

    # wOBA
    w = _WOBA_WEIGHTS
    woba_num = (w['wBB'] * bb + w['wHBP'] * hbp + w['w1B'] * singles +
                w['w2B'] * doubles + w['w3B'] * triples + w['wHR'] * hr)
    woba = np.clip(woba_num / pa, 0, 1) if pa > 0 else 0.312

    return {'OBP': obp, 'SLG': slg, 'wOBA': woba}


def marcel_batter_projections(
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    future_years: int = 15,
    cutoff_year: int = 2025,
    roster_ids: Optional[Set[int]] = None,
    use_xstats: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Generate Marcel-style batter projections.

    Projects rate stats (BB%, K%, AVG, OBP, SLG, wOBA) using Marcel method:
      1. Weighted average of last 3 seasons (5/4/3 × PA)
      2. Regress toward league-average
      3. Apply empirical aging curves year-by-year

    Then derives counting stats (HR, 2B, 3B, RBI, R, HBP) from projected wOBA
    using career-profile ratios, and reconstructs OBP/SLG/wOBA for consistency.

    When use_xstats=True, xwOBA replaces wOBA, xBA replaces AVG, and xSLG
    replaces SLG in the input (x-stats are more predictive of future
    performance, matching the LSTM pipeline's USE_XWOBA_FOR_PREDICTIONS).

    Output format matches the LSTM batter_predictions.csv exactly.

    Args:
        raw_df: Historical batting data with rate stats computed
        player_names: DataFrame with IDfg and Name columns
        future_years: Number of years to project
        cutoff_year: Last year of actual data
        roster_ids: Optional set of IDfg values for active roster players
        use_xstats: Whether to substitute x-stats (xwOBA, xBA, xSLG)

    Returns:
        DataFrame matching the batter_predictions.csv format
    """
    from configs.batter_config import BatterConfig

    curves = _load_aging_curves()
    batting_curves = curves.get('batting', {})

    min_pa_current = getattr(BatterConfig, 'MIN_PA_CURRENT', 70)
    pa_full = getattr(BatterConfig, 'COMPONENTS_FROM_WOBA_PA_WEIGHT', 1500)

    # xwOBA-wOBA gap adjustment: add back a regressed portion of persistent over/under-performance
    enable_gap_adj = use_xstats and getattr(BatterConfig, 'ENABLE_XWOBA_GAP_ADJUSTMENT', False)
    gap_skill_fraction = getattr(BatterConfig, 'XWOBA_GAP_SKILL_FRACTION', 0.5)
    gap_min_seasons = getattr(BatterConfig, 'XWOBA_GAP_MIN_SEASONS', 2)
    gap_regress_pa = getattr(BatterConfig, 'XWOBA_GAP_REGRESSION_PA', 800)

    # Get current-year qualifying batters
    current_players = raw_df[
        (raw_df['Season'] == cutoff_year) &
        (raw_df['PA'] >= min_pa_current)
    ]['IDfg'].unique()

    current_ids = set(current_players)

    # Roster recovery: add roster players from history
    if roster_ids is not None:
        missing = roster_ids - current_ids
        if missing:
            historical = set(raw_df[
                (raw_df['IDfg'].isin(missing)) &
                (raw_df['Season'] <= cutoff_year) &
                (raw_df['PA'] >= 50)
            ]['IDfg'].unique())
            current_ids = current_ids | historical
            if historical:
                logger.info(f"Marcel batting: recovered {len(historical)} roster players from history")

    logger.info(f"Marcel batting: {len(current_ids)} players")

    all_predictions = []
    stats = list(BATTER_MARCEL_RATE_STATS)

    # Reliability regression setup (matches LSTM pipeline behavior)
    enable_reliability_regression = getattr(
        BatterConfig, 'ENABLE_RELIABILITY_REGRESSION_PREDICTION', False
    )
    if enable_reliability_regression:
        from core.reliability import regress_player_sequence, get_era_for_features
        batter_era = get_era_for_features(stats)
        logger.info(f"Marcel batting: reliability regression ENABLED (era={batter_era})")
    else:
        regress_player_sequence = None  # unused
        batter_era = None

    for player_id in current_ids:
        player_hist = raw_df[
            (raw_df['IDfg'] == player_id) &
            (raw_df['Season'] <= cutoff_year) &
            (raw_df['PA'] >= 50)
        ].copy()

        if player_hist.empty:
            continue

        # Player name
        name_match = player_names[player_names['IDfg'] == player_id]
        player_name = name_match['Name'].iloc[0] if not name_match.empty else f"Unknown ({player_id})"

        player_hist = player_hist.sort_values('Season')
        last_age = player_hist['Age'].iloc[-1]
        last_season = player_hist['Season'].iloc[-1]

        # x-stat substitution: use xwOBA→wOBA, xBA→AVG, xSLG→SLG in the input
        if use_xstats:
            hist_for_marcel = player_hist.copy()
            if 'xwOBA' in hist_for_marcel.columns:
                mask = hist_for_marcel['xwOBA'].notna()
                hist_for_marcel.loc[mask, 'wOBA'] = hist_for_marcel.loc[mask, 'xwOBA']
            if 'xBA' in hist_for_marcel.columns:
                mask = hist_for_marcel['xBA'].notna()
                hist_for_marcel.loc[mask, 'AVG'] = hist_for_marcel.loc[mask, 'xBA']
            if 'xSLG' in hist_for_marcel.columns:
                mask = hist_for_marcel['xSLG'].notna()
                hist_for_marcel.loc[mask, 'SLG'] = hist_for_marcel.loc[mask, 'xSLG']
        else:
            hist_for_marcel = player_hist

        # Reliability regression: regress each season toward career/league prior
        # before the Marcel weighted average (matches LSTM pipeline behavior)
        if enable_reliability_regression:
            hist_for_marcel = regress_player_sequence(
                hist_for_marcel, stats,
                model_type='batter', era=batter_era,
                league_priors=BATTER_LEAGUE_AVG,
            )

        # Build recent seasons list (newest first)
        recent = hist_for_marcel.tail(len(SEASON_WEIGHTS)).iloc[::-1]
        seasons_data = []
        for _, row in recent.iterrows():
            s = {'PA': row.get('PA', 650)}
            for stat in stats:
                s[stat] = row.get(stat, np.nan)
            seasons_data.append(s)

        # Marcel weighted average with regression toward league average
        base = _compute_marcel_weighted_average_toward_league(
            seasons_data, stats, 'PA', BATTER_REGRESSION_PA, BATTER_LEAGUE_AVG
        )

        # xwOBA-wOBA gap adjustment: for players who consistently over/under-perform
        # their xwOBA (e.g. pull-heavy hitters like Ramirez, Raleigh), add back a
        # regressed portion of their historical gap so the projection isn't purely
        # anchored to expected stats.
        if enable_gap_adj:
            gap_pairs = [('wOBA', 'xwOBA'), ('AVG', 'xBA'), ('SLG', 'xSLG')]
            for real_col, x_col in gap_pairs:
                if x_col not in player_hist.columns:
                    continue
                recent_gap = player_hist.tail(len(SEASON_WEIGHTS)).copy()
                mask = recent_gap[x_col].notna() & recent_gap[real_col].notna()
                recent_gap = recent_gap[mask]
                if len(recent_gap) < gap_min_seasons:
                    continue
                # PA-weighted gap (real - expected)
                pa_vals = recent_gap['PA'].values
                gaps = (recent_gap[real_col] - recent_gap[x_col]).values
                total_pa = pa_vals.sum()
                if total_pa <= 0:
                    continue
                raw_gap = np.average(gaps, weights=pa_vals)
                # Regress toward 0: gap * total_pa / (total_pa + regression_pa)
                regressed_gap = raw_gap * total_pa / (total_pa + gap_regress_pa)
                adjustment = regressed_gap * gap_skill_fraction
                base[real_col] += adjustment

        # Build career counting profile from x-stat substituted data so that
        # the counting derivation ratio (projected_wOBA / career_wOBA) uses the
        # same x-stat basis as the Marcel rate-stat projection.
        career_profile = _build_batter_career_profile(hist_for_marcel)

        # Project forward with aging
        for year_offset in range(1, future_years + 1):
            proj_year = cutoff_year + year_offset
            proj_age = last_age + (cutoff_year - last_season) + year_offset

            # Apply cumulative aging
            projected = {}
            for stat in stats:
                stat_curves = batting_curves.get(stat, {})
                aging_total = 0.0
                for y in range(1, year_offset + 1):
                    age_at_y = last_age + (cutoff_year - last_season) + y
                    aging_total += _get_smoothed_aging_delta(stat_curves, int(age_at_y))

                val = base[stat] + aging_total

                # Apply physical bounds
                if stat in BATTER_BOUNDS:
                    lo, hi = BATTER_BOUNDS[stat]
                    val = np.clip(val, lo, hi)

                projected[stat] = val

            # Derive counting stats from projected wOBA × career profile
            if career_profile is not None:
                counting = _derive_counting_from_woba(
                    projected['wOBA'], career_profile, pa_full
                )
            else:
                counting = {s: 0.0 for s in BATTER_COUNTING_STATS}

            # Reconstruct OBP/SLG/wOBA from counting components for consistency.
            # All three must come from the same counting stats so that the
            # displayed slash line is internally coherent.  The x-stat signal
            # is preserved because counting stats were derived from the
            # x-stat-based projected wOBA via career-profile ratios.
            reconstructed = _reconstruct_rate_stats(
                projected['BB%'], projected['AVG'], counting
            )
            if reconstructed:
                projected['OBP'] = reconstructed['OBP']
                projected['SLG'] = reconstructed['SLG']
                projected['wOBA'] = reconstructed['wOBA']

            # Re-apply bounds after reconstruction
            for stat in stats:
                if stat in BATTER_BOUNDS:
                    lo, hi = BATTER_BOUNDS[stat]
                    projected[stat] = np.clip(projected[stat], lo, hi)

            # Build output row matching LSTM format
            row = {
                'Name': player_name,
                'IDfg': player_id,
                'Year': proj_year,
                'Age': proj_age,
                'PA': 650,
            }
            # Rate stats
            for stat in stats:
                row[stat] = projected[stat]
            # Counting stats
            for stat in BATTER_COUNTING_STATS:
                row[stat] = counting.get(stat, 0.0)

            all_predictions.append(row)

    if not all_predictions:
        logger.warning("No Marcel batter predictions generated")
        return None

    result_df = pd.DataFrame(all_predictions)

    # Ensure column order matches LSTM output
    output_cols = ['Name', 'IDfg', 'Year', 'Age', 'PA'] + stats + BATTER_COUNTING_STATS
    for col in output_cols:
        if col not in result_df.columns:
            result_df[col] = 0.0
    result_df = result_df[output_cols]
    result_df = result_df.sort_values(['Name', 'Year'])

    logger.info(f"Marcel batting: generated {len(result_df)} projections for "
                f"{result_df['Name'].nunique()} batters")
    return result_df


# ============================================================================
# MARCEL PITCHER PROJECTIONS
# ============================================================================

# Component rate stats that Marcel directly projects for pitchers.
# FIP, ERA, SIERA, K/9, BB/9, HR/9 are all reconstructed from these.
PITCHER_MARCEL_RATE_STATS = ['K%', 'BB%', 'HBP%', 'BABIP', 'HR/FB', 'GB%', 'FB%', 'LD%']

# Regression toward league average for pitchers (in IP-equivalents).
PITCHER_REGRESSION_IP = {
    'K%':    50,     # ~16 IP to stabilize (very sticky)
    'BB%':   80,     # ~40 IP to stabilize
    'HBP%': 150,     # ~230 IP to stabilize (noisy)
    'BABIP': 200,    # ~190 IP to stabilize (very noisy)
    'HR/FB': 200,    # ~170 TBF → ~40 IP to stabilize (noisy, low ICC)
    'GB%':   50,     # ~16 IP to stabilize (very sticky)
    'FB%':   50,     # ~16 IP to stabilize (very sticky)
    'LD%':  150,     # ~60 IP to stabilize
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

# FIP constant and BF/IP fallback (synced with pitcher_prediction.py)
_FIP_CONSTANT = 3.15
_BF_PER_IP_FALLBACK = 4.25

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
    using Marcel method:
      1. Weighted average of last 3 seasons (5/4/3 × IP)
      2. Regress toward league-average
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

    # Roster recovery
    if roster_ids is not None:
        missing = roster_ids - qualifying_ids
        if missing:
            for pid in missing:
                if pid in player_roles:
                    continue
                phist = raw_df[
                    (raw_df['IDfg'] == pid) &
                    (raw_df['Season'] <= cutoff_year) &
                    (raw_df['IP'] >= 10)
                ]
                if phist.empty:
                    continue
                last = phist.sort_values('Season').iloc[-1]
                if last['G'] > 0:
                    gs_rate = last['GS'] / last['G']
                    player_roles[pid] = 'SP' if gs_rate >= 0.8 else 'RP'
                    qualifying_ids.add(pid)
            recovered = len(qualifying_ids) - len(player_roles)
            if recovered > 0:
                logger.info(f"Marcel pitching: recovered {recovered} roster pitchers from history")

    logger.info(f"Marcel pitching: {len(player_roles)} pitchers "
                f"({sum(1 for r in player_roles.values() if r == 'SP')} SP, "
                f"{sum(1 for r in player_roles.values() if r == 'RP')} RP)")

    all_predictions = []

    # Reliability regression setup (matches LSTM pipeline behavior)
    # Check both SP and RP configs — enable if either has it on
    enable_reliability_regression = (
        getattr(PitcherSPConfig, 'ENABLE_RELIABILITY_REGRESSION_PREDICTION', False) or
        getattr(PitcherRPConfig, 'ENABLE_RELIABILITY_REGRESSION_PREDICTION', False)
    )
    if enable_reliability_regression:
        from core.reliability import regress_player_sequence, get_era_for_features
        pitcher_era = get_era_for_features(stats)
        logger.info(f"Marcel pitching: reliability regression ENABLED (era={pitcher_era})")
    else:
        regress_player_sequence = None  # unused
        pitcher_era = None

    for player_id, role in player_roles.items():
        min_ip_season = 20 if role == 'SP' else 10

        player_hist = raw_df[
            (raw_df['IDfg'] == player_id) &
            (raw_df['Season'] <= cutoff_year) &
            (raw_df['IP'] >= min_ip_season)
        ].copy()

        if player_hist.empty:
            continue

        # Player name
        name_match = player_names[player_names['IDfg'] == player_id]
        player_name = name_match['Name'].iloc[0] if not name_match.empty else f"Unknown ({player_id})"

        player_hist = player_hist.sort_values('Season')
        last_age = player_hist['Age'].iloc[-1]
        last_season = player_hist['Season'].iloc[-1]

        # Reliability regression: regress each season toward career/league prior
        # before the Marcel weighted average (matches LSTM pipeline behavior)
        if enable_reliability_regression:
            hist_for_marcel = regress_player_sequence(
                player_hist, stats,
                model_type='pitcher', era=pitcher_era,
                league_priors=PITCHER_LEAGUE_AVG,
            )
        else:
            hist_for_marcel = player_hist

        # Build recent seasons list (newest first)
        recent = hist_for_marcel.tail(len(SEASON_WEIGHTS)).iloc[::-1]
        seasons_data = []
        for _, row in recent.iterrows():
            s = {'IP': row.get('IP', 100)}
            for stat in stats:
                s[stat] = row.get(stat, np.nan)
            seasons_data.append(s)

        # Marcel weighted average with regression toward league average
        base = _compute_marcel_weighted_average_toward_league(
            seasons_data, stats, 'IP', PITCHER_REGRESSION_IP, PITCHER_LEAGUE_AVG
        )

        # Career ERA-FIP gap
        era_fip_gap = _compute_career_era_fip_gap_marcel(player_hist)

        # Project forward with aging
        for year_offset in range(1, future_years + 1):
            proj_year = cutoff_year + year_offset
            proj_age = last_age + (cutoff_year - last_season) + year_offset

            # Apply cumulative aging to each stat
            projected = {}
            for stat in stats:
                stat_curves = pitching_curves.get(stat, {})
                aging_total = 0.0
                for y in range(1, year_offset + 1):
                    age_at_y = last_age + (cutoff_year - last_season) + y
                    aging_total += _get_smoothed_aging_delta(stat_curves, int(age_at_y))

                val = base[stat] + aging_total

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

            # Derive per-9 rates
            k_per_9 = projected['K%'] * bf_per_ip * 9.0
            bb_per_9 = projected['BB%'] * bf_per_ip * 9.0
            hr_per_9 = hr_pct * bf_per_ip * 9.0

            # Build output row matching LSTM format
            row = {
                'Name': player_name,
                'Year': proj_year,
                'Age': proj_age,
                'Role': role,
                'IDfg': player_id,
                # Component rates (Marcel-projected)
                'K%': projected['K%'],
                'BB%': projected['BB%'],
                'HBP%': projected['HBP%'],
                'BABIP': projected['BABIP'],
                # Reconstructed metrics
                'FIP': fip,
                'ERA': era,
                # Batted ball profile
                'GB%': projected['GB%'],
                'FB%': projected['FB%'],
                'HR/FB': projected['HR/FB'],
                'SIERA': siera,
                'LD%': projected['LD%'],
                # Per-9 derived rates
                'K/9': k_per_9,
                'BB/9': bb_per_9,
                'HR/9': hr_per_9,
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
