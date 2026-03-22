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
        total_weight += reg_vol * SEASON_WEIGHTS[0]  # Use highest recency weight for regression

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
                        aging_total += _get_smoothed_aging_delta(stat_curves, int(age_at_y))
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
