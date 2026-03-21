"""
Batter-Specific Prediction Functions
=====================================

Analogous to core/pitcher_prediction.py.  Contains in-loop reconstruction of
counting stats (from wOBA × career profile) and rate stats (wOBA/OBP/SLG
from counting-stat components), so that the LSTM's autoregressive feedback
contains self-consistent values at each projection step.

Previously, this reconciliation happened post-hoc in the value determination
pipeline (calculate_war.py / counting_recalibration.py).  Moving it into the
prediction loop means each year's prediction "sees" consistent stats from
the prior year, analogous to how pitcher FIP/ERA reconstruction feeds back.

Functions:
    - build_career_profiles():           Build per-player counting-stat baselines
    - _apply_physical_bounds():          Clip predictions to sane ranges
    - _apply_counting_derivation():      In-loop counting stat derivation from wOBA
    - _apply_rate_reconstruction():      In-loop wOBA/OBP/SLG reconstruction
    - predict_future_stats_batter():     Single-player projection with reconstruction

All functions are re-exported from core/prediction.py for backward
compatibility, so existing imports continue to work.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# 2025 FanGraphs wOBA linear weights (must stay in sync with
# value_determination/config.py → Config.WAR.WOBA_WEIGHTS)
WOBA_WEIGHTS = {
    'wBB': 0.691,
    'wHBP': 0.722,
    'w1B': 0.882,
    'w2B': 1.252,
    'w3B': 1.584,
    'wHR': 2.037,
}

# Counting stats eligible for in-loop derivation (per-150-game rates)
COUNTING_STATS = ['HR', '2B', '3B', 'RBI', 'R', 'HBP']

# Physical bounds to prevent autoregressive divergence.
# Applied every step so extreme predictions in one year don't snowball.
BATTER_PHYSICAL_BOUNDS = {
    'BB%':  (0.02,  0.25),
    'K%':   (0.05,  0.40),
    'AVG':  (0.150, 0.380),
    'OBP':  (0.200, 0.500),
    'SLG':  (0.200, 0.850),
    'wOBA': (0.200, 0.500),
    'HR':   (0, 60),       # per 150 games
    '2B':   (0, 60),
    '3B':   (0, 25),
    'RBI':  (0, 170),
    'R':    (0, 160),
    'HBP':  (0, 30),
}


# =============================================================================
# CAREER PROFILE CONSTRUCTION
# =============================================================================

def build_career_profiles(
    raw_df: pd.DataFrame,
    n_recent: int = 3,
    min_pa: int = 50,
) -> Dict[int, Dict]:
    """
    Build PA-weighted career counting profiles for in-loop reconstruction.

    Expects ``raw_df`` where counting stats (HR, 2B, 3B, RBI, R, HBP) are
    already in per-150-game rates (after ``calculate_rate_stats``).

    Args:
        raw_df:    Historical batting DataFrame with per-150 counting stats.
        n_recent:  Number of most-recent qualifying seasons to include.
        min_pa:    Minimum PA for a season to qualify.

    Returns:
        Dict mapping ``IDfg`` → profile dict with keys:
            ``base_woba``:   PA-weighted recent wOBA
            ``base_counts``: {stat: per-150 rate} for each counting stat
            ``career_pa``:   Total career PA across all qualifying seasons
    """
    profiles: Dict[int, Dict] = {}

    required_cols = {'IDfg', 'Season', 'PA', 'wOBA'}
    if not required_cols.issubset(raw_df.columns):
        missing = required_cols - set(raw_df.columns)
        logger.warning(f"Missing columns for career profiles: {missing}")
        return profiles

    available_counting = [s for s in COUNTING_STATS if s in raw_df.columns]
    df = raw_df[raw_df['PA'] >= min_pa].copy()

    for player_id, group in df.groupby('IDfg'):
        group = group.sort_values('Season')
        recent = group.tail(n_recent)

        if len(recent) == 0:
            continue

        pa = recent['PA'].values.astype(float)
        total_pa = pa.sum()
        if total_pa == 0:
            continue
        weights = pa / total_pa

        # Baseline wOBA
        woba_vals = recent['wOBA'].values
        if pd.isna(woba_vals).all():
            continue
        woba_vals = np.nan_to_num(woba_vals, nan=0.0)
        base_woba = float(np.average(woba_vals, weights=weights))
        if base_woba < 0.15:
            continue

        # Per-150 counting stats — already per-150 in raw_df, just average
        base_counts: Dict[str, float] = {}
        for stat in available_counting:
            vals = np.nan_to_num(recent[stat].values.astype(float), nan=0.0)
            base_counts[stat] = float(np.average(vals, weights=weights))

        career_pa = float(group['PA'].sum())

        profiles[int(player_id)] = {
            'base_woba': base_woba,
            'base_counts': base_counts,
            'career_pa': career_pa,
        }

    logger.info(f"Built career counting profiles for {len(profiles)} batters")
    return profiles


# =============================================================================
# IN-LOOP RECONSTRUCTION FUNCTIONS
# =============================================================================

def _apply_physical_bounds(
    prediction: np.ndarray,
    input_features: List[str],
) -> np.ndarray:
    """Clip prediction values to physically plausible ranges."""
    for stat, (lo, hi) in BATTER_PHYSICAL_BOUNDS.items():
        if stat in input_features:
            idx = input_features.index(stat)
            prediction[idx] = np.clip(prediction[idx], lo, hi)
    return prediction


def _apply_counting_derivation(
    prediction: np.ndarray,
    input_features: List[str],
    career_profile: Dict,
    pa_full: float,
    player_name: str = '',
) -> np.ndarray:
    """
    Derive counting stats from wOBA × career profile (in-loop).

    Uses the model's predicted wOBA as a quality scalar.  Each counting
    stat is derived from the player's career per-150 profile scaled by
    (predicted_wOBA / career_wOBA), blended with the model's direct
    prediction based on career PA.

    Modifies the prediction array in-place and returns it.
    """
    if 'wOBA' not in input_features:
        return prediction

    base_woba = career_profile['base_woba']
    if base_woba < 0.15:
        return prediction

    woba_idx = input_features.index('wOBA')
    pred_woba = prediction[woba_idx]
    if np.isnan(pred_woba):
        return prediction

    ratio = np.clip(pred_woba / base_woba, 0.50, 1.50)
    blend = min(career_profile['career_pa'] / pa_full, 1.0)

    for stat in COUNTING_STATS:
        if stat not in input_features or stat not in career_profile['base_counts']:
            continue
        idx = input_features.index(stat)
        career_rate = career_profile['base_counts'][stat]
        model_pred = prediction[idx]
        derived = career_rate * ratio
        prediction[idx] = max(0.0, blend * derived + (1.0 - blend) * model_pred)

    return prediction


def _apply_rate_reconstruction(
    prediction: np.ndarray,
    input_features: List[str],
    player_name: str = '',
) -> np.ndarray:
    """
    Reconstruct wOBA, OBP, and SLG from counting stat components (in-loop).

    After counting stats have been derived (or left as model output), this
    ensures rate stats are mathematically consistent with the components.
    Uses standard wOBA linear weights and OBP/SLG formulas.

    Assumes per-150-game counting stats and PA ≈ 650.

    Modifies the prediction array in-place and returns it.
    """
    # Need AVG and BB% at minimum
    if 'AVG' not in input_features or 'BB%' not in input_features:
        return prediction

    pa = 650.0

    bb_pct = prediction[input_features.index('BB%')]
    avg = prediction[input_features.index('AVG')]

    bb = bb_pct * pa
    hbp = prediction[input_features.index('HBP')] if 'HBP' in input_features else pa * 0.01
    sf = pa * 0.007
    ab = pa - bb - hbp - sf

    if ab <= 0:
        return prediction

    h = avg * ab
    hr = prediction[input_features.index('HR')] if 'HR' in input_features else 0.0
    doubles = prediction[input_features.index('2B')] if '2B' in input_features else 0.0
    triples = prediction[input_features.index('3B')] if '3B' in input_features else 0.0
    singles = max(0.0, h - doubles - triples - hr)

    # Reconstruct OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    if 'OBP' in input_features:
        obp_den = ab + bb + hbp + sf
        if obp_den > 0:
            prediction[input_features.index('OBP')] = np.clip(
                (h + bb + hbp) / obp_den, 0, 1
            )

    # Reconstruct wOBA from linear weights
    if 'wOBA' in input_features:
        w = WOBA_WEIGHTS
        woba_num = (w['wBB'] * bb + w['wHBP'] * hbp + w['w1B'] * singles +
                    w['w2B'] * doubles + w['w3B'] * triples + w['wHR'] * hr)
        if pa > 0:
            prediction[input_features.index('wOBA')] = np.clip(woba_num / pa, 0, 1)

    # Reconstruct SLG = (1B + 2×2B + 3×3B + 4×HR) / AB
    if 'SLG' in input_features:
        slg_num = singles + 2 * doubles + 3 * triples + 4 * hr
        if ab > 0:
            prediction[input_features.index('SLG')] = np.clip(slg_num / ab, 0, 4)

    return prediction


# =============================================================================
# BATTER PREDICTION FUNCTION
# =============================================================================

def predict_future_stats_batter(
    player_id: str,
    input_features: List[str],
    model,
    scaler,
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    seq_length: int = 3,
    future_years: int = 16,
    cutoff_year: Optional[int] = None,
    league_priors: Optional[Dict[str, float]] = None,
    career_profile: Optional[Dict] = None,
) -> List[Dict]:
    """
    Predict future stats for a batter with in-loop counting stat reconstruction.

    Analogous to ``predict_future_stats_pitcher``.  Handles:

    1. Sequence building (x-stat substitution, reliability regression, park factors)
    2. Custom autoregressive loop with:
       - Physical bounds to prevent divergence
       - Counting stat derivation from wOBA × career profile (Mode A)
       - Rate stat (wOBA/OBP/SLG) reconstruction from components
       - Consistent unscaled feedback to next year's prediction

    Args:
        player_id:      FanGraphs player ID
        input_features: List of feature names the model was trained on
        model:          Trained LSTM model
        scaler:         Fitted scaler for features
        raw_df:         Historical batting data (pre-filtered to this player)
        player_names:   DataFrame mapping IDfg to Name
        seq_length:     Number of historical seasons for input window
        future_years:   Number of years to project
        cutoff_year:    Last year of actual data (projections start cutoff_year + 1)
        league_priors:  Pre-computed league averages per feature (for regression)
        career_profile: Career counting profile for this player
                       (from ``build_career_profiles``).  Required for Mode A
                       counting stat derivation; if None only rate reconstruction
                       is applied.

    Returns:
        List of prediction dictionaries (one per projected year)
    """
    from .prediction import _is_park_factor_enabled

    # ── Load config ──────────────────────────────────────────────────────
    components_from_woba = False
    woba_from_components = False
    pa_full = 1500.0

    try:
        from configs.batter_config import BatterConfig
        components_from_woba = getattr(BatterConfig, 'CALCULATE_COMPONENTS_FROM_WOBA', False)
        woba_from_components = getattr(BatterConfig, 'CALCULATE_WOBA_FROM_COMPONENTS', False)
        pa_full = getattr(BatterConfig, 'COMPONENTS_FROM_WOBA_PA_WEIGHT', 1500.0)
    except (ImportError, AttributeError):
        pass

    enable_counting_derivation = components_from_woba and career_profile is not None
    enable_rate_reconstruction = enable_counting_derivation or woba_from_components

    # ── Get player data ──────────────────────────────────────────────────
    player_data = raw_df[raw_df['IDfg'] == player_id].copy()
    if len(player_data) == 0:
        return []

    player_data = player_data.sort_values('Season')

    # Get player name
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        logger.warning(f"Player name not found for ID {player_id}")
        return []

    # Determine latest season and age
    if cutoff_year is not None:
        latest_season = cutoff_year
        actual_latest_season = player_data['Season'].max()
        latest_age = player_data[
            player_data['Season'] == actual_latest_season
        ]['Age'].iloc[-1]
        if cutoff_year > actual_latest_season:
            latest_age += (cutoff_year - actual_latest_season)
    else:
        latest_season = player_data['Season'].max()
        latest_age = player_data['Age'].iloc[-1]

    # ── Reliability regression check ─────────────────────────────────────
    use_regression = False
    try:
        from core.data_processing import _is_reliability_regression_enabled
        use_regression = _is_reliability_regression_enabled('batter', context='prediction')
    except ImportError:
        pass

    # ── x-stat substitution ──────────────────────────────────────────────
    # Save originals for counting-stat adjustment
    _orig_woba = player_data['wOBA'].values.copy() if 'wOBA' in player_data.columns else None
    _orig_slg = player_data['SLG'].values.copy() if 'SLG' in player_data.columns else None
    _orig_avg = player_data['AVG'].values.copy() if 'AVG' in player_data.columns else None

    try:
        from configs.batter_config import BatterConfig as _BC

        if (_BC.USE_XWOBA_FOR_PREDICTIONS and
                'wOBA' in input_features and 'xwOBA' in player_data.columns):
            mask = player_data['xwOBA'].notna()
            player_data.loc[mask, 'wOBA'] = player_data.loc[mask, 'xwOBA']

        elif (getattr(_BC, 'USE_XWOBA_BLEND_FOR_PREDICTIONS', False) and
                'wOBA' in input_features and 'xwOBA' in player_data.columns):
            mask = player_data['xwOBA'].notna()
            player_data.loc[mask, 'wOBA'] = (
                player_data.loc[mask, 'wOBA'] + player_data.loc[mask, 'xwOBA']
            ) / 2

        if (_BC.USE_XBA_FOR_PREDICTIONS and
                'AVG' in input_features and 'xBA' in player_data.columns):
            mask = player_data['xBA'].notna()
            player_data.loc[mask, 'AVG'] = player_data.loc[mask, 'xBA']

        if (_BC.USE_XSLG_FOR_PREDICTIONS and
                'SLG' in input_features and 'xSLG' in player_data.columns):
            mask = player_data['xSLG'].notna()
            player_data.loc[mask, 'SLG'] = player_data.loc[mask, 'xSLG']
    except (ImportError, AttributeError):
        pass

    # Counting-stat adjustment for x-stat consistency
    try:
        from configs.batter_config import BatterConfig as _BC
        if getattr(_BC, 'ADJUST_COUNTING_STATS_TO_XSTATS', False):
            from .prediction import _adjust_counting_stats_for_xstats
            _adjust_counting_stats_for_xstats(
                player_data, _orig_woba, _orig_slg, _orig_avg, input_features
            )
    except (ImportError, AttributeError):
        pass

    # ── Build sequence ───────────────────────────────────────────────────
    num_seasons = len(player_data)

    if use_regression:
        from core.reliability import (
            regress_player_sequence,
            compute_regressed_career_mean,
            get_era_for_features,
        )
        era = get_era_for_features(input_features)
        player_data_regressed = regress_player_sequence(
            player_data, input_features, model_type='batter',
            era=era, league_priors=league_priors
        )
        career_mean = compute_regressed_career_mean(
            player_data, input_features, model_type='batter',
            era=era, league_priors=league_priors
        )

        recent_data = player_data_regressed[input_features].iloc[-seq_length:].copy().reset_index(drop=True)
        n_actual = len(recent_data)

        # Pad with regressed career mean if not enough seasons
        if n_actual < seq_length:
            padding_vector = np.array(
                [career_mean.get(f, 0.0) for f in input_features], dtype=np.float64
            )
            n_pad = seq_length - n_actual
            padding_df = pd.DataFrame([padding_vector] * n_pad, columns=input_features)
            recent_data = pd.concat([recent_data, padding_df], ignore_index=True)

        # Fill remaining NaN with career mean
        if recent_data.isna().any().any():
            for feat in input_features:
                if recent_data[feat].isna().any():
                    recent_data[feat] = recent_data[feat].fillna(career_mean.get(feat, 0.0))

        sequence = recent_data.values.astype(np.float64)
        # For park factor: use regressed data to get team sequence
        _team_source = player_data_regressed
    else:
        # Non-regression path — pad by repeating last row
        if num_seasons < seq_length:
            recent_data = player_data[input_features].copy()
            while len(recent_data) < seq_length:
                recent_data = pd.concat([recent_data, recent_data.iloc[-1:]], ignore_index=True)
        else:
            recent_data = player_data[input_features].iloc[-seq_length:].copy().reset_index(drop=True)

        if recent_data.isna().any().any():
            return []

        sequence = recent_data.values.astype(np.float64)
        _team_source = player_data

    # ── Park factor neutralization ───────────────────────────────────────
    park_factor_enabled = _is_park_factor_enabled('batter')

    if park_factor_enabled and 'Team' in player_data.columns:
        from core.park_factors import get_park_factor, EXCLUDED_STATS
        adjustable_indices = [
            i for i, f in enumerate(input_features) if f not in EXCLUDED_STATS
        ]

        n_actual_teams = min(len(_team_source), seq_length)
        season_teams = (
            _team_source['Team'].iloc[-n_actual_teams:].tolist()
            if 'Team' in _team_source.columns else []
        )
        last_team = player_data['Team'].iloc[-1] if len(player_data) > 0 else None
        n_pad = seq_length - len(season_teams)
        all_teams = season_teams + [last_team] * n_pad

        for row_idx, team in enumerate(all_teams):
            pf = get_park_factor(team)
            if pf != 1.0:
                for col_idx in adjustable_indices:
                    sequence[row_idx, col_idx] = sequence[row_idx, col_idx] / pf

    # ── NaN / Inf safety ─────────────────────────────────────────────────
    if np.isnan(sequence).any() or np.isinf(sequence).any():
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        logger.warning(f"NaN/Inf cleaned from sequence for player {player_id}")

    # =====================================================================
    # AUTOREGRESSIVE PREDICTION LOOP
    # =====================================================================
    # current_sequence is kept in UNSCALED space (like pitcher_prediction.py).
    # Each iteration: scale → model → inverse scale → reconstruct → feedback.
    current_sequence = sequence.copy()
    n_features = len(input_features)
    device = next(model.parameters()).device
    predictions = []

    for year_offset in range(1, future_years + 1):
        year = latest_season + year_offset
        age = latest_age + year_offset

        # Scale and predict
        try:
            sequence_scaled = scaler.transform(current_sequence)
        except Exception as e:
            logger.error(f"Scaling error for player {player_id}: {e}")
            break

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
            lengths = torch.tensor([seq_length], dtype=torch.int64).to(device)
            output = model(seq_tensor, lengths)
            pred_numpy = output.cpu().numpy()[0]

        try:
            prediction_constrained = scaler.inverse_transform(
                pred_numpy.reshape(1, -1)
            )[0]
        except Exception as e:
            logger.error(f"Inverse transform error for player {player_id}, year {year}: {e}")
            break

        # ── Physical bounds ──────────────────────────────────────────
        prediction_constrained = _apply_physical_bounds(
            prediction_constrained, input_features
        )

        # ── In-loop counting stat derivation (Mode A) ────────────────
        if enable_counting_derivation:
            prediction_constrained = _apply_counting_derivation(
                prediction_constrained, input_features,
                career_profile, pa_full, player_name
            )

        # ── In-loop rate stat reconstruction ─────────────────────────
        # Ensures wOBA/OBP/SLG are mathematically consistent with the
        # (possibly derived) counting stats before feeding back.
        if enable_rate_reconstruction:
            prediction_constrained = _apply_rate_reconstruction(
                prediction_constrained, input_features, player_name
            )

        # ── Build prediction dict ────────────────────────────────────
        pred_dict = {
            'Name': player_name,
            'IDfg': player_id,
            'Year': year,
            'Age': age,
            'PA': 650,
        }

        for i, feature in enumerate(input_features):
            if feature == 'Age':
                pred_dict[feature] = age
            else:
                pred_dict[feature] = prediction_constrained[i]

        predictions.append(pred_dict)

        # ── Feed back into sequence (unscaled, like pitchers) ────────
        next_sequence = prediction_constrained.copy()
        age_index = input_features.index('Age')
        next_sequence[age_index] = age + 1  # Next year's age

        # Safety: replace NaN/Inf with last valid row
        if np.isnan(next_sequence).any() or np.isinf(next_sequence).any():
            logger.warning(
                f"NaN/Inf in prediction for {player_name} year {year} — "
                f"replacing with last valid values"
            )
            last_valid = current_sequence[-1].copy()
            bad = np.isnan(next_sequence) | np.isinf(next_sequence)
            next_sequence[bad] = last_valid[bad]
            next_sequence[age_index] = age + 1

        current_sequence = np.vstack([
            current_sequence[1:], next_sequence.reshape(1, -1)
        ])

    return predictions
