"""
Pitcher Post-Processing Pipeline
=================================

Applies post-prediction adjustments to LSTM pitcher predictions before WAR
calculation. This handles adjustments that do NOT need to feed back into the
autoregressive loop.

Steps handled HERE (value determination side):
    1. Output regression: shrink component rates toward league average (career TBF-dependent)
    2. Reconstruct FIP/ERA/SIERA from regressed components (avoids double-regression)
    3. Aging constraints: cap unrealistic year-over-year improvements

Steps that remain in the autoregressive loop (core/prediction.py):
    - HR% decomposition (HR% = HR/FB * FB% * BIP_rate)
    - FIP reconstruction from K%, BB%, HBP%, HR%
    - SIERA reconstruction from K%, BB%, GB%
    - ERA derivation (FIP + career gap, or SIERA + career gap)

The loop-side steps MUST stay there because the model receives FIP, ERA, and HR%
as input features. If we fed back inconsistent values, the next prediction step
would see corrupted inputs.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pitcher_prediction import (
    _apply_output_regression,
    _apply_pitcher_aging_constraints,
    BF_PER_IP_RATIO,
    reconstruct_fip_from_components,
    reconstruct_siera_from_components,
)
from core.reliability import (
    compute_league_priors_from_df,
    get_era_for_features,
)

logger = logging.getLogger(__name__)

# Rate stat features expected in the predictions CSV (excluding metadata)
PITCHER_RATE_FEATURES = [
    'K%', 'BB%', 'HR%', 'HBP%', 'FIP', 'ERA', 'GB%', 'FB%', 'HR/FB', 'SIERA'
]


def _get_config(role: str):
    """Load the appropriate pitcher config class."""
    if role == 'SP':
        from configs.pitcher_sp_config import PitcherSPConfig
        return PitcherSPConfig
    else:
        from configs.pitcher_rp_config import PitcherRPConfig
        return PitcherRPConfig


def post_process_pitcher_predictions(
    pitcher_df: pd.DataFrame,
    role: str,
    pitching_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply value-determination-side post-processing to pitcher predictions.

    Handles output regression and aging constraints only. FIP/ERA/HR%
    reconstruction stays in the autoregressive loop (core/prediction.py)
    because those values feed back into the model.

    Args:
        pitcher_df: Pitcher predictions DataFrame (from predictions CSV).
                    Must have columns: Name, Year, Age, Role, IDfg, plus rate stats.
        role: 'SP' or 'RP'
        pitching_history: Historical pitching data for career stats. If None,
                         loaded automatically via data_loader.

    Returns:
        DataFrame with post-processed predictions (copy -- original unchanged).
    """
    if pitcher_df.empty:
        return pitcher_df.copy()

    config = _get_config(role)
    result = pitcher_df.copy()

    # Read config toggles (only the ones handled here)
    enable_output_reg = getattr(config, 'ENABLE_OUTPUT_REGRESSION', False)
    enable_aging = getattr(config, 'ENABLE_PITCHER_AGING_CONSTRAINTS', False)

    if not enable_output_reg and not enable_aging:
        logger.info(f"{role} post-processing: nothing enabled, skipping")
        return result

    # Determine which features are present in the predictions
    available_features = [f for f in PITCHER_RATE_FEATURES if f in result.columns]
    if not available_features:
        logger.warning(f"No rate features found in pitcher predictions for {role}")
        return result

    logger.info(
        f"{role} post-processing: output_reg={'ON' if enable_output_reg else 'OFF'}, "
        f"aging={'ON' if enable_aging else 'OFF'}"
    )

    # Load historical data for output regression
    if enable_output_reg and pitching_history is None:
        from value_determination.data_loader import load_historical_data
        _, pitching_history = load_historical_data()

    # Compute league priors for output regression
    league_priors = None
    era = 'pitchfx'
    if enable_output_reg and pitching_history is not None:
        feature_list_for_era = ['Age'] + available_features
        era = get_era_for_features(feature_list_for_era)
        cutoff_year = pitching_history['Season'].max()
        league_priors = compute_league_priors_from_df(
            pitching_history, feature_list_for_era,
            model_type='pitcher', season=cutoff_year, window=3
        )
        logger.info(f"Computed league priors for {role} output regression ({len(league_priors)} features)")

    # Process each pitcher individually
    player_ids = result['IDfg'].unique()
    tbf_per_season = 700.0 if role == 'SP' else 250.0

    for player_id in player_ids:
        player_mask = result['IDfg'] == player_id
        player_rows = result.loc[player_mask].sort_values('Year')

        if player_rows.empty:
            continue

        player_name = player_rows.iloc[0]['Name']

        # Get career TBF from historical data
        career_tbf = 0.0
        if enable_output_reg and pitching_history is not None:
            player_hist = pitching_history[pitching_history['IDfg'] == player_id]
            if not player_hist.empty:
                if 'TBF' in player_hist.columns and player_hist['TBF'].notna().any():
                    career_tbf = float(player_hist['TBF'].sum())
                elif 'IP' in player_hist.columns:
                    career_tbf = float(player_hist['IP'].sum()) * BF_PER_IP_RATIO

        # Capture the ERA-FIP gap from the autoregressive loop output
        # (this reflects the pitcher's regressed career ERA-FIP differential)
        era_fip_gap = None
        if 'ERA' in available_features and 'FIP' in available_features:
            era_idx = available_features.index('ERA')
            fip_idx = available_features.index('FIP')

        # Process each prediction year in order
        previous_values = None

        for idx in player_rows.index:
            row = result.loc[idx]
            pred_array = np.array([row[f] for f in available_features], dtype=np.float64)
            age = row['Age']

            # Capture ERA-FIP gap BEFORE regression (stable per-pitcher offset
            # from the autoregressive loop's ERA-FIP adjustment)
            if 'ERA' in available_features and 'FIP' in available_features:
                era_fip_gap = pred_array[era_idx] - pred_array[fip_idx]

            # Output regression (regresses component rates; skips FIP/ERA/SIERA)
            if enable_output_reg and league_priors:
                pred_array = _apply_output_regression(
                    pred_array, available_features,
                    career_tbf, league_priors,
                    era=era, player_name=player_name
                )
                career_tbf += tbf_per_season

                # Reconstruct FIP from regressed components
                if 'FIP' in available_features:
                    k_pct = pred_array[available_features.index('K%')] if 'K%' in available_features else 0.0
                    bb_pct = pred_array[available_features.index('BB%')] if 'BB%' in available_features else 0.0
                    hbp_pct = pred_array[available_features.index('HBP%')] if 'HBP%' in available_features else 0.0
                    hr_pct = pred_array[available_features.index('HR%')] if 'HR%' in available_features else 0.0
                    new_fip = reconstruct_fip_from_components(k_pct, bb_pct, hbp_pct, hr_pct)
                    pred_array[fip_idx] = new_fip

                    # Derive ERA from reconstructed FIP + career gap
                    if 'ERA' in available_features and era_fip_gap is not None:
                        pred_array[era_idx] = np.clip(new_fip + era_fip_gap, 0.5, 10.0)

                # Reconstruct SIERA from regressed components
                if 'SIERA' in available_features:
                    siera_idx = available_features.index('SIERA')
                    k_pct = pred_array[available_features.index('K%')] if 'K%' in available_features else 0.0
                    bb_pct = pred_array[available_features.index('BB%')] if 'BB%' in available_features else 0.0
                    gb_pct = pred_array[available_features.index('GB%')] if 'GB%' in available_features else 0.0
                    pred_array[siera_idx] = reconstruct_siera_from_components(k_pct, bb_pct, gb_pct)

            # Aging constraints (skip first year -- no previous to compare)
            if enable_aging and previous_values is not None:
                pred_array = _apply_pitcher_aging_constraints(
                    pred_array, previous_values,
                    available_features, age, player_name
                )

            # Write back to DataFrame
            for i, feat in enumerate(available_features):
                result.at[idx, feat] = pred_array[i]

            previous_values = pred_array.copy()

    logger.info(f"Post-processed {len(player_ids)} {role} pitchers ({len(result)} total rows)")
    return result
