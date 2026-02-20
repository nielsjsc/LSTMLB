"""
Park Factor Adjustments
========================

Provides park factor neutralization and application for batter and pitcher stats.
Uses 5-year park factors to convert between park-influenced and park-neutral stats.

Key concepts:
- Neutralize: Divide stats by (PF/100) to remove park effects → true talent
- Apply: Multiply stats by (PF/100) to add park effects back → park-adjusted stats

Usage:
    from core.park_factors import neutralize_park_factors, apply_park_factors

    # Training: neutralize historical data before feeding to model
    df = neutralize_park_factors(df, input_features, team_column='Team')

    # WAR calculation: apply park factors to neutral predictions
    df = apply_park_factors(df, features, team_column='Team')
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 5-YEAR PARK FACTORS (2025)
# =============================================================================
# Source: FanGraphs 5-year park factors (Basic column)
# 100 = neutral park. >100 = hitter-friendly, <100 = pitcher-friendly.

PARK_FACTORS_5YR = {
    'LAA': 101,
    'BAL': 99,
    'BOS': 104,
    'CHW': 100,
    'CLE': 99,
    'DET': 100,
    'KC':  103,
    'MIN': 101,
    'NYY': 99,
    'ATH': 103,
    'SEA': 94,
    'TB':  101,
    'TEX': 99,
    'TOR': 99,
    'ARI': 101,
    'ATL': 100,
    'CHC': 98,
    'CIN': 105,
    'COL': 113,
    'MIA': 101,
    'HOU': 99,
    'LAD': 99,
    'MIL': 99,
    'WSH': 100,
    'NYM': 96,
    'PHI': 101,
    'PIT': 102,
    'STL': 98,
    'SD':  96,
    'SF':  97,
}

# Stats that should NEVER be park-adjusted
EXCLUDED_STATS = frozenset({'Age', 'wRC+'})


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_park_factor(team: str) -> float:
    """
    Get park factor as a multiplier (e.g., 1.13 for COL, 0.94 for SEA).
    
    Returns 1.0 (neutral) for unknown teams or free agents.
    """
    if team is None or (isinstance(team, float) and np.isnan(team)):
        return 1.0
    return PARK_FACTORS_5YR.get(str(team).upper().strip(), 100) / 100.0


def get_adjustable_features(input_features: List[str]) -> List[str]:
    """
    Get the subset of features that should receive park factor adjustments.
    
    Excludes 'Age' and 'wRC+' as specified.
    """
    return [f for f in input_features if f not in EXCLUDED_STATS]


# =============================================================================
# NEUTRALIZE (pre-model: remove park effects)
# =============================================================================

def neutralize_park_factors(
    df: pd.DataFrame,
    input_features: List[str],
    team_column: str = 'Team',
) -> pd.DataFrame:
    """
    Divide stats by park factor to get park-neutral values (true talent).
    
    A player in Coors (PF=113) hitting .300 AVG has a neutral AVG of .300/1.13 ≈ .265.
    
    Args:
        df: DataFrame with player stats (must contain team_column)
        input_features: List of model input features
        team_column: Name of the column containing team abbreviations
        
    Returns:
        DataFrame with park-neutralized stats (modifies in place for efficiency)
    """
    if team_column not in df.columns:
        logger.warning(f"Column '{team_column}' not found — skipping park factor neutralization")
        return df

    adjustable = get_adjustable_features(input_features)
    if not adjustable:
        return df

    # Build a Series of park factors aligned to the DataFrame index
    pf_series = df[team_column].map(
        lambda t: get_park_factor(t)
    )

    # Divide each adjustable feature by the park factor
    for feat in adjustable:
        if feat in df.columns:
            df[feat] = df[feat] / pf_series

    n_adjusted = sum(1 for f in adjustable if f in df.columns)
    logger.info(
        f"Park factor neutralization applied to {n_adjusted} features "
        f"(excluded: {EXCLUDED_STATS & set(input_features)})"
    )

    return df


# =============================================================================
# APPLY (post-model: add park effects back)
# =============================================================================

def apply_park_factors(
    df: pd.DataFrame,
    features: List[str],
    team_column: str = 'Team',
) -> pd.DataFrame:
    """
    Multiply stats by park factor to convert from park-neutral to park-adjusted.
    
    Used in WAR calculation after the model outputs park-neutral predictions
    and we know which team the player will be on.
    
    Args:
        df: DataFrame with park-neutral predicted stats
        features: List of features to adjust
        team_column: Name of the column containing team abbreviations
        
    Returns:
        DataFrame with park-adjusted stats
    """
    if team_column not in df.columns:
        logger.warning(f"Column '{team_column}' not found — skipping park factor application")
        return df

    adjustable = get_adjustable_features(features)
    if not adjustable:
        return df

    pf_series = df[team_column].map(
        lambda t: get_park_factor(t)
    )

    for feat in adjustable:
        if feat in df.columns:
            df[feat] = df[feat] * pf_series

    n_adjusted = sum(1 for f in adjustable if f in df.columns)
    logger.info(
        f"Park factor application (reverse) applied to {n_adjusted} features"
    )

    return df


def neutralize_array(
    sequence: np.ndarray,
    input_features: List[str],
    team: str,
) -> np.ndarray:
    """
    Neutralize a single player's sequence array (used in prediction pipeline).
    
    Args:
        sequence: Shape (seq_len, n_features) — unscaled feature values
        input_features: Feature names corresponding to columns of sequence
        team: Team abbreviation for park factor lookup
        
    Returns:
        Park-neutralized sequence array
    """
    pf = get_park_factor(team)
    if pf == 1.0:
        return sequence  # No adjustment needed

    adjustable_indices = [
        i for i, f in enumerate(input_features) if f not in EXCLUDED_STATS
    ]

    neutralized = sequence.copy()
    for idx in adjustable_indices:
        neutralized[:, idx] = neutralized[:, idx] / pf

    return neutralized
