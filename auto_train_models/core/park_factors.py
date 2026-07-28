"""
Park Factor Adjustments
========================

Provides park factor neutralization and application for batter and pitcher stats.

Two tables, two different jobs — do not mix them up:

- PARK_FACTORS_5YR   — runs-scale (FanGraphs Basic). Use for anything that's
  actually on a runs scale (e.g. FIP/ERA context, run-environment terms).
- WOBA_RESIDUAL_PF    — wOBA-scale residual (Savant wOBAcon/xwOBAcon). Use
  for anything touching wOBA/AVG/SLG/OBP-derived batting value. This is the
  leftover park effect AFTER xwOBA-based substitution already removed the
  quality-of-contact portion — using the runs-scale table here overstates
  the effect for extreme parks (Coors, T-Mobile).

Key concepts:
- Neutralize: Divide stats by (PF/100) to remove park effects → true talent
- Apply: Multiply stats by (PF/100) to add park effects back → park-adjusted stats

Usage:
    from core.park_factors import (
        neutralize_park_factors, apply_park_factors,
        get_park_factor, get_woba_residual_factor,
    )

    # Training: neutralize historical data before feeding to model
    # (runs-scale features)
    df = neutralize_park_factors(df, input_features, team_column='Team')

    # wOBA-scale features: pass the residual factor function explicitly
    df = neutralize_park_factors(
        df, ['wOBA', 'AVG', 'SLG'], team_column='Team',
        factor_fn=get_woba_residual_factor,
    )

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

# Alternate team codes used by other data sources (e.g. the historical
# batting CSV uses 3-letter codes for a few current teams where
# PARK_FACTORS_5YR uses the shorter/common FanGraphs form). Without this,
# these teams silently fall through to the "unknown team" neutral default
# in get_park_factor() below — no error, just quietly wrong (no park
# adjustment ever applied to them).
#
# Confirmed against the actual historical CSV (season ranges checked):
#   SFG/SDP/KCR/TBR/WSN — current teams' primary codes, need this.
#   TBD (1998-2007) — Devil Rays, same franchise/park as TBR, needs this.
#   CAL (1965-1996) / ANA (1997-2004) — same franchise/park as LAA
#     (Angel Stadium, same location since 1966), needs this.
#
# Genuinely different/relocated teams correctly default to neutral instead
# (no alias — different park, or no reliable modern PF for it):
#   WAS (1950-1971) — original Washington Senators; relocated to become the
#     Texas Rangers in 1972. NOT the same franchise as modern WSN (which is
#     the relocated Montreal Expos/MON).
#   KCA (1955-1967) — Kansas City Athletics; relocated to Oakland. Not the
#     same franchise as KCR (Royals, an expansion team from 1969).
#   MON, OAK, historical/dead codes, etc. — genuinely different parks.
TEAM_CODE_ALIASES = {
    'SFG': 'SF',
    'SDP': 'SD',
    'KCR': 'KC',
    'TBR': 'TB',
    'WSN': 'WSH',
    'TBD': 'TB',
    'CAL': 'LAA',
    'ANA': 'LAA',
}

# Stats that should NEVER be park-adjusted
# BABIP: park effect on BABIP is ~.010-.020, not proportional to run-scoring
# park factor. Applying the full 1.13x Coors factor to .290 BABIP gives .257,
# which massively overcorrects. Requires a BABIP-specific park factor.
EXCLUDED_STATS = frozenset({'Age', 'wRC+', 'BABIP'})


# =============================================================================
# wOBA-SCALE RESIDUAL PARK FACTORS (2024-2026, Baseball Savant)
# =============================================================================
# PARK_FACTORS_5YR (above) is calibrated to RUNS SCORED. Runs compound
# nonlinearly off the underlying rate stats, so a park that inflates run
# scoring by e.g. 13% (Coors) does NOT inflate wOBA by 13% — using the
# runs-scale number to adjust a rate stat overstates the effect.
#
# This table is the RESIDUAL park effect that's left over AFTER a player's
# xwOBA already accounts for quality of contact (exit velo/launch angle).
# It's built from each team's wOBAcon / xwOBAcon ratio on balls in play —
# same batted-ball quality, different observed outcome — which isolates the
# park's physical effect (altitude, wall distance/height, foul territory,
# air density) from contact quality itself.
#
# Since predict_models.py already substitutes xwOBA in for wOBA before
# Marcel ever sees a player's history, most of the park effect is already
# stripped out upstream. This table is for the *leftover* piece — use it
# anywhere the pipeline still needs to neutralize/reapply park effect on
# wOBA-derived value (calculate_wrc_plus, calculate_war_components,
# _apply_park_factors_to_batter_predictions), NOT PARK_FACTORS_5YR, which
# is 2-3x too large for this purpose (e.g. COL 113 vs. residual 112 is
# close, but SEA 94 vs. residual 94 only lines up by coincidence — TEX 99
# Basic vs. 95 residual, PHI 101 Basic vs. 106 residual show the runs-scale
# number is not a reliable stand-in).
#
# Source: Baseball Savant, wOBAcon/xwOBAcon, 2024-2026 (3-year window).
# Missing team (ATH/Athletics) has no stable park yet (Sacramento, temp
# home) — defaults to neutral (100) via .get() fallback below.
WOBA_RESIDUAL_PF = {
    'LAA': 102,
    'BAL': 102,
    'BOS': 104,
    'CHW': 98,
    'CLE': 100,
    'DET': 102,
    'KC':  98,
    'MIN': 102,
    'NYY': 98,
    'SEA': 94,
    'TB':  99,
    'TEX': 95,
    'TOR': 100,
    'ARI': 99,
    'ATL': 100,
    'CHC': 97,
    'CIN': 102,
    'COL': 112,
    'MIA': 99,
    'HOU': 104,
    'LAD': 99,
    'MIL': 101,
    'WSH': 99,
    'NYM': 98,
    'PHI': 106,
    'PIT': 99,
    'STL': 96,
    'SD':  96,
    'SF':  99,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_park_factor(team: str) -> float:
    """
    Get RUNS-SCALE park factor as a multiplier (e.g., 1.13 for COL, 0.94 for SEA).

    Returns 1.0 (neutral) for unknown teams or free agents.

    NOTE: Do not use this for wOBA/AVG/SLG/OBP-derived value — it will
    overstate the park effect for extreme parks since those stats already
    had most of their park effect removed via xwOBA/xBA/xSLG substitution
    upstream. Use get_woba_residual_factor() instead for anything touching
    wOBA-derived batting value (WAR, wRC+, rate-stat reapplication).
    """
    if team is None or (isinstance(team, float) and np.isnan(team)):
        return 1.0
    code = str(team).upper().strip()
    code = TEAM_CODE_ALIASES.get(code, code)
    return PARK_FACTORS_5YR.get(code, 100) / 100.0


def get_woba_residual_factor(team: str) -> float:
    """
    Get the RESIDUAL (wOBAcon/xwOBAcon) park factor as a multiplier
    (e.g., 1.12 for COL, 0.94 for SEA).

    This is the park effect left over after xwOBA-based substitution has
    already removed the quality-of-contact portion. Use this — not
    get_park_factor() — for any wOBA/AVG/SLG/OBP-scale adjustment.

    Returns 1.0 (neutral) for unknown teams or free agents.
    """
    if team is None or (isinstance(team, float) and np.isnan(team)):
        return 1.0
    code = str(team).upper().strip()
    code = TEAM_CODE_ALIASES.get(code, code)
    return WOBA_RESIDUAL_PF.get(code, 100) / 100.0


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
    factor_fn=get_park_factor,
) -> pd.DataFrame:
    """
    Divide stats by park factor to get park-neutral values (true talent).
    
    A player in Coors (PF=113) hitting .300 AVG has a neutral AVG of .300/1.13 ≈ .265.
    
    Args:
        df: DataFrame with player stats (must contain team_column)
        input_features: List of model input features
        team_column: Name of the column containing team abbreviations
        factor_fn: Function mapping team -> multiplier. Defaults to
            get_park_factor (runs-scale). Pass get_woba_residual_factor
            for wOBA/AVG/SLG/OBP-scale features instead.
        
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
        lambda t: factor_fn(t)
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
    factor_fn=get_park_factor,
) -> pd.DataFrame:
    """
    Multiply stats by park factor to convert from park-neutral to park-adjusted.
    
    Used in WAR calculation after the model outputs park-neutral predictions
    and we know which team the player will be on.
    
    Args:
        df: DataFrame with park-neutral predicted stats
        features: List of features to adjust
        team_column: Name of the column containing team abbreviations
        factor_fn: Function mapping team -> multiplier. Defaults to
            get_park_factor (runs-scale). Pass get_woba_residual_factor
            for wOBA/AVG/SLG/OBP-scale features instead.
        
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
        lambda t: factor_fn(t)
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
    factor_fn=get_park_factor,
) -> np.ndarray:
    """
    Neutralize a single player's sequence array (used in prediction pipeline).
    
    Args:
        sequence: Shape (seq_len, n_features) — unscaled feature values
        input_features: Feature names corresponding to columns of sequence
        team: Team abbreviation for park factor lookup
        factor_fn: Function mapping team -> multiplier. Defaults to
            get_park_factor (runs-scale). Pass get_woba_residual_factor
            for wOBA/AVG/SLG/OBP-scale features instead.
        
    Returns:
        Park-neutralized sequence array
    """
    pf = factor_fn(team)
    if pf == 1.0:
        return sequence  # No adjustment needed

    adjustable_indices = [
        i for i, f in enumerate(input_features) if f not in EXCLUDED_STATS
    ]

    neutralized = sequence.copy()
    for idx in adjustable_indices:
        neutralized[:, idx] = neutralized[:, idx] / pf

    return neutralized