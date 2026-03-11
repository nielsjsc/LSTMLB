"""
Reliability Regression for Player Statistics
==============================================

Implements partial regression to the mean based on sample size, following
the same methodology used by professional projection systems (Steamer, ZiPS, PECOTA).

The core formula (James-Stein / Bayesian shrinkage):

    true_talent = (observed * n + prior * n0) / (n + n0)

Where:
    observed = raw observed rate stat for a season
    n        = batters faced / plate appearances / games / innings (model-dependent)
    prior    = blended prior (career mean ↔ league average, weighted by career volume)
    n0       = stabilization point (stat-specific, in the relevant unit)

The prior is a continuous blend of the player's career mean and league average:

    career_weight = min(1.0, total_career_volume / full_weight_threshold)
    effective_prior = career_weight * career_mean + (1 - career_weight) * league_mean

Veterans above the full-weight threshold have 100% career prior (zero league-
average influence).  Rookies and small-sample players regress heavily toward
league average, with a smooth linear ramp in between.

Stabilization points represent the sample size required for a stat to become
50% signal / 50% noise. These are well-established from FanGraphs research
(Russell Carleton, Tom Tango, et al).

Supported model types:
    - pitcher (SP/RP): regresses rate stats based on Batters Faced (TBF)
    - batter: regresses rate stats based on Plate Appearances (PA)
    - baserunning: regresses rate stats based on Games (G)
    - defense (infield/outfield/catcher): regresses runs-based stats based on Innings (Inn)

Usage:
    # In training pipeline (operates on full DataFrame):
    df = regress_stats(df, features, model_type='pitcher', era='modern')

    # In prediction pipeline (operates on per-player sequence):
    sequence = regress_player_sequence(player_data, features, model_type='pitcher', era='modern')
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Literal

logger = logging.getLogger(__name__)

# =============================================================================
# STABILIZATION POINTS
# =============================================================================
# Source: FanGraphs "How Long Does It Take To Stabilize?" research
# (Russell Carleton, Pizza Cutter, Tom Tango)
#
# These represent the sample size needed for a stat to be 50% true talent /
# 50% noise. Lower = stabilizes faster = more trustworthy at small samples.
#
# PITCHER stabilization is in BATTERS FACED (BF)
# BATTER stabilization is in PLATE APPEARANCES (PA)
# BASERUNNING stabilization is in GAMES (G)
# DEFENSE stabilization is in INNINGS (Inn)

PITCHER_STABILIZATION_POINTS: Dict[str, int] = {
    # Classical stats (available 1950+)
    'K%':       500,
    'BB%':      500,
    'HR%':      300,    # ~258 IP to stabilize (data-derived)
    'HBP%':     300,    # ~231 IP to stabilize (data-derived)
    'BABIP':    1200,    # ~190 IP; YoY r~0.30, mostly noise
    'ERA':      1320,
    'FIP':      1320,
    # PITCHf/x era stats (2002+)
    'FBv':      50,
    'SwStr%':   200,
    'CSW%':     250,
    'GB%':      70,
    'FB%':      70,
    'LD%':      70,
    'HR/FB':    1600,    # ~300 FB x ~4 TBF/FB ~ 1200 TBF
    'Contact%': 100,
    'xFIP':     1320,
    'SIERA':    1320,
    # Statcast era stats (2020+)
    'Stuff+':    20,
    'Location+': 100,
    'Pitching+': 60,
    'xERA':      1320,
}

# =========================================================================
# BATTER STATS (unit: Plate Appearances)
# =========================================================================
# Source: Russell Carleton / Tom Tango stabilization research
# wOBA and wRC+ stabilize around the same rate since wRC+ derives from wOBA
BATTER_STABILIZATION_POINTS: Dict[str, int] = {
    'BB%':   120,   # Walk rate (fast — plate discipline is a stable skill)
    'K%':    60,    # Strikeout rate (very fast — contact ability is innate)
    'AVG':   300,   # Batting average (very slow — BABIP-dependent)
    'OBP':   460,   # On-base percentage (moderate — includes walks)
    'SLG':   320,   # Slugging (moderate — power is fairly stable)
    'wOBA':  200,   # Weighted on-base average (moderate)
    'wRC+':  350,   # Weighted runs created plus (same basis as wOBA)
    'ISO':   160,   # Isolated power (faster than SLG — pure power signal)
    'HR':    170,   # Home runs per 150G (derived from ISO/power, ~170 PA)
    '2B':    400,   # Doubles per 150G (gap power + speed, slower)
    '3B':    500,   # Triples per 150G (rare event, very slow)
    'RBI':   500,   # RBI per 150G (context-dependent, slow)
    'R':     500,   # Runs per 150G (context-dependent, slow)
    'HBP':   300,   # Hit-by-pitch per 150G (moderate)
    'SF':    500,   # Sacrifice flies per 150G (rare, very slow)
    # Statcast
    'EV':    40,    # Exit velocity (physical measure, very fast)
    'xwOBA': 60,    # Expected wOBA (Statcast, stabilizes very fast)
}

# =========================================================================
# BASERUNNING STATS (unit: Games)
# =========================================================================
# Sprint speed is a physical measure that stabilizes almost immediately.
# Baserunning runs are noisier and require more sample.
BASERUNNING_STABILIZATION_POINTS: Dict[str, int] = {
    'sc_sprint_speed':                     10,   # Physical measure, extremely stable
    'wSB_rate':                            80,   # Stolen base runs per 150G
    'SB_rate':                             80,   # Stolen bases per 150G
    'CS_rate':                             80,   # Caught stealing per 150G
    'sc_baserunning_runner_runs_tot_rate': 120,  # Total baserunning runs per 150G
    'sc_baserunning_runner_runs_XB_rate':  120,  # Extra-base taking runs per 150G
    'sc_baserunning_runner_runs_SBX_rate': 100,  # SB + extra-base runs per 150G
}

# =========================================================================
# DEFENSE STATS (unit: Innings)
# =========================================================================
# Defensive metrics are notoriously noisy. Even advanced metrics like OAA
# and DRS require large samples to stabilize. Framing is slightly more
# stable because it's measured on every pitch.
DEFENSE_STABILIZATION_POINTS: Dict[str, int] = {
    # Infield + Outfield shared
    'OAA/150':            700,
    'DRS/150':            800,
    'sc_total_runs/150':  1000,
    'sc_range_runs/150':  800,
    'sc_arm_runs/150':    800,
    # Infield-specific
    'sc_dp_runs/150':     1000,
    # Outfield-specific (arm runs shared above)
    # Catcher-specific
    'sc_framing_runs/150':  1200,  # Framing more stable (every pitch)
    'sc_throwing_runs/150': 1500,  # Throwing (SB attempts, sparse)
    'sc_blocking_runs/150': 1500,  # Blocking (wild pitches, sparse)
}


# Maximum number of recent seasons to use when computing the career prior.
PRIOR_MAX_SEASONS = 4

# Features where regression should NOT be applied
SKIP_FEATURES = {'Age', 'IP', 'Inn', 'G', 'PA', 'TBF'}

# Approximate BF per IP (used when TBF is not available)
BF_PER_IP = 4.3

# Approximate PA per game for batters
PA_PER_GAME = 3.9

# Floor fraction for adaptive stabilization — even the most established
# player still gets at least this fraction of the base stabilization point.
ADAPTIVE_STAB_FLOOR = 0.10

# Rookie boost — players with very little career data get n0 ABOVE base,
# meaning stronger regression.  The boost fades linearly to zero at
# ADAPTIVE_BOOST_CAREER_FRACTION × career_stab.
ADAPTIVE_BOOST_MAX = 1.0           # At 0 career volume, n0 is (1 + this) × base
ADAPTIVE_BOOST_CAREER_FRACTION = 0.4  # Fraction of career_stab where boost ends


# =============================================================================
# CAREER WEIGHT — prior blending between career mean and league average
# =============================================================================
# career_weight determines what fraction of the prior comes from the
# player's OWN career mean vs the LEAGUE average.
#
# We use a linear ramp that saturates at 1.0 once career volume reaches
# the full-weight threshold.  This ensures established veterans have ZERO
# league-average influence, while rookies regress heavily toward league avg.
#
# PER-STAT THRESHOLDS: When a stabilization point (n0) is provided, the
# threshold scales proportionally: threshold = n0 × CAREER_WEIGHT_N0_MULTIPLIER.
# This means fast-stabilizing stats (K%, GB%) reach full career weight sooner,
# while noisy stats (BABIP, HR/FB, ERA) require more career data.
#
# Examples with n0 multiplier = 1.25:
#   K%    (n0= 800): full career weight at 1000 TBF
#   BABIP (n0=1200): full career weight at 1500 TBF
#   ERA   (n0=1320): full career weight at 1650 TBF
#   GB%   (n0=  70): full career weight at   88 TBF
#
# Formula:
#   career_weight = min(1.0, career_volume / full_weight_threshold)
#   effective_prior = career_weight * career_mean + (1 - career_weight) * league_mean

# Multiplier applied to each feature's stabilization point to derive the
# career-weight saturation threshold.  Higher → more data needed before
# the career mean fully replaces the league average.
CAREER_WEIGHT_N0_MULTIPLIER = 1.25


def _get_career_stabilization(model_type: str) -> int:
    """
    Get the career volume threshold for prior blending.

    At this volume, career and league priors are weighted 50/50.
    Below this, league average dominates. Above, career data dominates.

    Args:
        model_type: 'pitcher', 'batter', 'baserunning', 'defense_*'

    Returns:
        Career stabilization threshold in the appropriate volume unit
    """
    if model_type == 'pitcher':
        return 1000   # ~230 IP worth of TBF
    elif model_type == 'batter':
        return 800   # ~2 full seasons of PA
    elif model_type == 'baserunning':
        return 200    # ~200 games (~1.3 full seasons)
    elif model_type.startswith('defense') or model_type.startswith('fielding'):
        return 1500   # ~1500 innings (~2 seasons for a starter)
    return 1200       # fallback


def _get_career_full_weight_threshold(model_type: str) -> float:
    """
    Volume at which career_weight saturates to 1.0 (no league-avg influence).

    At or above this volume, the prior is 100% career mean.
    Below, career_weight increases linearly: vol / threshold.

    Returns:
        Career full-weight threshold in the appropriate volume unit
    """
    if model_type == 'pitcher':
        return 1000    # ~460 IP of TBF
    elif model_type == 'batter':
        return 2500    # ~4 full seasons of PA
    elif model_type == 'baserunning':
        return 400     # ~2.5 full seasons of games
    elif model_type.startswith('defense') or model_type.startswith('fielding'):
        return 3000    # ~2 full seasons of innings as starter
    return 2500        # fallback


def _compute_career_weight(career_volume: float, model_type: str,
                           stab_point: Optional[int] = None) -> float:
    """
    Fraction of the prior that comes from the player's career mean (vs league avg).

    Uses a linear ramp that saturates at 1.0: once a player has enough career
    volume, their prior is 100% career mean with ZERO league-average influence.

    When stab_point is provided, the threshold is PER-FEATURE:
        threshold = stab_point × CAREER_WEIGHT_N0_MULTIPLIER
    This lets fast-stabilizing stats (K%, GB%) reach full career weight sooner
    than noisy ones (BABIP, ERA, HR/FB).

    When stab_point is None, falls back to the flat model-type threshold
    (backward-compatible with callers that don't have per-feature context).
    """
    if stab_point is not None:
        threshold = stab_point * CAREER_WEIGHT_N0_MULTIPLIER
        threshold = max(threshold, 50.0)  # floor: prevent degenerate thresholds
    else:
        threshold = _get_career_full_weight_threshold(model_type)
    if threshold <= 0:
        return 1.0
    return min(1.0, career_volume / threshold)


def _effective_stabilization_point(
    base_n0: int,
    career_volume: float,
    career_stab: float,
    floor_fraction: float = ADAPTIVE_STAB_FLOOR,
) -> float:
    """
    Scale a stat's stabilization point based on accumulated career volume.

    Two regimes:

    1. **Rookie boost** (career_volume < boost_threshold):
       Players with very little career data get n0 ABOVE the base, meaning
       *stronger* regression.  An extra additive term fades linearly to zero
       as career volume approaches the boost threshold.

    2. **Veteran reduction** (career_volume >= boost_threshold):
       The standard adaptive formula reduces n0 as career evidence grows,
       so established players are barely regressed.

    The two regimes are *continuous* — at the boost threshold, the additive
    boost term reaches zero and the standard formula takes over seamlessly.

    A floor ensures even the most established player still gets a minimal
    amount of regression (default 10% of base_n0).

    Examples (batter, wOBA n0=200, career_stab=1200, boost_threshold=480):
        Debut    (  10 PA): n0 ≈ 200 (base) + 196 (boost) + 1.6 (base reduction) = 398 → very heavy regression
        Rookie   (  69 PA): n0 ≈ 189 (base) + 171 (boost)                         = 360 → heavy regression
        Prospect ( 232 PA): n0 ≈ 168 (base) + 103 (boost)                         = 271 → strong regression
        Breakout ( 480 PA): n0 ≈ 143 (base) +   0 (boost)                         = 143 → moderate
        2nd year ( 700 PA): n0 ≈ 126                                               = 126 → moderate
        Veteran  (3000 PA): n0 ≈  57                                               =  57 → light
        Star     (5000 PA): n0 ≈  39                                               =  39 → minimal

    Args:
        base_n0: Raw stabilization point from the lookup tables
        career_volume: Total career exposure (PA / TBF / G / Inn)
        career_stab: Career stabilization threshold for the model type
        floor_fraction: Minimum fraction of base_n0 to preserve

    Returns:
        Effective stabilization point (always >= base_n0 * floor_fraction)
    """
    if career_volume <= 0 or career_stab <= 0:
        return float(base_n0) * (1 + ADAPTIVE_BOOST_MAX)

    # Standard adaptive reduction (applies at all career volumes)
    standard_n0 = base_n0 * (career_stab / (career_volume + career_stab))

    # Rookie boost: extra regression for very low career volume
    boost_threshold = career_stab * ADAPTIVE_BOOST_CAREER_FRACTION
    if career_volume < boost_threshold:
        boost_fraction = 1.0 - (career_volume / boost_threshold)
        extra = base_n0 * ADAPTIVE_BOOST_MAX * boost_fraction
        return standard_n0 + extra

    # Veteran regime: pure adaptive reduction with floor
    return max(standard_n0, base_n0 * floor_fraction)

def _get_stabilization_point(feature: str, era: str = 'statcast', model_type: str = 'pitcher') -> Optional[int]:
    """
    Get the stabilization point for a feature.

    Looks up the correct stabilization table based on model_type:
        - pitcher: PITCHER_STABILIZATION_POINTS (unit=BF)
        - batter: BATTER_STABILIZATION_POINTS (unit=PA)
        - baserunning: BASERUNNING_STABILIZATION_POINTS (unit=G)
        - defense_*: DEFENSE_STABILIZATION_POINTS (unit=Inn)

    Args:
        feature: Stat name (e.g., 'K%', 'ERA', 'wRC+', 'sc_total_runs/150')
        era: Kept for backward compatibility (unused).
        model_type: 'pitcher', 'batter', 'baserunning', 'defense_infield',
                    'defense_outfield', 'defense_catcher'

    Returns:
        Stabilization point in the appropriate unit, or None if not applicable
    """
    if feature in SKIP_FEATURES:
        return None

    # Route to the correct stabilization table
    if model_type == 'pitcher':
        return PITCHER_STABILIZATION_POINTS.get(feature)

    elif model_type == 'batter':
        return BATTER_STABILIZATION_POINTS.get(feature)

    elif model_type == 'baserunning':
        return BASERUNNING_STABILIZATION_POINTS.get(feature)

    elif model_type.startswith('defense'):
        return DEFENSE_STABILIZATION_POINTS.get(feature)

    return None


def _get_volume_column(model_type: str) -> str:
    """
    Get the name of the volume/exposure column for a model type.

    This is the column used as the denominator 'n' in the Bayesian shrinkage
    formula. Different model types use different exposure measures.

    Args:
        model_type: One of 'pitcher', 'batter', 'baserunning', 'defense_*'

    Returns:
        Column name string
    """
    if model_type == 'pitcher':
        return 'TBF'
    elif model_type == 'batter':
        return 'PA'
    elif model_type == 'baserunning':
        return 'G'
    elif model_type.startswith('defense'):
        return 'Inn'
    return 'PA'  # fallback


def _get_weight_column(model_type: str) -> str:
    """
    Get the column used for volume-weighting career priors.

    This determines how seasons are weighted when computing the career mean.
    More playing time = more weight.

    For pitchers, TBF (total batters faced) is used because rate stats like
    K%, BB%, HR%, HBP% are per-TBF rates. Weighting by TBF ensures that
    the career mean equals the true career rate (e.g., career HR% = total HR /
    total TBF). Falls back to IP if TBF is unavailable.

    Args:
        model_type: One of 'pitcher', 'batter', 'baserunning', 'defense_*'

    Returns:
        Column name string
    """
    if model_type == 'pitcher':
        return 'TBF'
    elif model_type == 'batter':
        return 'PA'
    elif model_type == 'baserunning':
        return 'G'
    elif model_type.startswith('defense'):
        return 'Inn'
    return 'PA'  # fallback


def _estimate_volume(row: pd.Series, model_type: str) -> float:
    """
    Estimate volume from available columns when the primary column is missing.

    Args:
        row: A DataFrame row
        model_type: Model type string

    Returns:
        Estimated volume value
    """
    vol_col = _get_volume_column(model_type)

    if vol_col in row.index and not np.isnan(row.get(vol_col, np.nan)):
        return row[vol_col]

    # Fallback estimations
    if model_type == 'pitcher':
        if 'IP' in row.index and not np.isnan(row.get('IP', np.nan)):
            return row['IP'] * BF_PER_IP
    elif model_type == 'batter':
        if 'G' in row.index and not np.isnan(row.get('G', np.nan)):
            return row['G'] * PA_PER_GAME
    elif model_type == 'baserunning':
        if 'PA' in row.index and not np.isnan(row.get('PA', np.nan)):
            return row['PA'] / PA_PER_GAME
    elif model_type.startswith('defense'):
        if 'G' in row.index and not np.isnan(row.get('G', np.nan)):
            return row['G'] * 8.5  # ~8.5 innings per game

    return 0.0


def _compute_career_prior(
    player_data: pd.DataFrame,
    feature: str,
    current_season: int,
    max_seasons: int = PRIOR_MAX_SEASONS,
    current_age: Optional[float] = None,
    model_type: str = 'pitcher',
    recency_halflife: float = 0,
) -> float:
    """
    Compute a volume-weighted recent career mean for a single feature.

    Uses only the most recent `max_seasons` seasons up to and including
    current_season. This prevents regression toward long-ago peak years
    for aging players (e.g., Kershaw's 2025 won't regress toward his
    2014 peak, only toward his 2022-2025 level).

    If `current_age` is provided, each historical season's value is
    age-adjusted to what it would be at `current_age` before averaging.
    This ensures the prior reflects the player's expected talent NOW,
    not what it was at a younger age.

    If `recency_halflife` > 0, an exponential decay is applied so that
    more recent seasons receive higher effective weight. A season N years
    ago gets a multiplier of 2^(-N / recency_halflife). For example, with
    halflife=3, a season from 3 years ago gets half the weight of the most
    recent season (in addition to volume weighting).

    The weight column is model-type-dependent:
        - pitcher: TBF (total batters faced — matches per-TBF rate stats)
        - batter: PA (plate appearances)
        - baserunning: G (games)
        - defense_*: Inn (innings)

    Args:
        player_data: All rows for one player, sorted by Season
        feature: The stat column name
        current_season: The season being regressed
        max_seasons: Maximum number of recent seasons to include in the prior
        current_age: If provided, age-adjust each season to this age
        model_type: 'pitcher', 'batter', etc. (used for aging curves and weight column)
        recency_halflife: Exponential decay half-life in seasons (0 = no decay)

    Returns:
        Volume-weighted recent career mean (optionally age-adjusted), or NaN
    """
    weight_col = _get_weight_column(model_type)

    career = player_data[player_data['Season'] <= current_season].copy()

    # Limit to most recent max_seasons
    if len(career) > max_seasons:
        career = career.tail(max_seasons)

    # Drop rows where the feature or weight column is missing
    # Fall back from TBF → IP for pitchers if TBF is unavailable
    if weight_col not in career.columns:
        if model_type == 'pitcher' and 'IP' in career.columns:
            weight_col = 'IP'
        else:
            # Fallback: unweighted mean
            valid = career.dropna(subset=[feature])
            if valid.empty:
                return np.nan
            return valid[feature].mean()

    valid = career.dropna(subset=[feature, weight_col])
    valid = valid[valid[weight_col] > 0]

    if valid.empty:
        return np.nan

    weights = valid[weight_col].values.astype(np.float64)
    values = valid[feature].values.astype(np.float64)

    # Apply recency decay: multiply volume weights by 2^(-years_ago / halflife)
    if recency_halflife > 0 and 'Season' in valid.columns:
        seasons = valid['Season'].values
        years_ago = current_season - seasons  # 0 for most recent, 1, 2, ...
        decay = np.power(2.0, -years_ago / recency_halflife)
        weights = weights * decay

    # Age-adjust if current_age is provided and Age column exists
    if current_age is not None and 'Age' in valid.columns:
        adjusted_values = []
        for _, row in valid.iterrows():
            obs_age = row['Age']
            adjusted = age_adjust_prior_value(
                row[feature], feature, obs_age, current_age, model_type
            )
            adjusted_values.append(adjusted)
        values = np.array(adjusted_values, dtype=np.float64)

    return np.average(values, weights=weights)


def _compute_league_prior(
    full_df: pd.DataFrame,
    feature: str,
    season: int,
    window: int = 3,
    model_type: str = 'pitcher',
) -> float:
    """
    Compute a league-average prior for a feature using a rolling window.

    Uses volume-weighted average over the last `window` seasons to smooth out
    year-to-year noise while adapting to era shifts.

    The weight column is model-type-dependent:
        - pitcher: IP
        - batter: PA
        - baserunning: G
        - defense_*: Inn

    Args:
        full_df: Full historical DataFrame (all players)
        feature: The stat column name
        season: The season to compute the prior for
        window: Number of seasons to average over
        model_type: 'pitcher', 'batter', 'baserunning', 'defense_*'

    Returns:
        Volume-weighted league average for the feature
    """
    # Statcast fielding metrics (FRV, OAA, DRS, framing, etc.) are all
    # zero-sum by construction — the league average is exactly 0 by definition.
    if model_type.startswith('defense_'):
        return 0.0

    weight_col = _get_weight_column(model_type)

    recent = full_df[
        (full_df['Season'] >= season - window + 1) &
        (full_df['Season'] <= season)
    ]

    if weight_col in recent.columns:
        valid = recent.dropna(subset=[feature, weight_col])
        valid = valid[valid[weight_col] > 0]
    else:
        valid = recent.dropna(subset=[feature])

    if valid.empty:
        # Broader fallback
        if weight_col in full_df.columns:
            valid = full_df.dropna(subset=[feature, weight_col])
            valid = valid[valid[weight_col] > 0]
        else:
            valid = full_df.dropna(subset=[feature])

    if valid.empty:
        return np.nan

    if weight_col in valid.columns:
        return np.average(valid[feature].values, weights=valid[weight_col].values)
    else:
        return valid[feature].mean()


def regress_single_value(
    observed: float,
    bf: float,
    prior: float,
    stabilization_bf: int,
) -> float:
    """
    Apply Bayesian shrinkage to a single observed value.

    true_talent = (observed * bf + prior * n0) / (bf + n0)

    At bf = n0, the estimate is exactly 50/50 observed vs prior.
    At bf >> n0, the estimate is almost entirely the observed value.
    At bf << n0, the estimate is almost entirely the prior.

    Args:
        observed: Raw observed stat value
        bf: Batters faced in this season
        prior: Prior estimate (career mean or league average)
        stabilization_bf: Stabilization point for this stat

    Returns:
        Regressed estimate
    """
    if np.isnan(observed) or np.isnan(prior):
        return observed  # Can't regress if either value is missing

    weight_observed = bf / (bf + stabilization_bf)
    return weight_observed * observed + (1 - weight_observed) * prior


# =============================================================================
# TRAINING PIPELINE: Full DataFrame regression
# =============================================================================

def regress_pitcher_stats(
    df: pd.DataFrame,
    features: List[str],
    era: str = 'classical',
    league_df: Optional[pd.DataFrame] = None,
    min_career_seasons: int = 2,
) -> pd.DataFrame:
    """
    Legacy wrapper — calls regress_stats with model_type='pitcher'.
    Kept for backward compatibility.
    """
    return regress_stats(df, features, model_type='pitcher', era=era,
                         league_df=league_df, min_career_seasons=min_career_seasons)


def regress_stats(
    df: pd.DataFrame,
    features: List[str],
    model_type: str = 'pitcher',
    era: str = 'classical',
    league_df: Optional[pd.DataFrame] = None,
    min_career_seasons: int = 2,
) -> pd.DataFrame:
    """
    Apply reliability regression to all player stats in a DataFrame.

    This is the main entry point for the TRAINING pipeline. It operates on
    the full DataFrame after filtering but BEFORE scaling.

    For each player-season:
    1. Compute the player's volume-weighted career mean up to that season
    2. Blend career mean with league average based on total career volume
       (more career exposure → more trust in career mean)
    3. Regress each rate stat toward the blended prior based on volume
       and stabilization rate

    The volume column (n in the shrinkage formula) is model-type-dependent:
        - pitcher: TBF (batters faced)
        - batter: PA (plate appearances)
        - baserunning: G (games)
        - defense_*: Inn (innings)

    Args:
        df: Filtered player DataFrame with Season, IDfg, and volume columns
        features: List of model input features to regress
        model_type: 'pitcher', 'batter', 'baserunning', 'defense_infield',
                    'defense_outfield', 'defense_catcher'
        era: Data era for stabilization points ('classical', 'pitchfx', 'statcast')
        league_df: Full DataFrame for computing league priors (defaults to df itself)
        min_career_seasons: Deprecated — blended priors are now used instead.
                           Kept for backward compatibility.

    Returns:
        DataFrame with regressed stat values (original columns overwritten)
    """
    if league_df is None:
        league_df = df

    vol_col = _get_volume_column(model_type)
    weight_col = _get_weight_column(model_type)

    # Identify which features actually need regression
    regressable = []
    stab_points = {}
    for feat in features:
        n0 = _get_stabilization_point(feat, era, model_type)
        if n0 is not None:
            regressable.append(feat)
            stab_points[feat] = n0

    if not regressable:
        logger.info("No regressable features found — skipping reliability regression")
        return df

    logger.info(f"Applying reliability regression to {len(regressable)} features: {regressable}")
    logger.info(f"Model type: {model_type}, era: {era}, volume column: {vol_col}")
    logger.info(f"Career stabilization: {_get_career_stabilization(model_type)} (blended priors)")

    # Pre-compute league priors for each (feature, season) combination
    seasons = df['Season'].unique()
    league_priors: Dict[str, Dict[int, float]] = {}
    for feat in regressable:
        league_priors[feat] = {}
        for season in seasons:
            league_priors[feat][season] = _compute_league_prior(
                league_df, feat, season, model_type=model_type
            )

    # Ensure volume column exists; estimate from available data if needed
    df = df.copy()
    if vol_col not in df.columns:
        logger.warning(f"{vol_col} column not found — estimating from available data")
        df[vol_col] = df.apply(lambda row: _estimate_volume(row, model_type), axis=1)

    df = df.sort_values(['IDfg', 'Season'])

    regressed_count = 0
    total_count = 0

    for player_id, player_group in df.groupby('IDfg'):
        player_idx = player_group.index
        player_seasons = player_group['Season'].values
        cumulative_volume = 0.0
        career_stab = _get_career_stabilization(model_type)

        for i, (idx, row) in enumerate(player_group.iterrows()):
            season = row['Season']
            volume = _estimate_volume(row, model_type)
            cumulative_volume += volume

            for feat in regressable:
                observed = row[feat]
                if np.isnan(observed):
                    continue

                # Per-feature career weight: fast-stabilizing stats reach full
                # career weight sooner than noisy ones.
                feat_career_wt = _compute_career_weight(
                    cumulative_volume, model_type, stab_point=stab_points[feat]
                )

                # Adaptive stabilization: reduce n0 as career evidence grows.
                # Established players barely get regressed; rookies get full regression.
                n0 = _effective_stabilization_point(
                    stab_points[feat], cumulative_volume, career_stab
                )
                total_count += 1

                # Compute career and league priors, then blend them.
                # NOTE: Training pipeline uses RAW career prior (no aging adjustment).
                # Aging curves are only applied in the prediction pipeline's regression
                # prior, so the model learns from unmodified historical relationships.
                career_prior = _compute_career_prior(
                    player_group, feat, season, model_type=model_type
                )
                league_prior = league_priors[feat].get(season, observed)

                # Blend career and league priors based on total career exposure.
                # Players with little career data regress toward league average;
                # established players regress toward their own career mean.
                if not np.isnan(career_prior):
                    prior = feat_career_wt * career_prior + (1 - feat_career_wt) * league_prior
                else:
                    prior = league_prior

                regressed = regress_single_value(observed, volume, prior, n0)
                df.at[idx, feat] = regressed

                if abs(regressed - observed) > 1e-6:
                    regressed_count += 1

    pct = (regressed_count / total_count * 100) if total_count > 0 else 0
    logger.info(
        f"Reliability regression complete: {regressed_count}/{total_count} "
        f"values adjusted ({pct:.1f}%)"
    )

    return df


# =============================================================================
# PREDICTION PIPELINE: Per-player sequence regression
# =============================================================================

def regress_player_sequence(
    player_data: pd.DataFrame,
    features: List[str],
    model_type: str = 'pitcher',
    era: str = 'classical',
    league_priors: Optional[Dict[str, float]] = None,
    min_career_seasons: int = 2,
    recency_halflife: float = 0,
    league_weight_overrides: Optional[Dict[str, float]] = None,
    seq_length: Optional[int] = None,
    sequence_ip_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply reliability regression to a single player's historical data.

    This is the entry point for the PREDICTION pipeline. It operates on
    one player's data before sequence construction.

    **Sequence-based regression** (when seq_length is provided):
    The regression prior is computed from the SEQUENCE WINDOW — the same
    seasons the model will actually see — rather than the full career.
    This prevents distant peak seasons (e.g. 2014 Kershaw) from pulling
    the regression target away from current talent.

    When the sequence window has sufficient volume (>= career_stab threshold,
    e.g. 1000 TBF for pitchers), the prior is 100% sequence mean.
    Below that threshold, the sequence mean is blended with league average:
        seq_weight = min(1.0, sequence_TBF / career_stab)
        prior = seq_weight * sequence_mean + (1 - seq_weight) * league_mean

    **Legacy mode** (when seq_length is None):
    Falls back to the previous behavior using PRIOR_MAX_SEASONS career mean.

    For features listed in league_weight_overrides, the league-average
    fraction of the prior is floored at the specified value regardless of
    sequence volume.

    The volume column (n in the shrinkage formula) is model-type-dependent:
        - pitcher: TBF (batters faced)
        - batter: PA (plate appearances)
        - baserunning: G (games)
        - defense_*: Inn (innings)

    Args:
        player_data: One player's historical DataFrame, sorted by Season
        features: List of model input features to regress
        model_type: 'pitcher', 'batter', 'baserunning', 'defense_infield',
                    'defense_outfield', 'defense_catcher'
        era: Data era for stabilization points
        league_priors: Pre-computed league averages per feature.
                      If None, uses the player's own data (less accurate for rookies)
        min_career_seasons: Deprecated — kept for backward compatibility.
        recency_halflife: Exponential decay half-life in seasons (0 = no decay)
        league_weight_overrides: Per-feature minimum league-average weight in
            the prior blend. Maps feature name → float in [0, 1].
        seq_length: Number of seasons in the model's input sequence. When
            provided, the prior is computed from the sequence window only.
        sequence_ip_threshold: Minimum IP for a season to qualify for the
            sequence (e.g. 20 for SP, 10 for RP). Required when seq_length
            is provided for pitchers.

    Returns:
        DataFrame with regressed values (copy — does not modify input)
    """
    if len(player_data) == 0:
        return player_data.copy()

    result = player_data.copy()
    vol_col = _get_volume_column(model_type)
    weight_col = _get_weight_column(model_type)

    # Ensure volume column exists
    if vol_col not in result.columns:
        result[vol_col] = result.apply(lambda row: _estimate_volume(row, model_type), axis=1)

    # Identify regressable features
    regressable = []
    stab_points = {}
    for feat in features:
        n0 = _get_stabilization_point(feat, era, model_type)
        if n0 is not None and feat in result.columns:
            regressable.append(feat)
            stab_points[feat] = n0

    if not regressable:
        return result

    career_stab = _get_career_stabilization(model_type)
    current_age = result['Age'].iloc[-1] if 'Age' in result.columns else None

    # Determine the aging model_type key for lookup
    aging_key = model_type
    if model_type.startswith('defense_'):
        aging_key = model_type.replace('defense_', 'fielding_')

    # Convert rate stat columns to float64 to avoid dtype warnings
    for feat in regressable:
        if feat in result.columns:
            result[feat] = result[feat].astype('float64')

    # =====================================================================
    # IDENTIFY SEQUENCE WINDOW
    # =====================================================================
    # When seq_length is provided, the prior is computed from the seasons
    # the model will actually see (matching predict_future_stats logic).
    # Otherwise fall back to PRIOR_MAX_SEASONS career window.
    if seq_length is not None:
        ip_thresh = sequence_ip_threshold or 0
        # Mirror predict_future_stats: take tail(seq_length + 2), filter by IP
        candidates = result.tail(seq_length + 2)
        if 'IP' in candidates.columns and ip_thresh > 0:
            seq_rows = candidates[candidates['IP'] >= ip_thresh].tail(seq_length)
        else:
            seq_rows = candidates.tail(seq_length)
        prior_source = seq_rows
        prior_volume = sum(
            _estimate_volume(row, model_type) for _, row in seq_rows.iterrows()
        )
    else:
        prior_source = result.tail(PRIOR_MAX_SEASONS)
        prior_volume = sum(
            _estimate_volume(row, model_type) for _, row in result.iterrows()
        )

    # Career weight: per-feature blend between sequence/career mean and league average.
    # Fast-stabilizing stats (K%, GB%) reach full career weight sooner than noisy ones.
    career_weights: Dict[str, float] = {}
    for feat in regressable:
        career_weights[feat] = _compute_career_weight(
            prior_volume, model_type, stab_point=stab_points[feat]
        )

    # =====================================================================
    # COMPUTE PRIOR MEANS (from sequence window or career window)
    # =====================================================================
    career_means: Dict[str, float] = {}
    for feat in regressable:
        if current_age is not None and 'Age' in prior_source.columns:
            # Age-adjusted, volume-weighted mean of the prior source window
            career_means[feat] = compute_age_adjusted_career_prior(
                prior_source, feat, current_age,
                max_seasons=len(prior_source), model_type=aging_key,
                recency_halflife=recency_halflife
            )
        else:
            if weight_col in prior_source.columns:
                valid = prior_source.dropna(subset=[feat, weight_col])
                valid = valid[valid[weight_col] > 0]
                if not valid.empty:
                    career_means[feat] = np.average(
                        valid[feat].values, weights=valid[weight_col].values
                    )
                else:
                    career_means[feat] = np.nan
            else:
                valid = prior_source.dropna(subset=[feat])
                if not valid.empty:
                    career_means[feat] = valid[feat].mean()
                else:
                    career_means[feat] = np.nan

    # =====================================================================
    # BLEND PRIORS (sequence/career mean ↔ league average)
    # =====================================================================
    blended_priors: Dict[str, float] = {}
    _overrides = league_weight_overrides or {}
    for feat in regressable:
        career_mean = career_means.get(feat, np.nan)
        league_mean = league_priors.get(feat, np.nan) if league_priors else np.nan
        feat_career_wt = career_weights.get(feat, 0.5)

        if not np.isnan(career_mean) and not np.isnan(league_mean):
            min_league_wt = _overrides.get(feat, 0.0)
            effective_career_wt = min(feat_career_wt, 1.0 - min_league_wt)
            blended_priors[feat] = effective_career_wt * career_mean + (1 - effective_career_wt) * league_mean
        elif not np.isnan(career_mean):
            blended_priors[feat] = career_mean
        elif not np.isnan(league_mean):
            blended_priors[feat] = league_mean

    # Log representative career weights (min/max range across features)
    _cw_values = list(career_weights.values())
    logger.debug(
        f"Prior blending: prior_vol={prior_volume:.0f}, "
        f"career_stab={career_stab}, "
        f"career_weight=[{min(_cw_values):.3f}..{max(_cw_values):.3f}], "
        f"source={'sequence' if seq_length else 'career'}({len(prior_source)} seasons)"
    )

    _example_feat = 'wOBA' if 'wOBA' in stab_points else regressable[0]
    _example_eff = _effective_stabilization_point(
        stab_points[_example_feat], prior_volume, career_stab
    )
    logger.debug(
        f"Adaptive stabilization: {_example_feat} base_n0={stab_points[_example_feat]} → "
        f"effective_n0={_example_eff:.1f} (prior_vol={prior_volume:.0f})"
    )

    # =====================================================================
    # REGRESS EACH SEASON
    # =====================================================================
    for idx, row in result.iterrows():
        volume = _estimate_volume(row, model_type)

        for feat in regressable:
            observed = row[feat]
            if np.isnan(observed):
                continue

            # Adaptive stabilization keyed off prior_volume (sequence or career)
            n0 = _effective_stabilization_point(
                stab_points[feat], prior_volume, career_stab
            )

            prior = blended_priors.get(feat)
            if prior is None or np.isnan(prior):
                continue

            result.at[idx, feat] = regress_single_value(observed, volume, prior, n0)

    return result


def compute_regressed_career_mean(
    player_data: pd.DataFrame,
    features: List[str],
    model_type: str = 'pitcher',
    era: str = 'classical',
    league_priors: Optional[Dict[str, float]] = None,
    recency_halflife: float = 0,
    league_weight_overrides: Optional[Dict[str, float]] = None,
    seq_length: Optional[int] = None,
    sequence_ip_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute the best available "true talent" estimate for padding.

    This produces a single set of feature values representing the player's
    true talent level — used for padding when the sequence is shorter than
    seq_length.

    Unlike raw career averages, this applies regression to each season first,
    then takes the volume-weighted mean of the regressed values. This prevents
    small-sample seasons from distorting the padding values.

    Args:
        player_data: One player's historical DataFrame
        features: List of features
        model_type: 'pitcher', 'batter', 'baserunning', 'defense_*'
        era: Data era
        league_priors: Pre-computed league averages
        recency_halflife: Exponential decay half-life in seasons (0 = no decay)
        league_weight_overrides: Per-feature minimum league weight
        seq_length: Forwarded to regress_player_sequence for sequence-based priors
        sequence_ip_threshold: Forwarded to regress_player_sequence

    Returns:
        Dict mapping feature name to regressed career mean
    """
    if len(player_data) == 0:
        return {}

    weight_col = _get_weight_column(model_type)

    # First regress the player's data (with sequence-based priors if configured)
    regressed = regress_player_sequence(
        player_data, features, model_type=model_type, era=era,
        league_priors=league_priors, recency_halflife=recency_halflife,
        league_weight_overrides=league_weight_overrides,
        seq_length=seq_length, sequence_ip_threshold=sequence_ip_threshold,
    )

    # When using sequence-based priors, limit the career mean to the sequence
    # window so padding matches the prior source.
    if seq_length is not None:
        ip_thresh = sequence_ip_threshold or 0
        candidates = regressed.tail(seq_length + 2)
        if 'IP' in candidates.columns and ip_thresh > 0:
            recent = candidates[candidates['IP'] >= ip_thresh].tail(seq_length)
        else:
            recent = candidates.tail(seq_length)
    else:
        recent = regressed.tail(PRIOR_MAX_SEASONS)

    result = {}
    for feat in features:
        if feat in SKIP_FEATURES:
            if feat == 'Age':
                result[feat] = regressed['Age'].iloc[-1]
            continue

        valid = recent.dropna(subset=[feat])

        if weight_col in valid.columns:
            valid_with_weight = valid[valid[weight_col] > 0]
        else:
            valid_with_weight = pd.DataFrame()

        if not valid_with_weight.empty and weight_col in valid_with_weight.columns:
            result[feat] = np.average(
                valid_with_weight[feat].values, weights=valid_with_weight[weight_col].values
            )
        elif not valid.empty:
            result[feat] = valid[feat].mean()
        elif league_priors and feat in league_priors:
            result[feat] = league_priors[feat]
        else:
            result[feat] = 0.0

    return result


def compute_league_priors_from_df(
    df: pd.DataFrame,
    features: List[str],
    model_type: str = 'pitcher',
    season: Optional[int] = None,
    window: int = 3,
) -> Dict[str, float]:
    """
    Compute league-average priors from a full DataFrame.

    Convenience function for the prediction pipeline to pre-compute league
    priors once and pass them to per-player regression.

    Args:
        df: Full historical DataFrame
        features: List of features to compute priors for
        model_type: 'pitcher', 'batter', 'baserunning', 'defense_*'
        season: Target season (uses rolling window ending here)
        window: Number of seasons to average

    Returns:
        Dict mapping feature name to league average
    """
    weight_col = _get_weight_column(model_type)
    priors = {}
    for feat in features:
        if feat in SKIP_FEATURES or feat not in df.columns:
            continue
        if season is not None:
            priors[feat] = _compute_league_prior(
                df, feat, season, window, model_type=model_type
            )
        else:
            # Use all data
            if weight_col in df.columns:
                valid = df.dropna(subset=[feat, weight_col])
                valid = valid[valid[weight_col] > 0]
                if not valid.empty:
                    priors[feat] = np.average(
                        valid[feat].values, weights=valid[weight_col].values
                    )
            else:
                valid = df.dropna(subset=[feat])
                if not valid.empty:
                    priors[feat] = valid[feat].mean()

    return priors


def get_era_for_season(season: int) -> str:
    """
    Determine the appropriate stabilization era for a given season.

    Args:
        season: MLB season year

    Returns:
        Era string: 'classical', 'pitchfx', or 'statcast'
    """
    if season >= 2020:
        return 'statcast'
    elif season >= 2002:
        return 'pitchfx'
    else:
        return 'classical'


def get_era_for_features(features: List[str]) -> str:
    """
    Determine the appropriate era based on which features are in use.

    If any Statcast features are present, use 'statcast'.
    If any PITCHf/x features are present, use 'pitchfx'.
    Otherwise 'classical'.

    Args:
        features: List of feature names

    Returns:
        Era string
    """
    statcast_features = {'Stuff+', 'Location+', 'Pitching+', 'xERA'}
    pitchfx_features = {'FBv', 'SwStr%', 'CSW%', 'GB%', 'FB%', 'Contact%', 'xFIP', 'SIERA'}

    feature_set = set(features)

    if feature_set & statcast_features:
        return 'statcast'
    elif feature_set & pitchfx_features:
        return 'pitchfx'
    else:
        return 'classical'


# =============================================================================
# AGING CURVE INTEGRATION
# =============================================================================
# Aging curves are integrated ONLY into the prediction pipeline's regression
# prior — NOT as a post-model adjustment, and NOT in training data.
#
# Design rationale:
#   - The LSTM learns from unmodified historical data, preserving natural
#     relationships between Age and performance
#   - At prediction time, the career prior used for regression is age-adjusted:
#     a pitcher's 2022 FIP at age 33 is adjusted to what it implies at age 37
#     before being used as the regression target
#   - This means low-sample seasons in the prediction sequence get pulled
#     toward an age-appropriate baseline, and padding values (for short
#     sequences) reflect expected current talent, not past peak
#   - The LSTM's own output is NOT post-processed with aging deltas, so the
#     model's learned Age signal is trusted as-is for the raw projection
#
# The aging parameters use `decline_per_year_corrected` which accounts for
# survivorship bias (poor performers exit → observed decline is artificially
# low → corrected values are larger and more realistic).
# =============================================================================

import json
from pathlib import Path
from functools import lru_cache

# Path to aging parameters (relative to this file's location)
_AGING_PARAMS_PATH = Path(__file__).parent.parent / 'analysis' / 'aging_parameters.json'


@lru_cache(maxsize=1)
def _load_aging_parameters() -> Dict:
    """
    Load aging parameters from JSON file (cached after first call).

    Returns:
        Full aging parameters dictionary
    """
    if not _AGING_PARAMS_PATH.exists():
        logger.warning(f"Aging parameters file not found: {_AGING_PARAMS_PATH}")
        return {}

    with open(_AGING_PARAMS_PATH, 'r') as f:
        return json.load(f)


def _get_age_band(age: float) -> Optional[str]:
    """
    Map a player's age to the corresponding age band.

    Args:
        age: Player's age (can be float)

    Returns:
        Age band string (e.g., '31-35') or None if out of range
    """
    age = int(age)
    if 21 <= age <= 25:
        return '21-25'
    elif 26 <= age <= 30:
        return '26-30'
    elif 31 <= age <= 35:
        return '31-35'
    elif 36 <= age <= 40:
        return '36-40'
    elif 41 <= age <= 45:
        return '41-45'
    elif age > 45:
        return '41-45'  # Use oldest band for ages 46+
    else:
        return None  # Under 21


def get_aging_delta(
    feature: str,
    age: float,
    model_type: str = 'pitcher',
) -> float:
    """
    Get the expected per-year change in a stat due to aging.

    Returns a SIGNED delta that should be ADDED to the stat value.
    For "inverted" stats (higher = worse, like ERA), the delta is positive
    (ERA increases with age). For normal stats (higher = better, like K%),
    the delta is negative (K% decreases with age).

    Uses `decline_per_year_corrected` which adjusts for survivorship bias.

    Args:
        feature: Stat name (e.g., 'ERA', 'K%', 'FIP')
        age: Player's current age
        model_type: 'pitcher', 'batter', etc.

    Returns:
        Signed delta to add to the stat per year of aging.
        Returns 0.0 if aging data is not available for this stat/age.

    Examples:
        >>> get_aging_delta('ERA', 37)  # ERA gets worse with age
        0.5522  # Add ~0.55 ERA per year at age 37
        >>> get_aging_delta('K%', 37)   # K% gets worse (decreases) with age
        -0.0122  # Subtract ~1.2% K rate per year at age 37
    """
    if feature in SKIP_FEATURES:
        return 0.0

    params = _load_aging_parameters()
    if not params or model_type not in params:
        return 0.0

    model_params = params[model_type]
    if feature not in model_params:
        return 0.0

    stat_params = model_params[feature]
    age_band = _get_age_band(age)
    if age_band is None:
        return 0.0

    band_data = stat_params.get('decline_by_age_band', {}).get(age_band)
    if band_data is None:
        return 0.0

    decline = band_data.get('decline_per_year_corrected')
    if decline is None:
        return 0.0

    is_inverted = stat_params.get('is_inverted', False)

    # For inverted stats (ERA, FIP — higher = worse):
    #   "decline" means getting worse, so the VALUE increases → return +decline
    # For normal stats (K% — higher = better):
    #   "decline" means getting worse, so the VALUE decreases → return -decline
    if is_inverted:
        return decline  # ERA goes up
    else:
        return -decline  # K% goes down


def apply_aging_to_prediction(
    prediction: np.ndarray,
    features: List[str],
    age: float,
    model_type: str = 'pitcher',
) -> np.ndarray:
    """
    Apply one year of aging adjustment to a model prediction vector.

    This is the main entry point for the autoregressive prediction loop.
    After the LSTM generates a raw prediction for year N+1, this function
    applies the expected aging delta for each stat based on the player's
    age in that projected year.

    Args:
        prediction: 1D array of predicted stat values (one per feature)
        features: List of feature names corresponding to prediction indices
        age: Player's age in the PROJECTED year (not current year)
        model_type: 'pitcher', 'batter', etc.

    Returns:
        Age-adjusted prediction array (copy — does not modify input)
    """
    adjusted = prediction.copy()

    for i, feat in enumerate(features):
        delta = get_aging_delta(feat, age, model_type)
        if delta != 0.0:
            adjusted[i] += delta

    return adjusted


def age_adjust_prior_value(
    value: float,
    feature: str,
    from_age: float,
    to_age: float,
    model_type: str = 'pitcher',
) -> float:
    """
    Age-adjust a historical stat value from one age to another.

    Used when computing career priors: if a pitcher had a 2.80 FIP at age 33,
    and we want to know what that talent level implies at age 37, we age-adjust
    by applying 4 years of aging delta.

    This works year-by-year to respect age band boundaries (e.g., aging from
    34 to 37 uses the 31-35 band for age 34 and the 36-40 band for ages 36-37).

    Args:
        value: Original stat value
        feature: Stat name
        from_age: Age when the stat was observed
        to_age: Target age to project the stat to
        model_type: 'pitcher', 'batter', etc.

    Returns:
        Age-adjusted value
    """
    if from_age >= to_age:
        return value  # No adjustment needed for same/younger age

    adjusted = value
    current_age = from_age

    while current_age < to_age:
        delta = get_aging_delta(feature, current_age, model_type)
        adjusted += delta
        current_age += 1.0

    return adjusted


def compute_age_adjusted_career_prior(
    player_data: pd.DataFrame,
    feature: str,
    current_age: float,
    max_seasons: int = PRIOR_MAX_SEASONS,
    model_type: str = 'pitcher',
    recency_halflife: float = 0,
) -> float:
    """
    Compute a volume-weighted career mean with aging adjustment.

    Each historical season's stat value is age-adjusted to what it would
    be at the player's CURRENT age before averaging. This way the prior
    reflects where the player's talent should be NOW, not where it was
    in past years.

    If `recency_halflife` > 0, an exponential decay is applied so that
    more recent seasons receive higher effective weight. A season N years
    ago gets a multiplier of 2^(-N / recency_halflife). This is applied
    on top of volume weighting.

    The weight column is model-type-dependent:
        - pitcher: TBF (total batters faced)
        - batter: PA (plate appearances)
        - baserunning: G (games)
        - defense_*: Inn (innings)

    Example (Chris Sale):
        Age 33: FIP 2.80 → age-adjusted to 37: ~3.86 (4 years × ~0.27/year)
        Age 34: FIP 3.10 → age-adjusted to 37: ~3.91 (3 years × ~0.27/year)
        Age 35: FIP 3.00 → age-adjusted to 37: ~3.53 (2 years × ~0.27/year)
        Age 36: FIP 3.20 → age-adjusted to 37: ~3.60 (1 year  × ~0.40/year)
        Weighted mean ≈ 3.72 (vs 3.03 without aging adjustment)

    Args:
        player_data: All rows for one player, sorted by Season
        feature: The stat column name
        current_age: The player's age to adjust all seasons to
        max_seasons: Maximum number of recent seasons to include
        model_type: 'pitcher', 'batter', 'baserunning', 'fielding_infield', etc.
        recency_halflife: Exponential decay half-life in seasons (0 = no decay)

    Returns:
        Volume-weighted, age-adjusted career mean (or NaN if no valid data)
    """
    weight_col = _get_weight_column(model_type)
    recent = player_data.tail(max_seasons)

    # Determine required columns for validation
    required_cols = [feature, 'Age']
    if weight_col in recent.columns:
        required_cols.append(weight_col)
        valid = recent.dropna(subset=required_cols)
        valid = valid[valid[weight_col] > 0]
    elif model_type == 'pitcher' and 'IP' in recent.columns:
        # Fallback from TBF to IP for pitchers
        weight_col = 'IP'
        required_cols.append(weight_col)
        valid = recent.dropna(subset=required_cols)
        valid = valid[valid[weight_col] > 0]
    else:
        valid = recent.dropna(subset=required_cols)

    if valid.empty:
        return np.nan

    adjusted_values = []
    weights = []

    for _, row in valid.iterrows():
        observed = row[feature]
        obs_age = row['Age']
        adjusted = age_adjust_prior_value(
            observed, feature, obs_age, current_age, model_type
        )
        adjusted_values.append(adjusted)
        if weight_col in row.index and not np.isnan(row.get(weight_col, np.nan)):
            weights.append(row[weight_col])
        else:
            weights.append(1.0)  # equal weight fallback

    weights = np.array(weights, dtype=np.float64)

    # Apply recency decay: multiply volume weights by 2^(-years_ago / halflife)
    if recency_halflife > 0 and 'Season' in valid.columns:
        most_recent_season = valid['Season'].max()
        years_ago = most_recent_season - valid['Season'].values
        decay = np.power(2.0, -years_ago / recency_halflife)
        weights = weights * decay

    return np.average(adjusted_values, weights=weights)
