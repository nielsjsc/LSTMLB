"""
Reliability Regression for Pitcher Statistics
==============================================

Implements partial regression to the mean based on sample size, following
the same methodology used by professional projection systems (Steamer, ZiPS, PECOTA).

The core formula (James-Stein / Bayesian shrinkage):

    true_talent = (observed * n + prior * n0) / (n + n0)

Where:
    observed = raw observed rate stat for a season
    n        = batters faced (TBF) in that season
    prior    = player's IP-weighted career mean (or league avg for rookies)
    n0       = stabilization point (stat-specific, in BF)

Stabilization points represent the number of batters faced required for a
stat to become 50% signal / 50% noise. These are well-established from
FanGraphs research (Russell Carleton, Tom Tango, et al).

Usage:
    # In training pipeline (operates on full DataFrame):
    df = regress_pitcher_stats(df, features, era='modern')

    # In prediction pipeline (operates on per-player sequence):
    sequence = regress_player_sequence(player_data, features, era='modern')
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Literal

logger = logging.getLogger(__name__)

# =============================================================================
# STABILIZATION POINTS (in Batters Faced)
# =============================================================================
# Source: FanGraphs "How Long Does It Take To Stabilize?" research
# (Russell Carleton, Pizza Cutter, Tom Tango)
#
# These represent the BF needed for a stat to be 50% true talent / 50% noise.
# Lower = stabilizes faster = more trustworthy at small samples.
#
# Three eras are defined because pitch-tracking data availability changed
# the reliability of certain metrics across time periods.

STABILIZATION_POINTS: Dict[str, Dict[str, int]] = {
    # Classical stats (available 1950+)
    # K% and BB% are the fastest-stabilizing counting-derived rates
    'K%':       {'classical': 70,   'pitchfx': 70,   'statcast': 70},
    'BB%':      {'classical': 170,  'pitchfx': 170,  'statcast': 170},
    'ERA':      {'classical': 1320, 'pitchfx': 1320, 'statcast': 1320},
    # FIP: True analytical stabilization is ~740 BF, but we match ERA's rate
    # (1320 BF) so that ERA and FIP regress in lockstep. Otherwise a 75 IP
    # pitcher gets FIP barely regressed but ERA heavily regressed, creating
    # unrealistic divergence between the two run estimators.
    'FIP':      {'classical': 1320, 'pitchfx': 1320, 'statcast': 1320},

    # PITCHf/x era stats (2002+)
    'FBv':      {'pitchfx': 50,   'statcast': 50},    # Physical measure, very stable
    'SwStr%':   {'pitchfx': 200,  'statcast': 200},
    'CSW%':     {'pitchfx': 250,  'statcast': 250},
    'GB%':      {'pitchfx': 70,   'statcast': 70},     # Batted ball profile, fast
    'FB%':      {'pitchfx': 70,   'statcast': 70},
    'Contact%': {'pitchfx': 100,  'statcast': 100},
    # xFIP and SIERA also matched to ERA's rate for consistency
    'xFIP':     {'pitchfx': 1320, 'statcast': 1320},
    'SIERA':    {'pitchfx': 1320, 'statcast': 1320},

    # Statcast era stats (2020+)
    # Stuff+ stabilizes incredibly fast — 80 pitches ≈ ~20 BF
    'Stuff+':    {'statcast': 20},
    'Location+': {'statcast': 100},
    'Pitching+': {'statcast': 60},
    'xERA':      {'statcast': 1320},
}

# Maximum number of recent seasons to use when computing the career prior.
# Prevents regression toward long-ago peak years (e.g., Kershaw 2011-2017
# inflating his 2025 prior). A 4-season window captures the player's
# current talent level while still smoothing year-to-year noise.
PRIOR_MAX_SEASONS = 4

# Features where regression should NOT be applied
# Age is not a rate stat. IP is a volume stat, not a rate.
SKIP_FEATURES = {'Age', 'IP'}

# Approximate BF per IP (used when TBF is not available)
BF_PER_IP = 4.3




def _get_stabilization_point(feature: str, era: str) -> Optional[int]:
    """
    Get the stabilization point for a feature in a given era.

    Args:
        feature: Stat name (e.g., 'K%', 'ERA')
        era: One of 'classical', 'pitchfx', 'statcast'

    Returns:
        Stabilization point in BF, or None if not applicable
    """
    if feature in SKIP_FEATURES:
        return None
    if feature not in STABILIZATION_POINTS:
        return None

    era_points = STABILIZATION_POINTS[feature]

    # Try exact era, then fall back to nearest available
    if era in era_points:
        return era_points[era]

    # Fallback order: statcast -> pitchfx -> classical
    for fallback in ['statcast', 'pitchfx', 'classical']:
        if fallback in era_points:
            return era_points[fallback]

    return None


def _compute_career_prior(
    player_data: pd.DataFrame,
    feature: str,
    current_season: int,
    max_seasons: int = PRIOR_MAX_SEASONS,
    current_age: Optional[float] = None,
    model_type: str = 'pitcher',
) -> float:
    """
    Compute an IP-weighted recent career mean for a single feature.

    Uses only the most recent `max_seasons` seasons up to and including
    current_season. This prevents regression toward long-ago peak years
    for aging players (e.g., Kershaw's 2025 won't regress toward his
    2014 peak, only toward his 2022-2025 level).

    If `current_age` is provided, each historical season's value is
    age-adjusted to what it would be at `current_age` before averaging.
    This ensures the prior reflects the player's expected talent NOW,
    not what it was at a younger age.

    Args:
        player_data: All rows for one player, sorted by Season
        feature: The stat column name
        current_season: The season being regressed
        max_seasons: Maximum number of recent seasons to include in the prior
        current_age: If provided, age-adjust each season to this age
        model_type: 'pitcher', 'batter', etc. (used for aging curves)

    Returns:
        IP-weighted recent career mean (optionally age-adjusted), or NaN
    """
    career = player_data[player_data['Season'] <= current_season].copy()

    # Limit to most recent max_seasons
    if len(career) > max_seasons:
        career = career.tail(max_seasons)

    # Drop rows where the feature or IP is missing
    valid = career.dropna(subset=[feature, 'IP'])
    valid = valid[valid['IP'] > 0]

    if valid.empty:
        return np.nan

    weights = valid['IP'].values
    values = valid[feature].values

    # Age-adjust if current_age is provided and Age column exists
    if current_age is not None and 'Age' in valid.columns:
        adjusted_values = []
        for _, row in valid.iterrows():
            obs_age = row['Age']
            adjusted = age_adjust_prior_value(
                row[feature], feature, obs_age, current_age, model_type
            )
            adjusted_values.append(adjusted)
        values = np.array(adjusted_values)

    return np.average(values, weights=weights)


def _compute_league_prior(
    full_df: pd.DataFrame,
    feature: str,
    season: int,
    window: int = 3,
) -> float:
    """
    Compute a league-average prior for a feature using a rolling window.

    Uses IP-weighted average over the last `window` seasons to smooth out
    year-to-year noise while adapting to era shifts.

    Args:
        full_df: Full historical DataFrame (all players)
        feature: The stat column name
        season: The season to compute the prior for
        window: Number of seasons to average over

    Returns:
        IP-weighted league average for the feature
    """
    recent = full_df[
        (full_df['Season'] >= season - window + 1) &
        (full_df['Season'] <= season)
    ]

    valid = recent.dropna(subset=[feature, 'IP'])
    valid = valid[valid['IP'] > 0]

    if valid.empty:
        # Broader fallback
        valid = full_df.dropna(subset=[feature, 'IP'])
        valid = valid[valid['IP'] > 0]

    if valid.empty:
        return np.nan

    return np.average(valid[feature].values, weights=valid['IP'].values)


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
    Apply reliability regression to all pitcher stats in a DataFrame.

    This is the main entry point for the TRAINING pipeline. It operates on
    the full DataFrame after filtering but BEFORE scaling.

    For each player-season:
    1. Compute the player's IP-weighted career mean up to that season (prior)
    2. If the player has fewer than `min_career_seasons`, use league average
    3. Regress each rate stat toward the prior based on BF and stabilization rate

    Args:
        df: Filtered pitcher DataFrame with Season, IDfg, IP, TBF columns
        features: List of model input features to regress
        era: Data era for stabilization points ('classical', 'pitchfx', 'statcast')
        league_df: Full DataFrame for computing league priors (defaults to df itself)
        min_career_seasons: Minimum career seasons before using career prior
                           (below this, league average is used)

    Returns:
        DataFrame with regressed stat values (original columns overwritten)
    """
    if league_df is None:
        league_df = df

    # Identify which features actually need regression
    regressable = []
    stab_points = {}
    for feat in features:
        n0 = _get_stabilization_point(feat, era)
        if n0 is not None:
            regressable.append(feat)
            stab_points[feat] = n0

    if not regressable:
        logger.info("No regressable features found — skipping reliability regression")
        return df

    logger.info(f"Applying reliability regression to {len(regressable)} features: {regressable}")
    logger.info(f"Era: {era}, min career seasons for career prior: {min_career_seasons}")

    # Pre-compute league priors for each (feature, season) combination
    seasons = df['Season'].unique()
    league_priors: Dict[str, Dict[int, float]] = {}
    for feat in regressable:
        league_priors[feat] = {}
        for season in seasons:
            league_priors[feat][season] = _compute_league_prior(league_df, feat, season)

    # Ensure TBF column exists; estimate from IP if needed
    if 'TBF' not in df.columns:
        logger.warning("TBF column not found — estimating from IP * 4.3")
        df['TBF'] = df['IP'] * BF_PER_IP

    df = df.copy()
    df = df.sort_values(['IDfg', 'Season'])

    regressed_count = 0
    total_count = 0

    for player_id, player_group in df.groupby('IDfg'):
        player_idx = player_group.index
        player_seasons = player_group['Season'].values
        n_seasons_so_far = 0

        for i, (idx, row) in enumerate(player_group.iterrows()):
            season = row['Season']
            bf = row['TBF'] if not np.isnan(row.get('TBF', np.nan)) else row['IP'] * BF_PER_IP
            n_seasons_so_far = i + 1

            for feat in regressable:
                observed = row[feat]
                if np.isnan(observed):
                    continue

                n0 = stab_points[feat]
                total_count += 1

                # Choose prior: career mean if enough history, else league average
                # NOTE: Training pipeline uses RAW career prior (no aging adjustment).
                # Aging curves are only applied in the prediction pipeline's regression
                # prior, so the model learns from unmodified historical relationships.
                if n_seasons_so_far >= min_career_seasons:
                    prior = _compute_career_prior(player_group, feat, season)
                    if np.isnan(prior):
                        prior = league_priors[feat].get(season, observed)
                else:
                    prior = league_priors[feat].get(season, observed)

                regressed = regress_single_value(observed, bf, prior, n0)
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
    era: str = 'classical',
    league_priors: Optional[Dict[str, float]] = None,
    min_career_seasons: int = 2,
) -> pd.DataFrame:
    """
    Apply reliability regression to a single player's historical data.

    This is the entry point for the PREDICTION pipeline. It operates on
    one player's data before sequence construction.

    The key difference from training: we compute the career prior using
    ALL available seasons (not just up-to-season), since at prediction time
    we want the best possible estimate of the player's true talent.

    Args:
        player_data: One player's historical DataFrame, sorted by Season
        features: List of model input features to regress
        era: Data era for stabilization points
        league_priors: Pre-computed league averages per feature.
                      If None, uses the player's own data (less accurate for rookies)
        min_career_seasons: Minimum seasons for career prior vs league prior

    Returns:
        DataFrame with regressed values (copy — does not modify input)
    """
    if len(player_data) == 0:
        return player_data.copy()

    result = player_data.copy()

    # Ensure TBF exists
    if 'TBF' not in result.columns:
        result['TBF'] = result['IP'] * BF_PER_IP

    # Identify regressable features
    regressable = []
    stab_points = {}
    for feat in features:
        n0 = _get_stabilization_point(feat, era)
        if n0 is not None and feat in result.columns:
            regressable.append(feat)
            stab_points[feat] = n0

    if not regressable:
        return result

    n_total_seasons = len(result)

    # Compute age-adjusted IP-weighted means using only the most recent
    # PRIOR_MAX_SEASONS. Each historical season is age-adjusted to the
    # player's CURRENT age so the prior reflects where talent is NOW,
    # not where it was at a younger age.
    recent_data = result.tail(PRIOR_MAX_SEASONS)
    current_age = result['Age'].iloc[-1] if 'Age' in result.columns else None
    career_means: Dict[str, float] = {}
    for feat in regressable:
        if current_age is not None and 'Age' in recent_data.columns:
            # Use age-adjusted career prior
            career_means[feat] = compute_age_adjusted_career_prior(
                result, feat, current_age,
                max_seasons=PRIOR_MAX_SEASONS, model_type='pitcher'
            )
        else:
            # Fallback: raw IP-weighted mean
            valid = recent_data.dropna(subset=[feat, 'IP'])
            valid = valid[valid['IP'] > 0]
            if not valid.empty:
                career_means[feat] = np.average(
                    valid[feat].values, weights=valid['IP'].values
                )
            else:
                career_means[feat] = np.nan

    # Regress each season
    for idx, row in result.iterrows():
        bf = row['TBF'] if not np.isnan(row.get('TBF', np.nan)) else row['IP'] * BF_PER_IP

        for feat in regressable:
            observed = row[feat]
            if np.isnan(observed):
                continue

            n0 = stab_points[feat]

            # Choose prior
            if n_total_seasons >= min_career_seasons and not np.isnan(career_means[feat]):
                prior = career_means[feat]
            elif league_priors and feat in league_priors:
                prior = league_priors[feat]
            else:
                # Last resort: don't regress (keep observed)
                continue

            result.at[idx, feat] = regress_single_value(observed, bf, prior, n0)

    return result


def compute_regressed_career_mean(
    player_data: pd.DataFrame,
    features: List[str],
    era: str = 'classical',
    league_priors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute the best available "true talent" estimate for padding.

    This produces a single set of feature values representing the player's
    true talent level — used for padding when the sequence is shorter than
    seq_length.

    Unlike raw career averages, this applies regression to each season first,
    then takes the IP-weighted mean of the regressed values. This prevents
    small-sample seasons from distorting the padding values.

    Args:
        player_data: One player's historical DataFrame
        features: List of features
        era: Data era
        league_priors: Pre-computed league averages

    Returns:
        Dict mapping feature name to regressed career mean
    """
    if len(player_data) == 0:
        return {}

    # First regress the player's data
    regressed = regress_player_sequence(
        player_data, features, era=era, league_priors=league_priors
    )

    result = {}
    for feat in features:
        if feat in SKIP_FEATURES:
            # For Age, use latest
            if feat == 'Age':
                result[feat] = regressed['Age'].iloc[-1]
            continue

        # Use only the most recent seasons for the career mean (same window
        # as the prior itself) so padding reflects current talent, not peak.
        recent = regressed.tail(PRIOR_MAX_SEASONS)
        valid = recent.dropna(subset=[feat])
        valid_with_ip = valid[valid['IP'] > 0] if 'IP' in valid.columns else valid

        if not valid_with_ip.empty and 'IP' in valid_with_ip.columns:
            result[feat] = np.average(
                valid_with_ip[feat].values, weights=valid_with_ip['IP'].values
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
        season: Target season (uses rolling window ending here)
        window: Number of seasons to average

    Returns:
        Dict mapping feature name to league average
    """
    priors = {}
    for feat in features:
        if feat in SKIP_FEATURES or feat not in df.columns:
            continue
        if season is not None:
            priors[feat] = _compute_league_prior(df, feat, season, window)
        else:
            # Use all data
            valid = df.dropna(subset=[feat, 'IP'])
            valid = valid[valid['IP'] > 0]
            if not valid.empty:
                priors[feat] = np.average(valid[feat].values, weights=valid['IP'].values)

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
) -> float:
    """
    Compute an IP-weighted career mean with aging adjustment.

    Each historical season's stat value is age-adjusted to what it would
    be at the player's CURRENT age before averaging. This way the prior
    reflects where the player's talent should be NOW, not where it was
    in past years.

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
        model_type: 'pitcher', 'batter', etc.

    Returns:
        IP-weighted, age-adjusted career mean (or NaN if no valid data)
    """
    recent = player_data.tail(max_seasons)

    valid = recent.dropna(subset=[feature, 'IP', 'Age'])
    valid = valid[valid['IP'] > 0]

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
        weights.append(row['IP'])

    return np.average(adjusted_values, weights=weights)
