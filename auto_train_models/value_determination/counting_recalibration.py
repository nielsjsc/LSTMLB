"""
Counting Stat Recalibration — Derive Components from wOBA
==========================================================

The LSTM's rate stat predictions (wOBA, SLG, OBP, BB%, K%) are well-calibrated,
but it systematically mean-regresses counting stats (HR, 2B, 3B, RBI, R, HBP)
toward the training population average.  A 30-HR hitter might project at 23 HR
even though his wOBA correctly implies ~28–30 HR given his hitting profile.

This module replaces the model's deflated counting stats with values derived
from each player's *own* career counting profile, scaled by the model's
projected quality level (predicted_wOBA / career_wOBA).

Formula for each counting stat:
    ratio          = predicted_wOBA / career_wOBA       (clipped to [0.5, 1.5])
    derived_count  = career_count_per150 × ratio
    final          = blend × derived_count  +  (1 - blend) × model_prediction
    blend          = min(career_PA / PA_FULL_WEIGHT, 1.0)

    - 'blend' gives full trust to career profiles for established players
      (>1500 PA) and partially trusts the model for young players with fewer
      career plate appearances.
    - The ratio clip prevents extreme extrapolation when the model projects a
      dramatic quality change (e.g., prospect breakout or aging cliff).

Key design choices:
    - Each player retains their OWN HR/2B/3B/RBI/R mix (José Ramírez is
      HR+2B heavy, Bobby Witt Jr. is 2B+3B heavy, Cal Raleigh is HR-dominant).
    - wOBA is NEVER modified — it is the source of truth.
    - This is the inverse of CALCULATE_WOBA_FROM_COMPONENTS: instead of
      deriving wOBA from counting stats, we derive counting stats from wOBA.

Usage in the value determination pipeline (main.py):
    from .counting_recalibration import recalibrate_batter_counting_stats
    batter_data = recalibrate_batter_counting_stats(batter_data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger('value_determination')

# Counting stats eligible for recalibration (per-150 games in predictions CSV)
COUNTING_STATS = ['HR', '2B', '3B', 'RBI', 'R', 'HBP']


# =============================================================================
# HISTORICAL DATA LOADING (cached)
# =============================================================================

_historical_cache: Optional[pd.DataFrame] = None


def _load_historical_batting() -> pd.DataFrame:
    """Load the full historical batting CSV (cached after first call)."""
    global _historical_cache
    if _historical_cache is not None:
        return _historical_cache

    from .config import Config
    hist_file = Config.Paths.HISTORIC_MLB_DIR / 'mlb_batting_data_1950_2025.csv'
    if not hist_file.exists():
        logger.warning(f"Historical batting file not found: {hist_file}")
        _historical_cache = pd.DataFrame()
        return _historical_cache

    _historical_cache = pd.read_csv(hist_file, low_memory=False)
    logger.info(f"Loaded historical batting data: {len(_historical_cache)} rows")
    return _historical_cache


# =============================================================================
# CAREER PROFILE CONSTRUCTION
# =============================================================================

def build_career_profiles(
    historical_df: pd.DataFrame,
    n_recent: int = 3,
    min_pa: int = 50,
) -> Dict[int, Dict]:
    """
    Build PA-weighted career counting profiles for each player.

    For each player, computes the PA-weighted average of their most recent
    ``n_recent`` qualifying seasons (PA ≥ ``min_pa``).  Counting stats are
    converted from raw totals to per-150-games rates.

    Args:
        historical_df: Full historical batting DataFrame (raw counting stats).
        n_recent: Number of most-recent qualifying seasons to include.
        min_pa: Minimum PA for a season to qualify.

    Returns:
        Dict mapping ``IDfg`` → profile dict with keys:
            ``base_woba``:   PA-weighted recent wOBA
            ``base_counts``: {stat: per-150 rate} for each counting stat
            ``career_pa``:   Total career PA across all qualifying seasons
    """
    profiles: Dict[int, Dict] = {}

    required_cols = {'IDfg', 'Season', 'PA', 'wOBA', 'G'}
    if not required_cols.issubset(historical_df.columns):
        missing = required_cols - set(historical_df.columns)
        logger.warning(f"Historical data missing columns for career profiles: {missing}")
        return profiles

    available_counting = [s for s in COUNTING_STATS if s in historical_df.columns]
    df = historical_df[historical_df['PA'] >= min_pa].copy()

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

        # Baseline wOBA (actual, not x-stat adjusted)
        woba_vals = recent['wOBA'].values
        if pd.isna(woba_vals).all():
            continue
        woba_vals = np.nan_to_num(woba_vals, nan=0.0)
        base_woba = float(np.average(woba_vals, weights=weights))
        if base_woba < 0.15:
            continue

        # Per-150 counting stats
        games = recent['G'].values.astype(float)
        base_counts: Dict[str, float] = {}
        for stat in available_counting:
            raw_vals = np.nan_to_num(recent[stat].values.astype(float), nan=0.0)
            # Convert each season's raw total to per-150-games rate
            per_150_vals = np.where(games > 0, raw_vals * (150.0 / games), 0.0)
            base_counts[stat] = float(np.average(per_150_vals, weights=weights))

        career_pa = float(group['PA'].sum())

        profiles[int(player_id)] = {
            'base_woba': base_woba,
            'base_counts': base_counts,
            'career_pa': career_pa,
        }

    logger.info(f"Built career counting profiles for {len(profiles)} players")
    return profiles


# =============================================================================
# RECALIBRATION — ENTRY POINT
# =============================================================================

def recalibrate_batter_counting_stats(batter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive counting stats from the model's predicted wOBA × each player's
    career counting profile.

    Reads configuration from ``BatterConfig``:
        ``CALCULATE_COMPONENTS_FROM_WOBA``  — master toggle (default False)
        ``COMPONENTS_FROM_WOBA_PA_WEIGHT``  — PA for full career-profile trust
        ``COMPONENTS_FROM_WOBA_RECENT_SEASONS`` — seasons for career average

    The model's wOBA is treated as the source of truth.  For each player, a
    quality ratio (predicted_wOBA / career_wOBA) scales their career counting
    profile to produce rate-consistent counting stats.  Players without a
    career profile are left unchanged.

    Args:
        batter_df: DataFrame with raw LSTM predictions.  Must contain ``wOBA``,
            ``IDfg``, and at least some of HR/2B/3B/RBI/R/HBP (per 150 games).

    Returns:
        DataFrame with recalibrated counting stats (wOBA unchanged).
    """
    # ── Load config ──────────────────────────────────────────────────────
    try:
        try:
            from ..configs.batter_config import BatterConfig
        except (ImportError, ValueError):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from configs.batter_config import BatterConfig

        enabled = getattr(BatterConfig, 'CALCULATE_COMPONENTS_FROM_WOBA', False)
        pa_full = getattr(BatterConfig, 'COMPONENTS_FROM_WOBA_PA_WEIGHT', 1500.0)
        n_recent = getattr(BatterConfig, 'COMPONENTS_FROM_WOBA_RECENT_SEASONS', 3)
    except (ImportError, AttributeError) as e:
        logger.info(f"Counting recalibration disabled (config load error: {e})")
        return batter_df

    if not enabled:
        logger.info("CALCULATE_COMPONENTS_FROM_WOBA is disabled — counting stats unchanged.")
        return batter_df

    # ── Validate input ───────────────────────────────────────────────────
    available_stats = [s for s in COUNTING_STATS if s in batter_df.columns]
    if 'wOBA' not in batter_df.columns or not available_stats:
        logger.warning("Cannot recalibrate: missing wOBA or counting stat columns")
        return batter_df

    # ── Build career profiles from historical data ───────────────────────
    hist_df = _load_historical_batting()
    if hist_df.empty:
        logger.warning("No historical data — counting stats unchanged")
        return batter_df

    career_profiles = build_career_profiles(hist_df, n_recent=n_recent, min_pa=50)
    if not career_profiles:
        logger.warning("No career profiles built — counting stats unchanged")
        return batter_df

    # ── Apply recalibration ──────────────────────────────────────────────
    df = batter_df.copy()
    n_recalibrated = 0
    n_total = len(df)

    for idx, row in df.iterrows():
        player_id = int(row['IDfg'])
        profile = career_profiles.get(player_id)
        if profile is None:
            continue

        pred_woba = row['wOBA']
        base_woba = profile['base_woba']

        if base_woba < 0.15 or pd.isna(pred_woba):
            continue

        # Quality ratio: how much better/worse the model projects vs career
        ratio = pred_woba / base_woba
        # Safety clip: prevent extreme extrapolation for breakouts or collapses
        ratio = max(0.50, min(1.50, ratio))

        # Blend weight: trust career profile more for established players
        blend = min(profile['career_pa'] / pa_full, 1.0)

        for stat in available_stats:
            if stat not in profile['base_counts']:
                continue
            career_rate = profile['base_counts'][stat]
            model_pred = row[stat]

            # Derived: career counting rate × quality ratio
            derived = career_rate * ratio

            # Blend career-derived value with model prediction
            recalibrated = blend * derived + (1.0 - blend) * model_pred
            df.at[idx, stat] = max(0.0, recalibrated)

        n_recalibrated += 1

    logger.info(
        f"Recalibrated counting stats for {n_recalibrated}/{n_total} player-year rows "
        f"({len(available_stats)} stats: {available_stats})"
    )
    return df
