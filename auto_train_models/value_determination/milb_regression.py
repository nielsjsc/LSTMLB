"""
MiLB Regression Module
======================

Reliability-weighted blending of MLB model predictions with MiLB-based
Major League Equivalencies (MLEs) for batters with small MLB samples.

Players with limited MLB plate appearances have noisy model predictions.
This module regresses those predictions toward their MiLB track record
(translated to MLB equivalents) using per-stat stabilization thresholds.

Pipeline Position: Step 2.25 — after predictions are loaded, before wRC+/WAR.

Formula (per stat):
    mlb_weight = career_mlb_pa / (career_mlb_pa + stabilization_pa)
    blended    = mlb_weight * model_prediction + (1 - mlb_weight) * milb_mle

Once a player reaches ~1200 career PA, mlb_weight ≈ 1.0 for all stats
and MiLB data has negligible influence.
"""

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from value_determination.config import Config

logger = logging.getLogger('value_determination')

# =============================================================================
# CONSTANTS
# =============================================================================

# Per-stat PA needed for the stat to stabilize (reach r=0.5 reliability).
# Reduced from pure-theory values so MiLB influence fades faster —
# players outgrow the regression within ~2 full seasons.
STABILIZATION_PA = {
    'K%':  40,
    'BB%': 80,
    'AVG': 500,
    'OBP': 300,
    'SLG': 200,
    'wOBA': 220,
}

# MLE translation multipliers by level.
# Calibrated against actual MLB performance of players promoted from
# each level.  Previous factors (AAA=0.90) overestimated MLB wOBA by
# ~52 points; these are derived from actual AAA→MLB translation data.
MLE_LEVEL_FACTORS = {
    # Rate stats (AVG, OBP, SLG, wOBA): multiply by this factor
    'rate': {
        'AAA': 0.78,
        'AA':  0.72,
        'A+':  0.66,
        'A':   0.62,
        'A-':  0.58,
    },
    # K% adjustment: add this to MiLB K% (strikeouts increase in MLB)
    'K%': {
        'AAA': 0.03,
        'AA':  0.05,
        'A+':  0.07,
        'A':   0.08,
        'A-':  0.09,
    },
    # BB% adjustment: multiply MiLB BB% by this factor (walks decrease)
    'BB%': {
        'AAA': 0.85,
        'AA':  0.78,
        'A+':  0.72,
        'A':   0.68,
        'A-':  0.65,
    },
}

# Age-appropriate benchmarks for each MiLB level.
# Players older than these ages have inflated stats ("AAAA" effect);
# players younger are more impressive and translate better.
AGE_BENCHMARKS = {
    'AAA': 24.0,
    'AA':  22.0,
    'A+':  21.0,
    'A':   20.0,
    'A-':  19.0,
}

# Per-year age-level adjustment applied to the MLE rate factor.
# Each year ABOVE the benchmark reduces the rate factor by this amount.
# Each year BELOW the benchmark increases the rate factor by this amount.
AGE_ADJUSTMENT_PER_YEAR = 0.025

# Bounds on age adjustment multiplier (guard against extremes)
AGE_ADJ_MIN = 0.85   # oldest-for-level: MLE drops to 85% of base
AGE_ADJ_MAX = 1.15   # youngest-for-level: MLE rises to 115% of base

# Season recency weights for MiLB data (most recent 3 seasons)
MILB_SEASON_WEIGHTS = {0: 0.55, 1: 0.30, 2: 0.15}  # 0 = most recent

# Minimum PA at a level for it to count
MIN_MILB_PA = 50

# Minimum MiLB level to include (exclude rookie/complex leagues for
# players who have upper-minors data)
RELEVANT_LEVELS = {'AAA', 'AA', 'A+', 'A'}

# Career PA threshold above which no regression is applied
FULL_STABILIZATION_PA = 800

# ── Prospect FV Grade Integration ────────────────────────────────────────────
# Derived from multivariate regression on 641 prospects with 200+ MLB PA:
#   career_wOBA = 0.155 + 0.00106*FV + 0.348*MLE_wOBA  (R²=0.241)
# FV adds R² of +0.042 beyond MLE stats alone.
# Per-FV-point adjustment folded into MLE: 0.00106 / 0.348 ≈ 0.003 wOBA.
FV_BASELINE = 50        # FV 50 = "average MLB regular" (scouting baseline)
FV_WOBA_PER_POINT = 0.003  # wOBA adjustment per FV point above baseline

# MiLB data path
MILB_DATA_FILE = Config.Paths.DATA_DIR / 'MiLB' / 'MiLB_Hitters.csv'

# Historic MLB batting data for career PA calculation
HISTORIC_BATTING_FILE = Config.Paths.HISTORIC_MLB_DIR / 'mlb_batting_data_1950_2025.csv'

# Prospect data + ID crosswalk
PROSPECT_DATA_FILE = Config.Paths.PROSPECT_FILE
REGISTER_DATA_DIR = Config.Paths.DATA_DIR / 'register' / 'data'


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_milb_data() -> pd.DataFrame:
    """Load and clean MiLB hitter data."""
    if not MILB_DATA_FILE.exists():
        logger.warning(f"MiLB data not found at {MILB_DATA_FILE}")
        return pd.DataFrame()

    milb = pd.read_csv(MILB_DATA_FILE, low_memory=False)

    # Normalize PlayerId to int for matching with IDfg
    milb['PlayerId'] = pd.to_numeric(milb['PlayerId'], errors='coerce')
    milb = milb.dropna(subset=['PlayerId'])
    milb['PlayerId'] = milb['PlayerId'].astype(int)

    # Keep only relevant levels
    milb = milb[milb['Level'].isin(RELEVANT_LEVELS)].copy()

    # Drop rows with insufficient PA
    milb = milb[milb['PA'] >= MIN_MILB_PA].copy()

    # Ensure numeric stat columns
    for col in ['BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA', 'PA', 'Age']:
        milb[col] = pd.to_numeric(milb[col], errors='coerce')

    milb = milb.dropna(subset=['PA', 'wOBA'])

    return milb


def _load_career_pa() -> pd.Series:
    """Load career MLB PA for each player (IDfg → total PA)."""
    if not HISTORIC_BATTING_FILE.exists():
        logger.warning(f"Historic batting data not found at {HISTORIC_BATTING_FILE}")
        return pd.Series(dtype=float)

    hist = pd.read_csv(HISTORIC_BATTING_FILE, usecols=['IDfg', 'PA'],
                       low_memory=False)
    hist['PA'] = pd.to_numeric(hist['PA'], errors='coerce').fillna(0)
    return hist.groupby('IDfg')['PA'].sum()


def _load_prospect_grades() -> dict:
    """Load peak FV grades for prospects, mapped to FanGraphs IDfg.

    Returns dict {IDfg: peak_fv_grade}.
    """
    if not PROSPECT_DATA_FILE.exists():
        logger.warning(f"Prospect data not found at {PROSPECT_DATA_FILE}")
        return {}

    prospects = pd.read_csv(PROSPECT_DATA_FILE, low_memory=False)

    # Extract mlbam_id from prospect URL (last segment after '-')
    prospects['mlbam_id'] = pd.to_numeric(
        prospects['prospect_url'].str.split('-').str[-1], errors='coerce'
    )
    batters = prospects.dropna(subset=['grade_hit', 'mlbam_id']).copy()
    batters['mlbam_id'] = batters['mlbam_id'].astype(int)

    if batters.empty:
        return {}

    # Build mlbam → IDfg crosswalk from register files
    people_files = glob.glob(str(REGISTER_DATA_DIR / 'people-*.csv'))
    if not people_files:
        logger.warning(f"No register files found in {REGISTER_DATA_DIR}")
        return {}

    xw_dfs = [
        pd.read_csv(f, usecols=['key_mlbam', 'key_fangraphs'], low_memory=False)
        for f in people_files
    ]
    xw = pd.concat(xw_dfs, ignore_index=True).dropna(
        subset=['key_mlbam', 'key_fangraphs']
    )
    xw['key_mlbam'] = xw['key_mlbam'].astype(int)
    xw['key_fangraphs'] = xw['key_fangraphs'].astype(int)
    id_map = dict(zip(xw['key_mlbam'], xw['key_fangraphs']))

    # Map prospects to IDfg
    batters['IDfg'] = batters['mlbam_id'].map(id_map)
    batters = batters.dropna(subset=['IDfg'])
    batters['IDfg'] = batters['IDfg'].astype(int)

    # Peak FV grade per player (highest grade they ever received)
    peak = (
        batters.sort_values('grade_overall', ascending=False)
        .drop_duplicates('IDfg')
        .set_index('IDfg')['grade_overall']
        .to_dict()
    )

    logger.info(f"Loaded FV grades for {len(peak)} prospect batters")
    return peak


# =============================================================================
# MLE CALCULATION
# =============================================================================

def _age_adjustment(age: float, level: str) -> float:
    """Compute age-based multiplier for MLE translation.

    Young-for-level players get a bonus (> 1.0); old-for-level get a
    penalty (< 1.0).  A 20-year-old in AA is 2 years young → 1.05.
    A 27-year-old in AAA is 3 years old → 0.925.
    """
    benchmark = AGE_BENCHMARKS.get(level)
    if benchmark is None or pd.isna(age):
        return 1.0

    years_over = age - benchmark  # positive = old, negative = young
    raw = 1.0 - years_over * AGE_ADJUSTMENT_PER_YEAR
    return max(AGE_ADJ_MIN, min(AGE_ADJ_MAX, raw))


def _translate_to_mle(milb_row: pd.Series) -> dict:
    """Translate a single MiLB season-level row to MLB equivalents.

    Applies level-based translation AND age-based adjustment.
    Young-for-level players get a more generous translation;
    old-for-level (AAAA types) get a harsher one.

    Returns a dict with translated stats: K%, BB%, AVG, OBP, SLG, wOBA.
    """
    level = milb_row['Level']
    age = milb_row.get('Age', None)

    base_rate_factor = MLE_LEVEL_FACTORS['rate'].get(level, 0.66)
    k_adj = MLE_LEVEL_FACTORS['K%'].get(level, 0.06)
    bb_factor = MLE_LEVEL_FACTORS['BB%'].get(level, 0.72)

    # Age adjustment: young-for-level → higher rate factor, old → lower
    age_mult = _age_adjustment(age, level)
    rate_factor = base_rate_factor * age_mult

    # For K%: old-for-level players strike out even more in MLB
    # Invert the age mult so old → higher K% addition, young → lower
    k_age_mult = 2.0 - age_mult  # if age_mult=0.925 → k_age=1.075
    adjusted_k_adj = k_adj * k_age_mult

    return {
        'K%':  min(milb_row['K%'] + adjusted_k_adj, 0.45),
        'BB%': max(milb_row['BB%'] * bb_factor * age_mult, 0.02),
        'AVG': milb_row['AVG'] * rate_factor,
        'OBP': milb_row['OBP'] * rate_factor,
        'SLG': milb_row['SLG'] * rate_factor,
        'wOBA': milb_row['wOBA'] * rate_factor,
    }


def _compute_player_mle(player_milb: pd.DataFrame, current_year: int) -> dict:
    """Compute PA-weighted, recency-weighted MLE for one player.

    Args:
        player_milb: All MiLB rows for one player (already filtered to
                     relevant levels and min PA).
        current_year: The projection year (for recency weighting).

    Returns:
        Dict with MLE stats, or empty dict if no usable data.
    """
    if player_milb.empty:
        return {}

    # Sort by season descending, take most recent 3 seasons
    recent_seasons = sorted(player_milb['Season'].unique(), reverse=True)[:3]

    stats = ['K%', 'BB%', 'AVG', 'OBP', 'SLG', 'wOBA']
    weighted_stats = {s: 0.0 for s in stats}
    total_weight = 0.0

    for recency_idx, season in enumerate(recent_seasons):
        season_data = player_milb[player_milb['Season'] == season]
        recency_weight = MILB_SEASON_WEIGHTS.get(recency_idx, 0.10)

        for _, row in season_data.iterrows():
            mle = _translate_to_mle(row)
            pa = row['PA']
            w = pa * recency_weight

            for stat in stats:
                weighted_stats[stat] += mle[stat] * w
            total_weight += w

    if total_weight == 0:
        return {}

    return {stat: weighted_stats[stat] / total_weight for stat in stats}


def _apply_fv_adjustment(mle: dict, fv_grade: float) -> dict:
    """Adjust MLE rate stats using the prospect's FV scouting grade.

    FV captures true-talent information that MiLB stats miss (e.g. swing
    mechanics, raw power, projection).  The adjustment is derived from
    regression: each FV point above baseline adds ~0.003 wOBA to expected
    MLB production, after controlling for MLE.  Applied as a multiplicative
    scale so AVG/OBP/SLG stay proportionally consistent.

    Only rate stats are adjusted — K% and BB% are left unchanged since
    FV grades capture outcome quality rather than plate discipline.
    """
    fv_delta = fv_grade - FV_BASELINE
    woba_adj = fv_delta * FV_WOBA_PER_POINT

    adjusted = mle.copy()
    base_woba = adjusted.get('wOBA', 0)
    if base_woba <= 0:
        return adjusted

    # Multiplicative scale: preserves relative stat proportions
    mult = (base_woba + woba_adj) / base_woba
    for stat in ('AVG', 'OBP', 'SLG', 'wOBA'):
        if stat in adjusted:
            adjusted[stat] *= mult

    return adjusted


# =============================================================================
# BLENDING
# =============================================================================

def _blend_predictions(
    model_pred: pd.Series,
    mle: dict,
    career_pa: float,
) -> dict:
    """Blend model predictions with MiLB MLE using reliability weights.

    Args:
        model_pred: One row from batter_predictions with model outputs.
        mle: Dict of MLE stats for this player.
        career_pa: Total career MLB PA.

    Returns:
        Dict of blended stat values.
    """
    blended = {}

    for stat, stab_pa in STABILIZATION_PA.items():
        if stat not in model_pred.index or stat not in mle:
            continue

        model_val = model_pred[stat]
        mle_val = mle[stat]

        # Reliability weight: how much to trust the MLB model
        mlb_weight = career_pa / (career_pa + stab_pa)

        blended[stat] = mlb_weight * model_val + (1 - mlb_weight) * mle_val

    return blended


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def apply_milb_regression(batter_data: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """Apply MiLB-based regression to batter predictions.

    For each batter with < FULL_STABILIZATION_PA career MLB PA:
    1. Look up their MiLB stats
    2. Translate MiLB stats to MLB equivalents (MLEs)
    3. Blend model predictions with MLEs using reliability weights

    Args:
        batter_data: DataFrame with model predictions (must have IDfg column
                     and stat columns: K%, BB%, AVG, OBP, SLG, wOBA).
        current_year: The season being projected (for MiLB recency weighting).

    Returns:
        DataFrame with blended predictions (same shape, modified in-place stats).
    """
    milb = _load_milb_data()
    if milb.empty:
        logger.warning("No MiLB data available — skipping regression")
        return batter_data

    career_pa_series = _load_career_pa()
    if career_pa_series.empty:
        logger.warning("No career PA data available — skipping regression")
        return batter_data

    prospect_grades = _load_prospect_grades()

    # Work on a copy
    result = batter_data.copy()

    # Get unique player IDs that need regression
    unique_ids = result['IDfg'].unique()
    candidates = [
        pid for pid in unique_ids
        if career_pa_series.get(pid, 0) < FULL_STABILIZATION_PA
    ]

    if not candidates:
        logger.info("No batters below stabilization threshold — skipping MiLB regression")
        return result

    logger.info(f"Applying MiLB regression to {len(candidates)} batters "
                f"(career PA < {FULL_STABILIZATION_PA})")

    # Pre-compute MLEs for all candidates
    milb_by_player = milb.groupby('PlayerId')
    player_mles = {}
    fv_adjusted_count = 0
    for pid in candidates:
        if pid in milb_by_player.groups:
            player_milb = milb_by_player.get_group(pid)
            mle = _compute_player_mle(player_milb, current_year)
            if mle:
                # Apply FV scouting grade adjustment if available
                if pid in prospect_grades:
                    fv = prospect_grades[pid]
                    mle = _apply_fv_adjustment(mle, fv)
                    fv_adjusted_count += 1
                player_mles[pid] = mle

    if fv_adjusted_count > 0:
        logger.info(f"Applied FV grade adjustments to {fv_adjusted_count} / "
                    f"{len(player_mles)} MLEs")

    logger.info(f"Computed MLEs for {len(player_mles)} batters with MiLB data")

    # Apply blending
    stats_to_blend = list(STABILIZATION_PA.keys())
    adjusted_count = 0

    for pid, mle in player_mles.items():
        career_pa = career_pa_series.get(pid, 0)
        mask = result['IDfg'] == pid

        if not mask.any():
            continue

        # Blend EACH projection year individually so the LSTM's
        # year-over-year progression is preserved.
        first_row = True
        for idx in result.index[mask]:
            row = result.loc[idx]
            blended = _blend_predictions(row, mle, career_pa)
            if not blended:
                continue

            for stat, val in blended.items():
                result.at[idx, stat] = val

            if first_row:
                adjusted_count += 1
                # Log notable adjustments (first year only)
                mlb_weight = career_pa / (career_pa + STABILIZATION_PA.get('wOBA', 220))
                if career_pa < 300:
                    name = row.get('Name', f'IDfg={pid}')
                    old_woba = row.get('wOBA', 0)
                    new_woba = blended.get('wOBA', old_woba)
                    logger.debug(
                        f"  {name}: PA={career_pa:.0f}, mlb_wt={mlb_weight:.2f}, "
                        f"wOBA {old_woba:.3f} → {new_woba:.3f} "
                        f"(MLE={mle.get('wOBA', 0):.3f})"
                    )
                first_row = False

    logger.info(f"MiLB regression applied to {adjusted_count} batters")

    # Summary statistics
    if adjusted_count > 0:
        orig_woba = batter_data.loc[
            batter_data['IDfg'].isin(player_mles.keys())
        ]['wOBA'].mean()
        new_woba = result.loc[
            result['IDfg'].isin(player_mles.keys())
        ]['wOBA'].mean()
        logger.info(
            f"Average wOBA change for regressed batters: "
            f"{orig_woba:.3f} → {new_woba:.3f} ({new_woba - orig_woba:+.3f})"
        )

    return result
