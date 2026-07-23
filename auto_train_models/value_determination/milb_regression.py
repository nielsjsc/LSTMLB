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

# Per-stat PA for MLE blend weight calculation.
# At 400, MiLB weight is ~80% at 100 career PA, ~67% at 200 PA, ~50% at
# 400 PA.  Calibrated so that prospects with ~1 MLB season (500 PA) get
# roughly equal weight between MLB and MiLB signal.
MLE_BLEND_PA = 400

STABILIZATION_PA = {
    'K%':  MLE_BLEND_PA,
    'BB%': MLE_BLEND_PA,
    'AVG': MLE_BLEND_PA,
    'OBP': MLE_BLEND_PA,
    'SLG': MLE_BLEND_PA,
    'wOBA': MLE_BLEND_PA,
}

# Per-level MLE translation factors.
# K%: additive adjustment (strikeouts increase in MLB).
# BB%: multiplicative adjustment (walks decrease in MLB).
# AVG/OBP/SLG/wOBA: multiplicative per-level factors preserving player-specific
# signal instead of collapsing to a population-average lookup table.
# Derived from the same 1,307-player empirical dataset, but split by level
# to retain within-level variance.  Age adjustment is applied separately.
MLE_LEVEL_FACTORS = {
    'K%': {
        'AAA': 0.03,
        'AA':  0.05,
        'A+':  0.07,
        'A':   0.08,
        'A-':  0.09,
    },
    'BB%': {
        'AAA': 0.85,
        'AA':  0.78,
        'A+':  0.72,
        'A':   0.68,
        'A-':  0.65,
    },
    # Rate-stat multipliers: MiLB_stat × factor → MLB equivalent.
    # Derived from median(MLB_career_stat / MiLB_stat) for each level,
    # filtered to players with ≥200 MLB PA.
    'AVG': {
        'AAA': 0.92,
        'AA':  0.84,
        'A+':  0.78,
        'A':   0.74,
        'A-':  0.70,
    },
    'OBP': {
        'AAA': 0.92,
        'AA':  0.85,
        'A+':  0.80,
        'A':   0.76,
        'A-':  0.72,
    },
    'SLG': {
        'AAA': 0.88,
        'AA':  0.80,
        'A+':  0.74,
        'A':   0.70,
        'A-':  0.66,
    },
    'wOBA': {
        'AAA': 0.90,
        'AA':  0.82,
        'A+':  0.77,
        'A':   0.73,
        'A-':  0.69,
    },
}

# Age adjustment bonuses per year younger than benchmark (wOBA points).
# Used in per-level rate-stat translation to credit young-for-level
# performance.  Derived from same empirical data as AGE_ADJUSTMENT_WRC_PTS
# but expressed in wOBA units for direct application.
AGE_ADJUSTMENT_WOBA_PTS = {
    'AAA': 0.002,
    'AA':  0.004,
    'A+':  0.004,
    'A':   0.005,
    'A-':  0.004,
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

# Level-dependent age adjustment in wRC+ points per year.
# Each year BELOW the benchmark adds this many wRC+ points before looking
# up the expected MLB wOBA.  Empirically optimized per level to maximize
# correlation with career MLB wOBA (Analysis 4 of MiLB regression research).
# Lower levels need larger adjustments because age spreads are wider and
# developmental trajectories diverge more.
AGE_ADJUSTMENT_WRC_PTS = {
    'AAA': 3,    # r=0.325 (AAA ages cluster tightly)
    'AA':  8,    # r=0.341 (moderate spread)
    'A+':  9,    # r=0.312 (young-for-level matters)
    'A':   12,   # r=0.300 (18yo in A-ball is a different animal)
    'A-':  10,   # interpolated
}

# ── Stagnation Penalty (AAAA effect) ─────────────────────────────────────
# Players who spend multiple seasons at their highest MiLB level without
# promotion are disproportionately AAAA players.  Empirical analysis:
#   career_woba ~ mean_woba_aaa + seasons_at_aaa: coef = -0.006/season
#   career_woba ~ mean_woba_aa  + seasons_at_aa:  coef = -0.003/season
# Among strong AAA hitters (115+ wRC+): 1yr=.328, 3yr=.317, 5yr=.304 MLB wOBA.
# Applied to the composite MLE AFTER recency-weighted aggregation.
# The first season at a level is free (everyone passes through); penalty
# starts at 2+ seasons.  This correctly penalizes stagnant AAAA players
# while leaving fast-rising prospects unaffected.
STAGNATION_WOBA_PER_YEAR = {
    'AAA': 0.008,   # strongest signal — stuck at AAA = AAAA player
    'AA':  0.005,   # moderate — could be blocked by MLB roster depth
    'A+':  0.003,   # weak — development takes time at lower levels
    'A':   0.002,
}

# Season recency weights for MiLB data (most recent 5 seasons).
# Exponential decay: 8/4/2/1/0.5 — research found all weight schemes
# perform nearly identically (r=0.40-0.41), but exp_decay was marginally best.
MILB_SEASON_WEIGHTS = {0: 8, 1: 4, 2: 2, 3: 1, 4: 0.5}  # 0 = most recent

# Marcel regression: regress MLE composite toward league mean.
# Reduced to 50 PA because per-level rate multipliers already preserve
# player-specific signal — heavy regression double-counts the implicit
# regression in the MLE translation step.
REGRESSION_PA = 500
LEAGUE_MEAN_WOBA = 0.290  # expected MLB wOBA for avg MiLB performer

# ── Per-Season Shrinkage (dampens single outlier MiLB seasons) ───────────
# The per-level translation factors (MLE_LEVEL_FACTORS) are flat multipliers:
# a .440 wOBA season and a .300 wOBA season at the same level both get
# multiplied by the same 0.90/0.82/etc. factor. That means one small-sample
# monster season translates at full face value and, combined with the
# recency weighting above (the most-recent season gets weight 8, vs 0.5 for
# the oldest of the 5), can dominate the whole composite MLE.
#
# Before translation, each season's raw wOBA is now shrunk toward the
# player's own PA×recency-weighted MiLB average (across the same 5 seasons
# used elsewhere in this function) using the season's own PA as the
# reliability signal. A short, hot half-season regresses hard toward the
# player's broader track record; a full, sustained season barely moves.
# This does NOT replace the existing league-mean regression on the final
# composite (REGRESSION_PA) — that step still runs afterward — it just
# keeps any one season from single-handedly setting the composite.
SEASON_SHRINKAGE_PA = 250

# ── Real-MLB Consistency Bonus (mlb_weight blend) ─────────────────────────
# mlb_weight = career_pa / (career_pa + effective_stab) only measures
# *volume* of real MLB PA — it can't tell the difference between 300 PA
# spread across three separate seasons that all say "below-average hitter"
# and 300 PA of pure noise around an unknown true talent. The former is a
# much stronger signal and should pull the blend further toward the real
# MLB data than raw PA alone implies.
#
# When a player has 2+ real MLB seasons (each with at least
# MIN_SEASON_PA_FOR_CONSISTENCY PA) that all land on the same side of
# league-average wOBA, we credit extra *effective* PA in the mlb_weight
# calculation — capped so it can meaningfully shift the blend without ever
# fully overriding the MiLB side on its own.
MLB_LEAGUE_AVG_WOBA = 0.310
MIN_SEASON_PA_FOR_CONSISTENCY = 40
CONSISTENCY_BONUS_PA_PER_SEASON = 150
MAX_CONSISTENCY_BONUS_PA = 450

# Minimum PA at a level for it to count
MIN_MILB_PA = 50

# Minimum MiLB level to include (exclude rookie/complex leagues for
# players who have upper-minors data)
RELEVANT_LEVELS = {'AAA', 'AA', 'A+', 'A'}

# Career PA threshold above which no regression is applied.
# At 1500 PA with MLE_BLEND_PA=570, MiLB weight is ~28% — still
# meaningful for young players but small enough to be a safe cutoff.
FULL_STABILIZATION_PA = 1500

# ── MLE Calibration Curve (Analysis 8) ───────────────────────────────────
# The old wRC+-lookup MLE systematically underestimated elite MiLB
# performers because it compressed all wOBAs into a narrow .278-.340 range.
# With per-level rate multipliers preserving player-specific signal, the
# calibration is only needed in the original .287-.312 range — above that,
# MLEs are already well-scaled and no correction is applied.
MLE_CALIBRATION = [
    (0.287, 0.281),
    (0.293, 0.292),
    (0.298, 0.300),
    (0.302, 0.313),
    (0.307, 0.323),
    (0.312, 0.341),
]

# Counting stats to scale when rate stats change
COUNTING_STATS = ['HR', '2B', '3B', 'RBI', 'R', 'HBP']

# wOBA linear weights (2025 FanGraphs) for reconstruction
_WOBA_WEIGHTS = {
    'wBB': 0.691, 'wHBP': 0.722, 'w1B': 0.882,
    'w2B': 1.252, 'w3B': 1.584, 'wHR': 2.037,
}

# ── Batting Tool Grade Integration ───────────────────────────────────────────
# Replaces FV-only adjustment with hit + power tool grades (20-80 scale).
# Derived from OLS on 690 prospects with 200+ MLB PA:
#   career_wOBA = 0.1879 + 0.00131*grade_hit + 0.00098*grade_power
#   (R²=0.203, vs R²=0.150 for FV-only)
# Hit tool captures contact/AVG signal, power tool captures SLG/ISO signal.
# The adjustment is the residual between grade-implied wOBA and a neutral
# baseline (50-hit / 50-power = .303 wOBA), scaled by GRADE_BLEND_WEIGHT
# so scouting grades nudge the MLE rather than dominate it.
GRADE_INTERCEPT = 0.1879
GRADE_HIT_COEF = 0.00131     # wOBA per hit-tool point
GRADE_POWER_COEF = 0.00098   # wOBA per power-tool point
GRADE_BASELINE_WOBA = 0.303  # implied wOBA at 50-hit / 50-power
GRADE_BLEND_WEIGHT = 0.20    # how much of the grade residual to apply

# MiLB data path
MILB_DATA_FILE = Config.Paths.DATA_DIR / 'MiLB' / 'MiLB_Hitters.csv'

# Historic MLB batting data for career PA calculation
HISTORIC_BATTING_FILE = Config.Paths.HISTORIC_MLB_DIR / 'mlb_batting_data_1950_2025.csv'

# Prospect data + ID crosswalk
PROSPECT_DATA_FILE = Config.Paths.PROSPECT_FILE
REGISTER_DATA_DIR = Config.Paths.DATA_DIR / 'register' / 'data'
CROSSWALK_FILE = Config.Paths.GENERATED_DIR / 'player_id_crosswalk.csv'


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_milb_data() -> pd.DataFrame:
    """Load and clean MiLB hitter data."""
    if not MILB_DATA_FILE.exists():
        logger.warning(f"MiLB data not found at {MILB_DATA_FILE}")
        return pd.DataFrame()

    milb = pd.read_csv(MILB_DATA_FILE, low_memory=False)

    # Find the player-ID column case-insensitively — FanGraphs exports this
    # as 'PlayerID' (capital ID), not 'PlayerId'.
    id_cols = [c for c in milb.columns if c.lower() in ('playerid', 'player_id')]
    if not id_cols:
        logger.error(f"No player-ID column found in {MILB_DATA_FILE} "
                     f"(columns: {list(milb.columns)})")
        return pd.DataFrame()
    id_col = id_cols[0]

    # Normalize to int for matching with IDfg. This drops rows whose ID is
    # non-numeric (FanGraphs' 'sa######' placeholder IDs for players who
    # haven't been assigned a numeric IDfg yet) — those can't be matched to
    # batter_data['IDfg'] by equality regardless, so they're excluded here
    # unless/until they're run through an ID crosswalk.
    milb['PlayerID'] = pd.to_numeric(milb[id_col], errors='coerce')
    n_dropped = milb['PlayerID'].isna().sum()
    if n_dropped:
        logger.info(f"{n_dropped}/{len(milb)} MiLB rows have no numeric "
                    f"IDfg-compatible ID and will be excluded from regression")
    milb = milb.dropna(subset=['PlayerID'])
    milb['PlayerID'] = milb['PlayerID'].astype(int)

    # Keep only relevant levels
    milb = milb[milb['Level'].isin(RELEVANT_LEVELS)].copy()

    # Drop rows with insufficient PA
    milb = milb[milb['PA'] >= MIN_MILB_PA].copy()

    # Ensure numeric stat columns
    for col in ['BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 'PA', 'Age']:
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


def _load_consistency_bonus() -> pd.Series:
    """Extra effective MLB PA for players whose real MLB seasons agree in direction.

    mlb_weight (see _blend_predictions) is a pure PA-volume formula — it
    can't distinguish 300 PA of noisy small-sample MLB performance from
    300 PA spread across three separate seasons that all say the same
    thing about a player's true talent. This computes a bonus for the
    latter case: when a player has 2+ real MLB seasons (each with at
    least MIN_SEASON_PA_FOR_CONSISTENCY PA) that all land on the same
    side of league-average wOBA, credit extra effective PA — capped at
    MAX_CONSISTENCY_BONUS_PA so it nudges the blend without ever fully
    overriding the MiLB side by itself.

    Returns:
        Series indexed by IDfg of bonus PA (0 for players without a
        qualifying consistent multi-season track record).
    """
    if not HISTORIC_BATTING_FILE.exists():
        logger.warning(f"Historic batting data not found at {HISTORIC_BATTING_FILE}")
        return pd.Series(dtype=float)

    needed_cols = {'IDfg', 'Season', 'PA', 'wOBA'}
    available_cols = set(pd.read_csv(HISTORIC_BATTING_FILE, nrows=0).columns)
    if not needed_cols.issubset(available_cols):
        logger.warning(
            "Historic batting data missing columns needed for the "
            f"consistency bonus ({needed_cols - available_cols}) — skipping"
        )
        return pd.Series(dtype=float)

    hist = pd.read_csv(HISTORIC_BATTING_FILE,
                       usecols=['IDfg', 'Season', 'PA', 'wOBA'],
                       low_memory=False)
    hist['PA'] = pd.to_numeric(hist['PA'], errors='coerce').fillna(0)
    hist['wOBA'] = pd.to_numeric(hist['wOBA'], errors='coerce')

    # Collapse to one row per player-season first (a player traded mid-year
    # can have multiple rows per season) before checking season count.
    per_season = hist.groupby(['IDfg', 'Season']).apply(
        lambda g: pd.Series({
            'PA': g['PA'].sum(),
            'wOBA': (g['wOBA'] * g['PA']).sum() / g['PA'].sum() if g['PA'].sum() > 0 else np.nan,
        })
    ).reset_index()

    qualifying = per_season[per_season['PA'] >= MIN_SEASON_PA_FOR_CONSISTENCY].dropna(subset=['wOBA'])

    bonuses = {}
    for pid, pdata in qualifying.groupby('IDfg'):
        directions = np.sign(pdata['wOBA'] - MLB_LEAGUE_AVG_WOBA)
        directions = directions[directions != 0]
        if len(directions) >= 2 and directions.nunique() == 1:
            bonuses[pid] = min(
                MAX_CONSISTENCY_BONUS_PA,
                (len(directions) - 1) * CONSISTENCY_BONUS_PA_PER_SEASON,
            )

    return pd.Series(bonuses, dtype=float)


def _load_prospect_grades() -> dict:
    """Load batting tool grades for prospects, mapped to FanGraphs IDfg.

    Returns dict {IDfg: {'hit': grade_hit, 'power': grade_power}}.
    Uses the most recent year's grades for each player.
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

    # Build mlbam → IDfg crosswalk from unified crosswalk (preferred) + register fallback
    id_map = {}

    # Fallback: Chadwick register files
    people_files = glob.glob(str(REGISTER_DATA_DIR / 'people-*.csv'))
    if people_files:
        xw_dfs = [
            pd.read_csv(f, usecols=['key_mlbam', 'key_fangraphs'], low_memory=False)
            for f in people_files
        ]
        xw = pd.concat(xw_dfs, ignore_index=True).dropna(
            subset=['key_mlbam', 'key_fangraphs']
        )
        xw['key_mlbam'] = xw['key_mlbam'].astype(int)
        xw['key_fangraphs'] = xw['key_fangraphs'].astype(int)
        id_map.update(zip(xw['key_mlbam'], xw['key_fangraphs']))

    # Primary: unified crosswalk built by scrapers pipeline (overrides register)
    if CROSSWALK_FILE.exists():
        xw_unified = pd.read_csv(CROSSWALK_FILE, low_memory=False)
        xw_unified['fg_numeric'] = pd.to_numeric(xw_unified['fg_id'], errors='coerce')
        xw_valid = xw_unified.dropna(subset=['fg_numeric'])
        id_map.update(zip(
            xw_valid['mlbam_id'].astype(int),
            xw_valid['fg_numeric'].astype(int),
        ))
        logger.info(f"Loaded unified crosswalk: {len(xw_valid)} numeric mappings")
    elif not people_files:
        logger.warning("No crosswalk files found — batting grades unavailable")
        return {}

    # Map prospects to IDfg
    batters['IDfg'] = batters['mlbam_id'].map(id_map)
    batters = batters.dropna(subset=['IDfg'])
    batters['IDfg'] = batters['IDfg'].astype(int)

    # Most recent grades per player (latest year with grade_hit populated)
    batters = batters.sort_values('year', ascending=False)
    latest = batters.drop_duplicates('IDfg')

    grades = {}
    for _, row in latest.iterrows():
        idfg = int(row['IDfg'])
        hit = row.get('grade_hit', np.nan)
        power = row.get('grade_power', np.nan)
        if pd.notna(hit):
            grades[idfg] = {
                'hit': float(hit),
                'power': float(power) if pd.notna(power) else 50.0,
            }

    logger.info(f"Loaded batting tool grades for {len(grades)} prospect batters")
    return grades


# =============================================================================
# MLE CALCULATION
# =============================================================================

def _age_woba_adjustment(age: float, level: str) -> float:
    """Compute wOBA bonus for being young-for-level.

    Young-for-level → positive adjustment (more impressive talent signal).
    Old-for-level → negative adjustment (stats inflated by experience).
    """
    benchmark = AGE_BENCHMARKS.get(level)
    if benchmark is None or pd.isna(age):
        return 0.0
    pts_per_yr = AGE_ADJUSTMENT_WOBA_PTS.get(level, 0.004)
    return pts_per_yr * (benchmark - age)


def _compute_stagnation_penalty(player_milb: pd.DataFrame) -> float:
    """Compute wOBA penalty for players stuck at their highest MiLB level.

    Players who spend 2+ seasons at their highest level are penalized because
    being passed over for promotion despite good numbers is a strong signal
    of AAAA-type players.  The first season at a level is free.

    Returns a non-negative wOBA penalty (to be subtracted from the MLE).
    """
    if player_milb.empty:
        return 0.0

    # Determine the player's highest level with meaningful playing time
    level_order = {'AAA': 4, 'AA': 3, 'A+': 2, 'A': 1}
    levels_played = player_milb['Level'].unique()
    ranked = [(lv, level_order.get(lv, 0)) for lv in levels_played if lv in level_order]
    if not ranked:
        return 0.0

    highest_level = max(ranked, key=lambda x: x[1])[0]
    seasons_at_highest = player_milb[
        player_milb['Level'] == highest_level
    ]['Season'].nunique()

    extra_seasons = max(0, seasons_at_highest - 1)
    if extra_seasons == 0:
        return 0.0

    per_year = STAGNATION_WOBA_PER_YEAR.get(highest_level, 0.003)
    penalty = extra_seasons * per_year

    return penalty


def _translate_to_mle(milb_row: pd.Series) -> dict:
    """Translate a single MiLB season-level row to MLB equivalents.

    Uses per-level multiplicative factors for all stats, preserving player-
    specific signal.  Each MiLB stat is multiplied by its level-appropriate
    translation factor (e.g. AAA wOBA × 0.90, AA wOBA × 0.82).  An age
    adjustment is added to wOBA to credit young-for-level performance.

    Returns a dict with translated stats: K%, BB%, AVG, OBP, SLG, wOBA.
    """
    level = milb_row['Level']
    age = milb_row.get('Age', None)

    # K% and BB% per-level adjustments (pitch quality gap)
    k_adj = MLE_LEVEL_FACTORS['K%'].get(level, 0.06)
    bb_factor = MLE_LEVEL_FACTORS['BB%'].get(level, 0.72)

    # Rate-stat per-level multipliers
    avg_factor = MLE_LEVEL_FACTORS['AVG'].get(level, 0.80)
    obp_factor = MLE_LEVEL_FACTORS['OBP'].get(level, 0.80)
    slg_factor = MLE_LEVEL_FACTORS['SLG'].get(level, 0.76)
    woba_factor = MLE_LEVEL_FACTORS['wOBA'].get(level, 0.78)

    # Age adjustment: credit young-for-level, penalize old-for-level
    age_adj = _age_woba_adjustment(age, level)

    base_woba = milb_row['wOBA'] * woba_factor
    translated_woba = base_woba + age_adj
    woba_ratio = translated_woba / base_woba if base_woba > 0 else 1.0

    return {
        'K%':  min(milb_row['K%'] + k_adj, 0.45),
        'BB%': max(milb_row['BB%'] * bb_factor, 0.02),
        'AVG': milb_row['AVG'] * avg_factor * woba_ratio,
        'OBP': milb_row['OBP'] * obp_factor * woba_ratio,
        'SLG': milb_row['SLG'] * slg_factor * woba_ratio,
        'wOBA': max(translated_woba, 0.200),
    }


def _compute_player_mle(player_milb: pd.DataFrame, current_year: int) -> tuple:
    """Compute Marcel-style composite MLE for one player.

    PA × recency-weighted average of translated MLEs across recent seasons,
    with light regression to league mean.

    Args:
        player_milb: All MiLB rows for one player (already filtered to
                     relevant levels and min PA).
        current_year: The projection year (for MiLB recency weighting).

    Returns:
        Tuple of (mle_dict, total_milb_pa) or (empty dict, 0) if no data.
    """
    if player_milb.empty:
        return {}, 0

    # Sort by season descending, take most recent 5 seasons
    recent_seasons = sorted(player_milb['Season'].unique(), reverse=True)[:5]

    stats = ['K%', 'BB%', 'AVG', 'OBP', 'SLG', 'wOBA']

    # ---- Pass 1: player's own PA×recency-weighted raw wOBA (untranslated) ----
    # This is the anchor each individual season gets shrunk toward before
    # translation, so a single outlier season (small-sample and/or hot)
    # doesn't pass through the flat per-level multiplier at full face value.
    season_rows = []  # (row, recency_weight, pa) for every level-row across recent_seasons
    anchor_weighted_sum = 0.0
    anchor_total_weight = 0.0
    for recency_idx, season in enumerate(recent_seasons):
        season_data = player_milb[player_milb['Season'] == season]
        recency_weight = MILB_SEASON_WEIGHTS.get(recency_idx, 0.5)

        for _, row in season_data.iterrows():
            pa = row['PA']
            w = pa * recency_weight
            anchor_weighted_sum += row['wOBA'] * w
            anchor_total_weight += w
            season_rows.append((row, recency_weight, pa))

    if anchor_total_weight == 0:
        return {}, 0

    player_anchor_woba = anchor_weighted_sum / anchor_total_weight

    # ---- Pass 2: shrink each season toward the anchor, then translate ----
    weighted_stats = {s: 0.0 for s in stats}
    total_weight = 0.0
    total_pa = 0.0

    for row, recency_weight, pa in season_rows:
        season_reliability = pa / (pa + SEASON_SHRINKAGE_PA)
        shrunk_row = row.copy()
        shrunk_woba = (
            season_reliability * row['wOBA']
            + (1 - season_reliability) * player_anchor_woba
        )
        # Scale AVG/OBP/SLG proportionally so the shrunk season stays
        # internally consistent before it goes through _translate_to_mle.
        if row['wOBA'] > 0:
            scale = shrunk_woba / row['wOBA']
            shrunk_row['AVG'] = row['AVG'] * scale
            shrunk_row['OBP'] = row['OBP'] * scale
            shrunk_row['SLG'] = row['SLG'] * scale
        shrunk_row['wOBA'] = shrunk_woba

        mle = _translate_to_mle(shrunk_row)
        w = pa * recency_weight

        for stat in stats:
            weighted_stats[stat] += mle[stat] * w
        total_weight += w
        total_pa += pa

    if total_weight == 0:
        return {}, 0

    result = {stat: weighted_stats[stat] / total_weight for stat in stats}

    # Light regression toward league mean (50 PA) — the per-level
    # multipliers already embed substantial regression, so this just
    # nudges small-sample MLEs toward a safe baseline.
    raw_woba = result['wOBA']
    player_weight = total_pa / (total_pa + REGRESSION_PA)
    regressed_woba = player_weight * raw_woba + (1 - player_weight) * LEAGUE_MEAN_WOBA

    if raw_woba > 0:
        scale = regressed_woba / raw_woba
        result['AVG'] *= scale
        result['OBP'] *= scale
        result['SLG'] *= scale
    result['wOBA'] = regressed_woba

    return result, int(total_pa)


def _calibrate_mle(mle: dict) -> dict:
    """Apply calibration correction to MLE wOBA and scale other rate stats.

    Only applies within the calibrated range (.287-.312 wOBA).  Above that
    range, per-level multipliers already produce well-scaled MLEs and no
    correction is needed.  Scales AVG/OBP/SLG proportionally.
    """
    raw_woba = mle.get('wOBA', 0)
    if raw_woba <= 0:
        return mle

    table = MLE_CALIBRATION
    # No correction outside the calibrated range
    if raw_woba <= table[0][0] or raw_woba >= table[-1][0]:
        return mle

    cal_woba = raw_woba
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= raw_woba <= x1:
            t = (raw_woba - x0) / (x1 - x0)
            cal_woba = y0 + t * (y1 - y0)
            break

    adjusted = mle.copy()
    mult = cal_woba / raw_woba
    for stat in ('AVG', 'OBP', 'SLG', 'wOBA'):
        if stat in adjusted:
            adjusted[stat] *= mult

    return adjusted


def _apply_grade_adjustment(mle: dict, grades: dict) -> dict:
    """Adjust MLE rate stats using batting tool grades (hit + power).

    The hit+power regression predicts expected career wOBA from scouting
    grades alone.  The residual between grade-implied wOBA and the neutral
    baseline (50/50 = .303) captures talent information that MiLB stats
    alone miss (mechanics, raw power, projection).  Only a fraction
    (GRADE_BLEND_WEIGHT) of this residual is applied so grades nudge
    the MLE without overwhelming it.

    Applied as a multiplicative scale so AVG/OBP/SLG stay proportionally
    consistent.  K% and BB% are left unchanged.
    """
    hit = grades.get('hit', 50.0)
    power = grades.get('power', 50.0)

    # Grade-implied wOBA from regression
    grade_woba = GRADE_INTERCEPT + GRADE_HIT_COEF * hit + GRADE_POWER_COEF * power
    residual = grade_woba - GRADE_BASELINE_WOBA
    woba_adj = residual * GRADE_BLEND_WEIGHT

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
    milb_pa: int = MLE_BLEND_PA,
) -> dict:
    """Blend model predictions with MiLB MLE using reliability weights.

    Blends rate stats (K%, BB%, AVG, OBP, SLG, wOBA), then scales counting
    stats (HR, 2B, 3B, RBI, R, HBP) proportionally to the wOBA change so
    everything stays internally consistent.  Finally reconstructs OBP, SLG,
    and wOBA from the adjusted components using standard linear weights.

    The MiLB side of the blend is capped at the actual MiLB PA so that a
    player with only 97 MiLB PA doesn't get the same weight as one with
    1,400 PA.

    Args:
        model_pred: One row from batter_predictions with model outputs.
        mle: Dict of MLE stats for this player.
        career_pa: Total career MLB PA.
        milb_pa: Total qualifying MiLB PA backing the MLE.

    Returns:
        Dict of blended stat values (rate stats + adjusted counting stats).
    """
    blended = {}

    for stat, stab_pa in STABILIZATION_PA.items():
        if stat not in model_pred.index or stat not in mle:
            continue

        model_val = model_pred[stat]
        mle_val = mle[stat]

        # Cap effective MiLB PA at the actual sample size backing the MLE
        effective_stab = min(stab_pa, milb_pa)

        # Reliability weight: how much to trust the MLB model
        mlb_weight = career_pa / (career_pa + effective_stab)

        blended_val = mlb_weight * model_val + (1 - mlb_weight) * mle_val
        if pd.isna(blended_val):
            blended_val = model_val
        blended[stat] = blended_val

    # ── Derive counting stats algebraically from blended rate stats ──────
    # We freeze the reliable blended rate stats (AVG, OBP, SLG, BB%, wOBA).
    # Then we mathematically derive Total Hits (H) and Total Bases (TB).
    # Finally, we distribute Extra Base Hits (2B, 3B, HR) according to the
    # model's original predicted extra-base distribution so that they perfectly
    # match the new blended SLG and wOBA.
    pa = 650.0
    bb_pct = blended.get('BB%', model_pred.get('BB%', 0.08))
    avg = blended.get('AVG', model_pred.get('AVG', 0.240))
    slg = blended.get('SLG', model_pred.get('SLG', 0.400))
    target_woba = blended.get('wOBA', model_pred.get('wOBA', 0.310))

    bb = pa * bb_pct
    orig_pa = 650.0
    
    # Estimate HBP based on original proportion
    hbp = model_pred.get('HBP', orig_pa * 0.01) * (pa / orig_pa)
    sf = orig_pa * 0.007 * (pa / orig_pa)
    
    ab = max(1.0, pa - bb - hbp - sf)
    h = avg * ab
    tb = slg * ab
    
    # Extra bases to distribute: TB - H = 2B + 2*3B + 3*HR
    diff = tb - h
    xb = max(0.0, diff) if not pd.isna(diff) else 0.0
    
    # Get original distribution of extra bases
    orig_2b = model_pred.get('2B', 0)
    orig_3b = model_pred.get('3B', 0)
    orig_hr = model_pred.get('HR', 0)
    orig_xb = orig_2b + 2 * orig_3b + 3 * orig_hr
    
    if orig_xb > 0:
        xb_ratio = xb / orig_xb
        blended['2B'] = orig_2b * xb_ratio
        blended['3B'] = orig_3b * xb_ratio
        blended['HR'] = orig_hr * xb_ratio
    else:
        # fallback distribution if model predicted 0 XB
        blended['2B'] = xb * (2.0 / 5.0)
        blended['3B'] = 0.0
        blended['HR'] = xb * (1.0 / 3.0)

    # Scale Run production stats by wOBA increase
    orig_woba = max(0.001, model_pred.get('wOBA', 0.310))
    r_ratio = target_woba / orig_woba
    blended['R'] = model_pred.get('R', 0) * r_ratio
    blended['RBI'] = model_pred.get('RBI', 0) * r_ratio
    blended['HBP'] = hbp

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

    consistency_bonus_series = _load_consistency_bonus()

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
    milb_by_player = milb.groupby('PlayerID')
    player_mles = {}      # pid → mle dict
    player_milb_pa = {}   # pid → total qualifying MiLB PA
    fv_adjusted_count = 0
    for pid in candidates:
        mle = None
        milb_total_pa = 0
        is_baseline = False

        if pid in milb_by_player.groups:
            player_milb = milb_by_player.get_group(pid)
            mle, milb_total_pa = _compute_player_mle(player_milb, current_year)

        if not mle:
            # Missing MiLB data (e.g. skipped minors or ID missing from crosswalk).
            # Fabricate a baseline MLE so they still get regressed appropriately
            # instead of keeping 100% of their volatile MLB sample.
            mle = {
                'K%': 0.240,
                'BB%': 0.080,
                'AVG': 0.240,
                'OBP': 0.310,
                'SLG': 0.400,
                'wOBA': GRADE_BASELINE_WOBA,
            }
            milb_total_pa = MLE_BLEND_PA
            is_baseline = True
        else:
            # Calibration correction (Analysis 8) for real MLEs
            mle = _calibrate_mle(mle)

        # Apply batting tool grade adjustment if available (works on baselines too)
        if pid in prospect_grades:
            grades = prospect_grades[pid]
            mle = _apply_grade_adjustment(mle, grades)
            fv_adjusted_count += 1

        # Stagnation penalty for AAAA-type players (real MLEs only)
        if not is_baseline:
            stag_penalty = _compute_stagnation_penalty(player_milb)
            if stag_penalty > 0:
                base_woba = mle.get('wOBA', 0)
                if base_woba > 0:
                    mult = max(0.85, (base_woba - stag_penalty) / base_woba)
                    for stat in ('AVG', 'OBP', 'SLG', 'wOBA'):
                        if stat in mle:
                            mle[stat] *= mult
                    logger.debug(
                        f"  IDfg={pid}: stagnation penalty {stag_penalty:.3f} wOBA "
                        f"(mult={mult:.3f})"
                    )

        player_mles[pid] = mle
        player_milb_pa[pid] = milb_total_pa

    if fv_adjusted_count > 0:
        logger.info(f"Applied batting grade adjustments to {fv_adjusted_count} / "
                    f"{len(player_mles)} MLEs")

    logger.info(f"Computed MLEs for {len(player_mles)} batters with MiLB data")

    # Apply blending
    stats_to_blend = list(STABILIZATION_PA.keys())
    adjusted_count = 0

    for pid, mle in player_mles.items():
        career_pa = career_pa_series.get(pid, 0)
        consistency_bonus = consistency_bonus_series.get(pid, 0.0)
        effective_career_pa = career_pa + consistency_bonus
        milb_pa = player_milb_pa.get(pid, MLE_BLEND_PA)
        mask = result['IDfg'] == pid

        if not mask.any():
            continue

        # Blend EACH projection year individually so the LSTM's
        # year-over-year progression is preserved.
        first_row = True
        for idx in result.index[mask]:
            row = result.loc[idx]
            blended = _blend_predictions(row, mle, effective_career_pa, milb_pa)
            if not blended:
                continue

            for stat, val in blended.items():
                result.at[idx, stat] = val

            if first_row:
                adjusted_count += 1
                # Log notable adjustments (first year only)
                effective_stab = min(STABILIZATION_PA.get('wOBA', MLE_BLEND_PA), milb_pa)
                mlb_weight = effective_career_pa / (effective_career_pa + effective_stab)
                if career_pa < 300:
                    name = row.get('Name', f'IDfg={pid}')
                    old_woba = row.get('wOBA', 0)
                    new_woba = blended.get('wOBA', old_woba)
                    logger.debug(
                        f"  {name}: PA={career_pa:.0f} (+{consistency_bonus:.0f} consistency), "
                        f"mlb_wt={mlb_weight:.2f}, "
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