"""
Standalone MiLB Projections
===========================
Generates Expected MLB stat lines for MiLB batters using Multiple Linear
Regression models for wRC+ and K%, then solves for a full, mathematically
coherent slash line (AVG/OBP/SLG/OPS/HR/counting stats) by scaling the
player's MiLB-derived baseline components (BB%, ISO, BABIP) to hit
the regression's wOBA target.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import brentq

from core.stat_composition import (
    compose_counting,
    compose_woba,
    compose_wrc_plus,
    WOBA_WEIGHTS, WOBA_SCALE, LG_WOBA, LG_RUNS_PER_PA, LG_WRC_PER_PA,
    DEFAULT_PA, DEFAULT_SF_RATE, DEFAULT_3B_SHARE,
)

logger = logging.getLogger(__name__)

# ── MLR Coefficients ──────────────────────────────────────────────────────────
# Expected MLB wRC+ = β0 + β1(MiLB wRC+) + β2(MiLB Age)
BATTER_WRC_REGRESSION = {
    'AAA': {'intercept': 54.25, 'wrc_coef': 0.373, 'age_coef': -0.177},
    'AA':  {'intercept': 73.35, 'wrc_coef': 0.309, 'age_coef': -0.772},
    'A+':  {'intercept': 78.07, 'wrc_coef': 0.243, 'age_coef': -0.730},
    'A':   {'intercept': 100.53, 'wrc_coef': 0.195, 'age_coef': -1.618},
    'A-':  {'intercept': 100.53, 'wrc_coef': 0.195, 'age_coef': -1.618},
    'CPX': {'intercept': 100.53, 'wrc_coef': 0.195, 'age_coef': -1.618},
}

# Expected MLB K% = β0 + β1(MiLB K%) + β2(MiLB Age)
K_REGRESSION = {
    'AAA': {'intercept': 14.88, 'k_coef': 0.859, 'age_coef': -0.337},
    'AA':  {'intercept': 11.54, 'k_coef': 0.746, 'age_coef': -0.097},
    'A+':  {'intercept': 9.53,  'k_coef': 0.635, 'age_coef': 0.100},
    'A':   {'intercept': 7.07,  'k_coef': 0.630, 'age_coef': 0.243},
    'A-':  {'intercept': 7.07,  'k_coef': 0.630, 'age_coef': 0.243},
    'CPX': {'intercept': 7.07,  'k_coef': 0.630, 'age_coef': 0.243},
}

# Marcel-style exponential decay weights  (year N-0 → N-4)
RECENCY_WEIGHTS = {0: 8, 1: 4, 2: 2, 3: 1, 4: 0.5}

# League-average baselines for missing component data
LG_BB_PCT   = 0.080
LG_ISO      = 0.140
LG_BABIP    = 0.290
LG_HR_FB    = 0.120
LG_GB_PCT   = 0.430
LG_LD_PCT   = 0.200
LG_HBP_PCT  = 0.010


# ── Phase 1: Evaluate individual MiLB season rows ────────────────────────────

def evaluate_milb_batter_seasons(milb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate Expected MLB wRC+ and Expected K% for every row in the MiLB
    dataset.  Also preserves the raw MiLB BB%, ISO, BABIP for later use
    as baseline component shapes.
    """
    df = milb_df.copy()

    # ── Identify FanGraphs ID column ──
    id_cols = [c for c in df.columns if c.lower() in ('playerid', 'player_id', 'idfg')]
    if id_cols:
        df['IDfg'] = pd.to_numeric(df[id_cols[0]], errors='coerce')

    # ── Clean percentage columns (may arrive as "23.5 %" strings) ──
    for col in ['K%', 'BB%']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0

    # ── Coerce all numeric columns ──
    for col in ['wRC+', 'Age', 'PA', 'Season', 'ISO', 'BABIP', 'K%', 'BB%']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].replace({',': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['IDfg', 'wRC+', 'Age', 'PA', 'Season'])
    df['IDfg'] = df['IDfg'].astype(int)
    df = df[df['PA'] > 0]

    # ── Expected wRC+ (MLR) ──
    for attr, key in [('_int', 'intercept'), ('_wc', 'wrc_coef'), ('_ac', 'age_coef')]:
        df[attr] = df['Level'].map({k: v[key] for k, v in BATTER_WRC_REGRESSION.items()})
        df[attr] = df[attr].fillna(BATTER_WRC_REGRESSION['A'][key])

    df['Expected_wRC+'] = (df['_int'] + df['_wc'] * df['wRC+'] + df['_ac'] * df['Age']).clip(lower=0)

    # ── Expected K% (MLR) — input is 0-100 scale ──
    milb_k_100 = df['K%'].fillna(0.20) * 100.0

    for attr, key in [('_ki', 'intercept'), ('_kc', 'k_coef'), ('_ka', 'age_coef')]:
        df[attr] = df['Level'].map({k: v[key] for k, v in K_REGRESSION.items()})
        df[attr] = df[attr].fillna(K_REGRESSION['A'][key])

    df['Expected_K%'] = (df['_ki'] + df['_kc'] * milb_k_100 + df['_ka'] * df['Age']).clip(lower=0) / 100.0

    df.drop(columns=['_int', '_wc', '_ac', '_ki', '_kc', '_ka'], inplace=True)
    return df


# ── Phase 2: Aggregate across seasons ────────────────────────────────────────

def aggregate_batter_projections(df: pd.DataFrame, projection_year: int = None) -> pd.DataFrame:
    """
    Collapse per-season rows into one projection per player using
    PA × Recency weighting for all components.

    Offset (and therefore Recency_Weight) is anchored to each player's OWN
    most recent season present in `df`, not to `projection_year`. This is
    what makes a player's MiLB prior a stable, fixed translation of "his
    last MiLB stint" rather than something that silently decays and
    eventually drops out of the 5-season RECENCY_WEIGHTS window purely
    because projection_year keeps advancing while he sits in the majors
    with no new MiLB games. A player's most recent MiLB season stays
    Offset=0 (full weight-8) for every future projection_year, until he
    actually plays more MiLB and that anchor updates. What SHOULD change
    year over year is how much this prior is trusted relative to his real
    MLB performance — that's handled separately, by mlb_weight in
    marcel_batter_projections (career_pa / (career_pa + 400)), not by
    aging this function's output toward irrelevance.

    `projection_year` is accepted for backward compatibility with existing
    callers but is no longer used to compute Offset.
    """
    if df.empty:
        return df
    df = df.copy()

    anchor_season = df.groupby('IDfg')['Season'].transform('max')
    df['Offset'] = anchor_season - df['Season']
    df = df[(df['Offset'] >= 0) & (df['Offset'] <= 4)].copy()
    if df.empty:
        return df

    df['Recency_Weight'] = df['Offset'].map(RECENCY_WEIGHTS).fillna(0)
    df['Total_Weight'] = df['PA'] * df['Recency_Weight']

    # Fill missing baselines with league averages
    df['BB%']   = df['BB%'].fillna(LG_BB_PCT)
    df['ISO']   = df['ISO'].fillna(LG_ISO)
    df['BABIP'] = df['BABIP'].fillna(LG_BABIP)

    # Weighted components
    weighted_cols = ['Expected_wRC+', 'Expected_K%', 'BB%', 'ISO', 'BABIP']
    for col in weighted_cols:
        df[f'W_{col}'] = df[col] * df['Total_Weight']

    # Descriptive info from the most recent season
    df = df.sort_values(['Season', 'PA'])
    idx_latest = df.groupby('IDfg')['Season'].idxmax()
    latest_info = df.loc[idx_latest].set_index('IDfg')[['Level', 'Name', 'Age']]

    # Aggregate
    agg = {'Total_Weight': 'sum', 'PA': 'sum'}
    for col in weighted_cols:
        agg[f'W_{col}'] = 'sum'

    grouped = df.groupby('IDfg').agg(agg)

    for col in weighted_cols:
        grouped[col] = grouped[f'W_{col}'] / grouped['Total_Weight']

    grouped['Name'] = latest_info['Name']
    grouped['Latest_Level'] = latest_info['Level']
    grouped['Age_in_Latest_Season'] = latest_info['Age']

    return grouped.reset_index()


# ── Phase 3: Constraint solver — full slash line from wRC+ + K% targets ──────

def _wrc_plus_to_woba(wrc_plus: float, park_factor: float = 100.0) -> float:
    """Invert compose_wrc_plus:  wRC+ → target wOBA (neutral or park-adjusted)."""
    pf = park_factor / 100.0
    park_adj = LG_RUNS_PER_PA - (pf * LG_RUNS_PER_PA)
    # wRC+ = 100 * ((wraa_pa + lg_r_pa + park_adj) / lg_wrc_pa)
    # ⇒ wraa_pa = (wrc+ / 100) * lg_wrc_pa - lg_r_pa - park_adj
    wraa_pa = (wrc_plus / 100.0) * LG_WRC_PER_PA - LG_RUNS_PER_PA - park_adj
    # wraa_pa = (woba - lg_woba) / woba_scale
    target_woba = wraa_pa * WOBA_SCALE + LG_WOBA
    return target_woba


def _solve_scale_factor(
    target_woba: float,
    k_pct: float,
    bb_pct: float,
    iso: float,
    babip: float,
) -> float:
    """
    Find a single scalar *M* such that scaling (BB%, ISO, BABIP) by M
    produces a composed wOBA that matches *target_woba*.

    Uses Brent's method on the monotonic function  wOBA(M) − target = 0.
    """
    # Enforce minimum floors so we never scale from zero
    bb_pct  = max(bb_pct, 0.04)
    iso     = max(iso, 0.060)
    babip   = max(babip, 0.200)

    def _residual(m: float) -> float:
        test_bb    = np.clip(bb_pct * m, 0.02, 0.25)
        test_iso   = np.clip(iso * m, 0.01, 0.450)
        test_babip = np.clip(babip * m, 0.15, 0.450)

        counts = compose_counting(
            k_pct   = k_pct,
            bb_pct  = test_bb,
            hbp_pct = LG_HBP_PCT,
            iso     = test_iso,
            babip   = test_babip,
            hr_fb   = np.clip(LG_HR_FB * m, 0.01, 0.35),
            gb_pct  = LG_GB_PCT,
            ld_pct  = LG_LD_PCT,
            pa      = DEFAULT_PA,
        )
        return compose_woba(counts) - target_woba

    try:
        best_m = brentq(_residual, 0.01, 5.0, xtol=1e-6)
    except ValueError:
        # Bracket doesn't straddle zero — fall back to scale=1
        logger.debug("Brent solver could not bracket; falling back to M=1.0")
        best_m = 1.0

    return best_m


def generate_full_slash_lines(df: pd.DataFrame, park_factor: float = 100.0) -> pd.DataFrame:
    """
    For each player row coming out of aggregation, solve for a full
    slash line that is:
      • locked to the MLR Expected wRC+ target  (via wOBA)
      • locked to the MLR Expected K% target
      • shaped by the player's historical MiLB BB%, ISO, BABIP profile

    Returns the input DataFrame with additional Proj_* columns appended.
    """
    results = []

    for _, row in df.iterrows():
        wrc_plus   = row['Expected_wRC+']
        k_pct      = row['Expected_K%']
        baseline_bb    = row.get('BB%',   LG_BB_PCT)
        baseline_iso   = row.get('ISO',   LG_ISO)
        baseline_babip = row.get('BABIP', LG_BABIP)

        # 1.  wRC+ → target wOBA
        target_woba = _wrc_plus_to_woba(wrc_plus, park_factor)

        # 2.  Find the scale factor M
        m = _solve_scale_factor(target_woba, k_pct, baseline_bb, baseline_iso, baseline_babip)

        # 3.  Build final components
        final_bb    = np.clip(max(baseline_bb, 0.04) * m, 0.02, 0.25)
        final_iso   = np.clip(max(baseline_iso, 0.060) * m, 0.01, 0.450)
        final_babip = np.clip(max(baseline_babip, 0.200) * m, 0.15, 0.450)
        final_hr_fb = np.clip(LG_HR_FB * m, 0.01, 0.35)

        # 4.  Compose counting stats (guaranteed coherent with rates)
        counts = compose_counting(
            k_pct   = k_pct,
            bb_pct  = final_bb,
            hbp_pct = LG_HBP_PCT,
            iso     = final_iso,
            babip   = final_babip,
            hr_fb   = final_hr_fb,
            gb_pct  = LG_GB_PCT,
            ld_pct  = LG_LD_PCT,
            pa      = DEFAULT_PA,
        )
        final_woba     = compose_woba(counts)
        final_wrc_plus = compose_wrc_plus(final_woba, park_factor)

        results.append({
            'Target_wRC+':  wrc_plus,
            'Proj_wRC+':    round(final_wrc_plus, 1),
            'Proj_wOBA':    round(final_woba, 3),
            'Proj_K%':      round(k_pct, 3),
            'Proj_BB%':     round(final_bb, 3),
            'Proj_AVG':     round(counts['AVG'], 3),
            'Proj_OBP':     round(counts['OBP'], 3),
            'Proj_SLG':     round(counts['SLG'], 3),
            'Proj_OPS':     round(counts['OBP'] + counts['SLG'], 3),
            'Proj_ISO':     round(final_iso, 3),
            'Proj_BABIP':   round(final_babip, 3),
            'Proj_HR/FB':   round(final_hr_fb, 3),
            'Proj_GB%':     round(LG_GB_PCT, 3),
            'Proj_LD%':     round(LG_LD_PCT, 3),
            'Proj_HBP%':    round(LG_HBP_PCT, 3),
            'Proj_HR':      round(counts['HR'], 1),
            'Proj_2B':      round(counts['2B'], 1),
            'Proj_3B':      round(counts['3B'], 1),
            'Proj_1B':      round(counts['1B'], 1),
            'Proj_H':       round(counts['H'], 1),
            'Proj_BB':      round(counts['BB'], 1),
            'Proj_K':       round(counts['K'], 1),
            'Proj_AB':      round(counts['AB'], 1),
            'Proj_PA':      DEFAULT_PA,
        })

    res_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, res_df], axis=1)


def get_milb_priors(
    projection_year: int,
    exclude_mlb_experienced: bool = False,
    mlb_pa_exclusion_threshold: int = 130,
) -> pd.DataFrame:
    """
    End-to-end pipeline to load MiLB data, evaluate MLR models, aggregate
    historical seasons, solve for wOBA/slash components, and return a clean
    DataFrame of Year 1 MiLB-derived priors (Proj_K%, Proj_BB%, Proj_ISO,
    etc.) for every player with qualifying recent MiLB history.

    By default (exclude_mlb_experienced=False) this returns a prior for
    EVERY such player, including those who already have a real MLB track
    record. This is the single source of MiLB signal for the batter
    pipeline: callers with an MLB-based projection are expected to blend
    it against this prior (reliability-weighted by career MLB PA), and
    callers with no MLB-based projection at all (pure prospects) use it
    directly as their Year 1 baseline. There is no longer a separate
    MiLB regression step downstream in value determination — this
    function is the only place MiLB → MLB translation happens.

    Pass exclude_mlb_experienced=True to restrict to players with no
    meaningful MLB history, matching the old rookie-only behavior.
    """
    import sys
    from pathlib import Path
    
    # Path hack to load config if running outside standard tree
    # (assuming we are inside auto_train_models/core/milb_projections)
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
        
    from value_determination.config import Config
    
    milb_path = Config.Paths.DATA_DIR / 'MiLB' / 'MiLB_Hitters.csv'
    if not milb_path.exists():
        logger.warning(f"MiLB hitters data not found at {milb_path}. Skipping MiLB projections.")
        return pd.DataFrame()
        
    logger.info("Evaluating MLR Expected wRC+ and K% for MiLB seasons...")
    milb_df = pd.read_csv(milb_path, low_memory=False)
    evaluated_df = evaluate_milb_batter_seasons(milb_df)

    # NOTE: previously filtered here to players who played MiLB in exactly
    # `projection_year - 1`. That made a player's MiLB prior disappear
    # entirely the moment projection_year advanced past his last MiLB
    # season — e.g. a rookie who debuts and never gets optioned down again
    # would lose his prior the very next projection cycle, even though
    # mlb_weight (in marcel_batter_projections) hadn't yet grown large
    # enough to make that prior irrelevant. No filter is needed here now:
    # aggregate_batter_projections() anchors each player's Offset window to
    # THEIR OWN most recent MiLB season, so it naturally keeps (and
    # correctly recency-weights) any player with qualifying MiLB history in
    # the last 5 seasons relative to himself — without requiring that
    # history to line up with projection_year at all.
    
    if exclude_mlb_experienced:
        # Filter out players with significant MLB experience (legacy
        # "rookie-eligible prospects only" behavior).
        mlb_path = Config.Paths.HISTORIC_MLB_DIR / 'mlb_batting_data_1950_2025.csv'
        if mlb_path.exists():
            mlb_df = pd.read_csv(mlb_path, low_memory=False)
            if 'IDfg' in mlb_df.columns and 'PA' in mlb_df.columns and 'Season' in mlb_df.columns:
                mlb_df = mlb_df[mlb_df['Season'] < projection_year].copy()
                mlb_df['IDfg'] = pd.to_numeric(mlb_df['IDfg'], errors='coerce')
                mlb_df['PA'] = pd.to_numeric(mlb_df['PA'], errors='coerce')
                career_mlb_pa = mlb_df.groupby('IDfg')['PA'].sum()
                experienced_players = career_mlb_pa[career_mlb_pa > mlb_pa_exclusion_threshold].index
                evaluated_df = evaluated_df[~evaluated_df['IDfg'].isin(experienced_players)].copy()

    logger.info(f"Found {evaluated_df['IDfg'].nunique()} players with qualifying MiLB history. "
                f"Generating Year 1 priors...")
    projections = aggregate_batter_projections(evaluated_df, projection_year=projection_year)
    projections = generate_full_slash_lines(projections)
    
    return projections


def get_milb_year1_baselines(projection_year: int) -> pd.DataFrame:
    """
    Backward-compatible alias: MiLB priors restricted to rookie-eligible
    prospects with no significant MLB experience. Prefer get_milb_priors()
    for new code — this only exists so old callers keep working.
    """
    return get_milb_priors(projection_year, exclude_mlb_experienced=True)


# ── Cached priors: the single on-disk handoff between project_milb.py /
#    the daily pipeline's MiLB-priors step, and marcel_projections.py ────────

def get_milb_priors_output_path() -> Path:
    """
    Canonical location the rest of the pipeline reads cached MiLB batter
    priors from. This is written once per daily-pipeline run (by
    save_milb_priors / project_milb.py) and then read by every
    marcel_batter_projections() call during that run, instead of each
    caller re-computing the full MiLB regression from raw MiLB_Hitters.csv
    on every invocation.
    """
    import sys
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from value_determination.config import Config

    return Config.Paths.DATA_DIR / 'generated' / 'pipeline' / 'MiLB_priors' / 'milb_batter_priors.csv'


def save_milb_priors(
    projection_year: int,
    exclude_mlb_experienced: bool = False,
    mlb_pa_exclusion_threshold: int = 130,
    out_path: Path = None,
) -> pd.DataFrame:
    """
    Compute MiLB batter priors via get_milb_priors() and write them to
    `out_path` (default: get_milb_priors_output_path(), i.e.
    data/generated/pipeline/MiLB_priors/milb_batter_priors.csv) so
    downstream callers can read a cached copy instead of recomputing.
    Returns the computed DataFrame (empty if there was nothing to save).
    """
    projections = get_milb_priors(
        projection_year,
        exclude_mlb_experienced=exclude_mlb_experienced,
        mlb_pa_exclusion_threshold=mlb_pa_exclusion_threshold,
    )

    if projections.empty:
        logger.warning("No MiLB priors generated — nothing to save.")
        return projections

    if out_path is None:
        out_path = get_milb_priors_output_path()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    projections.to_csv(out_path, index=False)
    logger.info(f"Saved {len(projections)} MiLB batter priors to {out_path}")

    return projections