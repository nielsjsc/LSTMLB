#!/usr/bin/env python3
"""
Free Agency & Extension Contract Analysis
==========================================

Matches Spotrac FA signings and extensions to historical projections to
understand what drives contract value.  Key questions:

    1. Which WAR components (bat, defense, baserunning) predict contract $?
    2. Do SP and RP get paid differently per WAR?
    3. Is defensive WAR discounted relative to offensive WAR?
    4. Do extensions differ from FA signings?

Approach:
    - For each signing/extension, find the player's projection from the
      cutoff_YYYY matching their signing year.
    - Also pull actual stats from the prior 3 seasons (trailing performance).
    - Regress AAV against projection features, age, position, etc.

Usage:
    python -m auto_train_models.analysis.fa_contract_analysis
"""

import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "auto_train_models"))

import pandas as pd
import numpy as np
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
HISTORIC_BATTING = DATA_DIR / "historic_mlb" / "mlb_batting_data_1950_2025.csv"
HISTORIC_PITCHING = DATA_DIR / "historic_mlb" / "mlb_pitching_data_1950_2025.csv"
SPOTRAC_TRANSACTIONS = DATA_DIR / "salary" / "spotrac_transactions.csv"
PROJECTIONS_DIR = DATA_DIR / "generated" / "historical_values" / "projections"
CROSSWALK = DATA_DIR / "generated" / "player_id_crosswalk.csv"
OUTPUT_DIR = DATA_DIR / "generated" / "historical_values"

# ── Name normalization ────────────────────────────────────────────────────
def _normalize_name(name: str) -> str:
    """Lowercase, strip accents-ish, collapse whitespace."""
    import unicodedata
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    # Remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Collapse whitespace, remove periods/suffixes
    name = " ".join(name.split())
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    name = name.replace(".", "").replace("'", "").replace("-", " ")
    return name


# ══════════════════════════════════════════════════════════════════════════
# Step 1: Load and filter contract data
# ══════════════════════════════════════════════════════════════════════════
def load_contracts(min_aav: float = 2_000_000, min_years: int = 1) -> pd.DataFrame:
    """
    Load FA signings and extensions from Spotrac.

    Spotrac labels ALL player signings as 'fa_signing', including pre-arb
    renewals, arb-avoided deals, arb settlements, minor-league contracts,
    and international signings.  We filter these out using the description
    field and salary thresholds so that only genuine free-agent contracts
    and extensions remain.
    """
    df = pd.read_csv(SPOTRAC_TRANSACTIONS)
    contracts = df[df["transaction_type"].isin(["fa_signing", "extension"])].copy()
    contracts["sign_date"] = pd.to_datetime(contracts["date"])
    contracts["sign_year"] = contracts["sign_date"].dt.year
    contracts["desc_lower"] = contracts["description"].str.lower().fillna("")

    # ── Exclude non-FA contracts masquerading as fa_signing ──────────────
    is_arb_avoid = contracts["desc_lower"].str.contains("avoiding arbitration")
    is_arb_settle = contracts["desc_lower"].str.contains("settling in arbitration")
    is_minor_league = contracts["desc_lower"].str.contains("minor league")
    # Pre-arb renewals at or near MLB minimum (varies by year, ~$700-750K)
    is_pre_arb = (
        contracts["annual_value"].notna()
        & (contracts["annual_value"] < 800_000)
        & (contracts["transaction_type"] == "fa_signing")
    )

    exclude = is_arb_avoid | is_arb_settle | is_minor_league | is_pre_arb
    n_excluded = exclude.sum()
    contracts = contracts[~exclude].copy()

    # ── Standard filters ────────────────────────────────────────────────
    contracts = contracts[
        (contracts["annual_value"] >= min_aav)
        & (contracts["years"] >= min_years)
        & (contracts["sign_year"] >= 2014)  # we have projections from 2013+
        & contracts["annual_value"].notna()
    ].copy()

    contracts["name_key"] = contracts["player_name"].apply(_normalize_name)
    contracts = contracts.sort_values("sign_date").reset_index(drop=True)

    n_fa = (contracts["transaction_type"] == "fa_signing").sum()
    n_ext = (contracts["transaction_type"] == "extension").sum()
    print(f"Loaded {len(contracts)} contracts (FA: {n_fa}, Extensions: {n_ext})")
    print(f"  Excluded {n_excluded} arb-avoided/arb-settled/MiLB/pre-arb deals")
    return contracts


# ══════════════════════════════════════════════════════════════════════════
# Step 2: Load historical actuals (trailing stats)
# ══════════════════════════════════════════════════════════════════════════
def load_historical_actuals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load actual batting and pitching stats with WAR components."""
    bat_cols = [
        "IDfg", "Season", "Name", "Age", "G", "PA", "wOBA", "wRC+", "wRAA",
        "Bat", "BsR", "Fld", "Def", "Pos", "WAR", "Off", "HR", "SB", "AVG",
        "OBP", "SLG", "BB%", "K%",
    ]
    bat = pd.read_csv(HISTORIC_BATTING, usecols=[c for c in bat_cols if c != "missing"])
    # Keep only columns that exist
    bat_cols_present = [c for c in bat_cols if c in bat.columns]
    bat = bat[bat_cols_present].copy()
    bat["name_key"] = bat["Name"].apply(_normalize_name)
    bat["player_type"] = "batter"

    pit_cols = [
        "IDfg", "Season", "Name", "Age", "G", "GS", "IP", "ERA", "FIP",
        "WAR", "K%", "BB%", "SIERA", "HR/9", "K/9", "BB/9", "gmLI",
        "SV", "HLD", "Starting", "Relieving", "Start-IP", "Relief-IP",
    ]
    pit = pd.read_csv(HISTORIC_PITCHING, usecols=[c for c in pit_cols if c != "missing"])
    pit_cols_present = [c for c in pit_cols if c in pit.columns]
    pit = pit[pit_cols_present].copy()
    pit["name_key"] = pit["Name"].apply(_normalize_name)
    # Classify SP vs RP
    pit["role"] = np.where(pit["GS"] > 5, "SP", "RP")
    pit["player_type"] = "pitcher"

    print(f"Historical batting: {len(bat)} rows, pitching: {len(pit)} rows")
    return bat, pit


def get_trailing_stats(
    player_name_key: str,
    sign_year: int,
    bat_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    window: int = 3,
) -> dict:
    """
    Compute trailing weighted stats for a player over the prior `window` seasons.

    Weights: most recent = 3, second = 2, third = 1 (recency-weighted).

    IMPORTANT: Check pitching data first with an IP threshold so that
    NL pitchers who batted pre-DH don't get misclassified as batters.
    """
    results = {}

    # Try pitcher FIRST (avoids misclassifying pitchers who had batting rows)
    pp = pit_df[(pit_df["name_key"] == player_name_key)
                & (pit_df["Season"] >= sign_year - window)
                & (pit_df["Season"] < sign_year)]
    if len(pp) >= 1 and pp["IP"].sum() >= 20:
        pp = pp.sort_values("Season", ascending=False).head(window).copy()
        weights = np.array([3, 2, 1][: len(pp)], dtype=float)
        weights /= weights.sum()

        results["player_type"] = "pitcher"
        results["trailing_ip"] = pp["IP"].sum()
        results["trailing_games"] = pp["G"].sum()
        results["trailing_gs"] = pp["GS"].sum()
        results["role"] = pp.iloc[0]["role"]
        for col in ["ERA", "FIP", "SIERA", "K%", "BB%", "K/9", "BB/9", "HR/9"]:
            if col in pp.columns:
                vals = pp[col].values
                results[f"trailing_{col}"] = np.average(vals, weights=weights) if not np.isnan(vals).all() else np.nan
        results["trailing_WAR"] = pp["WAR"].sum()
        results["trailing_WAR_avg"] = np.average(pp["WAR"].values, weights=weights)
        if "gmLI" in pp.columns:
            results["trailing_gmLI"] = np.average(pp["gmLI"].values, weights=weights)
        results["trailing_SV"] = pp["SV"].sum() if "SV" in pp.columns else 0
        results["trailing_HLD"] = pp["HLD"].sum() if "HLD" in pp.columns else 0
        return results

    # Then try batter
    pb = bat_df[(bat_df["name_key"] == player_name_key)
                & (bat_df["Season"] >= sign_year - window)
                & (bat_df["Season"] < sign_year)]
    if len(pb) >= 1 and pb["PA"].sum() >= 50:
        pb = pb.sort_values("Season", ascending=False).head(window).copy()
        weights = np.array([3, 2, 1][: len(pb)], dtype=float)
        weights /= weights.sum()

        results["player_type"] = "batter"
        results["trailing_pa"] = pb["PA"].sum()
        results["trailing_games"] = pb["G"].sum()
        for col in ["wOBA", "wRC+", "AVG", "OBP", "SLG", "BB%", "K%"]:
            if col in pb.columns:
                vals = pb[col].values
                results[f"trailing_{col}"] = np.average(vals, weights=weights) if not np.isnan(vals).all() else np.nan
        results["trailing_WAR"] = pb["WAR"].sum()
        results["trailing_WAR_avg"] = np.average(pb["WAR"].values, weights=weights)
        for col in ["Bat", "BsR", "Def", "Fld", "Pos", "Off"]:
            if col in pb.columns:
                results[f"trailing_{col}"] = pb[col].sum()
                results[f"trailing_{col}_avg"] = np.average(pb[col].values, weights=weights)
        results["trailing_HR_total"] = pb["HR"].sum() if "HR" in pb.columns else np.nan
        return results

    return {"player_type": "unknown"}


# ══════════════════════════════════════════════════════════════════════════
# Step 3: Match contracts to projections
# ══════════════════════════════════════════════════════════════════════════
def load_projections_for_year(cutoff_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load batter and pitcher projections for a given cutoff year."""
    proj_dir = PROJECTIONS_DIR / f"cutoff_{cutoff_year}"
    bp_path = proj_dir / "batter_predictions.csv"
    pp_path = proj_dir / "pitcher_predictions.csv"

    if not bp_path.exists() or not pp_path.exists():
        # Fall back to the most recent available cutoff
        return pd.DataFrame(), pd.DataFrame()

    bp = pd.read_csv(bp_path)
    bp["name_key"] = bp["Name"].apply(_normalize_name)

    pp = pd.read_csv(pp_path)
    pp["name_key"] = pp["Name"].apply(_normalize_name)

    return bp, pp


def get_projection_features(
    player_name_key: str,
    sign_year: int,
    batter_proj: pd.DataFrame,
    pitcher_proj: pd.DataFrame,
    contract_years: int,
) -> dict:
    """
    Extract projection features for a player at signing time.

    Uses projections for the years covered by the contract (sign_year
    through sign_year + contract_years - 1).
    """
    results = {}

    if batter_proj.empty and pitcher_proj.empty:
        return {"proj_type": "unmatched"}

    # Try batter projections
    if not batter_proj.empty and "name_key" in batter_proj.columns:
        bp = batter_proj[
            (batter_proj["name_key"] == player_name_key)
            & (batter_proj["Year"] >= sign_year)
            & (batter_proj["Year"] < sign_year + contract_years)
        ]
    else:
        bp = pd.DataFrame()
    if len(bp) >= 1:
        results["proj_type"] = "batter"
        results["proj_years_matched"] = len(bp)
        results["proj_wOBA_avg"] = bp["wOBA"].mean()
        results["proj_wOBA_yr1"] = bp.loc[bp["Year"] == bp["Year"].min(), "wOBA"].iloc[0] if len(bp) else np.nan
        results["proj_PA_total"] = bp["PA"].sum()
        results["proj_PA_avg"] = bp["PA"].mean()
        for col in ["AVG", "OBP", "SLG", "BB%", "K%"]:
            if col in bp.columns:
                results[f"proj_{col}_avg"] = bp[col].mean()
        results["proj_HR_total"] = bp["HR"].sum() if "HR" in bp.columns else np.nan
        results["proj_Age_start"] = bp["Age"].min() if "Age" in bp.columns else np.nan
        # IDfg for crosswalk
        results["IDfg"] = bp["IDfg"].iloc[0]
        return results

    # Try pitcher projections
    if not pitcher_proj.empty and "name_key" in pitcher_proj.columns:
        pp = pitcher_proj[
            (pitcher_proj["name_key"] == player_name_key)
            & (pitcher_proj["Year"] >= sign_year)
            & (pitcher_proj["Year"] < sign_year + contract_years)
        ]
    else:
        pp = pd.DataFrame()
    if len(pp) >= 1:
        results["proj_type"] = "pitcher"
        results["proj_years_matched"] = len(pp)
        results["proj_role"] = pp["Role"].mode().iloc[0] if "Role" in pp.columns else "SP"
        for col in ["FIP", "ERA", "K%", "BB%", "SIERA", "K/9", "BB/9", "HR/9"]:
            if col in pp.columns:
                results[f"proj_{col}_avg"] = pp[col].mean()
                results[f"proj_{col}_yr1"] = pp.loc[pp["Year"] == pp["Year"].min(), col].iloc[0]
        results["proj_Age_start"] = pp["Age"].min() if "Age" in pp.columns else np.nan
        results["IDfg"] = pp["IDfg"].iloc[0]
        return results

    return {"proj_type": "unmatched"}


# ══════════════════════════════════════════════════════════════════════════
# Step 4: Build the analysis dataset
# ══════════════════════════════════════════════════════════════════════════
def build_analysis_dataset() -> pd.DataFrame:
    """
    Build a dataset matching each contract to projection + trailing features.
    """
    contracts = load_contracts(min_aav=2_000_000, min_years=1)
    bat_hist, pit_hist = load_historical_actuals()

    # Cache loaded projections
    proj_cache: dict[int, tuple] = {}

    rows = []
    for _, contract in contracts.iterrows():
        sign_year = int(contract["sign_year"])
        name_key = contract["name_key"]
        contract_years = int(contract["years"])

        # Load projections for this cutoff year
        # For a signing in year Y, use cutoff_Y (the projection set
        # that would have been available at signing time).
        cutoff = sign_year
        if cutoff not in proj_cache:
            proj_cache[cutoff] = load_projections_for_year(cutoff)
        bp, pp = proj_cache[cutoff]

        # Get projection features
        proj = get_projection_features(name_key, sign_year, bp, pp, contract_years)

        # Get trailing actual stats
        trailing = get_trailing_stats(name_key, sign_year, bat_hist, pit_hist)

        # Combine
        row = {
            "player_name": contract["player_name"],
            "name_key": name_key,
            "sign_year": sign_year,
            "sign_date": contract["sign_date"],
            "team": contract["team"],
            "transaction_type": contract["transaction_type"],
            "years": contract_years,
            "total_value": contract["total_value"],
            "annual_value": contract["annual_value"],
        }
        row.update({f"proj_{k}" if not k.startswith("proj_") else k: v
                     for k, v in proj.items()})
        row.update(trailing)
        rows.append(row)

    df = pd.DataFrame(rows)
    matched = df[df.get("proj_type", pd.Series(dtype=str)) != "unmatched"]
    print(f"\nBuilt dataset: {len(df)} contracts, {len(matched)} matched to projections")
    print(f"  Batters: {(df['player_type'] == 'batter').sum()}")
    print(f"  Pitchers: {(df['player_type'] == 'pitcher').sum()}")
    print(f"  Unmatched: {(df.get('proj_type', '') == 'unmatched').sum()}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# Step 5: Regression analysis
# ══════════════════════════════════════════════════════════════════════════
def analyze_contracts(df: pd.DataFrame) -> dict:
    """
    Run regression analyses on the contract dataset.

    Returns a dict of results/summaries for display and export.
    """
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    results = {}

    # ── Filter to matched contracts ──────────────────────────────────────
    matched = df[df["proj_type"] != "unmatched"].copy()
    matched["log_aav"] = np.log(matched["annual_value"])

    # Inflation-adjust AAV to 2025 dollars (4% annual)
    matched["aav_2025"] = matched["annual_value"] * (1.04 ** (2025 - matched["sign_year"]))
    matched["log_aav_2025"] = np.log(matched["aav_2025"])

    print(f"\n{'='*70}")
    print("CONTRACT VALUE ANALYSIS")
    print(f"{'='*70}")
    print(f"Matched contracts: {len(matched)}")

    # ── 5A: Batter analysis ──────────────────────────────────────────────
    batters = matched[matched["player_type"] == "batter"].copy()
    print(f"\n{'─'*50}")
    print(f"BATTER CONTRACTS: {len(batters)}")
    print(f"{'─'*50}")

    if len(batters) >= 20:
        # Show correlations with AAV
        bat_features = ["trailing_WAR", "trailing_WAR_avg", "trailing_wRC+",
                        "trailing_wOBA", "trailing_Bat", "trailing_Def",
                        "trailing_BsR", "trailing_Off", "trailing_Bat_avg",
                        "trailing_Def_avg", "proj_wOBA_avg", "proj_PA_avg",
                        "proj_Age_start", "years"]
        available = [f for f in bat_features if f in batters.columns]

        print("\nCorrelation with AAV (inflation-adjusted to 2025$):")
        print("─" * 45)
        corrs = {}
        for feat in available:
            valid = batters[[feat, "aav_2025"]].dropna()
            if len(valid) >= 10:
                corr = valid[feat].corr(valid["aav_2025"])
                corrs[feat] = corr
                print(f"  {feat:30s}  r = {corr:+.3f}")
        results["batter_correlations"] = corrs

        # Regression: offensive vs defensive WAR components
        bat_reg_features = [f for f in ["trailing_Bat_avg", "trailing_Def_avg",
                                         "trailing_BsR", "proj_Age_start", "years"]
                            if f in batters.columns]
        reg_data = batters[bat_reg_features + ["aav_2025"]].dropna()
        if len(reg_data) >= 20:
            X = reg_data[bat_reg_features].values
            y = reg_data["aav_2025"].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)
            cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

            print(f"\nMultiple Regression: AAV ~ Bat + Def + BsR + Age + Years")
            print(f"  R² (5-fold CV): {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
            print(f"  Coefficients (per 1 SD):")
            for feat, coef in zip(bat_reg_features, model.coef_):
                print(f"    {feat:30s}  ${coef:>+14,.0f}")
            results["batter_regression"] = dict(zip(bat_reg_features, model.coef_))
            results["batter_r2"] = cv_r2.mean()

        # Key insight: $/WAR from offense vs defense
        # Compute marginal $/unit for batting runs vs defensive runs
        bat_off_def = [f for f in ["trailing_Bat", "trailing_Def", "trailing_pa"]
                       if f in batters.columns]
        reg_data2 = batters[bat_off_def + ["aav_2025", "years"]].dropna()
        if len(reg_data2) >= 20 and "trailing_Bat" in reg_data2.columns:
            # Per-year values
            reg_data2["bat_per_yr"] = reg_data2["trailing_Bat"] / 3  # trailing 3 yr avg
            reg_data2["def_per_yr"] = reg_data2["trailing_Def"] / 3
            X = reg_data2[["bat_per_yr", "def_per_yr"]].values
            y = reg_data2["aav_2025"].values
            model2 = LinearRegression()
            model2.fit(X, y)
            print(f"\n  ** KEY INSIGHT: $/run by component (raw regression) **")
            print(f"    Batting runs:   ${model2.coef_[0]:>+12,.0f} per run above avg")
            print(f"    Defensive runs: ${model2.coef_[1]:>+12,.0f} per run above avg")
            if model2.coef_[0] != 0:
                ratio = model2.coef_[1] / model2.coef_[0]
                print(f"    Def/Bat ratio:  {ratio:.2f}x")
                print(f"    → Defense is valued at {ratio*100:.0f}% of offense in FA contracts")
                results["def_bat_ratio"] = ratio

    # ── 5B: Pitcher analysis ─────────────────────────────────────────────
    pitchers = matched[matched["player_type"] == "pitcher"].copy()
    print(f"\n{'─'*50}")
    print(f"PITCHER CONTRACTS: {len(pitchers)}")
    print(f"{'─'*50}")

    if len(pitchers) >= 15:
        # Split by role
        sp = pitchers[pitchers.get("role", pd.Series(dtype=str)) == "SP"]
        rp = pitchers[pitchers.get("role", pd.Series(dtype=str)) == "RP"]
        print(f"  SP: {len(sp)}, RP: {len(rp)}")

        pit_features = ["trailing_WAR", "trailing_WAR_avg", "trailing_FIP",
                        "trailing_ERA", "trailing_ip", "trailing_K%",
                        "trailing_gmLI", "trailing_SV", "proj_FIP_avg",
                        "proj_ERA_avg", "proj_Age_start", "years"]
        available = [f for f in pit_features if f in pitchers.columns]

        print("\nCorrelation with AAV (all pitchers, 2025$):")
        print("─" * 45)
        pit_corrs = {}
        for feat in available:
            valid = pitchers[[feat, "aav_2025"]].dropna()
            if len(valid) >= 10:
                corr = valid[feat].corr(valid["aav_2025"])
                pit_corrs[feat] = corr
                print(f"  {feat:30s}  r = {corr:+.3f}")
        results["pitcher_correlations"] = pit_corrs

        # SP-specific
        if len(sp) >= 15:
            print(f"\n  SP correlations:")
            for feat in available:
                valid = sp[[feat, "aav_2025"]].dropna()
                if len(valid) >= 10:
                    corr = valid[feat].corr(valid["aav_2025"])
                    print(f"    {feat:30s}  r = {corr:+.3f}")

        # RP-specific
        if len(rp) >= 10:
            print(f"\n  RP correlations:")
            for feat in available:
                valid = rp[[feat, "aav_2025"]].dropna()
                if len(valid) >= 5:
                    corr = valid[feat].corr(valid["aav_2025"])
                    print(f"    {feat:30s}  r = {corr:+.3f}")

        # $/WAR by role
        print(f"\n  ** $/WAR by pitcher role (trailing WAR avg) **")
        for role_label, subset in [("SP", sp), ("RP", rp)]:
            valid = subset[["trailing_WAR_avg", "aav_2025"]].dropna()
            if len(valid) >= 5:
                avg_aav = valid["aav_2025"].mean()
                avg_war = valid["trailing_WAR_avg"].mean()
                if avg_war > 0:
                    dollar_per_war = avg_aav / avg_war
                    print(f"    {role_label}: avg AAV=${avg_aav/1e6:.1f}M, "
                          f"avg WAR={avg_war:.2f}, "
                          f"$/WAR=${dollar_per_war/1e6:.1f}M")
                    results[f"{role_label.lower()}_dollar_per_war"] = dollar_per_war

    # ── 5C: FA vs Extension comparison ───────────────────────────────────
    print(f"\n{'─'*50}")
    print("FA SIGNINGS vs EXTENSIONS")
    print(f"{'─'*50}")

    for ptype in ["batter", "pitcher"]:
        subset = matched[matched["player_type"] == ptype]
        fa = subset[subset["transaction_type"] == "fa_signing"]
        ext = subset[subset["transaction_type"] == "extension"]
        if len(fa) >= 5 and len(ext) >= 5:
            print(f"\n  {ptype.upper()}:")
            print(f"    FA signings (n={len(fa)}):  avg AAV=${fa['aav_2025'].mean()/1e6:.1f}M, "
                  f"avg years={fa['years'].mean():.1f}, "
                  f"avg age={fa['proj_Age_start'].mean():.1f}" if 'proj_Age_start' in fa.columns else "")
            print(f"    Extensions (n={len(ext)}):   avg AAV=${ext['aav_2025'].mean()/1e6:.1f}, "
                  f"avg years={ext['years'].mean():.1f}, "
                  f"avg age={ext['proj_Age_start'].mean():.1f}" if 'proj_Age_start' in ext.columns else "")

            # $/WAR comparison
            for label, sub in [("FA", fa), ("Extension", ext)]:
                valid = sub[["trailing_WAR_avg", "aav_2025"]].dropna()
                if len(valid) >= 5 and valid["trailing_WAR_avg"].mean() > 0:
                    ratio = valid["aav_2025"].mean() / valid["trailing_WAR_avg"].mean()
                    print(f"    {label} $/WAR: ${ratio/1e6:.1f}M")

    # ── 5D: Position group analysis ──────────────────────────────────────
    print(f"\n{'─'*50}")
    print("$/WAR BY POSITION GROUP")
    print(f"{'─'*50}")

    # Merge with historical data to get position info
    bat_hist_latest = pd.read_csv(HISTORIC_BATTING, usecols=["IDfg", "Season", "Name"])
    # We don't have clean position in the contract data, so use proxy from trailing stats
    # Instead, compute $/WAR for the entire batter group vs pitcher subgroups
    for ptype, label in [("batter", "Batters"), ("pitcher", "Pitchers")]:
        subset = matched[matched["player_type"] == ptype]
        valid = subset[["trailing_WAR_avg", "aav_2025", "years"]].dropna()
        if len(valid) >= 10:
            # Bin by trailing WAR level
            bins = [(-np.inf, 1), (1, 2), (2, 3), (3, 5), (5, np.inf)]
            print(f"\n  {label} $/WAR by performance tier:")
            for lo, hi in bins:
                tier = valid[(valid["trailing_WAR_avg"] >= lo) & (valid["trailing_WAR_avg"] < hi)]
                if len(tier) >= 3:
                    avg_aav = tier["aav_2025"].mean()
                    avg_war = tier["trailing_WAR_avg"].mean()
                    dollar_per_war = avg_aav / avg_war if avg_war > 0 else 0
                    print(f"    WAR {lo:+.0f} to {hi:+.0f}: n={len(tier):3d}, "
                          f"avg AAV=${avg_aav/1e6:6.1f}M, "
                          f"avg WAR={avg_war:.2f}, "
                          f"$/WAR=${dollar_per_war/1e6:.1f}M")

    # ── 5E: WAR component value decomposition ────────────────────────────
    print(f"\n{'─'*50}")
    print("WAR COMPONENT VALUE DECOMPOSITION (Batters)")
    print(f"{'─'*50}")

    bat_decomp = matched[
        (matched["player_type"] == "batter")
        & matched["trailing_Bat_avg"].notna()
        & matched["trailing_Def_avg"].notna()
    ].copy()

    if len(bat_decomp) >= 20:
        # Compute offensive WAR proxy and defensive WAR proxy
        # (using runs-above-average, ~10 runs/WAR)
        bat_decomp["off_war_proxy"] = (bat_decomp["trailing_Bat_avg"] +
                                        bat_decomp.get("trailing_BsR", 0)) / 10
        bat_decomp["def_war_proxy"] = bat_decomp["trailing_Def_avg"] / 10

        for component, label in [("off_war_proxy", "Offensive WAR"),
                                  ("def_war_proxy", "Defensive WAR"),
                                  ("trailing_WAR_avg", "Total WAR")]:
            if component in bat_decomp.columns:
                valid = bat_decomp[[component, "aav_2025"]].dropna()
                if len(valid) >= 10:
                    corr = valid[component].corr(valid["aav_2025"])
                    print(f"  {label:20s} correlation with AAV: r = {corr:+.3f}")

        # Direct regression
        features = ["off_war_proxy", "def_war_proxy", "proj_Age_start"]
        available = [f for f in features if f in bat_decomp.columns]
        reg_data = bat_decomp[available + ["aav_2025"]].dropna()
        if len(reg_data) >= 15:
            X = reg_data[available].values
            y = reg_data["aav_2025"].values
            model = LinearRegression()
            model.fit(X, y)
            print(f"\n  Regression: AAV ~ Offensive_WAR + Defensive_WAR + Age")
            print(f"  R² = {model.score(X, y):.3f}")
            for feat, coef in zip(available, model.coef_):
                print(f"    {feat:20s}  ${coef:>+14,.0f} per unit")
            if "off_war_proxy" in available and "def_war_proxy" in available:
                off_idx = available.index("off_war_proxy")
                def_idx = available.index("def_war_proxy")
                if model.coef_[off_idx] != 0:
                    ratio = model.coef_[def_idx] / model.coef_[off_idx]
                    print(f"\n  *** DEFENSIVE WAR IS VALUED AT {ratio*100:.0f}% OF OFFENSIVE WAR ***")
                    results["off_def_war_ratio"] = ratio

    return results


# ══════════════════════════════════════════════════════════════════════════
# Step 6: Summary and recommendations
# ══════════════════════════════════════════════════════════════════════════
def print_summary(results: dict):
    """Print actionable summary for the trade value system."""
    print(f"\n{'='*70}")
    print("ACTIONABLE FINDINGS FOR TRADE VALUE SYSTEM")
    print(f"{'='*70}")

    if "def_bat_ratio" in results:
        ratio = results["def_bat_ratio"]
        print(f"\n1. DEFENSIVE DISCOUNT:")
        print(f"   FA market values defensive runs at {ratio*100:.0f}% of batting runs.")
        print(f"   Recommendation: weight defensive WAR at {max(0.3, min(0.8, ratio)):.2f}x "
              f"in trade value calculations.")

    if "sp_dollar_per_war" in results and "rp_dollar_per_war" in results:
        sp_val = results["sp_dollar_per_war"]
        rp_val = results["rp_dollar_per_war"]
        print(f"\n2. PITCHER ROLE PREMIUMS:")
        print(f"   SP $/WAR in FA: ${sp_val/1e6:.1f}M")
        print(f"   RP $/WAR in FA: ${rp_val/1e6:.1f}M")
        if sp_val > 0:
            print(f"   RP premium over SP: {rp_val/sp_val:.2f}x")

    if "off_def_war_ratio" in results:
        ratio = results["off_def_war_ratio"]
        print(f"\n3. WAR COMPONENT WEIGHTING:")
        print(f"   Regression confirms defense valued at {ratio*100:.0f}% of offense.")
        print(f"   This means: a 2-WAR player from bat should be valued higher than")
        print(f"   a 2-WAR player from defense in trade calculations.")

    print(f"\n4. CONFIDENCE DISCOUNTING:")
    print(f"   Players like Bliss (minimal MLB track record) should have their")
    print(f"   projected WAR regressed toward replacement level proportional to")
    print(f"   sample size. Apply universal confidence scaling, not just for prospects.")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("Free Agency & Extension Contract Analysis")
    print("=" * 50)

    # Build dataset
    df = build_analysis_dataset()

    # Save intermediate dataset
    output_path = OUTPUT_DIR / "fa_contract_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved analysis dataset to {output_path}")

    # Run analysis
    results = analyze_contracts(df)

    # Print summary
    print_summary(results)

    return df, results


if __name__ == "__main__":
    main()
