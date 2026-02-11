"""
Trade Value Calculations
========================

Calculates trade value for each player:
    1. Sum projected WAR over remaining team-control years
    2. Convert WAR to dollar value
    3. Subtract total contract cost → trade value (surplus)
    4. For recent prospects with limited MLB experience, blend the
       performance-based value with their prospect grade value

Usage:
    from value_determination.trade_value import (
        calculate_trade_values, add_trade_ranking_metrics
    )
"""

import pandas as pd
import numpy as np
import re

from .config import Config, logger, CURRENT_YEAR


# ---------------------------------------------------------------------------
# Prospect dollar value from FV grade + ranking
# ---------------------------------------------------------------------------

def _prospect_dollar_value(fv, rank) -> float | None:
    """
    Convert a FanGraphs FV grade and ranking into a dollar value.

    Args:
        fv: Future Value grade (40-70 scale, may include '+' suffix).
        rank: Top-100 ranking (1-100) or NaN for org-only prospects.

    Returns:
        Dollar value, or None if inputs are invalid.
    """
    if pd.isna(fv):
        return None

    try:
        # Handle plus grades (e.g. '55+' → 57.5)
        fv = float(str(fv).replace("+", "")) + (2.5 if "+" in str(fv) else 0)

        # Find the highest FV tier at or below this grade
        tiers = Config.Prospects.FV_BASE_VALUES
        valid = [k for k in tiers if k <= fv]
        base_tier = max(valid) if valid else min(tiers)
        base_value = tiers[base_tier]

        # Apply top-100 rank bonus (non-top-100 get 1.0×)
        if pd.notna(rank):
            return base_value * Config.Prospects.calculate_rank_adjustment(float(rank))
        return base_value

    except Exception as e:
        logger.warning(f"Could not calculate prospect value (FV={fv}, rank={rank}): {e}")
        return None


# ---------------------------------------------------------------------------
# Contract option analysis (opt-outs, team/player options)
# ---------------------------------------------------------------------------

def analyze_contract_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine each player's FA year accounting for contract options.

    Adds columns:
        FA_Year           – first explicit Free Agent year
        probable_fa_year  – FA year after evaluating options
        earliest_fa_year  – earliest possible FA year (any option exercised)
    """
    result = df.copy()

    # --- Base FA year (first year with 'Free Agent' status) ------------------
    fa_years = (
        result.loc[result["Status"] == "Free Agent"]
        .groupby("IDfg")["Year"]
        .min()
        .rename("FA_Year")
    )
    result = result.merge(fa_years, on="IDfg", how="left")

    # Infer FA year for players without an explicit Free Agent row
    missing = result.loc[result["FA_Year"].isna(), "IDfg"].unique()
    if len(missing):
        logger.info(f"Inferring FA year for {len(missing)} players without explicit FA status")
        for pid in missing:
            rows = result[result["IDfg"] == pid]
            contract = rows.loc[
                (rows["contract_value"].notna() & (rows["contract_value"] > 0))
                | rows["Status"].isin(["Signed", "Unknown"])
            ]["Year"]
            if len(contract):
                result.loc[result["IDfg"] == pid, "FA_Year"] = contract.max() + 1

    result["probable_fa_year"] = result["FA_Year"]

    # --- Evaluate options year-by-year --------------------------------------
    OPTION_TYPES = {"Player Option", "Team Option", "Mutual Option",
                    "Vesting Option", "Opt-Out"}

    option_min = (
        result.loc[result["Status"].isin(OPTION_TYPES)]
        .groupby("IDfg")["Year"]
        .min()
        .rename("earliest_fa_year")
    )
    result = result.merge(option_min, on="IDfg", how="left")
    result["earliest_fa_year"] = result["earliest_fa_year"].fillna(result["FA_Year"])

    for pid in result.loc[result["Status"].isin(OPTION_TYPES), "IDfg"].unique():
        pdata = result[result["IDfg"] == pid].sort_values("Year")
        probable_fa = pdata["FA_Year"].iloc[0]

        for _, row in pdata[pdata["Status"].isin(OPTION_TYPES)].iterrows():
            yr, status, surplus = row["Year"], row["Status"], row["surplus_value"]
            if yr >= probable_fa:
                break

            declined = False
            if status in ("Player Option", "Opt-Out"):
                declined = pd.notna(surplus) and surplus > 0
            elif status == "Team Option":
                declined = pd.notna(surplus) and surplus < 0
            else:  # Mutual / Vesting
                declined = pd.notna(surplus) and surplus < 0

            if declined:
                probable_fa = yr
                break

        result.loc[result["IDfg"] == pid, "probable_fa_year"] = probable_fa

    return result


# ---------------------------------------------------------------------------
# Core trade value calculation
# ---------------------------------------------------------------------------

def calculate_trade_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trade value for every player.

    Trade value = Σ(projected WAR dollars) – Σ(contract cost)
    summed over each remaining team-control year (CURRENT_YEAR … FA_Year-1).

    Additional rules:
        • Arb/Pre-Arb players are floored at 0 (team can non-tender).
        • Signed players with ≤2 years left are floored at 0.
        • Recent prospects get a blended value (see _apply_prospect_adjustments).
    """
    result = df.copy()
    result["trade_value"] = np.nan

    for pid in result["IDfg"].unique():
        pmask = result["IDfg"] == pid
        pdata = result.loc[pmask]

        fa_year = pdata["probable_fa_year"].iloc[0]
        if pd.isna(fa_year):
            fa_year = pdata["FA_Year"].iloc[0]
        if pd.isna(fa_year):
            continue

        control = pdata[
            (pdata["Year"] >= CURRENT_YEAR)
            & (pdata["Year"] < fa_year)
            & pdata["Base_Value"].notna()
            & pdata["contract_value"].notna()
        ]

        if control.empty:
            continue

        war_dollars = control["Base_Value"].sum()
        contract_cost = control["contract_value"].sum()
        trade_value = war_dollars - contract_cost

        # Arb/Pre-Arb players can be non-tendered, so floor at 0
        is_team_control = pdata["Status"].str.contains("Arb|Pre-Arb", regex=True).any()
        if is_team_control:
            trade_value = max(0, trade_value)

        result.loc[pmask & (result["Year"] >= CURRENT_YEAR), "trade_value"] = trade_value

    logger.info(
        f"Trade values: {result['trade_value'].notna().sum()} players, "
        f"avg=${result['trade_value'].mean():,.0f}, "
        f"median=${result['trade_value'].median():,.0f}"
    )

    # Prospect adjustments
    prospect_file = Config.Paths.PROSPECT_FILE
    if prospect_file.exists():
        result = _apply_prospect_adjustments(result, prospect_file)
    else:
        logger.warning(f"Prospect file not found: {prospect_file}")

    return result


# ---------------------------------------------------------------------------
# Prospect blending
# ---------------------------------------------------------------------------

def _apply_prospect_adjustments(result_df: pd.DataFrame, prospect_file) -> pd.DataFrame:
    """
    Blend prospect value with performance-based trade value for young players.

    For a player with limited MLB experience their trade value is:
        trade_value = (MLB weight × performance value) + (prospect weight × prospect value)

    The prospect weight decays linearly from 1.0 (no MLB games) to 0.0 once the
    player reaches the experience threshold defined in Config.Prospects.

    Only prospects ranked within the last 3 years are considered.
    """
    prospect_df = pd.read_csv(prospect_file)
    logger.info(
        f"Loaded prospect data: {len(prospect_df)} records, "
        f"years {prospect_df['year'].min():.0f}–{prospect_df['year'].max():.0f}"
    )

    # Extract MLB ID from prospect URL
    prospect_df["prospect_mlb_id"] = prospect_df["prospect_url"].apply(
        lambda u: int(u.split("-")[-1]) if pd.notna(u) and u else None
    )

    # Keep each prospect's most recent ranking
    latest = (
        prospect_df[prospect_df["prospect_mlb_id"].notna()]
        .sort_values("year", ascending=False)
        .drop_duplicates("prospect_mlb_id")
    )
    logger.info(f"Unique prospects: {len(latest)}")

    # Only apply adjustments to recent prospects (ranked in last 3 years)
    recent_cutoff = CURRENT_YEAR - 3
    latest = latest[latest["year"] >= recent_cutoff]
    logger.info(f"Recent prospects (since {recent_cutoff}): {len(latest)}")

    if latest.empty:
        return result_df

    # Pre-compute career MLB games per player
    career_games = (
        result_df[
            (result_df["Year"] < CURRENT_YEAR)
            & (result_df["G_bat"].notna() | result_df["G_pit"].notna() | result_df["GS"].notna())
        ]
        .groupby("IDfg")
        .agg(G_bat=("G_bat", "sum"), G_pit=("G_pit", "sum"),
             GS=("GS", "sum"), position_group=("position_group", "first"))
        .reset_index()
    )

    # Determine matching key
    use_mlbam = "mlbam_id" in result_df.columns

    if use_mlbam:
        merge_cols = {"left_on": "mlbam_id", "right_on": "prospect_mlb_id"}
    else:
        logger.warning("mlbam_id not available — falling back to name matching")
        _norm = lambda s: re.sub(r"[^A-Z]", "", s.upper()) if pd.notna(s) else None
        result_df["_name_key"] = result_df["Name"].apply(_norm)
        latest["_name_key"] = latest["name"].apply(_norm)
        merge_cols = {"left_on": "_name_key", "right_on": "_name_key"}

    # Identify prospect rows in current-year data that already have trade values
    candidates = result_df[
        (result_df["Year"] >= CURRENT_YEAR)
        & result_df["trade_value"].notna()
    ].drop_duplicates("IDfg" if use_mlbam else "_name_key")

    matched = candidates.merge(
        latest[["prospect_mlb_id", "name", "year", "rank", "grade_overall",
                "top_100"] + (["_name_key"] if not use_mlbam else [])],
        **merge_cols, how="inner"
    )
    logger.info(f"Matched {len(matched)} prospects with trade values")

    adjusted = 0
    for _, row in matched.iterrows():
        pid = row["IDfg"]

        # Determine position type
        pos = row.get("position_group", "batter")
        pos_type = "sp" if pos == "SP" else ("rp" if pos == "RP" else "batter")

        # Career games
        if pid in career_games["IDfg"].values:
            cg = career_games[career_games["IDfg"] == pid].iloc[0]
            gs = cg.get("GS", 0) or 0
            g_pit = cg.get("G_pit", 0) or 0
            if pos_type == "sp":
                games = gs
            elif pos_type == "rp":
                games = g_pit - gs if gs < 50 else gs
                if gs >= 50:
                    pos_type = "sp"
            else:
                games = cg.get("G_bat", 0) or 0
        else:
            games = 0

        prospect_wt = Config.Prospects.calculate_prospect_weight(games, pos_type)
        if prospect_wt == 0.0:
            continue  # established — no adjustment needed

        # Determine which rank to use
        prospect_year = row.get("year", None)
        org_rank = row.get("rank", None)
        top_100 = row.get("top_100", None)

        # For current-year lists that only have top-100 (no org lists yet)
        if prospect_year == CURRENT_YEAR and pd.notna(org_rank):
            top_100 = org_rank
            org_rank = None

        prospect_val = _prospect_dollar_value(row.get("grade_overall"), top_100)
        if prospect_val is None:
            continue

        perf_value = row["trade_value"]
        blended = (1 - prospect_wt) * perf_value + prospect_wt * prospect_val

        # Sanity: don't let a negative-value player jump to 10× their absolute value
        if perf_value < 0 and blended > abs(perf_value) * 10:
            logger.warning(
                f"Skipping prospect adjustment for {row.get('Name', 'Unknown')}: "
                f"would create unrealistic value (perf=${perf_value:,.0f}, blended=${blended:,.0f})"
            )
            continue

        mask = (result_df["IDfg"] == pid) & (result_df["Year"] >= CURRENT_YEAR)
        result_df.loc[mask, "trade_value"] = blended
        adjusted += 1

        logger.debug(
            f"  {row.get('Name', f'ID{pid}')}: FV={row.get('grade_overall')}, "
            f"games={games}, pw={prospect_wt:.2f}, "
            f"prospect=${prospect_val:,.0f}, final=${blended:,.0f}"
        )

    logger.info(f"Applied prospect adjustments to {adjusted} players")

    # Clean up temp column
    result_df.drop("_name_key", axis=1, errors="ignore", inplace=True)

    return result_df


# ---------------------------------------------------------------------------
# Trade ranking metrics
# ---------------------------------------------------------------------------

def add_trade_ranking_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-computed ranking metrics for each player.

    Metrics (all merged back on IDfg):
        contract_war, avg_war, total_contract, avg_contract, total_surplus,
        years_control, control_through, contract_base_value,
        total_future_war, total_future_value,
        total_war, total_value, historical_war, historical_value
    """
    result = df.copy()
    rows = []

    for pid in result["IDfg"].unique():
        p = result[result["IDfg"] == pid].sort_values("Year")

        fa_year = p["probable_fa_year"].iloc[0]
        if pd.isna(fa_year):
            fa_year = p["FA_Year"].iloc[0]

        control = p[(p["Year"] >= CURRENT_YEAR) & (p["Year"] < fa_year)]
        future = p[p["Year"] >= CURRENT_YEAR]
        past = p[p["Year"] < CURRENT_YEAR]
        n = len(control)

        rows.append({
            "IDfg": pid,
            "contract_war": round(control["WAR"].sum(), 1),
            "contract_base_value": round(control["Base_Value"].sum(), 1),
            "avg_war": round(control["WAR"].mean(), 2) if n else 0,
            "total_contract": round(control["contract_value"].sum(), 1),
            "avg_contract": round(control["contract_value"].mean(), 2) if n else 0,
            "total_surplus": round(control["surplus_value"].sum(), 1),
            "years_control": n,
            "control_through": fa_year - 1 if pd.notna(fa_year) else None,
            "total_future_war": round(future.loc[future["WAR"] > 0, "WAR"].sum(), 1),
            "total_future_value": round(future["Base_Value"].sum(), 1),
            "total_war": round(past["WAR"].sum() + future.loc[future["WAR"] > 0, "WAR"].sum(), 1),
            "total_value": round(p["Base_Value"].sum(), 1),
            "historical_war": round(past["WAR"].sum(), 1),
            "historical_value": round(past["Base_Value"].sum(), 1),
        })

    metrics = pd.DataFrame(rows)
    return result.merge(metrics, on="IDfg", how="left")


# ---------------------------------------------------------------------------
# Update prospect histories with MLB-debut flags
# ---------------------------------------------------------------------------

def update_prospect_mlb_status(export_data: pd.DataFrame) -> None:
    """
    Mark prospects who have reached the majors in prospect_histories.csv.

    Adds a boolean ``has_mlb`` column so the web-app can distinguish
    prospects who have debuted from those who have not.
    """
    path = Config.Paths.GENERATED_DIR / "MiLB" / "prospect_histories.csv"
    if not path.exists():
        logger.warning(f"Prospect file not found: {path} — run generate_prospect_histories.py first")
        return

    try:
        pf = pd.read_csv(path)
        mlb_ids = set(str(i) for i in export_data["IDfg"].unique())
        pf["IDfg"] = pf["IDfg"].astype(str)
        pf["has_mlb"] = pf["IDfg"].isin(mlb_ids)
        pf.to_csv(path, index=False)

        total = pf["IDfg"].nunique()
        debuted = pf.loc[pf["has_mlb"], "IDfg"].nunique()
        logger.info(f"Prospect MLB status: {debuted}/{total} ({100*debuted/total:.1f}%) have MLB data")

    except Exception as e:
        logger.error(f"Failed to update prospect MLB status: {e}")
        raise
