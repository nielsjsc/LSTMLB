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

import glob
import pandas as pd
import numpy as np
import re
from pathlib import Path

from .config import Config, logger, CURRENT_YEAR
from core.name_utils import name_key_alpha_only as _name_key_norm

# Register data directory for mlbam → IDfg crosswalk
_REGISTER_DATA_DIR = Config.Paths.DATA_DIR / 'register' / 'data'


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

def analyze_contract_options(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """
    Determine each player's FA year accounting for contract options.

    Adds columns:
        FA_Year           – first explicit Free Agent year
        probable_fa_year  – FA year after evaluating options
        earliest_fa_year  – earliest possible FA year (any option exercised)
    """
    if current_year is None:
        current_year = CURRENT_YEAR
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
            # Only consider explicitly-signed contract years (Payroll from Spotrac),
            # not Pre-Arb/Arb minimum salaries imputed by calculate_contract_value.
            contract = rows.loc[
                rows["Status"].isin(["Signed", "Team Option", "Player Option",
                                     "Mutual Option", "Vesting Option", "Opt-Out",
                                     "Unknown"])
            ]["Year"]
            if len(contract):
                result.loc[result["IDfg"] == pid, "FA_Year"] = contract.max() + 1
            else:
                # No signed contract rows at all — player is likely already FA.
                # Set FA year to current year so trade value logic treats them
                # as immediately available.
                result.loc[result["IDfg"] == pid, "FA_Year"] = float(current_year)

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
                # Player opts out only if they can earn MORE on the open market
                # than the remaining guaranteed contract money.
                # Compare: total projected WAR dollars vs total remaining contract
                remaining = pdata[(pdata["Year"] >= yr) & (pdata["Year"] < probable_fa)]
                remaining_contract = remaining["contract_value"].sum()
                remaining_war_value = remaining["Base_Value"].sum()
                declined = remaining_war_value > remaining_contract
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

def calculate_trade_values(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """
    Calculate trade value for every player.

    Trade value = Σ(projected WAR dollars) – Σ(contract cost)
    summed over each remaining team-control year (current_year … FA_Year-1).

    Additional rules:
        • Arb/Pre-Arb players are floored at 0 (team can non-tender).
        • Signed players with ≤2 years left are floored at 0.
        • Recent prospects get a confidence-blended value (see _apply_confidence_adjustments).
    """
    if current_year is None:
        current_year = CURRENT_YEAR

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
            (pdata["Year"] >= current_year)
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

        future_mask = pmask & (result["Year"] >= current_year)
        result.loc[future_mask, "trade_value"] = trade_value
        result.loc[future_mask, "_proj_value_sum"] = war_dollars
        result.loc[future_mask, "_contract_sum"] = contract_cost

    logger.info(
        f"Trade values: {result['trade_value'].notna().sum()} players, "
        f"avg=${result['trade_value'].mean():,.0f}, "
        f"median=${result['trade_value'].median():,.0f}"
    )

    # Prospect adjustments → confidence-based blending
    prospect_file = Config.Paths.PROSPECT_FILE
    if prospect_file.exists():
        result = _apply_confidence_adjustments(result, prospect_file, current_year=current_year)
    else:
        logger.warning(f"Prospect file not found: {prospect_file}")
        # Still populate confidence columns for all players
        result["projection_confidence"] = 1.0
        result["raw_trade_value"] = result["trade_value"]

    return result


# ---------------------------------------------------------------------------
# Confidence-based trade-value blending (replaces legacy prospect blending)
# ---------------------------------------------------------------------------

def _apply_confidence_adjustments(result_df: pd.DataFrame,
                                  prospect_file,
                                  current_year: int | None = None) -> pd.DataFrame:
    """
    Blend prospect-grade value with performance-based trade value using
    a stabilisation-based confidence score.

    For every player:
        1. Compute ``projection_confidence`` from career MLB games vs
           ``TradeConfidence.STABILIZATION_GAMES`` thresholds.
        2. Store the performance-based trade value as ``raw_trade_value``.

    For recent prospects (FV grade available, ranked within the recency window):
        3. Compute a prospect-grade dollar value from FV + rank.
        4. Apply as a floor: ``trade_value = max(perf_value, prospect_value * prospect_weight)``
           where prospect_weight fades from 1.0 (0 games) to 0.0 (experience threshold).

    Players without prospect data or with full confidence are untouched.

    Config reference:
        ``Config.TradeConfidence`` — thresholds, floor, FV prior WAR, recency.
        ``Config.Prospects``       — FV_BASE_VALUES, rank adjustments.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

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

    # Only consider recent prospects (within recency window)
    recency = Config.TradeConfidence.PROSPECT_RECENCY_YEARS
    recent_cutoff = current_year - recency
    latest = latest[latest["year"] >= recent_cutoff]
    logger.info(f"Recent prospects (since {recent_cutoff}): {len(latest)}")

    # ── Career games per player ──────────────────────────────────────────
    # Column names vary depending on upstream pipeline steps; use whatever
    # game-count columns are available.
    has_g_bat = "G_bat" in result_df.columns
    has_g_pit = "G_pit" in result_df.columns
    has_gs    = "GS"    in result_df.columns
    has_g     = "G"     in result_df.columns  # fallback: un-suffixed G

    game_filter = result_df["Year"] < current_year
    if has_g_bat:
        game_filter = game_filter & (result_df["G_bat"].notna())
    elif has_g_pit:
        game_filter = game_filter & (result_df["G_pit"].notna())
    elif has_gs:
        game_filter = game_filter & (result_df["GS"].notna())
    elif has_g:
        game_filter = game_filter & (result_df["G"].notna())

    agg_dict: dict = {}
    if has_g_bat:
        agg_dict["G_bat"] = ("G_bat", "sum")
    if has_g_pit:
        agg_dict["G_pit"] = ("G_pit", "sum")
    elif has_g:
        agg_dict["G_pit"] = ("G", "sum")       # approximate pitcher G from total G
    if has_gs:
        agg_dict["GS"] = ("GS", "sum")
    if "position_group" in result_df.columns:
        agg_dict["position_group"] = ("position_group", "first")

    if agg_dict:
        career_games = (
            result_df[game_filter]
            .groupby("IDfg")
            .agg(**agg_dict)
            .reset_index()
        )
    else:
        career_games = pd.DataFrame(columns=["IDfg"])

    # ── Compute confidence for ALL players ───────────────────────────────
    result_df["projection_confidence"] = 1.0   # default: fully confident
    result_df["raw_trade_value"] = result_df["trade_value"]

    for pid in result_df.loc[result_df["trade_value"].notna(), "IDfg"].unique():
        pmask = (result_df["IDfg"] == pid) & (result_df["Year"] >= current_year)
        if not pmask.any():
            continue

        pos = result_df.loc[pmask, "position_group"].iloc[0]
        pos_type = "sp" if pos == "SP" else ("rp" if pos == "RP" else "batter")

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

        conf = Config.TradeConfidence.calculate_confidence(games, pos_type)
        result_df.loc[pmask, "projection_confidence"] = round(conf, 3)

    # ── Prospect-grade trade-value floor ─────────────────────────────────
    # MiLB regression (Step 2.25) handles stat-level blending, but a
    # prospect with a disastrous short MLB stint can still project ~0 WAR.
    # The prospect floor ensures their trade value reflects their pedigree:
    #   floor = prospect_dollar_value * prospect_weight
    # where prospect_weight fades linearly from 1.0 (0 games) to 0.0
    # (EXPERIENCE_THRESHOLD_GAMES reached).  The floor only RAISES value.

    # Build mlbam → IDfg crosswalk from register files
    people_files = glob.glob(str(_REGISTER_DATA_DIR / 'people-*.csv'))
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
        mlbam_to_idfg = dict(zip(xw['key_mlbam'], xw['key_fangraphs']))
    else:
        mlbam_to_idfg = {}
        logger.warning(f"No register files found in {_REGISTER_DATA_DIR} — "
                       "prospect floor disabled")

    # Map prospects to IDfg and apply floor
    latest["IDfg"] = latest["prospect_mlb_id"].astype(int).map(mlbam_to_idfg)
    matched = latest.dropna(subset=["IDfg"])
    matched = matched.copy()
    matched["IDfg"] = matched["IDfg"].astype(int)
    floor_count = 0

    for _, prospect in matched.iterrows():
        pid = prospect["IDfg"]
        fv = prospect.get("grade_overall")
        rank = prospect.get("top_100")

        pmask = (result_df["IDfg"] == pid) & (result_df["Year"] >= current_year)
        if not pmask.any() or pd.isna(result_df.loc[pmask, "trade_value"].iloc[0]):
            continue

        prospect_val = _prospect_dollar_value(fv, rank)
        if prospect_val is None or prospect_val <= 0:
            continue

        # Determine experience-based prospect weight
        pos = result_df.loc[pmask, "position_group"].iloc[0]
        pos_type = "sp" if pos == "SP" else ("rp" if pos == "RP" else "batter")
        if pid in career_games["IDfg"].values:
            cg = career_games[career_games["IDfg"] == pid].iloc[0]
            gs = cg.get("GS", 0) or 0
            g_pit = cg.get("G_pit", 0) or 0
            if pos_type == "sp":
                games = gs
            elif pos_type == "rp":
                games = g_pit - gs if gs < 50 else gs
            else:
                games = cg.get("G_bat", 0) or 0
        else:
            games = 0

        prospect_weight = Config.Prospects.calculate_prospect_weight(games, pos_type)
        if prospect_weight <= 0:
            continue  # Fully established — no floor needed

        prospect_floor = prospect_val * prospect_weight
        current_tv = result_df.loc[pmask, "trade_value"].iloc[0]

        if prospect_floor > current_tv:
            result_df.loc[pmask, "trade_value"] = prospect_floor
            result_df.loc[pmask, "prospect_floor_applied"] = True
            floor_count += 1

    logger.info(f"Prospect floor: matched {len(matched)} prospects, "
                f"raised trade value for {floor_count}")

    # Clean up temp columns
    result_df.drop(["_name_key", "_proj_value_sum", "_contract_sum"],
                   axis=1, errors="ignore", inplace=True)

    return result_df


# ---------------------------------------------------------------------------
# Trade ranking metrics
# ---------------------------------------------------------------------------

def add_trade_ranking_metrics(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """
    Add pre-computed ranking metrics for each player.

    Metrics (all merged back on IDfg):
        contract_war, avg_war, total_contract, avg_contract, total_surplus,
        years_control, control_through, contract_base_value,
        total_future_war, total_future_value,
        total_war, total_value, historical_war, historical_value
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    result = df.copy()
    rows = []

    for pid in result["IDfg"].unique():
        p = result[result["IDfg"] == pid].sort_values("Year")

        fa_year = p["probable_fa_year"].iloc[0]
        if pd.isna(fa_year):
            fa_year = p["FA_Year"].iloc[0]

        control = p[(p["Year"] >= current_year) & (p["Year"] < fa_year)]
        future = p[p["Year"] >= current_year]
        past = p[p["Year"] < current_year]
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
            "total_future_war": round(control.loc[control["WAR"] > 0, "WAR"].sum(), 1),
            "total_future_value": round(future["Base_Value"].sum(), 1),
            "total_war": round(past["WAR"].sum() + control["WAR"].sum(), 1),
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
