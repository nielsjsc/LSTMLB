"""
Historical Trade Analyser
=========================

Links actual MLB trades (2014-2024) with surplus values from
``surplus_calculator.py`` **and** prospect valuations from FanGraphs
scouting data, then fits the non-linear WAR transformation parameter β
that best explains observed trade behaviour.

Core idea
---------
If teams trade rationally, the *total value* sent by each side should be
roughly equal.  We compute value as:

    MLB players   →  WAR^β dollar surplus  (from projection models)
    Prospects     →  FV/rank-based surplus  (from scouting grades)

and find the β that minimises imbalance across all observed trades.

Matching pipeline
-----------------
    1. mlbam_id + snapshot_year   → surplus  (primary)
    2. Normalised name + year     → surplus  (fallback for MLB players)
    3. mlbam_id → prospect data   → FV-based value  (for minor leaguers)

Usage:
    python -m auto_train_models.trade_analysis.analyze_trades
    python -m auto_train_models.trade_analysis.analyze_trades --beta 1.4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_AUTO_TRAIN = Path(__file__).resolve().parents[1]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from trade_analysis.config import (
    Config, logger,
    DATA_DIR, SURPLUS_DIR, RESULTS_DIR,
    TRADE_PLAYERS_FILE, TRADES_FILE, PROSPECT_FILE,
)
from trade_analysis.surplus_calculator import _war_dollar_value, _inflation


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Load & prepare data
# ═══════════════════════════════════════════════════════════════════════════

def load_trades() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trades, trade_players) DataFrames."""
    trades  = pd.read_csv(TRADES_FILE)
    players = pd.read_csv(TRADE_PLAYERS_FILE)

    players["date"] = pd.to_datetime(players["date"])
    trades["date"]  = pd.to_datetime(trades["date"])

    players["trade_year"] = players["date"].dt.year
    trades["trade_year"]  = trades["date"].dt.year if "year" not in trades.columns else trades["year"]

    return trades, players


def load_all_surplus() -> pd.DataFrame:
    """Load and concatenate all per-year surplus files."""
    files = sorted(SURPLUS_DIR.glob("surplus_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No surplus files found in {SURPLUS_DIR}.  "
            "Run  surplus_calculator.py  first."
        )
    frames = [pd.read_csv(f) for f in files]
    surplus = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(surplus)} surplus rows from {len(files)} snapshot years")
    return surplus


def load_prospect_data() -> pd.DataFrame:
    """
    Load prospect data and extract mlbam_id from the prospect URL.

    Returns a DataFrame with columns including:
        mlbam_id, year, name, grade_overall, top_100, position, level, etc.
    """
    if not PROSPECT_FILE.exists():
        logger.warning(f"Prospect file not found: {PROSPECT_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(PROSPECT_FILE)
    # Extract 6-digit mlbam_id from end of prospect URL
    df["mlbam_id"] = (
        df["prospect_url"]
        .str.extract(r"(\d{6})$", expand=False)
        .astype(float)
    )
    df = df.dropna(subset=["mlbam_id"])
    df["mlbam_id"] = df["mlbam_id"].astype(int)
    df["year"] = df["year"].astype(int)

    logger.info(f"Loaded {len(df)} prospect rows ({df['mlbam_id'].nunique()} "
                f"unique players, {df['year'].nunique()} years)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Prospect valuation
# ═══════════════════════════════════════════════════════════════════════════

def _fv_to_surplus(fv: float) -> float:
    """
    Interpolate FV grade → surplus value from the Config lookup table.

    Values between defined tiers are linearly interpolated.
    """
    table = Config.FV_SURPLUS_VALUE
    tiers = sorted(table.keys())

    if pd.isna(fv):
        return 0.0
    fv = float(fv)

    if fv <= tiers[0]:
        return table[tiers[0]]
    if fv >= tiers[-1]:
        return table[tiers[-1]]

    # Linear interpolation between adjacent tiers
    for i in range(len(tiers) - 1):
        lo, hi = tiers[i], tiers[i + 1]
        if lo <= fv <= hi:
            frac = (fv - lo) / (hi - lo)
            return table[lo] + frac * (table[hi] - table[lo])
    return 0.0


def _top100_bonus(rank: float) -> float:
    """
    Compute a multiplicative bonus factor from top-100 rank.

    #1 overall → 1 + TOP_100_MAX_BONUS
    #100        → 1 + TOP_100_MIN_BONUS
    Unranked    → 1.0  (no bonus)
    """
    if pd.isna(rank):
        return 1.0
    rank = float(rank)
    if rank < 1 or rank > 100:
        return 1.0
    # Linear interpolation: rank 1 → max bonus, rank 100 → min bonus
    frac = (100 - rank) / 99.0
    bonus = Config.TOP_100_MIN_BONUS + frac * (Config.TOP_100_MAX_BONUS - Config.TOP_100_MIN_BONUS)
    return 1.0 + bonus


def prospect_surplus_value(
    fv: float,
    top_100_rank: float = np.nan,
    year: int = 2024,
) -> float:
    """
    Compute surplus-equivalent dollar value for a prospect.

    Combines FV-based value with top-100 rank bonus, adjusted for inflation.
    """
    base = _fv_to_surplus(fv)
    bonus = _top100_bonus(top_100_rank)
    return base * bonus * _inflation(year)


def _build_prospect_lookup(prospect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup table: for each (mlbam_id, year), compute prospect surplus.

    For trade matching, we want the most recent prospect ranking AT or
    BEFORE the trade year.  This function returns one row per
    (mlbam_id, year) with the computed surplus.
    """
    if prospect_df.empty:
        return pd.DataFrame(columns=["mlbam_id", "snapshot_year", "prospect_surplus",
                                      "prospect_fv", "prospect_rank", "prospect_level"])

    df = prospect_df.copy()
    df["prospect_surplus"] = df.apply(
        lambda r: prospect_surplus_value(r["grade_overall"], r.get("top_100", np.nan), int(r["year"])),
        axis=1,
    )
    df = df.rename(columns={
        "year": "snapshot_year",
        "grade_overall": "prospect_fv",
        "top_100": "prospect_rank",
        "level": "prospect_level",
    })

    # Keep the best row per (mlbam_id, snapshot_year) — highest FV
    df = (
        df.sort_values("prospect_surplus", ascending=False)
        .drop_duplicates(subset=["mlbam_id", "snapshot_year"], keep="first")
    )

    return df[["mlbam_id", "snapshot_year", "prospect_surplus",
               "prospect_fv", "prospect_rank", "prospect_level"]]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Match trade players → surplus + prospects
# ═══════════════════════════════════════════════════════════════════════════

def match_trade_players_to_surplus(
    trade_players: pd.DataFrame,
    surplus_df: pd.DataFrame,
    prospect_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach value information to each trade-player row.

    Three-tier matching:
        1. mlbam_id + snapshot_year → MLB surplus  (primary)
        2. Normalised name + year  → MLB surplus   (fallback)
        3. mlbam_id → prospect data → FV surplus   (minor leaguers)

    For players matched to BOTH MLB surplus and prospect data, the MLB
    surplus is used (it's based on actual projections and is more precise).

    Snapshot year logic
    -------------------
    The surplus snapshot for year Y uses stats through end of season Y-1
    (i.e. preseason Y projections).  For a trade occurring in:

      • Jan–Sep (in-season or pre-season):
            snapshot_year = trade_year
            (Use the preseason projections that were current at trade time.)

      • Oct–Dec (post-season offseason):
            snapshot_year = trade_year + 1
            (The regular season has concluded, so the NEXT snapshot already
            incorporates the current season's stats — e.g. Stanton traded in
            Dec 2017 should use the 2018 snapshot, not the stale 2017 one.)

    Adds columns:
        IDfg, surplus, total_future_WAR_value, total_future_salary, status,
        years_of_control, service_time, matched_by,
        prospect_fv, prospect_rank, prospect_surplus, snapshot_year
    """
    tp = trade_players.copy()

    # Determine the correct snapshot year based on trade month
    trade_month = tp["date"].dt.month
    tp["snapshot_year"] = tp["trade_year"].where(trade_month < 10, tp["trade_year"] + 1)

    surplus = surplus_df.copy()
    surplus["mlbam_id"] = surplus["mlbam_id"].astype("Int64")

    _SURPLUS_COLS = [
        "mlbam_id", "snapshot_year", "IDfg", "surplus",
        "total_future_WAR_value", "total_future_salary", "status",
        "years_of_control", "service_time",
    ]

    # ── Tier 1: match by mlbam_id + snapshot_year ─────────────────────────
    tp = tp.merge(
        surplus[_SURPLUS_COLS],
        left_on=["mlbam_id", "snapshot_year"],
        right_on=["mlbam_id", "snapshot_year"],
        how="left",
        suffixes=("", "_surplus"),
    )
    tp["matched_by"] = np.where(tp["surplus"].notna(), "mlbam_id", pd.NA)

    # ── Tier 2: fallback name match ───────────────────────────────────────
    unmatched = tp["surplus"].isna()
    if unmatched.any():
        # Normalise names for fuzzy matching
        def _norm(s: pd.Series) -> pd.Series:
            return (
                s.str.lower()
                .str.strip()
                .str.replace(r"[.\-']", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
            )

        tp["_name_key"] = _norm(tp["name"])
        surplus["_name_key"] = _norm(surplus["Name"])

        _FB_COLS = [
            "_name_key", "snapshot_year", "IDfg", "surplus",
            "total_future_WAR_value", "total_future_salary", "status",
            "years_of_control", "service_time",
        ]
        # Deduplicate surplus name keys to avoid merge fans
        surplus_dedup = surplus[_FB_COLS].drop_duplicates(
            subset=["_name_key", "snapshot_year"], keep="first"
        )

        fb = tp.loc[unmatched, ["_name_key", "snapshot_year"]].merge(
            surplus_dedup,
            on=["_name_key", "snapshot_year"],
            how="left",
        )
        for col in ["IDfg", "surplus", "total_future_WAR_value",
                     "total_future_salary", "status",
                     "years_of_control", "service_time"]:
            tp.loc[unmatched, col] = fb[col].values
        tp.loc[unmatched & tp["surplus"].notna(), "matched_by"] = "name"

        tp.drop(columns=["_name_key"], errors="ignore", inplace=True)
        surplus.drop(columns=["_name_key"], errors="ignore", inplace=True)

    # ── Tier 3: prospect data ─────────────────────────────────────────────
    tp["prospect_fv"] = np.nan
    tp["prospect_rank"] = np.nan
    tp["prospect_surplus"] = np.nan

    if prospect_df is not None and not prospect_df.empty:
        prosp_lookup = _build_prospect_lookup(prospect_df)

        if not prosp_lookup.empty:
            still_unmatched = tp["surplus"].isna()

            # For prospect matching, find the closest prospect year <= trade year
            # We'll merge all prospect years and filter post-merge
            tp_unmatch = tp.loc[still_unmatched, ["mlbam_id", "snapshot_year"]].copy()
            tp_unmatch["_idx"] = tp_unmatch.index

            # Cross-join with prospect lookup and keep only year <= trade year
            merged = tp_unmatch.merge(prosp_lookup, on="mlbam_id", how="inner",
                                       suffixes=("", "_prosp"))
            merged = merged[merged["snapshot_year_prosp"] <= merged["snapshot_year"]]

            if not merged.empty:
                # Keep the most recent prospect year for each trade-player row
                merged = (
                    merged
                    .sort_values("snapshot_year_prosp", ascending=False)
                    .drop_duplicates(subset=["_idx"], keep="first")
                    .set_index("_idx")
                )

                # Fill in prospect columns
                for col in ["prospect_fv", "prospect_rank", "prospect_surplus", "prospect_level"]:
                    if col in merged.columns:
                        tp.loc[merged.index, col] = merged[col].values

                # Set surplus = prospect_surplus for these players
                prosp_matched = merged.index
                tp.loc[prosp_matched, "surplus"] = tp.loc[prosp_matched, "prospect_surplus"]
                tp.loc[prosp_matched, "matched_by"] = "prospect"
                tp.loc[prosp_matched, "status"] = "Prospect"

            # Also attach prospect info to players already matched via surplus
            # (informational only — doesn't change their surplus value)
            already_matched = tp["surplus"].notna() & (tp["matched_by"] != "prospect")
            if already_matched.any():
                tp_matched = tp.loc[already_matched, ["mlbam_id", "snapshot_year"]].copy()
                tp_matched["_idx"] = tp_matched.index
                m2 = tp_matched.merge(prosp_lookup, on="mlbam_id", how="inner",
                                       suffixes=("", "_prosp"))
                m2 = m2[m2["snapshot_year_prosp"] <= m2["snapshot_year"]]
                if not m2.empty:
                    m2 = (
                        m2.sort_values("snapshot_year_prosp", ascending=False)
                        .drop_duplicates(subset=["_idx"], keep="first")
                        .set_index("_idx")
                    )
                    for col in ["prospect_fv", "prospect_rank"]:
                        if col in m2.columns:
                            tp.loc[m2.index, col] = m2[col].values

    # ── Summary ───────────────────────────────────────────────────────────
    n_matched = tp["surplus"].notna().sum()
    match_counts = tp["matched_by"].value_counts().to_dict()
    logger.info(
        f"Matched {n_matched}/{len(tp)} trade-player rows "
        f"({match_counts})"
    )
    return tp


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Convex trade-value model
# ═══════════════════════════════════════════════════════════════════════════

def _convex_war_value(
    war: float, year: int, alpha: float, beta: float,
) -> float:
    """
    Convex power-law annual WAR valuation.

        value_year = alpha * WAR^beta * inflation(year)

    With beta > 1 the curve is convex: elite players are worth
    disproportionately more per WAR than average players, reflecting
    scarcity, certainty, and roster-slot opportunity cost.
    """
    if pd.isna(war) or war <= 0:
        return 0.0
    return alpha * (war ** beta) * _inflation(year)


def _recompute_trade_value(
    trade_player_row: pd.Series,
    surplus_df: pd.DataFrame,
    alpha: float,
    beta: float,
) -> float:
    """
    Compute convex trade value for one player:

        trade_value = SUM_y[ alpha * WAR_y^beta * infl(y) ] - SUM_y[ salary_y ]

    For prospect-matched players (no per-year WAR columns), returns the
    original FV-based prospect surplus unchanged.
    """
    matched_by = trade_player_row.get("matched_by", "")
    if matched_by == "prospect":
        # Prospect value is already calibrated to trade value
        return trade_player_row.get("surplus", 0.0) or 0.0

    idfg = trade_player_row.get("IDfg")
    snap  = trade_player_row.get("snapshot_year")
    if pd.isna(idfg) or pd.isna(snap):
        return 0.0

    row_match = surplus_df.loc[
        (surplus_df["IDfg"] == idfg) & (surplus_df["snapshot_year"] == snap)
    ]
    if row_match.empty:
        return 0.0

    row = row_match.iloc[0]
    yoc = int(row.get("years_of_control", 0))
    if yoc <= 0:
        return 0.0

    total_val = 0.0
    total_sal = 0.0
    for yr in range(int(snap), int(snap) + yoc):
        war = row.get(f"WAR_{yr}", 0.0)
        if pd.isna(war):
            war = 0.0
        total_val += _convex_war_value(war, yr, alpha, beta)
        sal = row.get(f"salary_{yr}", 0.0)
        total_sal += 0.0 if pd.isna(sal) else sal

    return total_val - total_sal


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Build per-trade summary & optimise (alpha, beta)
# ═══════════════════════════════════════════════════════════════════════════

def build_trade_sides(
    matched_tp: pd.DataFrame,
    value_col: str = "surplus",
    min_players_per_side: int = 1,
) -> pd.DataFrame:
    """
    Aggregate matched trade-player rows into per-side summaries.

    ``value_col`` selects which column to sum ("surplus" for linear,
    "trade_value" for convex model).
    Returns one row per (trade_id, from_team_id).
    """
    df = matched_tp.dropna(subset=[value_col]).copy()

    sides = (
        df
        .groupby(["trade_id", "from_team_id", "from_team_name", "trade_year"])
        .agg(
            n_players=(value_col, "size"),
            total_surplus=(value_col, "sum"),
            total_war_value=("total_future_WAR_value", lambda x: x.sum(min_count=1)),
            total_salary=("total_future_salary", lambda x: x.sum(min_count=1)),
            players=("name", lambda x: ", ".join(x)),
        )
        .reset_index()
    )

    sides = sides[sides["n_players"] >= min_players_per_side]
    return sides


def pair_trade_sides(sides: pd.DataFrame) -> pd.DataFrame:
    """
    Pair the two sides of each 2-team trade into a single row.

    Returns columns: trade_id, trade_year, side_A_*, side_B_*, imbalance.
    """
    paired_rows = []
    for tid, grp in sides.groupby("trade_id"):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        paired_rows.append({
            "trade_id":            tid,
            "trade_year":          a["trade_year"],
            "side_A_team":         a["from_team_name"],
            "side_A_players":      a["players"],
            "side_A_n":            a["n_players"],
            "side_A_surplus":      a["total_surplus"],
            "side_A_war_value":    a.get("total_war_value", 0),
            "side_A_salary":       a.get("total_salary", 0),
            "side_B_team":         b["from_team_name"],
            "side_B_players":      b["players"],
            "side_B_n":            b["n_players"],
            "side_B_surplus":      b["total_surplus"],
            "side_B_war_value":    b.get("total_war_value", 0),
            "side_B_salary":       b.get("total_salary", 0),
            "imbalance":           abs(a["total_surplus"] - b["total_surplus"]),
        })

    result = pd.DataFrame(paired_rows)
    if not result.empty:
        result = result.sort_values("imbalance", ascending=False).reset_index(drop=True)
    logger.info(f"Paired {len(result)} 2-team trades with both sides matched")
    return result


def _compute_trade_imbalances(
    usable_tp: pd.DataFrame,
    surplus_df: pd.DataFrame,
    alpha: float,
    beta: float,
) -> list[float]:
    """
    Recompute convex trade values and return per-trade absolute imbalances.

    Helper shared by optimiser and final reporting.
    """
    new_vals = usable_tp.apply(
        lambda r: _recompute_trade_value(r, surplus_df, alpha, beta), axis=1
    )
    tmp = usable_tp.copy()
    tmp["tv"] = new_vals

    sides = (
        tmp
        .groupby(["trade_id", "from_team_id"])
        .agg(total=("tv", "sum"))
        .reset_index()
    )

    imbalances: list[float] = []
    for tid, grp in sides.groupby("trade_id"):
        if len(grp) != 2:
            continue
        imbalances.append(abs(grp["total"].iloc[0] - grp["total"].iloc[1]))
    return imbalances


def optimise_parameters(
    matched_tp: pd.DataFrame,
    surplus_df: pd.DataFrame,
    alpha_range: tuple[float, float] = (2_000_000, 15_000_000),
    beta_range: tuple[float, float] = (1.0, 2.5),
) -> dict:
    """
    Jointly fit (alpha, beta) to minimise **median** trade imbalance.

    Using the median (instead of mean/total) makes the fit robust to
    salary-dump trades that are intentionally lopsided.  Those trades
    will still show as big imbalances in the output, but they won't
    drag the optimal curve in the wrong direction.

    Returns dict with keys: alpha, beta, median_imbalance, mean_imbalance,
    n_trades, n_evals.
    """
    # Determine which trade_ids have two matchable sides
    sides_linear = build_trade_sides(matched_tp)
    paired_linear = pair_trade_sides(sides_linear)
    usable_trade_ids = set(paired_linear["trade_id"].unique())

    if not usable_trade_ids:
        logger.warning("No usable trades for parameter optimisation")
        return {"alpha": 8_000_000, "beta": 1.4,
                "median_imbalance": np.inf, "n_trades": 0}

    usable_tp = matched_tp[
        matched_tp["trade_id"].isin(usable_trade_ids) & matched_tp["surplus"].notna()
    ].copy()

    n_trades = len(usable_trade_ids)
    logger.info(f"Optimising (alpha, beta) over {n_trades} trades, "
                f"{len(usable_tp)} player-rows...")

    n_evals = [0]

    def _objective(params: np.ndarray) -> float:
        alpha, beta = float(params[0]), float(params[1])
        # Soft boundary penalty
        if (alpha < alpha_range[0] or alpha > alpha_range[1]
                or beta < beta_range[0] or beta > beta_range[1]):
            return 1e18
        n_evals[0] += 1
        imbalances = _compute_trade_imbalances(
            usable_tp, surplus_df, alpha, beta,
        )
        if not imbalances:
            return 1e18
        return float(np.median(imbalances))

    # Starting point: ~$8M/WAR base rate, beta = 1.4
    x0 = np.array([8_000_000.0, 1.4])

    result = minimize(
        _objective, x0,
        method="Nelder-Mead",
        options={
            "xatol": 50_000,   # alpha tolerance ($50K)
            "fatol": 50_000,   # objective tolerance ($50K)
            "maxiter": 300,
            "adaptive": True,
        },
    )

    best_alpha = float(np.clip(result.x[0], *alpha_range))
    best_beta  = float(np.clip(result.x[1], *beta_range))

    # Final stats at optimum
    final_imb = _compute_trade_imbalances(
        usable_tp, surplus_df, best_alpha, best_beta,
    )
    med_imb  = float(np.median(final_imb)) if final_imb else 0.0
    mean_imb = float(np.mean(final_imb))   if final_imb else 0.0

    out = {
        "alpha": round(best_alpha),
        "beta":  round(best_beta, 3),
        "median_imbalance": round(med_imb),
        "mean_imbalance":   round(mean_imb),
        "n_trades":  n_trades,
        "n_evals":   n_evals[0],
    }
    logger.info(
        f"Optimal alpha=${best_alpha/1e6:.2f}M  beta={best_beta:.3f}  "
        f"(median imbalance ${med_imb/1e6:.1f}M, "
        f"mean ${mean_imb/1e6:.1f}M, {n_evals[0]} evals)"
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Trade highlighting — human-readable output
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_dollars(v: float) -> str:
    """Format a dollar value compactly: $12.3M, $450K, etc."""
    if pd.isna(v) or v == 0:
        return "$0"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}K"
    return f"${v:,.0f}"


def _player_detail(row: pd.Series, value_col: str = "surplus") -> str:
    """One-line summary of a matched player."""
    name = row["name"]
    matched = row.get("matched_by", "")
    val = row.get(value_col, 0) or row.get("surplus", 0) or 0

    parts = [name]
    if matched == "prospect":
        fv = row.get("prospect_fv", "")
        rank = row.get("prospect_rank", "")
        fv_s = f"FV {int(fv)}" if pd.notna(fv) else ""
        rank_s = f"Top-100 #{int(rank)}" if pd.notna(rank) else ""
        tag = ", ".join(filter(None, [fv_s, rank_s]))
        if tag:
            parts.append(f"[{tag}]")
    parts.append(f"({_fmt_dollars(val)})")
    return " ".join(parts)


def print_highlighted_trades(
    matched_tp: pd.DataFrame,
    paired: pd.DataFrame,
    trades_df: pd.DataFrame,
    n_examples: int = 12,
    value_col: str = "surplus",
) -> str:
    """
    Print notable trade examples to stdout and return the text.

    Selects trades across categories:
      - Most balanced (lowest imbalance)
      - Biggest steals (highest imbalance)
      - Trades involving top prospects
      - Most total value exchanged
    """
    lines: list[str] = []

    def _add(text: str = ""):
        lines.append(text)
        print(text)

    def _show_trade(row: pd.Series, label: str = ""):
        tid = row["trade_id"]
        year = int(row["trade_year"])

        # Get trade description
        desc_row = trades_df[trades_df["trade_id"] == tid]
        desc = desc_row.iloc[0]["description"] if not desc_row.empty else ""
        # Truncate long descriptions
        if len(desc) > 200:
            desc = desc[:197] + "..."

        # Get per-player details
        side_a_players = matched_tp[
            (matched_tp["trade_id"] == tid) &
            (matched_tp["from_team_name"] == row["side_A_team"]) &
            matched_tp[value_col].notna()
        ]
        side_b_players = matched_tp[
            (matched_tp["trade_id"] == tid) &
            (matched_tp["from_team_name"] == row["side_B_team"]) &
            matched_tp[value_col].notna()
        ]

        _add(f"\n  {'-' * 70}")
        if label:
            _add(f"  [{label}]")
        _add(f"  {year} | Trade #{tid}")
        if desc:
            _add(f"  {desc}")

        # Side A
        _add(f"  >> {row['side_A_team']} sent ({_fmt_dollars(row['side_A_surplus'])} total):")
        for _, p in side_a_players.iterrows():
            _add(f"  |   {_player_detail(p, value_col)}")

        # Side B
        _add(f"  << {row['side_B_team']} sent ({_fmt_dollars(row['side_B_surplus'])} total):")
        for _, p in side_b_players.iterrows():
            _add(f"      {_player_detail(p, value_col)}")

        imb_pct = (row["imbalance"] / max(row["side_A_surplus"], row["side_B_surplus"], 1)) * 100
        _add(f"  Imbalance: {_fmt_dollars(row['imbalance'])} ({imb_pct:.0f}%)")

    if paired.empty:
        _add("No paired trades to highlight.")
        return "\n".join(lines)

    _add("\n" + "=" * 74)
    _add("  HIGHLIGHTED TRADES")
    _add("=" * 74)

    # --- Category 1: Most balanced trades ---
    n_each = max(n_examples // 4, 2)
    balanced = paired.nsmallest(n_each, "imbalance")
    _add(f"\n  > MOST BALANCED TRADES (top {n_each})")
    for _, row in balanced.iterrows():
        _show_trade(row, "Balanced")

    # --- Category 2: Biggest steals ---
    steals = paired.nlargest(n_each, "imbalance")
    _add(f"\n  > BIGGEST IMBALANCES (top {n_each})")
    for _, row in steals.iterrows():
        _show_trade(row, "Big imbalance")

    # --- Category 3: Trades involving top prospects ---
    prospect_tids = set(
        matched_tp.loc[
            matched_tp["prospect_rank"].notna() & (matched_tp["prospect_rank"] <= 25),
            "trade_id"
        ].unique()
    )
    prospect_trades = paired[paired["trade_id"].isin(prospect_tids)]
    if not prospect_trades.empty:
        top_prosp = prospect_trades.nlargest(min(n_each, len(prospect_trades)),
                                             "side_A_surplus")
        _add(f"\n  > TRADES INVOLVING TOP-25 PROSPECTS ({len(top_prosp)} shown)")
        for _, row in top_prosp.iterrows():
            _show_trade(row, "Top prospect")

    # --- Category 4: Highest total value exchanged ---
    paired["total_value"] = paired["side_A_surplus"].abs() + paired["side_B_surplus"].abs()
    biggest = paired.nlargest(n_each, "total_value")
    _add(f"\n  > HIGHEST TOTAL VALUE EXCHANGED (top {n_each})")
    for _, row in biggest.iterrows():
        _show_trade(row, "Big trade")

    _add("\n" + "=" * 74)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Main analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(
    alpha: float | None = None,
    beta: float | None = None,
    skip_optimise: bool = False,
) -> dict:
    """
    Full trade-analysis pipeline.

    1. Load trades, surplus, and prospect data.
    2. Match trade players to values (3-tier: mlbam -> name -> prospect).
    3. Build linear trade sides + pairs (baseline).
    4. Optimise (alpha, beta) convex model — or use fixed values.
    5. Recompute trade values with optimal params.
    6. Print highlighted trade examples.
    7. Save all results.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    trades, trade_players = load_trades()
    surplus_df = load_all_surplus()
    prospect_df = load_prospect_data()

    # ── Match ─────────────────────────────────────────────────────────────
    matched = match_trade_players_to_surplus(trade_players, surplus_df, prospect_df)

    # ── Linear baseline ───────────────────────────────────────────────────
    sides_lin   = build_trade_sides(matched, value_col="surplus")
    paired_lin  = pair_trade_sides(sides_lin)

    paired_lin_path = RESULTS_DIR / "trade_pairs_linear.csv"
    paired_lin.to_csv(paired_lin_path, index=False)
    logger.info(f"Saved {len(paired_lin)} paired trades (linear) to "
                f"{paired_lin_path.name}")

    # ── Parameter optimisation ────────────────────────────────────────────
    params: dict = {}
    if alpha is not None and beta is not None:
        params = {"alpha": alpha, "beta": beta, "fixed": True}
        logger.info(f"Using fixed alpha=${alpha/1e6:.2f}M  beta={beta}")
    elif not skip_optimise:
        params = optimise_parameters(matched, surplus_df)
    else:
        params = {"alpha": 8_000_000, "beta": 1.4, "skipped": True}

    best_alpha = float(params.get("alpha", 8_000_000))
    best_beta  = float(params.get("beta", 1.4))

    params_path = RESULTS_DIR / "convex_calibration.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved calibration to {params_path.name}")

    # ── Recompute per-player trade values with optimal (alpha, beta) ─────
    matched["trade_value"] = matched.apply(
        lambda r: _recompute_trade_value(r, surplus_df, best_alpha, best_beta)
        if pd.notna(r.get("surplus")) else np.nan,
        axis=1,
    )

    matched_path = RESULTS_DIR / "matched_trade_players.csv"
    matched.to_csv(matched_path, index=False)
    logger.info(f"Saved matched trade-player rows to {matched_path.name}")

    # ── Convex trade pairs ────────────────────────────────────────────────
    sides_cv  = build_trade_sides(matched, value_col="trade_value")
    paired_cv = pair_trade_sides(sides_cv)

    paired_cv_path = RESULTS_DIR / "trade_pairs_convex.csv"
    paired_cv.to_csv(paired_cv_path, index=False)
    logger.info(f"Saved {len(paired_cv)} paired trades (convex) to "
                f"{paired_cv_path.name}")

    # ── Summary stats ─────────────────────────────────────────────────────
    n_total_trades  = trades["trade_id"].nunique()
    n_matched_pairs = len(paired_cv)
    match_rate = matched["surplus"].notna().mean()
    match_counts = matched["matched_by"].value_counts().to_dict()

    summary = {
        "total_trades": n_total_trades,
        "paired_trades_analyzed": n_matched_pairs,
        "player_match_rate": round(float(match_rate), 4),
        "match_breakdown": {k: int(v) for k, v in match_counts.items()},
        "alpha": round(best_alpha),
        "beta": round(best_beta, 3),
        "median_imbalance_linear": round(float(paired_lin["imbalance"].median())) if len(paired_lin) else 0,
        "mean_imbalance_linear":   round(float(paired_lin["imbalance"].mean()))   if len(paired_lin) else 0,
        "median_imbalance_convex": round(float(paired_cv["imbalance"].median()))  if len(paired_cv) else 0,
        "mean_imbalance_convex":   round(float(paired_cv["imbalance"].mean()))    if len(paired_cv) else 0,
    }

    summary_path = RESULTS_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary + highlighted trades ────────────────────────────────
    print("\n" + "=" * 74)
    print("  TRADE ANALYSIS SUMMARY")
    print("=" * 74)
    print(f"  Total trades in database:       {n_total_trades:,}")
    print(f"  Trade-player rows:              {len(trade_players):,}")
    print(f"  Players matched to value:       {matched['surplus'].notna().sum():,} "
          f"({match_rate:.1%})")
    for src, cnt in sorted(match_counts.items(), key=lambda x: -x[1]):
        print(f"    via {src:12s}:  {cnt:,}")
    print(f"  Paired trades analysed:         {n_matched_pairs:,}")
    print()
    print(f"  --- Convex Model: alpha=${best_alpha/1e6:.2f}M  beta={best_beta:.3f} ---")
    print(f"  Median imbalance (convex):      {_fmt_dollars(summary['median_imbalance_convex'])}")
    print(f"  Mean imbalance (convex):        {_fmt_dollars(summary['mean_imbalance_convex'])}")
    print(f"  Median imbalance (linear):      {_fmt_dollars(summary['median_imbalance_linear'])}")
    print(f"  Mean imbalance (linear):        {_fmt_dollars(summary['mean_imbalance_linear'])}")

    # Show the value curve at a few WAR levels
    print()
    print("  --- Value Curve (single year, 2025) ---")
    print(f"  {'WAR':>5}  {'Trade value':>14}  {'Linear val':>14}")
    for w in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        cv = _convex_war_value(w, 2025, best_alpha, best_beta)
        lv = _war_dollar_value(w, 2025)
        print(f"  {w:>5.1f}  {_fmt_dollars(cv):>14}  {_fmt_dollars(lv):>14}")

    # Highlighted trades (use convex values)
    highlight_text = print_highlighted_trades(matched, paired_cv, trades,
                                              value_col="trade_value")

    highlight_path = RESULTS_DIR / "highlighted_trades.txt"
    with open(highlight_path, "w", encoding="utf-8") as f:
        f.write(highlight_text)
    logger.info(f"Saved highlighted trades to {highlight_path.name}")

    logger.info(f"Analysis summary: {summary}")

    return {
        "params": params,
        "paired_trades_convex": paired_cv,
        "paired_trades_linear": paired_lin,
        "matched_players": matched,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Analyse historical trades and fit convex (alpha, beta) model."
    )
    p.add_argument("--alpha", type=float, default=None,
                   help="Fix alpha ($/WAR base rate) instead of optimising.")
    p.add_argument("--beta", type=float, default=None,
                   help="Fix beta (convexity exponent) instead of optimising.")
    p.add_argument("--skip-optimise", action="store_true",
                   help="Skip optimisation (use alpha=8M, beta=1.4).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_analysis(alpha=args.alpha, beta=args.beta,
                 skip_optimise=args.skip_optimise)
