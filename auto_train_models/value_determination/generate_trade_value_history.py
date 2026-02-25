#!/usr/bin/env python
"""
Generate Trade Value History
============================

Builds a year-by-year trade value timeline for every player by combining:
  1. Prospect rankings (FV-based dollar values) for pre-MLB years
  2. MLB surplus values from each year's surplus file — uses the PROJECTED
     WAR at that point in time (never actual future stats), applies the
     convex trade-value model, and subtracts projected salary

The output is a compact CSV that the web-app backend serves to render a
rolling trade value chart on each player's detail page.

Output columns:
    mlb_id      - MLB player ID (primary key for web-app)
    IDfg        - FanGraphs player ID
    name        - Player name
    year        - Season
    value       - Dollar value for that year
    value_type  - 'prospect' | 'mlb_surplus'
    label       - Human-readable label (e.g. "FV 55, #46" or "6.3 WAR")

Output:  data/generated/value_by_year/trade_value_history.csv

Usage:
    python -m auto_train_models.value_determination.generate_trade_value_history
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
import logging

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from auto_train_models.value_determination.config import Config, CURRENT_YEAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("trade_value_history")


# ── Convex model for consistent historical valuation ──────────────────────
_ALPHA, _BETA = Config.ConvexModel.load_calibration()
_INFLATION_RATE = Config.Contracts.INFLATION_RATE
_BASE_YEAR = Config.Contracts.BASE_YEAR


def _convex_value(war: float, year: int) -> float:
    """Convert WAR to dollar value using the calibrated convex model."""
    if pd.isna(war) or war <= 0:
        return 0.0
    inflation = (1 + _INFLATION_RATE) ** (year - _BASE_YEAR)
    return _ALPHA * (war ** _BETA) * inflation


def _prospect_dollar_value(fv, rank) -> float:
    """Convert FV grade + rank into a dollar value (mirrors trade_value.py)."""
    if pd.isna(fv):
        return 0.0
    try:
        fv_num = float(str(fv).replace("+", "")) + (2.5 if "+" in str(fv) else 0)
        tiers = Config.Prospects.FV_BASE_VALUES
        valid = [k for k in tiers if k <= fv_num]
        base_tier = max(valid) if valid else min(tiers)
        base_value = tiers[base_tier]

        if pd.notna(rank) and float(rank) <= 100:
            return base_value * Config.Prospects.calculate_rank_adjustment(float(rank))
        return base_value
    except Exception:
        return 0.0


def build_prospect_timeline(player_values: pd.DataFrame) -> pd.DataFrame:
    """
    Build prospect value entries from the raw prospect data.
    
    Returns DataFrame with columns: mlb_id, IDfg, name, year, value, value_type, label
    """
    prospect_file = ROOT / "data" / "prospect_data" / "prospects_2014_2026_with_top100.csv"
    if not prospect_file.exists():
        logger.warning(f"Prospect file not found: {prospect_file}")
        return pd.DataFrame()

    prospects = pd.read_csv(prospect_file)
    logger.info(f"Loaded {len(prospects)} prospect entries")

    # Extract mlb_id from prospect URL
    prospects["mlb_id"] = prospects["prospect_url"].str.extract(r"(\d{5,7})$").astype(float)

    # Build lookup: mlb_id -> (IDfg, name) from player_values
    id_lookup = (
        player_values[player_values["mlb_id"].notna()]
        .drop_duplicates("mlb_id")
        .set_index("mlb_id")[["IDfg", "Player_Name"]]
    )

    rows = []
    for _, p in prospects[prospects["mlb_id"].notna()].iterrows():
        mid = int(p["mlb_id"])
        if mid not in id_lookup.index:
            continue

        idfg = id_lookup.loc[mid, "IDfg"]
        name = id_lookup.loc[mid, "Player_Name"]
        year = int(p["year"])
        fv = p.get("grade_overall")
        rank = p.get("top_100")

        value = _prospect_dollar_value(fv, rank)
        if value <= 0:
            continue

        # Build label
        fv_str = str(fv).replace(".0", "") if pd.notna(fv) else "?"
        label_parts = [f"FV {fv_str}"]
        if pd.notna(rank):
            label_parts.append(f"#{int(rank)}")
        label = ", ".join(label_parts)

        rows.append({
            "mlb_id": mid,
            "IDfg": int(idfg),
            "name": name,
            "year": year,
            "value": round(value),
            "value_type": "prospect",
            "label": label,
        })

    result = pd.DataFrame(rows)
    # Keep only the best ranking per player per year (in case of duplicates)
    if not result.empty:
        result = result.sort_values("value", ascending=False).drop_duplicates(["mlb_id", "year"])
    logger.info(f"Built {len(result)} prospect timeline entries for {result['mlb_id'].nunique()} players")
    return result


def _load_surplus_files() -> dict[int, pd.DataFrame]:
    """Load all per-year surplus files (surplus_YYYY.csv) into a dict keyed by year."""
    surplus_dir = ROOT / "data" / "generated" / "trade_analysis" / "surplus"
    surplus_by_year: dict[int, pd.DataFrame] = {}
    for path in sorted(surplus_dir.glob("surplus_*.csv")):
        try:
            yr = int(path.stem.split("_")[1])
            surplus_by_year[yr] = pd.read_csv(path, low_memory=False)
            logger.info(f"  Loaded {path.name}: {len(surplus_by_year[yr])} players")
        except Exception as e:
            logger.warning(f"  Failed to load {path.name}: {e}")
    return surplus_by_year


def build_mlb_timeline(player_values: pd.DataFrame) -> pd.DataFrame:
    """
    Build MLB trade-value entries using **projected** WAR from each year's
    surplus file — never actual future stats.

    For each surplus snapshot year Y and each player in that file:
      1. Collect WAR_Y, WAR_Y+1, … (the projections made at time Y)
      2. Apply the convex model to each year's projected WAR
      3. Sum the dollar values and subtract the projected salaries
      → that is the trade value *as it would have been estimated at time Y*

    IMPORTANT: We only use data from year Y's surplus file for the year Y
    entry.  We never cross-reference actual stats from future years so that
    the chart reflects what was projected at the time, not hindsight.

    For the current year (CURRENT_YEAR) we use the trade_value already
    computed by the value-determination pipeline and label it mlb_surplus
    (it IS the present, not the future).

    Returns DataFrame: mlb_id, IDfg, name, year, value, value_type, label
    """
    # ── Load all surplus files ────────────────────────────────────────────
    surplus_by_year = _load_surplus_files()
    if not surplus_by_year:
        logger.warning("No surplus files found — MLB timeline will be empty")
        return pd.DataFrame()

    # Build mlb_id ↔ IDfg lookup from player_values
    id_map = (
        player_values[player_values["mlb_id"].notna()]
        .drop_duplicates("mlb_id")
        .set_index("mlb_id")[["IDfg", "Player_Name"]]
    )
    idfg_to_mlb = {}
    for mlb_id, row in id_map.iterrows():
        idfg_to_mlb[int(row["IDfg"])] = int(mlb_id)

    rows: list[dict] = []

    # ── Historical years: use each surplus file's projections ─────────────
    for snap_year, sdf in surplus_by_year.items():
        war_cols = sorted([c for c in sdf.columns if c.startswith("WAR_")])
        sal_cols = sorted([c for c in sdf.columns if c.startswith("salary_") and c != "salary_source"])

        for _, player_row in sdf.iterrows():
            idfg = int(player_row["IDfg"])
            mlbam = int(player_row["mlbam_id"]) if pd.notna(player_row.get("mlbam_id")) else None

            # We need mlb_id for the web-app. Try mlbam_id first, then lookup via IDfg.
            mlb_id = mlbam
            if mlb_id is None or mlb_id not in id_map.index:
                mlb_id = idfg_to_mlb.get(idfg)
            if mlb_id is None:
                continue

            name = player_row.get("Name", "")
            yrs_ctrl = player_row.get("years_of_control", 0) or 0

            # Sum convex(projected WAR) - salary for each year of control
            total_value = 0.0
            total_salary = 0.0
            proj_war_this_year = 0.0  # WAR projected for the snapshot year itself

            for wc in war_cols:
                proj_year = int(wc.split("_")[1])
                war = player_row.get(wc)
                if pd.isna(war) or war <= 0:
                    continue

                total_value += _convex_value(war, proj_year)
                if proj_year == snap_year:
                    proj_war_this_year = war

            for sc in sal_cols:
                sal = player_row.get(sc)
                if pd.notna(sal):
                    total_salary += sal

            trade_val = total_value - total_salary

            # Skip players with no meaningful projection
            if proj_war_this_year <= 0 and total_value == 0:
                continue

            label = f"{proj_war_this_year:.1f} WAR" if proj_war_this_year > 0 else f"{yrs_ctrl}yr ctrl"

            rows.append({
                "mlb_id": mlb_id,
                "IDfg": idfg,
                "name": name,
                "year": snap_year,
                "value": round(trade_val),
                "value_type": "mlb_surplus",
                "label": label,
            })

    # ── Current year: use trade_value from the pipeline (already convex) ──
    current_pv = player_values[
        (player_values["Year"] == CURRENT_YEAR)
        & player_values["mlb_id"].notna()
        & player_values["trade_value"].notna()
    ].copy()

    for _, row in current_pv.iterrows():
        mlb_id = int(row["mlb_id"])
        idfg = int(row["IDfg"])
        name = row["Player_Name"]
        tv = row["trade_value"]
        war = row.get("WAR", 0)
        war_label = f"{war:.1f} WAR" if pd.notna(war) and war > 0 else "Trade Value"

        rows.append({
            "mlb_id": mlb_id,
            "IDfg": idfg,
            "name": name,
            "year": CURRENT_YEAR,
            "value": round(tv),
            "value_type": "mlb_surplus",
            "label": war_label,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        # Deduplicate: if a player appears multiple times in the same year
        # (e.g. from surplus file AND current-year pipeline), keep highest priority
        result = result.sort_values("value", ascending=False).drop_duplicates(
            ["mlb_id", "year"], keep="first"
        )
    logger.info(
        f"Built {len(result)} MLB timeline entries for "
        f"{result['mlb_id'].nunique() if not result.empty else 0} players"
    )
    return result


def main():
    """Generate the combined trade value history CSV."""
    logger.info("=" * 60)
    logger.info("Generating Trade Value History")
    logger.info("=" * 60)

    # Load player values
    pv_file = ROOT / "data" / "generated" / "value_by_year" / "player_values_complete.csv"
    if not pv_file.exists():
        logger.error(f"Player values file not found: {pv_file}")
        return

    pv = pd.read_csv(pv_file, low_memory=False)
    logger.info(f"Loaded {len(pv)} rows, {pv['IDfg'].nunique()} players")

    # Build timelines
    prospect_tl = build_prospect_timeline(pv)
    mlb_tl = build_mlb_timeline(pv)

    # Combine: prospect entries + MLB entries
    # Where a player has both prospect and MLB data for the same year,
    # prefer the MLB entry (they've already debuted)
    combined = pd.concat([prospect_tl, mlb_tl], ignore_index=True)

    # For duplicate (mlb_id, year), prefer mlb_surplus > prospect
    type_priority = {"mlb_surplus": 0, "prospect": 1}
    combined["_priority"] = combined["value_type"].map(type_priority).fillna(2)
    combined = (
        combined.sort_values("_priority")
        .drop_duplicates(["mlb_id", "year"], keep="first")
        .drop(columns=["_priority"])
        .sort_values(["mlb_id", "year"])
    )

    # ── Output ──
    out_dir = ROOT / "data" / "generated" / "value_by_year"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trade_value_history.csv"
    combined.to_csv(out_path, index=False)

    logger.info(f"Wrote {len(combined)} entries for {combined['mlb_id'].nunique()} players")
    logger.info(f"  Prospect entries: {(combined['value_type'] == 'prospect').sum()}")
    logger.info(f"  MLB surplus entries: {(combined['value_type'] == 'mlb_surplus').sum()}")
    logger.info(f"Output: {out_path}")

    # Print a few example players
    for name in ["Tarik Skubal", "Corbin Carroll", "Bobby Witt Jr."]:
        p = combined[combined["name"] == name]
        if not p.empty:
            print(f"\n{name}:")
            for _, r in p.iterrows():
                print(f"  {int(r['year'])}  {r['value_type']:14s}  ${r['value']:>14,.0f}  {r['label']}")


if __name__ == "__main__":
    main()
