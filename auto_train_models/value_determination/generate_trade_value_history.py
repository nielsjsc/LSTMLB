#!/usr/bin/env python
"""
Generate Trade Value History
============================

Builds a year-by-year trade value timeline for every player by combining:
  1. Prospect rankings (FV-based dollar values) for pre-MLB years
  2. Historical MLB surplus values (WAR production - contract cost) per season
  3. Current projected trade value from the value determination pipeline

The output is a compact CSV that the web-app backend serves to render a
rolling trade value chart on each player's detail page.

Output columns:
    mlb_id      - MLB player ID (primary key for web-app)
    IDfg        - FanGraphs player ID
    name        - Player name
    year        - Season
    value       - Dollar value for that year
    value_type  - 'prospect' | 'mlb_surplus' | 'projected'
    label       - Human-readable label (e.g. "FV 55, #46" or "3.2 WAR")

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


def build_mlb_timeline(player_values: pd.DataFrame) -> pd.DataFrame:
    """
    Build MLB surplus value entries from historical + projected data.
    
    For historical years: compute cumulative remaining trade value at each year
    (sum of surplus from that year forward until FA).
    
    For projected years: use the trade_value from the pipeline directly.
    
    Returns DataFrame with columns: mlb_id, IDfg, name, year, value, value_type, label
    """
    rows = []

    # Only process players with mlb_id (needed for web-app)
    valid = player_values[player_values["mlb_id"].notna()].copy()
    valid["mlb_id"] = valid["mlb_id"].astype(int)
    valid["IDfg"] = valid["IDfg"].astype(int)

    for pid, pdata in valid.groupby("IDfg"):
        pdata = pdata.sort_values("Year")
        name = pdata["Player_Name"].iloc[0]
        mid = pdata["mlb_id"].iloc[0]

        # Get the current trade value (the reference point)
        current = pdata[pdata["Year"] == CURRENT_YEAR]
        current_trade_value = None
        if not current.empty:
            tv = current["trade_value"].iloc[0]
            if pd.notna(tv):
                current_trade_value = tv

        # ── Historical years: compute rolling trade value ──
        # For each historical year, the "trade value" is approximated as:
        # sum of convex(WAR) - estimated_salary for remaining control years
        hist = pdata[pdata["Year"] < CURRENT_YEAR].copy()
        future = pdata[pdata["Year"] >= CURRENT_YEAR].copy()

        fa_year = pdata["FA_Year"].iloc[0] if pd.notna(pdata["FA_Year"].iloc[0]) else None

        for _, row in hist.iterrows():
            yr = int(row["Year"])
            war = row.get("WAR", 0) or 0
            base_val = row.get("Base_Value", 0) or 0
            contract_val = row.get("Contract_Value", 0) or 0
            surplus = row.get("Surplus_Value")

            if pd.isna(war) or war == 0:
                continue

            # Use the single-year surplus as a proxy for "what this player
            # was worth in trade value" in that snapshot year.
            # We accumulate forward-looking surplus from this year onward.
            remaining_years = pdata[
                (pdata["Year"] >= yr)
                & (pdata["Year"] < CURRENT_YEAR)
                & pdata["WAR"].notna()
            ]

            if remaining_years.empty:
                continue

            # Compute cumulative remaining surplus at this point in time
            cum_base = sum(
                _convex_value(r["WAR"], int(r["Year"]))
                for _, r in remaining_years.iterrows()
                if pd.notna(r["WAR"]) and r["WAR"] > 0
            )
            cum_contract = remaining_years["Contract_Value"].fillna(0).sum()
            rolling_value = cum_base - cum_contract

            # Add projected future value if available
            if current_trade_value is not None:
                # Scale future value by decay (older snapshots shouldn't get full future)
                pass  # We only show historical MLB production value
            
            label = f"{war:.1f} WAR"

            rows.append({
                "mlb_id": mid,
                "IDfg": int(pid),
                "name": name,
                "year": yr,
                "value": round(rolling_value),
                "value_type": "mlb_surplus",
                "label": label,
            })

        # ── Current projected trade value (single entry) ──
        if current_trade_value is not None:
            cur_war = future["WAR"].iloc[0] if not future.empty and pd.notna(future["WAR"].iloc[0]) else 0
            rows.append({
                "mlb_id": mid,
                "IDfg": int(pid),
                "name": name,
                "year": CURRENT_YEAR,
                "value": round(current_trade_value),
                "value_type": "projected",
                "label": f"Trade Value",
            })

    result = pd.DataFrame(rows)
    logger.info(
        f"Built {len(result)} MLB timeline entries for {result['mlb_id'].nunique()} players"
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

    # For duplicate (mlb_id, year), prefer mlb_surplus > projected > prospect
    type_priority = {"projected": 0, "mlb_surplus": 1, "prospect": 2}
    combined["_priority"] = combined["value_type"].map(type_priority)
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
    logger.info(f"  Projected entries: {(combined['value_type'] == 'projected').sum()}")
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
