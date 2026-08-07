"""
Daily Trade-Value Snapshots
============================

Moved from ``daily_ros/snapshots.py`` to
``value_determination/pipelines/snapshots.py``.

After each daily value computation:

1. **Full snapshot** — Copies ``player_values_complete.csv`` into
   ``data/generated/value_by_year/snapshots/YYYY-MM-DD.csv`` to give
   full revision history.

2. **Append to trade_value_history.csv** — Adds one row per player
   (current-year only) with ``date = today``.  If the pipeline runs
   twice in a day, that day's entries are replaced (not duplicated).

Storage: ~800 players × 180 game days = 144 K rows per season.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from value_determination.config import Config, logger, CURRENT_YEAR

# ── Paths ────────────────────────────────────────────────────────────────
_OUTPUT_DIR = Config.Paths.OUTPUT_DIR                        # data/generated/value_by_year/
_PVC_FILE = _OUTPUT_DIR / "player_values_complete.csv"
_TVH_FILE = _OUTPUT_DIR / "trade_value_history.csv"
_SNAPSHOT_DIR = _OUTPUT_DIR / "snapshots"


# ═════════════════════════════════════════════════════════════════════════
# 1. Full snapshot
# ═════════════════════════════════════════════════════════════════════════

def save_daily_snapshot(today: date | None = None) -> Path | None:
    """Copy ``player_values_complete.csv`` into ``snapshots/YYYY-MM-DD.csv``.

    If a snapshot for today already exists, it is overwritten.

    Returns:
        Path to the snapshot file, or None if the source file is missing.
    """
    if today is None:
        today = date.today()

    if not _PVC_FILE.exists():
        logger.warning(
            f"Cannot snapshot — {_PVC_FILE.name} not found at {_PVC_FILE}"
        )
        return None

    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    dest = _SNAPSHOT_DIR / f"{today.isoformat()}.csv"
    shutil.copy2(str(_PVC_FILE), str(dest))
    logger.info(f"Saved daily snapshot → {dest}")
    return dest


# ═════════════════════════════════════════════════════════════════════════
# 2. Append to trade_value_history.csv
# ═════════════════════════════════════════════════════════════════════════

# Columns that must appear in the output (matches DB model + existing file)
_TVH_COLS = [
    "mlb_id", "IDfg", "name", "date", "year",
    "value", "value_type", "transaction_type",
    "label", "years_control", "projected_war",
    "projected_salary", "war_per_year",
]


def _build_today_entries(pvc: pd.DataFrame, today: date) -> pd.DataFrame:
    """Extract one trade-value row per current-year player from the PVC.

    Mirrors the format produced by
    ``historical_values.timeline.build_mlb_timeline()`` for current-year
    entries, so both code paths produce identical columns.
    """
    date_str = today.isoformat()

    cur = pvc[
        (pvc["Year"] == CURRENT_YEAR)
        & pvc["mlb_id"].notna()
        & pvc["trade_value"].notna()
    ].copy()

    if cur.empty:
        logger.warning("No current-year rows with mlb_id + trade_value found")
        return pd.DataFrame(columns=_TVH_COLS)

    name_col = "Player_Name" if "Player_Name" in cur.columns else "name"

    rows: list[dict] = []
    for _, row in cur.iterrows():
        yrs = row.get("years_control", 0)
        if pd.isna(yrs):
            yrs = 0
        fut_war = row.get("total_future_war", 0)
        if pd.isna(fut_war):
            fut_war = 0
        total_sal = row.get("total_contract", 0)
        if pd.isna(total_sal) or total_sal == 0:
            total_sal = 0
        war_per_yr = fut_war / yrs if yrs > 0 else 0.0

        label = (
            f"{round(yrs)}yr control, {fut_war:.1f} WAR"
            if yrs > 0
            else f"{fut_war:.1f} WAR projected"
        )

        rows.append({
            "mlb_id":           int(row["mlb_id"]),
            "IDfg":             int(row["IDfg"]),
            "name":             row[name_col],
            "date":             date_str,
            "year":             CURRENT_YEAR,
            "value":            round(row["trade_value"]),
            "value_type":       "mlb_surplus",
            "transaction_type": pd.NA,
            "label":            label,
            "years_control":    round(yrs, 1),
            "projected_war":    round(fut_war, 1),
            "projected_salary": round(total_sal),
            "war_per_year":     round(war_per_yr, 2),
        })

    return pd.DataFrame(rows, columns=_TVH_COLS)


def append_to_trade_value_history(today: date | None = None) -> int:
    """Add today's entries to ``trade_value_history.csv``.

    Deduplication: any existing rows for ``(date == today)`` are removed
    before appending, so re-running the pipeline on the same day replaces
    rather than duplicates.

    Returns:
        Number of rows appended.
    """
    if today is None:
        today = date.today()

    if not _PVC_FILE.exists():
        logger.warning(
            f"Cannot append to history — {_PVC_FILE.name} not found"
        )
        return 0

    pvc = pd.read_csv(_PVC_FILE, low_memory=False, encoding='utf-8')
    new_entries = _build_today_entries(pvc, today)

    if new_entries.empty:
        logger.warning("No entries to append to trade_value_history.csv")
        return 0

    date_str = today.isoformat()

    if _TVH_FILE.exists():
        existing = pd.read_csv(_TVH_FILE, low_memory=False, encoding='utf-8')
        # Remove any rows for today (deduplication)
        existing = existing[existing["date"] != date_str]
        combined = pd.concat([existing, new_entries], ignore_index=True)
    else:
        combined = new_entries

    combined.to_csv(_TVH_FILE, index=False, na_rep="", encoding='utf-8')
    logger.info(
        f"Appended {len(new_entries)} rows to trade_value_history.csv "
        f"(date={date_str}, total={len(combined)})"
    )
    return len(new_entries)


# ═════════════════════════════════════════════════════════════════════════
# Convenience: run both steps
# ═════════════════════════════════════════════════════════════════════════

def save_daily_trade_value_snapshot(today: date | None = None) -> None:
    """Run both snapshot steps (full CSV copy + history append)."""
    if today is None:
        today = date.today()
    save_daily_snapshot(today)
    append_to_trade_value_history(today)