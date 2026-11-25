"""
Daily Trade-Value Snapshots (V2 Schema)
=======================================

1. Copies `player_values_complete.csv` into `snapshots/YYYY-MM-DD.csv`.
2. Appends/replaces today's current-year entries in `trade_value_history.csv`
   using canonical `snake_case` columns (`mlbam_id`, `fg_id`, `player_name`,
   `trade_value`, `trade_value_p10`, `trade_value_p90`, `control_war`, etc.).
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from value_determination.config import CURRENT_YEAR, Config, logger

_OUTPUT_DIR = Config.Paths.OUTPUT_DIR
_PVC_FILE = _OUTPUT_DIR / "player_values_complete.csv"
_TVH_FILE = _OUTPUT_DIR / "trade_value_history.csv"
_SNAPSHOT_DIR = _OUTPUT_DIR / "snapshots"

_TVH_COLS = [
    "mlbam_id", "fg_id", "player_name", "date", "year",
    "value", "value_p10", "value_p90",
    "value_type", "transaction_type",
    "label", "years_control", "projected_war",
    "projected_salary", "war_per_year",
]


def save_daily_snapshot(today: date | None = None) -> Path | None:
    if today is None:
        today = date.today()
    if not _PVC_FILE.exists():
        logger.warning("Cannot snapshot — %s not found", _PVC_FILE)
        return None
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    dest = _SNAPSHOT_DIR / f"{today.isoformat()}.csv"
    shutil.copy2(str(_PVC_FILE), str(dest))
    logger.info("Saved daily snapshot -> %s", dest)
    return dest


def _build_today_entries(pvc: pd.DataFrame, today: date) -> pd.DataFrame:
    date_str = today.isoformat()
    yr_col = "year" if "year" in pvc.columns else "Year"
    mlb_col = "mlbam_id" if "mlbam_id" in pvc.columns else "mlb_id"
    fg_col = "fg_id" if "fg_id" in pvc.columns else "IDfg"
    name_col = "player_name" if "player_name" in pvc.columns else ("Player_Name" if "Player_Name" in pvc.columns else "name")

    cur = pvc[
        (pvc[yr_col] == CURRENT_YEAR)
        & pvc[mlb_col].notna()
        & pvc["trade_value"].notna()
    ].copy()

    if cur.empty:
        return pd.DataFrame(columns=_TVH_COLS)

    rows: list[dict] = []
    for _, row in cur.iterrows():
        yrs = row.get("years_control", 0)
        yrs = 0.0 if pd.isna(yrs) else float(yrs)

        ctrl_war = row.get("control_war", row.get("total_future_war", 0.0))
        ctrl_war = 0.0 if pd.isna(ctrl_war) else float(ctrl_war)

        ctrl_sal = row.get("control_salary_cost", row.get("total_contract", 0.0))
        ctrl_sal = 0.0 if pd.isna(ctrl_sal) else float(ctrl_sal)

        war_per_yr = ctrl_war / yrs if yrs > 0 else 0.0
        label = (
            f"{round(yrs)}yr control, {ctrl_war:.1f} WAR"
            if yrs > 0
            else f"{ctrl_war:.1f} WAR projected"
        )

        rows.append({
            "mlbam_id": int(row[mlb_col]),
            "fg_id": int(row[fg_col]),
            "player_name": row[name_col],
            "date": date_str,
            "year": CURRENT_YEAR,
            "value": round(float(row["trade_value"])),
            "value_p10": round(float(row["trade_value_p10"])) if pd.notna(row.get("trade_value_p10")) else pd.NA,
            "value_p90": round(float(row["trade_value_p90"])) if pd.notna(row.get("trade_value_p90")) else pd.NA,
            "value_type": "mlb_surplus",
            "transaction_type": pd.NA,
            "label": label,
            "years_control": round(yrs, 1),
            "projected_war": round(ctrl_war, 1),
            "projected_salary": round(ctrl_sal),
            "war_per_year": round(war_per_yr, 2),
        })

    return pd.DataFrame(rows, columns=_TVH_COLS)


def append_to_trade_value_history(today: date | None = None) -> int:
    if today is None:
        today = date.today()
    if not _PVC_FILE.exists():
        return 0

    pvc = pd.read_csv(_PVC_FILE, low_memory=False, encoding="utf-8")
    new_entries = _build_today_entries(pvc, today)
    if new_entries.empty:
        return 0

    date_str = today.isoformat()
    if _TVH_FILE.exists():
        existing = pd.read_csv(_TVH_FILE, low_memory=False, encoding="utf-8")
        # Normalize legacy headers if appending to an older file
        existing = existing.rename(columns={"mlb_id": "mlbam_id", "IDfg": "fg_id", "name": "player_name"})
        existing = existing[existing["date"] != date_str]
        for col in _TVH_COLS:
            if col not in existing.columns:
                existing[col] = pd.NA
        combined = pd.concat([existing[_TVH_COLS], new_entries], ignore_index=True)
    else:
        combined = new_entries

    combined.to_csv(_TVH_FILE, index=False, na_rep="", encoding="utf-8")
    logger.info("Appended %d rows to trade_value_history.csv (date=%s)", len(new_entries), date_str)
    return len(new_entries)


def save_daily_trade_value_snapshot(today: date | None = None) -> None:
    if today is None:
        today = date.today()
    save_daily_snapshot(today)
    append_to_trade_value_history(today)
