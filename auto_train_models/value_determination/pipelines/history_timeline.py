"""
Historical Values — Unified Trade-Value Timeline Generator (V2)
===============================================================

Combines three V2 sources into `data/generated/value_by_year/trade_value_history.csv`:
  1. **Unified Prospect Valuations** (`prospect_values.csv` — pre-debut prospects only!)
  2. **V2 Monte Carlo Surplus Snapshots** (`surplus_2014..2026.csv` + `_eoy` + live `2026`)
  3. **Spotrac Contract & Trade Events** (`spotrac_transactions.csv`)

Output Schema (`snake_case`):
    `mlbam_id, fg_id, player_name, date, year, value, value_p10, value_p90,
     value_type, transaction_type, label, years_control, projected_war,
     projected_salary, war_per_year`
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_AUTO_TRAIN = Path(__file__).resolve().parents[2]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from core.id_crosswalk import get_crosswalk
from core.name_utils import name_key as _normalise_name
from value_determination.config import CURRENT_YEAR, Config, logger
from value_determination.dollar_valuation import calculate_production_value
from value_determination.prospect_valuation import generate_prospect_values

SURPLUS_DIR = Config.Paths.SURPLUS_DIR
OUTPUT_FILE = Config.Paths.TRADE_VALUE_HISTORY_FILE
PROSPECT_VALUES_FILE = Config.Paths.PROSPECT_VALUES_FILE
CROSSWALK_FILE = Config.Paths.CROSSWALK_FILE
PLAYER_VALUES_FILE = Config.Paths.PLAYER_VALUES_FILE
SPOTRAC_TRANSACTIONS_FILE = Config.Paths.SPOTRAC_TRANSACTIONS_FILE
SNAPSHOT_DIR = Config.Paths.OUTPUT_DIR / "snapshots"

_ZERO_VALUE_TYPES = {"elected_fa", "released", "designated", "non_tendered", "opt_out", "option_declined"}


def _build_id_and_name_maps(
    player_values: Optional[pd.DataFrame],
) -> tuple[dict[int, int], dict[int, int], dict[int, str], dict[str, int]]:
    """Return `(fg_to_mlbam, mlbam_to_fg, mlbam_to_name, name_key_to_mlbam)`."""
    mlbam_to_fg_base, fg_to_mlbam_base = get_crosswalk()
    fg_to_mlb = dict(fg_to_mlbam_base)
    mlb_to_fg = dict(mlbam_to_fg_base)
    mlb_to_name: dict[int, str] = {}
    name_to_mlb: dict[str, int] = {}

    if CROSSWALK_FILE.exists():
        cw = pd.read_csv(CROSSWALK_FILE, encoding="utf-8-sig", low_memory=False)
        cw.columns = cw.columns.str.strip()
        for _, r in cw.iterrows():
            m = pd.to_numeric(r.get("mlbam_id"), errors="coerce")
            nm = str(r.get("name", "")).strip()
            if pd.notna(m) and nm:
                mid = int(m)
                mlb_to_name[mid] = nm
                nk = _normalise_name(nm)
                if nk:
                    name_to_mlb[nk] = mid

    if player_values is not None and not player_values.empty:
        mlb_col = "mlbam_id" if "mlbam_id" in player_values.columns else "mlb_id"
        fg_col = "fg_id" if "fg_id" in player_values.columns else "IDfg"
        nm_col = "player_name" if "player_name" in player_values.columns else (
            "Player_Name" if "Player_Name" in player_values.columns else "name"
        )
        sub = player_values.dropna(subset=[mlb_col]).drop_duplicates(mlb_col)
        for _, r in sub.iterrows():
            mid = int(r[mlb_col])
            if pd.notna(r.get(fg_col)):
                fg = int(r[fg_col])
                fg_to_mlb[fg] = mid
                mlb_to_fg[mid] = fg
            nm = str(r.get(nm_col, "")).strip()
            if nm:
                mlb_to_name[mid] = nm
                nk = _normalise_name(nm)
                if nk:
                    name_to_mlb[nk] = mid

    return fg_to_mlb, mlb_to_fg, mlb_to_name, name_to_mlb


# =============================================================================
# 1. Prospect Timeline (`prospect_values.csv`)
# =============================================================================

def build_prospect_timeline(player_values: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build pre-debut prospect timeline entries directly from `prospect_values.csv`."""
    if not PROSPECT_VALUES_FILE.exists():
        generate_prospect_values()
    if not PROSPECT_VALUES_FILE.exists():
        return pd.DataFrame()

    pv_df = pd.read_csv(PROSPECT_VALUES_FILE, low_memory=False)
    _, mlb_to_fg, mlb_to_name, _ = _build_id_and_name_maps(player_values)

    # Keep only pre-debut observations (once a player debuts in MLB, MLB surplus tracks them!)
    pre_debut = pv_df[(pv_df["mlbam_id"].notna()) & (~pv_df["has_mlb"].fillna(False).astype(bool))].copy()

    rows: list[dict] = []
    for _, p in pre_debut.iterrows():
        mid = int(p["mlbam_id"])
        val = float(p.get("trade_value", 0) or 0)
        if val <= 0:
            continue

        yr = int(p["year"])
        fg_int = mlb_to_fg.get(mid)
        if fg_int is None and pd.notna(p.get("fg_id")) and str(p["fg_id"]).isdigit():
            fg_int = int(p["fg_id"])

        fv_str = str(p.get("fv") or "?").replace(".0", "")
        label_parts = [f"FV {fv_str}"]
        if pd.notna(p.get("top_100")):
            label_parts.append(f"#{int(p['top_100'])}")

        rows.append({
            "mlbam_id": mid,
            "fg_id": fg_int if fg_int is not None else pd.NA,
            "player_name": mlb_to_name.get(mid, str(p.get("player_name", ""))),
            "date": f"{yr - 1}-10-01",
            "year": yr,
            "value": round(val),
            "value_p10": round(float(p["trade_value_p10"])) if pd.notna(p.get("trade_value_p10")) else pd.NA,
            "value_p90": round(float(p["trade_value_p90"])) if pd.notna(p.get("trade_value_p90")) else pd.NA,
            "value_type": "prospect",
            "transaction_type": pd.NA,
            "label": ", ".join(label_parts),
            "years_control": pd.NA,
            "projected_war": pd.NA,
            "projected_salary": pd.NA,
            "war_per_year": pd.NA,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("value", ascending=False).drop_duplicates(["mlbam_id", "year"])
    logger.info("Built %d pre-debut prospect timeline entries", len(result))
    return result


# =============================================================================
# 2. MLB Surplus Timeline (`surplus_*.csv` + `player_values_complete.csv`)
# =============================================================================

def _load_surplus_files() -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    reg_by_yr: dict[int, pd.DataFrame] = {}
    eoy_by_yr: dict[int, pd.DataFrame] = {}
    if not SURPLUS_DIR.exists():
        return reg_by_yr, eoy_by_yr

    for path in sorted(SURPLUS_DIR.glob("surplus_*.csv")):
        try:
            parts = path.stem.split("_")
            yr = int(parts[1])
            is_eoy = len(parts) > 2 and parts[2] == "eoy"
            df = pd.read_csv(path, low_memory=False)
            if is_eoy:
                eoy_by_yr[yr] = df
            else:
                reg_by_yr[yr] = df
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path.name, exc)
    return reg_by_yr, eoy_by_yr


def build_mlb_timeline(
    player_values: Optional[pd.DataFrame],
    current_anchor_date: Optional[str] = None,
) -> pd.DataFrame:
    surplus_by_year, surplus_eoy_by_year = _load_surplus_files()
    fg_to_mlb, _, mlb_to_name, _ = _build_id_and_name_maps(player_values)

    rows: list[dict] = []

    def _process_sdf(sdf: pd.DataFrame, date_str: str, year_val: int) -> None:
        fg_col = "fg_id" if "fg_id" in sdf.columns else "IDfg"
        nm_col = "player_name" if "player_name" in sdf.columns else "Name"
        yrs_col = "years_control" if "years_control" in sdf.columns else "years_of_control"
        war_col = "control_war" if "control_war" in sdf.columns else "total_future_WAR"
        sal_col = "control_salary_cost" if "control_salary_cost" in sdf.columns else "total_future_salary"

        for _, pr in sdf.iterrows():
            mid = None
            if pd.notna(pr.get("mlbam_id")):
                try:
                    mid = int(pr["mlbam_id"])
                except (TypeError, ValueError):
                    mid = None
            if mid is None and pd.notna(pr.get(fg_col)):
                mid = fg_to_mlb.get(int(pr[fg_col]))
            if mid is None:
                continue

            tv = pr.get("trade_value", np.nan)
            if pd.isna(tv):
                continue

            yrs_ctrl = float(pr.get(yrs_col, 0) or 0)
            total_war = float(pr.get(war_col, 0) or 0)
            total_sal = float(pr.get(sal_col, 0) or 0)
            if total_war <= 0 and float(tv) == 0:
                continue

            war_per_yr = total_war / yrs_ctrl if yrs_ctrl > 0 else 0.0
            label = (
                f"{round(yrs_ctrl)}yr control, {total_war:.1f} WAR"
                if yrs_ctrl > 0
                else f"{total_war:.1f} WAR projected"
            )

            rows.append({
                "mlbam_id": mid,
                "fg_id": int(pr[fg_col]) if pd.notna(pr.get(fg_col)) else pd.NA,
                "player_name": mlb_to_name.get(mid, str(pr.get(nm_col, ""))),
                "date": date_str,
                "year": year_val,
                "value": round(float(tv)),
                "value_p10": round(float(pr["trade_value_p10"])) if pd.notna(pr.get("trade_value_p10")) else pd.NA,
                "value_p90": round(float(pr["trade_value_p90"])) if pd.notna(pr.get("trade_value_p90")) else pd.NA,
                "value_type": "mlb_surplus",
                "transaction_type": pd.NA,
                "label": label,
                "years_control": round(yrs_ctrl, 1),
                "projected_war": round(total_war, 1),
                "projected_salary": round(total_sal),
                "war_per_year": round(war_per_yr, 2),
            })

    for snap_year in sorted(surplus_by_year):
        if snap_year < CURRENT_YEAR:
            _process_sdf(surplus_by_year[snap_year], f"{snap_year}-04-01", snap_year)

    for snap_year in sorted(surplus_eoy_by_year):
        eoy_year = snap_year - 1
        if Config.History.CUTOFF_START <= eoy_year < CURRENT_YEAR:
            _process_sdf(surplus_eoy_by_year[snap_year], f"{eoy_year}-10-01", eoy_year)

    if player_values is not None and not player_values.empty:
        yr_col = "year" if "year" in player_values.columns else "Year"
        mlb_col = "mlbam_id" if "mlbam_id" in player_values.columns else "mlb_id"
        fg_col = "fg_id" if "fg_id" in player_values.columns else "IDfg"
        nm_col = "player_name" if "player_name" in player_values.columns else "Player_Name"

        pv = player_values[
            (player_values[yr_col] == CURRENT_YEAR)
            & player_values[mlb_col].notna()
            & player_values["trade_value"].notna()
        ].copy()

        for _, row in pv.iterrows():
            mid = int(row[mlb_col])
            yrs = float(row.get("years_control", 0) or 0)
            cwar = float(row.get("control_war", row.get("total_future_war", 0)) or 0)
            csal = float(row.get("control_salary_cost", row.get("total_contract", 0)) or 0)
            wpy = cwar / yrs if yrs > 0 else 0.0
            label = f"{round(yrs)}yr control, {cwar:.1f} WAR" if yrs > 0 else f"{cwar:.1f} WAR projected"

            rows.append({
                "mlbam_id": mid,
                "fg_id": int(row[fg_col]),
                "player_name": row[nm_col],
                "date": current_anchor_date or f"{CURRENT_YEAR}-03-01",
                "year": CURRENT_YEAR,
                "value": round(float(row["trade_value"])),
                "value_p10": round(float(row["trade_value_p10"])) if pd.notna(row.get("trade_value_p10")) else pd.NA,
                "value_p90": round(float(row["trade_value_p90"])) if pd.notna(row.get("trade_value_p90")) else pd.NA,
                "value_type": "mlb_surplus",
                "transaction_type": pd.NA,
                "label": label,
                "years_control": round(yrs, 1),
                "projected_war": round(cwar, 1),
                "projected_salary": round(csal),
                "war_per_year": round(wpy, 2),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("value", ascending=False).drop_duplicates(["mlbam_id", "date"], keep="first")
    logger.info("Built %d MLB surplus timeline entries", len(result))
    return result


# =============================================================================
# 3. Spotrac Transaction Overlay & Monthly Interpolation
# =============================================================================

_PRE_ARB_SALARY_RE = re.compile(r"Signed a 1 year \$[\d,]+(?:k|K)?\s+contract with", re.IGNORECASE)
_KEEP_OTHER_PATTERNS = re.compile(
    r"non.?tendered|option\s+for|player\s+option|termination\s+option|opted?\s+out",
    re.IGNORECASE,
)


def _build_cots_lookup(name_to_mlb: dict[str, int]) -> dict[tuple[int, int], dict]:
    lookup: dict[tuple[int, int], dict] = {}
    cots_dir = Config.Paths.COTS_BY_YEAR_DIR
    if not cots_dir.exists():
        return lookup

    for cots_file in sorted(cots_dir.glob("*.csv")):
        try:
            year = int(cots_file.stem)
            cots = pd.read_csv(cots_file, low_memory=False)
            if "player" not in cots.columns or "years_of_control" not in cots.columns:
                continue
            for _, row in cots.iterrows():
                nk = _normalise_name(str(row.get("player", "")))
                yoc = row.get("years_of_control")
                svc = row.get("service_time")
                if not nk or pd.isna(yoc):
                    continue
                mid = name_to_mlb.get(nk)
                if mid:
                    lookup[(mid, year)] = {
                        "years_control": int(float(yoc)),
                        "service_time": float(svc) if pd.notna(svc) else 0.0,
                    }
        except Exception:
            continue
    return lookup


def build_transaction_entries(
    player_values: Optional[pd.DataFrame],
    value_timeline: pd.DataFrame,
) -> pd.DataFrame:
    if not SPOTRAC_TRANSACTIONS_FILE.exists() or value_timeline.empty:
        return pd.DataFrame()

    txn = pd.read_csv(SPOTRAC_TRANSACTIONS_FILE, parse_dates=["date"], low_memory=False)
    _, mlb_to_fg, mlb_to_name, name_to_mlb = _build_id_and_name_maps(player_values)
    cots_lookup = _build_cots_lookup(name_to_mlb)

    # Reclassify arb / pre-arb / minor-league / initial signings & QOs
    arb_mask = txn["transaction_type"].isin(["fa_signing", "signing"]) & txn["description"].str.contains(
        "avoiding arbitration|settling in arbitration", case=False, na=False, regex=True
    )
    txn.loc[arb_mask, "transaction_type"] = "arbitration"

    min_sal_map = Config.History.HISTORICAL_MIN_SALARY
    for idx in txn.index[txn["transaction_type"].isin(["fa_signing", "signing"])]:
        desc = str(txn.at[idx, "description"])
        ev_date = txn.at[idx, "date"]
        pname = str(txn.at[idx, "player_name"])
        try:
            ts = pd.Timestamp(ev_date)
            yr = ts.year + (1 if ts.month >= 10 else 0)
        except Exception:
            continue
        mid = name_to_mlb.get(_normalise_name(pname))
        cots_svc = cots_lookup.get((mid, yr), {}).get("service_time") if mid else None
        if cots_svc is not None:
            if cots_svc < 3.0:
                txn.at[idx, "transaction_type"] = "pre_arb"
                continue
            if cots_svc < 6.0:
                txn.at[idx, "transaction_type"] = "arbitration"
                continue
        if _PRE_ARB_SALARY_RE.search(desc):
            sal_m = re.search(r"\$([\d,]+(?:\.\d+)?)(k|K)?\s", desc)
            if sal_m:
                sval = float(sal_m.group(1).replace(",", "")) * (1000.0 if sal_m.group(2) else 1.0)
                if sval <= min_sal_map.get(yr, 800_000) * 1.5:
                    txn.at[idx, "transaction_type"] = "pre_arb"

    txn.loc[
        txn["transaction_type"].isin(["fa_signing", "signing"])
        & txn["description"].str.contains("minor league", case=False, na=False),
        "transaction_type",
    ] = "minor_league_signing"

    txn.loc[
        txn["transaction_type"].isin(["fa_signing", "signing"])
        & ~txn["description"].str.contains(r"\$", na=False, regex=True),
        "transaction_type",
    ] = "initial_signing"

    qo_ext = txn["description"].str.contains(r"extended.*Qualifying Offer", case=False, na=False, regex=True)
    qo_dec = txn["description"].str.contains(r"Declined.*Qualifying Offer", case=False, na=False, regex=True)
    txn.loc[qo_ext, "transaction_type"] = "elected_fa"
    txn = txn[~qo_dec].copy()

    skip_types = {"pre_arb", "arbitration", "option_exercised", "minor_league_signing", "initial_signing"}
    txn = txn[~txn["transaction_type"].isin(skip_types)].copy()
    txn = txn[
        ~(
            (txn["transaction_type"] == "other")
            & txn["description"].str.contains(r"exercised.*option|option\s+for", case=False, na=False, regex=True)
        )
    ].copy()
    if "other" in txn["transaction_type"].values:
        other_m = txn["transaction_type"] == "other"
        keep_o = txn.loc[other_m, "description"].str.contains(_KEEP_OTHER_PATTERNS, na=False)
        txn = txn[~(other_m & ~keep_o)].copy()

    valid = value_timeline[value_timeline["value_type"].isin(["prospect", "mlb_surplus"])].copy()
    valid["_dt"] = pd.to_datetime(valid["date"], errors="coerce")
    valid = valid.dropna(subset=["_dt"]).sort_values("_dt")
    value_map = {int(m): g for m, g in valid.groupby("mlbam_id")}

    rows: list[dict] = []
    for _, p_txns in txn.groupby("spotrac_id"):
        pname = p_txns.iloc[0]["player_name"]
        mid = name_to_mlb.get(_normalise_name(str(pname)))
        if mid is None or mid not in value_map:
            continue

        fg_id = mlb_to_fg.get(mid)
        canon_name = mlb_to_name.get(mid, str(pname))
        p_tl = value_map[mid]

        for _, ev in p_txns.sort_values("date").iterrows():
            ev_type = str(ev["transaction_type"])
            ev_date = pd.to_datetime(ev["date"], errors="coerce")
            if pd.isna(ev_date) or ev_date.year < Config.History.CUTOFF_START:
                continue
            desc = str(ev.get("description", "") or "")

            if ev_type == "other":
                dl = desc.lower()
                if "non" in dl and "tender" in dl:
                    ev_type = "non_tendered"
                elif "termination" in dl or "opted out" in dl or "opt out" in dl:
                    ev_type = "opt_out"

            if ev_type in _ZERO_VALUE_TYPES:
                val, p10, p90 = 0, 0, 0
                vtype = "free_agent"
                yrs_ctrl = proj_war = proj_sal = war_yr = pd.NA
            else:
                if ev_type in ("extension", "fa_signing", "signing"):
                    cand = p_tl[(p_tl["_dt"] > ev_date) & (p_tl["value_type"] == "mlb_surplus")]
                    if cand.empty:
                        cand = p_tl[(p_tl["_dt"] <= ev_date) & (p_tl["value_type"] == "mlb_surplus")]
                elif ev_type == "drafted":
                    cand = p_tl[(p_tl["_dt"] <= ev_date) & (p_tl["value_type"] == "prospect")]
                    if cand.empty:
                        cand = p_tl[p_tl["value_type"] == "prospect"]
                else:
                    cand = p_tl[(p_tl["_dt"] <= ev_date) & (p_tl["value_type"] == "mlb_surplus")]
                    if cand.empty:
                        cand = p_tl[p_tl["_dt"] <= ev_date]

                if cand.empty:
                    continue
                ref = cand.iloc[0] if ev_type in ("extension", "fa_signing", "signing") and (cand["_dt"] > ev_date).any() else cand.iloc[-1]
                val = ref["value"]
                p10 = ref.get("value_p10", pd.NA)
                p90 = ref.get("value_p90", pd.NA)
                vtype = ref["value_type"]
                yrs_ctrl = ref.get("years_control")
                proj_war = ref.get("projected_war")
                proj_sal = ref.get("projected_salary")
                war_yr = ref.get("war_per_year")

            if (
                ev_type == "fa_signing"
                and pd.notna(yrs_ctrl)
                and float(yrs_ctrl) > 1
                and "1 year" in desc.lower()
            ):
                continue

            rows.append({
                "mlbam_id": mid,
                "fg_id": fg_id if fg_id is not None else pd.NA,
                "player_name": canon_name,
                "date": ev_date.strftime("%Y-%m-%d"),
                "year": ev_date.year,
                "value": round(float(val)),
                "value_p10": round(float(p10)) if pd.notna(p10) else pd.NA,
                "value_p90": round(float(p90)) if pd.notna(p90) else pd.NA,
                "value_type": vtype,
                "transaction_type": "fa_signing" if ev_type == "signing" else ev_type,
                "label": desc.strip() or ev_type.replace("_", " ").title(),
                "years_control": yrs_ctrl,
                "projected_war": proj_war,
                "projected_salary": proj_sal,
                "war_per_year": war_yr,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(["mlbam_id", "date", "transaction_type"])
    logger.info("Built %d Spotrac transaction timeline entries", len(out))
    return out


def _interpolate_monthly(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return combined

    combined = combined.copy()
    combined["_dt"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values(["mlbam_id", "_dt"]).reset_index(drop=True)

    def _season_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
        if start >= end:
            return 0
        total = 0
        for y in range(start.year, end.year + 1):
            lo = max(start, pd.Timestamp(y, 4, 1))
            hi = min(end, pd.Timestamp(y, 10, 1))
            if lo < hi:
                total += (hi - lo).days
        return total

    interp_rows: list[dict] = []
    for mid, grp in combined.groupby("mlbam_id", sort=False):
        grp = grp.sort_values("_dt").reset_index(drop=True)
        if len(grp) < 2:
            continue
        for idx in range(len(grp) - 1):
            ra, rb = grp.iloc[idx], grp.iloc[idx + 1]
            dt_a, dt_b = ra["_dt"], rb["_dt"]
            if pd.isna(dt_a) or pd.isna(dt_b) or (dt_b - dt_a).days <= 32:
                continue

            total_days = (dt_b - dt_a).days
            val_a, val_b = float(ra["value"]), float(rb["value"])
            b_txn = pd.notna(rb.get("transaction_type")) and str(rb.get("transaction_type")) != ""
            b_fa_exit = str(rb.get("transaction_type", "")) == "elected_fa"
            total_season = _season_days(dt_a, dt_b)

            cur = pd.Timestamp((dt_a + pd.DateOffset(months=1)).year, (dt_a + pd.DateOffset(months=1)).month, 1)
            while cur < dt_b:
                if b_txn and not b_fa_exit:
                    frac = 0.0
                elif b_fa_exit:
                    frac = min((cur - dt_a).days / total_days, 1.0)
                elif total_season > 0:
                    frac = min(_season_days(dt_a, cur) / total_season, 1.0)
                else:
                    frac = 0.0

                iv = val_a + frac * (val_b - val_a)
                p10_a, p10_b = ra.get("value_p10"), rb.get("value_p10")
                p90_a, p90_b = ra.get("value_p90"), rb.get("value_p90")
                ip10 = round(float(p10_a) + frac * (float(p10_b) - float(p10_a))) if pd.notna(p10_a) and pd.notna(p10_b) else p10_a
                ip90 = round(float(p90_a) + frac * (float(p90_b) - float(p90_a))) if pd.notna(p90_a) and pd.notna(p90_b) else p90_a

                interp_rows.append({
                    "mlbam_id": mid,
                    "fg_id": ra.get("fg_id") if pd.notna(ra.get("fg_id")) else rb.get("fg_id"),
                    "player_name": ra["player_name"],
                    "date": cur.strftime("%Y-%m-%d"),
                    "year": cur.year,
                    "value": round(iv),
                    "value_p10": ip10,
                    "value_p90": ip90,
                    "value_type": "mlb_surplus",
                    "transaction_type": pd.NA,
                    "label": ra.get("label", ""),
                    "years_control": ra.get("years_control"),
                    "projected_war": ra.get("projected_war"),
                    "projected_salary": ra.get("projected_salary"),
                    "war_per_year": ra.get("war_per_year"),
                })
                cur = pd.Timestamp((cur + pd.DateOffset(months=1)).year, (cur + pd.DateOffset(months=1)).month, 1)

    combined = combined.drop(columns=["_dt"])
    if interp_rows:
        combined = pd.concat([combined, pd.DataFrame(interp_rows)], ignore_index=True)
    return combined.sort_values(["mlbam_id", "date"]).reset_index(drop=True)


def generate_timeline() -> pd.DataFrame:
    """Build and write `trade_value_history.csv` on the V2 Monte Carlo scale."""
    pv = pd.read_csv(PLAYER_VALUES_FILE, low_memory=False) if PLAYER_VALUES_FILE.exists() else None
    current_anchor_date = f"{CURRENT_YEAR}-03-01"

    prospect_tl = build_prospect_timeline(pv)
    mlb_tl = build_mlb_timeline(pv, current_anchor_date=current_anchor_date)

    pre_combined = pd.concat([prospect_tl, mlb_tl], ignore_index=True)
    if not pre_combined.empty:
        pri = {"prospect": 0, "mlb_surplus": 1}
        pre_combined["_p"] = pre_combined["value_type"].map(pri).fillna(2)
        pre_combined = (
            pre_combined.sort_values("_p")
            .drop_duplicates(["mlbam_id", "date"], keep="first")
            .drop(columns=["_p"])
        )

    txn_entries = build_transaction_entries(pv, pre_combined)
    combined = pd.concat([pre_combined, txn_entries], ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values(["mlbam_id", "date"]).reset_index(drop=True)
        qo_mask = combined["transaction_type"] == "elected_fa"
        combined.loc[qo_mask, "label"] = pd.NA
        combined = _interpolate_monthly(combined)
        combined.loc[combined["transaction_type"] == "elected_fa", "transaction_type"] = pd.NA

    col_order = [
        "mlbam_id", "fg_id", "player_name", "date", "year",
        "value", "value_p10", "value_p90",
        "value_type", "transaction_type", "label",
        "years_control", "projected_war", "projected_salary", "war_per_year",
    ]
    for c in col_order:
        if c not in combined.columns:
            combined[c] = pd.NA
    combined = combined[col_order]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False, na_rep="")
    logger.info(
        "Wrote %d V2 timeline entries across %d players -> %s",
        len(combined),
        combined["mlbam_id"].nunique() if not combined.empty else 0,
        OUTPUT_FILE,
    )
    return combined


if __name__ == "__main__":
    generate_timeline()
