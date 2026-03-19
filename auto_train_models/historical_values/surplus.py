"""
Historical Values — Surplus Calculator
========================================

For each snapshot year S (cutoff = S − 1):
  1. Load the four prediction CSVs from ``projections/cutoff_{S-1}/``.
  2. Load the Cot's salary file ``salary/by_year/{S}.csv``.
  3. Match players between projections and Cot's by name (+ ID enrichment).
  4. Compute projected WAR per year, sum over remaining control years.
  5. Output:  ``surplus/surplus_{S}.csv``

Usage:
    cd auto_train_models
    python -m historical_values.surplus --start 2014 --end 2026
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

_AUTO_TRAIN = Path(__file__).resolve().parents[1]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from historical_values.config import (
    Config, logger,
    PROJECTIONS_DIR, SURPLUS_DIR,
    COTS_BY_YEAR_DIR, DATA_DIR,
    HISTORIC_BATTING_FILE, HISTORIC_PITCHING_FILE,
    HISTORIC_BATTING_FILE_CLASSIC, HISTORIC_PITCHING_FILE_CLASSIC,
    ROSTER_FILE,
)
from historical_values.war import calculate_batter_war, calculate_pitcher_war, war_to_dollars
from value_determination.value_calculator import (
    calculate_contract_value as _vd_contract_value,
)
from core.name_utils import name_key as _name_key_fn, normalize_team as _normalise_team


# ═══════════════════════════════════════════════════════════════════════════════
# ID crosswalk helpers
# ═══════════════════════════════════════════════════════════════════════════════

_mlbam_xw_cache: Optional[pd.DataFrame] = None
_idfg_team_cache: Optional[Dict[int, str]] = None
_hist_app_cache: Optional[pd.DataFrame] = None


def _build_mlbam_crosswalk() -> pd.DataFrame:
    """Build IDfg → mlbam_id mapping from statcast-enriched historic data."""
    global _mlbam_xw_cache
    if _mlbam_xw_cache is not None:
        return _mlbam_xw_cache

    frames = []
    for path in [HISTORIC_BATTING_FILE, HISTORIC_PITCHING_FILE]:
        if path.exists():
            df = pd.read_csv(path, usecols=["IDfg", "sc_mlbam_id"], low_memory=False)
            df = df.dropna(subset=["sc_mlbam_id"]).drop_duplicates("IDfg")
            frames.append(df)
    if frames:
        xw = pd.concat(frames, ignore_index=True).drop_duplicates("IDfg")
        xw = xw.rename(columns={"sc_mlbam_id": "mlbam_id"})
        xw["mlbam_id"] = xw["mlbam_id"].astype("Int64")
    else:
        xw = pd.DataFrame(columns=["IDfg", "mlbam_id"])
    _mlbam_xw_cache = xw
    return xw


def _build_idfg_team_map() -> Dict[int, str]:
    """Build IDfg → team abbreviation from the active roster."""
    global _idfg_team_cache
    if _idfg_team_cache is not None:
        return _idfg_team_cache

    team_info = DATA_DIR / "active_roster" / "team_info.csv"
    if not ROSTER_FILE.exists() or not team_info.exists():
        _idfg_team_cache = {}
        return _idfg_team_cache

    teams = pd.read_csv(team_info, usecols=["team_id", "abbreviation"])
    tid_to_abbr = dict(zip(teams["team_id"], teams["abbreviation"]))

    roster = pd.read_csv(ROSTER_FILE, usecols=["fg_id", "team_id"], low_memory=False)
    roster = roster.dropna(subset=["fg_id"])
    roster["fg_id"] = pd.to_numeric(roster["fg_id"], errors="coerce").dropna().astype(int)
    roster["team_abbr"] = roster["team_id"].map(tid_to_abbr)
    roster = roster.dropna(subset=["team_abbr"])

    _idfg_team_cache = dict(zip(roster["fg_id"], roster["team_abbr"]))
    return _idfg_team_cache



# ═══════════════════════════════════════════════════════════════════════════════
# Cot's ↔ projection matching
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_cots_with_idfg(
    cots: pd.DataFrame,
    proj_players: pd.DataFrame,
) -> pd.DataFrame:
    """Add an ``IDfg`` column to the Cot's DataFrame via name + team matching."""
    name_to_idfgs: Dict[str, list] = {}
    for _, row in proj_players.iterrows():
        key = row["_name_key"]
        name_to_idfgs.setdefault(key, []).append(int(row["IDfg"]))

    idfg_team_map = _build_idfg_team_map()
    idfg_col = []
    for _, row in cots.iterrows():
        candidates = name_to_idfgs.get(row["_name_key"], [])
        if len(candidates) == 1:
            idfg_col.append(candidates[0])
        elif len(candidates) > 1:
            cots_team = _normalise_team(str(row.get("team", "")))
            matched = None
            for cand in candidates:
                if _normalise_team(idfg_team_map.get(cand, "")) == cots_team:
                    matched = cand
                    break
            idfg_col.append(matched if matched else pd.NA)
        else:
            idfg_col.append(pd.NA)

    cots = cots.copy()
    cots["IDfg"] = pd.array(idfg_col, dtype="Int64")
    return cots


# ═══════════════════════════════════════════════════════════════════════════════
# Service-time helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_historical_appearances() -> pd.DataFrame:
    global _hist_app_cache
    if _hist_app_cache is not None:
        return _hist_app_cache

    frames = []
    for path in [HISTORIC_BATTING_FILE_CLASSIC, HISTORIC_PITCHING_FILE_CLASSIC]:
        if path.exists():
            df = pd.read_csv(path, usecols=["IDfg", "Season"], low_memory=False)
            frames.append(df[["IDfg", "Season"]].drop_duplicates())
    _hist_app_cache = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
    return _hist_app_cache


def _estimate_service_time(idfg: int, as_of_season: int) -> float:
    app = _load_historical_appearances()
    if app.empty:
        return 0.0
    return float(app.loc[(app["IDfg"] == idfg) & (app["Season"] <= as_of_season), "Season"].nunique())


def _years_of_control_from_svc(service_time: float) -> int:
    return max(0, math.ceil(Config.SERVICE_TIME_FA - service_time))


def _classify_status(service_time: float) -> str:
    if pd.isna(service_time):
        return "Unknown"
    if service_time < 3:
        return "Pre-Arb"
    elif service_time < 4:
        return "Arb-1"
    elif service_time < 5:
        return "Arb-2"
    elif service_time < 6:
        return "Arb-3"
    else:
        return "Signed"


def _estimate_control_salaries(
    idfg: int,
    svc: float,
    control_years: list[int],
    player_proj: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """Estimate annual salaries over control years.

    Delegates arb-year salary estimation to
    ``value_determination.value_calculator.calculate_contract_value()``
    so the escalation model (prev_value tracking, 1.1× floor) is
    identical to the live pipeline.

    Pre-arb years use ``Config.HISTORICAL_MIN_SALARY`` for the
    year-appropriate league minimum.
    """
    ctrl_rows = []
    for i, yr in enumerate(control_years):
        future_svc = svc + i
        future_status = _classify_status(future_svc)
        if future_status in ("FA", "Signed", "Unknown"):
            break
        yr_war_row = player_proj.loc[player_proj["Year"] == yr, "WAR"]
        yr_war = max(0.0, float(yr_war_row.iloc[0])) if len(yr_war_row) else 0.0
        bv = war_to_dollars(yr_war, yr)

        # Pre-arb: pass historical minimum as Payroll so VD uses it directly
        payroll = (
            Config.HISTORICAL_MIN_SALARY.get(yr, 720_000)
            if future_status == "Pre-Arb"
            else np.nan
        )
        ctrl_rows.append({
            "IDfg": idfg,
            "Year": yr,
            "Base_Value": bv,
            "Normalized_Status": future_status,
            "Payroll": payroll,
        })

    per_year: dict[str, float] = {}
    total = 0.0
    if ctrl_rows:
        ctrl_df = _vd_contract_value(pd.DataFrame(ctrl_rows))
        for _, cyr in ctrl_df.iterrows():
            sal = cyr.get("contract_value", 0)
            if pd.notna(sal):
                per_year[f"salary_{int(cyr['Year'])}"] = round(sal)
                total += sal
    return total, per_year


# ═══════════════════════════════════════════════════════════════════════════════
# Projected stat columns to carry forward
# ═══════════════════════════════════════════════════════════════════════════════

BATTER_PROJ_COLS  = ["BB%", "K%", "AVG", "OBP", "SLG", "wOBA", "HR", "2B", "3B", "RBI", "R", "HBP"]
PITCHER_PROJ_COLS = ["Role", "K%", "BB%", "FIP", "ERA"]


# ═══════════════════════════════════════════════════════════════════════════════
# Core surplus computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_surplus_for_snapshot(
    snapshot_year: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute per-player surplus value for a single snapshot year.

    snapshot_year S  →  cutoff_year = S − 1
    """
    cutoff_year = snapshot_year - Config.SNAPSHOT_LAG
    out_path = SURPLUS_DIR / f"surplus_{snapshot_year}.csv"

    if out_path.exists() and not force:
        logger.info(f"[{snapshot_year}]  surplus file already exists — skipping")
        return pd.read_csv(out_path)

    SURPLUS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Cot's salary ────────────────────────────────────────────────
    cots_file = COTS_BY_YEAR_DIR / f"{snapshot_year}.csv"
    if not cots_file.exists():
        logger.warning(f"[{snapshot_year}]  Cot's salary file missing: {cots_file}")
        return pd.DataFrame()

    cots = pd.read_csv(cots_file)
    cots = cots.dropna(subset=["player"]).copy()
    cots["_name_key"] = cots["player"].apply(_name_key_fn)
    cots = cots.sort_values("salary", ascending=False, na_position="last")
    logger.info(f"[{snapshot_year}]  loaded {len(cots)} Cot's rows")

    # ── Load projections ─────────────────────────────────────────────────
    proj_dir = PROJECTIONS_DIR / f"cutoff_{cutoff_year}"
    if not proj_dir.exists():
        logger.warning(f"[{snapshot_year}]  projections dir missing: {proj_dir}")
        return pd.DataFrame()

    pred_files = {
        "batter":      proj_dir / "batter_predictions.csv",
        "pitcher":     proj_dir / "pitcher_predictions.csv",
        "fielding":    proj_dir / "fielding_predictions.csv",
        "baserunning": proj_dir / "baserunning_predictions.csv",
    }
    missing = [k for k, v in pred_files.items() if not v.exists()]
    if missing:
        logger.warning(f"[{snapshot_year}]  missing prediction files: {missing}")
        return pd.DataFrame()

    batter_df      = pd.read_csv(pred_files["batter"])
    pitcher_df     = pd.read_csv(pred_files["pitcher"])
    fielding_df    = pd.read_csv(pred_files["fielding"])
    baserunning_df = pd.read_csv(pred_files["baserunning"])

    logger.info(
        f"[{snapshot_year}]  loaded projections — "
        f"{batter_df['IDfg'].nunique()} batters, {pitcher_df['IDfg'].nunique()} pitchers"
    )

    # ── Compute WAR ──────────────────────────────────────────────────────
    batter_df  = calculate_batter_war(batter_df, fielding_df, baserunning_df)
    pitcher_df = calculate_pitcher_war(pitcher_df)

    # ── Unify into single WAR table ──────────────────────────────────────
    batter_keep  = ["IDfg", "Name", "Year", "Age", "WAR"] + [c for c in BATTER_PROJ_COLS if c in batter_df.columns]
    pitcher_keep = ["IDfg", "Name", "Year", "Age", "WAR"] + [c for c in PITCHER_PROJ_COLS if c in pitcher_df.columns]

    batter_slim  = batter_df[batter_keep].copy();  batter_slim["player_type"]  = "batter"
    pitcher_slim = pitcher_df[pitcher_keep].copy(); pitcher_slim["player_type"] = "pitcher"

    all_proj = pd.concat([batter_slim, pitcher_slim], ignore_index=True)

    # Handle two-way players — sum WAR instead of taking max
    dup_mask = all_proj.duplicated(subset=["IDfg", "Year"], keep=False)
    two_way_ids = set(all_proj.loc[dup_mask, "IDfg"].unique())

    if two_way_ids:
        two_way = all_proj[all_proj["IDfg"].isin(two_way_ids)]
        non_two = all_proj[~all_proj["IDfg"].isin(two_way_ids)]
        summed = []
        for (idfg, yr), grp in two_way.groupby(["IDfg", "Year"]):
            base = grp.sort_values("WAR", ascending=False).iloc[0].copy()
            base["WAR"] = grp["WAR"].sum()
            base["player_type"] = "two_way"
            summed.append(base)
        all_proj = pd.concat([non_two, pd.DataFrame(summed)], ignore_index=True)
    else:
        all_proj = (
            all_proj.sort_values("WAR", ascending=False)
            .drop_duplicates(subset=["IDfg", "Year"], keep="first")
        )

    all_proj["_name_key"] = all_proj["Name"].apply(_name_key_fn)

    # ── Identify unique projected players ────────────────────────────────
    proj_players = (
        all_proj[all_proj["Year"] == snapshot_year]
        .drop_duplicates(subset=["IDfg"])
        [["IDfg", "Name", "_name_key", "Age", "player_type"]]
        .copy()
    )
    extra = (
        all_proj[~all_proj["IDfg"].isin(proj_players["IDfg"])]
        .sort_values("Year")
        .drop_duplicates(subset=["IDfg"])
        [["IDfg", "Name", "_name_key", "Age", "player_type"]]
        .copy()
    )
    proj_players = pd.concat([proj_players, extra], ignore_index=True)

    # ── Match Cot's to projections ───────────────────────────────────────
    cots = _enrich_cots_with_idfg(cots, proj_players)
    cots_id   = cots.dropna(subset=["IDfg"]).drop_duplicates(subset=["IDfg"], keep="first")
    cots_name = cots[cots["IDfg"].isna()].drop_duplicates(subset=["_name_key"], keep="first")
    cots = pd.concat([cots_id, cots_name], ignore_index=True)

    merged = proj_players.merge(
        cots.drop(columns=["_name_key"]),
        on="IDfg", how="left", suffixes=("", "_cots"),
    )
    n_matched = merged["player"].notna().sum()
    logger.info(f"[{snapshot_year}]  {n_matched} matched in Cot's, {merged['player'].isna().sum()} estimated")

    # ── Build surplus rows ───────────────────────────────────────────────
    records = []
    for _, p in merged.iterrows():
        idfg  = p["IDfg"]
        name  = p["Name"]
        ptype = p["player_type"]
        age   = p["Age"]
        in_cots = pd.notna(p.get("player"))

        if in_cots:
            team = p.get("team", "")
            pos  = p.get("position", "")
            svc  = p.get("service_time", np.nan)
            yoc  = p.get("years_of_control", 0)
            cots_total_future = p.get("total_future_salary", 0)
        else:
            team = ""
            pos  = ""
            svc  = _estimate_service_time(idfg, snapshot_year - 1)
            yoc  = _years_of_control_from_svc(svc)
            cots_total_future = 0

        if pd.isna(yoc) or yoc <= 0:
            continue

        status = _classify_status(svc)
        control_years = list(range(snapshot_year, snapshot_year + int(yoc)))
        player_proj = all_proj[all_proj["IDfg"] == idfg]

        total_war = 0.0
        total_war_value = 0.0
        per_year_war = {}
        proj_war_snap = 0.0

        for yr in control_years:
            war_row = player_proj.loc[player_proj["Year"] == yr, "WAR"]
            war = max(0.0, float(war_row.iloc[0])) if len(war_row) else 0.0
            total_war += war
            total_war_value += war_to_dollars(war, yr)
            per_year_war[f"WAR_{yr}"] = round(war, 3)
            if yr == snapshot_year:
                proj_war_snap = war

        # Salary estimation — delegates arb model to value_determination
        estimated_sal, per_year_salary = _estimate_control_salaries(
            idfg, svc, control_years, player_proj,
        )

        if in_cots:
            cots_sal = float(cots_total_future) if pd.notna(cots_total_future) else 0.0
            total_future_sal = max(cots_sal, estimated_sal)
            if cots_sal > estimated_sal > 0:
                scale = cots_sal / estimated_sal
                per_year_salary = {k: round(v * scale) for k, v in per_year_salary.items()}
            elif cots_sal > estimated_sal == 0:
                n_yrs = len(control_years)
                for yr in control_years:
                    per_year_salary[f"salary_{yr}"] = round(cots_sal / n_yrs)
        else:
            total_future_sal = estimated_sal

        surplus = total_war_value - total_future_sal

        # Projected stats for snapshot year
        snap_row = player_proj[player_proj["Year"] == snapshot_year]
        proj_stats = {}
        if not snap_row.empty:
            sr = snap_row.iloc[0]
            stat_cols = BATTER_PROJ_COLS if ptype == "batter" else PITCHER_PROJ_COLS
            for c in stat_cols:
                if c in sr.index and pd.notna(sr[c]):
                    proj_stats[f"proj_{c}"] = sr[c]
            proj_stats["proj_WAR"] = round(float(sr["WAR"]), 2)

        # mlbam_id from crosswalk
        xw = _build_mlbam_crosswalk()
        mlbam_row = xw.loc[xw["IDfg"] == idfg, "mlbam_id"]
        mlbam_id = int(mlbam_row.iloc[0]) if len(mlbam_row) else pd.NA

        record = {
            "Name": name,
            "IDfg": idfg,
            "mlbam_id": mlbam_id,
            "snapshot_year": snapshot_year,
            "Team": team,
            "Age": age,
            "Position": pos,
            "player_type": ptype,
        }
        record.update(proj_stats)
        record.update({
            "service_time": round(svc, 3) if pd.notna(svc) else np.nan,
            "years_of_control": int(yoc),
            "status": status,
            "salary_source": "cots" if in_cots else "estimated",
            "total_future_WAR": round(total_war, 2),
            "total_future_WAR_value": round(total_war_value),
            "total_future_salary": round(total_future_sal),
            "surplus": round(surplus),
        })
        record.update(per_year_war)
        record.update(per_year_salary)
        records.append(record)

    result = pd.DataFrame(records)
    if result.empty:
        logger.warning(f"[{snapshot_year}]  no surplus records produced")
        return result

    result = result.sort_values("surplus", ascending=False).reset_index(drop=True)
    result.to_csv(out_path, index=False)
    logger.info(
        f"[{snapshot_year}]  wrote {len(result)} players to {out_path.name}  "
        f"(median surplus ${result['surplus'].median():,.0f})"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_surpluses(
    start: int | None = None,
    end: int | None = None,
    force: bool = False,
) -> dict[int, pd.DataFrame]:
    """Compute surplus for every snapshot year in [start, end]."""
    start = start or (Config.CUTOFF_START + Config.SNAPSHOT_LAG)
    end   = end   or (Config.CUTOFF_END   + Config.SNAPSHOT_LAG)

    Config.ensure_directories()

    logger.info("=" * 60)
    logger.info("Historical Values — Surplus Calculator")
    logger.info(f"Snapshot years {start} → {end}")
    logger.info("=" * 60)

    results: dict[int, pd.DataFrame] = {}
    for snap_year in range(start, end + 1):
        df = compute_surplus_for_snapshot(snap_year, force=force)
        results[snap_year] = df

    total = sum(len(df) for df in results.values())
    logger.info(f"Surplus computation complete — {total} total player rows across {len(results)} years")
    return results
