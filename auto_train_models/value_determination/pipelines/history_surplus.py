"""
Historical Values — V2 Monte Carlo Surplus Snapshot Builder (`2014–2026`)
=========================================================================

Computes annual opening-day (`surplus_YYYY.csv`) and end-of-season
(`surplus_YYYY_eoy.csv`) trade-value snapshots using the exact same V2
valuation spine as the live pipeline:

  1. Enforce MLB debut boundary (`>= 1` MLB game as of `cutoff_year`) so zero
     pre-debut minor-leaguers leak from cached `projections/cutoff_*` files.
  2. Estimate individualized playing time (`estimate_playing_time` &
     `estimate_batter_playing_time`) and compute WAR via `war_engine`
     (including post-28 defensive spectrum drift).
  3. Build Cot's + Spotrac CBA salary/option timelines.
  4. Run the vectorized 1,000-path Monte Carlo simulator (`monte_carlo_valuation`)
     emitting `trade_value`, `trade_value_p10`, `trade_value_p50`, `trade_value_p90`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

_AUTO_TRAIN = Path(__file__).resolve().parents[2]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from core.id_crosswalk import fg_to_mlbam, normalize_team_code
from core.name_utils import name_key as _name_key_fn
from core.position_profiles import (
    build_position_profiles,
    load_batting_for_games,
    load_fielding_history,
)
from value_determination.config import Config, logger
from value_determination.contract_timeline import extend_fa_timeline
from value_determination.dollar_valuation import (
    calculate_annual_surplus,
    calculate_salary_cost,
    join_predictions_with_timeline,
)
from value_determination.monte_carlo_valuation import (
    add_control_horizon_metrics,
    analyze_contract_options,
    calculate_trade_values,
)
from value_determination.pipelines.cots_adapter import (
    _years_of_control_from_svc,
    build_salary_timeline,
)
from value_determination.playing_time import (
    estimate_batter_playing_time,
    estimate_playing_time,
)
from value_determination.prospect_valuation import _load_first_mlb_season_by_fg
from value_determination.salary_cleaner import clean_salary_data
from value_determination.war_engine import (
    calculate_batter_war_dataframe,
    calculate_pitcher_war_dataframe,
)

PROJECTIONS_DIR = Config.Paths.PROJECTIONS_DIR
SURPLUS_DIR = Config.Paths.SURPLUS_DIR
COTS_BY_YEAR_DIR = Config.Paths.COTS_BY_YEAR_DIR
DATA_DIR = Config.Paths.DATA_DIR
HISTORIC_BATTING_FILE_CLASSIC = Config.Paths.HISTORIC_BATTING_FILE_CLASSIC
HISTORIC_PITCHING_FILE_CLASSIC = Config.Paths.HISTORIC_PITCHING_FILE_CLASSIC

_hist_app_cache: Optional[pd.DataFrame] = None
_first_mlb_cache: Optional[Dict[int, int]] = None
_spotrac_clean_cache: Optional[pd.DataFrame] = None
_spotrac_txn_cache: Optional[pd.DataFrame] = None
_hist_fielding_cache: Optional[pd.DataFrame] = None
_hist_batting_cache: Optional[pd.DataFrame] = None


def _get_hist_fielding_and_batting() -> tuple[pd.DataFrame, pd.DataFrame]:
    global _hist_fielding_cache, _hist_batting_cache
    if _hist_fielding_cache is None:
        _hist_fielding_cache = load_fielding_history()
    if _hist_batting_cache is None:
        _hist_batting_cache = load_batting_for_games()
    return _hist_fielding_cache, _hist_batting_cache


def _get_first_mlb_seasons() -> Dict[int, int]:
    global _first_mlb_cache
    if _first_mlb_cache is None:
        _first_mlb_cache = _load_first_mlb_season_by_fg()
    return _first_mlb_cache


def _load_historical_appearances() -> pd.DataFrame:
    global _hist_app_cache
    if _hist_app_cache is not None:
        return _hist_app_cache

    frames = []
    for path in (HISTORIC_BATTING_FILE_CLASSIC, HISTORIC_PITCHING_FILE_CLASSIC):
        if path.exists():
            df = pd.read_csv(path, usecols=["IDfg", "Season", "Team"], low_memory=False)
            df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")
            df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
            df = df.dropna(subset=["IDfg", "Season"]).copy()
            df["IDfg"] = df["IDfg"].astype(int)
            df["Season"] = df["Season"].astype(int)
            df["Team"] = df["Team"].map(normalize_team_code)
            frames.append(df.drop_duplicates(subset=["IDfg", "Season"]))
    _hist_app_cache = pd.concat(frames, ignore_index=True).drop_duplicates(["IDfg", "Season"]) if frames else pd.DataFrame()
    return _hist_app_cache


def _estimate_service_and_team(idfg: int, as_of_season: int) -> tuple[float, str]:
    app = _load_historical_appearances()
    if app.empty:
        return 0.0, "MLB"
    sub = app[(app["IDfg"] == idfg) & (app["Season"] <= as_of_season)].sort_values("Season")
    if sub.empty:
        return 0.0, "MLB"
    seasons = int(sub["Season"].nunique())
    last_team = str(sub.iloc[-1]["Team"])
    if not last_team or last_team == "FA":
        last_team = "MLB"
    return float(max(0, seasons - 1)), last_team


def _enrich_cots_with_idfg(
    cots: pd.DataFrame,
    proj_players: pd.DataFrame,
    as_of_season: int,
) -> pd.DataFrame:
    """Match Cot's player names to projected `IDfg` with historical team disambiguation."""
    name_to_idfgs: Dict[str, list[int]] = {}
    for _, row in proj_players.iterrows():
        name_to_idfgs.setdefault(row["_name_key"], []).append(int(row["IDfg"]))

    app = _load_historical_appearances()
    recent_teams = (
        app[app["Season"].between(as_of_season - 1, as_of_season + 1)]
        .groupby("IDfg")["Team"]
        .apply(set)
        .to_dict()
        if not app.empty
        else {}
    )

    idfg_col = []
    for _, row in cots.iterrows():
        candidates = name_to_idfgs.get(row["_name_key"], [])
        if len(candidates) == 1:
            idfg_col.append(candidates[0])
        elif len(candidates) > 1:
            cots_team = normalize_team_code(row.get("team", ""))
            matched = next((c for c in candidates if cots_team in recent_teams.get(c, set())), None)
            idfg_col.append(matched if matched is not None else candidates[0])
        else:
            idfg_col.append(pd.NA)

    out = cots.copy()
    out["IDfg"] = pd.array(idfg_col, dtype="Int64")
    return out


def _get_cleaned_spotrac_lookup() -> pd.DataFrame:
    """Load CBA-cleaned Spotrac salaries & option statuses keyed by `(_name_key, Year)`."""
    global _spotrac_clean_cache
    if _spotrac_clean_cache is not None:
        return _spotrac_clean_cache

    spotrac_path = Config.Paths.SALARY_FILE
    if not spotrac_path.exists():
        _spotrac_clean_cache = pd.DataFrame()
        return _spotrac_clean_cache

    raw = pd.read_csv(spotrac_path, low_memory=False)
    cleaned = clean_salary_data(raw)
    cleaned["_name_key"] = cleaned["Player Name"].map(_name_key_fn)

    def _norm_status(s) -> str:
        txt = "" if pd.isna(s) else str(s).upper().strip()
        if "OPT-OUT" in txt or "OPT OUT" in txt:
            return "Opt-Out"
        if "CLUB" in txt:
            return "Team Option"
        if "MUTUAL" in txt:
            return "Mutual Option"
        if "VESTING" in txt:
            return "Vesting Option"
        if "PLAYER" in txt:
            return "Player Option"
        if any(k in txt for k in ("UFA", "RFA", "FA", "FREE AGENT")):
            return "Free Agent"
        return "Signed"

    cleaned["_norm_status"] = cleaned["Status"].map(_norm_status)
    _spotrac_clean_cache = cleaned.drop_duplicates(subset=["_name_key", "Year"], keep="first")
    return _spotrac_clean_cache


def _load_spotrac_transactions() -> pd.DataFrame:
    global _spotrac_txn_cache
    if _spotrac_txn_cache is not None:
        return _spotrac_txn_cache
    if not Config.Paths.SPOTRAC_TRANSACTIONS_FILE.exists():
        _spotrac_txn_cache = pd.DataFrame(columns=["_name_key", "date", "transaction_type", "years", "total_value"])
        return _spotrac_txn_cache

    df = pd.read_csv(Config.Paths.SPOTRAC_TRANSACTIONS_FILE, parse_dates=["date"], low_memory=False)
    if df.empty:
        _spotrac_txn_cache = pd.DataFrame(columns=["_name_key", "date", "transaction_type", "years", "total_value"])
        return _spotrac_txn_cache

    df = df.copy()
    df["_name_key"] = df["player_name"].map(_name_key_fn)
    df["transaction_type"] = df["transaction_type"].astype(str).str.lower().str.strip()
    df["years"] = pd.to_numeric(df.get("years"), errors="coerce")
    df["total_value"] = pd.to_numeric(df.get("total_value"), errors="coerce")
    _spotrac_txn_cache = df[["_name_key", "date", "transaction_type", "years", "total_value"]].copy()
    return _spotrac_txn_cache


def _apply_eoy_extension_overrides(cots: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:
    txns = _load_spotrac_transactions()
    if txns.empty:
        return cots

    season_end = pd.Timestamp(snapshot_year - 1, 10, 1)
    ext = txns[
        (txns["transaction_type"] == "extension")
        & txns["date"].notna()
        & (txns["date"] <= season_end)
    ].dropna(subset=["_name_key", "years"])
    if ext.empty:
        return cots

    ext = (
        ext.sort_values(["_name_key", "date"])
        .groupby("_name_key", as_index=False)
        .agg({"years": "max", "total_value": "max"})
        .rename(columns={"years": "_ext_yrs", "total_value": "_ext_val"})
    )

    out = cots.copy()
    out["_name_key"] = out["player"].map(_name_key_fn)
    out = out.merge(ext, on="_name_key", how="left")
    mask = out["_ext_yrs"].notna() & (out["_ext_yrs"] > out["years_of_control"])
    if mask.any():
        out.loc[mask, "years_of_control"] = out.loc[mask, "_ext_yrs"]
        ext_val = out.loc[mask, "_ext_val"].fillna(out.loc[mask, "total_future_salary"])
        out.loc[mask, "total_future_salary"] = np.maximum(
            pd.to_numeric(ext_val, errors="coerce").fillna(0.0),
            pd.to_numeric(out.loc[mask, "total_future_salary"], errors="coerce").fillna(0.0),
        )
        out.loc[mask, "Status"] = "Signed"
        out.loc[mask, "Normalized_Status"] = "Signed"
    return out.drop(columns=["_ext_yrs", "_ext_val"], errors="ignore")


def _overlay_spotrac_salaries_and_options(
    salary_timeline: pd.DataFrame,
    snapshot_year: int,
) -> pd.DataFrame:
    """
    Overlay CBA-accurate Spotrac annual salaries & option statuses where a
    Spotrac contract was already active as of `snapshot_year`.
    """
    sp = _get_cleaned_spotrac_lookup()
    if sp.empty:
        return salary_timeline

    st = salary_timeline.copy()
    st["_name_key"] = st["Name"].map(_name_key_fn)

    # Identify Spotrac contracts active by snapshot_year
    sp_non_fa = sp[sp["_norm_status"] != "Free Agent"]
    sp_bounds = sp_non_fa.groupby("_name_key")["Year"].agg(["min", "max"]).reset_index()
    active_keys = set(sp_bounds.loc[sp_bounds["min"] <= snapshot_year, "_name_key"])

    sp_active = sp[sp["_name_key"].isin(active_keys)][["_name_key", "Year", "Payroll", "_norm_status"]]
    merged = st.merge(sp_active, on=["_name_key", "Year"], how="left", suffixes=("", "_sp"))

    # Override Signed / Option rows where Spotrac has explicit salary or option clause
    is_not_fa = merged["Normalized_Status"] != "Free Agent"
    sal_override = is_not_fa & merged["Payroll_sp"].notna() & (
        merged["Normalized_Status"].eq("Signed")
        | merged["_norm_status"].isin(["Opt-Out", "Team Option", "Player Option", "Mutual Option", "Vesting Option"])
    )
    if sal_override.any():
        merged.loc[sal_override, "Payroll"] = merged.loc[sal_override, "Payroll_sp"]

    opt_override = is_not_fa & merged["_norm_status"].isin(
        ["Opt-Out", "Team Option", "Player Option", "Mutual Option", "Vesting Option"]
    )
    if opt_override.any():
        merged.loc[opt_override, "Status"] = merged.loc[opt_override, "_norm_status"]
        merged.loc[opt_override, "Normalized_Status"] = merged.loc[opt_override, "_norm_status"]

    return merged.drop(columns=["_name_key", "Payroll_sp", "_norm_status"], errors="ignore")


def compute_surplus_for_snapshot(
    snapshot_year: int,
    *,
    is_eoy: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute per-player V2 Monte Carlo trade value snapshot for `snapshot_year`.
    """
    cutoff_year = snapshot_year - Config.History.SNAPSHOT_LAG
    suffix = "_eoy" if is_eoy else ""
    out_path = SURPLUS_DIR / f"surplus_{snapshot_year}{suffix}.csv"

    if out_path.exists() and not force:
        logger.info("[%d%s] surplus file exists — skipping", snapshot_year, suffix)
        return pd.read_csv(out_path, low_memory=False)

    SURPLUS_DIR.mkdir(parents=True, exist_ok=True)

    cots_year = snapshot_year - 1 if is_eoy else snapshot_year
    cots_file = COTS_BY_YEAR_DIR / f"{cots_year}.csv"
    if not cots_file.exists():
        logger.warning("[%d] Cot's file missing: %s", snapshot_year, cots_file)
        return pd.DataFrame()

    cots = pd.read_csv(cots_file, low_memory=False)
    cots = cots.dropna(subset=["player"]).copy()
    cots = cots[~cots["player"].astype(str).str.contains("Running|Payroll", na=False)]

    if is_eoy:
        cots["service_time"] = pd.to_numeric(cots["service_time"], errors="coerce").fillna(0.0) + 1.0
        cots["years_of_control"] = (pd.to_numeric(cots["years_of_control"], errors="coerce").fillna(0.0) - 1.0).clip(lower=0)
        cots["total_future_salary"] = (
            pd.to_numeric(cots["total_future_salary"], errors="coerce").fillna(0.0)
            - pd.to_numeric(cots["salary"], errors="coerce").fillna(0.0)
        ).clip(lower=0)
        cots = cots[cots["years_of_control"] > 0].copy()
        cots = _apply_eoy_extension_overrides(cots, snapshot_year)

    cots["_name_key"] = cots["player"].map(_name_key_fn)
    cots = cots.sort_values("salary", ascending=False, na_position="last")

    proj_dir = PROJECTIONS_DIR / f"cutoff_{cutoff_year}"
    pred_files = {
        "batter": proj_dir / "batter_predictions.csv",
        "pitcher": proj_dir / "pitcher_predictions.csv",
        "fielding": proj_dir / "fielding_predictions.csv",
        "baserunning": proj_dir / "baserunning_predictions.csv",
    }
    if not proj_dir.exists() or any(not p.exists() for p in pred_files.values()):
        logger.warning("[%d] projections missing at %s", snapshot_year, proj_dir)
        return pd.DataFrame()

    batter_df = pd.read_csv(pred_files["batter"], low_memory=False)
    pitcher_df = pd.read_csv(pred_files["pitcher"], low_memory=False)
    fielding_df = pd.read_csv(pred_files["fielding"], low_memory=False)
    baserunning_df = pd.read_csv(pred_files["baserunning"], low_memory=False)

    # ── Enforce Hard MLB Debut Boundary (`>= 1` MLB game as of `cutoff_year`) ─
    first_mlb = _get_first_mlb_seasons()
    debuted_ids = {fg for fg, yr in first_mlb.items() if yr <= cutoff_year}
    batter_df = batter_df[batter_df["IDfg"].isin(debuted_ids) & (batter_df["Year"] >= snapshot_year)].copy()
    pitcher_df = pitcher_df[pitcher_df["IDfg"].isin(debuted_ids) & (pitcher_df["Year"] >= snapshot_year)].copy()

    if batter_df.empty and pitcher_df.empty:
        return pd.DataFrame()

    # ── Playing Time + WAR Engine (with post-28 spectrum drift) ──────────────
    hist_fielding, hist_batting = _get_hist_fielding_and_batting()
    batter_ids = batter_df["IDfg"].unique().tolist()
    position_profiles = build_position_profiles(
        hist_fielding, hist_batting, batter_ids, cutoff_year=cutoff_year
    )

    if not batter_df.empty:
        batter_df = estimate_batter_playing_time(
            batter_df, anchor_year=snapshot_year, position_profiles=position_profiles
        )
        batter_df = calculate_batter_war_dataframe(
            batter_df, fielding_df, baserunning_df, position_profiles=position_profiles, include_components=True
        )

    if not pitcher_df.empty:
        pit_chunks = []
        for pyr in sorted(pitcher_df["Year"].unique()):
            chunk = pitcher_df[pitcher_df["Year"] == pyr].copy()
            pit_chunks.append(estimate_playing_time(chunk, int(pyr), anchor_year=snapshot_year))
        pitcher_df = pd.concat(pit_chunks, ignore_index=True)
        pitcher_df = calculate_pitcher_war_dataframe(pitcher_df)

    # ── Unify projections & reconcile two-way players ────────────────────────
    bat_cols = [c for c in ("IDfg", "Name", "Year", "Age", "WAR", "PA", "Bat", "BsR", "Def", "Position") if c in batter_df.columns]
    batter_slim = batter_df[bat_cols].copy()
    batter_slim["player_type"] = "batter"
    batter_slim["player_role"] = "Batter"

    pit_cols = [c for c in ("IDfg", "Name", "Year", "Age", "WAR", "IP") if c in pitcher_df.columns]
    pitcher_slim = pitcher_df[pit_cols].copy()
    pitcher_slim["player_type"] = "pitcher"
    pitcher_slim["player_role"] = (
        pitcher_df["Role"].map(lambda r: "SP" if str(r).upper() == "SP" else "RP")
        if "Role" in pitcher_df.columns
        else "SP"
    )

    all_proj = pd.concat([batter_slim, pitcher_slim], ignore_index=True)
    dup_mask = all_proj.duplicated(subset=["IDfg", "Year"], keep=False)
    if dup_mask.any():
        two_way_ids = set(all_proj.loc[dup_mask, "IDfg"].unique())
        two_way = all_proj[all_proj["IDfg"].isin(two_way_ids)]
        non_two = all_proj[~all_proj["IDfg"].isin(two_way_ids)]
        summed = []
        for _, grp in two_way.groupby(["IDfg", "Year"]):
            bat_r = grp[grp["player_type"] == "batter"]
            pit_r = grp[grp["player_type"] == "pitcher"]
            bw = float(bat_r["WAR"].iloc[0]) if not bat_r.empty else 0.0
            pw = float(pit_r["WAR"].iloc[0]) if not pit_r.empty else 0.0
            if bw > 0 and pw > 0:
                base = grp.sort_values("WAR", ascending=False).iloc[0].copy()
                base["WAR"] = bw + pw
                base["player_type"] = "two_way"
            elif pw >= bw and not pit_r.empty:
                base = pit_r.iloc[0].copy()
            else:
                base = bat_r.iloc[0].copy() if not bat_r.empty else grp.iloc[0].copy()
            summed.append(base)
        all_proj = pd.concat([non_two, pd.DataFrame(summed)], ignore_index=True)

    all_proj["_name_key"] = all_proj["Name"].map(_name_key_fn)
    all_proj["prediction_year"] = all_proj["Year"]

    proj_players = (
        all_proj.sort_values("Year")
        .drop_duplicates(subset=["IDfg"])[["IDfg", "Name", "_name_key", "Age", "player_type", "player_role"]]
        .copy()
    )

    cots = _enrich_cots_with_idfg(cots, proj_players, as_of_season=cutoff_year)
    cots_id = cots.dropna(subset=["IDfg"]).drop_duplicates(subset=["IDfg"], keep="first")
    cots_matched = cots_id.merge(
        proj_players[["IDfg", "Name", "player_type", "player_role"]],
        on="IDfg",
        how="left",
        suffixes=("", "_proj"),
    )
    if "Name_proj" in cots_matched.columns:
        cots_matched["Name"] = cots_matched["Name_proj"].fillna(cots_matched.get("player", ""))
        cots_matched = cots_matched.drop(columns=["Name_proj"])

    matched_idfgs = set(cots_matched["IDfg"].astype(int))
    unmatched_idfgs = set(proj_players["IDfg"].astype(int)) - matched_idfgs

    est_rows = []
    for idfg in unmatched_idfgs:
        pp = proj_players[proj_players["IDfg"] == idfg]
        if pp.empty:
            continue
        pp_row = pp.iloc[0]
        svc, hist_team = _estimate_service_and_team(idfg, cutoff_year)
        yoc = _years_of_control_from_svc(svc)
        if yoc <= 0:
            continue
        est_rows.append({
            "IDfg": int(idfg),
            "Name": pp_row["Name"],
            "Team": hist_team,
            "player_type": pp_row["player_type"],
            "player_role": pp_row["player_role"],
            "service_time": svc,
            "years_of_control": yoc,
        })
    estimated_df = pd.DataFrame(est_rows) if est_rows else pd.DataFrame(
        columns=["IDfg", "Name", "Team", "player_type", "player_role", "service_time", "years_of_control"]
    )

    salary_timeline = build_salary_timeline(cots_matched, estimated_df, snapshot_year)
    if salary_timeline.empty:
        return pd.DataFrame()

    salary_timeline = _overlay_spotrac_salaries_and_options(salary_timeline, cots_year)
    salary_timeline["mlbam_id"] = salary_timeline["IDfg"].map(lambda x: fg_to_mlbam(int(x)))

    # ── Shared V2 Valuation Spine ────────────────────────────────────────────
    extended = extend_fa_timeline(salary_timeline)
    with_war = join_predictions_with_timeline(extended, all_proj)
    with_cost = calculate_salary_cost(with_war)
    with_surplus = calculate_annual_surplus(with_cost)
    with_options = analyze_contract_options(with_surplus, current_year=snapshot_year)
    with_mc = calculate_trade_values(
        with_options,
        current_year=snapshot_year,
        apply_ros_fraction=False,
    )
    result = add_control_horizon_metrics(
        with_mc,
        current_year=snapshot_year,
        apply_ros_fraction=False,
    )

    summary = _extract_player_summary(result, all_proj, snapshot_year)
    if summary.empty:
        return summary

    summary = summary.sort_values("trade_value", ascending=False, na_position="last").reset_index(drop=True)
    summary.to_csv(out_path, index=False)
    logger.info(
        "[%d%s] wrote %d MLB players to %s (max=$%.1fM, median=$%.1fM)",
        snapshot_year,
        suffix,
        len(summary),
        out_path.name,
        (summary["trade_value"].max() / 1e6) if summary["trade_value"].notna().any() else 0.0,
        (summary["trade_value"].median() / 1e6) if summary["trade_value"].notna().any() else 0.0,
    )
    return summary


def _extract_player_summary(
    pipeline_result: pd.DataFrame,
    all_proj: pd.DataFrame,
    snapshot_year: int,
) -> pd.DataFrame:
    """Extract one canonical summary row per player for `surplus_YYYY.csv`."""
    if pipeline_result.empty:
        return pd.DataFrame()

    records = []
    for idfg, pdata in pipeline_result.groupby("IDfg", sort=False):
        pdata = pdata.sort_values("Year")
        first = pdata.iloc[0]
        mlbam = fg_to_mlbam(int(idfg))

        snap = pdata[pdata["Year"] == snapshot_year]
        age = float(snap["Age"].iloc[0]) if not snap.empty and pd.notna(snap["Age"].iloc[0]) else np.nan

        pp = all_proj[all_proj["IDfg"] == idfg]
        player_type = pp["player_type"].iloc[0] if not pp.empty else "batter"
        role = str(first.get("player_role", "Batter"))

        tv = first.get("trade_value", np.nan)
        p10 = first.get("trade_value_p10", np.nan)
        p50 = first.get("trade_value_p50", np.nan)
        p90 = first.get("trade_value_p90", np.nan)
        yrs_ctrl = float(first.get("years_control", 0) or 0)
        ctrl_war = float(first.get("control_war", 0) or 0)
        ctrl_prod = float(first.get("control_production_value", 0) or 0)
        ctrl_sal = float(first.get("control_salary_cost", 0) or 0)
        ctrl_surp = float(first.get("control_surplus", 0) or 0)
        avg_war = float(first.get("avg_war", 0) or 0)

        svc = first.get("Years_of_Service", np.nan)
        status = first.get("Status", first.get("Normalized_Status", ""))
        salary_source = "cots" if pd.notna(first.get("Payroll")) else "estimated"
        war_per_year = ctrl_war / yrs_ctrl if yrs_ctrl > 0 else 0.0

        per_year = {}
        for _, row in pdata.iterrows():
            yr = int(row["Year"])
            w = row.get("WAR", np.nan)
            s = row.get("salary_cost", np.nan)
            if pd.notna(w):
                per_year[f"WAR_{yr}"] = round(float(w), 3)
            if pd.notna(s):
                per_year[f"salary_{yr}"] = round(float(s))

        rec = {
            "fg_id": int(idfg),
            "mlbam_id": int(mlbam) if mlbam is not None else pd.NA,
            "player_name": first.get("Name", ""),
            "snapshot_year": snapshot_year,
            "team": normalize_team_code(first.get("Team", "")),
            "age": age,
            "player_type": player_type,
            "player_role": role,
            "service_time": round(float(svc), 3) if pd.notna(svc) else np.nan,
            "years_control": round(yrs_ctrl, 2),
            "status": status,
            "salary_source": salary_source,
            "trade_value": round(float(tv)) if pd.notna(tv) else np.nan,
            "trade_value_p10": round(float(p10)) if pd.notna(p10) else np.nan,
            "trade_value_p50": round(float(p50)) if pd.notna(p50) else np.nan,
            "trade_value_p90": round(float(p90)) if pd.notna(p90) else np.nan,
            "control_war": round(ctrl_war, 2),
            "control_production_value": round(ctrl_prod),
            "control_salary_cost": round(ctrl_sal),
            "control_surplus": round(ctrl_surp),
            "avg_war": round(avg_war, 2),
            "war_per_year": round(war_per_year, 2),
        }
        rec.update(per_year)
        records.append(rec)

    return pd.DataFrame(records)


def compute_all_surpluses(
    start: Optional[int] = None,
    end: Optional[int] = None,
    force: bool = False,
    include_eoy: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute opening-day and EOY V2 surplus snapshots across `[start, end]`."""
    start = start or (Config.History.CUTOFF_START + Config.History.SNAPSHOT_LAG)
    end = end or (Config.History.CUTOFF_END + Config.History.SNAPSHOT_LAG)
    Config.Paths.ensure_directories()

    results: dict[str, pd.DataFrame] = {}
    for snap_year in range(start, end + 1):
        results[str(snap_year)] = compute_surplus_for_snapshot(snap_year, force=force)
        if include_eoy:
            try:
                df_eoy = compute_surplus_for_snapshot(snap_year, is_eoy=True, force=force)
                if not df_eoy.empty:
                    results[f"{snap_year}_eoy"] = df_eoy
            except Exception as exc:
                logger.warning("Could not compute EOY surplus for %d: %s", snap_year, exc)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Historical V2 Monte Carlo Surplus Snapshots")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--force", "-f", action="store_true")
    a = p.parse_args()
    compute_all_surpluses(start=a.start, end=a.end, force=a.force)
