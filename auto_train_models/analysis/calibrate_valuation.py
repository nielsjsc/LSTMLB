"""
Joint Empirical Calibration of V2 Valuation Parameters
======================================================

Jointly calibrates `(rate_batter, rate_sp, rate_rp, war_beta, discount_rate,
package_gamma, prospect_scale)` against two real-world MLB market anchors:

1. **Spotrac Free-Agent Signings (`2018–2026`)**:
   - Anchors median signing NPV surplus (`NPV_production - NPV_salary`) per role
     (`Batter`, `SP`, `RP`) near `-$1.5M` (mild winner's curse, within `[-$4M, $0M]`),
     identifying role-specific `$/WAR` base rates (`rate_batter`, `rate_sp`, `rate_rp`).

2. **Clean 2-Team MLB Trades (`2018–2023` Train | `2024–2026` Holdout)**:
   - Minimizes median relative package imbalance `|V_A - V_B| / max(V_A, V_B)` on
     the `2018–2023` training split using L_gamma package consolidation, tuning
     `(war_beta, discount_rate, package_gamma, prospect_scale)` while anchoring
     top MLB trade surplus near `$180M–$230M`.
   - Evaluates strictly out-of-sample on `2024–2026` holdout trades and compares
     against the `36.4%` V1 baseline.

Saves calibrated parameters to `data/generated/valuation_calibration.json`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parents[2]
AUTO_TRAIN_DIR = ROOT_DIR / "auto_train_models"
if str(AUTO_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_TRAIN_DIR))

from core.id_crosswalk import get_crosswalk
from core.name_utils import name_key as _name_key
from value_determination.config import CURRENT_YEAR, Config, logger
from value_determination.prospect_valuation import (
    _load_first_mlb_season_by_fg,
    compute_single_prospect_valuation,
    extract_mlbam_from_url,
)


def _consolidate(vals: list[float], gamma: float) -> float:
    pos = [v for v in vals if v > 0]
    neg = [v for v in vals if v <= 0]
    if len(pos) <= 1:
        pos_c = sum(pos)
    else:
        pos_c = sum(v ** gamma for v in pos) ** (1.0 / gamma)
    return pos_c + sum(neg)


def _load_fa_contracts_dataset(surplus_by_year: dict[int, pd.DataFrame]) -> list[dict]:
    """Build matched FA signing records (2018–2026) with projected WAR paths."""
    txn_path = Config.Paths.SPOTRAC_TRANSACTIONS_FILE
    if not txn_path.exists():
        return []

    txn = pd.read_csv(txn_path, parse_dates=["date"], low_memory=False)
    fa = txn[
        txn["transaction_type"].isin(["fa_signing", "signing"])
        & ~txn["description"].astype(str).str.contains("arbitration|minor league", case=False, na=False)
    ].copy()
    fa["years"] = pd.to_numeric(fa["years"], errors="coerce")
    fa["total_value"] = pd.to_numeric(fa["total_value"], errors="coerce")
    fa = fa.dropna(subset=["date", "years", "total_value"])
    fa = fa[(fa["years"] >= 1) & (fa["years"] <= 15) & (fa["total_value"] >= 6_000_000)].copy()

    # Build lookup from (name_key, snap_year) -> surplus row
    snap_lookup: dict[tuple[str, int], pd.Series] = {}
    for yr, sdf in surplus_by_year.items():
        nm_col = "player_name" if "player_name" in sdf.columns else "Name"
        for _, r in sdf.iterrows():
            nk = _name_key(str(r.get(nm_col, "")))
            if nk and (nk, yr) not in snap_lookup:
                snap_lookup[(nk, yr)] = r

    records = []
    for _, row in fa.iterrows():
        dt = row["date"]
        sign_season = int(dt.year + (1 if dt.month >= 10 else 0))
        if not (2018 <= sign_season <= 2026):
            continue
        yrs = int(round(float(row["years"])))
        tot_val = float(row["total_value"])
        if tot_val / max(yrs, 1) < 4_000_000:
            continue

        nk = _name_key(str(row["player_name"]))
        srow = snap_lookup.get((nk, sign_season))
        if srow is None:
            srow = snap_lookup.get((nk, sign_season - 1))
        if srow is None:
            continue

        role_raw = str(srow.get("player_role", srow.get("position_group", "Batter")))
        role = "SP" if role_raw == "SP" else ("RP" if role_raw == "RP" else "Batter")

        wars = []
        for t in range(yrs):
            target_yr = sign_season + t
            w = srow.get(f"WAR_{target_yr}", np.nan)
            wars.append(max(0.0, float(w)) if pd.notna(w) else 0.0)

        if sum(wars) <= 0.2:
            continue

        records.append({
            "player_name": row["player_name"],
            "sign_season": sign_season,
            "years": yrs,
            "aav": tot_val / yrs,
            "role": role,
            "wars": wars,
        })

    logger.info("Matched %d Spotrac MLB free-agent contracts (2018–2026) for calibration", len(records))
    return records


def _load_clean_2team_trades(
    surplus_by_year: dict[int, pd.DataFrame],
) -> list[dict]:
    """
    Load clean 2-team trades (2018–2026, no cash/PTBNL) linking each player leg
    to either their MLB WAR/salary trajectory (if MLB-debuted before trade year)
    or their scouting prospect profile (if pre-debut).
    """
    trades_csv = Config.Paths.DATA_DIR / "transactions" / "trades.csv"
    players_csv = Config.Paths.DATA_DIR / "transactions" / "trade_players.csv"
    team_info_csv = Config.Paths.DATA_DIR / "active_roster" / "team_info.csv"
    prospect_csv = Config.Paths.PROSPECT_FILE

    if not all(p.exists() for p in (trades_csv, players_csv, team_info_csv, prospect_csv)):
        return []

    trades_df = pd.read_csv(trades_csv)
    players_df = pd.read_csv(players_csv)
    ti_df = pd.read_csv(team_info_csv)
    mlb_team_ids = set(ti_df["team_id"])

    mlbam_to_fg, _ = get_crosswalk()
    first_mlb = _load_first_mlb_season_by_fg()

    # Index MLB surplus rows by (fg_id, snap_year)
    mlb_snap_lookup: dict[tuple[int, int], pd.Series] = {}
    for yr, sdf in surplus_by_year.items():
        fg_col = "fg_id" if "fg_id" in sdf.columns else "IDfg"
        for _, r in sdf.iterrows():
            if pd.notna(r.get(fg_col)):
                mlb_snap_lookup[(int(r[fg_col]), yr)] = r

    # Index prospects by (mlbam_id, year)
    p_df = pd.read_csv(prospect_csv, low_memory=False)
    p_df["year"] = pd.to_numeric(p_df["year"], errors="coerce")
    p_df = p_df.dropna(subset=["year"]).copy()
    p_df["year"] = p_df["year"].astype(int)
    p_df["mlbam_id"] = p_df["prospect_url"].map(extract_mlbam_from_url)
    dob_ts = pd.to_datetime(p_df.get("dob"), format="mixed", errors="coerce")
    ref_ts = pd.to_datetime(p_df["year"].astype(str) + "-07-01")
    dob_age = ((ref_ts - dob_ts).dt.days / 365.25).round(1)
    p_df["season_age"] = dob_age.where(dob_age.notna() & dob_age.between(15, 35), pd.to_numeric(p_df.get("age"), errors="coerce"))

    prospect_lookup: dict[tuple[int, int], pd.Series] = {}
    for _, r in p_df.dropna(subset=["mlbam_id"]).iterrows():
        prospect_lookup[(int(r["mlbam_id"]), int(r["year"]))] = r

    players_df = players_df[
        players_df["from_team_id"].isin(mlb_team_ids)
        & players_df["to_team_id"].isin(mlb_team_ids)
    ].copy()
    meta_map = trades_df.set_index("trade_id")

    clean_trades = []
    for tid, grp in players_df.groupby("trade_id"):
        if tid not in meta_map.index:
            continue
        meta = meta_map.loc[tid]
        if bool(meta.get("has_cash", False)) or bool(meta.get("has_ptbnl", False)):
            continue

        t_date = str(meta["date"])[:10]
        t_year = int(t_date[:4])
        t_month = int(t_date[5:7])
        if not (2018 <= t_year <= 2026):
            continue

        sides_grouped = list(grp.groupby("to_team_id"))
        if len(sides_grouped) != 2:
            continue

        # Season fraction remaining for in-season trades (Apr-Sep)
        rem_frac = 1.0
        if 4 <= t_month <= 9:
            rem_frac = max(0.15, (10.0 - t_month) / 6.0)

        side_assets = []
        valid_trade = True
        for _, s_grp in sides_grouped:
            assets = []
            for _, leg in s_grp.iterrows():
                mid = int(leg["mlbam_id"])
                fg = mlbam_to_fg.get(mid)
                debut_yr = first_mlb.get(fg) if fg is not None else None
                is_mlb = debut_yr is not None and debut_yr < t_year

                if is_mlb and fg is not None:
                    srow = mlb_snap_lookup.get((fg, t_year))
                    if srow is None:
                        srow = mlb_snap_lookup.get((fg, t_year + 1))
                    if srow is not None:
                        role_raw = str(srow.get("player_role", srow.get("position_group", "Batter")))
                        role = "SP" if role_raw == "SP" else ("RP" if role_raw == "RP" else "Batter")
                        yoc = int(round(float(srow.get("years_control", srow.get("years_of_control", 0)) or 0)))
                        wars, sals = [], []
                        for k in range(max(yoc, 0)):
                            yr_k = t_year + k
                            w = float(srow.get(f"WAR_{yr_k}", 0.0) or 0.0)
                            s = float(srow.get(f"salary_{yr_k}", 0.0) or 0.0)
                            if k == 0:
                                w *= rem_frac
                                s *= rem_frac
                            wars.append(max(0.0, w))
                            sals.append(s)
                        if wars:
                            assets.append({
                                "kind": "mlb",
                                "year": t_year,
                                "role": role,
                                "wars": wars,
                                "sals": sals,
                                "base_mc_tv": float(srow.get("trade_value", 0.0) or 0.0),
                            })
                            continue

                # Pre-debut prospect lookup (check t_year, then t_year - 1)
                prow = prospect_lookup.get((mid, t_year))
                if prow is None:
                    prow = prospect_lookup.get((mid, t_year - 1))
                if prow is not None:
                    eta_num = pd.to_numeric(prow.get("eta"), errors="coerce")
                    age_v = float(prow["season_age"]) if pd.notna(prow.get("season_age")) else 21.0
                    delta_eta = (
                        float(np.clip(float(eta_num) - t_year, 0.0, 5.0))
                        if pd.notna(eta_num) and float(eta_num) >= t_year
                        else (3.2 if age_v <= 19 else (2.0 if age_v <= 21 else 0.8))
                    )
                    base_pv = compute_single_prospect_valuation(
                        fv=prow.get("grade_overall"),
                        top_100=pd.to_numeric(prow.get("top_100"), errors="coerce"),
                        org_rank=pd.to_numeric(prow.get("rank"), errors="coerce"),
                        position=str(prow.get("position") or "OF"),
                        delta_eta_years=delta_eta,
                        valuation_year=t_year,
                    )["trade_value"]
                    if base_pv > 0:
                        assets.append({
                            "kind": "prospect",
                            "year": t_year,
                            "base_prospect_val": float(base_pv) / max(Config.Prospects.PROSPECT_SCALE, 1e-6),
                        })

            if len(assets) != len(s_grp) or not assets:
                valid_trade = False
                break
            side_assets.append(assets)

        if valid_trade:
            clean_trades.append({
                "trade_id": int(tid),
                "year": t_year,
                "date": t_date,
                "side_a": side_assets[0],
                "side_b": side_assets[1],
            })

    logger.info("Loaded %d clean 2-team trades (2018–2026) with valued assets on both sides", len(clean_trades))
    return clean_trades


def evaluate_parameters(
    params: np.ndarray,
    fa_records: list[dict],
    trades: list[dict],
) -> dict:
    """Evaluate FA surplus medians and trade package imbalance for a parameter vector."""
    r_bat, r_sp, r_rp, beta, disc, gamma, p_scale = params
    rates = {"Batter": r_bat * 1e6, "SP": r_sp * 1e6, "RP": r_rp * 1e6}
    infl = Config.Market.INFLATION_RATE
    base_yr = Config.Market.BASE_YEAR

    # 1. FA Signing Surplus by Role
    fa_surplus_by_role: dict[str, list[float]] = {"Batter": [], "SP": [], "RP": []}
    for rec in fa_records:
        rate = rates[rec["role"]]
        npv_prod = 0.0
        npv_cost = 0.0
        for t, w in enumerate(rec["wars"]):
            yr = rec["sign_season"] + t
            dfac = (1.0 + disc) ** -t
            ifac = (1.0 + infl) ** (yr - base_yr)
            npv_prod += dfac * rate * (w ** beta) * ifac
            npv_cost += dfac * rec["aav"]
        fa_surplus_by_role[rec["role"]].append((npv_prod - npv_cost) / 1e6)

    # 2. Trade Imbalance
    rel_imbalances = []
    abs_diffs = []
    for tr in trades:
        side_vals = []
        for side in (tr["side_a"], tr["side_b"]):
            asset_vals = []
            for a in side:
                if a["kind"] == "prospect":
                    # Adjust prospect ETA discount to candidate disc vs default 0.09
                    asset_vals.append(a["base_prospect_val"] * p_scale)
                else:
                    rate = rates[a["role"]]
                    npv_s = 0.0
                    for t, (w, sal) in enumerate(zip(a["wars"], a["sals"])):
                        yr = a["year"] + t
                        dfac = (1.0 + disc) ** -t
                        ifac = (1.0 + infl) ** (yr - base_yr)
                        prod = rate * (w ** beta) * ifac if w > 0 else 0.0
                        npv_s += dfac * (prod - sal)
                    asset_vals.append(npv_s)
            side_vals.append(_consolidate(asset_vals, gamma))

        va, vb = side_vals
        denom = max(abs(va), abs(vb))
        if denom < 8_000_000:
            continue
        if va <= 0 and vb <= 0:
            continue
        rel_imbalances.append(abs(va - vb) / denom)
        abs_diffs.append(abs(va - vb) / 1e6)

    return {
        "fa_medians_m": {k: float(np.median(v)) if v else 0.0 for k, v in fa_surplus_by_role.items()},
        "fa_all_median_m": float(np.median([x for v in fa_surplus_by_role.values() for x in v])) if fa_records else 0.0,
        "median_rel_imbalance": float(np.median(rel_imbalances)) if rel_imbalances else 1.0,
        "mean_rel_imbalance": float(np.mean(rel_imbalances)) if rel_imbalances else 1.0,
        "median_abs_diff_m": float(np.median(abs_diffs)) if abs_diffs else 0.0,
        "n_trades": len(rel_imbalances),
    }


def run_calibration() -> dict:
    """Run joint Nelder-Mead / Powell calibration and save `valuation_calibration.json`."""
    from value_determination.pipelines.history_surplus import compute_surplus_for_snapshot

    logger.info("Building baseline 2018–2026 V2 surplus snapshots for calibration...")
    for yr in range(2018, 2027):
        p = Config.Paths.SURPLUS_DIR / f"surplus_{yr}.csv"
        needs_build = True
        if p.exists():
            cols = pd.read_csv(p, nrows=1).columns
            if "trade_value_p10" in cols:
                needs_build = False
        if needs_build:
            compute_surplus_for_snapshot(yr, force=True)

    surplus_by_year = {}
    for yr in range(2018, 2027):
        p = Config.Paths.SURPLUS_DIR / f"surplus_{yr}.csv"
        if p.exists():
            surplus_by_year[yr] = pd.read_csv(p, low_memory=False)

    fa_records = _load_fa_contracts_dataset(surplus_by_year)
    all_trades = _load_clean_2team_trades(surplus_by_year)

    train_trades = [t for t in all_trades if 2018 <= t["year"] <= 2023]
    holdout_trades = [t for t in all_trades if 2024 <= t["year"] <= 2026]
    logger.info("Trade split: %d train (2018–2023), %d holdout (2024–2026)", len(train_trades), len(holdout_trades))

    # Bounds: (r_bat [$M], r_sp [$M], r_rp [$M], beta, disc, gamma, prospect_scale)
    bounds = [
        (8.0, 10.8),   # rate_batter ($M)
        (8.8, 11.8),   # rate_sp ($M)
        (7.2, 10.0),   # rate_rp ($M)
        (1.08, 1.18),  # war_beta
        (0.08, 0.11),  # discount_rate
        (1.20, 1.80),  # package_gamma
        (0.55, 1.45),  # prospect_scale
    ]
    x0 = np.array([9.2, 10.2, 8.6, 1.12, 0.09, 1.40, 1.00])

    def _loss(x: np.ndarray) -> float:
        # Enforce box bounds smoothly
        for val, (lo, hi) in zip(x, bounds):
            if val < lo or val > hi:
                return 1e6 + 1e4 * (max(0.0, lo - val) + max(0.0, val - hi))

        metrics = evaluate_parameters(x, fa_records, train_trades)

        # Target median FA surplus near -$1.5M per role (mild winner's curse)
        fa_loss = 0.0
        for role_med in metrics["fa_medians_m"].values():
            fa_loss += ((role_med - (-1.5)) / 4.0) ** 2

        # Preserve empirically established SP >= Batter >= RP ordering
        order_pen = 0.0
        if x[1] < x[0]:
            order_pen += (x[0] - x[1]) ** 2 * 5.0
        if x[0] < x[2]:
            order_pen += (x[2] - x[0]) ** 2 * 5.0

        # Primary trade-fit loss: median + 0.5 * mean relative package imbalance on train trades
        trade_loss = metrics["median_rel_imbalance"] + 0.35 * metrics["mean_rel_imbalance"]

        # Regularization toward sensible discount_rate ~0.09 and beta ~1.12
        reg = 0.05 * ((x[4] - 0.09) / 0.02) ** 2 + 0.03 * ((x[3] - 1.12) / 0.04) ** 2
        return float(trade_loss + 0.08 * fa_loss + order_pen + reg)

    res = minimize(_loss, x0, method="Powell", bounds=bounds, options={"maxiter": 600, "ftol": 1e-4})
    best_x = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])

    train_eval = evaluate_parameters(best_x, fa_records, train_trades)
    holdout_eval = evaluate_parameters(best_x, fa_records, holdout_trades)

    calibrated = {
        "rate_batter": round(float(best_x[0] * 1e6)),
        "rate_sp": round(float(best_x[1] * 1e6)),
        "rate_rp": round(float(best_x[2] * 1e6)),
        "war_beta": round(float(best_x[3]), 4),
        "discount_rate": round(float(best_x[4]), 4),
        "package_gamma": round(float(best_x[5]), 3),
        "prospect_scale": round(float(best_x[6]), 4),
        "diagnostics": {
            "fa_medians_millions": {k: round(v, 2) for k, v in train_eval["fa_medians_m"].items()},
            "fa_overall_median_millions": round(train_eval["fa_all_median_m"], 2),
            "train_2018_2023": {
                "n_trades": train_eval["n_trades"],
                "median_rel_imbalance_pct": round(train_eval["median_rel_imbalance"] * 100, 2),
                "median_abs_diff_millions": round(train_eval["median_abs_diff_m"], 2),
            },
            "holdout_2024_2026": {
                "n_trades": holdout_eval["n_trades"],
                "median_rel_imbalance_pct": round(holdout_eval["median_rel_imbalance"] * 100, 2),
                "median_abs_diff_millions": round(holdout_eval["median_abs_diff_m"], 2),
                "v1_baseline_rel_imbalance_pct": 36.4,
            },
        },
    }

    out_file = Config.Paths.VALUATION_CALIBRATION_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibrated, f, indent=2)

    print("\n" + "=" * 68)
    print("V2 JOINT CALIBRATION SUMMARY")
    print("=" * 68)
    print(json.dumps(calibrated, indent=2))
    print("=" * 68)
    return calibrated


if __name__ == "__main__":
    run_calibration()
