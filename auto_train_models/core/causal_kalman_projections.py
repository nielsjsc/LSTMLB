"""
Causal Hierarchical Bayesian Network + Era-Neutral Kalman Projection Engine
===========================================================================

Replaces flat 3-year box-score Marcel averages and single-season (`iloc[-1]`)
multivariate overrides with a 3-layer State-Space Causal DAG:

  [Pedigree / Waiver Prior P_0(FV, Role)]
               |
               v
  [Layer 1: Physical Primitives (Kalman-filtered, N_0 ~ 12-95 PA/IP)]
    Batters : maxEV, sc_ev50, Barrel%, HardHit%, Pull_FB, sc_sprint_speed
    Pitchers: Stuff+, Location+, Pitching+, botStf, botCmd, botERA, vFA, GB%
               |
               v
  [Layer 2: Plate-Discipline & Contact Process (Kalman-filtered, N_0 ~ 25-130)]
    Batters : Contact%, Z-Contact%, SwStr%, O-Swing%, Z-Swing%, xISO, xBABIP
    Pitchers: SwStr%, CSW%, Contact%, F-Strike%, Zone%, Swing%, SIERA, xERA
               |
               v
  [Layer 3: Composed MLB Rate & Counting Projections (Years 0 .. future_years)]

Key Properties:
- **Era-Neutral Filtering**: Subtracts contemporaneous PA/IP-weighted MLB season
  means before Kalman filtering so ball-drag, sticky-stuff, and sensor shifts
  cancel identically, re-anchoring to the current run environment at `cutoff_year`.
- **Survivorship-Corrected Aging**: Applies empirical aging deltas during both
  historical state propagation and future projection while damping survivorship-biased
  pre-27 positive growth by 85%.
- **Epistemic Horizon Reversion**: Multi-year future projections revert gently toward
  the player's prior at a rate inversely proportional to career MLB volume (~1.5%/yr
  for proven veterans vs ~5.5%/yr for unproven call-ups).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from core.id_crosswalk import load_recent_prospect_info
from core.marcel_projections import (
    BATTER_BOUNDS,
    PITCHER_BOUNDS,
    DEFAULT_TRIPLE_SHARE,
    _BF_PER_IP_FALLBACK,
    _build_batter_career_profile,
    _compute_career_era_fip_gap_marcel,
    _derive_bf_per_ip,
    _get_primary_batter_ids,
    _get_smoothed_aging_delta,
    _load_aging_curves,
    _normalize_batted_ball,
    _project_reliever_gmli,
    _reconstruct_fip,
    _reconstruct_siera,
    _safe_resolve_player_name,
)
from core.park_factors import get_woba_residual_factor
from core.stat_composition import compose_all

logger = logging.getLogger(__name__)

# =============================================================================
# Causal DAG Specifications & Sign Priors
# =============================================================================

BATTER_COMPONENTS = ["K%", "BB%", "HBP%", "ISO", "BABIP", "HR/FB", "GB%", "LD%"]
PITCHER_COMPONENTS = ["K%", "BB%", "HBP%", "BABIP", "HR/FB", "GB%", "FB%", "LD%"]

BATTER_JOINT_DAG: Dict[str, List[str]] = {
    "K%":    ["K%", "Contact% (sc)", "Z-Contact% (sc)", "SwStr%", "O-Swing% (sc)"],
    "BB%":   ["BB%", "O-Swing% (sc)", "Z-Swing% (sc)", "xwOBA", "maxEV"],
    "HBP%":  ["HBP%", "O-Swing% (sc)", "Pull%"],
    "ISO":   ["ISO", "xISO", "Barrel%", "sc_ev50_imp", "maxEV", "HardHit%", "Pull_FB", "GB%"],
    "HR/FB": ["HR/FB", "Barrel%", "xISO", "sc_ev50_imp", "maxEV", "Pull_FB", "HardHit%"],
    "BABIP": ["BABIP", "xBABIP_imp", "LD%", "FB%", "GB%", "HardHit%", "sc_sprint_speed", "Pull%"],
    "GB%":   ["GB%", "O-Swing% (sc)"],
    "LD%":   ["LD%", "HardHit%", "Contact% (sc)"],
}

PITCHER_JOINT_DAG: Dict[str, List[str]] = {
    "K%":    ["K%", "SwStr%", "CSW%", "Contact% (sc)", "Z-Contact% (sc)", "Stuff+", "vFA_eff", "GB%"],
    "BB%":   ["BB%", "F-Strike%", "Zone% (sc)", "Swing% (sc)", "Location+", "botCmd"],
    "HBP%":  ["HBP%", "Zone% (sc)", "Location+"],
    "HR/FB": ["HR/FB", "Barrel%", "Stuff+", "GB%"],
    "BABIP": ["BABIP", "GB%", "HardHit%", "SIERA"],
    "GB%":   ["GB%", "SIERA"],
    "FB%":   ["FB%", "GB%"],
    "LD%":   ["LD%"],
}

BATTER_GOOD_SIGN: Dict[str, float] = {
    "K%": -1.0, "SwStr%": -1.0, "O-Swing% (sc)": -1.0, "GB%": -0.25,
    "BB%": +1.0, "Z-Contact% (sc)": +1.0, "Contact% (sc)": +1.0,
    "ISO": +1.0, "xISO": +1.0, "xwOBA": +1.0,
    "BABIP": +0.5, "xBABIP_imp": +0.6, "HR/FB": +1.0,
    "Barrel%": +1.0, "HardHit%": +1.0, "maxEV": +1.0,
    "sc_ev50_imp": +1.0, "Pull_FB": +0.6, "sc_sprint_speed": +0.4,
}

PITCHER_GOOD_SIGN: Dict[str, float] = {
    "K%": +1.0, "SwStr%": +1.0, "CSW%": +1.0,
    "Stuff+": +1.0, "Location+": +1.0, "Pitching+": +1.0,
    "botCmd": +1.0, "vFA_eff": +0.8,
    "Zone% (sc)": +0.6, "F-Strike%": +0.6, "Swing% (sc)": +0.4,
    "Contact% (sc)": -0.8, "Z-Contact% (sc)": -0.8,
    "BB%": -1.0, "HBP%": -0.4, "HR/FB": -0.8,
    "BABIP": -0.4, "Barrel%": -1.0, "HardHit%": -1.0,
    "FIP": -1.0, "xERA": -1.0, "SIERA": -1.0, "botERA": -1.0,
}

_BATTER_FIT_CACHE: Dict[int, tuple] = {}
_PITCHER_FIT_CACHE: Dict[int, tuple] = {}


# =============================================================================
# Feature Preparation & Era-Neutralization Helpers
# =============================================================================

def _prepare_batter_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = raw_df.copy()
    pa = pd.to_numeric(out["PA"], errors="coerce").fillna(0.0)
    if "HBP%" not in out.columns:
        out["HBP%"] = pd.to_numeric(out.get("HBP", 0.0), errors="coerce") / pa.replace(0, np.nan)

    # Neutralize park-distorted surface outcomes (e.g. Coors / T-Mobile) onto neutral scale
    if "Team" in out.columns:
        pf = out["Team"].astype(str).map(get_woba_residual_factor).fillna(1.0)
        for pcol in ("ISO", "BABIP", "HR/FB"):
            if pcol in out.columns:
                out[pcol] = pd.to_numeric(out[pcol], errors="coerce") / pf

    xba = pd.to_numeric(out.get("xBA"), errors="coerce")
    xslg = pd.to_numeric(out.get("xSLG"), errors="coerce")
    iso_num = pd.to_numeric(out.get("ISO"), errors="coerce")
    out["xISO"] = (xslg - xba).clip(lower=0.01, upper=0.45).fillna(iso_num)

    ab = pd.to_numeric(out.get("AB"), errors="coerce")
    hr = pd.to_numeric(out.get("HR"), errors="coerce")
    so = pd.to_numeric(out.get("SO"), errors="coerce")
    sf = pd.to_numeric(out.get("SF", 0.0), errors="coerce").fillna(0.0)
    denom = (ab - so - hr + sf).clip(lower=1.0)
    babip_num = pd.to_numeric(out.get("BABIP"), errors="coerce")
    out["xBABIP_imp"] = ((xba * ab - hr) / denom).clip(lower=0.18, upper=0.40).fillna(babip_num)

    ev50 = pd.to_numeric(out.get("sc_ev50"), errors="coerce")
    ev_fb = pd.to_numeric(out.get("EV"), errors="coerce") * 0.85 + 24.15
    out["sc_ev50_imp"] = ev50.fillna(ev_fb)
    out["Pull_FB"] = pd.to_numeric(out.get("Pull%"), errors="coerce") * pd.to_numeric(out.get("FB%"), errors="coerce")
    return out


def _prepare_pitcher_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = raw_df.copy()
    ip = pd.to_numeric(out["IP"], errors="coerce").fillna(0.0)
    if "HBP%" not in out.columns:
        tbf = pd.to_numeric(out.get("TBF"), errors="coerce").fillna(ip * _BF_PER_IP_FALLBACK)
        out["HBP%"] = pd.to_numeric(out.get("HBP", 0.0), errors="coerce") / tbf.replace(0, np.nan)
    vfa = pd.to_numeric(out.get("vFA (sc)"), errors="coerce")
    for alt in ("vSI (sc)", "vFT (sc)", "vFC (sc)", "FBv"):
        if alt in out.columns:
            vfa = vfa.fillna(pd.to_numeric(out[alt], errors="coerce"))
    out["vFA_eff"] = vfa
    return out


def _compute_season_env_means(df: pd.DataFrame, cols: List[str], vol_col: str) -> Dict[int, Dict[str, float]]:
    env: Dict[int, Dict[str, float]] = {}
    valid_df = df[pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0) > 0]
    for yr, grp in valid_df.groupby("Season"):
        w = pd.to_numeric(grp[vol_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        yr_means: Dict[str, float] = {}
        for c in cols:
            if c not in grp.columns:
                continue
            vals = pd.to_numeric(grp[c], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(vals) & (w > 0)
            if m.sum() >= 15:
                yr_means[c] = float(np.average(vals[m], weights=w[m]))
        env[int(yr)] = yr_means
    return env


def _center_by_season_env(df: pd.DataFrame, cols: List[str], env: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    seasons = out["Season"].astype(int)
    for c in cols:
        if c not in out.columns:
            continue
        yr_map = {yr: m.get(c, np.nan) for yr, m in env.items()}
        out[c] = pd.to_numeric(out[c], errors="coerce") - seasons.map(yr_map)
    return out


def _get_target_env_anchor(env: Dict[int, Dict[str, float]], cols: List[str], cutoff_year: int) -> Dict[str, float]:
    e0 = env.get(cutoff_year, {})
    e1 = env.get(cutoff_year - 1, e0)
    anchor: Dict[str, float] = {}
    for c in cols:
        v0 = e0.get(c, np.nan)
        v1 = e1.get(c, v0)
        if not pd.isna(v0) and not pd.isna(v1):
            anchor[c] = float(0.65 * v0 + 0.35 * v1)
        elif not pd.isna(v0):
            anchor[c] = float(v0)
        elif not pd.isna(v1):
            anchor[c] = float(v1)
        else:
            avail = [env[y][c] for y in sorted(env) if y <= cutoff_year and c in env[y]]
            anchor[c] = float(avail[-1]) if avail else 0.0
    return anchor


# =============================================================================
# Empirical-Bayes Kalman Hyperparameter & Causal Ridge Fitter
# =============================================================================

def _estimate_kalman_hyperparams(
    centered_df: pd.DataFrame,
    cols: List[str],
    vol_col: str,
    min_season: int,
    max_season: int,
    min_vol: float,
) -> Dict[str, dict]:
    sub = centered_df[
        (centered_df["Season"] >= min_season)
        & (centered_df["Season"] <= max_season)
        & (pd.to_numeric(centered_df[vol_col], errors="coerce").fillna(0.0) >= min_vol)
    ].sort_values(["IDfg", "Season"]).copy()

    params: Dict[str, dict] = {}
    for col in cols:
        if col not in sub.columns:
            continue
        s = sub[["IDfg", "Season", vol_col, col]].dropna().copy()
        if len(s) < 80:
            params[col] = {"mu": 0.0, "std": 0.03, "tau2": 9e-4, "q2": 1.8e-4, "n0": 200.0}
            continue

        w = s[vol_col].to_numpy(dtype=float)
        v = s[col].to_numpy(dtype=float)
        mu = float(np.average(v, weights=w))
        tot_var = max(float(np.average((v - mu) ** 2, weights=w)), 1e-10)

        s_next = s.rename(columns={"Season": "Season_prev", vol_col: "vol_next", col: "val_next"})
        s_next["Season_prev"] -= 1
        p1 = s.merge(s_next, left_on=["IDfg", "Season"], right_on=["IDfg", "Season_prev"])

        s_next2 = s.rename(columns={"Season": "Season_prev", vol_col: "vol_next", col: "val_next"})
        s_next2["Season_prev"] -= 2
        p2 = s.merge(s_next2, left_on=["IDfg", "Season"], right_on=["IDfg", "Season_prev"])

        if len(p1) < 40:
            tau2 = tot_var * 0.50
            q2 = tau2 * 0.18
            n0 = float(np.median(w))
        else:
            w1 = 2.0 * p1[vol_col] * p1["vol_next"] / (p1[vol_col] + p1["vol_next"])
            cov1 = max(float(np.average((p1[col] - mu) * (p1["val_next"] - mu), weights=w1)), tot_var * 0.08)
            if len(p2) >= 35:
                w2 = 2.0 * p2[vol_col] * p2["vol_next"] / (p2[vol_col] + p2["vol_next"])
                cov2 = float(np.average((p2[col] - mu) * (p2["val_next"] - mu), weights=w2))
                q2 = float(np.clip(max(0.0, cov1 - cov2) * 1.5, cov1 * 0.14, cov1 * 0.25))
            else:
                q2 = float(cov1 * 0.18)

            tau2 = float(cov1 + 0.35 * q2)
            mean_inv_n = float(np.average(1.0 / s[vol_col], weights=w))
            noise_var = max(tot_var - cov1, cov1 * 0.08)
            sigma_obs2 = noise_var / max(mean_inv_n, 1e-6)
            n0 = float(np.clip(sigma_obs2 / max(tau2, 1e-12), 15.0, 2200.0))

        params[col] = {
            "mu": 0.0,
            "std": float(math.sqrt(max(tau2, 1e-12))),
            "tau2": float(tau2),
            "q2": float(q2),
            "n0": float(n0),
        }
    return params


def _build_prior_z_offset(fv: Optional[float], career_vol: float, stab_vol: float) -> float:
    surv_fade = max(0.0, 1.0 - career_vol / max(stab_vol, 1.0))
    if surv_fade <= 0.0:
        return 0.0
    if fv is None or pd.isna(fv) or float(fv) < 40.0:
        base_z = -0.68
    else:
        base_z = -0.45 + 1.10 * float(np.clip((float(fv) - 40.0) / 25.0, 0.0, 1.0))
    return float(base_z * surv_fade)


def _build_aging_step_table(
    active_cols: List[str],
    curves_dict: dict,
    good_signs: Dict[str, float],
) -> Dict[int, np.ndarray]:
    table: Dict[int, np.ndarray] = {}
    for age in range(18, 48):
        vec = np.zeros(len(active_cols), dtype=float)
        for j, col in enumerate(active_cols):
            ckey = {"xISO": "ISO", "xBABIP_imp": "BABIP"}.get(col, col)
            sc = curves_dict.get(ckey)
            if not sc:
                continue
            delta = float(_get_smoothed_aging_delta(sc, age))
            gs = good_signs.get(col, 0.0)
            # Dampen survivorship-biased pre-27 positive drift by 85%
            if age <= 26 and gs != 0.0 and (delta * gs > 0):
                delta *= 0.15
            vec[j] = delta
        table[age] = vec
    return table


def _run_kalman_panel(
    centered_df: pd.DataFrame,
    cols: List[str],
    hyper: Dict[str, dict],
    vol_col: str,
    cutoff_year: int,
    good_signs: Dict[str, float],
    aging_curves_group: dict,
    prospect_lookup: Optional[Dict[int, dict]] = None,
    stab_vol: float = 600.0,
    max_lookback_years: int = 5,
    include_next_year_aging: bool = True,
) -> pd.DataFrame:
    all_hist = centered_df[
        (centered_df["Season"] <= cutoff_year)
        & (pd.to_numeric(centered_df[vol_col], errors="coerce").fillna(0.0) > 0)
    ]
    if all_hist.empty:
        return pd.DataFrame()

    career_vol_map = all_hist.groupby("IDfg")[vol_col].sum().to_dict()
    min_yr = cutoff_year - max_lookback_years + 1
    sub = all_hist[all_hist["Season"] >= min_yr].sort_values(["IDfg", "Season"]).copy()
    if sub.empty:
        sub = all_hist.sort_values(["IDfg", "Season"]).groupby("IDfg").tail(2).copy()

    active_cols = [c for c in cols if c in hyper and c in sub.columns]
    std_vec = np.array([hyper[c]["std"] for c in active_cols], dtype=float)
    sign_vec = np.array([good_signs.get(c, 0.0) for c in active_cols], dtype=float)
    tau2_vec = np.array([hyper[c]["tau2"] for c in active_cols], dtype=float)
    q2_vec = np.array([hyper[c]["q2"] for c in active_cols], dtype=float)
    sigma_obs2_vec = tau2_vec * np.array([hyper[c]["n0"] for c in active_cols], dtype=float)

    age_table = _build_aging_step_table(active_cols, aging_curves_group, good_signs)
    zero_age = np.zeros(len(active_cols), dtype=float)

    ids = sub["IDfg"].to_numpy(dtype=int)
    seasons = sub["Season"].to_numpy(dtype=int)
    ages = pd.to_numeric(sub["Age"], errors="coerce").fillna(27.0).to_numpy(dtype=float)
    vols = pd.to_numeric(sub[vol_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    val_mat = np.column_stack([pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float) for c in active_cols])

    split_idx = np.flatnonzero(np.diff(ids) != 0) + 1
    starts = np.concatenate(([0], split_idx))
    ends = np.concatenate((split_idx, [len(ids)]))

    out_rows = []
    for s_i, e_i in zip(starts, ends):
        pid = int(ids[s_i])
        c_vol = float(career_vol_map.get(pid, vols[s_i:e_i].sum()))
        fv_val = prospect_lookup[pid].get("fv") if (prospect_lookup and pid in prospect_lookup) else None
        z0 = _build_prior_z_offset(fv_val, c_vol, stab_vol=stab_vol)

        prior_m = z0 * sign_vec * std_vec
        m = prior_m.copy()
        P = tau2_vec.copy()

        prev_yr = None
        prev_age = None
        for row_k in range(s_i, e_i):
            yr = int(seasons[row_k])
            age_k = float(ages[row_k])
            n_k = max(float(vols[row_k]), 1.0)
            if prev_yr is not None:
                dt = max(1, yr - prev_yr)
                P = P + dt * q2_vec
                for step in range(1, dt + 1):
                    a_step = int(round((prev_age or age_k) + step))
                    m = m + age_table.get(a_step, zero_age)
            prev_yr = yr
            prev_age = age_k

            obs = val_mat[row_k]
            valid = ~np.isnan(obs)
            if valid.any():
                R = sigma_obs2_vec[valid] / n_k
                K = P[valid] / (P[valid] + R)
                m[valid] = m[valid] + K * (obs[valid] - m[valid])
                P[valid] = (1.0 - K) * P[valid]

        last_yr = int(seasons[e_i - 1])
        last_age = float(ages[e_i - 1])
        gap = max(0, cutoff_year - last_yr)
        if gap > 0:
            P = P + gap * q2_vec
            for step in range(1, gap + 1):
                m = m + age_table.get(int(round(last_age + step)), zero_age)
            miss_decay = 0.90 ** gap
            m = prior_m + miss_decay * (m - prior_m)

        m_out = m + age_table.get(int(round(last_age + gap + 1)), zero_age) if include_next_year_aging else m

        rec = {
            "IDfg": pid,
            "career_vol": c_vol,
            "last_season": last_yr,
            "last_age": last_age,
            "_prior_z0": z0,
        }
        for j, c in enumerate(active_cols):
            rec[c] = m_out[j]
            rec[f"_cur_{c}"] = m[j]
            rec[f"_prior_{c}"] = prior_m[j]
        out_rows.append(rec)

    return pd.DataFrame(out_rows)


def _fit_joint_causal_models(
    centered_df: pd.DataFrame,
    all_cols: List[str],
    dag: Dict[str, List[str]],
    hyper: Dict[str, dict],
    vol_col: str,
    good_signs: Dict[str, float],
    aging_curves_group: dict,
    train_cutoff_years: range,
    min_next_vol: float,
    stab_vol: float,
    role_filter: Optional[str] = None,
) -> Dict[str, dict]:
    panels = []
    for co_yr in train_cutoff_years:
        pros = load_recent_prospect_info(co_yr + 1, recency_years=3)
        filt = _run_kalman_panel(
            centered_df, all_cols, hyper, vol_col, cutoff_year=co_yr,
            good_signs=good_signs, aging_curves_group=aging_curves_group,
            prospect_lookup=pros, stab_vol=stab_vol, include_next_year_aging=True,
        )
        nxt = centered_df[
            (centered_df["Season"] == co_yr + 1)
            & (pd.to_numeric(centered_df[vol_col], errors="coerce").fillna(0.0) >= min_next_vol)
        ].copy()
        if role_filter is not None and not nxt.empty:
            is_sp = nxt["GS"] >= nxt["G"] * 0.5
            nxt = nxt[is_sp if role_filter == "SP" else (~is_sp)].copy()
        if nxt.empty or filt.empty:
            continue
        keep = ["IDfg", vol_col] + [t for t in dag if t in nxt.columns]
        m = filt.merge(nxt[keep], on="IDfg", suffixes=("_hat", "_next"))
        m["co_yr"] = co_yr
        panels.append(m)

    if not panels:
        return {t: {"parents": [t], "intercept": 0.0, "coefs": {t: 0.75}} for t in dag}

    train_df = pd.concat(panels, ignore_index=True)
    years_u = sorted(train_df["co_yr"].unique())
    models: Dict[str, dict] = {}

    for target, parents in dag.items():
        y_col = f"{target}_next" if f"{target}_next" in train_df.columns else target
        avail_parents = [p for p in parents if (f"{p}_hat" in train_df.columns or p in train_df.columns)]
        if not avail_parents:
            avail_parents = [target]
        p_cols = [f"{p}_hat" if f"{p}_hat" in train_df.columns else p for p in avail_parents]

        sub = train_df[train_df[y_col].notna()].copy()
        for pc in p_cols:
            sub[pc] = pd.to_numeric(sub[pc], errors="coerce").fillna(0.0)

        if len(sub) < 60:
            models[target] = {"parents": [target], "intercept": 0.0, "coefs": {target: 0.75}}
            continue

        w = sub[f"{vol_col}_next"].to_numpy(dtype=float) if f"{vol_col}_next" in sub.columns else sub[vol_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        X = np.column_stack([sub[pc].to_numpy(dtype=float) for pc in p_cols])
        yr_arr = sub["co_yr"].to_numpy(dtype=int)

        best_alpha = 0.20
        best_cv_mse = float("inf")
        if len(years_u) >= 2:
            for alpha in (0.01, 0.05, 0.20, 0.60, 2.0, 6.0):
                sq_errs, sq_wts = [], []
                for hold_yr in years_u:
                    tr_m = yr_arr != hold_yr
                    va_m = yr_arr == hold_yr
                    if not tr_m.any() or not va_m.any():
                        continue
                    wt = w[tr_m] / w[tr_m].sum()
                    xm = np.sum(X[tr_m] * wt[:, None], axis=0)
                    xs = np.sqrt(np.maximum(np.sum(((X[tr_m] - xm) ** 2) * wt[:, None], axis=0), 1e-10))
                    ym = float(np.sum(y[tr_m] * wt))
                    Z = (X[tr_m] - xm) / xs
                    ZW = Z * wt[:, None]
                    b_std = np.linalg.solve(Z.T @ ZW + alpha * np.eye(Z.shape[1]), ZW.T @ (y[tr_m] - ym))
                    c_raw = b_std / xs
                    ic = float(ym - np.dot(c_raw, xm))
                    pred_va = ic + X[va_m] @ c_raw
                    sq_errs.append((y[va_m] - pred_va) ** 2 * w[va_m])
                    sq_wts.append(w[va_m])
                if sq_errs:
                    cv_mse = float(np.concatenate(sq_errs).sum() / np.concatenate(sq_wts).sum())
                    if cv_mse < best_cv_mse:
                        best_cv_mse = cv_mse
                        best_alpha = alpha

        wt = w / w.sum()
        xm = np.sum(X * wt[:, None], axis=0)
        xs = np.sqrt(np.maximum(np.sum(((X - xm) ** 2) * wt[:, None], axis=0), 1e-10))
        ym = float(np.sum(y * wt))
        Z = (X - xm) / xs
        ZW = Z * wt[:, None]
        b_std = np.linalg.solve(Z.T @ ZW + best_alpha * np.eye(Z.shape[1]), ZW.T @ (y - ym))
        coefs = b_std / xs
        intercept = float(ym - np.dot(coefs, xm))

        models[target] = {
            "parents": avail_parents,
            "intercept": float(intercept),
            "coefs": {p: float(c) for p, c in zip(avail_parents, coefs)},
        }
    return models


# =============================================================================
# Public Entry Points (Drop-in Compatible with Marcel Batter & Pitcher Outputs)
# =============================================================================

def causal_kalman_batter_projections(
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    future_years: int = 15,
    cutoff_year: int = 2025,
    roster_ids: Optional[Set[int]] = None,
    use_xstats: bool = True,
) -> Optional[pd.DataFrame]:
    """Generate Era-Neutral Causal Kalman batter projections (`year_offset = 0..future_years`)."""
    from configs.batter_config import BatterConfig

    curves = _load_aging_curves()
    batting_curves = curves.get("batting", {})
    min_pa_current = getattr(BatterConfig, "MIN_PA_CURRENT", 70)

    # For pre-Statcast historical backfills (< 2018), delegate cleanly to Marcel
    if cutoff_year < 2018:
        from core.marcel_projections import marcel_batter_projections
        return marcel_batter_projections(
            raw_df=raw_df, player_names=player_names, future_years=future_years,
            cutoff_year=cutoff_year, roster_ids=roster_ids, use_xstats=use_xstats,
        )

    prep_df = _prepare_batter_df(raw_df)
    current_ids = set(
        prep_df[(prep_df["Season"] == cutoff_year) & (prep_df["PA"] >= min_pa_current)]["IDfg"].unique()
    )
    if roster_ids is not None:
        missing = roster_ids - current_ids
        if missing:
            rec_ids = set(
                prep_df[(prep_df["IDfg"].isin(missing)) & (prep_df["Season"] <= cutoff_year) & (prep_df["PA"] >= 1)]["IDfg"].unique()
            )
            current_ids |= rec_ids

    if not current_ids:
        return None

    all_cols = sorted(set(BATTER_JOINT_DAG.keys()) | {p for ps in BATTER_JOINT_DAG.values() for p in ps} | {"xwOBA"})
    statcast_era_df = prep_df[(prep_df["Season"] >= 2015) & (prep_df["Season"] <= cutoff_year)].copy()
    bat_env = _compute_season_env_means(statcast_era_df, all_cols, "PA")
    centered_bat = _center_by_season_env(statcast_era_df, all_cols, bat_env)

    max_train_co = min(cutoff_year - 1, 2024)
    cache_key = max_train_co
    if cache_key in _BATTER_FIT_CACHE:
        bat_hyper, bat_models = _BATTER_FIT_CACHE[cache_key]
    else:
        bat_hyper = _estimate_kalman_hyperparams(centered_bat, all_cols, "PA", min_season=2016, max_season=max_train_co + 1, min_vol=80.0)
        bat_models = _fit_joint_causal_models(
            centered_bat, all_cols, BATTER_JOINT_DAG, bat_hyper, "PA",
            good_signs=BATTER_GOOD_SIGN, aging_curves_group=batting_curves,
            train_cutoff_years=range(max(2016, max_train_co - 5), max_train_co + 1),
            min_next_vol=150.0, stab_vol=600.0,
        )
        _BATTER_FIT_CACHE[cache_key] = (bat_hyper, bat_models)

    from core.name_utils import name_key as _nk_fn
    name_to_fg = {
        _nk_fn(_safe_resolve_player_name(int(pid), player_names, raw_df)): int(pid)
        for pid in current_ids
    }
    pros = load_recent_prospect_info(cutoff_year, recency_years=3, name_to_fg_fallback=name_to_fg)

    filt = _run_kalman_panel(
        centered_bat[centered_bat["IDfg"].isin(current_ids)],
        list(bat_hyper.keys()), bat_hyper, "PA", cutoff_year=cutoff_year,
        good_signs=BATTER_GOOD_SIGN, aging_curves_group=batting_curves,
        prospect_lookup=pros, stab_vol=600.0, include_next_year_aging=True,
    )
    if filt.empty:
        return None

    env_anchor = _get_target_env_anchor(bat_env, BATTER_COMPONENTS, cutoff_year=cutoff_year)
    age_step_comp = _build_aging_step_table(BATTER_COMPONENTS, batting_curves, BATTER_GOOD_SIGN)
    zero_step = np.zeros(len(BATTER_COMPONENTS), dtype=float)

    all_predictions = []
    for _, r in filt.iterrows():
        pid = int(r["IDfg"])
        pname = _safe_resolve_player_name(pid, player_names, raw_df)
        last_age = float(r["last_age"])
        last_season = int(r["last_season"])
        c_pa = float(r["career_vol"])

        p_hist_raw = raw_df[(raw_df["IDfg"] == pid) & (raw_df["Season"] <= cutoff_year) & (raw_df["PA"] >= 20)]
        if p_hist_raw.empty:
            p_hist_raw = raw_df[(raw_df["IDfg"] == pid) & (raw_df["Season"] <= cutoff_year)]
        career_prof = _build_batter_career_profile(p_hist_raw)
        triple_share = career_prof["triple_share"] if career_prof else DEFAULT_TRIPLE_SHARE
        rbi_rate = career_prof["rbi_rate"] if career_prof else 75.0
        r_rate = career_prof["r_rate"] if career_prof else 75.0
        base_woba = career_prof["base_woba"] if career_prof else 0.315

        # Year-1 (cutoff_year + 1) Causal DAG deviation vector & Year-0 (cutoff_year) state
        d_y1 = np.array([
            bat_models[t]["intercept"] + sum(c * float(r.get(p, 0.0) or 0.0) for p, c in bat_models[t]["coefs"].items())
            for t in BATTER_COMPONENTS
        ], dtype=float)
        d_y0 = np.array([float(r.get(f"_cur_{t}", d_y1[j])) for j, t in enumerate(BATTER_COMPONENTS)], dtype=float)
        d_prior = np.array([float(r.get(f"_prior_{t}", 0.0)) for t in BATTER_COMPONENTS], dtype=float)

        # Epistemic annual persistence factor for horizons >= 2 (~0.985 for veterans, ~0.94 for low-PA rookies)
        epistemic_keep = 0.940 + 0.045 * float(np.clip(c_pa / 1200.0, 0.0, 1.0))

        d_running = d_y1.copy()
        for year_offset in range(0, future_years + 1):
            proj_year = cutoff_year + year_offset
            proj_age = last_age + (cutoff_year - last_season) + year_offset

            if year_offset == 0:
                d_cur = 0.55 * d_y0 + 0.45 * d_y1
            elif year_offset == 1:
                d_cur = d_y1
            else:
                d_running = d_prior + epistemic_keep * (d_running - d_prior) + age_step_comp.get(int(round(proj_age)), zero_step)
                d_cur = d_running

            projected = {}
            for j, stat in enumerate(BATTER_COMPONENTS):
                val = env_anchor[stat] + float(d_cur[j])
                # Apply tiny -0.004 BABIP calibration to remove +0.006 compose_all wOBA offset
                if stat == "BABIP":
                    val -= 0.0045
                if stat in BATTER_BOUNDS:
                    lo, hi = BATTER_BOUNDS[stat]
                    val = float(np.clip(val, lo, hi))
                projected[stat] = val

            gb, ld = projected["GB%"], projected["LD%"]
            if 1.0 - gb - ld < 0.10:
                tot = gb + ld
                if tot > 0:
                    projected["GB%"] = gb / tot * 0.90
                    projected["LD%"] = ld / tot * 0.90

            composed = compose_all(
                k_pct=projected["K%"],
                bb_pct=projected["BB%"],
                hbp_pct=projected["HBP%"],
                iso=projected["ISO"],
                babip=projected["BABIP"],
                hr_fb=projected["HR/FB"],
                gb_pct=projected["GB%"],
                ld_pct=projected["LD%"],
                triple_share=triple_share,
            )
            proj_woba = composed["wOBA"]
            woba_ratio = float(np.clip(proj_woba / base_woba, 0.50, 1.50)) if base_woba > 0.15 else 1.0

            all_predictions.append({
                "Name": pname,
                "IDfg": pid,
                "Year": proj_year,
                "Age": proj_age,
                "PA": 650,
                "K%": projected["K%"],
                "BB%": projected["BB%"],
                "HBP%": projected["HBP%"],
                "ISO": projected["ISO"],
                "BABIP": projected["BABIP"],
                "HR/FB": projected["HR/FB"],
                "GB%": projected["GB%"],
                "LD%": projected["LD%"],
                "AVG": composed["AVG"],
                "OBP": composed["OBP"],
                "SLG": composed["SLG"],
                "wOBA": proj_woba,
                "wRC+": composed["wRC+"],
                "FB%": composed["FB%"],
                "HR": composed["HR"],
                "2B": composed["2B"],
                "3B": composed["3B"],
                "1B": composed["1B"],
                "H": composed["H"],
                "RBI": max(0.0, rbi_rate * woba_ratio),
                "R": max(0.0, r_rate * woba_ratio),
                "HBP": composed["HBP"],
            })

    if not all_predictions:
        return None

    output_cols = [
        "Name", "IDfg", "Year", "Age", "PA",
        "BB%", "K%", "AVG", "OBP", "SLG", "wOBA",
        "HR", "2B", "3B", "RBI", "R", "HBP",
        "ISO", "BABIP", "HR/FB", "GB%", "LD%", "FB%",
        "HBP%", "wRC+", "1B", "H",
    ]
    res = pd.DataFrame(all_predictions)
    res = res[[c for c in output_cols if c in res.columns]].sort_values(["Name", "Year"])
    logger.info(
        "Causal Kalman batting: %d rows for %d batters (cutoff=%d, avg wOBA=%.3f)",
        len(res), res["IDfg"].nunique(), cutoff_year, res["wOBA"].mean(),
    )
    return res


def causal_kalman_pitcher_projections(
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    future_years: int = 15,
    cutoff_year: int = 2025,
    roster_ids: Optional[Set[int]] = None,
) -> Optional[pd.DataFrame]:
    """Generate Era-Neutral Causal Kalman pitcher projections (`year_offset = 0..future_years`)."""
    from configs.pitcher_rp_config import PitcherRPConfig
    from configs.pitcher_sp_config import PitcherSPConfig

    if cutoff_year < 2019:
        from core.marcel_projections import marcel_pitcher_projections
        return marcel_pitcher_projections(
            raw_df=raw_df, player_names=player_names, future_years=future_years,
            cutoff_year=cutoff_year, roster_ids=roster_ids,
        )

    curves = _load_aging_curves()
    pitching_curves = curves.get("pitching", {})
    sp_min_ip = getattr(PitcherSPConfig, "MIN_IP_CURRENT", 25)
    rp_min_ip = getattr(PitcherRPConfig, "MIN_IP_CURRENT", 15)

    prep_df = _prepare_pitcher_df(raw_df)
    pitchers_cur = prep_df[prep_df["Season"] == cutoff_year]
    pitchers_prev = prep_df[prep_df["Season"] == cutoff_year - 1]

    player_roles: Dict[int, str] = {}
    qualifying_ids: Set[int] = set()

    for frame in (pitchers_cur, pitchers_prev):
        if frame.empty or "GS" not in frame.columns or "G" not in frame.columns:
            continue
        for pid in frame["IDfg"].unique():
            if int(pid) in player_roles:
                continue
            pdata = frame[frame["IDfg"] == pid]
            tot_g = float(pdata["G"].sum())
            tot_gs = float(pdata["GS"].sum())
            tot_ip = float(pdata["IP"].sum())
            if tot_g <= 0:
                continue
            gs_rate = tot_gs / tot_g
            if gs_rate >= 0.8 and tot_ip >= sp_min_ip:
                player_roles[int(pid)] = "SP"
                qualifying_ids.add(int(pid))
            elif gs_rate < 0.8 and tot_ip >= rp_min_ip:
                player_roles[int(pid)] = "RP"
                qualifying_ids.add(int(pid))

    primary_batter_ids = _get_primary_batter_ids(cutoff_year)
    if roster_ids is not None:
        for pid in (roster_ids - qualifying_ids):
            if int(pid) in player_roles:
                continue
            phist = prep_df[(prep_df["IDfg"] == pid) & (prep_df["Season"] <= cutoff_year) & (prep_df["IP"] >= 1.0)]
            if phist.empty:
                continue
            if int(pid) in primary_batter_ids:
                if float(phist["IP"].max() or 0.0) < 15.0 and float(phist["GS"].max() or 0.0) < 1.0:
                    continue
            last = phist.sort_values("Season").iloc[-1]
            if float(last["G"]) > 0:
                player_roles[int(pid)] = "SP" if (float(last["GS"]) / float(last["G"]) >= 0.8) else "RP"
                qualifying_ids.add(int(pid))

    for pid in list(player_roles.keys()):
        if int(pid) in primary_batter_ids:
            ph = prep_df[(prep_df["IDfg"] == pid) & (prep_df["Season"] <= cutoff_year)]
            if float(ph["IP"].max() or 0.0) < 15.0 and float(ph["GS"].max() or 0.0) < 1.0:
                player_roles.pop(pid, None)
                qualifying_ids.discard(pid)

    if not player_roles:
        return None

    all_cols = sorted(
        set(PITCHER_JOINT_DAG.keys())
        | {p for ps in PITCHER_JOINT_DAG.values() for p in ps}
        | {"botERA", "SIERA", "xERA", "Pitching+", "FIP"}
    )
    pit_recent = prep_df[(prep_df["Season"] >= 2015) & (prep_df["Season"] <= cutoff_year)].copy()
    is_sp_row = pit_recent["GS"] >= pit_recent["G"] * 0.5
    sp_env = _compute_season_env_means(pit_recent[is_sp_row], all_cols, "IP")
    rp_env = _compute_season_env_means(pit_recent[~is_sp_row], all_cols, "IP")
    centered_pit = pd.concat([
        _center_by_season_env(pit_recent[is_sp_row], all_cols, sp_env),
        _center_by_season_env(pit_recent[~is_sp_row], all_cols, rp_env),
    ], ignore_index=True)

    max_train_co = min(cutoff_year - 1, 2024)
    if max_train_co in _PITCHER_FIT_CACHE:
        sp_hyper, rp_hyper, sp_models, rp_models = _PITCHER_FIT_CACHE[max_train_co]
    else:
        sp_hyper = _estimate_kalman_hyperparams(
            centered_pit[centered_pit["GS"] >= centered_pit["G"] * 0.5],
            all_cols, "IP", min_season=2017, max_season=max_train_co + 1, min_vol=40.0,
        )
        rp_hyper = _estimate_kalman_hyperparams(
            centered_pit[centered_pit["GS"] < centered_pit["G"] * 0.5],
            all_cols, "IP", min_season=2017, max_season=max_train_co + 1, min_vol=20.0,
        )
        sp_models = _fit_joint_causal_models(
            centered_pit, all_cols, PITCHER_JOINT_DAG, sp_hyper, "IP",
            good_signs=PITCHER_GOOD_SIGN, aging_curves_group=pitching_curves,
            train_cutoff_years=range(max(2017, max_train_co - 5), max_train_co + 1),
            min_next_vol=50.0, stab_vol=150.0, role_filter="SP",
        )
        rp_models = _fit_joint_causal_models(
            centered_pit, all_cols, PITCHER_JOINT_DAG, rp_hyper, "IP",
            good_signs=PITCHER_GOOD_SIGN, aging_curves_group=pitching_curves,
            train_cutoff_years=range(max(2017, max_train_co - 5), max_train_co + 1),
            min_next_vol=25.0, stab_vol=70.0, role_filter="RP",
        )
        _PITCHER_FIT_CACHE[max_train_co] = (sp_hyper, rp_hyper, sp_models, rp_models)

    pros = load_recent_prospect_info(cutoff_year + 1, recency_years=3)
    target_ids = set(player_roles.keys())
    sub_pit = centered_pit[centered_pit["IDfg"].isin(target_ids)]

    filt_sp = _run_kalman_panel(
        sub_pit, list(sp_hyper.keys()), sp_hyper, "IP", cutoff_year=cutoff_year,
        good_signs=PITCHER_GOOD_SIGN, aging_curves_group=pitching_curves,
        prospect_lookup=pros, stab_vol=150.0, include_next_year_aging=True,
    ).set_index("IDfg")
    filt_rp = _run_kalman_panel(
        sub_pit, list(rp_hyper.keys()), rp_hyper, "IP", cutoff_year=cutoff_year,
        good_signs=PITCHER_GOOD_SIGN, aging_curves_group=pitching_curves,
        prospect_lookup=pros, stab_vol=70.0, include_next_year_aging=True,
    ).set_index("IDfg")

    anchor_cols = PITCHER_COMPONENTS + ["botERA", "SIERA", "xERA"]
    sp_anchor = _get_target_env_anchor(sp_env, anchor_cols, cutoff_year=cutoff_year)
    rp_anchor = _get_target_env_anchor(rp_env, anchor_cols, cutoff_year=cutoff_year)
    age_step_pit = _build_aging_step_table(PITCHER_COMPONENTS, pitching_curves, PITCHER_GOOD_SIGN)
    zero_step = np.zeros(len(PITCHER_COMPONENTS), dtype=float)

    all_predictions = []
    for pid, role in player_roles.items():
        if pid not in filt_sp.index and pid not in filt_rp.index:
            continue
        r = filt_sp.loc[pid] if (role == "SP" and pid in filt_sp.index) else filt_rp.loc[pid]
        models = sp_models if role == "SP" else rp_models
        anc = sp_anchor if role == "SP" else rp_anchor

        pname = _safe_resolve_player_name(pid, player_names, raw_df)
        last_age = float(r["last_age"])
        last_season = int(r["last_season"])
        c_ip = float(r["career_vol"])

        p_hist_raw = prep_df[(prep_df["IDfg"] == pid) & (prep_df["Season"] <= cutoff_year) & (prep_df["IP"] >= 5.0)]
        if p_hist_raw.empty:
            p_hist_raw = prep_df[(prep_df["IDfg"] == pid) & (prep_df["Season"] <= cutoff_year)]

        era_fip_gap = _compute_career_era_fip_gap_marcel(p_hist_raw)

        d_y1 = np.array([
            models[t]["intercept"] + sum(c * float(r.get(p, 0.0) or 0.0) for p, c in models[t]["coefs"].items())
            for t in PITCHER_COMPONENTS
        ], dtype=float)
        d_y0 = np.array([float(r.get(f"_cur_{t}", d_y1[j])) for j, t in enumerate(PITCHER_COMPONENTS)], dtype=float)
        d_prior = np.array([float(r.get(f"_prior_{t}", 0.0)) for t in PITCHER_COMPONENTS], dtype=float)

        # Holistic run-prevention offset (botERA / SIERA / xERA) to prevent component-reconstruction drift
        hol_y1 = (
            0.40 * (anc["botERA"] + float(r.get("botERA", 0.0) or 0.0))
            + 0.35 * (anc["SIERA"] + float(r.get("SIERA", 0.0) or 0.0))
            + 0.25 * (anc["xERA"] + float(r.get("xERA", 0.0) or 0.0))
        )

        # Compute Year-1 SIERA for reliever gmLI projection
        gb1, fb1, _ = _normalize_batted_ball(
            anc["GB%"] + d_y1[PITCHER_COMPONENTS.index("GB%")],
            anc["FB%"] + d_y1[PITCHER_COMPONENTS.index("FB%")],
            anc["LD%"] + d_y1[PITCHER_COMPONENTS.index("LD%")],
        )
        siera_y1 = float(_reconstruct_siera(
            float(np.clip(anc["K%"] + d_y1[0], 0.08, 0.48)),
            float(np.clip(anc["BB%"] + d_y1[1], 0.02, 0.20)),
            gb1,
        ))
        gmli_y1 = _project_reliever_gmli(p_hist_raw, siera_y1) if role == "RP" else 1.0

        stab_ref = 300.0 if role == "SP" else 120.0
        epistemic_keep = 0.935 + 0.050 * float(np.clip(c_ip / stab_ref, 0.0, 1.0))

        d_running = d_y1.copy()
        for year_offset in range(0, future_years + 1):
            proj_year = cutoff_year + year_offset
            proj_age = last_age + (cutoff_year - last_season) + year_offset

            if year_offset == 0:
                d_cur = 0.55 * d_y0 + 0.45 * d_y1
            elif year_offset == 1:
                d_cur = d_y1
            else:
                d_running = d_prior + epistemic_keep * (d_running - d_prior) + age_step_pit.get(int(round(proj_age)), zero_step)
                d_cur = d_running

            projected = {}
            for j, stat in enumerate(PITCHER_COMPONENTS):
                val = anc[stat] + float(d_cur[j])
                if stat in PITCHER_BOUNDS:
                    lo, hi = PITCHER_BOUNDS[stat]
                    val = float(np.clip(val, lo, hi))
                projected[stat] = val

            gb, fb, ld = _normalize_batted_ball(projected["GB%"], projected["FB%"], projected["LD%"])
            projected["GB%"], projected["FB%"], projected["LD%"] = gb, fb, ld

            bip_rate = max(1.0 - projected["K%"] - projected["BB%"] - projected["HBP%"], 0.20)
            hr_pct = projected["HR/FB"] * projected["FB%"] * bip_rate
            bf_per_ip = _derive_bf_per_ip(
                projected["K%"], projected["BB%"], projected["HBP%"], hr_pct, projected["BABIP"]
            )
            fip_comp = _reconstruct_fip(
                projected["K%"], projected["BB%"], projected["HBP%"],
                projected["HR/FB"], projected["FB%"], bf_per_ip,
            )

            # Blend 25% toward holistic botERA/SIERA/xERA anchor (decayed with horizon)
            hol_w = 0.25 * (epistemic_keep ** max(0, year_offset - 1))
            fip_aging_shift = fip_comp - _reconstruct_fip(
                float(np.clip(anc["K%"] + d_y1[0], 0.08, 0.48)),
                float(np.clip(anc["BB%"] + d_y1[1], 0.02, 0.20)),
                float(np.clip(anc["HBP%"] + d_y1[2], 0.002, 0.03)),
                float(np.clip(anc["HR/FB"] + d_y1[4], 0.04, 0.22)),
                fb1,
                bf_per_ip,
            )
            fip = float((1.0 - hol_w) * fip_comp + hol_w * (hol_y1 + max(0.0, fip_aging_shift)))
            era = float(np.clip(fip + era_fip_gap, 0.5, 10.0))
            siera = float(_reconstruct_siera(projected["K%"], projected["BB%"], projected["GB%"]))

            if role == "RP":
                decay_steps = max(0, year_offset - 1)
                proj_gmli = float(np.clip(1.08 + (gmli_y1 - 1.08) * (0.88 ** decay_steps), 0.55, 2.00))
            else:
                proj_gmli = 1.0

            all_predictions.append({
                "Name": pname,
                "Year": proj_year,
                "Age": proj_age,
                "Role": role,
                "IDfg": pid,
                "K%": projected["K%"],
                "BB%": projected["BB%"],
                "HBP%": projected["HBP%"],
                "BABIP": projected["BABIP"],
                "FIP": fip,
                "ERA": era,
                "GB%": projected["GB%"],
                "FB%": projected["FB%"],
                "HR/FB": projected["HR/FB"],
                "SIERA": siera,
                "LD%": projected["LD%"],
                "K/9": projected["K%"] * bf_per_ip * 9.0,
                "BB/9": projected["BB%"] * bf_per_ip * 9.0,
                "HR/9": hr_pct * bf_per_ip * 9.0,
                "gmLI": round(proj_gmli, 3),
            })

    if not all_predictions:
        return None

    res = pd.DataFrame(all_predictions).sort_values(["Name", "Year"])
    logger.info(
        "Causal Kalman pitching: %d rows for %d pitchers (cutoff=%d, avg FIP=%.2f)",
        len(res), res["IDfg"].nunique(), cutoff_year, res["FIP"].mean(),
    )
    return res
