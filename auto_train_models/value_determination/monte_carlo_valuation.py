"""
Vectorized Monte Carlo MLB Trade Valuation Engine
=================================================

Replaces deterministic surplus sums, arbitrary confidence weights, and prospect
value floors with a fast vectorized NumPy simulation (`N` players x `M=1,000`
correlated career paths x `T` control years, `SEED=42`).

Economic Mechanics Priced Automatically Along Every Path:
1. **Correlated AR(1) Talent Shocks** (`rho = 0.75`) with experience-scaled and
   playing-time-scaled `sigma`, plus age/role career-attrition hazards.
2. **Bench / DFA Replacement Floor**: On-field production is floored at `0 WAR`
   (`W_eff = max(W, 0)`), while guaranteed salaries remain owed in full.
3. **Time & Risk Discounting**: Future surplus discounted at `r = 9%/yr`
   alongside `4%/yr` MLB WAR dollar inflation.
4. **Non-Anticipative Dynamic Arbitration & Non-Tenders**: Before each season `t`,
   using only information available at `t - 1` (`F_{t-1}`), arbitration salaries
   update from realized production and clubs non-tender negative-EV `Pre-Arb` /
   `Arb` contracts at `$0`.
5. **Asymmetric Contract Options**: Player Opt-Outs / Player Options (exercised
   when expected open-market value exceeds remaining guaranteed cash) and Club /
   Mutual Options (exercised when expected surplus beats `-buyout`).
6. **Smooth Debut Continuity**: Freshly debuted rookies (`< 150 PA` / `< 40 IP`)
   with a recent scouting FV grade transition smoothly from their pre-debut
   `prospect_valuation` distribution to their MLB Monte Carlo distribution.
7. **Free Agents (`Team == 'FA'`)**: Assigned `trade_value = NaN` (untradable).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import ndtr

from core.id_crosswalk import load_recent_prospect_info
from .config import CURRENT_YEAR, Config, logger
from .dollar_valuation import get_min_salary_for_year
from .prospect_valuation import compute_single_prospect_valuation


def _solve_martingale_location(mu_pos: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Solve `m * Phi(m / sigma) + sigma * phi(m / sigma) = mu_pos` elementwise
    via 6 vectorized Newton-Raphson iterations so `E[max(m + sigma * Z, 0)] == mu_pos`.

    Eliminates Jensen's-inequality / truncation 'vega bias' on low-WAR and
    sub-replacement players while leaving high-WAR stars (`mu >> sigma`) unchanged.
    """
    s = np.maximum(sigma, 1e-6)
    zero_mask = mu_pos <= 1e-6
    q = np.where(zero_mask, 0.0, mu_pos / s)

    u = np.where(
        q >= 1.2,
        q,
        np.clip(np.log(np.maximum(q, 1e-9) / 0.3989422804014327), -8.5, 2.0),
    )
    for _ in range(6):
        Phi = ndtr(u)
        phi = 0.3989422804014327 * np.exp(-0.5 * u * u)
        f_val = u * Phi + phi - q
        u = np.clip(u - f_val / np.maximum(Phi, 1e-12), -10.0, 30.0)

    return np.where(zero_mask, -12.0 * s, u * s)


def _truncated_normal_pos_mean(loc: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    """Exact analytical expectation `E[max(X, 0)]` for `X ~ N(loc, scale^2)`."""
    s = np.maximum(scale, 1e-6)
    u = np.clip(loc / s, -12.0, 30.0)
    return np.maximum(0.0, loc * ndtr(u) + s * 0.3989422804014327 * np.exp(-0.5 * u * u))


def _worp_prod_val(
    eff_war: np.ndarray,
    w_slot_full: np.ndarray | float,
    s_min_full: np.ndarray | float,
    role_rate: np.ndarray | float,
    inflation: np.ndarray | float,
    beta: float,
    sf: np.ndarray | float = 1.0,
) -> np.ndarray:
    """
    Vectorized Roster-Constrained Production Value (WORP), matching
    `Config.Market.calculate_production_value`:
      - w_slot = w_slot_full * sf
      - s_min  = s_min_full  * sf
      - For 0 <= W <= w_slot: V = s_min * (W / w_slot)
      - For W > w_slot:       V = s_min + sf * R * (((W - w_slot) / sf) ** beta) * inflation
    """
    sf_arr = np.asarray(sf, dtype=float)
    sf_safe = np.maximum(sf_arr, 1e-6)
    w_slot = w_slot_full * sf_safe
    s_min = s_min_full * sf_safe
    w_s = np.maximum(w_slot, 1e-6)
    sub_val = s_min * np.clip(eff_war / w_s, 0.0, 1.0)
    excess_full = np.maximum(eff_war - w_slot, 0.0) / sf_safe
    above_val = s_min + sf_safe * role_rate * (excess_full ** beta) * inflation
    return np.where(
        sf_arr <= 1e-6,
        0.0,
        np.where(eff_war <= w_slot, sub_val, above_val),
    )

# Status integer codes for fast vectorized branching
_ST_FA = 0
_ST_SIGNED = 1
_ST_PREARB = 2
_ST_ARB1 = 3
_ST_ARB2 = 4
_ST_ARB3 = 5
_ST_ARB4 = 6
_ST_TEAM_OPT = 7
_ST_PLAYER_OPT = 8
_ST_MUTUAL_OPT = 9

_STATUS_TO_CODE = {
    "Free Agent": _ST_FA,
    "Signed": _ST_SIGNED,
    "Unknown": _ST_SIGNED,
    "Pre-Arb": _ST_PREARB,
    "Arb-1": _ST_ARB1,
    "Arb-1 (Super 2)": _ST_ARB1,
    "Arb-2": _ST_ARB2,
    "Arb-3": _ST_ARB3,
    "Arb-4": _ST_ARB4,
    "Team Option": _ST_TEAM_OPT,
    "Vesting Option": _ST_TEAM_OPT,
    "Player Option": _ST_PLAYER_OPT,
    "Opt-Out": _ST_PLAYER_OPT,
    "Mutual Option": _ST_MUTUAL_OPT,
}

_CODE_TO_ARB_SHARE = {
    _ST_ARB1: Config.Contracts.ARB_PERCENT["Arb-1"],
    _ST_ARB2: Config.Contracts.ARB_PERCENT["Arb-2"],
    _ST_ARB3: Config.Contracts.ARB_PERCENT["Arb-3"],
    _ST_ARB4: Config.Contracts.ARB_PERCENT["Arb-4"],
}


def analyze_contract_options(df: pd.DataFrame, current_year: Optional[int] = None) -> pd.DataFrame:
    """
    Populate baseline `fa_year`, `earliest_fa_year`, and deterministic
    `probable_fa_year` on the player timeline DataFrame.
    """
    if current_year is None:
        current_year = CURRENT_YEAR

    result = df.copy()
    status_col = "Status" if "Status" in result.columns else "Normalized_Status"
    sal_col = "salary_cost" if "salary_cost" in result.columns else "contract_value"
    prod_col = "production_value" if "production_value" in result.columns else "Base_Value"
    surp_col = "annual_surplus" if "annual_surplus" in result.columns else "surplus_value"

    for old_c in ("FA_Year", "fa_year", "probable_fa_year", "earliest_fa_year"):
        if old_c in result.columns:
            result = result.drop(columns=[old_c])

    fa_years = (
        result.loc[result[status_col] == "Free Agent"]
        .groupby("IDfg")["Year"]
        .min()
        .rename("fa_year")
    )
    result = result.merge(fa_years, on="IDfg", how="left")

    missing = result.loc[result["fa_year"].isna(), "IDfg"].unique()
    for pid in missing:
        rows = result[result["IDfg"] == pid]
        controlled = rows.loc[
            rows[status_col].isin([
                "Signed", "Pre-Arb", "Arb-1", "Arb-1 (Super 2)", "Arb-2", "Arb-3", "Arb-4",
                "Team Option", "Player Option", "Mutual Option", "Vesting Option", "Opt-Out", "Unknown",
            ])
        ]["Year"]
        result.loc[result["IDfg"] == pid, "fa_year"] = (
            float(controlled.max() + 1) if len(controlled) else float(current_year)
        )

    result["probable_fa_year"] = result["fa_year"]

    option_types = {"Player Option", "Team Option", "Mutual Option", "Vesting Option", "Opt-Out"}
    option_min = (
        result.loc[result[status_col].isin(option_types) & (result["Year"] >= current_year)]
        .groupby("IDfg")["Year"]
        .min()
        .rename("earliest_fa_year")
    )
    result = result.merge(option_min, on="IDfg", how="left")
    result["earliest_fa_year"] = result["earliest_fa_year"].fillna(result["fa_year"])

    r_disc = Config.Market.DISCOUNT_RATE
    for pid in result.loc[result[status_col].isin(option_types), "IDfg"].unique():
        pdata = result[(result["IDfg"] == pid) & (result["Year"] >= current_year)].sort_values("Year")
        if pdata.empty:
            continue
        probable_fa = pdata["fa_year"].iloc[0]

        for _, row in pdata[pdata[status_col].isin(option_types)].iterrows():
            yr = int(row["Year"])
            if yr >= probable_fa:
                break
            st = row[status_col]
            surplus = row.get(surp_col, np.nan)

            declined = False
            if st in ("Player Option", "Opt-Out"):
                rem = pdata[(pdata["Year"] >= yr) & (pdata["Year"] < probable_fa)]
                disc = (1.0 + r_disc) ** -(rem["Year"] - yr)
                rem_sal = (rem[sal_col].fillna(0.0) * disc).sum()
                rem_prod = (rem[prod_col].fillna(0.0) * disc).sum()
                declined = rem_prod > rem_sal
            else:
                declined = pd.notna(surplus) and float(surplus) < 0.0

            if declined:
                probable_fa = float(yr)
                break

        result.loc[result["IDfg"] == pid, "probable_fa_year"] = probable_fa

    # Keep uppercase aliases on internal DataFrame until exporter normalizes
    result["FA_Year"] = result["fa_year"]
    return result


def _load_career_volume_lookup(as_of_year: int) -> dict[int, dict[str, float]]:
    """
    Compute career MLB PA and IP through `as_of_year` (prior seasons + any
    in-progress `as_of_year` stats) for experience-scaled dispersion & rookie blending.
    """
    vol: dict[int, dict[str, float]] = {}

    bat_files = [Config.Paths.HISTORIC_BATTING_FILE_CLASSIC]
    cur_bat = Config.Paths.DATA_DIR / "current_season" / f"mlb_batting_data_{as_of_year}_{as_of_year}.csv"
    if cur_bat.exists():
        bat_files.append(cur_bat)

    for path in bat_files:
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["IDfg", "Season", "PA"], low_memory=False)
        df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce").fillna(0.0)
        df = df[(df["IDfg"].notna()) & (df["Season"] <= as_of_year)]
        # Avoid double counting as_of_year if present in both historic and current_season
        if path != Config.Paths.HISTORIC_BATTING_FILE_CLASSIC:
            df = df[df["Season"] == as_of_year]
        else:
            if cur_bat.exists():
                df = df[df["Season"] < as_of_year]
        for fg, s in df.groupby("IDfg")["PA"].sum().items():
            entry = vol.setdefault(int(fg), {"pa": 0.0, "ip": 0.0})
            entry["pa"] += float(s)

    pit_files = [Config.Paths.HISTORIC_PITCHING_FILE_CLASSIC]
    cur_pit = Config.Paths.DATA_DIR / "current_season" / f"mlb_pitching_data_{as_of_year}_{as_of_year}.csv"
    if cur_pit.exists():
        pit_files.append(cur_pit)

    for path in pit_files:
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["IDfg", "Season", "IP"], low_memory=False)
        df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce").fillna(0.0)
        df = df[(df["IDfg"].notna()) & (df["Season"] <= as_of_year)]
        if path != Config.Paths.HISTORIC_PITCHING_FILE_CLASSIC:
            df = df[df["Season"] == as_of_year]
        else:
            if cur_pit.exists():
                df = df[df["Season"] < as_of_year]
        for fg, s in df.groupby("IDfg")["IP"].sum().items():
            entry = vol.setdefault(int(fg), {"pa": 0.0, "ip": 0.0})
            entry["ip"] += float(s)

    return vol


def calculate_trade_values(
    df: pd.DataFrame,
    current_year: Optional[int] = None,
    apply_ros_fraction: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Run the vectorized Monte Carlo career simulator across all controlled players.

    Populates `trade_value` (expected NPV surplus), `trade_value_p10`,
    `trade_value_p50`, `trade_value_p90`, and updates `probable_fa_year`
    where Monte Carlo modal control differs.
    """
    if current_year is None:
        current_year = CURRENT_YEAR
    if apply_ros_fraction is None:
        apply_ros_fraction = (current_year == CURRENT_YEAR)

    Config.Market.load_calibration()
    mc_cfg = Config.MonteCarlo
    mkt_cfg = Config.Market

    result = df.copy()
    status_col = "Status" if "Status" in result.columns else "Normalized_Status"
    role_col = "player_role" if "player_role" in result.columns else "position_group"
    sal_col = "salary_cost" if "salary_cost" in result.columns else "contract_value"

    for col in ("trade_value", "trade_value_p10", "trade_value_p50", "trade_value_p90"):
        result[col] = np.nan

    # Remaining-season fraction per team (1.0 in preseason or historical snapshots)
    games_map: dict[str, int] = {}
    if apply_ros_fraction:
        try:
            from value_determination.pipelines.ros import fetch_team_games_played, _team_remaining_fraction
            games_map = fetch_team_games_played(current_year)
        except Exception:
            games_map = {}

    career_vol_map = _load_career_volume_lookup(current_year)
    recent_prospects = load_recent_prospect_info(
        as_of_year=current_year,
        recency_years=Config.Prospects.RECENCY_YEARS,
    )

    future_df = result[result["Year"] >= current_year].copy()
    if future_df.empty:
        return result

    # Filter to active team-controlled players (exclude unsigned Free Agents 'FA')
    max_fa_year = int(min(2042, max(current_year + 16, future_df["fa_year"].dropna().max() if "fa_year" in future_df.columns and future_df["fa_year"].notna().any() else current_year + 15)))
    T = max(1, max_fa_year - current_year)
    years_grid = np.arange(current_year, current_year + T, dtype=int)

    player_ids: list[int] = []
    player_roles: list[str] = []
    player_teams: list[str] = []
    rem_fracs: list[float] = []
    career_vols: list[float] = []
    has_signed_extension: list[bool] = []

    mu_war_list = []
    age_list = []
    status_list = []
    fixed_sal_list = []
    pt_share_list = []

    for pid, grp in future_df.groupby("IDfg", sort=False):
        grp_sorted = grp.sort_values("Year")
        cur_rows = grp_sorted[grp_sorted["Year"] == current_year]
        team_val = str(cur_rows["Team"].iloc[0] if not cur_rows.empty else grp_sorted["Team"].iloc[0] or "FA").strip()

        # Free agents / restricted-list players without an active MLB club have no trade value
        if not team_val or team_val.upper() == "FA":
            continue

        fa_yr = grp_sorted["fa_year"].iloc[0] if "fa_year" in grp_sorted.columns else np.nan
        if pd.isna(fa_yr) or float(fa_yr) <= current_year:
            continue

        role_raw = str(grp_sorted[role_col].iloc[0] if role_col in grp_sorted.columns else "Batter")
        role = "SP" if role_raw == "SP" else ("RP" if role_raw == "RP" else "Batter")

        r_frac = 1.0
        if apply_ros_fraction and games_map:
            try:
                from value_determination.pipelines.ros import _team_remaining_fraction
                r_frac = float(_team_remaining_fraction(team_val, games_map))
            except Exception:
                r_frac = 1.0

        # If current season is complete (offseason) and player reaches FA before next season, 0 control remains
        if r_frac <= 1e-6 and float(fa_yr) <= current_year + 1:
            continue

        c_entry = career_vol_map.get(int(pid), {"pa": 0.0, "ip": 0.0})
        c_vol = c_entry["pa"] if role == "Batter" else c_entry["ip"]

        yr_index = grp_sorted.drop_duplicates("Year").set_index("Year")
        mu_w = np.zeros(T, dtype=float)
        ages = np.zeros(T, dtype=float)
        st_codes = np.zeros(T, dtype=int)  # default _ST_FA = 0
        f_sal = np.zeros(T, dtype=float)
        pt_sh = np.ones(T, dtype=float)

        base_age = float(grp_sorted["Age"].dropna().iloc[0]) if grp_sorted["Age"].notna().any() else 27.0
        base_age_yr = int(grp_sorted.loc[grp_sorted["Age"].notna(), "Year"].iloc[0]) if grp_sorted["Age"].notna().any() else current_year

        spot_col = "Spotrac_Salary" if "Spotrac_Salary" in yr_index.columns else (
            "spotrac_salary" if "spotrac_salary" in yr_index.columns else None
        )
        has_multi_yr_contract = False
        for chk_yr in yr_index.index:
            if chk_yr >= float(fa_yr):
                continue
            chk_r = yr_index.loc[chk_yr]
            chk_st = str(chk_r.get(status_col, "Free Agent") or "Free Agent")
            chk_code = _STATUS_TO_CODE.get(chk_st, _ST_FA)
            if chk_code in (_ST_SIGNED, _ST_TEAM_OPT, _ST_PLAYER_OPT, _ST_MUTUAL_OPT):
                has_multi_yr_contract = True
                break
            if chk_yr > current_year and spot_col is not None:
                sp_v = chk_r.get(spot_col, np.nan)
                if pd.notna(sp_v) and float(sp_v) > 0:
                    has_multi_yr_contract = True
                    break

        signed_any = has_multi_yr_contract
        for t_idx, yr in enumerate(years_grid):
            ages[t_idx] = base_age + (yr - base_age_yr)
            if yr >= float(fa_yr) or yr not in yr_index.index:
                st_codes[t_idx] = _ST_FA
                continue

            r = yr_index.loc[yr]
            st_str = str(r.get(status_col, "Free Agent") or "Free Agent")
            code = _STATUS_TO_CODE.get(st_str, _ST_SIGNED if pd.notna(r.get(sal_col)) else _ST_FA)

            # Pre-FA extension buyout years: Spotrac labels contracted Pre-Arb/Arb buyout years
            # as 'Pre-Arb' / 'Arb-1..4' with explicit Spotrac_Salary > 0. Those years are
            # guaranteed club obligations (not dynamic arb or free non-tender years).
            spot_v = r.get(spot_col, np.nan) if spot_col is not None else np.nan
            has_spot_sal = pd.notna(spot_v) and float(spot_v) > 0
            if code in (_ST_PREARB, _ST_ARB1, _ST_ARB2, _ST_ARB3, _ST_ARB4):
                if (t_idx >= 1 and has_spot_sal) or (t_idx == 0 and has_multi_yr_contract and has_spot_sal):
                    code = _ST_SIGNED
                    signed_any = True

            st_codes[t_idx] = code
            if code in (_ST_SIGNED, _ST_TEAM_OPT, _ST_PLAYER_OPT, _ST_MUTUAL_OPT):
                signed_any = True

            w_val = r.get("WAR", 0.0)
            mu_w[t_idx] = 0.0 if pd.isna(w_val) else float(w_val)

            s_val = r.get(sal_col, np.nan)
            f_sal[t_idx] = 0.0 if pd.isna(s_val) else float(s_val)

            # Playing-time share scales random WAR shock amplitude & roster-slot hurdle
            if role == "Batter":
                pa_v = r.get("PA", r.get("pa", 600.0))
                denom = 600.0 * (r_frac if t_idx == 0 else 1.0)
                pt_sh[t_idx] = float(np.clip((600.0 if pd.isna(pa_v) else float(pa_v)) / max(denom, 60.0), 0.15, 1.0))
            elif role == "SP":
                ip_v = r.get("IP", r.get("ip", 160.0))
                denom = 160.0 * (r_frac if t_idx == 0 else 1.0)
                pt_sh[t_idx] = float(np.clip((160.0 if pd.isna(ip_v) else float(ip_v)) / max(denom, 20.0), 0.15, 1.0))
            else:
                ip_v = r.get("IP", r.get("ip", 65.0))
                denom = 65.0 * (r_frac if t_idx == 0 else 1.0)
                pt_sh[t_idx] = float(np.clip((65.0 if pd.isna(ip_v) else float(ip_v)) / max(denom, 10.0), 0.25, 1.0))

        if (st_codes == _ST_FA).all():
            continue

        # Upstream Bayesian scouting prior blend on `mu_war` for young (<25.5yo) pedigreed
        # Top-100 / FV>=50 rookies with <1 full MLB season (`c_vol < stab_vol`).
        # Blending `mu_war` BEFORE contract simulation ensures pre-FA rookie extensions
        # (e.g. Colt Emerson, Roman Anthony, Konnor Griffin, Samuel Basallo) honor BOTH
        # their elite scouting prior AND their exact guaranteed Spotrac contract obligations.
        pinfo_up = recent_prospects.get(int(pid))
        if pinfo_up is not None and base_age <= 25.5:
            fv_num = float(pinfo_up.get("fv") or 0.0)
            t100 = pinfo_up.get("top_100")
            has_elite_pedigree = (fv_num >= 50.0) or pd.notna(t100)
            s_vol = float(mc_cfg.STABILIZATION_VOLUME[role])
            if has_elite_pedigree and c_vol < s_vol:
                eff_fv = fv_num if fv_num >= 45.0 else 50.0
                # Map scouting FV (45..70) -> prime annual fWAR expectation
                fv_prime_war = 1.1 + 4.1 * float(np.clip((eff_fv - 45.0) / 25.0, 0.0, 1.0)) ** 1.15
                if pd.notna(t100) and float(t100) <= 25.0:
                    fv_prime_war += 0.35 * (1.0 - (float(t100) - 1.0) / 25.0)
                if role == "RP":
                    fv_prime_war *= 0.42
                elif role == "SP":
                    fv_prime_war *= 0.92
                w_mlb_up = float(np.clip(c_vol / s_vol, 0.15, 1.0))
                for t_idx in range(T):
                    if st_codes[t_idx] == _ST_FA:
                        continue
                    age_t = ages[t_idx]
                    # Ramp in age 20-21, plateau 22-28, decline 29+
                    age_mult = min(1.0, 0.82 + 0.09 * max(0.0, age_t - 20.0)) if age_t < 22.0 else max(0.25, 1.0 - 0.075 * max(0.0, age_t - 28.5))
                    scout_w = fv_prime_war * age_mult * (r_frac if t_idx == 0 else 1.0)
                    mu_w[t_idx] = w_mlb_up * mu_w[t_idx] + (1.0 - w_mlb_up) * scout_w
                    if t_idx >= 1:
                        pt_sh[t_idx] = max(pt_sh[t_idx], w_mlb_up * pt_sh[t_idx] + (1.0 - w_mlb_up) * 0.88)

        player_ids.append(int(pid))
        player_roles.append(role)
        player_teams.append(team_val)
        rem_fracs.append(r_frac)
        career_vols.append(c_vol)
        has_signed_extension.append(signed_any)
        mu_war_list.append(mu_w)
        age_list.append(ages)
        status_list.append(st_codes)
        fixed_sal_list.append(f_sal)
        pt_share_list.append(pt_sh)

    N = len(player_ids)
    if N == 0:
        return result

    M = mc_cfg.N_SIMS
    rng = np.random.default_rng(mc_cfg.SEED)

    mu_war = np.vstack(mu_war_list)          # (N, T)
    ages = np.vstack(age_list)               # (N, T)
    st_codes = np.vstack(status_list)        # (N, T)
    fixed_sal = np.vstack(fixed_sal_list)    # (N, T)
    pt_share = np.vstack(pt_share_list)      # (N, T)
    rem_frac_arr = np.array(rem_fracs)       # (N,)

    # Vectorized role parameters (N,)
    role_rates = np.array([mkt_cfg.role_rate(r) for r in player_roles])
    base_sigmas = np.array([mc_cfg.SIGMA_ESTABLISHED[r] for r in player_roles])
    stab_vols = np.array([mc_cfg.STABILIZATION_VOLUME[r] for r in player_roles])

    # Gate rookie ceiling dispersion by age & scouting pedigree so unpedigreed 26-30yo
    # call-ups do not manufacture phantom star-upside variance out of low MLB sample size
    has_pedigree = np.array([
        (
            pid in recent_prospects
            and (
                float(recent_prospects[pid].get("fv") or 0.0) >= 45.0
                or pd.notna(recent_prospects[pid].get("top_100"))
            )
        )
        for pid in player_ids
    ], dtype=bool)
    upside_gate = np.where(has_pedigree, 1.0, np.clip((26.0 - ages[:, 0]) / 2.5, 0.0, 1.0))
    exp_mult = 1.0 + upside_gate * (mc_cfg.SIGMA_ROOKIE_MULTIPLIER - 1.0) * np.clip(
        1.0 - np.array(career_vols) / stab_vols, 0.0, 1.0
    )
    player_sigma = base_sigmas * exp_mult    # (N,)

    h_base = np.array([mc_cfg.ATTRITION_BASE_HAZARD[r] for r in player_roles])[:, None]
    h_slope = np.array([mc_cfg.ATTRITION_AGE_SLOPE[r] for r in player_roles])[:, None]
    hazards = np.clip(h_base + h_slope * np.maximum(0.0, ages - mc_cfg.ATTRITION_AGE_PIVOT), 0.0, 0.45)

    # Pre-compute year-level inflation, WORP minimum salary, & remaining-season-aware discount matrix
    inflation_vec = (1.0 + mkt_cfg.INFLATION_RATE) ** (years_grid - mkt_cfg.BASE_YEAR)
    s_min_vec = np.array([mkt_cfg.min_salary_for_year(int(y)) for y in years_grid])
    elapsed_tau = np.maximum(0.0, np.arange(T, dtype=float)[None, :] - (1.0 - rem_frac_arr[:, None]))
    discount_mat = (1.0 + mkt_cfg.DISCOUNT_RATE) ** -elapsed_tau
    prearb_min_vec = np.array([get_min_salary_for_year(int(y), "Pre-Arb") for y in years_grid])
    arb_floor_by_code = {
        c: np.array([
            get_min_salary_for_year(
                int(y),
                {3: "Arb-1", 4: "Arb-2", 5: "Arb-3", 6: "Arb-4"}[c],
            )
            for y in years_grid
        ])
        for c in (_ST_ARB1, _ST_ARB2, _ST_ARB3, _ST_ARB4)
    }

    # ── 1. Simulate Martingale AR(1) shocks & survival indicators (N, M, T) ──
    eps = rng.standard_normal((N, M, T))
    unif = rng.random((N, M, T))

    rho = mc_cfg.AR1_RHO
    innov_scale = math.sqrt(max(0.0, 1.0 - rho * rho))

    # Full-season Year-0 reference WAR & playing-time share so late-season/offseason proration
    # never shrinks latent AR(1) state variance or next-year arbitration baselines
    ref_war = mu_war.copy()
    pt_full = pt_share.copy()
    if T > 1:
        low_rem = rem_frac_arr < 0.25
        ref_war[low_rem, 0] = mu_war[low_rem, 1]
        ref_war[~low_rem, 0] = mu_war[~low_rem, 0] / np.maximum(rem_frac_arr[~low_rem], 0.25)
        pt_full[low_rem, 0] = pt_share[low_rem, 1]

    # Full-season 26-man WORP roster-slot hurdle per player-year (N, T) — no part-time discount
    hurdle_base = np.array([mkt_cfg.WORP_HURDLE_FULL[r] for r in player_roles], dtype=float)[:, None]
    w_slot_full = np.broadcast_to(hurdle_base, (N, T)).copy()

    # Skill-conditioned 40-man roster churn hazard on non-guaranteed years (t >= 1):
    # sub-hurdle depth faces high annual DFA / waiver / AAA replacement turnover
    sub_hurdle_gap = np.clip((w_slot_full - ref_war) / np.maximum(w_slot_full, 0.25), 0.0, 1.0)
    non_guar_mask = st_codes != _ST_SIGNED
    hazards = np.where(non_guar_mask, np.clip(hazards + 0.42 * sub_hurdle_gap, 0.0, 0.72), hazards)
    hazards[:, 0] = 0.0  # Current-season IL already reflected in Year-0 playing time

    # Talent-amplitude moderator (scaled by role's star WAR benchmark) so 0-WAR / middle-depth
    # players do not invent superstar or elite-closer noise out of symmetric Gaussian shocks
    star_war_scale = np.array([1.4 if r == "RP" else 2.8 for r in player_roles], dtype=float)[:, None]
    talent_mod = 0.42 + 0.58 * np.clip(np.maximum(ref_war, 0.0) / star_war_scale, 0.0, 1.0)  # (N, T)
    sigma_it = player_sigma[:, None] * pt_full * talent_mod                                  # (N, T)

    # Exact marginal variance of the AR(1) shock process z_{i,t} for Martingale conservation
    s_var = np.zeros((N, T), dtype=float)
    s_var[:, 0] = sigma_it[:, 0] ** 2
    for t in range(1, T):
        s_var[:, t] = (rho * rho) * s_var[:, t - 1] + (1.0 - rho * rho) * (sigma_it[:, t] ** 2)
    s_marginal = np.sqrt(np.maximum(s_var, 1e-12))

    # Solve Martingale location `m_loc` so E[max(m_loc + z, 0)] == max(ref_war, 0)
    m_loc = _solve_martingale_location(np.maximum(ref_war, 0.0), s_marginal)  # (N, T)

    z = np.zeros((N, M, T), dtype=float)
    z[:, :, 0] = sigma_it[:, 0][:, None] * eps[:, :, 0]
    for t in range(1, T):
        z[:, :, t] = rho * z[:, :, t - 1] + sigma_it[:, t][:, None] * innov_scale * eps[:, :, t]

    survived = np.ones((N, M, T), dtype=bool)
    for t in range(1, T):
        survived[:, :, t] = survived[:, :, t - 1] & (unif[:, :, t] > hazards[:, t][:, None])

    sim_war = np.where(survived, m_loc[:, None, :] + z, 0.0)
    s_rem_0 = sigma_it[:, 0] * np.sqrt(np.clip(rem_frac_arr, 0.0, 1.0))
    m_loc_0 = _solve_martingale_location(np.maximum(mu_war[:, 0], 0.0), s_rem_0)
    z_rem_0 = s_rem_0[:, None] * eps[:, :, 0]
    sim_war[:, :, 0] = np.where(rem_frac_arr[:, None] <= 1e-6, 0.0, m_loc_0[:, None] + z_rem_0)
    eff_war = np.maximum(sim_war, 0.0)

    # Vectorized WORP production value (sf = rem_frac in Year 0, 1.0 in Years 1..T-1)
    sf_mat = np.ones((N, 1, T), dtype=float)
    sf_mat[:, 0, 0] = rem_frac_arr
    prod_val = _worp_prod_val(
        eff_war=eff_war,
        w_slot_full=w_slot_full[:, None, :],
        s_min_full=s_min_vec[None, None, :],
        role_rate=role_rates[:, None, None],
        inflation=inflation_vec[None, None, :],
        beta=mkt_cfg.WAR_BETA,
        sf=sf_mat,
    )  # (N, M, T)
    full_prod_0 = _worp_prod_val(
        eff_war=np.maximum(0.0, m_loc[:, 0][:, None] + z[:, :, 0]),
        w_slot_full=w_slot_full[:, 0:1],
        s_min_full=s_min_vec[0],
        role_rate=role_rates[:, None],
        inflation=inflation_vec[0],
        beta=mkt_cfg.WAR_BETA,
        sf=1.0,
    )  # (N, M)

    # ── 2. Non-anticipative pathwise contract & option valuation ─────────────
    control_alive = np.broadcast_to(st_codes[:, 0:1] != _ST_FA, (N, M)).copy()
    npv_surplus = np.zeros((N, M), dtype=float)
    prev_arb_sal = np.zeros((N, M), dtype=float)
    prev_prod_val = np.zeros((N, M), dtype=float)

    for t in range(T):
        st_t = st_codes[:, t]                       # (N,)
        active_fa = st_t == _ST_FA
        control_alive = control_alive & (~active_fa[:, None])
        if not control_alive.any():
            break

        sf_t = rem_frac_arr[:, None] if t == 0 else 1.0

        # Exact conditional expectation E[eff_war_t | F_{t-1}] before year t is played
        if t == 0:
            z_prev = np.zeros((N, M), dtype=float)
            surv_prev = np.ones((N, M), dtype=float)
            exp_war_t = np.broadcast_to(np.maximum(mu_war[:, 0:1], 0.0), (N, M)).copy()
        else:
            z_prev = z[:, :, t - 1]
            surv_prev = survived[:, :, t - 1].astype(float)
            cond_pos_w = _truncated_normal_pos_mean(
                m_loc[:, t][:, None] + rho * z_prev,
                (sigma_it[:, t] * innov_scale)[:, None],
            )
            exp_war_t = surv_prev * (1.0 - hazards[:, t][:, None]) * cond_pos_w

        exp_prod_t = _worp_prod_val(
            eff_war=exp_war_t,
            w_slot_full=w_slot_full[:, t:t + 1],
            s_min_full=s_min_vec[t],
            role_rate=role_rates[:, None],
            inflation=inflation_vec[t],
            beta=mkt_cfg.WAR_BETA,
            sf=sf_t,
        )

        sal_t = np.zeros((N, M), dtype=float)
        buyout_cost_t = np.zeros((N, M), dtype=float)

        # (a) Guaranteed Signed years
        m_signed = st_t == _ST_SIGNED
        if m_signed.any():
            sal_t[m_signed] = fixed_sal[m_signed, t][:, None]

        # (b) Pre-Arb years: club tenders for 26-man roster iff expected production >= p_min
        # (equivalently exp_war_t >= w_slot under WORP; sub-hurdle Pre-Arb is replaced via waivers/AAA)
        m_prearb = st_t == _ST_PREARB
        if m_prearb.any():
            n_pre = int(m_prearb.sum())
            p_min = prearb_min_vec[t] * (rem_frac_arr[m_prearb] if t == 0 else np.ones(n_pre, dtype=float))
            if t == 0:
                sal_t[m_prearb] = np.where(
                    fixed_sal[m_prearb, 0][:, None] > 0,
                    fixed_sal[m_prearb, 0][:, None],
                    p_min[:, None],
                )
            else:
                keep_pre = exp_prod_t[m_prearb] >= p_min[:, None]
                control_alive[m_prearb] = control_alive[m_prearb] & keep_pre
                sal_t[m_prearb] = p_min[:, None]

        # (c) Arbitration years (Arb-1 .. Arb-4): dynamic salary + non-tender decision
        for arb_code in (_ST_ARB1, _ST_ARB2, _ST_ARB3, _ST_ARB4):
            m_arb = st_t == arb_code
            if not m_arb.any():
                continue
            arb_share = _CODE_TO_ARB_SHARE[arb_code]
            floor_sal = arb_floor_by_code[arb_code][t]

            if t == 0:
                r_f = rem_frac_arr[m_arb]
                est_prod_0 = exp_prod_t[m_arb, 0]
                settled = np.where(
                    r_f <= 1e-6,
                    0.0,
                    np.where(
                        fixed_sal[m_arb, 0] > 0,
                        fixed_sal[m_arb, 0],
                        np.maximum(floor_sal * r_f, est_prod_0 * arb_share),
                    ),
                )
                sal_t[m_arb] = settled[:, None]
            else:
                based_on_prev = prev_prod_val[m_arb] * arb_share
                based_on_exp = exp_prod_t[m_arb] * arb_share
                proposed_arb = np.maximum(
                    floor_sal,
                    np.maximum(0.55 * based_on_prev + 0.45 * based_on_exp, prev_arb_sal[m_arb] * 0.80),
                )
                tender = exp_prod_t[m_arb] >= proposed_arb
                control_alive[m_arb] = control_alive[m_arb] & tender
                sal_t[m_arb] = proposed_arb

        # (d) Option years (Team Option, Player Option / Opt-Out, Mutual Option)
        m_opt = np.isin(st_t, [_ST_TEAM_OPT, _ST_PLAYER_OPT, _ST_MUTUAL_OPT])
        if m_opt.any():
            opt_indices = np.where(m_opt)[0]
            for idx_i in opt_indices:
                code_i = st_t[idx_i]
                opt_sal = fixed_sal[idx_i, t]
                if code_i == _ST_TEAM_OPT:
                    buyout = opt_sal * mc_cfg.OPTION_BUYOUT_FRACTION
                    exercise = (exp_prod_t[idx_i] - opt_sal) >= -buyout
                    declined_now = control_alive[idx_i] & (~exercise)
                    buyout_cost_t[idx_i, declined_now] = buyout
                    control_alive[idx_i] = control_alive[idx_i] & exercise
                    sal_t[idx_i] = opt_sal
                elif code_i == _ST_PLAYER_OPT:
                    rem_prod_npv = np.zeros(M, dtype=float)
                    rem_sal_npv = np.zeros(M, dtype=float)
                    cum_surv = surv_prev[idx_i].copy()
                    for tau in range(t, T):
                        c_tau = st_codes[idx_i, tau]
                        if c_tau in (_ST_FA, _ST_TEAM_OPT, _ST_MUTUAL_OPT):
                            break
                        cum_surv = cum_surv * (1.0 - hazards[idx_i, tau])
                        if t == 0:
                            e_war_tau = cum_surv * max(0.0, float(ref_war[idx_i, tau]))
                        else:
                            steps_ahead = tau - t + 1
                            rho_pow = rho ** steps_ahead
                            cond_var_tau = max(1e-8, s_var[idx_i, tau] - (rho_pow * rho_pow) * s_var[idx_i, t - 1])
                            e_war_tau = cum_surv * _truncated_normal_pos_mean(
                                m_loc[idx_i, tau] + rho_pow * z_prev[idx_i],
                                math.sqrt(cond_var_tau),
                            )
                        e_prod_tau = _worp_prod_val(
                            eff_war=np.asarray(e_war_tau, dtype=float),
                            w_slot_full=w_slot_full[idx_i, tau],
                            s_min_full=s_min_vec[tau],
                            role_rate=role_rates[idx_i],
                            inflation=inflation_vec[tau],
                            beta=mkt_cfg.WAR_BETA,
                            sf=1.0,
                        )
                        dfac = (1.0 + mkt_cfg.DISCOUNT_RATE) ** -(tau - t)
                        rem_prod_npv += e_prod_tau * dfac
                        rem_sal_npv += fixed_sal[idx_i, tau] * dfac

                    player_stays = rem_sal_npv >= rem_prod_npv
                    control_alive[idx_i] = control_alive[idx_i] & player_stays
                    sal_t[idx_i] = opt_sal
                elif code_i == _ST_MUTUAL_OPT:
                    buyout = opt_sal * mc_cfg.OPTION_BUYOUT_FRACTION * 0.5
                    both_agree = (exp_prod_t[idx_i] >= opt_sal * 0.95) & (exp_prod_t[idx_i] <= opt_sal * 1.15)
                    declined_now = control_alive[idx_i] & (~both_agree)
                    buyout_cost_t[idx_i, declined_now] = buyout
                    control_alive[idx_i] = control_alive[idx_i] & both_agree
                    sal_t[idx_i] = opt_sal

        # Accrue discounted realized year-t surplus for paths still under club control.
        # Optionable / renewable Pre-Arb players (salary near MLB minimum) can be shuttled to AAA
        # on sub-hurdle seasons rather than burning a full guaranteed MLB salary on the 26-man roster.
        raw_net = prod_val[:, :, t] - sal_t
        p_min_full_t = prearb_min_vec[t] * (rem_frac_arr if t == 0 else 1.0)
        renewable_prearb = (st_t == _ST_PREARB) & (fixed_sal[:, t] <= p_min_full_t * 1.35)
        sub_hurdle_prearb = renewable_prearb & (ref_war[:, t] < w_slot_full[:, t])
        raw_net = np.where(
            sub_hurdle_prearb[:, None],
            np.maximum(0.0, raw_net),
            np.where(renewable_prearb[:, None], np.maximum(-0.25 * sal_t, raw_net), raw_net),
        )
        year_surplus = np.where(control_alive, raw_net, 0.0) - buyout_cost_t
        npv_surplus += discount_mat[:, t][:, None] * year_surplus

        if t == 0:
            rf_col = rem_frac_arr[:, None]
            prev_prod_val = np.where(rf_col > 0.25, prod_val[:, :, 0] / np.maximum(rf_col, 0.25), full_prod_0)
            prev_arb_sal = np.where(
                control_alive,
                np.where(rf_col > 0.25, sal_t / np.maximum(rf_col, 0.25), 0.0),
                0.0,
            )
        else:
            prev_arb_sal = np.where(control_alive, sal_t, 0.0)
            prev_prod_val = prod_val[:, :, t]

    # ── 3. Summarize expected value & quantiles + smooth rookie pedigree blend ─
    tv_mean = npv_surplus.mean(axis=1)
    tv_p10 = np.percentile(npv_surplus, 10, axis=1)
    tv_p50 = np.percentile(npv_surplus, 50, axis=1)
    tv_p90 = np.percentile(npv_surplus, 90, axis=1)

    for i, pid in enumerate(player_ids):
        role = player_roles[i]
        c_vol = career_vols[i]
        blend_thresh = mc_cfg.ROOKIE_PROSPECT_BLEND_VOLUME[role]

        # Cup-of-coffee (<120 PA / <35 IP) smoothing for un-extended recent prospects
        # whose MLB playing-time projection hasn't ramped to full-time starter volume yet
        cup_thresh = 0.35 * blend_thresh
        pinfo = recent_prospects.get(pid)
        if (
            pinfo is not None
            and (float(pinfo.get("fv") or 0.0) >= 45.0 or pd.notna(pinfo.get("top_100")))
            and c_vol < cup_thresh
            and not has_signed_extension[i]
        ):
            pval = compute_single_prospect_valuation(
                fv=pinfo.get("fv"),
                top_100=pinfo.get("top_100"),
                org_rank=pinfo.get("org_rank"),
                position=str(pinfo.get("position") or role),
                delta_eta_years=0.0,  # Already reached MLB
                valuation_year=current_year,
            )
            w_mlb = float(np.clip(c_vol / max(cup_thresh, 1.0), 0.20, 1.0))
            tv_mean[i] = w_mlb * tv_mean[i] + (1.0 - w_mlb) * pval["trade_value"]
            tv_p10[i] = w_mlb * tv_p10[i] + (1.0 - w_mlb) * pval["trade_value_p10"]
            tv_p50[i] = w_mlb * tv_p50[i] + (1.0 - w_mlb) * pval["trade_value_p50"]
            tv_p90[i] = w_mlb * tv_p90[i] + (1.0 - w_mlb) * pval["trade_value_p90"]

    summary_df = pd.DataFrame({
        "IDfg": player_ids,
        "_tv": np.round(tv_mean),
        "_p10": np.round(tv_p10),
        "_p50": np.round(tv_p50),
        "_p90": np.round(tv_p90),
    })

    result = result.merge(summary_df, on="IDfg", how="left")
    fut_mask = result["Year"] >= current_year
    result.loc[fut_mask, "trade_value"] = result.loc[fut_mask, "_tv"]
    result.loc[fut_mask, "trade_value_p10"] = result.loc[fut_mask, "_p10"]
    result.loc[fut_mask, "trade_value_p50"] = result.loc[fut_mask, "_p50"]
    result.loc[fut_mask, "trade_value_p90"] = result.loc[fut_mask, "_p90"]
    result = result.drop(columns=["_tv", "_p10", "_p50", "_p90"])

    cur_slice = result[(result["Year"] == current_year) & result["trade_value"].notna()]
    logger.info(
        "Monte Carlo trade valuation (%d sims, r=%.0f%%, beta=%.2f): %d controlled MLB players | "
        "max=$%.1fM, p90=$%.1fM, median=$%.1fM",
        M,
        mkt_cfg.DISCOUNT_RATE * 100,
        mkt_cfg.WAR_BETA,
        len(cur_slice),
        (cur_slice["trade_value"].max() / 1e6) if not cur_slice.empty else 0.0,
        (cur_slice["trade_value"].quantile(0.90) / 1e6) if not cur_slice.empty else 0.0,
        (cur_slice["trade_value"].median() / 1e6) if not cur_slice.empty else 0.0,
    )
    return result


def add_control_horizon_metrics(
    df: pd.DataFrame,
    current_year: Optional[int] = None,
    apply_ros_fraction: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Compute unambiguous horizon summary metrics for every player (`control_war`,
    `control_production_value`, `control_salary_cost`, `projected_15yr_war`,
    `past_career_war`, etc.).
    """
    if current_year is None:
        current_year = CURRENT_YEAR
    if apply_ros_fraction is None:
        apply_ros_fraction = (current_year == CURRENT_YEAR)

    games_map: dict[str, int] = {}
    if apply_ros_fraction:
        try:
            from value_determination.pipelines.ros import fetch_team_games_played, _team_remaining_fraction
            games_map = fetch_team_games_played(current_year)
        except Exception:
            games_map = {}

    prod_col = "production_value" if "production_value" in df.columns else "Base_Value"
    sal_col = "salary_cost" if "salary_cost" in df.columns else "contract_value"
    surp_col = "annual_surplus" if "annual_surplus" in df.columns else "surplus_value"

    rows = []
    for pid, p in df.groupby("IDfg", sort=False):
        p = p.sort_values("Year")
        cur_rows = p[p["Year"] == current_year]
        team_val = str(cur_rows["Team"].iloc[0] if not cur_rows.empty else p["Team"].iloc[0] or "FA").strip()

        fa_year = p["probable_fa_year"].iloc[0] if "probable_fa_year" in p.columns else np.nan
        if pd.isna(fa_year) and "fa_year" in p.columns:
            fa_year = p["fa_year"].iloc[0]

        is_fa_player = (not team_val) or (team_val.upper() == "FA")
        if is_fa_player:
            control = p.iloc[0:0]
        else:
            control = p[(p["Year"] >= current_year) & (p["Year"] < fa_year)]

        future = p[p["Year"] >= current_year]
        past = p[p["Year"] < current_year]
        n = len(control)

        if n > 0 and apply_ros_fraction and games_map:
            try:
                from value_determination.pipelines.ros import _team_remaining_fraction
                frac = _team_remaining_fraction(team_val, games_map)
            except Exception:
                frac = 1.0
            n_fractional = max(0, n - 1) + frac
        else:
            n_fractional = float(n)

        ctrl_war = float(control["WAR"].fillna(0.0).sum())
        ctrl_prod = float(control[prod_col].fillna(0.0).sum())
        ctrl_sal = float(control[sal_col].fillna(0.0).sum())
        ctrl_surp = float(control[surp_col].fillna(0.0).sum())
        fut_war = float(future["WAR"].fillna(0.0).clip(lower=0.0).sum())
        fut_prod = float(future[prod_col].fillna(0.0).sum())
        past_war = float(past["WAR"].fillna(0.0).sum())
        past_prod = float(past[prod_col].fillna(0.0).sum())

        rows.append({
            "IDfg": pid,
            "control_war": round(ctrl_war, 1),
            "avg_war": round(ctrl_war / n_fractional, 2) if n_fractional > 0 else 0.0,
            "control_production_value": round(ctrl_prod, 1),
            "control_salary_cost": round(ctrl_sal, 1),
            "avg_salary_cost": round(ctrl_sal / n_fractional, 2) if n_fractional > 0 else 0.0,
            "control_surplus": round(ctrl_surp, 1),
            "years_control": int(round(n_fractional)) if abs(n_fractional - round(n_fractional)) < 1e-6 else round(n_fractional, 1),
            "control_through": int(fa_year - 1) if (not is_fa_player and pd.notna(fa_year) and n_fractional > 1e-6) else None,
            "projected_15yr_war": round(fut_war, 1),
            "projected_15yr_value": round(fut_prod, 1),
            "career_war": round(past_war + ctrl_war, 1),
            "career_production_value": round(past_prod + fut_prod, 1),
            "past_career_war": round(past_war, 1),
            "past_career_value": round(past_prod, 1),
        })

    metrics_df = pd.DataFrame(rows)
    return df.merge(metrics_df, on="IDfg", how="left")


# Alias for transitional callers
add_trade_ranking_metrics = add_control_horizon_metrics
