"""
Dollar Valuation & Arbitration Cost Rules
=========================================

Converts single-season projected WAR into `production_value` (nominal season
dollars) and estimates CBA `Pre-Arb` / `Arb-1..4` `salary_cost` and
`annual_surplus` along a player's contract timeline.

Note:
    Deterministic `salary_cost` and `annual_surplus` provide clean per-year
    baseline numbers for UI tables; actual `trade_value` (and `P10/P50/P90`)
    is priced path-by-path with dynamic arbitration, non-tenders, and options
    inside `monte_carlo_valuation.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, logger


def calculate_production_value(
    war: float,
    year: int,
    player_role: str = "Batter",
    pt_share: float = 1.0,
    season_fraction: float = 1.0,
    **kwargs,
) -> float:
    """Convert single-season WAR into nominal year-specific production dollars."""
    return Config.Market.calculate_production_value(
        war=war,
        year=year,
        player_role=player_role,
        pt_share=pt_share,
        season_fraction=season_fraction,
    )


# Clean transition alias
calculate_war_value = calculate_production_value


def get_min_salary_for_year(year: int, status: str = "Pre-Arb") -> float:
    """
    Return the inflation-adjusted CBA minimum/floor salary for *status* in *year*.

    Uses exact historical MLB minimums for 2014–2026 and scales arbitration
    tier floors proportionally so historical snapshots (2014–2025) never apply
    2026 dollar floors to earlier eras.
    """
    yr = int(year)
    base_year = Config.Contracts.BASE_YEAR  # 2026
    base_prearb = float(Config.Contracts.MIN_SALARY["Pre-Arb"])  # 780_000

    if yr in Config.History.HISTORICAL_MIN_SALARY:
        yr_prearb = float(Config.History.HISTORICAL_MIN_SALARY[yr])
    else:
        yr_prearb = base_prearb * ((1.0 + Config.Contracts.INFLATION_RATE) ** (yr - base_year))

    if status == "Pre-Arb":
        return yr_prearb

    base_tier = float(Config.Contracts.MIN_SALARY.get(status, Config.Contracts.MIN_SALARY["Arb-1"]))
    era_scale = (
        (1.0 + Config.Contracts.INFLATION_RATE) ** (yr - base_year)
        if yr >= base_year
        else (yr_prearb / base_prearb)
    )
    return float(base_tier * era_scale)


def join_predictions_with_timeline(
    extended_timeline: pd.DataFrame,
    player_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join multi-year WAR predictions onto the contract timeline and compute
    single-season `production_value`.
    """
    preds = player_predictions.copy()
    if "prediction_year" not in preds.columns:
        preds["prediction_year"] = preds["Year"]

    # Collapse two-way players to one combined WAR row per (IDfg, prediction_year):
    # sum positive contributions; if one side is sub-replacement (e.g. pitcher batting),
    # keep the primary positive side.
    component_cols = [c for c in ("Bat", "BsR", "Def", "PA", "IP") if c in preds.columns]

    if preds.duplicated(subset=["IDfg", "prediction_year"]).any():
        dup_mask = preds.duplicated(subset=["IDfg", "prediction_year"], keep=False)
        non_dup = preds[~dup_mask][["IDfg", "prediction_year", "WAR"] + component_cols].copy()
        dup_rows = []
        for (pid, pyr), grp in preds[dup_mask].groupby(["IDfg", "prediction_year"]):
            pos_wars = grp.loc[grp["WAR"] > 0, "WAR"]
            combined_war = float(pos_wars.sum()) if len(pos_wars) >= 2 else float(grp["WAR"].max())
            rec = {"IDfg": pid, "prediction_year": pyr, "WAR": combined_war}
            for c in component_cols:
                rec[c] = float(grp[c].fillna(0.0).sum())
            dup_rows.append(rec)
        war_agg = pd.concat([non_dup, pd.DataFrame(dup_rows)], ignore_index=True)
    else:
        war_agg = preds[["IDfg", "prediction_year", "WAR"] + component_cols].copy()

    if "Age" in preds.columns:
        age_agg = preds.groupby(["IDfg", "prediction_year"], as_index=False)["Age"].first()
        war_agg = war_agg.merge(age_agg, on=["IDfg", "prediction_year"], how="left")

    if "Position" in preds.columns:
        pos_agg = preds.groupby(["IDfg", "prediction_year"], as_index=False)["Position"].first()
        war_agg = war_agg.merge(pos_agg, on=["IDfg", "prediction_year"], how="left")

    timeline_with_war = extended_timeline.merge(
        war_agg,
        left_on=["IDfg", "Year"],
        right_on=["IDfg", "prediction_year"],
        how="left",
        suffixes=("", "_pred"),
    )
    if "Age_pred" in timeline_with_war.columns:
        timeline_with_war["Age"] = timeline_with_war.get("Age", pd.Series(index=timeline_with_war.index)).fillna(
            timeline_with_war["Age_pred"]
        )
        timeline_with_war = timeline_with_war.drop(columns=["Age_pred"])

    for col in ("Bat", "BsR", "Def"):
        if col not in timeline_with_war.columns:
            timeline_with_war[col] = 0.0
        else:
            timeline_with_war[col] = timeline_with_war[col].fillna(0.0)

    role_col = "player_role" if "player_role" in timeline_with_war.columns else "position_group"
    roles = timeline_with_war.get(role_col, pd.Series("Batter", index=timeline_with_war.index)).fillna("Batter")
    wars = pd.to_numeric(timeline_with_war["WAR"], errors="coerce")
    years = pd.to_numeric(timeline_with_war["Year"], errors="coerce").fillna(Config.Market.BASE_YEAR).astype(int)
    pa_col = "PA" if "PA" in timeline_with_war.columns else ("PA_pred" if "PA_pred" in timeline_with_war.columns else None)
    ip_col = "IP" if "IP" in timeline_with_war.columns else ("IP_pred" if "IP_pred" in timeline_with_war.columns else None)
    pas = pd.to_numeric(timeline_with_war[pa_col], errors="coerce") if pa_col else pd.Series(np.nan, index=timeline_with_war.index)
    ips = pd.to_numeric(timeline_with_war[ip_col], errors="coerce") if ip_col else pd.Series(np.nan, index=timeline_with_war.index)

    prod_vals = []
    for w, y, r, pa_v, ip_v in zip(wars, years, roles, pas, ips):
        r_str = str(r)
        if r_str == "SP":
            pt_sh = float(np.clip((165.0 if pd.isna(ip_v) else float(ip_v)) / 165.0, 0.45, 1.0))
        elif r_str == "RP":
            pt_sh = float(np.clip((62.0 if pd.isna(ip_v) else float(ip_v)) / 62.0, 0.65, 1.0))
        else:
            pt_sh = float(np.clip((600.0 if pd.isna(pa_v) else float(pa_v)) / 600.0, 0.45, 1.0))
        prod_vals.append(calculate_production_value(w, y, player_role=r_str, pt_share=pt_sh))
    timeline_with_war["production_value"] = prod_vals

    for extra_col in ("prediction_year", "PA_pred", "IP_pred"):
        if extra_col in timeline_with_war.columns:
            timeline_with_war = timeline_with_war.drop(columns=[extra_col])

    logger.info(
        "Joined predictions onto timeline: %d rows (mean production_value=$%.2fM)",
        len(timeline_with_war),
        timeline_with_war["production_value"].mean() / 1e6,
    )
    return timeline_with_war


def calculate_salary_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate deterministic single-season `salary_cost` along the contract timeline.

    Uses explicit `Payroll` when present, otherwise applies era-appropriate
    `Pre-Arb` minimum and `Arb-1..4` production-sharing formulas.
    """
    result = df.copy().sort_values(["IDfg", "Year"])
    result["salary_cost"] = np.nan

    arb_pct_map = Config.Contracts.ARB_PERCENT

    for player_id, player_data in result.groupby("IDfg", sort=False):
        prev_salary = 0.0
        prev_arb_pct = 1.0

        for idx, row in player_data.iterrows():
            prod_val = float(row.get("production_value", 0.0) or 0.0)
            status = str(row.get("Normalized_Status") or row.get("Status") or "")
            yr = int(row["Year"])

            if pd.notna(row.get("Payroll")):
                cost = float(row["Payroll"])
            elif status == "Pre-Arb":
                cost = get_min_salary_for_year(yr, "Pre-Arb")
                prev_arb_pct = 1.0
            elif status in arb_pct_map:
                min_sal = get_min_salary_for_year(yr, status)
                arb_pct = arb_pct_map[status]
                if prod_val > 0:
                    step_up = prev_salary * (arb_pct / prev_arb_pct) * 1.05 if prev_salary > 0 else 0.0
                    cost = max(min_sal, prod_val * arb_pct, step_up)
                else:
                    cost = min_sal
                prev_arb_pct = arb_pct
            else:
                cost = np.nan

            result.at[idx, "salary_cost"] = cost
            if pd.notna(cost):
                prev_salary = float(cost)

    logger.info(
        "Calculated salary_cost for %d/%d timeline rows",
        int(result["salary_cost"].notna().sum()),
        len(result),
    )
    return result


def calculate_annual_surplus(df: pd.DataFrame) -> pd.DataFrame:
    """Compute `annual_surplus = production_value - salary_cost` where controlled."""
    result = df.copy()
    if "salary_cost" not in result.columns:
        raise ValueError("salary_cost column missing from DataFrame")

    result["annual_surplus"] = np.where(
        result["salary_cost"].notna(),
        result["production_value"] - result["salary_cost"],
        np.nan,
    )
    return result
