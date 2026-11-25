"""
Player Timeline Assembler
=========================

Integrates historical pre-`CURRENT_YEAR` MLB seasons with future projected
seasons, merges full hitting and pitching rate/counting stats, reconciles
two-way players, resolves `mlbam_id` via `core.id_crosswalk`, and normalizes
team abbreviations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.id_crosswalk import fg_to_mlbam, normalize_team_code
from core.name_utils import name_key as _name_key_fn
from .config import CURRENT_YEAR, Config, HITTER_COLUMNS, PITCHER_COLUMNS, logger
from .dollar_valuation import calculate_production_value

_historical_salary_cache: Optional[dict[tuple[str, int], float]] = None


def _load_historical_salary() -> dict[tuple[str, int], float]:
    """Load `(normalized_name, year) -> salary` from `universal_salary.csv`."""
    global _historical_salary_cache
    if _historical_salary_cache is not None:
        return _historical_salary_cache

    lookup: dict[tuple[str, int], float] = {}
    sal_file = Config.Paths.UNIVERSAL_SALARY_FILE
    if not sal_file.exists():
        _historical_salary_cache = lookup
        return lookup

    try:
        df = pd.read_csv(sal_file, encoding="utf-8", low_memory=False)
        for _, row in df.iterrows():
            name = _name_key_fn(str(row.get("player", "")))
            yr = row.get("year")
            sal = row.get("salary")
            if not name or pd.isna(yr) or pd.isna(sal):
                continue
            try:
                sal_f = float(sal)
            except (TypeError, ValueError):
                continue
            if sal_f > 0:
                lookup[(name, int(yr))] = sal_f
    except Exception as exc:
        logger.warning("Could not read universal_salary.csv: %s", exc)

    _historical_salary_cache = lookup
    return lookup


def integrate_historical_stats(
    timeline_df: pd.DataFrame,
    batting_history: pd.DataFrame,
    pitching_history: pd.DataFrame,
    current_year: int = CURRENT_YEAR,
) -> pd.DataFrame:
    """
    Attach accrued historical MLB seasons (`Season < current_year`) to the
    player timeline with `production_value`, `salary_cost`, and `annual_surplus`.
    """
    role_col = "player_role" if "player_role" in timeline_df.columns else "position_group"
    current_players = timeline_df[["IDfg", "Name", role_col]].drop_duplicates(subset=["IDfg"]).copy()
    current_players = current_players.rename(columns={role_col: "player_role"})

    batter_cols = [
        "IDfg", "Season", "Name", "Team", "G", "PA", "WAR", "BB%", "K%", "AVG",
        "OBP", "SLG", "OPS", "wOBA", "wRC+", "Bat", "BsR", "Def", "Age",
        "HR", "2B", "3B", "R", "RBI", "SB", "CS",
        "xBA", "xSLG", "xwOBA", "sc_est_ba", "sc_est_slg", "sc_est_woba",
        "sc_ba", "sc_slg", "sc_woba",
    ]
    avail_bat = [c for c in batter_cols if c in batting_history.columns]
    batting_filtered = (
        batting_history[
            batting_history["IDfg"].isin(current_players["IDfg"])
            & (batting_history["Season"] < current_year)
        ][avail_bat]
        .rename(columns={
            "Season": "Year",
            "WAR": "WAR_batter",
            "BB%": "BB%_bat",
            "K%": "K%_bat",
            "G": "G_bat",
        })
    )

    pitcher_cols = [
        "IDfg", "GS", "Season", "Name", "Team", "G", "WAR", "ERA", "FIP", "SIERA", "IP",
        "K%", "BB%", "Age", "HR/FB", "FB%", "GB%", "K/9", "BB/9", "HR/9",
    ]
    avail_pit = [c for c in pitcher_cols if c in pitching_history.columns]
    pitching_filtered = (
        pitching_history[
            pitching_history["IDfg"].isin(current_players["IDfg"])
            & (pitching_history["Season"] < current_year)
        ][avail_pit]
        .rename(columns={
            "Season": "Year",
            "WAR": "WAR_pitcher",
            "K%": "K%_pit",
            "BB%": "BB%_pit",
            "G": "G_pit",
            "HR/FB": "HR/FB_pit",
            "FB%": "FB%_pit",
            "GB%": "GB%_pit",
        })
    )

    historical = batting_filtered.merge(
        pitching_filtered,
        on=["IDfg", "Year"],
        how="outer",
        suffixes=("_bat", "_pit"),
    )

    for col in ("Name", "Team", "Age"):
        c_bat, c_pit = f"{col}_bat", f"{col}_pit"
        if c_bat in historical.columns and c_pit in historical.columns:
            historical[col] = historical[c_bat].fillna(historical[c_pit])
            historical = historical.drop(columns=[c_bat, c_pit])

    historical = historical.merge(current_players[["IDfg", "player_role"]], on="IDfg", how="left")
    historical["WAR_batter"] = historical.get("WAR_batter", pd.Series(0.0, index=historical.index)).fillna(0.0)
    historical["WAR_pitcher"] = historical.get("WAR_pitcher", pd.Series(0.0, index=historical.index)).fillna(0.0)
    historical["WAR"] = historical["WAR_batter"] + historical["WAR_pitcher"]

    historical["Status"] = "NA"
    historical["Normalized_Status"] = "NA"
    historical["Payroll"] = np.nan
    historical["Team"] = historical["Team"].map(normalize_team_code)

    historical["production_value"] = [
        calculate_production_value(w, int(y), player_role=r)
        for w, y, r in zip(
            historical["WAR"],
            historical["Year"],
            historical["player_role"].fillna("Batter"),
        )
    ]

    salary_lookup = _load_historical_salary()
    historical["salary_cost"] = [
        salary_lookup.get((_name_key_fn(str(nm)), int(yr)), np.nan)
        for nm, yr in zip(historical.get("Name", ""), historical["Year"])
    ]
    historical["annual_surplus"] = np.where(
        historical["salary_cost"].notna(),
        historical["production_value"] - historical["salary_cost"],
        np.nan,
    )

    # Preserve Spotrac salary_cost on any overlapping pre-current_year rows
    timeline_salary = (
        timeline_df[timeline_df["salary_cost"].notna()]
        .set_index(["IDfg", "Year"])["salary_cost"]
        .to_dict()
    )

    complete = pd.concat([timeline_df, historical], ignore_index=True)
    complete = (
        complete.sort_values(["IDfg", "Year"])
        .drop_duplicates(subset=["IDfg", "Year"], keep="last")
        .reset_index(drop=True)
    )

    if timeline_salary:
        for (idfg, yr), sal in timeline_salary.items():
            mask = (complete["IDfg"] == idfg) & (complete["Year"] == yr) & complete["salary_cost"].isna()
            if mask.any():
                complete.loc[mask, "salary_cost"] = sal
                complete.loc[mask, "annual_surplus"] = complete.loc[mask, "production_value"] - sal

    logger.info("Integrated historical MLB stats: %d total player-year rows", len(complete))
    return complete


def assemble_player_timeline(
    value_data: pd.DataFrame,
    batter_data: pd.DataFrame,
    sp_data: pd.DataFrame,
    rp_data: pd.DataFrame,
    current_year: int = CURRENT_YEAR,
) -> pd.DataFrame:
    """
    Merge projected hitting and pitching stat lines onto the timeline,
    reconcile two-way players, attach `mlbam_id`, and finalize clean columns.
    """
    historical_data = value_data[value_data["Year"] < current_year].copy()
    prediction_data = value_data[value_data["Year"] >= current_year].copy()

    if "Payroll" in prediction_data.columns:
        prediction_data["spotrac_salary"] = prediction_data["Payroll"]

    role_col = "player_role" if "player_role" in prediction_data.columns else "position_group"
    if role_col != "player_role" and role_col in prediction_data.columns:
        prediction_data["player_role"] = prediction_data[role_col]
    prediction_data["player_role"] = (
        prediction_data.get("player_role", pd.Series("Batter", index=prediction_data.index))
        .replace({"POS": "Batter", "batter": "Batter", "hitter": "Batter"})
        .fillna("Batter")
    )

    essential_cols = [
        "Name", "IDfg", "player_role", "Year", "Team",
        "Payroll", "spotrac_salary", "Years_of_Service",
        "Status", "Normalized_Status", "WAR", "production_value",
        "salary_cost", "annual_surplus",
    ]
    avail_essential = [c for c in essential_cols if c in prediction_data.columns]
    prediction_data = prediction_data[avail_essential].copy()

    batter_ids = set(batter_data["IDfg"].unique()) if not batter_data.empty else set()
    pitcher_ids = (
        (set(sp_data["IDfg"].unique()) if not sp_data.empty else set())
        | (set(rp_data["IDfg"].unique()) if not rp_data.empty else set())
    )
    two_way_players = batter_ids & pitcher_ids
    prediction_data["Two_Way"] = prediction_data["IDfg"].isin(two_way_players)

    # Batter stat line
    batter_stat_cols = ["IDfg", "prediction_year", "WAR", "Position"] + [
        c for c in HITTER_COLUMNS if c not in ("Name", "IDfg", "WAR", "Position")
    ] + ["xBA", "xSLG", "xwOBA", "sc_est_ba", "sc_est_slg", "sc_est_woba", "sc_ba", "sc_slg", "sc_woba"]
    avail_bat = [c for c in dict.fromkeys(batter_stat_cols) if c in batter_data.columns]
    batter_stats = batter_data[avail_bat].rename(columns={
        "prediction_year": "Year",
        "BB%": "BB%_bat",
        "K%": "K%_bat",
        "G": "G_bat",
        "Age": "Age_bat",
        "WAR": "WAR_batter",
        "Position": "Position_batter",
    }).drop_duplicates(subset=["IDfg", "Year"])

    # Pitcher stat line
    pitcher_stat_cols = ["IDfg", "prediction_year", "WAR", "Position"] + [
        c for c in PITCHER_COLUMNS if c not in ("Name", "IDfg", "WAR", "Position")
    ]
    sp_avail = [c for c in dict.fromkeys(pitcher_stat_cols) if c in sp_data.columns]
    rp_avail = [c for c in dict.fromkeys(pitcher_stat_cols) if c in rp_data.columns]
    pitcher_stats = (
        pd.concat([sp_data[sp_avail], rp_data[rp_avail]], ignore_index=True)
        .rename(columns={
            "prediction_year": "Year",
            "BB%": "BB%_pit",
            "K%": "K%_pit",
            "HR/FB": "HR/FB_pit",
            "FB%": "FB%_pit",
            "GB%": "GB%_pit",
            "Age": "Age_pit",
            "G": "G_pit",
            "WAR": "WAR_pitcher",
            "Position": "Position_pitcher",
        })
        .drop_duplicates(subset=["IDfg", "Year"])
    )

    prediction_data = prediction_data.merge(batter_stats, on=["IDfg", "Year"], how="left")
    prediction_data = prediction_data.merge(pitcher_stats, on=["IDfg", "Year"], how="left")

    mask_tw = prediction_data["Two_Way"]
    if "Position_batter" in prediction_data.columns and "Position_pitcher" in prediction_data.columns:
        prediction_data.loc[mask_tw, "Position"] = prediction_data.loc[mask_tw].apply(
            lambda x: f"{x['Position_pitcher']}/{x['Position_batter']}"
            if pd.notna(x["Position_pitcher"]) and pd.notna(x["Position_batter"])
            else (x["Position_batter"] if pd.notna(x["Position_batter"]) else x["Position_pitcher"]),
            axis=1,
        )
        prediction_data.loc[~mask_tw, "Position"] = prediction_data.loc[~mask_tw, "Position_batter"].fillna(
            prediction_data.loc[~mask_tw, "Position_pitcher"]
        )
        prediction_data = prediction_data.drop(columns=["Position_batter", "Position_pitcher"], errors="ignore")
    elif "Position_batter" in prediction_data.columns:
        prediction_data["Position"] = prediction_data["Position_batter"]
        prediction_data = prediction_data.drop(columns=["Position_batter"], errors="ignore")
    elif "Position_pitcher" in prediction_data.columns:
        prediction_data["Position"] = prediction_data["Position_pitcher"]
        prediction_data = prediction_data.drop(columns=["Position_pitcher"], errors="ignore")

    # Reconcile WAR & production_value for two-way players
    if "WAR_batter" in prediction_data.columns and "WAR_pitcher" in prediction_data.columns:
        prediction_data.loc[mask_tw, "WAR"] = (
            prediction_data.loc[mask_tw, "WAR_batter"].fillna(0.0).clip(lower=0.0)
            + prediction_data.loc[mask_tw, "WAR_pitcher"].fillna(0.0).clip(lower=0.0)
        )
        prediction_data.loc[~mask_tw, "WAR"] = prediction_data.loc[~mask_tw, "WAR_batter"].fillna(
            prediction_data.loc[~mask_tw, "WAR_pitcher"]
        )
        if mask_tw.any():
            prediction_data.loc[mask_tw, "production_value"] = [
                calculate_production_value(w, int(y), player_role=r)
                for w, y, r in zip(
                    prediction_data.loc[mask_tw, "WAR"],
                    prediction_data.loc[mask_tw, "Year"],
                    prediction_data.loc[mask_tw, "player_role"],
                )
            ]
            has_sal = mask_tw & prediction_data["salary_cost"].notna()
            prediction_data.loc[has_sal, "annual_surplus"] = (
                prediction_data.loc[has_sal, "production_value"] - prediction_data.loc[has_sal, "salary_cost"]
            )
    elif "WAR_batter" in prediction_data.columns:
        prediction_data["WAR"] = prediction_data["WAR_batter"]
    elif "WAR_pitcher" in prediction_data.columns:
        prediction_data["WAR"] = prediction_data["WAR_pitcher"]

    # Reconcile Age
    if "Age_bat" in prediction_data.columns and "Age_pit" in prediction_data.columns:
        prediction_data["Age"] = prediction_data["Age_bat"].fillna(prediction_data["Age_pit"])
        prediction_data = prediction_data.drop(columns=["Age_bat", "Age_pit"], errors="ignore")
    elif "Age_bat" in prediction_data.columns:
        prediction_data["Age"] = prediction_data["Age_bat"]
        prediction_data = prediction_data.drop(columns=["Age_bat"], errors="ignore")
    elif "Age_pit" in prediction_data.columns:
        prediction_data["Age"] = prediction_data["Age_pit"]
        prediction_data = prediction_data.drop(columns=["Age_pit"], errors="ignore")

    result = pd.concat([historical_data, prediction_data], ignore_index=True)
    result = result.sort_values(["IDfg", "Year"]).reset_index(drop=True)

    # Backfill Age on salary-extended post-horizon rows (e.g. 2040 FA rows)
    for pid, grp in result.groupby("IDfg", sort=False):
        valid_age = grp.dropna(subset=["Age", "Year"])
        if not valid_age.empty and grp["Age"].isna().any():
            base_age = float(valid_age.iloc[0]["Age"])
            base_yr = int(valid_age.iloc[0]["Year"])
            miss_idx = grp.index[grp["Age"].isna()]
            result.loc[miss_idx, "Age"] = base_age + (result.loc[miss_idx, "Year"] - base_yr)

    if "Position" in result.columns:
        result["Position"] = result.groupby("IDfg")["Position"].bfill().ffill()

    # Replace raw Status with Normalized_Status and drop temporary columns
    if "Normalized_Status" in result.columns:
        result["Status"] = result["Normalized_Status"]
        result = result.drop(columns=["Normalized_Status"], errors="ignore")
    result = result.drop(columns=["Payroll"], errors="ignore")

    if "OBP" in result.columns and "SLG" in result.columns:
        result["OPS"] = (result["OBP"] + result["SLG"]).round(3)

    result["Team"] = result["Team"].map(normalize_team_code)
    result["mlbam_id"] = result["IDfg"].map(lambda x: fg_to_mlbam(int(x)) if pd.notna(x) else None)

    return result
