"""
Canonical Value Data Exporter
=============================

Exports player valuation timelines in a single, self-documenting `snake_case`
CSV schema shared identically across CSV outputs, SQLAlchemy models, FastAPI
responses, and TypeScript interfaces — eliminating mixed-case rename churn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from core.id_crosswalk import fg_to_mlbam, normalize_team_code
from .config import logger

# Map internal upstream stat/metadata columns to canonical snake_case CSV columns
_EXPORT_RENAME_MAP = {
    "IDfg": "fg_id",
    "Name": "player_name",
    "Player_Name": "player_name",
    "Player Name": "player_name",
    "Team": "team",
    "Position": "position",
    "Status": "status",
    "Age": "age",
    "Year": "year",
    "Years_of_Service": "years_of_service",
    "Spotrac_Salary": "spotrac_salary",
    "WAR": "war",
    "FA_Year": "fa_year",
    "Probable_FA_Year": "probable_fa_year",
    "Earliest_FA_Year": "earliest_fa_year",
    # Hitting
    "G_bat": "g_bat",
    "PA": "pa",
    "BB%_bat": "bb_pct_bat",
    "K%_bat": "k_pct_bat",
    "BB_bat": "bb_bat",
    "K_bat": "k_bat",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "wOBA": "woba",
    "wRC+": "wrc_plus",
    "Bat": "bat",
    "BsR": "bsr",
    "Def": "def_value",
    "WAR_batter": "war_bat",
    "HR": "hr",
    "2B": "doubles",
    "3B": "triples",
    "R": "r",
    "RBI": "rbi",
    "SB": "sb",
    "CS": "cs",
    "xBA": "xba",
    "xSLG": "xslg",
    "xwOBA": "xwoba",
    "sc_est_ba": "xba",
    "sc_est_slg": "xslg",
    "sc_est_woba": "xwoba",
    # Pitching
    "G_pit": "g_pit",
    "GS": "gs",
    "IP": "ip",
    "ERA": "era",
    "FIP": "fip",
    "SIERA": "siera",
    "K%_pit": "k_pct_pit",
    "BB%_pit": "bb_pct_pit",
    "HR/FB_pit": "hr_fb",
    "FB%_pit": "fb_pct",
    "GB%_pit": "gb_pct",
    "K/9": "k_9",
    "BB/9": "bb_9",
    "HR/9": "hr_9",
    "K_pit": "k_pit",
    "BB_pit": "bb_pit",
    "ER_pit": "er_pit",
    "WAR_pitcher": "war_pit",
}

CANONICAL_COLUMN_ORDER = [
    # Identity & Contract
    "fg_id", "mlbam_id", "player_name", "team", "position", "player_role",
    "year", "age", "status", "years_of_service", "spotrac_salary",
    # Control & FA Horizon
    "fa_year", "probable_fa_year", "earliest_fa_year", "years_control", "control_through",
    # Single-Season Production & Surplus
    "war", "production_value", "salary_cost", "annual_surplus",
    # Monte Carlo Trade Valuation & Horizon Aggregates
    "trade_value", "trade_value_p10", "trade_value_p50", "trade_value_p90",
    "control_war", "avg_war",
    "control_production_value", "control_salary_cost", "avg_salary_cost", "control_surplus",
    "projected_15yr_war", "projected_15yr_value",
    "career_war", "career_production_value",
    "past_career_war", "past_career_value",
    # Hitting Stats
    "g_bat", "pa", "bb_pct_bat", "k_pct_bat", "bb_bat", "k_bat",
    "avg", "obp", "slg", "ops", "woba", "wrc_plus",
    "bat", "bsr", "def_value", "war_bat",
    "hr", "doubles", "triples", "r", "rbi", "sb", "cs",
    "xba", "xslg", "xwoba", "sc_ba", "sc_slg", "sc_woba",
    # Pitching Stats
    "g_pit", "gs", "ip", "era", "fip", "siera",
    "k_pct_pit", "bb_pct_pit", "hr_fb", "fb_pct", "gb_pct",
    "k_9", "bb_9", "hr_9", "k_pit", "bb_pit", "er_pit", "war_pit",
]


def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    collapsed = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in collapsed.columns:
            continue
        cols = df.columns[df.columns == col]
        if len(cols) == 1:
            collapsed[col] = df[col]
        else:
            collapsed[col] = df.loc[:, cols].bfill(axis=1).iloc[:, 0]
    return collapsed


def export_value_data(
    df: pd.DataFrame,
    output_dir: Path,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Format and export player valuation data to `output_dir / filename` using
    canonical `snake_case` columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = filename or "player_values_complete.csv"

    export_df = df.copy().rename(columns=_EXPORT_RENAME_MAP)
    export_df = _collapse_duplicate_columns(export_df)

    # Ensure fg_id & mlbam_id are clean integer-backed columns
    export_df["fg_id"] = pd.to_numeric(export_df["fg_id"], errors="coerce")
    export_df = export_df.dropna(subset=["fg_id"]).copy()
    export_df["fg_id"] = export_df["fg_id"].astype(int)

    if "mlbam_id" not in export_df.columns:
        export_df["mlbam_id"] = export_df["fg_id"].map(fg_to_mlbam)
    else:
        missing_mlb = export_df["mlbam_id"].isna()
        if missing_mlb.any():
            export_df.loc[missing_mlb, "mlbam_id"] = export_df.loc[missing_mlb, "fg_id"].map(fg_to_mlbam)
    export_df["mlbam_id"] = pd.to_numeric(export_df["mlbam_id"], errors="coerce").astype("Int64")

    export_df["team"] = export_df["team"].map(normalize_team_code)

    if "player_role" not in export_df.columns:
        export_df["player_role"] = export_df.get("position_group", "Batter")
    export_df["player_role"] = export_df["player_role"].replace(
        {"POS": "Batter", "batter": "Batter", "hitter": "Batter"}
    )

    for col in ("production_value", "salary_cost", "annual_surplus"):
        if col in export_df.columns:
            export_df[col] = pd.to_numeric(export_df[col], errors="coerce").round(2)
    if "war" in export_df.columns:
        export_df["war"] = pd.to_numeric(export_df["war"], errors="coerce").round(2)

    ordered_cols = [c for c in CANONICAL_COLUMN_ORDER if c in export_df.columns]
    export_df = export_df[ordered_cols].sort_values(
        ["year", "team", "war"],
        ascending=[True, True, False],
    )

    output_file = output_dir / output_filename
    export_df.to_csv(output_file, index=False, na_rep="", encoding="utf-8")
    logger.info(
        "Exported %d canonical player-year records (%d unique players, %d with mlbam_id) to %s",
        len(export_df),
        export_df["fg_id"].nunique(),
        int(export_df["mlbam_id"].notna().sum()),
        output_file,
    )
    return export_df
