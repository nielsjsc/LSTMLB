"""
Prediction & Historical Data Loader
===================================

Loads Marcel prediction CSVs (`pitcher_predictions.csv`, `batter_predictions.csv`,
`baserunning_predictions.csv`, `fielding_predictions.csv`), current-year Statcast
expected stats, Spotrac salary data, and historical MLB leaderboards.

Assigns canonical `player_role` in `{'Batter', 'SP', 'RP'}` at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import CURRENT_YEAR, Config, logger

PIPELINE_DIR = Config.Paths.PIPELINE_DIR
PREDICTION_YEARS = Config.Pipeline.PREDICTION_YEARS
REQUIRED_COLUMNS = Config.Columns.REQUIRED
SALARY_DIR = Config.Paths.SALARY_DIR
HISTORIC_MLB_DIR = Config.Paths.HISTORIC_MLB_DIR


def validate_files_exist(directory: Path, filename: str) -> None:
    if not (directory / filename).exists():
        raise FileNotFoundError(f"Missing file: {filename} in {directory}")


def load_prediction_files(
    pipeline_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load consolidated prediction CSVs and raw Spotrac salary data.

    Returns:
        `(sp_data, rp_data, batter_data, baserunning_data, fielding_data, salary_data)`
    """
    src_dir = Path(pipeline_dir) if pipeline_dir else PIPELINE_DIR
    for fname in (
        "pitcher_predictions.csv",
        "batter_predictions.csv",
        "baserunning_predictions.csv",
        "fielding_predictions.csv",
    ):
        validate_files_exist(src_dir, fname)

    pitcher_df = pd.read_csv(src_dir / "pitcher_predictions.csv", low_memory=False)
    missing_pit = set(REQUIRED_COLUMNS["pitcher_predictions"]) - set(pitcher_df.columns)
    if missing_pit:
        raise ValueError(f"Missing columns in pitcher_predictions.csv: {missing_pit}")

    if "Season" in pitcher_df.columns and "Year" not in pitcher_df.columns:
        pitcher_df["Year"] = pitcher_df["Season"]
    pitcher_df["prediction_year"] = pitcher_df["Year"]

    sp_data = pitcher_df[pitcher_df["Role"] == "SP"].copy()
    rp_data = pitcher_df[pitcher_df["Role"] == "RP"].copy()

    sp_data["player_role"] = "SP"
    rp_data["player_role"] = "RP"
    sp_data["position_group"] = "SP"
    rp_data["position_group"] = "RP"
    sp_data["Position"] = "SP"
    rp_data["Position"] = "RP"

    batter_data = pd.read_csv(src_dir / "batter_predictions.csv", low_memory=False)
    baserunning_data = pd.read_csv(src_dir / "baserunning_predictions.csv", low_memory=False)
    fielding_data = pd.read_csv(src_dir / "fielding_predictions.csv", low_memory=False)

    core_batter_cols = {"Name", "IDfg", "Year", "Age", "wOBA", "BB%", "K%", "AVG", "OBP", "SLG"}
    missing_bat = core_batter_cols - set(batter_data.columns)
    if missing_bat:
        raise ValueError(f"Missing columns in batter_predictions.csv: {missing_bat}")

    # Merge current-year Statcast expected stats if present
    current_year_data = batter_data[batter_data["Year"] == CURRENT_YEAR]
    if not current_year_data.empty:
        expected_file = (
            Config.Paths.ROOT_DIR
            / "data"
            / "statcast"
            / f"statcast_batter_expected_stats_{CURRENT_YEAR}_{CURRENT_YEAR}.csv"
        )
        if expected_file.exists() and "IDfg" in pd.read_csv(expected_file, nrows=0).columns:
            try:
                sc = pd.read_csv(expected_file, low_memory=False)
                if "year" in sc.columns and "Year" not in sc.columns:
                    sc = sc.rename(columns={"year": "Year"})
                keep = ["IDfg", "Year"] + [
                    c for c in sc.columns
                    if c.lower().startswith(("x", "sc_est", "est_"))
                ]
                sc = sc[[c for c in dict.fromkeys(keep) if c in sc.columns]].drop_duplicates(["IDfg", "Year"])
                batter_data = batter_data.merge(sc, on=["IDfg", "Year"], how="left")
            except Exception as exc:
                logger.warning("Could not merge current-year Statcast expected stats: %s", exc)

    batter_data["prediction_year"] = batter_data["Year"]
    batter_data["player_role"] = "Batter"
    batter_data["position_group"] = "Batter"

    salary_data = pd.read_csv(SALARY_DIR / "mlb_salary_data.csv", low_memory=False)

    logger.info(
        "Loaded predictions: %d SP rows, %d RP rows, %d Batter rows",
        len(sp_data),
        len(rp_data),
        len(batter_data),
    )
    return sp_data, rp_data, batter_data, baserunning_data, fielding_data, salary_data


def merge_prediction_data(
    sp_df: pd.DataFrame,
    rp_df: pd.DataFrame,
    batter_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine SP, RP, and Batter prediction DataFrames into one table."""
    required_cols = ["Name", "IDfg", "player_role", "Age", "prediction_year", "WAR", "Position"]
    stat_cols = [
        "position_group", "BsR", "Def", "Bat", "G", "PA", "BB%", "K%", "AVG", "OBP", "SLG",
        "wOBA", "wRC+", "HR", "2B", "3B", "RBI", "R", "SB", "CS", "HBP", "SF",
        "ERA", "FIP", "SIERA", "IP", "GS", "Role",
    ]
    all_cols = set(required_cols + stat_cols)

    frames = []
    for df, default_role in ((sp_df, "SP"), (rp_df, "RP"), (batter_df, "Batter")):
        if df.empty:
            continue
        tmp = df.copy()
        if "player_role" not in tmp.columns:
            tmp["player_role"] = default_role
        frames.append(tmp[[c for c in tmp.columns if c in all_cols]])

    player_predictions = pd.concat(frames, ignore_index=True)
    if "Position" not in player_predictions.columns:
        raise ValueError("Position column missing from merged predictions")

    logger.info("Merged %d total player-year predictions", len(player_predictions))
    return player_predictions


def load_historical_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical MLB batting and pitching leaderboards (preferring `_with_statcast`)."""
    batting_file = HISTORIC_MLB_DIR / "mlb_batting_data_1950_2025_with_statcast.csv"
    pitching_file = HISTORIC_MLB_DIR / "mlb_pitching_data_1950_2025_with_statcast.csv"

    if not batting_file.exists():
        batting_file = HISTORIC_MLB_DIR / "mlb_batting_data_1950_2025.csv"
    if not pitching_file.exists():
        pitching_file = HISTORIC_MLB_DIR / "mlb_pitching_data_1950_2025.csv"

    batting_history = pd.read_csv(batting_file, low_memory=False)
    pitching_history = pd.read_csv(pitching_file, low_memory=False)

    if "HR%" not in pitching_history.columns and {"HR", "TBF"}.issubset(pitching_history.columns):
        pitching_history["HR%"] = pitching_history["HR"] / pitching_history["TBF"].replace(0, float("nan"))

    logger.info(
        "Loaded historical MLB data: %d batting rows, %d pitching rows",
        len(batting_history),
        len(pitching_history),
    )
    return batting_history, pitching_history
