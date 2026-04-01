"""
Phase 1b: Merge Current-Season Data into Historic Files
========================================================

Takes freshly-scraped current-season CSVs (from ingest.py) and merges
them into the canonical historic files in data/historic_mlb/.

Strategy:
  1. Load the historic base file (e.g. mlb_batting_data_1950_2025.csv)
  2. Remove any existing rows for the current season
  3. Append the fresh current-season rows
  4. Save back to the same path (keeps all hardcoded references working)
  5. Merge current-season statcast files into the cumulative statcast files
  6. Re-run join_statcast_data.py to regenerate the _with_statcast variants
"""

import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]  # LSTMLB/
DATA_DIR = ROOT_DIR / 'data'
CURRENT_SEASON_DIR = DATA_DIR / 'current_season'
HISTORIC_DIR = DATA_DIR / 'historic_mlb'
STATCAST_DIR = DATA_DIR / 'statcast'

# ── Historic base files ─────────────────────────────────────────────────────

HISTORIC_FILES = {
    'batting': {
        'historic': HISTORIC_DIR / 'mlb_batting_data_1950_2025.csv',
        'current_pattern': 'mlb_batting_data_{year}_{year}.csv',
        'season_col': 'Season',
        'id_col': 'IDfg',
    },
    'pitching': {
        'historic': HISTORIC_DIR / 'mlb_pitching_data_1950_2025.csv',
        'current_pattern': 'mlb_pitching_data_{year}_{year}.csv',
        'season_col': 'Season',
        'id_col': 'IDfg',
    },
    'fielding': {
        'historic': HISTORIC_DIR / 'mlb_fielding_data_2000_2025.csv',
        'current_pattern': 'mlb_fielding_data_{year}_{year}.csv',
        'season_col': 'Season',
        'id_col': 'IDfg',
    },
}

# ── Statcast cumulative files ───────────────────────────────────────────────
# Map: statcast type -> (cumulative filename stem, year column name)
STATCAST_FILES = {
    'batter_exitvelo': ('statcast_batter_exitvelo_barrels', 'year'),
    'batter_expected': ('statcast_batter_expected_stats', 'year'),
    'sprint_speed': ('statcast_sprint_speed', 'year'),
    'pitcher_exitvelo': ('statcast_pitcher_exitvelo_barrels', 'year'),
    'pitcher_expected': ('statcast_pitcher_expected_stats', 'year'),
    'pitcher_arsenal_speed': ('statcast_pitcher_arsenal_speed', 'year'),
    'pitcher_arsenal_spin': ('statcast_pitcher_arsenal_spin', 'year'),
    'fielding_run_value': ('statcast_fielding_run_value', 'year'),
}

# Cumulative statcast files span 2015-2025
STATCAST_YEAR_RANGE = '2015_2025'


def _backup(path: Path) -> Path:
    """Create a timestamped backup of a file before overwriting."""
    if not path.exists():
        return path
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = path.with_suffix(f'.backup_{ts}.csv')
    shutil.copy2(path, backup)
    logger.info(f"  Backup: {backup.name}")
    return backup


def merge_standard_stats(year: int, backup: bool = True) -> dict[str, int]:
    """
    Merge current-season batting/pitching/fielding into the historic base files.

    For each type:
      1. Read historic file
      2. Drop any rows where Season == year
      3. Read current-season file from data/
      4. Append and save

    Returns dict mapping type name -> number of new rows merged.
    """
    results = {}

    for name, cfg in HISTORIC_FILES.items():
        current_path = CURRENT_SEASON_DIR / cfg['current_pattern'].format(year=year)
        historic_path = cfg['historic']

        if not current_path.exists():
            logger.warning(f"  {name}: current-season file not found ({current_path.name}), skipping")
            continue

        if not historic_path.exists():
            logger.error(f"  {name}: historic file not found ({historic_path.name}), skipping")
            continue

        logger.info(f"  Merging {name} for {year} ...")

        # Load
        current_df = pd.read_csv(current_path)
        historic_df = pd.read_csv(historic_path, low_memory=False)

        season_col = cfg['season_col']

        # Validate season col
        if season_col not in current_df.columns:
            logger.error(f"  {name}: '{season_col}' column missing from current-season file")
            continue

        # Ensure current data has the right season
        current_df[season_col] = year

        if backup:
            _backup(historic_path)

        # Drop old rows for this season, then append new
        before_len = len(historic_df)
        historic_df = historic_df[historic_df[season_col] != year]
        dropped = before_len - len(historic_df)

        # Align columns — current-season may have new or missing cols
        # Use the union of both column sets; missing values will be NaN
        all_cols = list(historic_df.columns)
        for col in current_df.columns:
            if col not in all_cols:
                all_cols.append(col)
                logger.info(f"    New column in {year} data: {col}")

        merged = pd.concat([historic_df, current_df], ignore_index=True, sort=False)

        merged.to_csv(historic_path, index=False)
        n_new = len(current_df)
        logger.info(
            f"  ✓ {name}: dropped {dropped} old rows, added {n_new} new rows "
            f"(total {len(merged)})"
        )
        results[name] = n_new

    return results


def merge_statcast(year: int, backup: bool = True) -> dict[str, int]:
    """
    Merge current-season statcast files into the cumulative statcast files.

    Current-season files are in data/statcast/ as {stem}_{year}_{year}.csv
    Cumulative files are in data/statcast/ as {stem}_2015_2025.csv
    """
    results = {}

    for name, (stem, year_col) in STATCAST_FILES.items():
        current_path = STATCAST_DIR / f"{stem}_{year}_{year}.csv"
        cumulative_path = STATCAST_DIR / f"{stem}_{STATCAST_YEAR_RANGE}.csv"

        if not current_path.exists():
            logger.warning(f"  statcast {name}: current file not found ({current_path.name}), skipping")
            continue

        if not cumulative_path.exists():
            logger.warning(f"  statcast {name}: cumulative file not found ({cumulative_path.name}), skipping")
            continue

        logger.info(f"  Merging statcast {name} for {year} ...")

        current_df = pd.read_csv(current_path)
        cumul_df = pd.read_csv(cumulative_path, low_memory=False)

        if backup:
            _backup(cumulative_path)

        # Ensure year column exists in current data
        if year_col not in current_df.columns:
            current_df[year_col] = year

        # Drop old rows for this year
        before_len = len(cumul_df)
        cumul_df = cumul_df[cumul_df[year_col] != year]
        dropped = before_len - len(cumul_df)

        merged = pd.concat([cumul_df, current_df], ignore_index=True, sort=False)
        merged.to_csv(cumulative_path, index=False)

        n_new = len(current_df)
        logger.info(
            f"  ✓ statcast {name}: dropped {dropped}, added {n_new} (total {len(merged)})"
        )
        results[name] = n_new

    return results


def rejoin_statcast() -> bool:
    """
    Re-run data/historic_mlb/join_statcast_data.py to regenerate
    the _with_statcast.csv variants from the updated base + statcast files.
    """
    join_script = HISTORIC_DIR / 'join_statcast_data.py'
    if not join_script.exists():
        logger.error(f"  join_statcast_data.py not found at {join_script}")
        return False

    logger.info("  Re-running join_statcast_data.py ...")
    try:
        result = subprocess.run(
            [sys.executable, str(join_script)],
            cwd=str(HISTORIC_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"  join_statcast_data.py failed:\n{result.stderr}")
            return False
        logger.info("  ✓ Statcast join complete")
        return True
    except subprocess.TimeoutExpired:
        logger.error("  join_statcast_data.py timed out after 5 minutes")
        return False


def merge_current_season(year: Optional[int] = None, backup: bool = True) -> bool:
    """
    Full merge pipeline: standard stats + statcast + rejoin.

    Args:
        year: Season to merge (defaults to current calendar year)
        backup: Whether to back up historic files before overwriting

    Returns:
        True if all steps succeeded
    """
    if year is None:
        year = datetime.now().year

    logger.info(f"=== Merging {year} data into historic files ===")

    # Step 1: Merge standard stats
    std_results = merge_standard_stats(year, backup=backup)
    if not std_results:
        logger.warning("  No standard stats were merged")

    # Step 2: Merge statcast
    sc_results = merge_statcast(year, backup=backup)

    # Step 3: Rejoin statcast to base files
    join_ok = rejoin_statcast()

    total = sum(std_results.values()) + sum(sc_results.values())
    logger.info(f"=== Merge complete: {total} total rows merged, statcast join {'OK' if join_ok else 'FAILED'} ===")

    return join_ok or bool(std_results)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    merge_current_season()
