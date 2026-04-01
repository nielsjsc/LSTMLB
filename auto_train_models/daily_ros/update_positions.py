"""
Phase 1c: Position Profile Updater
====================================

After merge.py updates the historic files with current-season fielding data,
this module rebuilds position profiles and exports a summary CSV so the
daily pipeline can verify which players had position changes.

The heavy lifting is done by the existing build_position_profiles() in
auto_train_models/core/position_profiles.py — which automatically uses
the most recent season of fielding data per player.  This module simply
orchestrates a targeted rebuild and writes a snapshot to disk.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]  # LSTMLB/
HISTORIC_DIR = ROOT_DIR / 'data' / 'historic_mlb'
OUTPUT_DIR = ROOT_DIR / 'auto_train_models' / 'daily_ros' / 'snapshots'

# Ensure import path for core modules
_CORE_PARENT = str(Path(__file__).resolve().parents[1])
if _CORE_PARENT not in sys.path:
    sys.path.insert(0, _CORE_PARENT)


def rebuild_position_profiles(
    year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rebuild position profiles from the (already-merged) historic files
    for every player who appeared in the given season.

    Returns a DataFrame with columns:
        IDfg, Name, Season, primary_position, profile
    """
    from core.position_profiles import (
        build_position_profiles,
        get_display_position,
        load_batting_for_games,
        load_fielding_history,
    )

    if year is None:
        year = datetime.now().year

    logger.info(f"  Rebuilding position profiles (cutoff_year={year}) ...")

    fld_df = load_fielding_history()
    bat_df = load_batting_for_games()

    # Get all players with fielding data in the current year
    current_year_players = fld_df.loc[fld_df['Season'] == year, 'IDfg'].unique().tolist()
    if not current_year_players:
        logger.warning(f"  No fielding data for {year} — no profiles to rebuild")
        return pd.DataFrame()

    logger.info(f"  Found {len(current_year_players)} players with {year} fielding data")

    # Use a lower min_games threshold (3) for the current season so that
    # position changes show up early in the year, before 10 games are reached.
    profiles = build_position_profiles(
        fld_df, bat_df, current_year_players, cutoff_year=year, min_games=3,
    )

    # Build summary records
    rows = []
    for pid, profile in profiles.items():
        name_row = fld_df.loc[fld_df['IDfg'] == pid, 'Name']
        name = name_row.iloc[0] if not name_row.empty else ''
        rows.append({
            'IDfg': pid,
            'Name': name,
            'Season': year,
            'primary_position': get_display_position(profile),
            'profile': profile,
        })

    summary = pd.DataFrame(rows)
    logger.info(f"  ✓ Rebuilt profiles for {len(summary)} players")
    return summary


def export_position_snapshot(
    year: Optional[int] = None,
) -> Optional[Path]:
    """
    Rebuild profiles and save a dated snapshot CSV.

    Returns the path to the snapshot file, or None on failure.
    """
    if year is None:
        year = datetime.now().year

    df = rebuild_position_profiles(year)
    if df.empty:
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    out_path = OUTPUT_DIR / f'position_profiles_{today}.csv'

    # Convert profile dict to string for CSV serialization
    export = df.copy()
    export['profile'] = export['profile'].astype(str)
    export.to_csv(out_path, index=False)

    logger.info(f"  ✓ Snapshot saved: {out_path.name}")
    return out_path


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    export_position_snapshot()
