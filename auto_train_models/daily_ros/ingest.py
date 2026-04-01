"""
Phase 1a: Current-Season Data Scraper
======================================

Non-interactive scraper that downloads the current MLB season's batting,
pitching, fielding, and Statcast data via pybaseball.

Outputs go to data/ as mlb_{type}_data_{year}_{year}.csv and
statcast_{type}_{year}_{year}.csv, matching the existing naming convention
used by MLB_data_script.py.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pybaseball import (
    batting_stats,
    pitching_stats,
    fielding_stats,
    statcast_batter_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_sprint_speed,
    statcast_pitcher_exitvelo_barrels,
    statcast_pitcher_expected_stats,
    statcast_pitcher_pitch_arsenal,
)
from pybaseball.statcast_fielding import (
    statcast_outs_above_average,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
CURRENT_SEASON_DIR = DATA_DIR / 'current_season'

# ── Standard FanGraphs stats ────────────────────────────────────────────────

STANDARD_TYPES = {
    'batting': {
        'func': batting_stats,
        'filename': 'mlb_batting_data',
    },
    'pitching': {
        'func': pitching_stats,
        'filename': 'mlb_pitching_data',
    },
    'fielding': {
        'func': fielding_stats,
        'filename': 'mlb_fielding_data',
    },
}

# ── Statcast season-level aggregates ────────────────────────────────────────

STATCAST_TYPES = {
    'batter_exitvelo': {
        'func': statcast_batter_exitvelo_barrels,
        'filename': 'statcast_batter_exitvelo_barrels',
        'kwargs': {'minBBE': 9},
    },
    'batter_expected': {
        'func': statcast_batter_expected_stats,
        'filename': 'statcast_batter_expected_stats',
        'kwargs': {'minPA': 9},
    },
    'sprint_speed': {
        'func': statcast_sprint_speed,
        'filename': 'statcast_sprint_speed',
        'kwargs': {},
    },
    'pitcher_exitvelo': {
        'func': statcast_pitcher_exitvelo_barrels,
        'filename': 'statcast_pitcher_exitvelo_barrels',
        'kwargs': {'minBBE': 9},
    },
    'pitcher_expected': {
        'func': statcast_pitcher_expected_stats,
        'filename': 'statcast_pitcher_expected_stats',
        'kwargs': {'minPA': 9},
    },
    'pitcher_arsenal_speed': {
        'func': statcast_pitcher_pitch_arsenal,
        'filename': 'statcast_pitcher_arsenal_speed',
        'kwargs': {'minP': 9, 'arsenal_type': 'avg_speed'},
    },
    'pitcher_arsenal_spin': {
        'func': statcast_pitcher_pitch_arsenal,
        'filename': 'statcast_pitcher_arsenal_spin',
        'kwargs': {'minP': 9, 'arsenal_type': 'avg_spin'},
    },
    'fielding_run_value': {
        'func': statcast_outs_above_average,
        'filename': 'statcast_fielding_run_value',
        'kwargs': {'pos': 'all', 'min_att': 9},
    },
}


def _fetch_with_retry(func, *args, max_retries: int = 3, **kwargs) -> Optional[pd.DataFrame]:
    """Call a pybaseball function with exponential-backoff retries."""
    for attempt in range(max_retries):
        try:
            data = func(*args, **kwargs)
            return data
        except Exception as e:
            logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def scrape_standard_stats(year: int) -> dict[str, Path]:
    """
    Download batting, pitching, and fielding stats for a single season.

    Returns dict mapping type name -> saved file path.
    """
    saved = {}
    for name, info in STANDARD_TYPES.items():
        logger.info(f"  Downloading {name} stats for {year} ...")
        try:
            df = _fetch_with_retry(info['func'], year, year, qual=0)
            if df is None or df.empty:
                logger.warning(f"  No {name} data returned for {year}")
                continue

            CURRENT_SEASON_DIR.mkdir(parents=True, exist_ok=True)
            out = CURRENT_SEASON_DIR / f"{info['filename']}_{year}_{year}.csv"
            df.to_csv(out, index=False)
            logger.info(f"  ✓ {name}: {len(df)} rows -> {out.name}")
            saved[name] = out
            time.sleep(2)
        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}")
    return saved


def scrape_statcast(year: int) -> dict[str, Path]:
    """
    Download all Statcast aggregate datasets for a single season (non-interactive).

    Returns dict mapping type name -> saved file path.
    """
    if year < 2015:
        logger.warning(f"Statcast data not available before 2015 (requested {year})")
        return {}

    statcast_dir = DATA_DIR / 'statcast'
    statcast_dir.mkdir(exist_ok=True)

    saved = {}
    for name, info in STATCAST_TYPES.items():
        logger.info(f"  Downloading statcast {name} for {year} ...")
        try:
            # Fielding run value needs per-position download
            if name == 'fielding_run_value':
                frames = _scrape_frv_by_position(info['func'], year, info['kwargs'])
            else:
                frames = [_fetch_with_retry(info['func'], year, **info['kwargs'])]

            parts = [f for f in frames if f is not None and not f.empty]
            if not parts:
                logger.warning(f"  No statcast {name} data for {year}")
                continue

            combined = pd.concat(parts, ignore_index=True)
            if 'year' not in combined.columns:
                combined['year'] = year

            out = statcast_dir / f"{info['filename']}_{year}_{year}.csv"
            combined.to_csv(out, index=False)
            logger.info(f"  ✓ statcast {name}: {len(combined)} rows -> {out.name}")
            saved[name] = out
            time.sleep(2)
        except Exception as e:
            logger.error(f"  ✗ statcast {name} failed: {e}")
    return saved


def _scrape_frv_by_position(func, year: int, base_kwargs: dict) -> list[pd.DataFrame]:
    """Download Fielding Run Value per position, combine into one frame."""
    positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
    frames = []
    for pos in positions:
        try:
            kwargs = {k: v for k, v in base_kwargs.items() if k != 'pos'}
            kwargs['pos'] = pos
            df = _fetch_with_retry(func, year, **kwargs)
            if df is not None and not df.empty:
                df['position'] = pos
                frames.append(df)
            time.sleep(1)
        except Exception as e:
            logger.warning(f"  FRV {pos}: {e}")
    return frames


def scrape_current_season() -> dict[str, Path]:
    """
    Convenience entry-point: scrape everything for the current calendar year.

    Returns dict of all saved file paths keyed by type name.
    """
    year = datetime.now().year
    logger.info(f"=== Scraping {year} season data ===")

    saved = scrape_standard_stats(year)
    saved.update(scrape_statcast(year))

    logger.info(f"=== Scraping complete: {len(saved)} datasets saved ===")
    return saved


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    scrape_current_season()
