"""
Constants and configuration for the Value Determination module.
"""

import os
from pathlib import Path
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases
PathLike = str | Path

# Directory paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data'
PIPELINE_DIR = DATA_DIR / 'generated' / 'pipeline'
HISTORIC_MLB_DIR = DATA_DIR / 'historic_mlb'
SALARY_DIR = DATA_DIR / 'salary'
GENERATED_DIR = DATA_DIR / 'generated'
OUTPUT_DIR = GENERATED_DIR / 'value_by_year'

# Prediction years
PREDICTION_YEARS = range(2025, 2041)

# Required columns for validation
REQUIRED_COLUMNS = {
    'predictions': ['Name', 'IDfg', 'Age', 'WAR'],
    'salary': ['player_name', 'player_id', 'payroll_annual', 'status']
}

# Column definitions
HITTER_COLUMNS = [
    'Name', 'Age', 'G', 'IDfg', 'BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA',
    'wRC+', 'Off', 'BsR', 'Def', 'WAR', 'HR', '2B', '3B', 'SB', 'CS', 'R', 'RBI'
]

PITCHER_COLUMNS = [
    'Name', 'Age', 'GS', 'G', 'IDfg', 'ERA', 'FIP', 'K%', 'BB%', 'WAR', 'SIERA'
]


class PlayerStatus(Enum):
    """Enum for player contract status."""
    PRE_ARB = "Pre-ARB"
    ARB1 = "ARB1"
    ARB2 = "ARB2"
    ARB3 = "ARB3"
    FREE_AGENT = "FA"
    SIGNED = "Signed"
    UNKNOWN = "Unknown"


STATUS_MAPPINGS = {
    'PRE-ARB': PlayerStatus.PRE_ARB,
    'PRE ARB': PlayerStatus.PRE_ARB,
    'ROOKIE': PlayerStatus.PRE_ARB,
    'MIN': PlayerStatus.PRE_ARB,
    'ARB 1': PlayerStatus.ARB1,
    'ARB1': PlayerStatus.ARB1,
    'ARB 2': PlayerStatus.ARB2,
    'ARB2': PlayerStatus.ARB2,
    'ARB 3': PlayerStatus.ARB3,
    'ARB3': PlayerStatus.ARB3,
    'ARB 4': PlayerStatus.ARB3,
    'UFA': PlayerStatus.FREE_AGENT,
    'FA': PlayerStatus.FREE_AGENT,
}

# WAR value calculation constants
WAR_VALUE_TIERS = {
    'tier1': {'max': 2, 'value': 8_000_000},
    'tier2': {'max': 4, 'value': 9_000_000},
    'tier3': {'value': 10_000_000}
}
INFLATION_RATE = 0.04
BASE_YEAR = 2025

# Historical WAR values by year
HISTORICAL_WAR_VALUE = {
    2002: 4800000,
    2003: 4800000,
    2004: 4800000,
    2005: 4800000,
    2006: 5200000,
    2007: 5700000,
    2008: 6200000,
    2009: 6400000,
    2010: 6000000,
    2011: 7500000,
    2012: 6500000,
    2013: 7400000,
    2014: 7600000,
    2015: 8000000,
    2016: 8000000,
    2017: 7900000,
    2018: 8000000,
    2019: 8100000,
    2020: 7900000,
    2021: 8100000,
    2022: 8200000,
    2023: 8100000,
    2024: 8200000
}
WAR_VALUE = 8200000  # Default value for future years

# Contract value constants
MIN_SALARY = {
    'Pre-Arb': 720000,
    'Arb-1': 1000000,
    'Arb-1 (Super 2)': 1200000,
    'Arb-2': 2500000,
    'Arb-3': 4000000,
    'Arb-4': 5000000
}

ARB_PERCENT = {
    'Arb-1': 0.15,
    'Arb-1 (Super 2)': 0.15,
    'Arb-2': 0.25,
    'Arb-3': 0.4,
    'Arb-4': 0.6
}


def ensure_directories():
    """Ensure required directories exist."""
    for directory in [DATA_DIR, GENERATED_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Verified directory exists: {directory}")


def get_war_value(year: int) -> float:
    """Get WAR value for specific year, default to current WAR_VALUE if not found."""
    return HISTORICAL_WAR_VALUE.get(year, WAR_VALUE)
