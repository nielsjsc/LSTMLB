"""
Historical Values — Central Configuration
==========================================

Single source of truth for all paths and year ranges used across the
historical trade-value pipeline.

Constants for WAR, contracts, prospects, and the convex model are
imported from value_determination.config — the canonical source — so
there is exactly one place to update them.

Usage:
    from historical_values.config import Config, logger, VDConfig
"""

import logging
from pathlib import Path

# ── Import canonical constants from value_determination ──────────────────────
from value_determination.config import (
    Config as VDConfig,
    CURRENT_YEAR,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("historical_values")


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parents[2]          # LSTMLB/
AUTO_TRAIN_DIR = Path(__file__).resolve().parents[1]          # auto_train_models/
DATA_DIR       = ROOT_DIR / "data"

# Input data — salary
SALARY_DIR            = DATA_DIR / "salary"
COTS_BY_YEAR_DIR      = SALARY_DIR / "by_year"

# Input data — historical stats (provides IDfg ↔ mlbam_id crosswalk)
HISTORIC_BATTING_FILE  = DATA_DIR / "historic_mlb" / "mlb_batting_data_1950_2025_with_statcast.csv"
HISTORIC_PITCHING_FILE = DATA_DIR / "historic_mlb" / "mlb_pitching_data_1950_2025_with_statcast.csv"
HISTORIC_BATTING_FILE_CLASSIC  = DATA_DIR / "historic_mlb" / "mlb_batting_data_1950_2025.csv"
HISTORIC_PITCHING_FILE_CLASSIC = DATA_DIR / "historic_mlb" / "mlb_pitching_data_1950_2025.csv"
HISTORIC_FIELDING_FILE = DATA_DIR / "historic_mlb" / "mlb_fielding_data_2000_2025_with_statcast.csv"

# Input data — prospects
PROSPECT_FILE = DATA_DIR / "prospect_data" / "prospects_2014_2026_with_top100.csv"

# Input data — active rosters (for team assignments)
ROSTER_DIR  = DATA_DIR / "active_roster"
ROSTER_FILE = ROSTER_DIR / "current_rosters.csv"

# Input data — Spotrac transactions
SPOTRAC_TRANSACTIONS_FILE = ROOT_DIR / "scrapers" / "data" / "salary" / "spotrac_transactions.csv"

# Input data — ID crosswalk (mlb_id ↔ IDfg ↔ other IDs)
CROSSWALK_FILE = DATA_DIR / "generated" / "player_id_crosswalk.csv"

# Input data — current-year value-determination output (for timeline overlay)
PLAYER_VALUES_FILE = DATA_DIR / "generated" / "value_by_year" / "player_values_complete.csv"

# Output directories — all self-contained under historical_values/
PROJECTIONS_DIR = DATA_DIR / "generated" / "historical_values" / "projections"
SURPLUS_DIR     = DATA_DIR / "generated" / "historical_values" / "surplus"

# Final output — same location the web-app expects
OUTPUT_DIR  = DATA_DIR / "generated" / "value_by_year"
OUTPUT_FILE = OUTPUT_DIR / "trade_value_history.csv"


class Config:
    """
    Pipeline-wide settings.

    WAR constants, contract/arb model, prospect valuation, and the convex
    $/WAR model are all imported from ``value_determination.config.Config``
    (aliased as ``VDConfig``) so updates propagate automatically.
    """

    # ── Year ranges ──────────────────────────────────────────────────────────
    CUTOFF_START       = 2013   # first cutoff year  (predictions 2014-2028)
    CUTOFF_END         = 2025   # last cutoff year   (predictions 2026-2040)
    SNAPSHOT_LAG       = 1      # snapshot_year = cutoff_year + 1
    PROJECTION_HORIZON = 15    # years into the future per snapshot
    CURRENT_YEAR       = CURRENT_YEAR

    # ── Service-time model ───────────────────────────────────────────────────
    SERVICE_TIME_FA = 6     # years to free agency

    # Historical minimum salary by year (for pre-arb estimation in past years)
    HISTORICAL_MIN_SALARY = {
        2014: 500_000, 2015: 507_500, 2016: 507_500, 2017: 535_000,
        2018: 545_000, 2019: 555_000, 2020: 563_500, 2021: 570_500,
        2022: 700_000, 2023: 720_000, 2024: 740_000, 2025: 760_000,
        2026: 780_000,
    }

    # ── Spotrac transaction types ────────────────────────────────────────────
    ZERO_VALUE_TYPES = {"elected_fa", "released", "designated"}
    CONTRACT_TYPES   = {"fa_signing", "extension", "signing"}

    # ── Delegated to VDConfig (single source of truth) ───────────────────────
    # WAR constants
    WAR              = VDConfig.WAR                # WARConstants class
    WOBA_SCALE       = VDConfig.WAR.WOBA_SCALE
    RPA              = VDConfig.WAR.RPA
    LG_WOBA          = VDConfig.WAR.LG_WOBA
    RPW              = VDConfig.WAR.RPW
    LG_PA            = VDConfig.WAR.LG_PA
    LG_FIP           = VDConfig.WAR.LG_FIP
    LG_RA9           = VDConfig.WAR.LG_RA9
    REPLACEMENT_LEVEL_RUNS_200IP = VDConfig.WAR.REPLACEMENT_LEVEL_RUNS_200IP
    DEFAULT_IP_PER_START         = VDConfig.WAR.DEFAULT_IP_PER_START
    DEFAULT_IP_PER_APPEARANCE_RP = VDConfig.WAR.DEFAULT_IP_PER_APPEARANCE_RP
    DEFAULT_SP_IP    = VDConfig.WAR.DEFAULT_SP_IP
    DEFAULT_RP_IP    = VDConfig.WAR.DEFAULT_RP_IP
    POSITIONAL_ADJUSTMENTS = VDConfig.WAR.POSITIONAL_ADJUSTMENTS
    BALLPARK_FACTORS = VDConfig.WAR.BALLPARK_FACTORS

    # Convex $/WAR model
    ConvexModel      = VDConfig.ConvexModel
    ALPHA            = VDConfig.ConvexModel.ALPHA_DEFAULT
    BETA             = VDConfig.ConvexModel.BETA_DEFAULT
    INFLATION_RATE   = VDConfig.Contracts.INFLATION_RATE
    BASE_YEAR        = VDConfig.Contracts.BASE_YEAR

    # Contract / arb salary model
    Contracts        = VDConfig.Contracts           # ContractConstants class
    MIN_SALARY       = VDConfig.Contracts.MIN_SALARY
    ARB_PERCENT      = VDConfig.Contracts.ARB_PERCENT

    # Prospect valuation
    Prospects        = VDConfig.Prospects            # ProspectConstants class
    FV_SURPLUS_VALUE = VDConfig.Prospects.FV_BASE_VALUES
    RANK_ADJ_TOP100_MAX = VDConfig.Prospects.RANK_ADJ_TOP100_MAX
    RANK_ADJ_TOP100_MIN = VDConfig.Prospects.RANK_ADJ_TOP100_MIN

    @classmethod
    def ensure_directories(cls):
        """Create all output directories."""
        for d in [PROJECTIONS_DIR, SURPLUS_DIR, OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)
