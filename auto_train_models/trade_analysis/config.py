"""
Trade Analysis — Central Configuration
========================================

Single source of truth for all paths, constants, and year ranges used across
the trade-analysis pipeline.  Import from here, not from individual modules.

Usage:
    from trade_analysis.config import Config, logger
"""

import logging
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("trade_analysis")


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parents[2]          # LSTMLB/
AUTO_TRAIN_DIR = Path(__file__).resolve().parents[1]          # auto_train_models/
DATA_DIR       = ROOT_DIR / "data"
SCRIPTS_DIR    = AUTO_TRAIN_DIR / "scripts"

# Input data — salary
SALARY_DIR            = DATA_DIR / "salary"
COTS_BY_YEAR_DIR      = SALARY_DIR / "by_year"                  # Cot's per-year CSVs (2014.csv … 2025.csv)
HISTORICAL_ROSTERS    = SALARY_DIR / "historical_rosters.csv"   # from Cot's scraper (legacy)
SPOTRAC_SALARY_FILE   = SALARY_DIR / "mlb_salary_data.csv"      # from Spotrac scraper

# Input data — transactions
TRADE_PLAYERS_FILE    = DATA_DIR / "transactions" / "trade_players.csv"
TRADES_FILE           = DATA_DIR / "transactions" / "trades.csv"

# Input data — historical stats (provides IDfg ↔ mlbam_id crosswalk)
HISTORIC_BATTING_FILE = DATA_DIR / "historic_mlb" / "mlb_batting_data_1950_2025_with_statcast.csv"
HISTORIC_PITCHING_FILE= DATA_DIR / "historic_mlb" / "mlb_pitching_data_1950_2025_with_statcast.csv"
HISTORIC_BATTING_FILE_CLASSIC = DATA_DIR / "historic_mlb" / "mlb_batting_data_1950_2025.csv"
HISTORIC_PITCHING_FILE_CLASSIC = DATA_DIR / "historic_mlb" / "mlb_pitching_data_1950_2025.csv"

# Input data — prospects
PROSPECT_FILE         = DATA_DIR / "prospect_data" / "prospects_2014_2026_with_top100.csv"

# Input data — active rosters (for team assignments / park factors)
ROSTER_DIR            = DATA_DIR / "active_roster"

# Output — projections are stored per-cutoff-year
PROJECTIONS_DIR = DATA_DIR / "generated" / "trade_analysis" / "projections"
SURPLUS_DIR     = DATA_DIR / "generated" / "trade_analysis" / "surplus"
RESULTS_DIR     = DATA_DIR / "generated" / "trade_analysis" / "results"


class Config:
    """Pipeline-wide settings."""

    # ── Year ranges ──────────────────────────────────────────────────────────
    # Cutoff years: for each cutoff_year Y we generate projections Y+1 … Y+15
    # using only data available up to (and including) season Y.
    CUTOFF_START   = 2013   # first cutoff year (predictions 2014-2028)
    CUTOFF_END     = 2024   # last cutoff year  (predictions 2025-2039)
    PROJECTION_HORIZON = 15 # years into the future per snapshot

    # ── WAR valuation (tiered — same as value_determination) ─────────────────
    WAR_VALUE_TIERS = {
        "tier1": {"max": 2, "value": 8_000_000},   # 0–2 WAR
        "tier2": {"max": 4, "value": 9_000_000},   # 2–4 WAR
        "tier3": {"value": 10_000_000},             # 4+ WAR
    }
    INFLATION_RATE = 0.04
    BASE_YEAR      = 2025

    # Historical $/WAR by year (FanGraphs estimates)
    HISTORICAL_WAR_VALUE = {
        2014: 7_600_000,
        2015: 8_000_000,
        2016: 8_000_000,
        2017: 7_900_000,
        2018: 8_000_000,
        2019: 8_100_000,
        2020: 7_900_000,
        2021: 8_100_000,
        2022: 8_200_000,
        2023: 8_100_000,
        2024: 8_200_000,
    }

    # ── Pre-arb / arb salary model ──────────────────────────────────────────
    MIN_SALARY = {
        "Pre-Arb":  720_000,
        "Arb-1":  1_000_000,
        "Arb-2":  2_500_000,
        "Arb-3":  4_000_000,
    }
    ARB_PERCENT = {
        "Arb-1": 0.15,
        "Arb-2": 0.25,
        "Arb-3": 0.40,
    }

    # Historical minimum MLB salary by year (approximate)
    HISTORICAL_MIN_SALARY = {
        2014: 500_000, 2015: 507_500, 2016: 507_500, 2017: 535_000,
        2018: 545_000, 2019: 555_000, 2020: 563_500, 2021: 570_500,
        2022: 700_000, 2023: 720_000, 2024: 740_000, 2025: 760_000,
    }

    # ── Service-time model ───────────────────────────────────────────────────
    # Years of MLB service time needed for each milestone
    SERVICE_TIME_ARB    = 3    # years to arbitration eligibility
    SERVICE_TIME_FA     = 6    # years to free agency
    # We approximate service from the number of *seasons* a player appears in
    # the historical data.  A full season ≈ 1 year of service.

    # ── Pitcher WAR constants (FanGraphs-style FIP-WAR) ─────────────────────
    LG_FIP  = 4.18        # league-average FIP
    RPW     = 9.786       # runs-per-win conversion
    REPLACEMENT_LEVEL_RUNS_200IP = 18.9   # replacement-level runs over 200 IP
    DEFAULT_SP_IP = 180.0
    DEFAULT_RP_IP =  60.0

    # ── Prediction infrastructure ────────────────────────────────────────────
    # Which model-types to run for each cutoff year.
    # We use the pretrained (classical-feature) models because they cover
    # 1950–present and work for every cutoff year.  Finetuned models only
    # have data from 2015/2020 onward.
    USE_PRETRAINED = True

    # ── Trade matching ───────────────────────────────────────────────────────
    # Window around the trade date to match the "pre-trade" snapshot year.
    # e.g. a July 2018 trade uses the cutoff_year = 2017 projections.
    SNAPSHOT_LAG = 1   # projections made from data through (trade_year - 1)

    # ── Prospect FV → surplus value mapping ──────────────────────────────────
    # Base surplus value for each FV tier (in dollars).  These represent the
    # expected total future surplus a prospect with that FV grade will produce
    # on a pre-arb + arb control window (~6 years).  Values are calibrated to
    # the same scale as MLB surplus so they can be directly compared.
    #
    # The scale is anchored around: a 50 FV prospect ≈ future average regular
    # ≈ ~2 WAR/yr for ~6 years ≈ ~$50–60M surplus after min-salary costs.
    FV_SURPLUS_VALUE = {
        35:   2_000_000,
        40:   5_000_000,
        45:  15_000_000,
        50:  40_000_000,
        55:  70_000_000,
        60: 110_000_000,
        65: 160_000_000,
        70: 220_000_000,
        75: 280_000_000,
        80: 350_000_000,
    }

    # Top-100 rank bonus: top-ranked prospects get a premium on top of their
    # FV-based value.  This captures the extra certainty / upside that comes
    # with being ranked on the consensus top-100 list.
    TOP_100_MAX_BONUS = 0.30   # up to 30% bonus for #1 overall
    TOP_100_MIN_BONUS = 0.02   # ~2% bonus for #100

    @classmethod
    def ensure_directories(cls):
        """Create all output directories."""
        for d in [PROJECTIONS_DIR, SURPLUS_DIR, RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
