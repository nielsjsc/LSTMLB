"""
Central Configuration for the Value Determination Engine (V2)
=============================================================

Consolidates paths, MLB league constants, contract rules, market valuation
parameters, Monte Carlo simulation settings, and pre-debut prospect valuation
curves in one self-documenting module.

Usage:
    from value_determination.config import Config, CURRENT_YEAR
"""

from __future__ import annotations

import json
import logging
import math
from enum import Enum
from pathlib import Path

from core.park_factors import PARK_FACTORS_5YR, TEAM_CODE_ALIASES, get_park_factor  # noqa: F401

CURRENT_YEAR = 2026

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("value_determination")


# =============================================================================
# PATH CONFIGURATION
# =============================================================================
class Paths:
    """All directory and file paths used in the valuation pipeline."""

    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "data"

    # Input directories
    PIPELINE_DIR = DATA_DIR / "generated" / "pipeline"
    HISTORIC_MLB_DIR = DATA_DIR / "historic_mlb"
    SALARY_DIR = DATA_DIR / "salary"
    PROSPECT_DIR = DATA_DIR / "prospect_data"
    ROSTER_DIR = DATA_DIR / "active_roster"

    # Output directories
    GENERATED_DIR = DATA_DIR / "generated"
    OUTPUT_DIR = GENERATED_DIR / "value_by_year"
    MILB_GENERATED_DIR = GENERATED_DIR / "MiLB"

    # Canonical files
    ROSTER_FILE = ROSTER_DIR / "current_rosters.csv"
    SALARY_FILE = SALARY_DIR / "mlb_salary_data.csv"
    UNIVERSAL_SALARY_FILE = SALARY_DIR / "universal_salary.csv"
    PROSPECT_FILE = PROSPECT_DIR / "prospects_2014_2026_with_top100.csv"
    PROSPECT_VALUES_FILE = MILB_GENERATED_DIR / "prospect_values.csv"
    VALUATION_CALIBRATION_FILE = GENERATED_DIR / "valuation_calibration.json"

    # Historical trade-value pipeline paths
    COTS_BY_YEAR_DIR = SALARY_DIR / "by_year"
    HISTORIC_BATTING_FILE = HISTORIC_MLB_DIR / "mlb_batting_data_1950_2025_with_statcast.csv"
    HISTORIC_PITCHING_FILE = HISTORIC_MLB_DIR / "mlb_pitching_data_1950_2025_with_statcast.csv"
    HISTORIC_BATTING_FILE_CLASSIC = HISTORIC_MLB_DIR / "mlb_batting_data_1950_2025.csv"
    HISTORIC_PITCHING_FILE_CLASSIC = HISTORIC_MLB_DIR / "mlb_pitching_data_1950_2025.csv"
    HISTORIC_FIELDING_FILE = HISTORIC_MLB_DIR / "mlb_fielding_data_2000_2025_with_statcast.csv"
    CROSSWALK_FILE = GENERATED_DIR / "player_id_crosswalk.csv"
    PLAYER_VALUES_FILE = OUTPUT_DIR / "player_values_complete.csv"
    SPOTRAC_TRANSACTIONS_FILE = SALARY_DIR / "spotrac_transactions.csv"
    PROJECTIONS_DIR = GENERATED_DIR / "historical_values" / "projections"
    SURPLUS_DIR = GENERATED_DIR / "historical_values" / "surplus"
    TRADE_VALUE_HISTORY_FILE = OUTPUT_DIR / "trade_value_history.csv"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required output directories."""
        for directory in (
            cls.DATA_DIR,
            cls.GENERATED_DIR,
            cls.OUTPUT_DIR,
            cls.MILB_GENERATED_DIR,
            cls.PROJECTIONS_DIR,
            cls.SURPLUS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
class Columns:
    """Column name definitions for upstream prediction validation and stat extraction."""

    REQUIRED = {
        "batter_predictions": ["Name", "IDfg", "Age", "wOBA"],
        "pitcher_predictions": ["Name", "IDfg", "Age", "FIP"],
        "salary": ["player_name", "player_id", "payroll_annual", "status"],
    }

    HITTER_COLUMNS = [
        "Name", "Age", "G", "PA", "BB_bat", "K_bat", "IDfg", "BB%", "K%", "AVG", "OBP", "SLG", "wOBA",
        "wRC+", "Bat", "BsR", "Def", "WAR", "HR", "2B", "3B", "SB", "CS", "R", "RBI",
    ]

    PITCHER_COLUMNS = [
        "Name", "Age", "GS", "G", "IP", "BB_pit", "K_pit", "ER_pit", "IDfg", "ERA", "FIP", "K%", "BB%", "WAR",
        "HR/FB", "K/9", "BB/9", "HR/9", "FB%", "GB%", "SIERA",
    ]


# =============================================================================
# PLAYER STATUS DEFINITIONS
# =============================================================================
class PlayerStatus(Enum):
    """Canonical player contract status tiers."""

    PRE_ARB = "Pre-ARB"
    ARB1 = "ARB1"
    ARB2 = "ARB2"
    ARB3 = "ARB3"
    FREE_AGENT = "FA"
    SIGNED = "Signed"
    UNKNOWN = "Unknown"


STATUS_MAPPINGS = {
    "PRE-ARB": PlayerStatus.PRE_ARB,
    "PRE ARB": PlayerStatus.PRE_ARB,
    "ROOKIE": PlayerStatus.PRE_ARB,
    "MIN": PlayerStatus.PRE_ARB,
    "ARB 1": PlayerStatus.ARB1,
    "ARB1": PlayerStatus.ARB1,
    "ARB 2": PlayerStatus.ARB2,
    "ARB2": PlayerStatus.ARB2,
    "ARB 3": PlayerStatus.ARB3,
    "ARB3": PlayerStatus.ARB3,
    "ARB 4": PlayerStatus.ARB3,
    "UFA": PlayerStatus.FREE_AGENT,
    "FA": PlayerStatus.FREE_AGENT,
}


# =============================================================================
# WAR CALCULATION CONSTANTS
# =============================================================================
class WARConstants:
    """League environment and FanGraphs WAR calculation constants."""

    BALLPARK_FACTORS = PARK_FACTORS_5YR

    # League offensive environment (2025 baseline)
    WOBA_SCALE = 1.232
    RPA = 0.118
    LG_WOBA = 0.313
    RPW = 9.774
    LG_PA = 186188
    LG_RUNS_PER_PA = 0.118
    LG_WRC_PER_PA = 0.117

    WOBA_WEIGHTS = {
        "wBB": 0.691,
        "wHBP": 0.722,
        "w1B": 0.882,
        "w2B": 1.252,
        "w3B": 1.584,
        "wHR": 2.037,
    }

    # Positional adjustments (runs per 162 games)
    POSITIONAL_ADJUSTMENTS = {
        "C": 12.5,
        "SS": 7.5,
        "2B": 2.5,
        "CF": 2.5,
        "3B": 2.5,
        "LF": -7.5,
        "RF": -7.5,
        "1B": -12.5,
        "DH": -17.5,
    }

    # Defensive spectrum drift: past age 28, defenders lose range and migrate
    # down the defensive spectrum toward corner/DH (-0.8 runs per 150 G per year).
    DEFENSIVE_SPECTRUM_DRIFT_AGE_PIVOT = 28.0
    DEFENSIVE_SPECTRUM_DRIFT_RUNS_PER_YEAR = -0.8

    # Pitcher environment constants (calibrated to 0.00 mean bias vs official FanGraphs fWAR:
    # SP ~0.12 wins/9 IP = 25.5 runs/200 IP; RP ~0.03 wins/9 IP = 9.8 runs/200 IP)
    LG_FIP = 4.16
    LG_RA9 = 4.35
    REPLACEMENT_LEVEL_RUNS_200IP = 25.5
    REPLACEMENT_LEVEL_RUNS_200IP_SP = 25.5
    REPLACEMENT_LEVEL_RUNS_200IP_RP = 9.8
    DEFAULT_IP_PER_START = 5.75
    DEFAULT_IP_PER_APPEARANCE_RP = 1.0
    DEFAULT_SP_IP = 175
    DEFAULT_RP_IP = 65
    BF_PER_IP = 4.33

    TEAM_ABBREVIATIONS = {
        "Athletics": "ATH",
        "Oakland Athletics": "ATH",
        "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD",
        "Seattle Mariners": "SEA",
        "San Francisco Giants": "SF",
        "Arizona Diamondbacks": "ARI",
        "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC",
        "Chicago White Sox": "CHW",
        "Cincinnati Reds": "CIN",
        "Cleveland Guardians": "CLE",
        "Cleveland Indians": "CLE",
        "Colorado Rockies": "COL",
        "Detroit Tigers": "DET",
        "Houston Astros": "HOU",
        "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA",
        "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN",
        "New York Mets": "NYM",
        "New York Yankees": "NYY",
        "Philadelphia Phillies": "PHI",
        "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB",
        "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR",
        "Washington Nationals": "WSH",
    }


# =============================================================================
# CONTRACT & ARBITRATION CONSTANTS
# =============================================================================
class ContractConstants:
    """CBA minimum salary and arbitration progression parameters."""

    HISTORICAL_WAR_VALUE = {
        2002: 4_800_000, 2003: 4_800_000, 2004: 4_800_000, 2005: 4_800_000,
        2006: 5_200_000, 2007: 5_700_000, 2008: 6_200_000, 2009: 6_400_000,
        2010: 6_000_000, 2011: 7_500_000, 2012: 6_500_000, 2013: 7_400_000,
        2014: 7_600_000, 2015: 8_000_000, 2016: 8_000_000, 2017: 7_900_000,
        2018: 8_000_000, 2019: 8_100_000, 2020: 7_900_000, 2021: 8_100_000,
        2022: 8_200_000, 2023: 8_500_000, 2024: 8_800_000, 2025: 9_000_000,
    }

    WAR_VALUE_DEFAULT = 9_200_000
    INFLATION_RATE = 0.04
    BASE_YEAR = 2026

    # Minimum salaries by status (2026 baseline; scaled by year when historical)
    MIN_SALARY = {
        "Pre-Arb": 780_000,
        "Arb-1": 2_000_000,
        "Arb-1 (Super 2)": 2_000_000,
        "Arb-2": 3_500_000,
        "Arb-3": 5_500_000,
        "Arb-4": 7_000_000,
    }

    # Share of single-season production value paid in each arbitration tier
    # (Anchored to empirical Spotrac MLB Arb-1..Arb-4 settlements: ~18% / 28% / 42% / 52%)
    ARB_PERCENT = {
        "Arb-1": 0.18,
        "Arb-1 (Super 2)": 0.16,
        "Arb-2": 0.28,
        "Arb-3": 0.42,
        "Arb-4": 0.52,
    }

    @classmethod
    def get_war_value(cls, year: int) -> float:
        """Linear $/WAR benchmark for a given season."""
        if year in cls.HISTORICAL_WAR_VALUE:
            return float(cls.HISTORICAL_WAR_VALUE[year])
        return float(cls.WAR_VALUE_DEFAULT * ((1.0 + cls.INFLATION_RATE) ** (year - cls.BASE_YEAR)))


# =============================================================================
# MARKET VALUATION & CONSOLIDATION MODEL
# =============================================================================
class MarketValuation:
    """
    Empirically calibrated WAR-to-dollar production model + package consolidation.

    Single-season production value:
        production_value = rate[player_role] * max(WAR, 0)^WAR_BETA * (1 + INFLATION_RATE)^(year - BASE_YEAR)

    Note:
        WAR already prices offensive runs, baserunning, fielding, positional scarcity,
        and post-28 defensive spectrum drift. No double-counted wRC+ or defensive
        dollar multipliers are layered on top.

    Multi-asset trade package consolidation (applied in /trades/analyze):
        consolidated_pos = (sum_{v_i > 0} v_i^PACKAGE_GAMMA)^(1 / PACKAGE_GAMMA)
        consolidated_surplus = consolidated_pos + sum_{v_j <= 0} v_j
    """

    BASE_YEAR = 2026
    INFLATION_RATE = 0.04
    DISCOUNT_RATE = 0.13       # Annual GM contention-window + roster/health risk discount rate

    RATE_BATTER = 7_500_000.0
    RATE_SP = 9_600_000.0
    RATE_RP = 14_800_000.0     # Compressed 60-IP high-leverage closer slot value above 0.44 RP hurdle
    WAR_BETA = 1.36            # Roster-slot scarcity convexity above W_slot
    PACKAGE_GAMMA = 1.35       # Sub-additive multi-piece package norm exponent

    # Full 26-man MLB roster-slot waiver hurdle (WORP) on the FanGraphs fWAR scale:
    # - A waiver #6/#7 SP (4.75 FIP in 150 IP at 25.5 rep runs/200 IP) earns ~0.90 fWAR.
    # - A waiver utility/corner bat (~87 wRC+) earns ~0.55 fWAR.
    # - A waiver middle RP (4.05 FIP, 0.90 gmLI in 60 IP at 9.8 rep runs/200 IP) earns ~0.44 fWAR.
    WORP_HURDLE_FULL = {
        "Batter": 0.55,
        "SP": 0.90,
        "RP": 0.48,
    }

    _calibration_loaded = False

    @classmethod
    def load_calibration(cls, force: bool = False) -> dict:
        """Load empirical calibration parameters from valuation_calibration.json if present."""
        if cls._calibration_loaded and not force:
            return cls.current_parameters()

        cal_path = Paths.VALUATION_CALIBRATION_FILE
        if cal_path.exists():
            try:
                with open(cal_path, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                cls.RATE_BATTER = float(cal.get("rate_batter", cls.RATE_BATTER))
                cls.RATE_SP = float(cal.get("rate_sp", cls.RATE_SP))
                cls.RATE_RP = float(cal.get("rate_rp", cls.RATE_RP))
                cls.WAR_BETA = float(cal.get("war_beta", cls.WAR_BETA))
                cls.DISCOUNT_RATE = float(cal.get("discount_rate", cls.DISCOUNT_RATE))
                cls.PACKAGE_GAMMA = float(cal.get("package_gamma", cls.PACKAGE_GAMMA))
                if "prospect_scale" in cal:
                    ProspectValuationConfig.PROSPECT_SCALE = float(cal["prospect_scale"])
                logger.info(
                    "Loaded calibrated V2 valuation parameters from %s "
                    "(Batter=$%.2fM, SP=$%.2fM, RP=$%.2fM, beta=%.3f, r=%.3f, gamma=%.2f)",
                    cal_path.name,
                    cls.RATE_BATTER / 1e6,
                    cls.RATE_SP / 1e6,
                    cls.RATE_RP / 1e6,
                    cls.WAR_BETA,
                    cls.DISCOUNT_RATE,
                    cls.PACKAGE_GAMMA,
                )
            except Exception as exc:
                logger.warning("Could not load %s (%s) — using defaults", cal_path, exc)

        cls._calibration_loaded = True
        return cls.current_parameters()

    @classmethod
    def current_parameters(cls) -> dict:
        return {
            "rate_batter": cls.RATE_BATTER,
            "rate_sp": cls.RATE_SP,
            "rate_rp": cls.RATE_RP,
            "war_beta": cls.WAR_BETA,
            "discount_rate": cls.DISCOUNT_RATE,
            "inflation_rate": cls.INFLATION_RATE,
            "package_gamma": cls.PACKAGE_GAMMA,
            "prospect_scale": ProspectValuationConfig.PROSPECT_SCALE,
        }

    @classmethod
    def role_rate(cls, player_role: str | None) -> float:
        cls.load_calibration()
        role_upper = str(player_role or "Batter").strip().upper()
        if role_upper == "SP":
            return cls.RATE_SP
        if role_upper == "RP":
            return cls.RATE_RP
        return cls.RATE_BATTER

    @classmethod
    def roster_hurdle(cls, player_role: str | None, pt_share: float = 1.0) -> float:
        """Return the full 26-man roster-slot waiver WAR hurdle W_slot (no part-time discount)."""
        role_upper = str(player_role or "Batter").strip().upper()
        if role_upper == "RP":
            return float(cls.WORP_HURDLE_FULL["RP"])
        if role_upper == "SP":
            return float(cls.WORP_HURDLE_FULL["SP"])
        return float(cls.WORP_HURDLE_FULL["Batter"])

    @classmethod
    def min_salary_for_year(cls, year: int) -> float:
        yr = int(year)
        if yr in HistorySettings.HISTORICAL_MIN_SALARY:
            return float(HistorySettings.HISTORICAL_MIN_SALARY[yr])
        return float(ContractConstants.MIN_SALARY["Pre-Arb"] * ((1.0 + cls.INFLATION_RATE) ** (yr - cls.BASE_YEAR)))

    @classmethod
    def calculate_production_value(
        cls,
        war: float,
        year: int,
        player_role: str = "Batter",
        pt_share: float = 1.0,
        season_fraction: float = 1.0,
        **kwargs,
    ) -> float:
        """
        Convert single-season WAR into nominal year-specific production dollars
        using the continuous Roster-Constrained WORP curve:
          - For 0 <= W <= W_slot: V = S_min * (W / W_slot)   (0 surplus at S_min)
          - For W > W_slot:       V = S_min + R * inflation * (W - W_slot)^beta
        """
        cls.load_calibration()
        if war is None or (isinstance(war, float) and math.isnan(war)) or war <= 0.0:
            return 0.0
        sf = float(max(0.0, min(1.0, season_fraction)))
        if sf <= 1e-6:
            return 0.0
        w_slot = cls.roster_hurdle(player_role, pt_share=pt_share) * sf
        s_min = cls.min_salary_for_year(int(year)) * sf
        w = float(war)
        if w <= w_slot:
            return float(s_min * (w / max(w_slot, 1e-6)))
        rate = cls.role_rate(player_role)
        inflation = (1.0 + cls.INFLATION_RATE) ** (int(year) - cls.BASE_YEAR)
        # Scale excess-WAR term homogeneously when season_fraction < 1.0
        full_excess = (w - w_slot) / sf
        return float(s_min + sf * rate * (full_excess ** cls.WAR_BETA) * inflation)

    # Alias for callers transitioning from MarketValuation.calculate_value
    @classmethod
    def calculate_value(cls, war: float, year: int, position_group: str = "Batter", **kwargs) -> float:
        return cls.calculate_production_value(war=war, year=year, player_role=position_group)

    @classmethod
    def consolidate_package(cls, asset_values: list[float]) -> tuple[float, float, float]:
        """
        Consolidate a package of trade assets using the sub-additive L_gamma norm.

        Positive-surplus assets combine sub-linearly (so 4 depth pieces cannot
        buy 1 superstar), while negative-surplus salary dumps subtract linearly.

        Returns:
            (consolidated_surplus, raw_surplus_sum, consolidation_penalty)
        """
        cls.load_calibration()
        clean_vals = [float(v) for v in asset_values if v is not None and not math.isnan(float(v))]
        if not clean_vals:
            return 0.0, 0.0, 0.0

        raw_sum = sum(clean_vals)
        positives = [v for v in clean_vals if v > 0.0]
        negatives = [v for v in clean_vals if v <= 0.0]

        if len(positives) <= 1:
            pos_consolidated = sum(positives)
        else:
            gamma = max(1.0, cls.PACKAGE_GAMMA)
            pos_consolidated = sum(v ** gamma for v in positives) ** (1.0 / gamma)

        consolidated = pos_consolidated + sum(negatives)
        penalty = consolidated - raw_sum  # <= 0 when >1 positive assets
        return float(consolidated), float(raw_sum), float(penalty)


# =============================================================================
# MONTE CARLO SIMULATION SETTINGS
# =============================================================================
class MonteCarloConfig:
    """Parameters for the vectorized career-path Monte Carlo trade valuation engine."""

    N_SIMS = 1000
    SEED = 42

    # Year-to-year AR(1) persistence of talent/health shocks
    AR1_RHO = 0.75

    # Annual WAR shock standard deviation (established full-time MLB player;
    # scaled proportionally to each role's full-season fWAR range)
    SIGMA_ESTABLISHED = {
        "Batter": 1.15,
        "SP": 1.20,
        "RP": 0.42,
    }

    # Unproven call-ups have wider outcome dispersion until stabilized
    SIGMA_ROOKIE_MULTIPLIER = 1.45
    STABILIZATION_VOLUME = {
        "Batter": 600.0,   # Career MLB PA
        "SP": 150.0,       # Career MLB IP
        "RP": 70.0,        # Career MLB IP
    }

    # Annual permanent MLB-viability / career-attrition hazard
    ATTRITION_BASE_HAZARD = {
        "Batter": 0.03,
        "SP": 0.055,
        "RP": 0.085,
    }
    ATTRITION_AGE_PIVOT = 29.0
    ATTRITION_AGE_SLOPE = {
        "Batter": 0.018,
        "SP": 0.022,
        "RP": 0.025,
    }

    # Career volume below which recent ranked prospects blend smoothly with
    # their pre-debut scouting distribution to prevent MLB-debut value cliffs
    ROOKIE_PROSPECT_BLEND_VOLUME = {
        "Batter": 350.0,   # Career MLB PA
        "SP": 90.0,        # Career MLB IP
        "RP": 40.0,        # Career MLB IP
    }

    # Default buyout cost as a fraction of option salary when buyout is unspecified
    OPTION_BUYOUT_FRACTION = 0.10


# =============================================================================
# UNIFIED PRE-DEBUT PROSPECT VALUATION SETTINGS
# =============================================================================
class ProspectValuationConfig:
    """Single source of truth for pre-debut (0 MLB game) prospect trade values."""

    BASE_YEAR = 2026
    PROSPECT_SCALE = 1.0       # Global calibration scalar tuned by calibrate_valuation.py

    # Expected NPV surplus in BASE_YEAR (2026) dollars for an MLB-ready Batter
    # calibrated to MLB.com Pipeline's empirical FV grade percentiles
    # (~18 FV 60+, ~82 FV 55s in Top-100, ~131 FV 50s in #101-230, ~252 FV 45s, ~416 FV 40s).
    FV_BASE_VALUES: dict[int, float] = {
        80: 120_000_000.0,
        75: 98_000_000.0,
        70: 78_000_000.0,
        65: 56_000_000.0,
        60: 38_000_000.0,
        55: 19_000_000.0,
        50: 8_500_000.0,
        45: 3_500_000.0,
        40: 1_200_000.0,
        35: 400_000.0,
        30: 150_000.0,
    }

    # Top-100 rank bonus: smooth decay from 1.35x (#1 overall) to 1.00x (#100)
    TOP100_MAX_BONUS = 1.35
    TOP100_MIN_BONUS = 1.00

    # Role multiplier reflecting pitcher UCL / attrition risk ("TINSTAAPP")
    ROLE_MULTIPLIER = {
        "Batter": 1.00,
        "SP": 0.78,
        "RP": 0.55,
    }

    # Expected years to MLB arrival by minor-league level when explicit ETA is absent
    LEVEL_ETA_YEARS = {
        "MLB": 0.0,
        "AAA": 0.3,
        "AA": 1.2,
        "A+": 2.2,
        "HIGH-A": 2.2,
        "A": 3.0,
        "LOW-A": 3.0,
        "A-": 3.5,
        "ROK": 4.0,
        "CPX": 4.0,
        "DSL": 4.5,
    }

    # Bounded MiLB age-adjusted performance modifier [min, max]
    PERF_MODIFIER_BOUNDS = (0.85, 1.15)

    # Prospect recency window (years) when checking recent scouting pedigree
    RECENCY_YEARS = 3


# =============================================================================
# PIPELINE & HISTORICAL SETTINGS
# =============================================================================
class PipelineSettings:
    """Settings controlling pipeline execution."""

    PREDICTION_YEARS = range(2026, 2031)
    CURRENT_YEAR = 2026
    APPLY_PARK_FACTORS = True
    FAIL_ON_MISSING_DATA = False
    LOG_UNMATCHED_PLAYERS = True
    EXPORT_INTERMEDIATE_FILES = False
    VERBOSE_OUTPUT = True


class HistorySettings:
    """Settings for the 2014-2026 historical trade-value snapshot builder."""

    CUTOFF_START = 2013
    CUTOFF_END = 2025
    SNAPSHOT_LAG = 1
    PROJECTION_HORIZON = 15
    SERVICE_TIME_FA = 6

    HISTORICAL_MIN_SALARY = {
        2014: 500_000, 2015: 507_500, 2016: 507_500, 2017: 535_000,
        2018: 545_000, 2019: 555_000, 2020: 563_500, 2021: 570_500,
        2022: 700_000, 2023: 720_000, 2024: 740_000, 2025: 760_000,
        2026: 780_000,
    }

    ZERO_VALUE_TYPES = {"elected_fa", "released", "designated"}
    CONTRACT_TYPES = {"fa_signing", "extension", "signing"}


# =============================================================================
# UNIFIED CONFIG CLASS
# =============================================================================
class Config:
    """Unified access point for all valuation configuration."""

    Paths = Paths
    Columns = Columns
    PlayerStatus = PlayerStatus
    StatusMappings = STATUS_MAPPINGS
    WAR = WARConstants
    Contracts = ContractConstants
    Market = MarketValuation
    MonteCarlo = MonteCarloConfig
    Prospects = ProspectValuationConfig
    Pipeline = PipelineSettings
    History = HistorySettings
    logger = logger


# Module-level convenience exports
ROOT_DIR = Paths.ROOT_DIR
DATA_DIR = Paths.DATA_DIR
PIPELINE_DIR = Paths.PIPELINE_DIR
HISTORIC_MLB_DIR = Paths.HISTORIC_MLB_DIR
SALARY_DIR = Paths.SALARY_DIR
GENERATED_DIR = Paths.GENERATED_DIR
OUTPUT_DIR = Paths.OUTPUT_DIR

PREDICTION_YEARS = PipelineSettings.PREDICTION_YEARS
HITTER_COLUMNS = Columns.HITTER_COLUMNS
PITCHER_COLUMNS = Columns.PITCHER_COLUMNS

BALLPARK_FACTORS = WARConstants.BALLPARK_FACTORS
WOBA_SCALE = WARConstants.WOBA_SCALE
RPA = WARConstants.RPA
LG_WOBA = WARConstants.LG_WOBA
RPW = WARConstants.RPW
LG_FIP = WARConstants.LG_FIP

WAR_VALUE = ContractConstants.WAR_VALUE_DEFAULT
HISTORICAL_WAR_VALUE = ContractConstants.HISTORICAL_WAR_VALUE
get_war_value = ContractConstants.get_war_value
ensure_directories = Paths.ensure_directories
