"""
Central Configuration for the Value Determination Module
=========================================================

This module consolidates ALL configuration values used throughout the value determination
pipeline. Edit values here rather than in individual files.

Configuration Categories:
    - Directory Paths
    - Column Definitions
    - WAR Calculation Constants
    - Contract/Salary Constants
    - Prospect Valuation Constants
    - Pipeline Settings

Usage:
    from value_determination.config import Config, CURRENT_YEAR
    
    # Access any config value
    data_dir = Config.Paths.DATA_DIR
    war_value = Config.WAR.get_value_for_year(2025)
    prospect_threshold = Config.Prospects.EXPERIENCE_THRESHOLD_GAMES['batter']
"""

from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
import logging

# Module-level constant for easy import
CURRENT_YEAR = 2026
# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# All modules should use this logger for consistent output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('value_determination')


# =============================================================================
# PATH CONFIGURATION
# =============================================================================
class Paths:
    """All directory and file paths used in the pipeline."""
    
    # Root directories
    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / 'data'
    
    # Input directories
    PIPELINE_DIR = DATA_DIR / 'generated' / 'pipeline'
    HISTORIC_MLB_DIR = DATA_DIR / 'historic_mlb'
    SALARY_DIR = DATA_DIR / 'salary'
    PROSPECT_DIR = DATA_DIR / 'prospect_data'
    ROSTER_DIR = DATA_DIR / 'active_roster'
    
    # Output directories
    GENERATED_DIR = DATA_DIR / 'generated'
    OUTPUT_DIR = GENERATED_DIR / 'value_by_year'
    
    # Specific files
    ROSTER_FILE = ROSTER_DIR / 'current_rosters.csv'
    SALARY_FILE = SALARY_DIR / 'mlb_salary_data.csv'
    PROSPECT_FILE = PROSPECT_DIR / 'prospects_2014_2026_with_top100.csv'
    
    @classmethod
    def ensure_directories(cls):
        """Create all required output directories."""
        for directory in [cls.DATA_DIR, cls.GENERATED_DIR, cls.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Verified directory exists: {directory}")


# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
class Columns:
    """Column name definitions for data validation and processing."""
    
    # Required columns by data type
    REQUIRED = {
        'batter_predictions': ['Name', 'IDfg', 'Age', 'wOBA'],
        'pitcher_predictions': ['Name', 'IDfg', 'Age', 'FIP'],
        'salary': ['player_name', 'player_id', 'payroll_annual', 'status']
    }
    
    # Output column sets
    HITTER_COLUMNS = [
        'Name', 'Age', 'G', 'PA', 'BB_bat', 'K_bat', 'IDfg', 'BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA',
        'wRC+', 'Bat', 'BsR', 'Def', 'WAR', 'HR', '2B', '3B', 'SB', 'CS', 'R', 'RBI'
    ]

    PITCHER_COLUMNS = [
        'Name', 'Age', 'GS', 'G', 'IP', 'BB_pit', 'K_pit', 'ER_pit', 'IDfg', 'ERA', 'FIP', 'K%', 'BB%', 'WAR',
        'HR%', 'HR/FB', 'K/9', 'BB/9', 'HR/9', 'FB%', 'GB%', 'SIERA',
    ]
    # ID column mappings (for transitioning from FG ID to MLB ID)
    # TODO: Complete migration to mlbam_id as primary identifier
    ID_COLUMNS = {
        'fangraphs': 'IDfg',
        'mlb': 'mlbam_id',
        'roster_fangraphs': 'fg_id',
        'salary': 'player_id'
    }


# =============================================================================
# PLAYER STATUS DEFINITIONS
# =============================================================================
class PlayerStatus(Enum):
    """Enum for player contract status."""
    PRE_ARB = "Pre-ARB"
    ARB1 = "ARB1"
    ARB2 = "ARB2"
    ARB3 = "ARB3"
    FREE_AGENT = "FA"
    SIGNED = "Signed"
    UNKNOWN = "Unknown"


# Status string mappings to normalize various input formats
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


# =============================================================================
# WAR CALCULATION CONSTANTS
# =============================================================================
class WARConstants:
    """Constants for WAR (Wins Above Replacement) calculations."""
    
    # Park factors by team (100 = neutral) — FanGraphs 5-year Basic park factors (2025)
    # Updated to match core/park_factors.py for consistency across neutralization and WAR calc
    BALLPARK_FACTORS = {
        'LAA': 101,  # Angel Stadium
        'BAL': 99,   # Oriole Park at Camden Yards
        'BOS': 104,  # Fenway Park
        'CHW': 100,  # Guaranteed Rate Field
        'CLE': 99,   # Progressive Field
        'DET': 100,  # Comerica Park
        'KC':  103,  # Kauffman Stadium
        'MIN': 101,  # Target Field
        'NYY': 99,   # Yankee Stadium
        'ATH': 103,  # Sutter Health Park (Sacramento)
        'SEA': 94,   # T-Mobile Park
        'TB':  101,  # Tropicana Field
        'TEX': 99,   # Globe Life Field
        'TOR': 99,   # Rogers Centre
        'ARI': 101,  # Chase Field
        'ATL': 100,  # Truist Park
        'CHC': 98,   # Wrigley Field
        'CIN': 105,  # Great American Ball Park
        'COL': 113,  # Coors Field
        'MIA': 101,  # loanDepot park
        'HOU': 99,   # Minute Maid Park
        'LAD': 99,   # Dodger Stadium
        'MIL': 99,   # American Family Field
        'WSH': 100,  # Nationals Park
        'NYM': 96,   # Citi Field
        'PHI': 101,  # Citizens Bank Park
        'PIT': 102,  # PNC Park
        'STL': 98,   # Busch Stadium
        'SD':  96,   # Petco Park
        'SF':  97,   # Oracle Park
    }
    
    # League constants for offensive calculations (2025 season)
    WOBA_SCALE = 1.232
    RPA = 0.118  # League runs per PA
    LG_WOBA = 0.313
    RPW = 9.774  # Runs per Win
    LG_PA = 186188  # League total PA
    LG_RUNS_PER_PA = 0.118
    LG_WRC_PER_PA = 0.117
    
    # 2025 wOBA weights for calculating wOBA from counting stats
    WOBA_WEIGHTS = {
        'wBB': 0.691,    # Unintentional walk
        'wHBP': 0.722,   # Hit by pitch
        'w1B': 0.882,    # Single
        'w2B': 1.252,    # Double
        'w3B': 1.584,    # Triple
        'wHR': 2.037     # Home run
    }
    
    # Positional adjustments (runs per 162 games)
    POSITIONAL_ADJUSTMENTS = {
        'C': 12.5,
        'SS': 7.5,
        '2B': 2.5,
        'CF': 2.5,
        '3B': 2.5,
        'LF': -7.5,
        'RF': -7.5,
        '1B': -12.5,
        'DH': -17.5
    }
    
    # Pitcher-specific constants
    LG_FIP = 4.14  # League average FIP (2024 actual ≈ 4.14)
    LG_RA9 = 4.50  # League average runs allowed per 9 IP (incl. unearned)
    REPLACEMENT_LEVEL_RUNS_200IP = 22.0  # Replacement level runs per 200 IP
    DEFAULT_IP_PER_START = 5.75  # Average IP per start (for dynamic RPW)
    DEFAULT_IP_PER_APPEARANCE_RP = 1.0  # Average IP per RP appearance
    
    # Default IP assumptions for projections
    DEFAULT_SP_IP = 180  # Standard full season for SP
    DEFAULT_RP_IP = 70   # Standard full season for RP
    BF_PER_IP = 4.33     # Average batters faced per inning (MLB average)
    
    # Team name to abbreviation mapping (for roster data)
    TEAM_ABBREVIATIONS = {
        'Athletics': 'ATH',
        'Pittsburgh Pirates': 'PIT',
        'San Diego Padres': 'SD',
        'Seattle Mariners': 'SEA',
        'San Francisco Giants': 'SF',
        'Arizona Diamondbacks': 'ARI',
        'Atlanta Braves': 'ATL',
        'Baltimore Orioles': 'BAL',
        'Boston Red Sox': 'BOS',
        'Chicago Cubs': 'CHC',
        'Chicago White Sox': 'CHW',
        'Cincinnati Reds': 'CIN',
        'Cleveland Guardians': 'CLE',
        'Colorado Rockies': 'COL',
        'Detroit Tigers': 'DET',
        'Houston Astros': 'HOU',
        'Kansas City Royals': 'KC',
        'Los Angeles Angels': 'LAA',
        'Los Angeles Dodgers': 'LAD',
        'Miami Marlins': 'MIA',
        'Milwaukee Brewers': 'MIL',
        'Minnesota Twins': 'MIN',
        'New York Mets': 'NYM',
        'New York Yankees': 'NYY',
        'Philadelphia Phillies': 'PHI',
        'St. Louis Cardinals': 'STL',
        'Tampa Bay Rays': 'TB',
        'Texas Rangers': 'TEX',
        'Toronto Blue Jays': 'TOR',
        'Washington Nationals': 'WSH'
    }


# =============================================================================
# CONTRACT/SALARY CONSTANTS
# =============================================================================
# =============================================================================
# CONVEX WAR VALUATION MODEL
# =============================================================================
class ConvexModel:
    """
    Empirically calibrated convex power-law model for WAR-to-dollar conversion.
    
    Formula:
        value = alpha * max(WAR, 0)^beta * (1 + inflation_rate)^(year - base_year)
    
    Parameters were derived by minimizing the median absolute trade imbalance
    across 744 MLB trades (2014-2024) using Nelder-Mead optimization on the
    joint (alpha, beta) surface.
    
    The convex shape (beta > 1) means each additional WAR is worth MORE than
    the last, reflecting real market dynamics:
        - Scarcity premium: elite players are rare and irreplaceable
        - Certainty premium: high-WAR players have lower variance
        - Optionality: surplus WAR can be traded for prospects/assets
        - Roster-slot cost: each player occupies a 26-man roster spot,
          so one 6-WAR player is worth more than two 3-WAR players
    
    Reference values (2025 dollars, alpha=$8.59M, beta=1.18, replacement floor=0.5):
        0.5 WAR =      $0       3 WAR  =  $24.5M
        1.0 WAR =   $3.8M       5 WAR  =  $49.2M
        2.0 WAR =  $13.8M       8 WAR  = $121.9M
    
    Usage:
        from value_determination.config import Config
        alpha, beta = Config.ConvexModel.load_calibration()
        value = Config.ConvexModel.calculate_value(war=4.5, year=2026)
    """
    
    # Default parameters (from trade analysis calibration 2014-2024)
    ALPHA_DEFAULT = 8_592_188   # Base $/WAR coefficient ($8.59M)
    BETA_DEFAULT = 1.18        # Convexity exponent (>1 = superlinear)
    
    # Path to trade-analysis calibration output (auto-generated by analyze_trades.py)
    CALIBRATION_FILE = Paths.GENERATED_DIR / 'trade_analysis' / 'results' / 'convex_calibration.json'
    
    @classmethod
    def load_calibration(cls) -> tuple:
        """
        Load calibrated (alpha, beta) from the trade analysis pipeline.
        
        Falls back to hardcoded defaults if the calibration file is missing
        or unreadable. This ensures the value determination module works
        independently of the trade analysis pipeline.
        
        Returns:
            Tuple of (alpha, beta) as floats.
        """
        import json
        try:
            if cls.CALIBRATION_FILE.exists():
                with open(cls.CALIBRATION_FILE, 'r') as f:
                    cal = json.load(f)
                alpha = float(cal.get('alpha', cls.ALPHA_DEFAULT))
                beta = float(cal.get('beta', cls.BETA_DEFAULT))
                logger.info(
                    f"Loaded convex calibration from {cls.CALIBRATION_FILE.name}: "
                    f"alpha=${alpha:,.0f}, beta={beta:.3f}"
                )
                return alpha, beta
        except Exception as e:
            logger.warning(f"Could not load calibration file: {e}")
        
        logger.info(
            f"Using default convex parameters: "
            f"alpha=${cls.ALPHA_DEFAULT:,.0f}, beta={cls.BETA_DEFAULT:.3f}"
        )
        return cls.ALPHA_DEFAULT, cls.BETA_DEFAULT
    
    @classmethod
    def calculate_value(cls, war: float, year: int,
                        alpha: float = None, beta: float = None) -> float:
        """
        Convert WAR to dollar value using the convex power-law model.

        Applies the replacement-level floor from ``TradeConfidence`` before
        the convex curve — only marginal WAR above the floor produces value.

        Args:
            war: WAR value (sub-replacement returns $0).
            year: Season year (for inflation adjustment from BASE_YEAR).
            alpha: Override alpha parameter (defaults to ALPHA_DEFAULT).
            beta: Override beta parameter (defaults to BETA_DEFAULT).

        Returns:
            Dollar value of player's WAR production for that year.
        """
        if alpha is None:
            alpha = cls.ALPHA_DEFAULT
        if beta is None:
            beta = cls.BETA_DEFAULT
        
        import math
        if war is None or (isinstance(war, float) and math.isnan(war)) or war <= 0:
            return 0.0
        
        inflation = (1 + ContractConstants.INFLATION_RATE) ** (year - ContractConstants.BASE_YEAR)
        return alpha * (war ** beta) * inflation


class ContractConstants:
    """Constants for contract value and salary calculations."""
    
    # Historical WAR dollar values by year
    HISTORICAL_WAR_VALUE = {
        2002: 4_800_000,
        2003: 4_800_000,
        2004: 4_800_000,
        2005: 4_800_000,
        2006: 5_200_000,
        2007: 5_700_000,
        2008: 6_200_000,
        2009: 6_400_000,
        2010: 6_000_000,
        2011: 7_500_000,
        2012: 6_500_000,
        2013: 7_400_000,
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
        2024: 8_200_000
    }
    
    # Current/default WAR value and inflation
    WAR_VALUE_DEFAULT = 8_200_000
    INFLATION_RATE = 0.04
    BASE_YEAR = 2025
    
    # WAR value tiers (for tiered calculations)
    WAR_VALUE_TIERS = {
        'tier1': {'max': 2, 'value': 8_000_000},
        'tier2': {'max': 4, 'value': 9_000_000},
        'tier3': {'value': 10_000_000}
    }
    
    # Minimum salaries by status
    MIN_SALARY = {
        'Pre-Arb': 720_000,
        'Arb-1': 1_000_000,
        'Arb-1 (Super 2)': 1_200_000,
        'Arb-2': 2_500_000,
        'Arb-3': 4_000_000,
        'Arb-4': 5_000_000
    }
    
    # Arbitration percentage of market value
    ARB_PERCENT = {
        'Arb-1': 0.15,
        'Arb-1 (Super 2)': 0.15,
        'Arb-2': 0.25,
        'Arb-3': 0.40,
        'Arb-4': 0.60
    }
    
    @classmethod
    def get_war_value(cls, year: int) -> float:
        """
        Get WAR dollar value for a specific year.
        
        Uses historical data for past years, applies inflation for future years.
        """
        if year in cls.HISTORICAL_WAR_VALUE:
            return cls.HISTORICAL_WAR_VALUE[year]
        
        # Apply inflation for future years
        years_from_base = year - cls.BASE_YEAR
        return cls.WAR_VALUE_DEFAULT * ((1 + cls.INFLATION_RATE) ** years_from_base)


# =============================================================================
# PROSPECT VALUATION CONSTANTS
# =============================================================================
class ProspectConstants:
    """Constants for prospect value calculations using FanGraphs methodology."""
    
    # FV (Future Value) grade base values in dollars
    # Based on FanGraphs research on prospect value
    FV_BASE_VALUES = {
        70: 120_000_000,  # Generational talent
        65: 90_000_000,  # Perennial All-Star
        60: 70_000_000,   # All-Star caliber
        55: 50_000_000,   # Above average regular
        50: 30_000_000,   # Average regular
        45: 15_000_000,   # Platoon/utility player
        40: 7_000_000,   # Fringe major leaguer
        35: 2_000_000,    # Organizational depth
        30: 1_000_000     # Minor league depth
    }
    
    # Rank adjustment factors (BONUS multipliers for top 100)
    # Top 100 prospects: bonus from 1.5 (rank 1) to 1.0 (rank 100)
    # This ensures top 100 prospects are NEVER valued less than non-top-100 with same FV
    RANK_ADJ_TOP100_MAX = 1.5   # Rank 1 gets 50% bonus
    RANK_ADJ_TOP100_MIN = 1.0   # Rank 100 gets no bonus (base FV value)
    RANK_ADJ_NON_TOP100 = 1.0   # Non-top-100 gets base FV value
    
    # Experience thresholds for prospect weight diminishing
    # These define when a player transitions from "prospect" to "established"
    # The weight diminishes linearly from 1.0 (no experience) to 0.0 (threshold reached)
    EXPERIENCE_THRESHOLD_GAMES = {
        'batter': 300,        # ~2 full seasons
        'sp': 45,             # ~1.5 seasons of starts
        'rp': 65,             # ~1.5 seasons of appearances
    }
    
    # Minimum games for ANY prospect adjustment (below this = pure prospect value)
    EXPERIENCE_MINIMUM_GAMES = {
        'batter': 0,
        'sp': 0,
        'rp': 0
    }
    
    @classmethod
    def calculate_prospect_weight(cls, games_played: float, position_type: str) -> float:
        """
        Calculate the prospect weight (0.0 to 1.0) based on MLB experience.
        
        As players gain experience, their value shifts from prospect-based to
        performance-based (MLB projections).
        
        Args:
            games_played: Total MLB games played
            position_type: 'batter', 'sp', or 'rp'
            
        Returns:
            Float from 0.0 (fully established) to 1.0 (pure prospect)
        """
        threshold = cls.EXPERIENCE_THRESHOLD_GAMES.get(position_type, 300)
        
        if games_played >= threshold:
            return 0.0  # Fully established player
        
        # Linear decrease from 1.0 to 0.0
        return max(0.0, 1.0 - (games_played / threshold))
    
    @classmethod
    def calculate_rank_adjustment(cls, rank: float) -> float:
        """
        Calculate the rank-based value adjustment multiplier.
        
        Top 100 prospects get a BONUS (1.0-1.5x) based on rank.
        Non-top-100 prospects get base FV value (1.0x).
        
        This ensures a top 100 prospect is NEVER valued less than a
        non-top-100 prospect with the same FV grade.
        
        Args:
            rank: Prospect ranking (1-100 for top 100, higher for org rankings)
            
        Returns:
            Multiplier between 1.0 and 1.5 for top 100, 1.0 for others
        """
        if rank <= 100:
            # Top 100: bonus from 1.5 (rank 1) down to 1.0 (rank 100)
            return cls.RANK_ADJ_TOP100_MAX - (rank - 1) * (
                cls.RANK_ADJ_TOP100_MAX - cls.RANK_ADJ_TOP100_MIN
            ) / 99  # Use 99 so rank 100 gets exactly 1.0
        else:
            # Non-top-100: base FV value only
            return cls.RANK_ADJ_NON_TOP100


# =============================================================================
# TRADE CONFIDENCE & REPLACEMENT LEVEL
# =============================================================================
class TradeConfidence:
    """
    Replacement-level floor and projection confidence for trade values.

    Two mechanisms adjust raw trade values to better reflect real market behavior:

    1. **Convex power-law valuation** — WAR is converted to dollar value using
       a convex curve (value = alpha * WAR^beta * inflation). Any positive WAR
       contributes trade value.

    2. **Projection confidence** — Players with limited MLB track records have
       their trade value blended between the performance projection and a
       prospect-grade prior.  Confidence scales linearly from
       ``CONFIDENCE_FLOOR`` (no games) to 1.0 (fully stabilised).
       Only players with a prospect FV grade receive blending; established
       players and non-prospects pass through at full confidence.

    All parameters are configurable here so the pipeline can be tuned without
    editing calculation code.
    """

    # -- Stabilisation thresholds (career games for full confidence) -----------
    # Below these thresholds trade value is blended with the prospect prior.
    # ~3 full batter seasons, ~2 SP seasons, ~1.5 RP seasons.
    STABILIZATION_GAMES: dict[str, int] = {
        'batter': 450,
        'sp':      65,   # starts
        'rp':     100,   # appearances
    }

    # Confidence range
    CONFIDENCE_FLOOR = 0.10   # 0-game prospect still gets 10 % performance weight
    CONFIDENCE_CEILING = 1.0  # fully stabilised — no blending

    # -- FV-grade prior (annual WAR expectation by prospect grade) -------------
    # Used to compute the prospect-side trade value for blending.
    FV_PRIOR_WAR: dict[int, float] = {
        70: 4.0,   # Generational — perennial MVP candidate
        65: 3.0,   # Perennial All-Star
        60: 2.5,   # All-Star caliber
        55: 2.0,   # Above-average regular
        50: 1.5,   # Average regular
        45: 1.0,   # Bench / utility
        40: 0.5,   # Fringe major-leaguer
    }
    DEFAULT_PRIOR_WAR = 1.0   # fallback for non-prospects with limited games

    # -- Recent-prospect window ------------------------------------------------
    # Only prospects ranked within this many years of CURRENT_YEAR are eligible
    # for confidence blending.  Older rankings are considered stale.
    PROSPECT_RECENCY_YEARS = 3

    # ── helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def calculate_confidence(cls, games: float, position_type: str) -> float:
        """
        Projection confidence from career MLB games.

        Returns a float in [CONFIDENCE_FLOOR, CONFIDENCE_CEILING].  Scales
        linearly between the two based on ``games / threshold``.
        """
        threshold = cls.STABILIZATION_GAMES.get(position_type, 450)
        if threshold <= 0:
            return cls.CONFIDENCE_CEILING
        raw = games / threshold
        return min(cls.CONFIDENCE_CEILING, max(cls.CONFIDENCE_FLOOR, raw))

    @classmethod
    def get_prior_war(cls, fv_grade) -> float:
        """
        Map a FanGraphs FV grade to an expected annual WAR prior.

        Handles '+' grades (e.g. '55+' → 57.5) and falls back to
        ``DEFAULT_PRIOR_WAR`` when the grade is missing or unrecognised.
        """
        import math
        if fv_grade is None or (isinstance(fv_grade, float) and math.isnan(fv_grade)):
            return cls.DEFAULT_PRIOR_WAR
        try:
            fv = float(str(fv_grade).replace('+', ''))
        except (TypeError, ValueError):
            return cls.DEFAULT_PRIOR_WAR
        # Find highest tier ≤ fv
        tiers = sorted(cls.FV_PRIOR_WAR.keys(), reverse=True)
        for t in tiers:
            if fv >= t:
                return cls.FV_PRIOR_WAR[t]
        return cls.DEFAULT_PRIOR_WAR


# =============================================================================
# PIPELINE SETTINGS
# =============================================================================
class PipelineSettings:
    """Settings controlling pipeline behavior."""
    
    # Year range for predictions (5 years for all players)
    PREDICTION_YEARS = range(2026, 2031)  # Save 5 years of projections for each player
    CURRENT_YEAR = 2026
    
    # Processing flags
    APPLY_PROSPECT_ADJUSTMENTS = True
    APPLY_PARK_FACTORS = True
    
    # Error handling
    FAIL_ON_MISSING_DATA = False  # If True, pipeline fails on missing data
    LOG_UNMATCHED_PLAYERS = True  # Log players that couldn't be matched
    
    # Output options
    EXPORT_INTERMEDIATE_FILES = False  # Save intermediate DataFrames
    VERBOSE_OUTPUT = True  # Print detailed progress


# =============================================================================
# UNIFIED CONFIG CLASS
# =============================================================================
class Config:
    """
    Unified access point for all configuration.
    
    Usage:
        from value_determination.config import Config
        
        # Access paths
        Config.Paths.DATA_DIR
        
        # Access WAR constants
        Config.WAR.LG_FIP
        
        # Get war value for year
        Config.Contracts.get_war_value(2025)
    """
    Paths = Paths
    Columns = Columns
    PlayerStatus = PlayerStatus
    StatusMappings = STATUS_MAPPINGS
    WAR = WARConstants
    Contracts = ContractConstants
    ConvexModel = ConvexModel
    Prospects = ProspectConstants
    TradeConfidence = TradeConfidence
    Pipeline = PipelineSettings
    
    # Convenience re-export of logger
    logger = logger


# =============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# =============================================================================
# These exports maintain compatibility with existing code that imports from constants.py
# TODO: Remove after full migration to Config class

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
