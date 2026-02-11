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
        'Name', 'Age', 'G', 'IDfg', 'BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA',
        'wRC+', 'Off', 'BsR', 'Def', 'WAR', 'HR', '2B', '3B', 'SB', 'CS', 'R', 'RBI'
    ]
    
    PITCHER_COLUMNS = [
        'Name', 'Age', 'GS', 'G', 'IDfg', 'ERA', 'FIP', 'K%', 'BB%', 'WAR', 'SIERA'
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
    
    # Park factors by team (100 = neutral)
    BALLPARK_FACTORS = {
        'COL': 108,  # Coors Field
        'BOS': 107,  # Fenway Park  
        'CIN': 105,  # Great American Ball Park
        'BAL': 104,  # Oriole Park at Camden Yards
        'TEX': 104,  # Globe Life Field
        'NYY': 103,  # Yankee Stadium
        'MIN': 103,  # Target Field
        'PHI': 103,  # Citizens Bank Park
        'HOU': 102,  # Minute Maid Park
        'TOR': 102,  # Rogers Centre
        'ARI': 101,  # Chase Field
        'ATL': 101,  # Truist Park
        'WSH': 101,  # Nationals Park
        'LAA': 100,  # Angel Stadium
        'CLE': 100,  # Progressive Field
        'DET': 100,  # Comerica Park
        'KC': 100,   # Kauffman Stadium
        'LAD': 100,  # Dodger Stadium
        'NYM': 100,  # Citi Field
        'STL': 100,  # Busch Stadium
        'CHW': 99,   # Guaranteed Rate Field
        'OAK': 99,   # Oakland Coliseum
        'PIT': 99,   # PNC Park
        'SF': 98,    # Oracle Park
        'MIA': 98,   # loanDepot park
        'MIL': 97,   # American Family Field
        'CHC': 97,   # Wrigley Field
        'SD': 96,    # Petco Park
        'TB': 96,    # Tropicana Field
        'SEA': 91    # T-Mobile Park
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
    LG_FIP = 4.20  # League average FIP
    REPLACEMENT_LEVEL_RUNS_200IP = 20.0  # Replacement level runs per 200 IP
    
    # Default IP assumptions for projections
    DEFAULT_SP_IP = 180  # Standard full season for SP
    DEFAULT_RP_IP = 70   # Standard full season for RP
    
    # Team name to abbreviation mapping (for roster data)
    TEAM_ABBREVIATIONS = {
        'Athletics': 'OAK',
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
    Prospects = ProspectConstants
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
