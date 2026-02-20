"""
Configuration constants for playing time projection.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# ==============================================================================
# PATHS
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parents[2]  # LSTMLB/
DATA_DIR = ROOT_DIR / 'data'
PIPELINE_DIR = DATA_DIR / 'generated' / 'pipeline'
INJURY_DIR = DATA_DIR / 'injury'
ROSTER_DIR = DATA_DIR / 'active_roster'
PROSPECT_DIR = DATA_DIR / 'prospect_data'
OUTPUT_DIR = DATA_DIR / 'generated' / 'playing_time'

# ==============================================================================
# TEAM BUDGETS (per season)
# ==============================================================================

@dataclass
class TeamBudget:
    """Team-level playing time constraints."""
    # Position player games (162 game season)
    C_GAMES: int = 162
    INF_GAMES: int = 162 * 4      # 1B, 2B, SS, 3B
    OF_GAMES: int = 162 * 3       # LF, CF, RF
    DH_GAMES: int = 162
    TOTAL_POSITION_GAMES: int = 162 * 9
    
    # PA-based budgets (FanGraphs methodology)
    # Each position slot gets 700 PA, except C which gets 640
    C_PA: int = 640
    POSITION_PA: int = 700        # Per position slot
    INF_PA: int = 700 * 4         # 2800 total for 1B, 2B, SS, 3B
    OF_PA: int = 700 * 3          # 2100 total for LF, CF, RF
    DH_PA: int = 700
    TOTAL_PA: int = 640 + (700 * 8)  # 6240 total
    
    # PA per game approximation
    PA_PER_GAME: float = 4.3
    
    # Maximum individual player PA
    MAX_PLAYER_PA: int = 700      # Elite everyday player cap
    MAX_CATCHER_PA: int = 550     # Catchers need rest
    
    # Pitching innings
    SP_IP: float = 1450.0         # ~5 starters × 29 starts × 5.5 IP
    RP_IP: float = 500.0          # Bullpen remainder
    TOTAL_IP: float = 1458.0      # 162 games × 9 innings
    
    # Maximum individual player allocations
    MAX_CATCHER_GAMES: int = 140  # Catchers need rest
    MAX_POSITION_GAMES: int = 162 # Can't exceed season length
    MAX_SP_IP: float = 220.0      # Even elite aces rarely exceed this
    MAX_RP_IP: float = 80.0       # Elite closers/setup men cap


# ==============================================================================
# POSITION MAPPINGS
# ==============================================================================

# Map roster position codes to position groups
POSITION_CODE_MAP: Dict[str, str] = {
    # Catchers
    '2': 'C',
    'C': 'C',
    
    # Infielders
    '3': '1B',
    '4': '2B', 
    '5': '3B',
    '6': 'SS',
    '1B': '1B',
    '2B': '2B',
    '3B': '3B',
    'SS': 'SS',
    'I': 'INF',      # Generic infield
    'INF': 'INF',
    
    # Outfielders
    '7': 'LF',
    '8': 'CF',
    '9': 'RF',
    'LF': 'LF',
    'CF': 'CF',
    'RF': 'RF',
    'O': 'OF',       # Generic outfield
    'OF': 'OF',
    
    # Pitchers
    '1': 'P',
    'P': 'P',
    
    # DH
    'D': 'DH',
    'DH': 'DH',
    'Y': 'DH',       # Two-way player
}

# Map position names to groups
POSITION_NAME_MAP: Dict[str, str] = {
    'Catcher': 'C',
    'First Base': '1B',
    'Second Base': '2B',
    'Third Base': '3B',
    'Shortstop': 'SS',
    'Infielder': 'INF',
    'Outfielder': 'OF',
    'Left Field': 'LF',
    'Center Field': 'CF',
    'Right Field': 'RF',
    'Pitcher': 'P',
    'Designated Hitter': 'DH',
}

# Prospect position mapping
PROSPECT_POSITION_MAP: Dict[str, str] = {
    'C': 'C',
    'C/OF': 'C',
    'C/1B': 'C',
    '1B': '1B',
    '2B': '2B',
    '3B': '3B',
    'SS': 'SS',
    'SS/3B': 'SS',
    'INF': 'INF',
    'OF': 'OF',
    'LF': 'LF',
    'CF': 'CF',
    'RF': 'RF',
    'LHP': 'SP',
    'RHP': 'SP',
    'LHRP': 'RP',
    'RHRP': 'RP',
}

# Position groups for allocation
POSITION_GROUPS: Dict[str, List[str]] = {
    'C': ['C'],
    'INF': ['1B', '2B', 'SS', '3B', 'INF'],
    'OF': ['LF', 'CF', 'RF', 'OF'],
    'DH': ['DH'],
    'SP': ['SP'],
    'RP': ['RP'],
}

# Default games/IP by position (before allocation)
DEFAULT_PLAYING_TIME: Dict[str, float] = {
    'C': 135.0,
    '1B': 150.0,
    '2B': 150.0,
    'SS': 150.0,
    '3B': 150.0,
    'INF': 150.0,
    'LF': 150.0,
    'CF': 150.0,
    'RF': 150.0,
    'OF': 150.0,
    'DH': 150.0,
    'SP': 180.0,   # IP
    'RP': 65.0,    # IP
}


# ==============================================================================
# INJURY ADJUSTMENTS
# ==============================================================================

@dataclass  
class InjuryConfig:
    """Injury-related adjustment parameters."""
    
    # Major surgeries: (surgery_pattern, year1_multiplier, year2_multiplier)
    MAJOR_SURGERIES: Dict[str, tuple] = None
    
    # Injury history penalty
    IL_STINT_THRESHOLD: int = 3          # IL stints in lookback period
    IL_LOOKBACK_YEARS: int = 3
    HISTORY_PENALTY_BASE: float = 0.05   # 5% reduction
    RECURRENCE_PENALTY: float = 0.05     # Additional 5% for same injury type
    
    def __post_init__(self):
        self.MAJOR_SURGERIES = {
            'tommy john': (0.0, 0.6),           # 0 IP year 1, 60% year 2
            'ucl reconstruction': (0.0, 0.6),
            'shoulder surgery': (0.0, 0.7),
            'labrum': (0.0, 0.7),
            'knee surgery': (0.0, 0.75),
            'acl': (0.0, 0.5),
            'achilles': (0.0, 0.5),
            'back surgery': (0.0, 0.7),
            'hip surgery': (0.0, 0.7),
        }


# ==============================================================================
# TEAM NAME MAPPINGS
# ==============================================================================

TEAM_NAME_TO_ABBR: Dict[str, str] = {
    'Angels': 'LAA',
    'Los Angeles Angels': 'LAA',
    'Astros': 'HOU', 
    'Houston Astros': 'HOU',
    'Athletics': 'ATH',
    'Oakland Athletics': 'ATH',
    'Sacramento Athletics': 'ATH',
    'Blue Jays': 'TOR',
    'Toronto Blue Jays': 'TOR',
    'Braves': 'ATL',
    'Atlanta Braves': 'ATL',
    'Brewers': 'MIL',
    'Milwaukee Brewers': 'MIL',
    'Cardinals': 'STL',
    'St. Louis Cardinals': 'STL',
    'Cubs': 'CHC',
    'Chicago Cubs': 'CHC',
    'Diamondbacks': 'ARI',
    'Arizona Diamondbacks': 'ARI',
    'Dodgers': 'LAD',
    'Los Angeles Dodgers': 'LAD',
    'Giants': 'SF',
    'San Francisco Giants': 'SF',
    'Guardians': 'CLE',
    'Cleveland Guardians': 'CLE',
    'Mariners': 'SEA',
    'Seattle Mariners': 'SEA',
    'Marlins': 'MIA',
    'Miami Marlins': 'MIA',
    'Mets': 'NYM',
    'New York Mets': 'NYM',
    'Nationals': 'WSH',
    'Washington Nationals': 'WSH',
    'Orioles': 'BAL',
    'Baltimore Orioles': 'BAL',
    'Padres': 'SD',
    'San Diego Padres': 'SD',
    'Phillies': 'PHI',
    'Philadelphia Phillies': 'PHI',
    'Pirates': 'PIT',
    'Pittsburgh Pirates': 'PIT',
    'Rangers': 'TEX',
    'Texas Rangers': 'TEX',
    'Rays': 'TB',
    'Tampa Bay Rays': 'TB',
    'Red Sox': 'BOS',
    'Boston Red Sox': 'BOS',
    'Reds': 'CIN',
    'Cincinnati Reds': 'CIN',
    'Rockies': 'COL',
    'Colorado Rockies': 'COL',
    'Royals': 'KC',
    'Kansas City Royals': 'KC',
    'Tigers': 'DET',
    'Detroit Tigers': 'DET',
    'Twins': 'MIN',
    'Minnesota Twins': 'MIN',
    'White Sox': 'CHW',
    'Chicago White Sox': 'CHW',
    'Yankees': 'NYY',
    'New York Yankees': 'NYY',
}


# ==============================================================================
# MAIN CONFIG CLASS
# ==============================================================================

class Config:
    """Central configuration object."""
    
    # Paths
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    PIPELINE_DIR = PIPELINE_DIR
    INJURY_DIR = INJURY_DIR
    ROSTER_DIR = ROSTER_DIR
    PROSPECT_DIR = PROSPECT_DIR
    OUTPUT_DIR = OUTPUT_DIR
    
    # Data files
    BATTER_PREDICTIONS = PIPELINE_DIR / 'batter_predictions.csv'
    PITCHER_PREDICTIONS = PIPELINE_DIR / 'pitcher_predictions.csv'
    FIELDING_PREDICTIONS = PIPELINE_DIR / 'fielding_predictions.csv'
    BASERUNNING_PREDICTIONS = PIPELINE_DIR / 'baserunning_predictions.csv'
    INJURY_DATA = INJURY_DIR / 'fangraphs_injury_data.csv'
    ROSTER_DATA = ROSTER_DIR / 'current_rosters.csv'
    PROSPECT_DATA = PROSPECT_DIR / 'prospects_2014_2026_with_top100.csv'
    
    # Budgets
    budget = TeamBudget()
    
    # Injury config
    injury = InjuryConfig()
    
    # Mappings
    POSITION_CODE_MAP = POSITION_CODE_MAP
    POSITION_NAME_MAP = POSITION_NAME_MAP
    PROSPECT_POSITION_MAP = PROSPECT_POSITION_MAP
    POSITION_GROUPS = POSITION_GROUPS
    DEFAULT_PLAYING_TIME = DEFAULT_PLAYING_TIME
    TEAM_NAME_TO_ABBR = TEAM_NAME_TO_ABBR
    
    # Projection settings
    CURRENT_YEAR = 2026
    PROJECTION_YEARS = range(2026, 2041)
