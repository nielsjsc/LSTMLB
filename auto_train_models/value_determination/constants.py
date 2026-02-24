"""
Legacy Constants Module
=======================

DEPRECATED: Use config.py instead.

This file is maintained for backward compatibility only.
All new code should import from config.py:

    from value_determination.config import Config, logger
    
    # Access values via Config class
    data_dir = Config.Paths.DATA_DIR
    war_value = Config.Contracts.get_war_value(2025)

The exports below are re-exports from config.py to maintain compatibility
with existing code that imports from constants.py.
"""

# Re-export everything from config for backward compatibility
from .config import (
    # Logger
    logger,
    
    # Path constants
    ROOT_DIR,
    DATA_DIR,
    PIPELINE_DIR,
    HISTORIC_MLB_DIR,
    SALARY_DIR,
    GENERATED_DIR,
    OUTPUT_DIR,
    
    # Column definitions
    HITTER_COLUMNS,
    PITCHER_COLUMNS,
    
    # WAR constants
    BALLPARK_FACTORS,
    WOBA_SCALE,
    RPA,
    LG_WOBA,
    RPW,
    LG_FIP,
    
    # Contract constants
    WAR_VALUE,
    HISTORICAL_WAR_VALUE,
    get_war_value,
    
    # Utility functions
    ensure_directories,
    
    # Pipeline settings
    PREDICTION_YEARS,
    
    # Enums and mappings
    PlayerStatus,
    STATUS_MAPPINGS,
)

# Additional exports for backward compatibility
from .config import Config

REQUIRED_COLUMNS = Config.Columns.REQUIRED
WAR_VALUE_TIERS = Config.Contracts.WAR_VALUE_TIERS
INFLATION_RATE = Config.Contracts.INFLATION_RATE
BASE_YEAR = Config.Contracts.BASE_YEAR
MIN_SALARY = Config.Contracts.MIN_SALARY
ARB_PERCENT = Config.Contracts.ARB_PERCENT

# Convex model exports
ConvexModel = Config.ConvexModel
CONVEX_ALPHA_DEFAULT = Config.ConvexModel.ALPHA_DEFAULT
CONVEX_BETA_DEFAULT = Config.ConvexModel.BETA_DEFAULT

# Type alias
PathLike = str
