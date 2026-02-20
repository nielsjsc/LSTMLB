"""
Rate statistics configuration for data processing.

This file defines how to calculate rate-based statistics from raw counting stats.
Add new rate stat types here without modifying the core data processing code.
"""

from typing import Dict, List, Callable
import pandas as pd

# =============================================================================
# BATTING COUNTING STATS — per-150-games conversion
# =============================================================================
# Master list of ALL batting counting stats that require per-150-games scaling.
# calculate_rate_stats() divides each by G and multiplies by 150, overwriting
# the column in-place (no suffix).  This list is the single source of truth,
# referenced by both the data processing pipeline and the batter config.
#
# To add a new counting stat for batters:
#   1. Add it here
#   2. Add it to CLASSICAL_COUNTING_FEATURES (or STATCAST_COUNTING_FEATURES)
#      in configs/batter_config.py if you also want the model to train on it.
BATTING_COUNTING_STATS = [
    'HR', '2B', '3B', 'RBI', 'R', 'HBP', 'SF',
]
# Rate stat configurations
RATE_STAT_CONFIGS = {
    # Defensive stats (per 150 games)
    'defensive_per_150': {
        'condition': lambda df: 'G' in df.columns and any(col in df.columns for col in ['DRS', 'RngR', 'ErrR', 'DPR', 'UZR', 'ARM', 'FRM', 'rSB', 'rCERA']),
        'stats': ['DRS', 'RngR', 'ErrR', 'DPR', 'UZR', 'ARM', 'FRM', 'rSB', 'rCERA'],
        'denominator': 'G',
        'multiplier': 150,
        'suffix': '/150',
        'description': 'Defensive stats per 150 games'
    },
    
    # Baserunning stats (per 150 games)
    'baserunning_per_150': {
        'condition': lambda df: any(col in df.columns for col in ['wSB', 'SB', 'CS', 'sc_baserunning_runner_runs_tot', 'sc_baserunning_runner_runs_XB', 'sc_baserunning_runner_runs_SBX']),
        'stats': ['wSB', 'SB', 'CS', 'sc_baserunning_runner_runs_tot', 'sc_baserunning_runner_runs_XB', 'sc_baserunning_runner_runs_SBX'],
        'denominator': 'G',
        'multiplier': 150,
        'suffix': '_rate',
        'description': 'Baserunning stats per 150 games'
    },
    
    # Batting stats (per game)
    'batting_per_game': {
        'condition': lambda df: 'G' in df.columns and any(col in df.columns for col in ['HR', '2B', '3B', 'RBI', 'R']),
        'stats': ['HR', '2B', '3B', 'RBI', 'R'],
        'denominator': 'G', 
        'multiplier': 1,
        'suffix': '_rate',
        'description': 'Batting counting stats per game'
    },
    
    # Advanced batting rates (per plate appearance)
    'batting_per_pa': {
        'condition': lambda df: 'PA' in df.columns and any(col in df.columns for col in ['BB', 'K', 'SF', 'HBP']),
        'stats': ['BB', 'K', 'SF', 'HBP'],
        'denominator': 'PA',
        'multiplier': 1,
        'suffix': '_per_pa',
        'description': 'Batting discipline stats per plate appearance'
    },
    
    # Pitching stats (per 9 innings)
    'pitching_per_inning': {
        'condition': lambda df: 'IP' in df.columns and any(col in df.columns for col in ['H', 'ER', 'BB', 'K', 'HR']),
        'stats': ['H', 'ER', 'BB', 'K', 'HR'],
        'denominator': 'IP',
        'multiplier': 9,
        'suffix': '_per9',
        'description': 'Pitching stats per 9 innings'
    },
    
    # Add more configurations here as needed...
    # Example for contact quality:
    # 'contact_quality': {
    #     'condition': lambda df: 'PA' in df.columns and any(col in df.columns for col in ['Barrel%', 'HardHit%']),
    #     'stats': ['Barrel%', 'HardHit%'],
    #     'denominator': 'PA',
    #     'multiplier': 100,  # Convert to percentage
    #     'suffix': '_pct',
    #     'description': 'Contact quality percentages'
    # }
}

def get_rate_stat_suffixes() -> List[str]:
    """Get all possible rate stat suffixes for cleanup operations"""
    suffixes = []
    for config in RATE_STAT_CONFIGS.values():
        suffixes.append(config['suffix'])
    return suffixes

def get_expected_rate_columns(df: pd.DataFrame) -> List[str]:
    """Get list of expected rate columns based on available data"""
    expected_columns = []
    
    for config_name, config in RATE_STAT_CONFIGS.items():
        if config['condition'](df):
            for stat in config['stats']:
                if stat in df.columns:
                    expected_columns.append(f"{stat}{config['suffix']}")
    
    return expected_columns
