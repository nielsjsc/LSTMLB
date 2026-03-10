"""
Data Loading Module
===================

Handles loading and validation of prediction, salary, and historical data.

This module is responsible for:
- Loading pitcher/batter prediction files
- Loading salary data
- Merging and validating data sources
- Loading historical MLB data

TODO: Add mlbam_id support
    - Load mlbam_id from predictions alongside IDfg
    - Prefer mlbam_id for matching when available
    - Fall back to name matching when IDs unavailable
"""

import pandas as pd
from pathlib import Path
from typing import Tuple

# Import from central config
from .config import Config, logger

# Backward compatibility
PIPELINE_DIR = Config.Paths.PIPELINE_DIR
PREDICTION_YEARS = Config.Pipeline.PREDICTION_YEARS
REQUIRED_COLUMNS = Config.Columns.REQUIRED
SALARY_DIR = Config.Paths.SALARY_DIR
HISTORIC_MLB_DIR = Config.Paths.HISTORIC_MLB_DIR


def validate_files_exist(directory: Path, filename: str) -> None:
    """Validate prediction files exist."""
    if not (directory / filename).exists():
        raise FileNotFoundError(f"Missing file: {filename} in {directory}")


def load_prediction_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process consolidated prediction files from pipeline directory.
    
    Returns:
        Tuple containing (sp_data, rp_data, batter_data, baserunning_data, fielding_data, salary_data)
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If required columns are missing
        
    TODO: Track mlbam_id alongside IDfg for future migration
    """
    # Validate files exist
    required_files = [
        'pitcher_predictions.csv',
        'batter_predictions.csv',
        'baserunning_predictions.csv',
        'fielding_predictions.csv'
    ]
    for file in required_files:
        validate_files_exist(PIPELINE_DIR, file)
    
    try:
        # Load pitcher data
        pitcher_df = pd.read_csv(PIPELINE_DIR / 'pitcher_predictions.csv')
        
        # Validate required columns for pitchers (WAR calculated in pipeline)
        required_pitcher_cols = set(REQUIRED_COLUMNS['pitcher_predictions'])
        missing_cols = required_pitcher_cols - set(pitcher_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in pitcher_predictions.csv: {missing_cols}")
        
        # Handle Season vs Year column
        if 'Season' in pitcher_df.columns:
            pitcher_df['Year'] = pitcher_df['Season']
        
        # Add prediction_year (no filtering - keep all years in file)
        pitcher_df['prediction_year'] = pitcher_df['Year']
        
        # Split into SP and RP
        sp_data = pitcher_df[pitcher_df['Role'] == 'SP'].copy()
        rp_data = pitcher_df[pitcher_df['Role'] == 'RP'].copy()
        
        # Add position grouping
        sp_data['position_group'] = 'SP'
        rp_data['position_group'] = 'RP'
        sp_data['Position'] = sp_data['position_group']
        rp_data['Position'] = rp_data['position_group']
        
        # Load raw batter prediction files (WAR will be calculated in main.py)
        batter_data = pd.read_csv(PIPELINE_DIR / 'batter_predictions.csv')
        baserunning_data = pd.read_csv(PIPELINE_DIR / 'baserunning_predictions.csv')
        fielding_data = pd.read_csv(PIPELINE_DIR / 'fielding_predictions.csv')
        
        # Validate required columns for batters
        # Only core prediction columns needed (WAR calculated later)
        core_batter_cols = {'Name', 'IDfg', 'Year', 'Age', 'wOBA', 'BB%', 'K%', 'AVG', 'OBP', 'SLG'}
        missing_cols = core_batter_cols - set(batter_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in batter_predictions.csv: {missing_cols}")
        
        # Add prediction_year (no filtering - keep all years in file)
        batter_data['prediction_year'] = batter_data['Year']
        batter_data['position_group'] = 'POS'
        
        # Load salary data
        salary_data = pd.read_csv(SALARY_DIR / 'mlb_salary_data.csv')
        
        min_year = min(sp_data['Year'].min(), rp_data['Year'].min(), batter_data['Year'].min())
        max_year = max(sp_data['Year'].max(), rp_data['Year'].max(), batter_data['Year'].max())
        logger.info(f"Loaded {len(sp_data)} SP predictions, {len(rp_data)} RP predictions, "
                   f"{len(batter_data)} batter predictions for years {min_year}-{max_year}")
        logger.info(f"Loaded {len(baserunning_data)} baserunning predictions, {len(fielding_data)} fielding predictions")
        
        return sp_data, rp_data, batter_data, baserunning_data, fielding_data, salary_data
        
    except Exception as e:
        logger.error(f"Error loading prediction files: {str(e)}")
        raise


def merge_prediction_data(sp_df: pd.DataFrame, 
                          rp_df: pd.DataFrame, 
                          batter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge prediction datasets with validation.
    
    Args:
        sp_df: Starting pitcher predictions
        rp_df: Relief pitcher predictions
        batter_df: Batter predictions
        
    Returns:
        Combined DataFrame with all player predictions
    """
    # Core columns that must exist
    required_cols = [
        'Name', 'IDfg', 'position_group', 'Age',
        'prediction_year', 'WAR', 'Position'
    ]
    
    # Additional stat columns to preserve if they exist
    stat_cols = ['BsR', 'Def', 'Bat', 'G', 'PA', 'BB%', 'K%', 'AVG', 'OBP', 'SLG',
                 'wOBA', 'wRC+', 'HR', '2B', '3B', 'RBI', 'R', 'SB', 'CS', 'HBP', 'SF',
                 'ERA', 'FIP', 'SIERA', 'IP', 'GS', 'Role']
    
    logger.info(f"SP columns: {sp_df.columns.tolist()}")
    logger.info(f"RP columns: {rp_df.columns.tolist()}")
    logger.info(f"Batter columns: {batter_df.columns.tolist()}")
    
    # Combine datasets - keep all available columns
    all_cols = set(required_cols + stat_cols)
    player_predictions = pd.concat([
        sp_df[sp_df.columns.intersection(all_cols)],
        rp_df[rp_df.columns.intersection(all_cols)],
        batter_df[batter_df.columns.intersection(all_cols)]
    ], ignore_index=True)
    
    # Verify Position exists
    if 'Position' not in player_predictions.columns:
        raise ValueError("Position column lost during merge")
    
    # Validate no missing values
    missing_values = player_predictions.isnull().sum()
    if missing_values.any():
        logger.warning(f"\nMissing values found:\n{missing_values[missing_values > 0]}")
    
    logger.info(f"Successfully merged {len(player_predictions)} player predictions")
    
    return player_predictions


def load_historical_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load historical MLB batting and pitching data.
    
    Returns:
        Tuple of (batting_history, pitching_history)
    """
    batting_file = HISTORIC_MLB_DIR / 'mlb_batting_data_1950_2025.csv'
    pitching_file = HISTORIC_MLB_DIR / 'mlb_pitching_data_1950_2025.csv'
    
    # Try the original filenames first, then alternatives
    if not batting_file.exists():
        batting_file = HISTORIC_MLB_DIR / 'mlb_batting_data_2000_2024.csv'
    if not pitching_file.exists():
        pitching_file = HISTORIC_MLB_DIR / 'mlb_pitching_data_2000_2024.csv'
    
    batting_history = pd.read_csv(batting_file)
    pitching_history = pd.read_csv(pitching_file)
    
    # Compute HR% for pitching if not present (HR / TBF)
    if 'HR%' not in pitching_history.columns and 'HR' in pitching_history.columns and 'TBF' in pitching_history.columns:
        pitching_history['HR%'] = pitching_history['HR'] / pitching_history['TBF'].replace(0, float('nan'))
    
    logger.info(f"Loaded historical data: {len(batting_history)} batting, {len(pitching_history)} pitching records")
    
    return batting_history, pitching_history
