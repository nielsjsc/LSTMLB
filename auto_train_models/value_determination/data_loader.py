"""
Data loading and merging functionality for prediction data.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple

from .constants import (
    logger, PIPELINE_DIR, PREDICTION_YEARS, REQUIRED_COLUMNS,
    SALARY_DIR, HISTORIC_MLB_DIR
)


def validate_files_exist(directory: Path, filename: str) -> None:
    """Validate prediction files exist."""
    if not (directory / filename).exists():
        raise FileNotFoundError(f"Missing file: {filename} in {directory}")


def load_prediction_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process consolidated prediction files.
    
    Returns:
        Tuple containing (sp_data, rp_data, batter_data, salary_data)
    """
    # Validate files exist
    required_files = [
        'pitcher_predictions.csv',
        'batter_predictions.csv'
    ]
    for file in required_files:
        validate_files_exist(PIPELINE_DIR, file)
    
    try:
        # Load pitcher data
        pitcher_df = pd.read_csv(PIPELINE_DIR / 'pitcher_predictions.csv')
        
        # Validate required columns
        missing_cols = set(REQUIRED_COLUMNS['predictions']) - set(pitcher_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in pitcher_predictions.csv: {missing_cols}")
        
        # Handle Season vs Year column
        if 'Season' in pitcher_df.columns:
            pitcher_df['Year'] = pitcher_df['Season']
        
        # Filter for prediction years and add prediction_year
        pitcher_df = pitcher_df[pitcher_df['Year'].isin(PREDICTION_YEARS)]
        pitcher_df['prediction_year'] = pitcher_df['Year']
        
        # Split into SP and RP
        sp_data = pitcher_df[pitcher_df['Role'] == 'SP'].copy()
        rp_data = pitcher_df[pitcher_df['Role'] == 'RP'].copy()
        
        # Add position grouping
        sp_data['position_group'] = 'SP'
        rp_data['position_group'] = 'RP'
        sp_data['Position'] = sp_data['position_group']
        rp_data['Position'] = rp_data['position_group']
        
        # Load batter data (with WAR, Position, BsR, Def columns)
        batter_data = pd.read_csv(PIPELINE_DIR / 'batter_predictions_with_war.csv')
        
        # Validate required columns
        missing_cols = set(REQUIRED_COLUMNS['predictions']) - set(batter_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in batter_predictions.csv: {missing_cols}")
        
        # Filter years and add prediction_year
        batter_data = batter_data[batter_data['Year'].isin(PREDICTION_YEARS)]
        batter_data['prediction_year'] = batter_data['Year']
        batter_data['position_group'] = 'POS'
        
        # Load salary data
        salary_data = pd.read_csv(SALARY_DIR / 'mlb_salary_data.csv')
        
        logger.info(f"Loaded {len(sp_data)} SP predictions, {len(rp_data)} RP predictions, "
                   f"{len(batter_data)} batter predictions for years {min(PREDICTION_YEARS)}-{max(PREDICTION_YEARS)}")
        
        return sp_data, rp_data, batter_data, salary_data
        
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
    stat_cols = ['BsR', 'Def', 'Off', 'G', 'PA', 'BB%', 'K%', 'AVG', 'OBP', 'SLG',
                 'wOBA', 'wRC+', 'HR', '2B', '3B', 'RBI', 'R', 'SB', 'CS',
                 'HR_rate', '2B_rate', 'RBI_rate', 'R_rate',
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
    
    logger.info(f"Loaded historical data: {len(batting_history)} batting, {len(pitching_history)} pitching records")
    
    return batting_history, pitching_history
