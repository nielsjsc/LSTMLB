"""
Salary Data Processing Module
=============================

Handles cleaning, normalization, and player ID integration for salary data.

Key responsibilities:
- Clean raw Sportrac salary data
- Normalize player names for matching
- Merge salary data with player IDs from predictions
- Handle various payroll formats and FA markers

TODO: mlbam_id Migration
    - Current: Uses name matching to find IDfg
    - Target: Use mlbam_id as primary match key
    - Sportrac data includes player_id which may be MLBAM
    - Need to validate and use this for direct matching
"""

import pandas as pd
import numpy as np
import unidecode
from typing import Tuple

from .config import Config, logger


def normalize_name(name: str) -> str:
    """
    Normalize player names by removing accents and standardizing format.
    
    Args:
        name: Raw player name
        
    Returns:
        Normalized uppercase name without accents
    """
    if pd.isna(name):
        return name
    return unidecode.unidecode(str(name)).upper().strip()


def clean_salary_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize salary data from Sportrac.
    
    Args:
        df: Raw salary data with payroll and status columns
        
    Returns:
        Cleaned salary data with standardized values
        
    Note:
        Handles multiple column naming conventions (snake_case vs Title Case)
    """
    logger.info("Starting salary data cleaning process")
    
    cleaned_df = df.copy()
    
    try:
        # Handle different column naming conventions
        name_col = 'player_name' if 'player_name' in cleaned_df.columns else 'Player Name'
        payroll_col = 'payroll_annual' if 'payroll_annual' in cleaned_df.columns else 'Payroll'
        status_col = 'status' if 'status' in cleaned_df.columns else 'Status'
        year_col = 'year' if 'year' in cleaned_df.columns else 'Year'
        team_col = 'team' if 'team' in cleaned_df.columns else 'Team'
        id_col = 'player_id' if 'player_id' in cleaned_df.columns else 'IDfg'
        
        # Remove non-player rows (options, buyouts, etc)
        cleaned_df = cleaned_df[~cleaned_df[name_col].str.contains(
            'OPT-OUT|UFA|PLAYER OPT|CLUB OPT',
            na=False,
            case=False
        )]
        
        # Clean Year column
        cleaned_df['Year'] = pd.to_numeric(cleaned_df[year_col], errors='coerce')
        cleaned_df = cleaned_df.dropna(subset=['Year'])
        
        # Clean Payroll column and detect FA markers
        payroll_str = cleaned_df[payroll_col].astype(str)
        
        # Mark rows where payroll_annual contains FA indicators
        is_fa_marker = payroll_str.str.contains('UFA|RFA|FA', case=False, na=False, regex=True)
        
        # Clean payroll values
        payroll = (payroll_str
                  .str.replace('$', '', regex=False)
                  .str.replace(',', '', regex=False)
                  .str.replace('-', '', regex=False))
        
        cleaned_df['Payroll'] = pd.to_numeric(payroll, errors='coerce')
        
        # Override Status for rows with FA markers in payroll column
        if status_col in cleaned_df.columns:
            cleaned_df.loc[is_fa_marker, status_col] = 'UFA'
        
        # Standardize column names
        cleaned_df['Player Name'] = cleaned_df[name_col]
        cleaned_df['Status'] = cleaned_df[status_col] if status_col in cleaned_df.columns else None
        cleaned_df['Team'] = cleaned_df[team_col]
        
        # Add IDfg if available
        if id_col in df.columns:
            cleaned_df['IDfg'] = df[id_col]
        
        # Status validation and cleaning
        if 'Status' in cleaned_df.columns and cleaned_df['Status'] is not None:
            status_counts = cleaned_df['Status'].value_counts()
            logger.info("\nStatus distribution:")
            logger.info(status_counts)
        
        # Generate summary statistics
        stats = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'valid_salary_rows': cleaned_df['Payroll'].notna().sum(),
            'min_salary': cleaned_df['Payroll'].min(),
            'max_salary': cleaned_df['Payroll'].max(),
            'mean_salary': cleaned_df['Payroll'].mean()
        }
        
        logger.info("\nSalary cleaning summary:")
        for key, value in stats.items():
            logger.info(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Select output columns
        output_cols = ['Player Name', 'Year', 'Team', 'Payroll', 'Status']
        if 'IDfg' in cleaned_df.columns:
            output_cols.append('IDfg')
            
        return cleaned_df[output_cols].copy()
        
    except Exception as e:
        logger.error(f"Error cleaning salary data: {str(e)}")
        raise


def create_player_reference(sp_df: pd.DataFrame,
                           rp_df: pd.DataFrame,
                           batter_df: pd.DataFrame) -> pd.DataFrame:
    """Create unified player reference with normalized names."""
    player_ref = pd.concat([
        sp_df[['Name', 'IDfg', 'position_group', 'Year']],
        rp_df[['Name', 'IDfg', 'position_group', 'Year']],
        batter_df[['Name', 'IDfg', 'position_group', 'Year']]
    ])
    
    player_ref['Name_Normalized'] = player_ref['Name'].apply(normalize_name)
    return player_ref


def merge_salary_with_ids(salary_df: pd.DataFrame, 
                         sp_df: pd.DataFrame,
                         rp_df: pd.DataFrame,
                         batter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge salary data with player reference, handling duplicate names.
    
    Args:
        salary_df: Cleaned salary data
        sp_df: Starting pitcher data
        rp_df: Relief pitcher data
        batter_df: Batter data
        
    Returns:
        Merged DataFrame with salary and ID information
    """
    # Create player reference
    player_ref = create_player_reference(sp_df, rp_df, batter_df)
    
    # Create copies to avoid modifying originals
    salary_df = salary_df.copy()
    player_ref = player_ref.copy()
    
    # Normalize salary data names
    salary_df['Name_Normalized'] = salary_df['Player Name'].apply(normalize_name)
    
    # Debug: Check Luis Garcia names before modifications
    print("\nBefore modifications:")
    print("Salary DF Luis Garcia entries:")
    luis_garcia_salary = salary_df[salary_df['Name_Normalized'].str.contains('LUIS GARCIA', na=False)]
    print(luis_garcia_salary[['Name_Normalized', 'Team', 'Year']].to_string())
    print("\nPlayer Ref Luis Garcia entries:")
    luis_garcia_ref = player_ref[player_ref['Name_Normalized'].str.contains('LUIS GARCIA', na=False)]
    print(luis_garcia_ref[['Name_Normalized', 'IDfg', 'position_group']].to_string())
    
    # Fix Luis Garcia names - normalize team abbreviations
    salary_df['Team_lower'] = salary_df['Team'].str.lower()
    mask_garcia_hou = (salary_df['Name_Normalized'] == 'LUIS GARCIA') & (salary_df['Team_lower'].str.contains('hou|astros', na=False))
    mask_garcia_wsh = (salary_df['Name_Normalized'] == 'LUIS GARCIA') & (salary_df['Team_lower'].str.contains('wsh|washington|nationals', na=False))
    
    salary_df.loc[mask_garcia_hou, 'Name_Normalized'] = 'LUIS GARCIA HOU'
    salary_df.loc[mask_garcia_wsh, 'Name_Normalized'] = 'LUIS GARCIA JR.'
    
    # Update player reference - active players
    player_ref.loc[player_ref['IDfg'] == 23735, 'Name_Normalized'] = 'LUIS GARCIA HOU'
    player_ref.loc[player_ref['Name_Normalized'] == 'LUIS GARCIA JR.', 'Name_Normalized'] = 'LUIS GARCIA JR.'
    
    # Handle FA players
    player_ref.loc[
        (player_ref['Name_Normalized'] == 'LUIS GARCIA') &
        (player_ref['position_group'] == 'RP'),
        'Name_Normalized'
    ] = 'LUIS GARCIA FA'
    
    # Fix Will Smith names
    player_ref.loc[player_ref['IDfg'] == 19197, 'Name_Normalized'] = 'WILL SMITH LAD'
    mask_smith_lad = (salary_df['Name_Normalized'] == 'WILL SMITH') & (salary_df['Team_lower'].str.contains('lad|dodgers', na=False))
    salary_df.loc[mask_smith_lad, 'Name_Normalized'] = 'WILL SMITH LAD'
    
    # FA player
    player_ref.loc[
        (player_ref['Name_Normalized'] == 'WILL SMITH') &
        (player_ref['position_group'] == 'RP'),
        'Name_Normalized'
    ] = 'WILL SMITH FA'
    
    # Clean up temp column
    salary_df = salary_df.drop('Team_lower', axis=1)
    
    # Debug: Check names after modifications
    print("\nAfter modifications:")
    print("Salary DF Luis Garcia entries:")
    luis_garcia_salary = salary_df[salary_df['Name_Normalized'].str.contains('LUIS GARCIA', na=False)]
    print(luis_garcia_salary[['Name_Normalized', 'Team', 'Year']].to_string())
    print("\nPlayer Ref Luis Garcia entries:")
    luis_garcia_ref = player_ref[player_ref['Name_Normalized'].str.contains('LUIS GARCIA', na=False)]
    print(luis_garcia_ref[['Name_Normalized', 'IDfg', 'position_group']].to_string())
    
    # Regular merge
    merged_df = player_ref.merge(
        salary_df[['Name_Normalized', 'Year', 'Team', 'Payroll', 'Status']],
        on=['Name_Normalized', 'Year'],
        how='left'
    )
    
    print("\nMerged DF Info:")
    print(f"Total rows: {len(merged_df)}")
    print(f"Null Payroll: {merged_df['Payroll'].isna().sum()}")
    print(f"Null Status: {merged_df['Status'].isna().sum()}")
    print(f"2025 rows: {len(merged_df[merged_df['Year'] == 2025])}")
    
    # Check each condition separately
    has_payroll = merged_df['Payroll'].notna()
    has_status = merged_df['Status'].notna()
    is_2025 = merged_df['Year'] == 2025
    
    print("\nCondition Counts:")
    print(f"Rows with Payroll: {has_payroll.sum()}")
    print(f"Rows with Status: {has_status.sum()}")
    print(f"Rows in 2025: {is_2025.sum()}")
    print(f"Rows meeting any condition: {(has_payroll | has_status | is_2025).sum()}")
    
    valid_data = merged_df[has_payroll | has_status | is_2025]
    
    return valid_data.drop('Name_Normalized', axis=1)
