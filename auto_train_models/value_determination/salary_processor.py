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
    normalized = unidecode.unidecode(str(name)).upper().strip()
    
    # Remove FA suffix that sometimes appears in salary data
    if normalized.endswith(' FA'):
        normalized = normalized[:-3].strip()
    
    return normalized


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
        # First remove parenthetical notes like "(Arb. Midpoint)"
        payroll = payroll_str.str.split('(').str[0]
        payroll = (payroll
                  .str.replace('$', '', regex=False)
                  .str.replace(',', '', regex=False)
                  .str.replace('-', '', regex=False))
        
        cleaned_df['Payroll'] = pd.to_numeric(payroll, errors='coerce')
        
        # Override Status for rows with FA markers in payroll column
        if status_col in cleaned_df.columns:
            cleaned_df.loc[is_fa_marker, status_col] = 'UFA'
        
        # Clean years_of_service for downstream pre-arb / arb classification
        yos_col = 'years_of_service' if 'years_of_service' in cleaned_df.columns else 'Years of Service'
        if yos_col in cleaned_df.columns:
            cleaned_df['Years_of_Service'] = pd.to_numeric(
                cleaned_df[yos_col].astype(str).str.replace('-', '', regex=False).str.strip(),
                errors='coerce'
            )
        else:
            cleaned_df['Years_of_Service'] = np.nan
        
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
        
        # Deduplicate (player_id, Year) pairs — same player can appear multiple times because:
        # (1) Traded players are scraped from both old and new team's Spotrac roster page
        #     (e.g. Devers appears under both Red Sox and Giants after a mid-season trade).
        # (2) Spotrac player pages contain multiple HTML contract tables that the scraper
        #     can parse independently, producing two rows from one page.
        #
        # Strategy: group by (Spotrac player_id, Year), sum Payroll (prorated amounts
        # across two teams correctly sum to the full-year salary obligation), and take
        # the first non-null value for all categorical columns.
        #
        # Team is kept for name-disambiguation purposes in merge_salary_with_ids()
        # (e.g. distinguishing Luis Garcia HOU vs Luis Garcia WSH). It is NOT the
        # canonical team source — that comes from current_rosters.csv in main.py.
        if 'IDfg' in cleaned_df.columns:  # IDfg here holds the raw Spotrac player_id
            def _agg_salary_dupes(g):
                return pd.Series({
                    'Player Name': g['Player Name'].dropna().iloc[0] if g['Player Name'].notna().any() else np.nan,
                    'Team': g['Team'].dropna().iloc[0] if g['Team'].notna().any() else np.nan,
                    'Payroll': g['Payroll'].sum() if g['Payroll'].notna().any() else np.nan,
                    'Status': g['Status'].dropna().iloc[0] if g['Status'].notna().any() else np.nan,
                    'Years_of_Service': g['Years_of_Service'].dropna().iloc[0] if g['Years_of_Service'].notna().any() else np.nan,
                })
            pre_dedup = len(cleaned_df)
            cleaned_df = (
                cleaned_df.groupby(['IDfg', 'Year'])
                .apply(_agg_salary_dupes)
                .reset_index()  # brings IDfg and Year back as columns
            )
            logger.info(
                f"Deduplication: {pre_dedup} → {len(cleaned_df)} rows "
                f"({pre_dedup - len(cleaned_df)} duplicates removed)"
            )

        output_cols = ['Player Name', 'Year', 'Team', 'Payroll', 'Status', 'Years_of_Service']
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

    # Fix Luis Garcia names - normalize team abbreviations
    # Two different players share this name: IDfg 23735 (HOU pitcher) and Luis Garcia Jr. (WSH batter).
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

    # Use OUTER merge to keep both prediction years AND salary-only years (for long contracts like Soto's 2031-2039)
    # Team is intentionally excluded — canonical team assignment happens in main.py via current_rosters.csv.
    salary_merge_cols = ['Name_Normalized', 'Year', 'Payroll', 'Status']
    if 'Years_of_Service' in salary_df.columns:
        salary_merge_cols.append('Years_of_Service')
    merged_df = player_ref.merge(
        salary_df[salary_merge_cols],
        on=['Name_Normalized', 'Year'],
        how='outer'
    )
    
    # For rows that came from salary but not predictions (salary-only years beyond 2030,
    # or pre-arb players whose salary rows have Status but no Payroll),
    # fill in IDfg and position_group by matching on Name_Normalized.
    missing_id_mask = merged_df['IDfg'].isna() & (merged_df['Payroll'].notna() | merged_df['Status'].notna())
    if missing_id_mask.any():
        # Create lookup: Name_Normalized -> (IDfg, position_group, Name)
        id_lookup = player_ref[['Name_Normalized', 'IDfg', 'position_group', 'Name']].drop_duplicates('Name_Normalized')
        
        for idx in merged_df[missing_id_mask].index:
            name_norm = merged_df.loc[idx, 'Name_Normalized']
            match = id_lookup[id_lookup['Name_Normalized'] == name_norm]
            if len(match) > 0:
                merged_df.loc[idx, 'IDfg'] = match.iloc[0]['IDfg']
                merged_df.loc[idx, 'position_group'] = match.iloc[0]['position_group']
                merged_df.loc[idx, 'Name'] = match.iloc[0]['Name']
    
    logger.info(
        f"Salary merge: {len(merged_df)} rows total, "
        f"{merged_df['IDfg'].notna().sum()} with predictions, "
        f"{merged_df['Payroll'].notna().sum()} with payroll data"
    )

    # Keep rows that have predictions (IDfg not null — includes FAs without salary)
    # or rows with salary data (for long contracts beyond prediction years, e.g. Soto 2031-2039).
    has_prediction = merged_df['IDfg'].notna()
    has_payroll = merged_df['Payroll'].notna()
    has_status = merged_df['Status'].notna()
    valid_data = merged_df[has_prediction | has_payroll | has_status]

    # Drop the internal name-matching helper. Team is also dropped here — the canonical
    # team source (current_rosters.csv) is attached in main.py after this function returns.
    return valid_data.drop(columns=['Name_Normalized'], errors='ignore')
