"""
Value data export functionality.
"""

import pandas as pd
from pathlib import Path

from .constants import logger


def export_value_data(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export sorted value data by year.
    
    Args:
        df: DataFrame with all calculated values
        output_dir: Directory to export files to
    """
    logger.info("Starting value data export")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define column groups
    base_cols = [
        'Player Name', 'Team', 'Status', 'Position', 'Age', 'WAR',
        'Base_Value', 'Contract_Value', 'Surplus_Value', 'IDfg', 'mlb_id', 'Year', 'FA_Year',
        'Probable_FA_Year', 'Earliest_FA_Year', 'trade_value', 'contract_war', 'avg_war',
        'total_contract', 'avg_contract', 'total_surplus', 'years_control', 'control_through',
        'total_future_war', 'total_future_value', 'total_value', 'total_war',
        'historical_value', 'historical_war', 'contract_base_value'
    ]
    
    hitting_cols = [
        'BB%_bat', 'K%_bat', 'BB_bat', 'K_bat', 'G_bat', 'PA', 'AVG', 'OBP', 'SLG', 'OPS',
        'wOBA', 'wRC+', 'Off', 'BsR', 'Def', 'WAR_batter', 'HR', '2B', '3B', 'SB', 'CS', 'R', 'RBI'
    ]

    pitching_cols = [
        'ERA', 'FIP', 'K%_pit', 'BB%_pit', 'K_pit', 'BB_pit', 'ER_pit', 'WAR_pitcher', 'IP', 'GS', 'G_pit']
    
    export_cols = base_cols + hitting_cols + pitching_cols
    
    try:
        # Create copy for export
        export_df = df.copy()
        
        # Merge with roster data to get mlb_id (mlbam_id) from fg_id
        roster_path = output_dir.parent.parent / 'active_roster' / 'current_rosters.csv'
        if roster_path.exists():
            roster_df = pd.read_csv(roster_path)
            # fg_id in roster is string with '.0' suffix, IDfg in export is float
            roster_df['fg_id_clean'] = pd.to_numeric(roster_df['fg_id'], errors='coerce')
            id_map = roster_df.dropna(subset=['fg_id_clean']).drop_duplicates('fg_id_clean')[['mlbam_id', 'fg_id_clean']]
            export_df['IDfg_num'] = pd.to_numeric(export_df.get('IDfg', pd.Series()), errors='coerce')
            export_df = export_df.merge(
                id_map, left_on='IDfg_num', right_on='fg_id_clean', how='left'
            )
            export_df['mlb_id'] = export_df['mlbam_id'].apply(
                lambda x: int(x) if pd.notna(x) else None
            )
            export_df = export_df.drop(columns=['mlbam_id', 'fg_id_clean', 'IDfg_num'], errors='ignore')
            logger.info(f"Matched {export_df['mlb_id'].notna().sum()}/{len(export_df)} players to mlb_id")
        else:
            export_df['mlb_id'] = None
            logger.warning(f"Roster file not found at {roster_path}, mlb_id will be empty")
        
        # Rename Name column if exists
        if 'Name' in export_df.columns:
            export_df = export_df.rename(columns={'Name': 'Player_Name'})
        
        # Standardize column names for export
        column_mapping = {
            'contract_value': 'Contract_Value',
            'surplus_value': 'Surplus_Value',
            'probable_fa_year': 'Probable_FA_Year',
            'earliest_fa_year': 'Earliest_FA_Year'
        }
        export_df = export_df.rename(columns=column_mapping)
        
        # Sort data
        export_df = export_df.sort_values(['Year', 'Team', 'WAR'],
                                          ascending=[True, True, False])
        
        # Round numeric columns
        numeric_cols = ['Base_Value', 'Contract_Value', 'Surplus_Value', 'WAR']
        for col in numeric_cols:
            if col in export_df.columns:
                export_df[col] = export_df[col].round(2)
        
        # Export to single file
        output_file = output_dir / 'player_values_complete.csv'
        export_df.to_csv(output_file, index=False, na_rep='')
        
        logger.info(f"Exported {len(export_df)} records to {output_file}")
        
        # Print status distribution
        print("\nStatus Distribution:")
        if 'Status' in export_df.columns and 'Year' in export_df.columns:
            status_dist = export_df.groupby(['Year', 'Status']).size().unstack(fill_value=0)
            print(status_dist)
        
    except Exception as e:
        logger.error(f"Export process failed: {str(e)}")
        raise
