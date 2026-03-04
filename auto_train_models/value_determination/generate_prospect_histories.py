"""
Generate Prospect History File
================================

Creates a prospect history file from MLB.com prospect rankings data,
calculating prospect values based on FV grades and rankings.

This file is designed to be loaded by the backend and follows the
format of player_histories.csv.

Usage:
    python -m value_determination.generate_prospect_histories
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config, logger


# Team slug to MLB abbreviation mapping
# Updated to match backend abbreviations (SFG not SF, SDP not SD, etc.)
TEAM_ABBREVIATIONS = {
    'diamondbacks': 'ARI', 'dbacks': 'ARI',
    'braves': 'ATL',
    'orioles': 'BAL',
    'red-sox': 'BOS', 'redsox': 'BOS',
    'cubs': 'CHC',
    'white-sox': 'CHW', 'whitesox': 'CHW',
    'reds': 'CIN',
    'guardians': 'CLE', 'indians': 'CLE',
    'rockies': 'COL',
    'tigers': 'DET',
    'astros': 'HOU',
    'royals': 'KCR', 'kcroyals': 'KCR',
    'angels': 'LAA',
    'dodgers': 'LAD',
    'marlins': 'MIA',
    'brewers': 'MIL',
    'twins': 'MIN',
    'mets': 'NYM',
    'yankees': 'NYY',
    'athletics': 'ATH',  # Frontend/DB now uses ATH
    'phillies': 'PHI',
    'pirates': 'PIT',
    'padres': 'SDP', 'sdpadres': 'SDP',  # Backend uses SDP, not SD
    'giants': 'SFG', 'sfgiants': 'SFG',  # Backend uses SFG, not SF
    'mariners': 'SEA',
    'cardinals': 'STL',
    'rays': 'TBR', 'tbr': 'TBR',  # Backend uses TBR, not TB
    'rangers': 'TEX',
    'blue-jays': 'TOR', 'bluejays': 'TOR',
    'nationals': 'WSH', 'wsh': 'WSH'
}


def get_team_abbreviation(team_slug: str) -> str:
    """
    Convert team slug to standard 3-letter MLB abbreviation.
    
    Args:
        team_slug: Team slug from prospect data (e.g., 'pirates', 'red-sox')
        
    Returns:
        3-letter team abbreviation (e.g., 'PIT', 'BOS') or 'FA' if unknown
    """
    if pd.isna(team_slug):
        return 'FA'
    
    team_slug_lower = str(team_slug).lower().strip()
    return TEAM_ABBREVIATIONS.get(team_slug_lower, team_slug.upper()[:3])


def calculate_prospect_value(
    grade: float,
    top_100_rank: Optional[float],
    year: int
) -> float:
    """
    Calculate prospect value based on FV grade and top 100 rank.
    
    Uses the same logic as the main value calculation pipeline:
    - Base value from FV grade (grade_overall)
    - Multiplier adjustment based on top 100 rank (if applicable)
    - Org rank is NOT used for value calculation (not comparable across orgs)
    
    Args:
        grade: FV grade (30-70 scale)
        top_100_rank: MLB top 100 rank (1-100) if applicable, None otherwise
        year: Prospect list year
        
    Returns:
        Estimated prospect value in dollars
    """
    # Base value from FV grade
    # Use the FV to dollar conversion from config
    fv_values = Config.Prospects.FV_BASE_VALUES
    
    # Find closest FV grade
    if pd.isna(grade):
        return 0.0
    
    # Convert grade to nearest 5
    grade_rounded = round(grade / 5) * 5
    grade_rounded = max(30, min(70, grade_rounded))
    
    base_value = fv_values.get(int(grade_rounded), 0)
    
    if base_value == 0:
        return 0.0
    
    # Apply top 100 rank multiplier (only if they're in the top 100)
    # Org ranks are NOT used because they're not comparable across organizations
    rank_multiplier = 1.0
    if pd.notna(top_100_rank):
        # Use the rank adjustment from config
        rank_multiplier = Config.Prospects.calculate_rank_adjustment(top_100_rank)
    
    # Calculate final value
    value = base_value * rank_multiplier
    
    return value


def load_prospect_data() -> pd.DataFrame:
    """Load prospect data from CSV file."""
    prospect_file = Config.Paths.DATA_DIR / 'prospect_data' / 'prospects_2014_2026_with_top100.csv'
    
    if not prospect_file.exists():
        raise FileNotFoundError(f"Prospect data file not found: {prospect_file}")
    
    logger.info(f"Loading prospect data from {prospect_file}")
    df = pd.read_csv(prospect_file)
    logger.info(f"Loaded {len(df)} prospect records from {df['year'].min():.0f}-{df['year'].max():.0f}")
    
    return df


def fill_missing_fv_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing FV grades (grade_overall) for preliminary rankings.
    
    Strategy:
    A) If player has previous year data, use their previous FV grade
    B) If no previous data, interpolate based on ranking position
    
    Args:
        df: Prospect DataFrame with potentially missing grade_overall values
        
    Returns:
        DataFrame with filled grade_overall values
    """
    df = df.copy().reset_index(drop=True)
    
    # Get unique years sorted
    years = sorted(df['year'].dropna().unique())
    
    for year in years:
        year_mask = df['year'] == year
        
        # Count missing
        missing_count = df.loc[year_mask, 'grade_overall'].isna().sum()
        if missing_count == 0:
            continue
        
        logger.info(f"  Filling {missing_count} missing FV grades for {int(year)}")
        
        # Strategy A: Fill from previous year for same player
        filled_count = 0
        if year > min(years):
            prev_year = year - 1
            
            for idx in df[year_mask].index:
                if pd.notna(df.loc[idx, 'grade_overall']):
                    continue
                
                player_name = df.loc[idx, 'name']
                
                # Find this player's previous year data
                prev_data = df[(df['year'] == prev_year) & (df['name'] == player_name)]
                
                if len(prev_data) > 0 and pd.notna(prev_data.iloc[0]['grade_overall']):
                    df.loc[idx, 'grade_overall'] = prev_data.iloc[0]['grade_overall']
                    filled_count += 1
            
            if filled_count > 0:
                logger.info(f"    Filled {filled_count} from previous year")
        
        # Strategy B: Interpolate based on rank for remaining missing values
        year_indices = df[year_mask].index.tolist()
        year_subset = df.loc[year_indices].copy()
        
        # Sort by rank or top_100
        if 'top_100' in year_subset.columns:
            year_subset['sort_rank'] = year_subset['top_100'].fillna(year_subset['rank'])
        else:
            year_subset['sort_rank'] = year_subset['rank']
        
        year_subset = year_subset.sort_values('sort_rank')
        
        # Interpolate missing values
        year_subset['grade_overall'] = year_subset['grade_overall'].interpolate(
            method='linear', limit_direction='both'
        )
        
        # If still missing (all NaN case), use a default declining scale
        if year_subset['grade_overall'].isna().any():
            # Create a scale from 70 (rank 1) declining to 40 (rank 100+)
            year_subset['grade_overall'] = year_subset['grade_overall'].fillna(
                70 - (year_subset['sort_rank'] - 1) * 0.3
            )
            year_subset['grade_overall'] = year_subset['grade_overall'].clip(lower=40, upper=70)
        
        # Update main DataFrame
        for idx in year_subset.index:
            df.loc[idx, 'grade_overall'] = year_subset.loc[idx, 'grade_overall']
    
    return df


def _extract_mlbam_id(url) -> Optional[int]:
    """Extract the MLBAM player ID from an MLB.com prospect URL.

    URLs look like: ``https://www.mlb.com/milb/prospects/dbacks/ryan-waldschmidt-814439``
    """
    if not url or not isinstance(url, str) or pd.isna(url):
        return None
    m = re.search(r"-(\d+)$", url)
    return int(m.group(1)) if m else None


def generate_prospect_histories() -> pd.DataFrame:
    """
    Generate prospect history file matching player_histories.csv format.
    
    Creates columns for:
    - Basic info: Name, Team, Position, Year, Level, Age, FV, IDfg
    - Grade columns: Hit, Game, Raw, Spd, FB, SL, CMD, CB, CH
    - Yearly composites and values: {Year}_Composite, {Year}_Rank, {Year}_Value
    - MLB status flag: has_mlb
    
    Returns:
        DataFrame with prospect histories
    """
    # Load prospect data
    df = pd.read_csv(Config.Paths.DATA_DIR / 'prospect_data' / 'prospects_2014_2026_with_top100.csv')
    
    # Include all available years (2026 rankings are published by the season start)
    # df = df[df['year'] < 2026]  # Removed: 2026 data is now final
    
    # Filter to relevant years (last 5 years of data)
    years = sorted(df['year'].dropna().unique())[-5:]
    df = df[df['year'].isin(years)]
    
    logger.info(f"Processing {len(df)} prospects across years {min(years):.0f}-{max(years):.0f}")
    
    # Fill missing FV grades (for preliminary 2026 rankings)
    df = fill_missing_fv_grades(df)
    
    # Calculate prospect value for each record (using only top_100 rank, not org rank)
    df['prospect_value'] = df.apply(
        lambda row: calculate_prospect_value(
            row['grade_overall'],
            row.get('top_100'),
            row['year']
        ),
        axis=1
    )
    
    # Group by player and get latest data + historical values
    prospect_histories = []
    
    for name in df['name'].unique():
        player_df = df[df['name'] == name].sort_values('year')
        
        if len(player_df) == 0:
            continue
        
        # Get most recent record for basic info
        latest = player_df.iloc[-1]
        
        # Build base record
        record = {
            'Name': latest['name'],
            'Team': get_team_abbreviation(latest['team_slug']),
            'Position': latest['position'] if pd.notna(latest['position']) else 'Unknown',
            'Year': int(latest['year']),
            'Level': latest['level'] if pd.notna(latest['level']) else '',
            'Age': latest['age'] if pd.notna(latest['age']) else None,
            'FV': int(latest['grade_overall']) if pd.notna(latest['grade_overall']) else None,
            'IDfg': None,  # Resolved at data-load time from crosswalk
            'mlbam_id': _extract_mlbam_id(latest.get('prospect_url', '')),
            'Org_Rank': int(latest['rank']) if pd.notna(latest['rank']) else None,  # Org rank as info only
            
            # Hitting grades (split format like "30 / 40")
            'Hit': f"{int(latest['grade_hit'])} / {int(latest['grade_hit'])}" if pd.notna(latest.get('grade_hit')) else '',
            'Game': f"{int(latest['grade_power'])} / {int(latest['grade_power'])}" if pd.notna(latest.get('grade_power')) else '',
            'Raw': f"{int(latest['grade_power'])} / {int(latest['grade_power'])}" if pd.notna(latest.get('grade_power')) else '',
            'Spd': f"{int(latest['grade_run'])} / {int(latest['grade_run'])}" if pd.notna(latest.get('grade_run')) else '',
            
            # Pitching grades
            'FB': f"{int(latest['grade_fastball'])} / {int(latest['grade_fastball'])}" if pd.notna(latest.get('grade_fastball')) else '',
            'SL': f"{int(latest['grade_slider'])} / {int(latest['grade_slider'])}" if pd.notna(latest.get('grade_slider')) else '',
            'CMD': f"{int(latest['grade_control'])} / {int(latest['grade_control'])}" if pd.notna(latest.get('grade_control')) else '',
            'CB': f"{int(latest['grade_curveball'])} / {int(latest['grade_curveball'])}" if pd.notna(latest.get('grade_curveball')) else '',
            'CH': f"{int(latest['grade_changeup'])} / {int(latest['grade_changeup'])}" if pd.notna(latest.get('grade_changeup')) else '',
            
            'has_mlb': False  # Prospects haven't reached MLB yet in this dataset
        }
        
        # Add yearly values
        for _, year_row in player_df.iterrows():
            year_int = int(year_row['year'])
            
            # Top 100 rank (if applicable) - this is the MLB-wide ranking
            top_100 = year_row.get('top_100') if pd.notna(year_row.get('top_100')) else None
            record[f'{year_int}_Top100'] = top_100
            
            # Org rank (informational only, not used for sorting/value)
            record[f'{year_int}_OrgRank'] = year_row['rank'] if pd.notna(year_row['rank']) else None
            
            # FV grade for this year
            record[f'{year_int}_FV'] = int(year_row['grade_overall']) if pd.notna(year_row['grade_overall']) else None
            
            # Calculated value
            record[f'{year_int}_Value'] = year_row['prospect_value']
            
            # Composite ranking: only for top 100 prospects, otherwise None (avoids FV-based ties)
            record[f'{year_int}_Composite'] = float(top_100) if top_100 is not None else None
        
        prospect_histories.append(record)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(prospect_histories)
    
    # Sort by: top_100 rank first (if available), then by FV grade, then by value
    latest_top100_col = f'{int(max(years))}_Top100'
    latest_fv_col = f'{int(max(years))}_FV'
    latest_value_col = f'{int(max(years))}_Value'
    
    sort_cols = []
    if latest_top100_col in result_df.columns:
        sort_cols.append(latest_top100_col)
    if latest_fv_col in result_df.columns:
        sort_cols.append(latest_fv_col)
    if latest_value_col in result_df.columns:
        sort_cols.append(latest_value_col)
    
    if sort_cols:
        result_df = result_df.sort_values(
            sort_cols,
            ascending=[True] + [False] * (len(sort_cols) - 1),  # Top100 ascending, rest descending
            na_position='last'
        )
    
    logger.info(f"Generated {len(result_df)} prospect history records")
    
    return result_df


def save_prospect_histories(df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    """
    Save prospect histories to CSV file.
    
    Args:
        df: Prospect histories DataFrame
        output_path: Output file path (default: data/generated/MiLB/prospect_histories.csv)
    """
    if output_path is None:
        output_path = Config.Paths.DATA_DIR / 'generated' / 'MiLB' / 'prospect_histories.csv'
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved prospect histories to {output_path}")
    logger.info(f"  Total prospects: {len(df)}")
    
    # Print summary statistics
    latest_year = str(int(df.columns[df.columns.str.contains('_Value')].str.extract(r'(\d+)_Value')[0].max()))
    value_col = f'{latest_year}_Value'
    top100_col = f'{latest_year}_Top100'
    
    if value_col in df.columns:
        logger.info(f"  {latest_year} values: min=${df[value_col].min():,.0f}, "
                   f"max=${df[value_col].max():,.0f}, "
                   f"avg=${df[value_col].mean():,.0f}")
        
        if top100_col in df.columns:
            top100_count = df[top100_col].notna().sum()
            logger.info(f"  {latest_year} top 100 prospects: {top100_count}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Generating Prospect Histories")
    logger.info("=" * 60)
    
    try:
        # Generate prospect histories
        df = generate_prospect_histories()
        
        # Save to file
        save_prospect_histories(df)
        
        logger.info("\n" + "=" * 60)
        logger.info("Prospect histories generated successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to generate prospect histories: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
