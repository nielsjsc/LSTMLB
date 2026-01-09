#!/usr/bin/env python3
"""
WAR Calculation Post-Processing Script

This script combines batter, baserunning, and fielding predictions to calculate 
comprehensive WAR values, exactly matching the methodology used in the batter notebook.

Usage:
    python calculate_war.py [--year YEAR] [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants from batter notebook
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

# League constants (from batter notebook)
WOBA_SCALE = 1.23
RPA = 0.117  # League runs per PA
LG_WOBA = 0.309
RPW = 9.8  # Runs per Win
LG_PA = 186188  # League total PA
LG_RUNS_PER_PA = 0.114
LG_WRC_PER_PA = 0.117

# Positional adjustments (runs per 162 games)
POSITIONAL_ADJUSTMENTS = {
    'C': 12.5,
    'SS': 7.5,
    '2B': 2.5,
    'CF': 7.5,
    '3B': 0.0,
    'LF': -7.5,
    'RF': -7.5,
    '1B': -12.5,
    'DH': -17.5
}

def load_player_orgs(data_dir: Path) -> pd.DataFrame:
    """
    Load player organizations from current rosters file.
    Returns DataFrame with IDfg (fg_id) and their current team.
    """
    # Try to load from active_roster directory first (correct location)
    roster_file = data_dir.parent / "active_roster" / "current_rosters_with_fg_id.csv"
    
    # Fallback to data root directory
    if not roster_file.exists():
        roster_file = data_dir.parent / "current_rosters_with_fg_id.csv"
    
    if not roster_file.exists():
        logger.warning(f"Current rosters file not found at: {roster_file}")
        logger.warning("WAR calculations will use park factor of 1.0 for all players")
        return pd.DataFrame(columns=['IDfg', 'Team'])
    
    # Load roster data
    roster_df = pd.read_csv(roster_file)
    
    # Filter out players with no fg_id (fg_id == -1.0 means no mapping found)
    roster_df = roster_df[roster_df['fg_id'].notna() & (roster_df['fg_id'] != -1.0)]
    
    # Map team_name to team abbreviations for park factors
    team_mapping = {
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
        'San Diego Padres': 'SD',
        'San Francisco Giants': 'SF',
        'Seattle Mariners': 'SEA',
        'Tampa Bay Rays': 'TB',
        'Texas Rangers': 'TEX',
        'Toronto Blue Jays': 'TOR',
        'Washington Nationals': 'WSH'
    }
    
    # Create mapping from fg_id to team abbreviation
    roster_df['Team'] = roster_df['team_name'].map(team_mapping)
    
    # Select and rename columns to match expected format
    org_data = roster_df[['fg_id', 'Team']].rename(columns={'fg_id': 'IDfg'})
    
    # Convert IDfg to int to match prediction data format
    org_data['IDfg'] = org_data['IDfg'].astype(int)
    
    logger.info(f"Loaded {len(org_data)} player organizations from current rosters")
    
    return org_data

def calculate_wrc_plus(woba: float, team: str, pa: float, 
                      lg_runs_per_pa: float = LG_RUNS_PER_PA,
                      lg_wrc_per_pa: float = LG_WRC_PER_PA,
                      lg_woba: float = LG_WOBA,
                      woba_scale: float = WOBA_SCALE) -> float:
    """
    Calculate wRC+ using the proper formula with park factors.
    Exactly matches the calculation from the batter notebook.
    """
    # Calculate wRAA per PA
    wraa_per_pa = (woba - lg_woba) / woba_scale
    
    # Get park factor (default to 100 if no team/FA - meaning no adjustment)
    park_factor = BALLPARK_FACTORS.get(str(team).upper().strip(), 100) / 100
    
    # Calculate Park Adjustment
    park_adjustment = lg_runs_per_pa - (park_factor * lg_runs_per_pa)
    
    # Calculate the numerator for wRC+
    numerator = (wraa_per_pa + lg_runs_per_pa) + park_adjustment
    
    # Calculate wRC+
    wrc_plus = (numerator / lg_wrc_per_pa) * 100
    
    return wrc_plus

def calculate_baserunning_value(row: pd.Series, games: int) -> float:
    """
    Calculate baserunning value (BsR) from baserunning predictions.
    Uses rate statistics (per 150 games) and scales by actual games played.
    Now uses Statcast baserunning run values (XB + SBX components).
    """
    # Convert rates from per-150 to actual games
    # rates are now per 150 games, so: (rate / 150) * actual_games
    runner_runs_xb = row.get('sc_baserunning_runner_runs_XB_rate', 0) * (games / 150.0)
    runner_runs_sbx = row.get('sc_baserunning_runner_runs_SBX_rate', 0) * (games / 150.0)
    
    # BsR = extra base advancement + stolen base runs
    bsr = runner_runs_xb + runner_runs_sbx
    
    return bsr

def infer_position_from_fielding(fielding_df: pd.DataFrame, player_id: int, year: int) -> str:
    """
    Infer primary position from fielding data based on most innings played.
    Returns 'DH' if no fielding data exists.
    """
    player_fielding = fielding_df[(fielding_df['IDfg'] == player_id) & 
                                  (fielding_df['Year'] == year)]
    
    if player_fielding.empty:
        return 'DH'  # No fielding data = DH
    
    # Check which position column exists - try 'Pos' first
    pos_col = None
    if 'Pos' in player_fielding.columns:
        pos_col = 'Pos'
    elif 'Position' in player_fielding.columns:
        pos_col = 'Position'
    elif 'Position_Group' in player_fielding.columns:
        pos_col = 'Position_Group'
    else:
        # No position column found, default to DH
        return 'DH'
    
    # Map position values to standard positions (keep OF positions separate)
    position_map = {
        'C': 'C',
        'Catcher': 'C',
        '1B': '1B',
        'First Base': '1B',
        '2B': '2B', 
        'Second Base': '2B',
        '3B': '3B',
        'Third Base': '3B',
        'SS': 'SS',
        'Shortstop': 'SS',
        'LF': 'LF',
        'CF': 'CF', 
        'RF': 'RF',
        'OF': 'OF',
        'Outfield': 'OF'
    }
    
    # Find position with most innings played
    if 'Inn' in player_fielding.columns:
        max_inn_idx = player_fielding['Inn'].idxmax()
        primary_pos = player_fielding.loc[max_inn_idx, pos_col]
        return position_map.get(primary_pos, 'OF')
    else:
        # No innings column, use most common position
        pos_value = player_fielding[pos_col].mode()
        if not pos_value.empty:
            return position_map.get(pos_value.iloc[0], 'OF')
    
    return 'OF'

def calculate_defensive_value(fielding_data: pd.DataFrame, player_id: int, year: int) -> tuple[float, float]:
    """
    Calculate defensive value and positional adjustment from fielding predictions using Statcast FRV metrics.
    Accounts for multi-position players by weighting each position by innings played.
    The /150 metrics are per 150 GAMES, so we scale by games played at each position.
    
    Returns:
        tuple: (defensive_value, positional_adjustment)
    """
    # Get all fielding rows for this player-year
    player_fielding = fielding_data[(fielding_data['IDfg'] == player_id) & 
                                    (fielding_data['Year'] == year)]
    
    if player_fielding.empty:
        return 0.0, 0.0
    
    # Calculate total innings across all positions
    total_innings = player_fielding['Inn'].sum()
    
    if total_innings == 0 or pd.isna(total_innings):
        return 0.0, 0.0
    
    total_def_value = 0.0
    total_pos_adjustment = 0.0
    
    # Process each position the player played
    for idx, row in player_fielding.iterrows():
        position = row.get('Pos') or row.get('Position') or row.get('Position_Group', 'OF')
        innings = row.get('Inn', 0)
        
        # Calculate percentage of time at this position
        pct_at_position = innings / total_innings if total_innings > 0 else 0
        
        # Convert innings to games (9 innings = 1 game)
        games_at_position = innings / 9.0
        
        # Determine target games for extrapolation (135 for C, 150 for others)
        target_games = 135 if (position == 'C' or position.lower() == 'catcher') else 150
        
        # Extrapolate defensive value to target games
        if games_at_position > 0:
            extrapolation_factor = target_games / games_at_position
        else:
            extrapolation_factor = 1.0
        
        # The /150 metrics are per 150 GAMES, so scale by target games
        scaling_factor = target_games / 150.0
        
        # Map position for positional adjustment lookup
        pos_for_adjustment = position
        if position in ['LF', 'CF', 'RF']:
            # Use specific OF position for adjustment
            pos_for_adjustment = position
        elif position.lower() == 'outfield':
            pos_for_adjustment = 'OF'
        elif position.lower() == 'infield':
            pos_for_adjustment = '2B'
        elif position.lower() == 'catcher':
            pos_for_adjustment = 'C'
        
        # Calculate position-specific defensive value
        if position == 'C' or position.lower() == 'catcher':
            # Catchers: framing + throwing + blocking
            framing = row.get('sc_framing_runs/150', 0) * scaling_factor
            throwing = row.get('sc_throwing_runs/150', 0) * scaling_factor
            blocking = row.get('sc_blocking_runs/150', 0) * scaling_factor
            pos_value = framing + throwing + blocking
            
        elif position in ['1B', '2B', '3B', 'SS'] or position.lower() in ['infield', 'first base', 'second base', 'third base', 'shortstop']:
            # Infielders: range + arm + double play
            range_runs = row.get('sc_range_runs/150', 0) * scaling_factor
            arm_runs = row.get('sc_arm_runs/150', 0) * scaling_factor
            dp_runs = row.get('sc_dp_runs/150', 0) * scaling_factor
            pos_value = range_runs + arm_runs + dp_runs
            
        else:
            # Outfielders: range + arm
            range_runs = row.get('sc_range_runs/150', 0) * scaling_factor
            arm_runs = row.get('sc_arm_runs/150', 0) * scaling_factor
            pos_value = range_runs + arm_runs
        
        # Calculate positional adjustment for target games (extrapolated)
        # Adjustments are per 162 games, scale to target games
        pos_adj_per_162 = POSITIONAL_ADJUSTMENTS.get(pos_for_adjustment, 0.0)
        pos_adjustment = pos_adj_per_162 * (target_games / 162.0)
        
        # Weight by percentage of time at this position
        weighted_def = pos_value * pct_at_position
        weighted_pos = pos_adjustment * pct_at_position
        
        total_def_value += weighted_def
        total_pos_adjustment += weighted_pos
    
    return total_def_value, total_pos_adjustment

def calculate_war_components(row: pd.Series, baserunning_data: pd.DataFrame, 
                           fielding_data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate comprehensive WAR components combining all three prediction types.
    Exactly matches the methodology from the batter notebook.
    """
    # Infer position from fielding data
    player_id = row['IDfg']
    year = row['Year']
    position = infer_position_from_fielding(fielding_data, player_id, year)
    
    # Position-based games (catchers play fewer games)
    games = 135 if position == 'C' else 150
    pa = games * 4.2  # Standard PA per game
    
    # Get team for park factor
    team = row.get('Team', '')
    if pd.isnull(team):
        park_factor = 1.0
    else:
        park_factor = BALLPARK_FACTORS.get(str(team).upper().strip(), 100) / 100
    
    # Batting value calculation (wRAA + park adjustment)
    woba = row['wOBA']
    wraa = ((woba - LG_WOBA) / WOBA_SCALE) * pa
    batting_runs = wraa + (RPA - (RPA * park_factor)) * pa
    
    # Get baserunning value
    bsr_row = baserunning_data[(baserunning_data['IDfg'] == player_id) & 
                               (baserunning_data['Year'] == year)]
    
    if not bsr_row.empty:
        bsr = calculate_baserunning_value(bsr_row.iloc[0], games)
    else:
        # Use league average baserunning if no data (slightly negative for most players)
        bsr = -0.5
    
    # Get defensive value and positional adjustment (now handles multi-position players)
    fld_value, pos_adjustment = calculate_defensive_value(fielding_data, player_id, year)
    
    # Handle DHs - they have no fielding data but need DH positional adjustment
    if position == 'DH':
        # DHs get -17.5 positional adjustment for full season
        games_for_dh = 150  # DHs play 150 games
        pos_adjustment = POSITIONAL_ADJUSTMENTS.get('DH', 0.0) * (games_for_dh / 162.0)
    
    # Total defensive value = fielding + positional adjustment
    def_value = fld_value + pos_adjustment
    
    # Offensive value
    off = batting_runs + bsr
    
    # Replacement level
    rep_level = 570 * RPW * pa / LG_PA
    
    # RAR (Runs Above Replacement)
    rar = off + def_value + rep_level
    
    # WAR
    war = rar / RPW
    
    # Counting stats (scale rates by games)
    counting_stats = {}
    for stat in ['HR', '2B', 'RBI', 'R']:
        rate_col = f'{stat}_rate'
        if rate_col in row:
            counting_stats[stat] = round(row[rate_col] * games, 1)
        else:
            counting_stats[stat] = 0.0
    
    # Add baserunning counting stats (SB_rate and CS_rate are per 150 games)
    if not bsr_row.empty:
        counting_stats['SB'] = round(bsr_row.iloc[0].get('SB_rate', 0) * (games / 150.0), 1)
        counting_stats['CS'] = round(bsr_row.iloc[0].get('CS_rate', 0) * (games / 150.0), 1)
    else:
        counting_stats['SB'] = 0.0
        counting_stats['CS'] = 0.0
    
    return war, {
        'Off': off,
        'BsR': bsr,
        'Fld': fld_value,  # Raw Statcast FRV fielding runs
        'Pos': pos_adjustment,  # Positional adjustment
        'Def': def_value,  # Total defensive value (Fld + Pos)
        'WAR': war,
        'PA': pa,
        'G': games,
        'Position': position,
        'Team': team,
        **counting_stats
    }

def process_predictions(data_dir: Path, output_dir: Path, target_year: Optional[int] = None) -> None:
    """
    Main processing function that combines all predictions and calculates comprehensive WAR.
    """
    logger.info("Starting WAR calculation post-processing...")
    
    # Load all prediction files from pipeline subdirectory
    batter_file = data_dir / "pipeline" / "batter_predictions.csv"
    baserunning_file = data_dir / "pipeline" / "baserunning_predictions.csv"
    fielding_file = data_dir / "pipeline" / "fielding_predictions.csv"
    
    if not all(f.exists() for f in [batter_file, baserunning_file, fielding_file]):
        missing = [f for f in [batter_file, baserunning_file, fielding_file] if not f.exists()]
        raise FileNotFoundError(f"Missing prediction files: {missing}")
    
    logger.info("Loading prediction files...")
    batter_df = pd.read_csv(batter_file)
    baserunning_df = pd.read_csv(baserunning_file)
    fielding_df = pd.read_csv(fielding_file)
    
    logger.info(f"Loaded {len(batter_df)} batter predictions")
    logger.info(f"Loaded {len(baserunning_df)} baserunning predictions") 
    logger.info(f"Loaded {len(fielding_df)} fielding predictions")
    
    # Filter by year if specified
    if target_year:
        batter_df = batter_df[batter_df['Year'] == target_year]
        baserunning_df = baserunning_df[baserunning_df['Year'] == target_year]
        fielding_df = fielding_df[fielding_df['Year'] == target_year]
        logger.info(f"Filtered to {target_year}: {len(batter_df)} batters")
    
    # Load organization data for park factors
    org_data = load_player_orgs(data_dir)
    
    # Merge organization data with batter predictions
    batter_df = batter_df.merge(org_data, on='IDfg', how='left')
    
    # Calculate wRC+ with proper park factors
    logger.info("Calculating wRC+ with park factors...")
    batter_df['wRC+_new'] = batter_df.apply(
        lambda row: calculate_wrc_plus(row['wOBA'], row.get('Team', ''), row.get('PA', 630)),
        axis=1
    )
    
    # Calculate comprehensive WAR components
    logger.info("Calculating comprehensive WAR components...")
    war_components_list = []
    
    for idx, row in batter_df.iterrows():
        try:
            war, components = calculate_war_components(row, baserunning_df, fielding_df)
            components['IDfg'] = row['IDfg']
            components['Year'] = row['Year']
            war_components_list.append(components)
        except Exception as e:
            logger.error(f"Error calculating WAR for {row['Name']} ({row['IDfg']}): {e}")
            continue
    
    # Convert to DataFrame and merge back
    war_df = pd.DataFrame(war_components_list)
    batter_df = batter_df.merge(war_df, on=['IDfg', 'Year'], how='left', suffixes=('_old', ''))
    
    # Clean up columns - remove old WAR components and keep new ones
    columns_to_remove = [col for col in batter_df.columns if col.endswith('_old')]
    batter_df = batter_df.drop(columns=columns_to_remove)
    
    # Update wRC+ 
    if 'wRC+_new' in batter_df.columns:
        batter_df['wRC+'] = batter_df['wRC+_new']
        batter_df = batter_df.drop(columns=['wRC+_new'])
    
    # Reorder columns for better readability
    column_order = ['Name', 'IDfg', 'Year', 'Age', 'Team', 'Position', 'BB%', 'K%', 'AVG', 'OBP', 'SLG', 
                   'wOBA', 'wRC+',
                   'Off', 'BsR', 'Fld', 'Pos', 'Def', 'WAR', 'PA', 'G', 
                   'HR', '2B', 'RBI', 'R', 'SB', 'CS']
    
    # Keep only columns that exist in the DataFrame
    final_columns = [col for col in column_order if col in batter_df.columns]
    batter_df = batter_df[final_columns]
    
    # Sort by Year and WAR
    batter_df = batter_df.sort_values(['Year', 'WAR'], ascending=[True, False])
    
    # Save updated predictions to pipeline subdirectory
    output_file = output_dir / "pipeline" / "batter_predictions_with_war.csv"
    batter_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved comprehensive predictions to {output_file}")
    logger.info(f"Processed {len(batter_df)} player seasons")
    
    # Display top performers for latest year
    latest_year = batter_df['Year'].max()
    top_performers = (batter_df[batter_df['Year'] == latest_year]
                     .nlargest(10, 'WAR')[['Name', 'Age', 'Position', 'wOBA', 'wRC+', 'Off', 'BsR', 'Fld', 'Pos', 'Def', 'WAR']])
    
    print(f"\nTop 10 Predicted WAR for {latest_year}:")
    print(top_performers.to_string(index=False))

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Calculate comprehensive WAR from prediction files')
    parser.add_argument('--year', type=int, help='Filter to specific year (optional)')
    parser.add_argument('--data-dir', type=Path, default=Path('../data/generated'),
                       help='Directory containing prediction CSV files')
    parser.add_argument('--output-dir', type=Path, default=Path('../data/generated'),
                       help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        process_predictions(args.data_dir, args.output_dir, args.year)
        logger.info("WAR calculation completed successfully!")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
