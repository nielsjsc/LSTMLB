#!/usr/bin/env python3
"""
WAR Calculation Module
======================

This module provides all WAR (Wins Above Replacement) calculation functions for both
batters and pitchers. It combines projected stats with baserunning and fielding data
to calculate comprehensive WAR values using FanGraphs methodology.

This is the SINGLE SOURCE OF TRUTH for WAR calculations in the value_determination module.
Do not duplicate these functions elsewhere.

Functions:
    - calculate_pitcher_war(): FIP-based WAR for pitchers
    - calculate_war_components(): Full WAR breakdown for batters
    - calculate_woba(): Calculate wOBA from counting stats (2025 weights)
    - calculate_woba_from_predictions(): Calculate wOBA from batter prediction DataFrame
    - calculate_wrc_plus(): wRC+ calculation with park factors
    - load_player_orgs(): Load team assignments from roster data

Usage:
    from value_determination.calculate_war import (
        calculate_pitcher_war, calculate_war_components, 
        calculate_woba_from_predictions, load_player_orgs
    )
    
    # Pitcher WAR
    war, components = calculate_pitcher_war(fip=3.50, ip=180, team='NYY', role='SP')
    
    # Batter WAR with calculated wOBA from predictions
    batter_data = calculate_woba_from_predictions(batter_predictions_df)
    war, components = calculate_war_components(player_row, baserunning_df, fielding_df)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Import from central config - SINGLE SOURCE OF TRUTH for constants
from .config import (
    Config, logger,
    # Backward compatibility exports
    BALLPARK_FACTORS, WOBA_SCALE, RPA, LG_WOBA, RPW, LG_FIP
)

# Additional constants from config
LG_PA = Config.WAR.LG_PA
LG_RUNS_PER_PA = Config.WAR.LG_RUNS_PER_PA
LG_WRC_PER_PA = Config.WAR.LG_WRC_PER_PA
POSITIONAL_ADJUSTMENTS = Config.WAR.POSITIONAL_ADJUSTMENTS
REPLACEMENT_LEVEL_RUNS_200IP = Config.WAR.REPLACEMENT_LEVEL_RUNS_200IP
TEAM_ABBREVIATIONS = Config.WAR.TEAM_ABBREVIATIONS

def load_player_orgs(data_dir: Path = None) -> pd.DataFrame:
    """
    Load player organizations from current rosters file.
    
    Returns DataFrame with IDfg (fg_id), mlbam_id, and their current team.
    This function provides the link between player IDs and team assignments
    needed for park factor adjustments.
    
    Args:
        data_dir: Path to data directory. If None, uses Config.Paths.DATA_DIR
        
    Returns:
        DataFrame with columns: IDfg, mlbam_id, Team, player_name
        
    Note:
        TODO: Transition to using mlbam_id as primary identifier
    """
    if data_dir is None:
        data_dir = Config.Paths.DATA_DIR
    
    # Use roster file from config
    roster_file = Config.Paths.ROSTER_FILE
    
    # Fallback paths if config path doesn't exist
    if not roster_file.exists():
        roster_file = data_dir / "active_roster" / "current_rosters.csv"
    if not roster_file.exists():
        roster_file = data_dir / "current_rosters.csv"
    
    if not roster_file.exists():
        logger.warning(f"Current rosters file not found at: {roster_file}")
        logger.warning("WAR calculations will use park factor of 1.0 for all players")
        return pd.DataFrame(columns=['IDfg', 'mlbam_id', 'Team', 'player_name'])
    
    # Load roster data
    roster_df = pd.read_csv(roster_file)
    
    # Filter out players with no fg_id (fg_id == -1.0 means no mapping found)
    roster_df = roster_df[roster_df['fg_id'].notna() & (roster_df['fg_id'] != -1.0)]
    
    # Use team abbreviation mapping from config
    roster_df['Team'] = roster_df['team_name'].map(TEAM_ABBREVIATIONS)
    
    # Select columns - include mlbam_id for future ID migration
    # TODO: Make mlbam_id the primary identifier
    org_data = roster_df[['fg_id', 'mlbam_id', 'Team', 'player_name']].copy()
    org_data = org_data.rename(columns={'fg_id': 'IDfg'})
    
    # Convert IDfg to int to match prediction data format
    org_data['IDfg'] = org_data['IDfg'].astype(int)
    
    logger.info(f"Loaded {len(org_data)} player organizations from current rosters")
    
    return org_data

def calculate_woba(ab: float, bb: float, ibb: float, hbp: float, sf: float,
                  singles: float, doubles: float, triples: float, hr: float,
                  pa: float = None,
                  wbb: float = 0.691, whbp: float = 0.722, w1b: float = 0.882,
                  w2b: float = 1.252, w3b: float = 1.584, whr: float = 2.037) -> float:
    """
    Calculate wOBA from counting stats using modified formula.
    
    Formula: wOBA = (wBB*BB + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR) / PA
    
    Note: This uses ALL walks (including IBB), not just unintentional walks.
    
    Args:
        ab: At bats
        bb: Total walks (includes IBB)
        ibb: Intentional walks (not used in this formula)
        hbp: Hit by pitch
        sf: Sacrifice flies
        singles: Singles (1B)
        doubles: Doubles (2B)
        triples: Triples (3B)
        hr: Home runs
        pa: Plate appearances (if None, calculated from AB+BB+HBP+SF)
        wbb, whbp, w1b, w2b, w3b, whr: 2025 wOBA weights
    
    Returns:
        wOBA value
    """
    # Numerator: weighted sum of positive offensive events (using ALL BB)
    numerator = (wbb * bb + whbp * hbp + w1b * singles + 
                 w2b * doubles + w3b * triples + whr * hr)
    
    # Denominator: plate appearances
    if pa is None:
        pa = ab + bb + hbp + sf
    
    # Avoid division by zero
    if pa == 0:
        return 0.0
    
    return numerator / pa


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

def calculate_woba_from_predictions(batter_df: pd.DataFrame, use_calculated_woba: bool = None) -> pd.DataFrame:
    """
    Calculate wOBA from batter prediction counting stats.
    
    Expects rate stats (HR_rate, 2B_rate, 3B_rate, HBP_rate, SF_rate) to be per 150 games.
    Calculates actual counting stats, then applies the wOBA formula with 2025 weights.
    
    Args:
        batter_df: DataFrame with batter predictions (PA, rate stats, AVG, BB%, K%)
        use_calculated_woba: If True, calculate wOBA from components. If False, use LSTM's wOBA.
                            If None, reads from BatterConfig.CALCULATE_WOBA_FROM_COMPONENTS
        
    Returns:
        DataFrame with 'wOBA' column (calculated or original based on use_calculated_woba)
    """
    from .config import Config
    
    # Import batter config to check the toggle
    if use_calculated_woba is None:
        try:
            # Try multiple import strategies to handle different execution contexts
            try:
                from ..configs.batter_config import BatterConfig
            except (ImportError, ValueError):
                # Relative import failed, try absolute from auto_train_models root
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from configs.batter_config import BatterConfig
            
            use_calculated_woba = BatterConfig.CALCULATE_WOBA_FROM_COMPONENTS
            logger.info(f"Loaded BatterConfig.CALCULATE_WOBA_FROM_COMPONENTS = {use_calculated_woba}")
        except (ImportError, AttributeError) as e:
            # Default to True if config not available
            use_calculated_woba = True
            logger.warning(f"Could not load BatterConfig.CALCULATE_WOBA_FROM_COMPONENTS (error: {e}), defaulting to True")
    else:
        logger.info(f"Using provided use_calculated_woba parameter = {use_calculated_woba}")
    
    df = batter_df.copy()
    
    # If not calculating wOBA from components, return original DataFrame
    if not use_calculated_woba:
        logger.info("Using LSTM's direct wOBA predictions (CALCULATE_WOBA_FROM_COMPONENTS=False)")
        return df
    
    # Force PA to 650 per 150 games for consistent wOBA calculation
    # The model predicts varying PA, but for wOBA we want a standard baseline
    df['PA'] = 650.0
    
    # PA is now standardized at 650 per 150 games
    games_estimate = 150.0  # Since PA is per-150, games = 150
    
    # Calculate walks and strikeouts from percentages
    # BB% and K% are already decimals (0.083 = 8.3%), not percentages, so don't divide by 100
    df['BB'] = df['BB%'] * df['PA']
    df['K'] = df['K%'] * df['PA']
    
    # Extra base hits, HBP, SF come directly from predictions (already per 150 games from preprocessing)
    # Scale to actual games: stat is per 150, so (stat / 150) * games = stat * (games / 150)
    df['HR_count'] = df['HR'] * (games_estimate / 150)
    df['2B_count'] = df['2B'] * (games_estimate / 150)
    df['3B_count'] = df['3B'] * (games_estimate / 150)
    
    # Calculate HBP and SF from predictions if available, otherwise estimate
    if 'HBP' in df.columns:
        df['HBP_count'] = df['HBP'] * (games_estimate / 150)
    else:
        df['HBP_count'] = df['PA'] * 0.01  # ~1% of PA
    
    if 'SF' in df.columns:
        df['SF_count'] = df['SF'] * (games_estimate / 150)
    else:
        df['SF_count'] = df['PA'] * 0.007  # ~0.7% of PA
    
    # Calculate AB and hits
    df['AB'] = df['PA'] - df['BB'] - df['HBP_count'] - df['SF_count']
    df['H'] = df['AVG'] * df['AB']
    
    # Calculate singles: 1B = H - 2B - 3B - HR
    df['1B'] = df['H'] - df['2B_count'] - df['3B_count'] - df['HR_count']
    
    # Estimate IBB as ~10% of BB (league average)
    df['IBB'] = df['BB'] * 0.10
    
    # DEBUG: Log first player's values
    if len(df) > 0:
        first = df.iloc[0]
        logger.info(f"DEBUG - First player: {first.get('Name', 'Unknown')}")
        logger.info(f"  PA={first['PA']:.1f}, BB={first['BB']:.1f}, HBP={first['HBP_count']:.1f}, SF={first['SF_count']:.1f}")
        logger.info(f"  AB={first['AB']:.1f}, H={first['H']:.1f}, AVG={first['AVG']:.3f}")
        logger.info(f"  HR={first['HR_count']:.1f}, 2B={first['2B_count']:.1f}, 3B={first['3B_count']:.1f}, 1B={first['1B']:.1f}")
    
    # Calculate wOBA using the formula with 2025 weights
    weights = Config.WAR.WOBA_WEIGHTS
    
    df['wOBA_calculated'] = df.apply(
        lambda row: calculate_woba(
            ab=row['AB'], bb=row['BB'], ibb=row['IBB'], hbp=row['HBP_count'], sf=row['SF_count'],
            singles=row['1B'], doubles=row['2B_count'], triples=row['3B_count'], hr=row['HR_count'],
            pa=row['PA'],
            wbb=weights['wBB'], whbp=weights['wHBP'], w1b=weights['w1B'],
            w2b=weights['w2B'], w3b=weights['w3B'], whr=weights['wHR']
        ),
        axis=1
    )
    
    # Log comparison if wOBA already exists
    if 'wOBA' in df.columns:
        logger.info(f"Average wOBA - LSTM: {df['wOBA'].mean():.3f}, Calculated: {df['wOBA_calculated'].mean():.3f}")
        # DEBUG: Show first player's calculation
        if len(df) > 0:
            first = df.iloc[0]
            logger.info(f"  First player wOBA: LSTM={first.get('wOBA', 0):.4f}, Calculated={first['wOBA_calculated']:.4f}")
    
    # Replace or add wOBA column with calculated value
    logger.info("Using calculated wOBA from component stats (CALCULATE_WOBA_FROM_COMPONENTS=True)")
    df['wOBA'] = df['wOBA_calculated']
    df = df.drop(columns=['wOBA_calculated'])
    
    return df

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

def calculate_pitcher_war(fip: float,
                         ip: float,
                         team: str,
                         role: str = 'SP',
                         rate_stats: Optional[Dict] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate pitcher WAR from FIP and allocated innings.
    
    FIP-based WAR formula:
    - Runs prevented = (LG_FIP - FIP) / 9 * IP
    - Replacement level = IP / 9 * (replacement_runs_per_9)
    - WAR = (Runs prevented + Replacement) / RPW
    
    Args:
        fip: Projected FIP
        ip: Allocated innings pitched
        team: Team abbreviation for park factor
        role: 'SP' or 'RP'
        rate_stats: Dict with rate stats (K%, BB%, ERA, etc.)
        
    Returns:
        Tuple of (war, components_dict) with full breakdown
    """
    # Park factor adjustment for pitchers (inverse of batters)
    park_factor = BALLPARK_FACTORS.get(str(team).upper().strip(), 100) / 100
    
    # Adjust FIP for park (pitcher in a hitter's park has inflated FIP)
    park_adjusted_fip = fip / park_factor if park_factor != 0 else fip
    
    # FIP runs saved (positive = better than league)
    fip_runs = (LG_FIP - park_adjusted_fip) / 9.0 * ip
    
    # Replacement level runs
    replacement_runs = REPLACEMENT_LEVEL_RUNS_200IP * (ip / 200.0)
    
    # Total runs above replacement
    rar = fip_runs + replacement_runs
    
    # WAR
    war = rar / RPW
    
    # Build components dict
    components = {
        'FIP_Runs': fip_runs,
        'Replacement_Runs': replacement_runs,
        'WAR': war,
        'IP': ip,
        'Team': team,
        'Role': role
    }
    
    # Add rate stats if provided
    if rate_stats:
        for key, value in rate_stats.items():
            components[key] = value
    
    return war, components

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
    
    # After primary position fix, there should only be one row per player-year
    # Just use the first row's position (Inn column is empty anyway)
    if len(player_fielding) > 0:
        primary_pos = player_fielding.iloc[0][pos_col]
        return position_map.get(primary_pos, 'OF')
    
    return 'OF'

def calculate_defensive_value(fielding_data: pd.DataFrame, player_id: int, year: int) -> tuple[float, float]:
    """
    Calculate defensive value and positional adjustment from fielding predictions using Statcast FRV metrics.
    Accounts for multi-position players by using their primary position (most recent prediction).
    The /150 metrics are ALREADY per 150 games from the predictions, so we use them directly.
    
    Returns:
        tuple: (defensive_value, positional_adjustment)
    """
    # Get all fielding rows for this player-year
    player_fielding = fielding_data[(fielding_data['IDfg'] == player_id) & 
                                    (fielding_data['Year'] == year)]
    
    if player_fielding.empty:
        return 0.0, 0.0
    
    # Use the first row (should only be one row per player-year now after primary position fix)
    row = player_fielding.iloc[0]
    
    position = row.get('Pos') or row.get('Position') or row.get('Position_Group', 'OF')
    
    # All positions use 150 games as baseline
    target_games = 150
    
    # The predictions are already per 150 games, scale to target games
    scaling_factor = target_games / 150.0
    
    # Map position for positional adjustment lookup
    pos_for_adjustment = position
    if position in ['LF', 'CF', 'RF']:
        pos_for_adjustment = position
    elif position.lower() == 'outfield':
        pos_for_adjustment = 'OF'
    elif position.lower() == 'infield':
        pos_for_adjustment = '2B'
    elif position.lower() == 'catcher':
        pos_for_adjustment = 'C'
    
    # Calculate position-specific defensive value
    # The /150 metrics are already rate stats, just scale by target games
    if position == 'C' or position.lower() == 'catcher':
        # Catchers: framing + throwing + blocking
        framing = row.get('sc_framing_runs/150', 0) * scaling_factor
        throwing = row.get('sc_throwing_runs/150', 0) * scaling_factor
        blocking = row.get('sc_blocking_runs/150', 0) * scaling_factor
        def_value = framing + throwing + blocking
        
    elif position in ['1B', '2B', '3B', 'SS'] or position.lower() in ['infield', 'first base', 'second base', 'third base', 'shortstop']:
        # Infielders: use sc_total_runs (which already combines range + arm + dp)
        def_value = row.get('sc_total_runs/150', 0) * scaling_factor
        
    else:
        # Outfielders: use sc_total_runs (which already combines range + arm)
        def_value = row.get('sc_total_runs/150', 0) * scaling_factor
    
    # Calculate positional adjustment for target games
    # Adjustments are per 162 games, scale to target games
    pos_adj_per_162 = POSITIONAL_ADJUSTMENTS.get(pos_for_adjustment, 0.0)
    pos_adjustment = pos_adj_per_162 * (target_games / 162.0)
    
    return def_value, pos_adjustment

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
    
    # Use predicted PA (per 150 games) scaled to 150 games
    # PA is predicted per 150 games, so scale to actual proportion
    pa_per_150 = row.get('PA', 650)  # Default 650 PA per 150 games if not present
    games = 150
    pa = pa_per_150 * (games / 150)  # This equals pa_per_150, but keeping for clarity
    
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
    
    # Cap negative defensive value at DH level (worst case = just be a DH)
    # DH penalty is -17.5 per 162 games, scaled to player's games
    dh_penalty = POSITIONAL_ADJUSTMENTS.get('DH', -17.5) * (games / 162.0)
    if def_value < dh_penalty:
        def_value = dh_penalty
    
    # Offensive value
    off = batting_runs + bsr
    
    # Replacement level
    rep_level = 570 * RPW * pa / LG_PA
    
    # RAR (Runs Above Replacement)
    rar = off + def_value + rep_level
    
    # WAR
    war = rar / RPW
    
    # Counting stats (scale from per-150 rates by games)
    counting_stats = {}
    for stat in ['HR', '2B', '3B', 'RBI', 'R']:
        if stat in row:
            # Stats are per 150 games, scale to actual games
            counting_stats[stat] = round(row[stat] * (games / 150.0), 1)
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
    
    # Calculate wOBA (either from components or use LSTM direct prediction based on config)
    batter_df = calculate_woba_from_predictions(batter_df)
    
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
                   'HR', '2B','3B', 'RBI', 'R', 'SB', 'CS']
    
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
