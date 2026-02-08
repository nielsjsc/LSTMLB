"""
Trade Value Calculations
========================

This module handles trade value calculations including:
- Contract option analysis (player/team options, opt-outs)
- Base trade value from surplus value  
- Prospect adjustments with experience-based weighting
- Trade ranking metrics

The key insight for prospect valuation is that a player's trade value should
transition smoothly from prospect-based (for players with little MLB experience)
to performance-based (for established players). The transition is controlled by
games played thresholds defined in Config.Prospects.EXPERIENCE_THRESHOLD_GAMES.

Usage:
    from value_determination.trade_value import (
        calculate_trade_values, add_trade_ranking_metrics
    )
"""

import pandas as pd
import numpy as np

# Import from central config
from .config import Config, logger, CURRENT_YEAR


def calculate_prospect_value_fangraphs(fv: float, rank: float) -> float:
    """
    Calculate prospect value based on FV grade and ranking using FanGraphs methodology.
    
    Uses FV base values and rank adjustments from Config.Prospects.
    
    Args:
        fv: Future Value grade (40-70+ scale). Can include '+' suffix (e.g., '55+')
        rank: Prospect ranking (1-100 for top 100, higher for organizational)
        
    Returns:
        Dollar value of prospect, or None if calculation fails
        
    Example:
        >>> calculate_prospect_value_fangraphs(60, 15)  # 60 FV, #15 prospect
        67_200_000  # $80M base * 0.84 rank adjustment
    """
    if pd.isna(fv):
        return None
        
    try:
        # Handle FV with plus grades (e.g., '55+' -> 57.5)
        if '+' in str(fv):
            fv = float(str(fv).replace('+', '')) + 2.5
        else:
            fv = float(fv)
        
        # Get base value from config
        fv_values = Config.Prospects.FV_BASE_VALUES
        
        # Find closest FV tier (round down to nearest 5)
        valid_tiers = [k for k in fv_values.keys() if k <= fv]
        if not valid_tiers:
            logger.warning(f"FV {fv} below minimum tier, using lowest value")
            base_fv = min(fv_values.keys())
        else:
            base_fv = max(valid_tiers)
        
        base_value = fv_values[base_fv]
        
        # Calculate rank adjustment using config method
        # Only apply rank adjustment for top 100 rank (comparable across orgs)
        # Org ranks are NOT comparable and should not get bonuses
        if pd.notna(rank):
            rank_adj = Config.Prospects.calculate_rank_adjustment(float(rank))
            return base_value * rank_adj
        
        # Default to base value (1.0x) if no rank
        return base_value * 1.0
        
    except Exception as e:
        logger.warning(f"Error calculating prospect value for FV={fv}, rank={rank}: {e}")
        return None


def analyze_contract_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FA year and probable FA year analysis.
    
    Args:
        df: DataFrame with contract status information
        
    Returns:
        DataFrame with FA_Year, probable_fa_year, earliest_fa_year columns
    """
    result = df.copy()
    
    # Find base FA year
    fa_years = (result[result['Status'] == 'Free Agent']
                .groupby('IDfg')['Year']
                .min()
                .reset_index()
                .rename(columns={'Year': 'FA_Year'}))
    
    result = result.merge(fa_years, on='IDfg', how='left')
    
    # Fallback: For players without FA_Year, infer from last contract year
    players_without_fa = result[result['FA_Year'].isna()]['IDfg'].unique()
    if len(players_without_fa) > 0:
        logger.info(f"Inferring FA year for {len(players_without_fa)} players without explicit FA status")
        
        for player_id in players_without_fa:
            player_data = result[result['IDfg'] == player_id]
            
            # Find last year with contract value or Signed status
            contract_years = player_data[
                (player_data['contract_value'].notna() & (player_data['contract_value'] > 0)) | 
                (player_data['Status'].isin(['Signed', 'Unknown']))
            ]['Year']
            
            if len(contract_years) > 0:
                last_contract_year = contract_years.max()
                inferred_fa_year = last_contract_year + 1
                result.loc[result['IDfg'] == player_id, 'FA_Year'] = inferred_fa_year
                logger.debug(f"Player {player_id}: Inferred FA_Year = {inferred_fa_year} (contract ends {last_contract_year})")
    
    result['probable_fa_year'] = result['FA_Year']
    
    # Find players with any type of option
    option_types = ['Player Option', 'Team Option', 'Mutual Option', 'Vesting Option', 'Opt-Out']
    
    # Set earliest_fa_year to option year if exists, otherwise FA_Year
    option_years = (result[result['Status'].isin(option_types)]
                   .groupby('IDfg')['Year']
                   .min()
                   .reset_index()
                   .rename(columns={'Year': 'option_year'}))
    
    result['earliest_fa_year'] = result['FA_Year']
    result = result.merge(option_years, on='IDfg', how='left')
    result.loc[result['option_year'].notna(), 'earliest_fa_year'] = result.loc[result['option_year'].notna(), 'option_year']
    
    # Process each option type
    for player_id in result[result['Status'].isin(option_types)]['IDfg'].unique():
        player_data = result[result['IDfg'] == player_id].sort_values('Year')
        option_status = player_data[player_data['Status'].isin(option_types)]['Status'].iloc[0]
        option_year = player_data[player_data['Status'].isin(option_types)]['Year'].min()
        fa_year = player_data['FA_Year'].iloc[0]
        
        # Calculate surplus sum from option year to FA year
        surplus_sum = player_data[
            (player_data['Year'] >= option_year) &
            (player_data['Year'] < fa_year)
        ]['surplus_value'].sum()
        
        # Apply option-specific logic
        if option_status in ['Player Option', 'Opt-Out']:
            if surplus_sum > 0:  # Player opts out if positive surplus
                result.loc[result['IDfg'] == player_id, 'probable_fa_year'] = option_year
        elif option_status == 'Team Option':
            if surplus_sum < 0:  # Team declines if negative surplus
                result.loc[result['IDfg'] == player_id, 'probable_fa_year'] = option_year
        else:  # Other option types (Mutual, Vesting)
            if surplus_sum < 0:  # Option declined if negative surplus
                result.loc[result['IDfg'] == player_id, 'probable_fa_year'] = option_year
    
    # Clean up temporary column
    result = result.drop('option_year', axis=1, errors='ignore')
    
    # Log examples at debug level
    adjusted_fa_count = (result['FA_Year'] != result['probable_fa_year']).sum()
    if adjusted_fa_count > 0:
        logger.debug(f"Players with adjusted FA years: {adjusted_fa_count}")
    
    return result


def calculate_trade_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trade values and handle prospect adjustments with arb floor.
    
    Args:
        df: DataFrame with surplus values and contract information
        
    Returns:
        DataFrame with trade_value column added
    """
    result_df = df.copy()
    result_df['trade_value'] = None
    
    # Process each player's base trade value
    for player_id in result_df['IDfg'].unique():
        player_data = result_df[result_df['IDfg'] == player_id]
        
        # Get FA year
        fa_year = player_data['probable_fa_year'].iloc[0]
        if pd.isna(fa_year):
            fa_year = player_data['FA_Year'].iloc[0]
        
        # Sum surplus values from current year to FA year (exclusive)
        valid_surplus = player_data[
            (player_data['Year'] >= CURRENT_YEAR) &
            (player_data['Year'] < fa_year) &
            (player_data['surplus_value'].notna())
        ]['surplus_value']
        
        # Check if player is in arbitration
        is_team_control = player_data['Status'].str.contains('Arb|Pre-Arb', regex=True).any()
        
        # Only assign trade value if we have valid surplus values
        if not valid_surplus.empty:
            trade_value = valid_surplus.sum()
            # Floor at 0 for arbitration players
            if is_team_control:
                trade_value = max(0, trade_value)
            
            result_df.loc[
                (result_df['IDfg'] == player_id) &
                (result_df['Year'] >= CURRENT_YEAR),
                'trade_value'
            ] = trade_value
    
    logger.info(f"Players with initial trade values: {result_df['trade_value'].notna().sum()}")
    
    # Try to load prospect data for adjustments
    prospect_file = Config.Paths.PROSPECT_FILE
    if prospect_file.exists():
        result_df = _apply_prospect_adjustments(result_df, prospect_file)
    else:
        logger.warning(f"Prospect file not found: {prospect_file}")
    
    # Log statistics
    total_with_values = result_df['trade_value'].notna().sum()
    avg_value = result_df['trade_value'].mean()
    median_value = result_df['trade_value'].median()
    logger.info(f"Trade values calculated: {total_with_values} players, avg=${avg_value:,.0f}, median=${median_value:,.0f}")
    
    return result_df


def _apply_prospect_adjustments(result_df: pd.DataFrame, prospect_file) -> pd.DataFrame:
    """
    Apply prospect value adjustments to trade values.
    
    Uses MLB.com prospect rankings and grades to adjust trade values for young players.
    Players with high prospect grades get a bonus, especially if they haven't proven themselves yet.
    """
    
    prospect_df = pd.read_csv(prospect_file)
    logger.info(f"Loaded prospect data: {prospect_df.shape[0]} records, years {prospect_df['year'].min():.0f}-{prospect_df['year'].max():.0f}")
    
    # Normalize names for matching
    result_df['name_normalized'] = result_df['Name'].str.lower().str.strip()
    prospect_df['name_normalized'] = prospect_df['name'].str.lower().str.strip()
    
    # Get latest prospect ranking for each player (most recent year)
    latest_prospect_data = (
        prospect_df
        .sort_values('year', ascending=False)
        .groupby('name_normalized')
        .first()
        .reset_index()
    )
    
    logger.info(f"Unique prospects in rankings: {len(latest_prospect_data)}")
    
    # Get latest pre-current-year MLB experience for each player
    latest_mlb_experience = (
        result_df[
            (result_df['Year'] < CURRENT_YEAR) &
            ((result_df['G_bat'].notna()) | (result_df['G_pit'].notna()) | (result_df['GS'].notna()))
        ]
        .groupby('name_normalized')
        .agg({
            'G_bat': 'sum',
            'G_pit': 'sum',
            'GS': 'sum',
            'position_group': 'first'
        })
        .reset_index()
    )
    logger.info(f"Players with pre-{CURRENT_YEAR} MLB experience: {len(latest_mlb_experience)}")
    
    # Match prospects with trade values
    prospects_with_values = result_df[
        (result_df['Year'] >= CURRENT_YEAR) &
        (result_df['name_normalized'].isin(latest_prospect_data['name_normalized'])) &
        (result_df['trade_value'].notna())
    ].copy()
    
    # Merge with prospect data
    prospects_with_values = prospects_with_values.merge(
        latest_prospect_data[['name_normalized', 'year', 'rank', 'grade_overall', 'top_100', 'organization']],
        on='name_normalized',
        how='left',
        suffixes=('', '_prospect')
    )
    
    matched_count = len(prospects_with_values.drop_duplicates('name_normalized'))
    logger.info(f"Matched {matched_count} prospects with trade values")
    
    if len(prospects_with_values) == 0:
        result_df = result_df.drop(columns=['name_normalized'], errors='ignore')
        return result_df
    
    # Process each unique prospect
    adjusted_count = 0
    for name in prospects_with_values['name_normalized'].unique():
        prospect_data = prospects_with_values[prospects_with_values['name_normalized'] == name].iloc[0]
        
        # Determine position type for experience threshold
        position_group = prospect_data.get('position_group', 'batter')
        if position_group == 'SP':
            position_type = 'sp'
        elif position_group == 'RP':
            position_type = 'rp'
        else:
            position_type = 'batter'
        
        # Get MLB experience if exists
        if name in latest_mlb_experience['name_normalized'].values:
            career_stats = latest_mlb_experience[latest_mlb_experience['name_normalized'] == name].iloc[0]
            
            # Calculate MLB games based on position type
            if position_type == 'sp':
                games_played = career_stats.get('GS', 0) or 0
            elif position_type == 'rp':
                gs = career_stats.get('GS', 0) or 0
                g_pit = career_stats.get('G_pit', 0) or 0
                games_played = g_pit - gs  # RP appearances
            else:
                games_played = career_stats.get('G_bat', 0) or 0
        else:
            games_played = 0
        
        # Use centralized prospect weight calculation from config
        # This properly diminishes prospect weight as players gain experience
        prospect_weight = Config.Prospects.calculate_prospect_weight(games_played, position_type)
        mlb_weight = 1.0 - prospect_weight
        
        # Skip prospect adjustment entirely for established players
        if prospect_weight == 0.0:
            logger.debug(f"Skipping {prospect_data['Name']}: established player ({games_played} games)")
            continue
        
        # Calculate prospect value using FanGraphs methodology
        fv = prospect_data.get('grade_overall', None)
        org_rank = prospect_data.get('rank', None)
        top_100_rank = prospect_data.get('top_100', None)
        year = prospect_data.get('year', None)
        
        # For 2026, 'rank' is actually the top_100 value (no org lists yet)
        if year == 2026 and pd.notna(org_rank):
            top_100_rank = org_rank
            org_rank = None
        
        # Only use top_100 rank for value calculation (org ranks not comparable)
        # This ensures only true top 100 prospects get the rank bonus
        rank = top_100_rank
        
        prospect_value = calculate_prospect_value_fangraphs(fv, rank)
        
        if prospect_value is None:
            logger.debug(f"Skipping {prospect_data['Name']}: could not calculate prospect value")
            continue
        
        # Calculate weighted value
        # MLB component: value from projected performance * MLB experience weight
        # Prospect component: value from prospect grade * prospect weight
        mlb_component = prospect_data['trade_value'] * mlb_weight
        prospect_component = prospect_value * prospect_weight
        weighted_value = mlb_component + prospect_component
        
        # Update trade values for all future years
        mask = (result_df['name_normalized'] == name) & (result_df['Year'] >= CURRENT_YEAR)
        if mask.sum() > 0:
            result_df.loc[mask, 'trade_value'] = weighted_value
            adjusted_count += 1
            
            # Log details at debug level
            rank_display = f"top100={top_100_rank:.0f}" if pd.notna(top_100_rank) else f"org={org_rank:.0f}" if pd.notna(org_rank) else "no rank"
            logger.debug(
                f"  {prospect_data['Name']}: FV={fv}, {rank_display}, "
                f"games={games_played}, prospect_wt={prospect_weight:.2f}, "
                f"prospect_val=${prospect_value:,.0f}, final=${weighted_value:,.0f}"
            )
    
    logger.info(f"Applied prospect adjustments to {adjusted_count} players")
    
    # Clean up temporary columns before returning
    result_df = result_df.drop(columns=['name_normalized'], errors='ignore')
    
    return result_df


def add_trade_ranking_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-calculated metrics needed for trade value rankings.
    
    Metrics added:
        - contract_war: WAR while under team control
        - contract_base_value: Dollar value of contract WAR
        - avg_war: Average WAR per control year
        - total_contract: Total contract cost
        - avg_contract: Average annual cost
        - total_surplus: Sum of surplus values under control
        - years_control: Number of control years remaining
        - control_through: Last year of team control
        - total_future_war: All WAR from 2025 onward
        - total_future_value: Dollar value of future WAR
        - historical_war: Career WAR before 2025
        - historical_value: Dollar value of historical WAR
    
    Args:
        df: DataFrame with trade values calculated
        
    Returns:
        DataFrame with ranking metrics added
    """
    result = df.copy()
    
    # Group by player and calculate metrics
    player_metrics = []
    
    for player_id in result['IDfg'].unique():
        player_data = result[result['IDfg'] == player_id].sort_values('Year')
        
        # Get control years (2025 through FA year)
        fa_year = player_data['probable_fa_year'].iloc[0]
        if pd.isna(fa_year):
            fa_year = player_data['FA_Year'].iloc[0]
        
        # Contract years data
        control_years = player_data[
            (player_data['Year'] >= CURRENT_YEAR) &
            (player_data['Year'] < fa_year)
        ]
        
        # Future years data (all years from current year onward)
        future_years = player_data[player_data['Year'] >= CURRENT_YEAR]
        
        # Historical years data (before current year)
        historical_years = player_data[player_data['Year'] < CURRENT_YEAR]
        
        # All career years
        all_years = player_data
        
        years_control = len(control_years)
        
        metrics = {
            'IDfg': player_id,
            'contract_war': control_years['WAR'].sum() if years_control > 0 else 0,
            'contract_base_value': control_years['Base_Value'].sum() if years_control > 0 else 0,
            'avg_war': control_years['WAR'].mean() if years_control > 0 else 0,
            'total_contract': control_years['contract_value'].sum() if years_control > 0 else 0,
            'avg_contract': control_years['contract_value'].mean() if years_control > 0 else 0,
            'total_surplus': control_years['surplus_value'].sum() if years_control > 0 else 0,
            'years_control': years_control,
            'control_through': fa_year - 1 if pd.notna(fa_year) else None,
            'total_future_war': future_years['WAR'].sum(),
            'total_future_value': future_years['Base_Value'].sum(),
            'total_war': all_years['WAR'].sum(),
            'total_value': all_years['Base_Value'].sum(),
            'historical_war': historical_years['WAR'].sum(),
            'historical_value': historical_years['Base_Value'].sum()
        }
        player_metrics.append(metrics)
    
    # Convert to DataFrame and merge back
    metrics_df = pd.DataFrame(player_metrics)
    result = result.merge(metrics_df, on='IDfg', how='left')
    
    # Round values for cleaner display
    result['contract_war'] = result['contract_war'].round(1)
    result['avg_war'] = result['avg_war'].round(2)
    result['total_contract'] = result['total_contract'].round(1)
    result['avg_contract'] = result['avg_contract'].round(2)
    result['total_surplus'] = result['total_surplus'].round(1)
    result['total_future_war'] = result['total_future_war'].round(1)
    result['total_future_value'] = result['total_future_value'].round(1)
    result['total_war'] = result['total_war'].round(1)
    result['total_value'] = result['total_value'].round(1)
    result['historical_war'] = result['historical_war'].round(1)
    result['historical_value'] = result['historical_value'].round(1)
    
    return result


def update_prospect_mlb_status(export_data: pd.DataFrame) -> None:
    """
    Add MLB status to prospect data.
    
    Updates the prospect_histories.csv file (used by backend) with has_mlb flags
    to indicate which prospects have reached the majors.
    
    Args:
        export_data: Final export data with all players
    """
    # Use prospect_histories.csv (the current/correct file, not the legacy player_histories.csv)
    prospect_file = Config.Paths.GENERATED_DIR / 'MiLB' / 'prospect_histories.csv'
    
    if not prospect_file.exists():
        logger.warning(f"Prospect file not found: {prospect_file}")
        logger.info(f"Run generate_prospect_histories.py first to create this file")
        return
    
    try:
        # Load prospect data
        prospect_df = pd.read_csv(prospect_file)
        
        # Get unique IDfg values from export data
        mlb_ids = export_data['IDfg'].unique()
        
        # Convert IDfg to string in both datasets for consistent comparison
        prospect_df['IDfg'] = prospect_df['IDfg'].astype(str)
        mlb_ids = [str(id) for id in mlb_ids]
        
        # Add has_mlb column
        prospect_df['has_mlb'] = prospect_df['IDfg'].isin(mlb_ids)
        
        # Save updated prospect file
        prospect_df.to_csv(prospect_file, index=False)
        
        # Log summary
        total_prospects = len(prospect_df['IDfg'].unique())
        mlb_prospects = len(prospect_df[prospect_df['has_mlb']]['IDfg'].unique())
        pct_mlb = (mlb_prospects / total_prospects) * 100 if total_prospects > 0 else 0
        
        logger.info(f"Prospect MLB status: {mlb_prospects}/{total_prospects} ({pct_mlb:.1f}%) have MLB data")
        
    except Exception as e:
        logger.error(f"Failed to update prospect MLB status: {str(e)}")
        raise
