"""
Trade value calculations including prospect adjustments and ranking metrics.
"""

import pandas as pd
import numpy as np

from .constants import logger, DATA_DIR


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
    
    # Log examples
    option_examples = result[
        result['FA_Year'] != result['probable_fa_year']
    ][['Name', 'Year', 'Status', 'surplus_value', 'FA_Year', 'probable_fa_year']].head()
    
    print("\nExample players with adjusted FA years:")
    print(option_examples)
    
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
        
        # Sum surplus values from 2025 to FA year (exclusive)
        valid_surplus = player_data[
            (player_data['Year'] >= 2025) &
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
                (result_df['Year'] >= 2025),
                'trade_value'
            ] = trade_value
    
    print(f"\nPlayers with initial trade values: {result_df['trade_value'].notna().sum()}")
    
    # Try to load prospect data for adjustments
    prospect_file = DATA_DIR / 'generated/MiLB/player_histories.csv'
    if prospect_file.exists():
        result_df = _apply_prospect_adjustments(result_df, prospect_file)
    else:
        logger.warning(f"Prospect file not found: {prospect_file}")
    
    print(f"\nFinal players with trade values: {result_df['trade_value'].notna().sum()}")
    
    # Log statistics
    print("\nTrade Value Statistics:")
    print(f"Total players with trade values: {result_df['trade_value'].notna().sum()}")
    print(f"Average trade value: ${result_df['trade_value'].mean():,.2f}")
    print(f"Median trade value: ${result_df['trade_value'].median():,.2f}")
    
    return result_df


def _apply_prospect_adjustments(result_df: pd.DataFrame, prospect_file) -> pd.DataFrame:
    """Apply prospect value adjustments to trade values."""
    
    prospect_df = pd.read_csv(prospect_file)
    print(f"\nProspect data shape: {prospect_df.shape}")
    print(f"Sample prospect IDs: {prospect_df['IDfg'].head().tolist()}")
    
    result_df['IDfg'] = result_df['IDfg'].astype(str).str.strip()
    prospect_df['IDfg'] = prospect_df['IDfg'].astype(str).str.strip()
    
    # Get latest pre-2025 WAR for each player
    latest_mlb_experience = (
        result_df[
            (result_df['Year'] < 2025) &
            ((result_df['G_bat'].notna()) | (result_df['G_pit'].notna()) | (result_df['GS'].notna()))
        ]
        .groupby('IDfg')
        .agg({
            'G_bat': 'sum',
            'G_pit': 'sum',
            'GS': 'sum',
            'position_group': 'first'
        })
        .reset_index()
    )
    print(f"\nPlayers with MLB experience: {len(latest_mlb_experience)}")
    
    # Debug matching conditions
    print("\nChecking matching conditions:")
    condition1 = result_df['Year'] >= 2025
    condition2 = result_df['IDfg'].isin(prospect_df['IDfg'])
    condition3 = result_df['trade_value'].notna()
    condition4 = result_df['IDfg'].isin(latest_mlb_experience['IDfg'])
    
    print(f"Players in 2025+: {condition1.sum()}")
    print(f"Players matching prospect IDs: {condition2.sum()}")
    print(f"Players with trade values: {condition3.sum()}")
    print(f"Players with MLB experience: {condition4.sum()}")
    
    # Process recent prospects that have both trade values and prospect values
    recent_prospects = result_df[
        condition1 & condition2 & condition3 & condition4
    ].drop_duplicates('IDfg')
    
    print(f"\nMatched prospects to process: {len(recent_prospects)}")
    if len(recent_prospects) > 0:
        print("\nSample matched prospect:")
        sample_prospect = recent_prospects.iloc[0]
        print(f"ID: {sample_prospect['IDfg']}")
        print(f"Name: {sample_prospect.get('Name', 'N/A')}")
        print(f"Trade Value: {sample_prospect['trade_value']}")
    
    for _, prospect in recent_prospects.iterrows():
        # Get career MLB experience
        career_stats = latest_mlb_experience[
            latest_mlb_experience['IDfg'] == prospect['IDfg']
        ].iloc[0]
        
        # Calculate MLB games based on position and role
        if prospect['position_group'] in ['SP', 'RP']:
            gs = career_stats.get('GS', 0) or 0
            g_pit = career_stats.get('G_pit', 0) or 0
            
            # If they have significant starts, only use GS
            if gs > 0 and g_pit > 0 and gs / g_pit > 0.5:
                games_played = gs
                max_games = 45
            else:
                games_played = g_pit
                max_games = 65
        else:
            # For position players
            games_played = career_stats.get('G_bat', 0) or 0
            max_games = 300
        
        print(f"\nProcessing prospect {prospect.get('Name', prospect['IDfg'])}:")
        print(f"Position group: {prospect['position_group']}")
        if prospect['position_group'] in ['SP', 'RP']:
            print(f"Career Games Started: {gs}")
            print(f"Career Games Pitched: {g_pit}")
        print(f"Career Games counted: {games_played}")
        print(f"Max games threshold: {max_games}")
        
        # Get prospect value with debug info
        prospect_matches = prospect_df[prospect_df['IDfg'] == prospect['IDfg']]
        print(f"Found {len(prospect_matches)} matching prospect records")
        
        # Attempt to find a valid year column in descending order
        prospect_value = 0
        for year_col in ["2025_Value", "2024_Value", "2023_Value", "2022_Value"]:
            if year_col in prospect_matches.columns:
                year_vals = prospect_matches[year_col].dropna()
                if not year_vals.empty:
                    prospect_value = year_vals.iloc[0]
                    print(f"Using {year_col}: {prospect_value}")
                    break
        
        print(f"MLB trade value: {prospect['trade_value']}")
        
        # Calculate weights based on games played
        mlb_weight = min(1.0, games_played / max_games)
        prospect_weight = 1 - mlb_weight
        
        print(f"MLB weight: {mlb_weight:.2f}")
        print(f"Prospect weight: {prospect_weight:.2f}")
        
        # Add null checks before calculation
        if pd.isna(prospect['trade_value']):
            print("Warning: MLB trade value is nan")
            mlb_component = 0
        else:
            mlb_component = prospect['trade_value'] * mlb_weight
        
        if pd.isna(prospect_value):
            print("Warning: Prospect value is nan")
            prospect_component = 0
        else:
            prospect_component = prospect_value * prospect_weight
        
        # Calculate weighted value with components
        weighted_value = mlb_component + prospect_component
        
        print(f"MLB component: {mlb_component:,.2f}")
        print(f"Prospect component: {prospect_component:,.2f}")
        print(f"Final weighted value: {weighted_value:,.2f}")
        
        # Update trade values for 2025+ years
        result_df.loc[
            (result_df['IDfg'] == prospect['IDfg']) &
            (result_df['Year'] >= 2025),
            'trade_value'
        ] = weighted_value
    
    return result_df


def add_trade_ranking_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-calculated metrics needed for trade value rankings:
    - Contract WAR (WAR while under contract)
    - Average WAR per season under contract
    - Total contract value
    - Average contract value per season
    - Total future WAR (all future seasons)
    - Total future value (all future seasons)
    - Total WAR (all career seasons)
    - Total value (all career seasons)
    - Historical WAR (all seasons before 2025)
    - Historical value (all seasons before 2025)
    
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
            (player_data['Year'] >= 2025) &
            (player_data['Year'] < fa_year)
        ]
        
        # Future years data (all years 2025+)
        future_years = player_data[player_data['Year'] >= 2025]
        
        # Historical years data (before 2025)
        historical_years = player_data[player_data['Year'] < 2025]
        
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
    
    Args:
        export_data: Final export data with all players
    """
    prospect_file = DATA_DIR / 'generated/MiLB/player_histories.csv'
    
    if not prospect_file.exists():
        logger.warning(f"Prospect file not found: {prospect_file}")
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
        
        # Print summary
        total_prospects = len(prospect_df['IDfg'].unique())
        mlb_prospects = len(prospect_df[prospect_df['has_mlb']]['IDfg'].unique())
        
        print(f"\nProspect MLB Status Summary:")
        print(f"Total unique prospects: {total_prospects}")
        print(f"Prospects with MLB data: {mlb_prospects}")
        print(f"Percentage with MLB: {(mlb_prospects/total_prospects)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Failed to update prospect MLB status: {str(e)}")
        raise
