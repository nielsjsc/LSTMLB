"""
Contract status processing and timeline generation.
"""

import pandas as pd
import numpy as np

from .constants import logger


def normalize_contract_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize MLB player contract statuses into standardized format.
    
    Process:
    1. Sort by player and year
    2. Check for long-term contracts
    3. Process status patterns in priority order
    4. Handle special cases (Estimate, Arb Avoided)
    
    Args:
        df: DataFrame must contain:
            - IDfg: Player ID
            - Year: Contract year
            - Status: Raw contract status
            - Payroll: Salary information
            
    Returns:
        DataFrame with new 'Normalized_Status' column
    """
    result_df = df.copy()
    result_df = result_df.sort_values(['IDfg', 'Year'])
    
    def has_long_term_contract(group):
        """Check if player has signed contract years"""
        # First check: Any future years with payroll but no status
        future_signed = group[
            (group['Payroll'].notna()) &
            (group['Status'].isna())
        ]
        if len(future_signed) > 0:
            return True
        
        # Second check: Signed years beyond arb
        arb_years = group[group['Status'].str.contains('ARB', na=False, case=True)]
        if len(arb_years) > 0:
            last_arb_year = arb_years['Year'].max()
            future_signed = group[
                (group['Year'] > last_arb_year) &
                (group['Payroll'].notna()) &
                (group['Status'].isna())
            ]
            return len(future_signed) > 0
        
        return False
    
    def get_next_year_status(group):
        """Look ahead one year to determine current status for 'Estimate'"""
        group = group.copy()
        group['Next_Status'] = group['Status'].shift(-1)
        group['Has_Long_Contract'] = has_long_term_contract(group)
        return group
    
    def _normalize_single_status(row):
        """Normalize individual status values."""
        if pd.isna(row['Status']):
            if pd.notna(row['Payroll']):
                return 'Signed'
            if row['Year'] == 2025:
                return 'Free Agent'
            return None
        
        status = str(row['Status']).upper().strip()
        if status == '-' and pd.notna(row['Payroll']):
            return 'Signed'
        
        # Handle dollar amounts
        if status.startswith('$'):
            return 'Signed'
        
        # Handle options
        if 'PLAYER' in status:
            return 'Player Option'
        if 'CLUB' in status:
            return 'Team Option'
        if 'MUTUAL' in status:
            return 'Mutual Option'
        if 'VESTING' in status:
            return 'Vesting Option'
        if 'OPT-OUT' in status:
            return 'Opt-Out'
        
        # Handle 'Estimate' based on next year's status
        if status == 'ESTIMATE':
            next_status = str(row['Next_Status']).upper().strip() if pd.notna(row['Next_Status']) else ''
            if any(x in next_status for x in ['UFA', 'FA']):
                return 'Arb-3'
            if 'ARB 1' in next_status:
                return 'Pre-Arb'
            if 'ARB 2' in next_status:
                return 'Arb-1'
            if 'ARB 3' in next_status:
                return 'Arb-2'
            if 'ARB 4' in next_status:
                return 'Arb-3'
            return 'Pre-Arb'
        
        # Handle 'Arb Avoided' and 'Arb Filed'
        if 'AVOIDED' in status or 'BYPASSED' in status or 'FILED' in status or 'SETTLED' in status:
            next_status = str(row['Next_Status']).upper().strip() if pd.notna(row['Next_Status']) else ''
            if 'ARB 2' in next_status:
                return 'Arb-1'
            if 'ARB 3' in next_status:
                return 'Arb-2'
            if 'ARB 4' in next_status:
                return 'Arb-3'
            if 'UFA' in next_status:
                return 'Arb-3'
            if pd.isna(row['Next_Status']):
                return 'Signed'
            return 'Arb-1'
        
        # Handle regular arbitration
        if 'ARB' in status:
            if 'S2' in status:
                return 'Arb-1 (Super 2)'
            if 'ARB 4' in status:
                return 'Arb-4'
            if 'ARB 3' in status:
                return 'Arb-3'
            if 'ARB 2' in status:
                return 'Arb-2'
            if 'ARB 1' in status:
                return 'Arb-1'
        
        # Handle pre-arbitration
        if 'PRE' in status and 'ARB' in status:
            return 'Pre-Arb'
        
        # Handle free agency
        if any(x in status for x in ['UFA', 'RFA', 'FA', 'FREE AGENT']):
            return 'Free Agent'
        
        # Handle contract statuses that indicate signed players
        # These are players with active contracts (Active, Injured, Reserve, etc.)
        contract_indicators = [
            'ACTIVE', 'INJURED', 'RESERVE', 'RETAINED', 'DEFERRED', 
            'BURIED', 'BUYOUT', 'BONUS', 'VOIDED', 'SIGNING BONUS'
        ]
        if any(indicator in status for indicator in contract_indicators):
            # If they have a payroll, they're signed
            if pd.notna(row['Payroll']):
                return 'Signed'
        
        # If we still don't know but they have a payroll, assume signed
        if pd.notna(row['Payroll']):
            return 'Signed'
        
        return 'Unknown'
    
    # Process by player group
    result_df = result_df.groupby('IDfg', group_keys=False).apply(get_next_year_status)
    
    # Apply normalization
    result_df['Normalized_Status'] = result_df.apply(_normalize_single_status, axis=1)
    
    # Log distribution
    status_counts = result_df['Normalized_Status'].value_counts()
    logger.info("\nStatus distribution after normalization:")
    for status, count in status_counts.items():
        logger.info(f"{status}: {count}")
    
    return result_df.drop(['Next_Status', 'Has_Long_Contract'], axis=1)


def check_none_statuses(contract_data: pd.DataFrame) -> list:
    """Find players with None status in their latest year."""
    problems = []
    
    for player_id in contract_data['IDfg'].unique():
        player_data = contract_data[contract_data['IDfg'] == player_id].sort_values('Year')
        latest_status = player_data.iloc[-1]['Normalized_Status']
        
        if pd.isna(latest_status):
            problems.append({
                'IDfg': player_id,
                'Name': player_data.iloc[0]['Name'],
                'Latest Year': player_data['Year'].max(),
                'All Statuses': player_data[['Year', 'Status', 'Normalized_Status']].to_dict('records')
            })
    
    if problems:
        print(f"\nFound {len(problems)} players with None as latest status:")
        for p in problems:
            print(f"\nPlayer: {p['Name']} (ID: {p['IDfg']}, Last Year: {p['Latest Year']})")
            print("Status History:")
            for status in p['All Statuses']:
                print(f"Year {status['Year']}: Status = {status['Status']}, Normalized = {status['Normalized_Status']}")
    
    return problems


def generate_contract_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate complete contract timeline for each player.
    """
    result_df = df.copy()
    
    def process_player_timeline(group):
        player_rows = group.copy()
        
        # Check for Super 2
        is_super2 = any('Super 2' in str(status) for status in player_rows['Normalized_Status'])
        
        # Get latest year and status
        latest_year = player_rows['Year'].max()
        latest_status_series = player_rows.loc[player_rows['Year'] == latest_year, 'Normalized_Status']
        
        if len(latest_status_series) == 0:
            logger.warning(f"No status found for player {player_rows['IDfg'].iloc[0]} in year {latest_year}")
            return player_rows
        
        latest_status = latest_status_series.iloc[0]
        new_rows = []
        
        if latest_status == 'Free Agent':
            return player_rows
        
        # Modified status checking
        if latest_status.startswith('Arb-4'):
            new_rows.append({'Year': latest_year + 1, 'Normalized_Status': 'Free Agent'})
        
        elif latest_status.startswith('Arb-3'):
            if is_super2:
                new_rows.append({'Year': latest_year + 1, 'Normalized_Status': 'Arb-4'})
                new_rows.append({'Year': latest_year + 2, 'Normalized_Status': 'Free Agent'})
            else:
                new_rows.append({'Year': latest_year + 1, 'Normalized_Status': 'Free Agent'})
        
        elif latest_status.startswith('Arb-2'):
            new_rows.append({'Year': latest_year + 1, 'Normalized_Status': 'Arb-3'})
            if is_super2:
                new_rows.append({'Year': latest_year + 2, 'Normalized_Status': 'Arb-4'})
                new_rows.append({'Year': latest_year + 3, 'Normalized_Status': 'Free Agent'})
            else:
                new_rows.append({'Year': latest_year + 2, 'Normalized_Status': 'Free Agent'})
        
        elif latest_status.startswith('Arb-1'):
            arb2_status = 'Arb-2'
            new_rows.append({'Year': latest_year + 1, 'Normalized_Status': arb2_status})
            new_rows.append({'Year': latest_year + 2, 'Normalized_Status': 'Arb-3'})
            if is_super2:
                new_rows.append({'Year': latest_year + 3, 'Normalized_Status': 'Arb-4'})
                new_rows.append({'Year': latest_year + 4, 'Normalized_Status': 'Free Agent'})
            else:
                new_rows.append({'Year': latest_year + 3, 'Normalized_Status': 'Free Agent'})
        
        elif latest_status == 'Pre-Arb':
            pre_arb_years = len(player_rows[player_rows['Normalized_Status'] == 'Pre-Arb'])
            remaining_pre_arb = 3 - pre_arb_years
            
            current_year = latest_year
            for i in range(remaining_pre_arb):
                current_year += 1
                new_rows.append({'Year': current_year, 'Normalized_Status': 'Pre-Arb'})
            
            arb1_status = 'Arb-1 (Super 2)' if is_super2 else 'Arb-1'
            new_rows.append({'Year': current_year + 1, 'Normalized_Status': arb1_status})
            new_rows.append({'Year': current_year + 2, 'Normalized_Status': 'Arb-2'})
            new_rows.append({'Year': current_year + 3, 'Normalized_Status': 'Arb-3'})
            
            if is_super2:
                new_rows.append({'Year': current_year + 4, 'Normalized_Status': 'Arb-4'})
                new_rows.append({'Year': current_year + 5, 'Normalized_Status': 'Free Agent'})
            else:
                new_rows.append({'Year': current_year + 4, 'Normalized_Status': 'Free Agent'})
        
        # Handle Signed and Unknown statuses - infer FA year from contract end
        elif latest_status in ['Signed', 'Unknown']:
            # Find the last year with a Payroll value
            payroll_years = player_rows[player_rows['Payroll'].notna()]['Year']
            if len(payroll_years) > 0:
                last_contract_year = payroll_years.max()
                # FA year is the year after the last contract year
                fa_year = last_contract_year + 1
                if fa_year > latest_year:
                    new_rows.append({'Year': fa_year, 'Normalized_Status': 'Free Agent'})
        
        # Add new rows to player timeline
        if new_rows:
            for row in new_rows:
                row.update({col: group.iloc[0][col] for col in group.columns
                           if col not in ['Year', 'Normalized_Status', 'Status', 'Payroll']})
            return pd.concat([player_rows, pd.DataFrame(new_rows)], ignore_index=True)
        
        return player_rows
    
    # Process each player
    result_df = result_df.groupby('IDfg', group_keys=False).apply(process_player_timeline)
    
    return result_df.sort_values(['IDfg', 'Year']).reset_index(drop=True)


def validate_fa_years(contract_timeline: pd.DataFrame) -> list:
    """Validate all players have FA year."""
    missing_fa = []
    for idfg, group in contract_timeline.groupby('IDfg'):
        if not any(group['Normalized_Status'] == 'Free Agent'):
            player_name = group['Name'].iloc[0]
            last_status = group.sort_values('Year')['Normalized_Status'].iloc[-1]
            missing_fa.append({
                'IDfg': idfg,
                'Player': player_name,
                'Last Status': last_status,
                'Last Year': group['Year'].max()
            })
    
    if missing_fa:
        print("\nPlayers missing Free Agent status:")
        missing_fa_df = pd.DataFrame(missing_fa)
        print(missing_fa_df.to_string())
        logger.warning(f"Found {len(missing_fa)} players missing FA status")
    else:
        logger.info("All players have Free Agent status")
    
    return missing_fa


def extend_fa_timeline(timeline_df: pd.DataFrame) -> pd.DataFrame:
    """Extend timeline beyond first FA year through 2040."""
    
    # Find first FA year for each player
    fa_years = (timeline_df[timeline_df['Normalized_Status'] == 'Free Agent']
                .groupby('IDfg')['Year']
                .min()
                .reset_index())
    
    # Generate future FA rows
    future_rows = []
    for _, row in fa_years.iterrows():
        idfg = row['IDfg']
        start_year = int(row['Year']) + 1
        player_info = timeline_df[timeline_df['IDfg'] == idfg].iloc[0]
        
        for year in range(start_year, 2041):
            future_rows.append({
                'Name': player_info['Name'],
                'IDfg': idfg,
                'position_group': player_info['position_group'],
                'Year': year,
                'Team': np.nan,
                'Payroll': np.nan,
                'Status': np.nan,
                'Normalized_Status': 'Free Agent'
            })
    
    # Add new rows to timeline
    extended_timeline = pd.concat([
        timeline_df,
        pd.DataFrame(future_rows)
    ])
    
    # Sort and deduplicate
    extended_timeline = (extended_timeline
                        .sort_values(['IDfg', 'Year'])
                        .drop_duplicates(subset=['IDfg', 'Year'], keep='first'))
    
    return extended_timeline
