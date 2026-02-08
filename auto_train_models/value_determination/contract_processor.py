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
    Fills in predicted contract progression and calculates FA years.
    """
    result_df = df.copy()
    
    def process_player_timeline(group):
        player_rows = group.copy().sort_values('Year')
        
        # Check for Super 2
        is_super2 = any('Super 2' in str(status) for status in player_rows['Normalized_Status'])
        
        # Find the last row with a valid (non-None) status
        valid_status_mask = player_rows['Normalized_Status'].notna()
        if not valid_status_mask.any():
            # All statuses are None - mark all as Free Agent
            player_rows['Normalized_Status'] = 'Free Agent'
            return player_rows
        
        last_valid_idx = player_rows[valid_status_mask].index[-1]
        latest_status = player_rows.loc[last_valid_idx, 'Normalized_Status']
        latest_year = player_rows.loc[last_valid_idx, 'Year']
        
        # If already at FA, nothing to add
        if latest_status == 'Free Agent':
            # Fill any remaining None years with FA
            player_rows.loc[player_rows['Normalized_Status'].isna(), 'Normalized_Status'] = 'Free Agent'
            return player_rows
        
        # Calculate what statuses should follow
        new_statuses = []
        
        if latest_status.startswith('Arb-4'):
            new_statuses = [('Free Agent', 1)]
        
        elif latest_status.startswith('Arb-3'):
            if is_super2:
                new_statuses = [('Arb-4', 1), ('Free Agent', 2)]
            else:
                new_statuses = [('Free Agent', 1)]
        
        elif latest_status.startswith('Arb-2'):
            if is_super2:
                new_statuses = [('Arb-3', 1), ('Arb-4', 2), ('Free Agent', 3)]
            else:
                new_statuses = [('Arb-3', 1), ('Free Agent', 2)]
        
        elif latest_status.startswith('Arb-1'):
            if is_super2:
                new_statuses = [('Arb-2', 1), ('Arb-3', 2), ('Arb-4', 3), ('Free Agent', 4)]
            else:
                new_statuses = [('Arb-2', 1), ('Arb-3', 2), ('Free Agent', 3)]
        
        elif latest_status == 'Pre-Arb':
            pre_arb_years = len(player_rows[player_rows['Normalized_Status'] == 'Pre-Arb'])
            remaining_pre_arb = max(0, 3 - pre_arb_years)
            
            year_offset = 0
            for i in range(remaining_pre_arb):
                year_offset += 1
                new_statuses.append(('Pre-Arb', year_offset))
            
            arb1_status = 'Arb-1 (Super 2)' if is_super2 else 'Arb-1'
            if is_super2:
                new_statuses.extend([
                    (arb1_status, year_offset + 1),
                    ('Arb-2', year_offset + 2),
                    ('Arb-3', year_offset + 3),
                    ('Arb-4', year_offset + 4),
                    ('Free Agent', year_offset + 5)
                ])
            else:
                new_statuses.extend([
                    (arb1_status, year_offset + 1),
                    ('Arb-2', year_offset + 2),
                    ('Arb-3', year_offset + 3),
                    ('Free Agent', year_offset + 4)
                ])  
        
        elif latest_status in ['Signed', 'Team Option', 'Player Option', 'Unknown', 
                              'Deferred', 'Buyout', 'Retained', 'RetainedBuyout', 'Active']:
            # For all these statuses, treat as end of contract - FA comes next
            # Find the last year with a Payroll value or just use latest_year
            payroll_years = player_rows[player_rows['Payroll'].notna()]['Year']
            if len(payroll_years) > 0:
                last_contract_year = payroll_years.max()
            else:
                # No payroll data, assume current year is last
                last_contract_year = latest_year
            
            # FA starts the year after last contract year
            year_offset = int(last_contract_year - latest_year + 1)
            new_statuses = [('Free Agent', year_offset)]
        
        # Now fill in the predicted statuses for existing None years
        for status, year_offset in new_statuses:
            target_year = int(latest_year + year_offset)
            # Check if this year exists in player_rows
            year_mask = player_rows['Year'] == target_year
            if year_mask.any():
                # Update existing row
                player_rows.loc[year_mask, 'Normalized_Status'] = status
            else:
                # Add new row (only if within our timeline)
                if target_year <= 2040:
                    new_row = {col: player_rows.iloc[0][col] for col in player_rows.columns
                             if col not in ['Year', 'Normalized_Status', 'Status', 'Payroll']}
                    new_row['Year'] = target_year
                    new_row['Normalized_Status'] = status
                    new_row['Status'] = np.nan
                    new_row['Payroll'] = np.nan
                    player_rows = pd.concat([player_rows, pd.DataFrame([new_row])], ignore_index=True)
        
        # Fill any remaining None years after FA year with FA
        player_rows = player_rows.sort_values('Year')
        fa_years = player_rows[player_rows['Normalized_Status'] == 'Free Agent']['Year']
        if len(fa_years) > 0:
            first_fa_year = fa_years.min()
            none_after_fa = (player_rows['Year'] >= first_fa_year) & (player_rows['Normalized_Status'].isna())
            player_rows.loc[none_after_fa, 'Normalized_Status'] = 'Free Agent'
        
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
    """
    Extend FA timeline through 2040 for all free agents.
    By this point, generate_contract_timeline should have already filled in FA statuses.
    """
    
    timeline_df = timeline_df.copy()
    
    # Find first FA year for each player
    fa_years = (timeline_df[timeline_df['Normalized_Status'] == 'Free Agent']
                .groupby('IDfg')['Year']
                .min()
                .reset_index())
    
    if fa_years.empty:
        logger.warning("No Free Agent statuses found - timeline may be incomplete")
        return timeline_df
    
    # Generate future FA rows for years beyond current timeline
    future_rows = []
    for _, row in fa_years.iterrows():
        idfg = row['IDfg']
        first_fa_year = int(row['Year'])
        
        # Get player info
        player_info = timeline_df[timeline_df['IDfg'] == idfg].iloc[0]
        
        # Find max year in timeline for this player
        max_year = int(timeline_df[timeline_df['IDfg'] == idfg]['Year'].max())
        
        # Add FA years from max_year+1 to 2040
        for year in range(max_year + 1, 2041):
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
    
    # Add new rows
    if future_rows:
        extended_timeline = pd.concat([
            timeline_df,
            pd.DataFrame(future_rows)
        ], ignore_index=True)
    else:
        extended_timeline = timeline_df
    
    # Sort and deduplicate
    extended_timeline = (extended_timeline
                        .sort_values(['IDfg', 'Year'])
                        .drop_duplicates(subset=['IDfg', 'Year'], keep='first')
                        .reset_index(drop=True))
    
    return extended_timeline
