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
    
    contract_indicators = [
        'ACTIVE', 'INJURED', 'RESERVE', 'RETAINED', 'DEFERRED',
        'BURIED', 'BUYOUT', 'BONUS', 'VOIDED', 'SIGNING BONUS'
    ]

    def _status_text(value):
        return '' if pd.isna(value) else str(value).upper().strip()

    def _is_roster_status(value):
        status = _status_text(value)
        return status in ('', '-') or any(
            indicator in status for indicator in contract_indicators
        )

    def _is_guaranteed_contract_year(row, group):
        """Identify roster-looking rows that are actually contract years.

        Spotrac uses ``Active`` (and similar roster states) both for a
        player's current roster status and for years inside a guaranteed
        multi-year deal.  Service time is appropriate for the former, but
        must not turn the latter back into Pre-Arb/Arb years.

        A roster-looking paid row is treated as guaranteed when either:
        - it occurs after the player's final explicit arbitration year; or
        - the player has a multi-year paid schedule with no arbitration
          statuses (typical for major-league extensions and free-agent deals).
        """
        if pd.isna(row.get('Payroll')) or not _is_roster_status(row.get('Status')):
            return False

        paid_rows = group[group['Payroll'].notna()]
        if paid_rows.empty:
            return False

        arb_mask = group['Status'].map(_status_text).str.contains(
            r'\bARB\b', regex=True, na=False
        )
        if arb_mask.any():
            last_arb_year = pd.to_numeric(
                group.loc[arb_mask, 'Year'], errors='coerce'
            ).max()
            return (
                pd.notna(last_arb_year)
                and pd.to_numeric(row.get('Year'), errors='coerce') > last_arb_year
            )

        # Without arbitration rows, multiple paid roster-looking years are
        # the strongest available indication of a guaranteed contract.
        paid_years = pd.to_numeric(paid_rows['Year'], errors='coerce').dropna().unique()
        return len(paid_years) > 1
    
    def get_next_year_status(group):
        """Look ahead one year to determine current status for 'Estimate'"""
        group = group.copy()
        group['Next_Status'] = group['Status'].shift(-1)
        group['Guaranteed_Contract_Year'] = group.apply(
            lambda row: _is_guaranteed_contract_year(row, group), axis=1
        )
        return group

    def _status_from_service_time(row):
        """Map service time to team-control status when raw status is roster-only."""
        yos = row.get('Years_of_Service')
        if pd.isna(yos):
            return None
        try:
            yos_float = float(yos)
        except (ValueError, TypeError):
            return None

        if yos_float < 3:
            return 'Pre-Arb'
        if yos_float < 4:
            return 'Arb-1'
        if yos_float < 5:
            return 'Arb-2'
        if yos_float < 6:
            return 'Arb-3'
        return 'Signed'
    
    def _normalize_single_status(row):
        """Normalize individual status values."""
        if pd.isna(row['Status']):
            if pd.notna(row['Payroll']):
                if row.get('Guaranteed_Contract_Year', False):
                    return 'Signed'
                # A paid current-year roster row can still be a pre-arb or
                # arbitration player. Prediction-only rows have no payroll and
                # remain unresolved so the timeline generator can project them.
                return _status_from_service_time(row) or 'Signed'
            # No status and no payroll — cannot determine from this row alone.
            # Let generate_contract_timeline infer from the player's history.
            return None
        
        status = str(row['Status']).upper().strip()
        if status == '-':
            if pd.notna(row['Payroll']):
                if row.get('Guaranteed_Contract_Year', False):
                    return 'Signed'
                return _status_from_service_time(row) or 'Signed'
            return None
        
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
            # Active/Buried/Reserve/etc. are usually roster states, so use
            # service time for a lone current-year salary row.  However,
            # Spotrac also uses them inside guaranteed multi-year contracts;
            # those rows must remain signed contract years.
            if row.get('Guaranteed_Contract_Year', False):
                return 'Signed'
            service_status = _status_from_service_time(row)
            if service_status is not None:
                return service_status
            if pd.notna(row['Payroll']):
                return 'Signed'
            # No payroll AND no usable service time → could be a Spotrac placeholder
            # row beyond the actual contract (e.g. Active + NaN payroll for years
            # after the last option year).  Return None so that
            # generate_contract_timeline() infers the correct status from the last
            # valid contract year.  Real pre-arb players are caught by the
            # years_of_service check above or by ARB/UFA status tags.
            return None
        
        # If we still don't know but they have a payroll, assume signed
        if pd.notna(row['Payroll']):
            return 'Signed'
        
        return 'Unknown'
    
    # Process by player group (explicit loop to preserve IDfg across pandas 3.0+)
    groups = []
    for player_id, group in result_df.groupby('IDfg'):
        processed = get_next_year_status(group)
        if 'IDfg' not in processed.columns:
            processed['IDfg'] = player_id
        groups.append(processed)
    result_df = pd.concat(groups, ignore_index=True)
    
    # Apply normalization
    result_df['Normalized_Status'] = result_df.apply(_normalize_single_status, axis=1)
    
    # Log distribution
    status_counts = result_df['Normalized_Status'].value_counts()
    logger.info("\nStatus distribution after normalization:")
    for status, count in status_counts.items():
        logger.info(f"{status}: {count}")
    
    return result_df.drop(
        ['Next_Status', 'Guaranteed_Contract_Year'], axis=1
    )


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

        def _project_yos(anchor_yos, anchor_year, target_year, partial_anchor=False):
            """Project display YoS while preserving a partial current season."""
            if pd.isna(anchor_yos):
                return np.nan
            offset = int(target_year) - int(anchor_year)
            if partial_anchor and offset > 0:
                offset -= 1
            return max(0.0, float(anchor_yos) + offset)
        
        # Check for Super 2
        is_super2 = any('Super 2' in str(status) for status in player_rows['Normalized_Status'])
        
        # Log potential Super Two candidates based on service time
        if not is_super2:
            # Use the first/current season's service time. Future prediction
            # rows may already contain projected YoS and must not be used as
            # the starting point for the control timeline.
            first_year = player_rows['Year'].min()
            yos_vals = player_rows.loc[
                (player_rows['Year'] == first_year)
                & player_rows['Years_of_Service'].notna(),
                'Years_of_Service',
            ]
            if len(yos_vals) > 0:
                starting_yos = float(yos_vals.iloc[0])
                partial_current = bool(
                    player_rows.loc[
                        player_rows['Year'] == first_year,
                        'Partial_Current_Year',
                    ].any()
                ) if 'Partial_Current_Year' in player_rows.columns else False
                end_yos = starting_yos + (0.0 if partial_current else 1.0)
                if 2.0 <= end_yos <= 3.0:
                    player_name = player_rows['Name'].iloc[0] if 'Name' in player_rows.columns else player_rows.get('Player Name', pd.Series(['?'])).iloc[0]
                    logger.debug(f"Potential Super Two candidate: {player_name} (end-of-season YoS={end_yos:.3f})")
        
        # Find an existing FA boundary before projecting anything. Explicit
        # Spotrac FA markers are authoritative: a service-time projection
        # must never turn a known UFA year back into team control.
        explicit_fa_years = player_rows.loc[
            player_rows['Normalized_Status'] == 'Free Agent', 'Year'
        ]
        first_explicit_fa_year = (
            int(explicit_fa_years.min()) if len(explicit_fa_years) else None
        )
        if first_explicit_fa_year is not None:
            # Salary/prediction merges can leave duplicate rows for the same
            # player-year (for example, an Active projection beside Spotrac's
            # UFA marker). Once an explicit FA boundary exists, every row at
            # or after it belongs to the FA portion of the timeline.
            player_rows.loc[
                player_rows['Year'] >= first_explicit_fa_year,
                'Normalized_Status',
            ] = 'Free Agent'

        # Find the last row with a valid status. Prefer the last non-FA status,
        # because Spotrac-style timelines often include an explicit Free Agent
        # marker year that should terminate the control window rather than
        # define it.
        valid_status_mask = player_rows['Normalized_Status'].notna()
        non_fa_mask = valid_status_mask & (player_rows['Normalized_Status'] != 'Free Agent')

        if not valid_status_mask.any():
            # All statuses are None — no salary/contract data matched.
            # If the player is on an active roster (has Team), they are under
            # team control.  Use Years_of_Service to determine remaining control.
            if 'Team' in player_rows.columns and player_rows['Team'].notna().any():
                import math
                first_year = int(player_rows['Year'].min())
                yos_vals = player_rows.loc[
                    (player_rows['Year'] == first_year)
                    & player_rows['Years_of_Service'].notna(),
                    'Years_of_Service',
                ]
                if len(yos_vals) == 0:
                    yos_vals = player_rows.loc[
                        (player_rows['Year'] <= first_year)
                        & player_rows['Years_of_Service'].notna(),
                        'Years_of_Service',
                    ]
                partial_current = bool(
                    player_rows.loc[
                        player_rows['Year'] == first_year,
                        'Partial_Current_Year',
                    ].any()
                ) if 'Partial_Current_Year' in player_rows.columns else False
                if len(yos_vals) > 0:
                    starting_yos = float(yos_vals.iloc[-1])
                else:
                    starting_yos = 0.0
                end_of_season_yos = starting_yos + (
                    0.0 if partial_current else 1.0
                )

                if end_of_season_yos >= 6.0:
                    player_rows['Normalized_Status'] = 'Free Agent'
                else:
                    # Assign per-year statuses based on projected service time.
                    # YoS represents start-of-season service time for the
                    # CURRENT_YEAR.  For each prediction year we advance it
                    # by one per season and map to the right control tier.
                    for _, r in player_rows.iterrows():
                        yr = int(r['Year'])
                        proj_yos = _project_yos(
                            starting_yos,
                            first_year,
                            yr,
                            partial_anchor=partial_current,
                        )  # start-of-season YoS for yr
                        if proj_yos >= 6.0:
                            status = 'Free Agent'
                        elif proj_yos >= 5.0:
                            status = 'Arb-3'
                        elif proj_yos >= 4.0:
                            status = 'Arb-2'
                        elif proj_yos >= 3.0:
                            status = 'Arb-1'
                        else:
                            status = 'Pre-Arb'
                        player_rows.loc[r.name, 'Normalized_Status'] = status

                    # Add a Free Agent row at the transition year if not already present
                    fa_start_year = (
                        first_year
                        + math.ceil(max(0, 6.0 - starting_yos))
                        + (1 if partial_current else 0)
                    )
                    if fa_start_year <= 2040 and not (player_rows['Year'] == fa_start_year).any():
                        new_row = {col: player_rows.iloc[0][col] for col in player_rows.columns
                                  if col not in ['Year', 'Normalized_Status', 'Status', 'Payroll']}
                        new_row['Year'] = fa_start_year
                        if 'Years_of_Service' in new_row:
                            new_row['Years_of_Service'] = _project_yos(
                                starting_yos,
                                first_year,
                                fa_start_year,
                                partial_anchor=partial_current,
                            )
                        if 'Partial_Current_Year' in new_row:
                            new_row['Partial_Current_Year'] = False
                        new_row['Normalized_Status'] = 'Free Agent'
                        new_row['Status'] = np.nan
                        new_row['Payroll'] = np.nan
                        player_rows = pd.concat([player_rows, pd.DataFrame([new_row])], ignore_index=True)
            else:
                player_rows['Normalized_Status'] = 'Free Agent'
            return player_rows
        
        if non_fa_mask.any():
            last_valid_idx = player_rows[non_fa_mask].index[-1]
        else:
            last_valid_idx = player_rows[valid_status_mask].index[-1]
        latest_status = player_rows.loc[last_valid_idx, 'Normalized_Status']
        latest_year = player_rows.loc[last_valid_idx, 'Year']

        anchor_yos = player_rows.loc[last_valid_idx, 'Years_of_Service']
        if pd.isna(anchor_yos):
            prior_yos = player_rows.loc[
                (player_rows['Year'] <= latest_year)
                & player_rows['Years_of_Service'].notna(),
                'Years_of_Service',
            ]
            anchor_yos = prior_yos.iloc[-1] if len(prior_yos) else np.nan
        anchor_partial = bool(
            player_rows.loc[player_rows.index == last_valid_idx, 'Partial_Current_Year'].any()
        ) if 'Partial_Current_Year' in player_rows.columns else False
        
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
            # Use Years_of_Service (start-of-season service time) to determine
            # remaining pre-arb years.  End-of-season YoS ≈ start_yos + 1.0.
            # Arb eligible at 3.000 years; FA eligible at 6.000 years.
            latest_yos = anchor_yos
            if pd.notna(latest_yos):
                latest_yos = float(latest_yos)
                # End-of-season service time after latest_year
                end_of_season_yos = latest_yos + (
                    0.0 if anchor_partial else 1.0
                )

                import math
                remaining_to_arb = max(0, 3.0 - end_of_season_yos)
                remaining_pre_arb = math.ceil(remaining_to_arb)

                remaining_to_fa = max(0, 6.0 - end_of_season_yos)
                fa_offset_from_latest = math.ceil(remaining_to_fa)
            else:
                # Fallback: count Pre-Arb rows if no YoS data
                pre_arb_years = len(player_rows[player_rows['Normalized_Status'] == 'Pre-Arb'])
                remaining_pre_arb = max(0, 3 - pre_arb_years)
                fa_offset_from_latest = remaining_pre_arb + (4 if is_super2 else 3) + 1
            
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
        
        elif latest_status in ['Signed', 'Team Option', 'Player Option', 'Mutual Option',
                              'Vesting Option', 'Opt-Out', 'Unknown', 'Deferred', 'Buyout',
                              'Retained', 'RetainedBuyout', 'Active']:
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

            # Do not project control statuses through an explicit FA marker.
            # Any blank rows at/after that boundary are filled with FA below.
            if (
                first_explicit_fa_year is not None
                and target_year >= first_explicit_fa_year
            ):
                continue

            # Check if this year exists in player_rows
            year_mask = player_rows['Year'] == target_year
            if year_mask.any():
                # Existing normalized statuses—including options and FA
                # markers—are authoritative. Only fill prediction-only rows.
                blank_mask = year_mask & player_rows['Normalized_Status'].isna()
                player_rows.loc[blank_mask, 'Normalized_Status'] = status
            else:
                # Add new row (only if within our timeline)
                if target_year <= 2040:
                    new_row = {col: player_rows.iloc[0][col] for col in player_rows.columns
                             if col not in ['Year', 'Normalized_Status', 'Status', 'Payroll']}
                    new_row['Year'] = target_year
                    if 'Years_of_Service' in new_row:
                        new_row['Years_of_Service'] = _project_yos(
                            anchor_yos,
                            latest_year,
                            target_year,
                            partial_anchor=anchor_partial,
                        )
                    if 'Partial_Current_Year' in new_row:
                        new_row['Partial_Current_Year'] = False
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
    
    # Process each player (explicit loop to preserve IDfg across pandas versions)
    groups = []
    for player_id, group in result_df.groupby('IDfg'):
        processed = process_player_timeline(group)
        if 'IDfg' not in processed.columns:
            processed['IDfg'] = player_id
        groups.append(processed)
    result_df = pd.concat(groups, ignore_index=True)
    
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
