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
    - Current: Uses name matching + roster fallback (fg_id bridge) to find IDfg
    - Target: Use mlbam_id as primary match key once available in all data sources
    - Sportrac player_id is Spotrac-internal (NOT MLBAM)
    - Roster file bridges via fg_id (= FanGraphs IDfg)
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple

from .config import Config, logger

# Canonical name normalization — single source of truth
from core.name_utils import (
    normalize_name,
    _SUFFIX_RE as _SUFFIXES,
)


def normalize_name_no_suffix(name: str) -> str:
    """Normalize name and strip common suffixes (Jr., III, etc.) for fuzzy matching."""
    normalized = normalize_name(name)
    if pd.isna(normalized):
        return normalized
    return _SUFFIXES.sub('', normalized).strip()


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
        
        # Helper to parse dollar-string columns into numeric
        def _parse_dollar_col(series: pd.Series) -> pd.Series:
            s = series.astype(str).str.split('(').str[0]
            s = (s.str.replace('$', '', regex=False)
                  .str.replace(',', '', regex=False)
                  .str.replace('-', '', regex=False))
            return pd.to_numeric(s, errors='coerce')
        
        # Parse payroll_annual — the primary salary basis, reflecting the
        # actual cash  value of the contract year-by-year.
        cleaned_df['_payroll_annual'] = _parse_dollar_col(cleaned_df[payroll_col])
        
        # Use payroll_annual directly as the player's salary for all downstream
        # calculations (contract value, surplus, trade value).
        cleaned_df['Payroll'] = cleaned_df['_payroll_annual']
        
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
        
        # ── Deduplication ─────────────────────────────────────────────────
        # Same player can appear multiple times per year because:
        # (1) Traded players are scraped from BOTH old and new team's Spotrac
        #     page — e.g. Devers appears under Red Sox AND Giants with identical
        #     salary figures → these are pure cross-team duplicates.
        # (2) Spotrac splits salary into multiple rows on the same page — e.g.
        #     base salary + signing-bonus amortization → these should be SUMMED.
        #
        # Strategy:
        #   Step A: Drop cross-team duplicates.  Group by (player_id, Year,
        #           Payroll) and keep only the first team.  Rows with the same
        #           dollar amount on multiple teams are the same contract line
        #           scraped twice.
        #   Step B: Sum the remaining rows per (player_id, Year).  This
        #           correctly adds base salary + bonus amortization rows.
        #
        # Team is kept for name-disambiguation purposes in merge_salary_with_ids()
        # (e.g. distinguishing Luis Garcia HOU vs Luis Garcia WSH). It is NOT the
        # canonical team source — that comes from current_rosters.csv in main.py.
        if 'IDfg' in cleaned_df.columns:  # IDfg here holds the raw Spotrac player_id
            pre_dedup = len(cleaned_df)
            
            # Step A: Drop cross-team duplicates.
            # Rows that share (player_id, Year, Payroll) across different teams
            # are the same contract line scraped from multiple team pages.
            # Keep only the first occurrence.
            # Handle NaN Payroll separately to avoid dropping distinct NaN rows.
            cleaned_df['_payroll_key'] = cleaned_df['Payroll'].fillna(-9999)
            cleaned_df = cleaned_df.drop_duplicates(
                subset=['IDfg', 'Year', '_payroll_key'],
                keep='first'
            )
            cross_team_dupes = pre_dedup - len(cleaned_df)
            
            # Step B: Sum within-player-year rows.
            # After removing cross-team duplicates, remaining multiple rows for the
            # same (player_id, Year) are distinct contract components (base salary +
            # signing-bonus amortization).  Sum the salary and take first categorical.
            #
            # IMPORTANT: Spotrac encodes option years with TWO rows for the same
            # year — e.g., Cal Raleigh 2031 has "conditional-mutual, $20M" AND
            # "UFA, NaN".  We must preserve the FA marker so downstream
            # normalization can detect it.  Strategy: if ANY row in the group
            # has a status containing UFA/RFA/FA, emit a *second* row with
            # that FA status and NaN payroll (mirroring how Spotrac encodes it).
            _FA_KEYWORDS = {'UFA', 'RFA', 'FA', 'FREE AGENT'}

            def _has_fa_status(status_str) -> bool:
                if pd.isna(status_str):
                    return False
                return any(kw in str(status_str).upper() for kw in _FA_KEYWORDS)

            def _agg_salary_components(g):
                # Pick the primary (non-FA) status row
                non_fa = g[~g['Status'].apply(_has_fa_status)]
                primary = non_fa if len(non_fa) > 0 else g

                result = pd.Series({
                    'Player Name': g['Player Name'].dropna().iloc[0] if g['Player Name'].notna().any() else np.nan,
                    'Team': g['Team'].dropna().iloc[0] if g['Team'].notna().any() else np.nan,
                    'Payroll': primary['Payroll'].sum() if primary['Payroll'].notna().any() else np.nan,
                    'Status': primary['Status'].dropna().iloc[0] if primary['Status'].notna().any() else np.nan,
                    'Years_of_Service': g['Years_of_Service'].dropna().iloc[0] if g['Years_of_Service'].notna().any() else np.nan,
                    '_has_fa_marker': g['Status'].apply(_has_fa_status).any(),
                })
                return result
            pre_agg = len(cleaned_df)
            cleaned_df = (
                cleaned_df.groupby(['IDfg', 'Year'])
                .apply(_agg_salary_components, include_groups=False)
                .reset_index()  # brings IDfg and Year back as columns
            )
            components_merged = pre_agg - len(cleaned_df)

            # Re-emit FA marker rows that were absorbed during aggregation.
            # For option years where Spotrac had a separate UFA row, we need
            # to keep that signal so normalize_contract_status sees it.
            fa_marker_rows = cleaned_df[cleaned_df['_has_fa_marker'] & ~cleaned_df['Status'].apply(_has_fa_status)]
            if len(fa_marker_rows) > 0:
                fa_rows = fa_marker_rows[['IDfg', 'Year', 'Player Name', 'Team']].copy()
                fa_rows['Payroll'] = np.nan
                fa_rows['Status'] = 'UFA'
                fa_rows['Years_of_Service'] = np.nan
                cleaned_df = pd.concat([cleaned_df, fa_rows], ignore_index=True)
                logger.info(f"Preserved {len(fa_rows)} FA marker rows from dual-row option years")

            cleaned_df = cleaned_df.drop(columns=['_has_fa_marker'], errors='ignore')
            
            logger.info(
                f"Deduplication: {pre_dedup} → {len(cleaned_df)} rows "
                f"({cross_team_dupes} cross-team dupes dropped, "
                f"{components_merged} component rows merged)"
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
    # Two different players share this name: IDfg 23735 (HOU pitcher, now NYM) and Luis Garcia Jr. (WSH batter).
    salary_df['Team_lower'] = salary_df['Team'].str.lower()
    mask_garcia_pitcher = (salary_df['Name_Normalized'] == 'LUIS GARCIA') & (salary_df['Team_lower'].str.contains('hou|astros|nym|mets', na=False))
    mask_garcia_wsh = (salary_df['Name_Normalized'] == 'LUIS GARCIA') & (salary_df['Team_lower'].str.contains('wsh|washington|nationals', na=False))
    
    salary_df.loc[mask_garcia_pitcher, 'Name_Normalized'] = 'LUIS GARCIA HOU'
    salary_df.loc[mask_garcia_wsh, 'Name_Normalized'] = 'LUIS GARCIA JR'
    
    # Update player reference - active players
    player_ref.loc[player_ref['IDfg'] == 23735, 'Name_Normalized'] = 'LUIS GARCIA HOU'
    # normalize_name now strips periods, so "Luis Garcia Jr." → "LUIS GARCIA JR"
    player_ref.loc[player_ref['Name_Normalized'] == 'LUIS GARCIA JR', 'Name_Normalized'] = 'LUIS GARCIA JR'
    
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
    
    # ── Fallback matching via roster bridge + suffix stripping ──────────
    # Some salary names don't match prediction names due to nickname differences
    # (e.g. "Mike Burrows" vs "Michael Burrows") or missing suffixes
    # (e.g. "Jazz Chisholm" vs "Jazz Chisholm Jr.").
    # The roster file has mapped_name which resolves many of these.
    still_missing = merged_df['IDfg'].isna() & (merged_df['Payroll'].notna() | merged_df['Status'].notna())
    if still_missing.any():
        matched_via_fallback = 0
        try:
            roster_df = pd.read_csv(Config.Paths.ROSTER_FILE)
            
            # Build roster lookups: normalized name → IDfg (using fg_id from roster)
            roster_lookups = {}  # norm_name -> (fg_id, player_name)
            for _, row in roster_df.iterrows():
                fgid = row.get('fg_id')
                if pd.isna(fgid):
                    continue
                try:
                    fgid = int(float(fgid))
                except (ValueError, TypeError):
                    continue
                # Add mapped_name variant
                mapped = row.get('mapped_name')
                if pd.notna(mapped):
                    roster_lookups[normalize_name(mapped)] = (fgid, str(mapped))
                # Add player_name variant
                pname = row.get('player_name')
                if pd.notna(pname):
                    roster_lookups[normalize_name(pname)] = (fgid, str(pname))
                # Add suffix-stripped variants of both
                if pd.notna(mapped):
                    roster_lookups[normalize_name_no_suffix(mapped)] = (fgid, str(mapped))
                if pd.notna(pname):
                    roster_lookups[normalize_name_no_suffix(pname)] = (fgid, str(pname))
            
            # Also build a prediction lookup: IDfg -> (position_group, Name)
            pred_lookup = player_ref[['IDfg', 'position_group', 'Name']].drop_duplicates('IDfg')
            pred_lookup_dict = {
                int(row['IDfg']): (row['position_group'], row['Name'])
                for _, row in pred_lookup.iterrows()
                if pd.notna(row['IDfg'])
            }
            
            for idx in merged_df[still_missing].index:
                name_norm = merged_df.loc[idx, 'Name_Normalized']
                # Try exact roster match first, then suffix-stripped match
                fgid_match = None
                for try_name in [name_norm, _SUFFIXES.sub('', name_norm).strip() if name_norm else None]:
                    if try_name and try_name in roster_lookups:
                        fgid_match = roster_lookups[try_name]
                        break
                
                if fgid_match is not None:
                    fgid, display_name = fgid_match
                    merged_df.loc[idx, 'IDfg'] = fgid
                    if fgid in pred_lookup_dict:
                        merged_df.loc[idx, 'position_group'] = pred_lookup_dict[fgid][0]
                        merged_df.loc[idx, 'Name'] = pred_lookup_dict[fgid][1]
                    else:
                        merged_df.loc[idx, 'Name'] = display_name
                    matched_via_fallback += 1
            
            if matched_via_fallback > 0:
                logger.info(f"Roster fallback matching: resolved {matched_via_fallback} additional salary rows")
        except FileNotFoundError:
            logger.warning("Roster file not found — skipping fallback salary matching")
    
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
