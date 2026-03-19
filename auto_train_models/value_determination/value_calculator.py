"""
WAR value and contract value calculations.

Uses an empirically calibrated convex power-law model for WAR-to-dollar
conversion. The model parameters (alpha, beta) are loaded from the trade
analysis calibration file if available, otherwise hardcoded defaults are used.

Convex Model:
    value = alpha * max(WAR, 0)^beta * inflation(year)
    
    With beta > 1, the value of each additional WAR is INCREASING:
    a 5-WAR player is worth more than five 1-WAR players combined.
    This captures the scarcity premium, certainty premium, optionality,
    and roster-slot opportunity cost of elite players.

Legacy Tiered Model (deprecated, kept for reference):
    Tier 1 (0-2 WAR):  $8M/WAR
    Tier 2 (2-4 WAR):  $9M/WAR
    Tier 3 (4+ WAR):  $10M/WAR
"""

import pandas as pd
import numpy as np
from pathlib import Path

from .constants import (
    logger, WAR_VALUE_TIERS, INFLATION_RATE, BASE_YEAR,
    MIN_SALARY, ARB_PERCENT, HISTORICAL_WAR_VALUE, WAR_VALUE,
    HITTER_COLUMNS, PITCHER_COLUMNS, get_war_value
)
from .config import Config, CURRENT_YEAR
from core.name_utils import name_key as _name_key_fn

# ── Universal salary lookup ───────────────────────────────────────────────
_UNIVERSAL_SALARY_FILE = Path(__file__).resolve().parents[2] / "data" / "salary" / "universal_salary.csv"
_historical_salary_cache: dict | None = None


def _load_historical_salary() -> dict:
    """Load universal_salary.csv into a lookup dict.

    Returns ``{(name_lower, year) -> salary}`` where *salary* is the
    annual salary for that year.

    This replaces the old per-year Cot's CSV loader — universal_salary.csv
    already merges Lahman + Spotrac + Cot's with proper dedup/priority.
    """
    global _historical_salary_cache
    if _historical_salary_cache is not None:
        return _historical_salary_cache
    lookup: dict[tuple[str, int], float] = {}
    if not _UNIVERSAL_SALARY_FILE.exists():
        logger.warning("Universal salary file not found: %s", _UNIVERSAL_SALARY_FILE)
        _historical_salary_cache = lookup
        return lookup
    try:
        df = pd.read_csv(_UNIVERSAL_SALARY_FILE)
    except Exception:
        logger.warning("Could not read universal salary file: %s", _UNIVERSAL_SALARY_FILE)
        _historical_salary_cache = lookup
        return lookup
    for _, row in df.iterrows():
        name = _name_key_fn(str(row.get("player", "")))
        yr = row.get("year")
        sal = row.get("salary")
        if not name or pd.isna(yr) or pd.isna(sal):
            continue
        try:
            sal_f = float(sal)
        except (TypeError, ValueError):
            continue
        if sal_f > 0:
            lookup[(name, int(yr))] = sal_f
    logger.info("Loaded historical salary data: %d entries from universal_salary.csv", len(lookup))
    _historical_salary_cache = lookup
    return lookup


# ---------------------------------------------------------------------------
# Load convex model parameters at module initialization
# ---------------------------------------------------------------------------
_CONVEX_ALPHA, _CONVEX_BETA = Config.ConvexModel.load_calibration()


# Team name to abbreviation mapping (handles both full names and existing abbreviations)
TEAM_NAME_TO_ABBREV = {
    # Full names from salary data
    'Athletics': 'ATH',
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
    'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSH',
    # Already abbreviations (pass through)
    'ATH': 'ATH', 'OAK': 'ATH', 'PIT': 'PIT', 'SD': 'SD', 'SEA': 'SEA', 'SF': 'SF',
    'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS', 'CHC': 'CHC',
    'CHW': 'CHW', 'CIN': 'CIN', 'CLE': 'CLE', 'COL': 'COL', 'DET': 'DET',
    'HOU': 'HOU', 'KC': 'KC', 'LAA': 'LAA', 'LAD': 'LAD', 'MIA': 'MIA',
    'MIL': 'MIL', 'MIN': 'MIN', 'NYM': 'NYM', 'NYY': 'NYY', 'PHI': 'PHI',
    'STL': 'STL', 'TB': 'TB', 'TEX': 'TEX', 'TOR': 'TOR', 'WSH': 'WSH',
    # Free agent indicators
    'FA': 'FA', '- - -': 'FA', '---': 'FA', '': 'FA',
}


def normalize_team_name(team: str) -> str:
    """Convert team name (full or abbreviation) to standard 3-letter abbreviation."""
    if pd.isna(team) or team is None:
        return 'FA'
    team_str = str(team).strip()
    return TEAM_NAME_TO_ABBREV.get(team_str, team_str[:3].upper() if len(team_str) >= 3 else 'FA')


def calculate_inflation_multiplier(year: int) -> float:
    """Calculate inflation multiplier from base year."""
    return (1 + INFLATION_RATE) ** (year - BASE_YEAR)


def calculate_war_value(war: float, year: int) -> float:
    """
    Calculate WAR dollar value using the empirically calibrated convex power-law.

    Formula:
        value = alpha * WAR^beta * inflation(year)

    Reference values (2025 dollars, alpha=$8.59M, beta=1.18):
        0.5 WAR →   $1.7M       3 WAR →  $28.5M
        1.0 WAR →   $8.6M       5 WAR →  $53.2M
        2.0 WAR →  $19.4M       8 WAR → $93.8M

    Args:
        war: WAR value (negative returns $0)
        year: Year for inflation adjustment (relative to BASE_YEAR)

    Returns:
        Dollar value of WAR production for that year
    """
    if pd.isna(war) or war <= 0:
        return 0.0

    inflation = calculate_inflation_multiplier(year)
    return _CONVEX_ALPHA * (war ** _CONVEX_BETA) * inflation


def _calculate_war_value_tiered(war: float, year: int) -> float:
    """
    DEPRECATED: Legacy tiered WAR valuation (kept for reference/comparison).
    
    Uses a piecewise linear function:
        Tier 1 (0-2 WAR): $8M per WAR
        Tier 2 (2-4 WAR): $9M per WAR
        Tier 3 (4+ WAR): $10M per WAR
    
    Replaced by the convex model which better captures the superlinear 
    relationship between WAR and market value observed in real trades.
    
    Args:
        war: WAR value
        year: Year for inflation adjustment
    
    Returns:
        Dollar value of WAR (tiered, linear within each tier)
    """
    if pd.isna(war) or war <= 0:
        return 0.0
    
    value = 0.0
    remaining_war = war
    
    # Tier 1: 0-2 WAR
    tier1_war = min(remaining_war, WAR_VALUE_TIERS['tier1']['max'])
    value += tier1_war * WAR_VALUE_TIERS['tier1']['value']
    remaining_war -= tier1_war
    
    if remaining_war <= 0:
        return value * calculate_inflation_multiplier(year)
    
    # Tier 2: 2-4 WAR
    tier2_war = min(remaining_war, WAR_VALUE_TIERS['tier2']['max'] - WAR_VALUE_TIERS['tier1']['max'])
    value += tier2_war * WAR_VALUE_TIERS['tier2']['value']
    remaining_war -= tier2_war
    
    if remaining_war <= 0:
        return value * calculate_inflation_multiplier(year)
    
    # Tier 3: 4+ WAR
    value += remaining_war * WAR_VALUE_TIERS['tier3']['value']
    
    return value * calculate_inflation_multiplier(year)


def join_predictions_with_timeline(extended_timeline: pd.DataFrame,
                                   player_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Join predictions with timeline and calculate WAR values.
    
    Args:
        extended_timeline: Extended contract timeline
        player_predictions: Player prediction data
        
    Returns:
        Timeline with WAR values calculated
    """
    # Aggregate WAR by (IDfg, prediction_year) before merging so that
    # two-way players (who appear in both batter and pitcher datasets)
    # produce ONE timeline row with their combined WAR, not two rows.
    war_agg = (
        player_predictions
        .groupby(['IDfg', 'prediction_year'], as_index=False)['WAR']
        .sum()
    )

    # Join predictions with timeline
    timeline_with_war = extended_timeline.merge(
        war_agg,
        left_on=['IDfg', 'Year'],
        right_on=['IDfg', 'prediction_year'],
        how='left'
    )
    
    # Calculate WAR values
    timeline_with_war['Base_Value'] = timeline_with_war.apply(
        lambda x: calculate_war_value(x['WAR'], x['Year']),
        axis=1
    )
    
    # Clean up and validate
    if 'prediction_year' in timeline_with_war.columns:
        timeline_with_war = timeline_with_war.drop('prediction_year', axis=1)
    
    logger.info(f"Processed {len(timeline_with_war)} rows")
    logger.info(f"Average WAR value: ${timeline_with_war['Base_Value'].mean():,.2f}")
    
    return timeline_with_war


def calculate_contract_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate contract values comparing current and previous arb percentages.
    
    Args:
        df: DataFrame with Base_Value and Normalized_Status columns
        
    Returns:
        DataFrame with contract_value column added
    """
    result = df.copy()
    result = result.sort_values(['IDfg', 'Year'])
    result['contract_value'] = np.nan
    
    for player_id in result['IDfg'].unique():
        player_mask = result['IDfg'] == player_id
        player_data = result[player_mask].copy()
        
        prev_value = 0
        prev_arb_pct = 1  # Start with 1 for first arb year
        
        for idx, row in player_data.iterrows():
            current_value = row['Base_Value']
            status = row['Normalized_Status']
            
            if pd.notna(row['Payroll']):
                contract_value = float(row['Payroll'])
            
            elif status == 'Pre-Arb':
                contract_value = max(MIN_SALARY['Pre-Arb'], prev_value)
                prev_arb_pct = 1
            
            elif status in ARB_PERCENT:
                min_salary = MIN_SALARY.get(status, MIN_SALARY['Arb-1'])
                arb_pct = ARB_PERCENT[status]
                
                if current_value >= 0:
                    # Calculate value based on current production vs previous salary adjusted
                    current_level_value = max(
                        current_value * arb_pct,
                        prev_value * (arb_pct / prev_arb_pct) * 1.1
                    )
                    
                    contract_value = max(
                        min_salary,
                        current_level_value
                    )
                else:
                    # For negative production, allow decrease but not below minimum
                    contract_value = max(
                        min_salary,
                        current_value * arb_pct
                    )
                
                prev_arb_pct = arb_pct
            
            else:
                contract_value = None
            
            result.loc[idx, 'contract_value'] = contract_value
            
            if pd.notna(contract_value):
                prev_value = contract_value
    
    # Validate results
    valid_contracts = result['contract_value'].notna().sum()
    logger.info(f"Processed {len(result)} rows")
    logger.info(f"Contract values calculated: {valid_contracts}")
    
    return result


def calculate_surplus_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate surplus value.
    Surplus = Base Value - Contract Value
    Only calculated for rows with existing contract values.
    
    Args:
        df: DataFrame with Base_Value and contract_value columns
        
    Returns:
        DataFrame with surplus_value column added
    """
    result = df.copy()
    
    # Verify contract_value exists
    if 'contract_value' not in result.columns:
        raise ValueError("contract_value column not found in dataframe")
    
    # Calculate surplus value only where contract_value exists
    result['surplus_value'] = np.where(
        result['contract_value'].notna(),
        result['Base_Value'] - result['contract_value'],
        np.nan
    )
    
    # Validate results
    valid_surplus = result['surplus_value'].notna().sum()
    avg_surplus = result['surplus_value'].mean()
    
    logger.info(f"Calculated {valid_surplus} surplus values")
    logger.info(f"Average surplus value: ${avg_surplus:,.2f}")
    
    return result


def integrate_historical_stats(timeline_df: pd.DataFrame,
                               batting_history: pd.DataFrame,
                               pitching_history: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical stats (2002-2024) for prediction players.
    
    Args:
        timeline_df: Current timeline with predictions
        batting_history: Historical batting data
        pitching_history: Historical pitching data
        
    Returns:
        Combined timeline with historical data
    """
    # Get current players info
    current_players = (timeline_df[['IDfg', 'Name', 'position_group']]
                      .drop_duplicates(subset=['IDfg']))
    
    # Format batting data
    batter_cols = ['IDfg', 'Season', 'Name', 'Team', 'G', 'WAR', 'BB%', 'K%', 'AVG',
                   'OBP', 'SLG', 'OPS', 'wOBA', 'wRC+', 'Bat', 'BsR', 'Def', 'Age', 
                   'HR', '2B', '3B', 'R', 'RBI', 'SB', 'CS']
    
    # Filter to columns that exist
    available_batter_cols = [c for c in batter_cols if c in batting_history.columns]
    
    batting_filtered = (batting_history[batting_history['IDfg'].isin(current_players['IDfg'])]
                       [available_batter_cols]
                       .rename(columns={'Season': 'Year', 'WAR': 'WAR_batter',
                                       'BB%': 'BB%_bat', 'K%': 'K%_bat', 'G': 'G_bat'}))
    
    # Format pitching data
    pitcher_cols = ['IDfg', 'GS', 'Season', 'Name', 'Team', 'G', 'WAR', 'ERA', 'FIP', 'SIERA', 'IP',
                    'K%', 'BB%', 'Age', 'HR/FB', 'FB%', 'GB%', 'K/9', 'BB/9', 'HR/9']
    
    # Filter to columns that exist
    available_pitcher_cols = [c for c in pitcher_cols if c in pitching_history.columns]
    
    pitching_filtered = (pitching_history[pitching_history['IDfg'].isin(current_players['IDfg'])]
                        [available_pitcher_cols]
                        .rename(columns={'Season': 'Year', 'WAR': 'WAR_pitcher',
                                        'K%': 'K%_pit', 'BB%': 'BB%_pit', 'G': 'G_pit',
                                        'HR/FB': 'HR/FB_pit',
                                        'FB%': 'FB%_pit', 'GB%': 'GB%_pit'}))
    
    # Merge batting and pitching data - use only IDfg and Year to avoid duplicates
    # Name, Team, Age can differ slightly between batting/pitching datasets
    historical = batting_filtered.merge(
        pitching_filtered,
        on=['IDfg', 'Year'],
        how='outer',
        suffixes=('_bat', '_pit')
    )
    
    # Consolidate duplicate columns (Name, Team, Age)
    # Prefer batting data if available, otherwise use pitching data
    if 'Name_bat' in historical.columns and 'Name_pit' in historical.columns:
        historical['Name'] = historical['Name_bat'].fillna(historical['Name_pit'])
        historical = historical.drop(['Name_bat', 'Name_pit'], axis=1)
    
    if 'Team_bat' in historical.columns and 'Team_pit' in historical.columns:
        historical['Team'] = historical['Team_bat'].fillna(historical['Team_pit'])
        historical = historical.drop(['Team_bat', 'Team_pit'], axis=1)
    
    if 'Age_bat' in historical.columns and 'Age_pit' in historical.columns:
        historical['Age'] = historical['Age_bat'].fillna(historical['Age_pit'])
        historical = historical.drop(['Age_bat', 'Age_pit'], axis=1)
    
    # Add position info from current data
    historical = historical.merge(current_players[['IDfg', 'position_group']], on='IDfg')
    
    # Fill NaN WAR values with 0
    historical['WAR_batter'] = historical['WAR_batter'].fillna(0)
    historical['WAR_pitcher'] = historical['WAR_pitcher'].fillna(0)
    
    # Calculate total WAR
    historical['WAR'] = historical['WAR_batter'] + historical['WAR_pitcher']
    
    # Add status columns
    historical['Status'] = 'NA'
    historical['Normalized_Status'] = 'NA'
    historical['Payroll'] = np.nan
    
    # Calculate base value using the convex model (consistent with prediction years)
    historical['Base_Value'] = historical.apply(
        lambda x: calculate_war_value(x['WAR'], int(x['Year'])), axis=1
    )

    # ── Populate salary for historical rows from Cot's by-year data ──────
    # Use lowercase ``contract_value`` to match the column already present in
    # the timeline DataFrame (which was created by calculate_contract_value).
    salary_lookup = _load_historical_salary()
    contract_vals = []
    for _, row in historical.iterrows():
        name_key = _name_key_fn(str(row.get("Name", "")))
        year_key = int(row["Year"]) if pd.notna(row["Year"]) else 0
        sal = salary_lookup.get((name_key, year_key))
        contract_vals.append(sal)
    historical['contract_value'] = contract_vals
    historical['surplus_value'] = np.where(
        pd.notna(historical['contract_value']),
        historical['Base_Value'] - historical['contract_value'],
        np.nan,
    )

    # ── Preserve salary data from timeline before merging ────────────────
    # The timeline_df has real contract_value (from Spotrac/contract_processor)
    # for current-contract years.  Historical rows will overwrite these via
    # drop_duplicates(keep='last'), so we save a mapping and patch afterwards.
    timeline_salary = (
        timeline_df[timeline_df['contract_value'].notna()]
        .set_index(['IDfg', 'Year'])['contract_value']
        .to_dict()
    )

    # Combine with timeline
    complete_timeline = pd.concat([timeline_df, historical], ignore_index=True)
    
    # Sort and remove duplicates, keeping LAST (historical data) so that
    # real game stats replace prediction-based rows.
    complete_timeline = (complete_timeline
                        .sort_values(['IDfg', 'Year'])
                        .drop_duplicates(subset=['IDfg', 'Year'], keep='last')
                        .reset_index(drop=True))

    # ── Restore salary for overlapping years ─────────────────────────────
    # For rows where the historical dedup clobbered the timeline salary,
    # reinstate the contract_value from the timeline (Spotrac).
    if timeline_salary:
        for (idfg, yr), sal in timeline_salary.items():
            mask = (complete_timeline['IDfg'] == idfg) & (complete_timeline['Year'] == yr)
            idx = complete_timeline.index[mask]
            if len(idx) > 0:
                existing = complete_timeline.loc[idx[0], 'contract_value']
                if pd.isna(existing):
                    complete_timeline.loc[idx[0], 'contract_value'] = sal
                    bv = complete_timeline.loc[idx[0], 'Base_Value']
                    if pd.notna(bv):
                        complete_timeline.loc[idx[0], 'surplus_value'] = bv - sal
    
    logger.info(f"Added historical records. New shape: {complete_timeline.shape}")
    
    return complete_timeline


def integrate_player_statistics(value_data: pd.DataFrame,
                                batter_data: pd.DataFrame,
                                sp_data: pd.DataFrame,
                                rp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate stats with combined positions for two-way players.
    
    Args:
        value_data: Timeline with value calculations
        batter_data: Batter prediction data
        sp_data: Starting pitcher data
        rp_data: Relief pitcher data
        
    Returns:
        Combined data with integrated statistics
    """
    # Split data
    historical_data = value_data[value_data['Year'] < CURRENT_YEAR].copy()
    prediction_data = value_data[value_data['Year'] >= CURRENT_YEAR].copy()
    
    # Clean prediction data - keep only essential columns
    essential_cols = ['Name', 'IDfg', 'position_group', 'Year', 'Team',
                     'Payroll', 'Status', 'Normalized_Status', 'WAR', 'Base_Value',
                     'contract_value', 'surplus_value']
    available_essential = [c for c in essential_cols if c in prediction_data.columns]
    prediction_data = prediction_data[available_essential].copy()
    
    # Find two-way players
    batter_ids = set(batter_data['IDfg'].unique())
    pitcher_ids = set(sp_data['IDfg'].unique()) | set(rp_data['IDfg'].unique())
    two_way_players = batter_ids.intersection(pitcher_ids)
    print(f"Found {len(two_way_players)} two-way players")
    
    # Add two-way flag
    prediction_data['Two_Way'] = prediction_data['IDfg'].isin(two_way_players)
    
    # Prepare batter stats for merging
    batter_stat_cols = ['IDfg', 'prediction_year', 'WAR', 'Position'] + \
                       [col for col in HITTER_COLUMNS if col not in ['Name', 'IDfg', 'WAR', 'Position']]
    available_batter_cols = [c for c in batter_stat_cols if c in batter_data.columns]
    
    batter_stats = (batter_data[available_batter_cols]
                   .rename(columns={
                       'prediction_year': 'Year',
                       'BB%': 'BB%_bat',
                       'K%': 'K%_bat',
                       'G': 'G_bat',
                       'Age': 'Age_bat',
                       'WAR': 'WAR_batter',
                       'Position': 'Position_batter'
                   }))
    
    # Prepare pitcher stats for merging
    pitcher_stat_cols = ['IDfg', 'prediction_year', 'WAR', 'Position'] + \
                        [col for col in PITCHER_COLUMNS if col not in ['Name', 'IDfg', 'WAR', 'Position']]
    
    sp_available = [c for c in pitcher_stat_cols if c in sp_data.columns]
    rp_available = [c for c in pitcher_stat_cols if c in rp_data.columns]
    
    pitcher_stats = (pd.concat([
        sp_data[sp_available],
        rp_data[rp_available]
    ])
    .rename(columns={
        'prediction_year': 'Year',
        'BB%': 'BB%_pit',
        'K%': 'K%_pit',
        'HR/FB': 'HR/FB_pit',
        'FB%': 'FB%_pit',
        'GB%': 'GB%_pit',
        'Age': 'Age_pit',
        'G': 'G_pit',
        'WAR': 'WAR_pitcher',
        'Position': 'Position_pitcher'
    })
    .drop_duplicates(subset=['IDfg', 'Year']))
    
    # Merge stats
    prediction_data = prediction_data.merge(batter_stats, on=['IDfg', 'Year'], how='left')
    prediction_data = prediction_data.merge(pitcher_stats, on=['IDfg', 'Year'], how='left')
    
    # Handle positions and WAR for two-way players
    mask = prediction_data['Two_Way']
    
    # Combine positions - check if columns exist first
    if 'Position_batter' in prediction_data.columns and 'Position_pitcher' in prediction_data.columns:
        prediction_data.loc[mask, 'Position'] = prediction_data.loc[mask].apply(
            lambda x: f"{x['Position_pitcher']}/{x['Position_batter']}" if pd.notna(x['Position_pitcher']) else x['Position_batter'],
            axis=1
        )
        
        # Single position for non-two-way players
        prediction_data.loc[~mask, 'Position'] = prediction_data.loc[~mask, 'Position_batter'].fillna(
            prediction_data.loc[~mask, 'Position_pitcher']
        )
        
        # Clean up position columns
        prediction_data = prediction_data.drop(['Position_batter', 'Position_pitcher'], axis=1, errors='ignore')
    elif 'Position_batter' in prediction_data.columns:
        prediction_data['Position'] = prediction_data['Position_batter']
        prediction_data = prediction_data.drop('Position_batter', axis=1, errors='ignore')
    elif 'Position_pitcher' in prediction_data.columns:
        prediction_data['Position'] = prediction_data['Position_pitcher']
        prediction_data = prediction_data.drop('Position_pitcher', axis=1, errors='ignore')
    
    # Handle WAR
    if 'WAR_batter' in prediction_data.columns and 'WAR_pitcher' in prediction_data.columns:
        prediction_data.loc[mask, 'WAR'] = (
            prediction_data.loc[mask, 'WAR_batter'].fillna(0) +
            prediction_data.loc[mask, 'WAR_pitcher'].fillna(0)
        )
        prediction_data.loc[~mask, 'WAR'] = prediction_data.loc[~mask, 'WAR_batter'].fillna(
            prediction_data.loc[~mask, 'WAR_pitcher']
        )
    elif 'WAR_batter' in prediction_data.columns:
        prediction_data['WAR'] = prediction_data['WAR_batter']
    elif 'WAR_pitcher' in prediction_data.columns:
        prediction_data['WAR'] = prediction_data['WAR_pitcher']
    
    # Handle Age
    if 'Age_bat' in prediction_data.columns and 'Age_pit' in prediction_data.columns:
        prediction_data['Age'] = prediction_data['Age_bat'].fillna(prediction_data['Age_pit'])
        prediction_data = prediction_data.drop(['Age_bat', 'Age_pit'], axis=1, errors='ignore')
    elif 'Age_bat' in prediction_data.columns:
        prediction_data['Age'] = prediction_data['Age_bat']
        prediction_data = prediction_data.drop('Age_bat', axis=1, errors='ignore')
    elif 'Age_pit' in prediction_data.columns:
        prediction_data['Age'] = prediction_data['Age_pit']
        prediction_data = prediction_data.drop('Age_pit', axis=1, errors='ignore')
    
    # Combine and sort
    result = pd.concat([historical_data, prediction_data])
    return result.sort_values(['IDfg', 'Year'])


def post_process_export_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process export data:
    - Replace status with normalized status
    - Drop unnecessary columns
    - Calculate OPS
    - Handle two-way player values
    - Round and handle negative values
    - Fill team NA values
    
    Args:
        df: Export data DataFrame
        
    Returns:
        Processed DataFrame
    """
    export_data = df.copy()
    
    # Replace status with normalized status
    if 'Normalized_Status' in export_data.columns:
        export_data['Status'] = export_data['Normalized_Status']
        export_data = export_data.drop('Normalized_Status', axis=1, errors='ignore')
    
    # Drop unnecessary columns
    export_data = export_data.drop('Contract_Value', axis=1, errors='ignore')
    export_data = export_data.drop('Payroll', axis=1, errors='ignore')
    
    # Calculate combined WAR value for two-way players using convex model
    if 'Two_Way' in export_data.columns:
        two_way_mask = export_data['Two_Way'] == True
        if two_way_mask.any():
            total_war = (export_data.loc[two_way_mask, 'WAR_batter'].fillna(0) + 
                        export_data.loc[two_way_mask, 'WAR_pitcher'].fillna(0))
            export_data.loc[two_way_mask, 'Base_Value'] = [
                calculate_war_value(w, int(y))
                for w, y in zip(total_war, export_data.loc[two_way_mask, 'Year'])
            ]
            export_data.loc[two_way_mask, 'surplus_value'] = (
                export_data.loc[two_way_mask, 'Base_Value'] - 
                export_data.loc[two_way_mask, 'contract_value']
            )
    
    # Round and handle negative values, preserving NaN
    columns_to_process = ['HR', '2B', '3B', 'RBI', 'R', 'SB', 'CS']
    for col in columns_to_process:
        if col in export_data.columns:
            export_data[col] = export_data[col].apply(lambda x: max(x, 0) if pd.notna(x) else x)
            export_data[col] = export_data[col].apply(lambda x: round(x) if pd.notna(x) else x)
    
    # Add OPS
    if 'OBP' in export_data.columns and 'SLG' in export_data.columns:
        export_data['OPS'] = np.where(
            export_data['OBP'].notna() & export_data['SLG'].notna(),
            export_data['OBP'] + export_data['SLG'],
            np.nan
        )
    
    # If team value is nan fill with "FA", and normalize all team names to abbreviations
    if 'Team' in export_data.columns:
        export_data['Team'] = export_data['Team'].apply(normalize_team_name)
    
    return export_data
