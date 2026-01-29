#!/usr/bin/env python
"""
MLB Trade Simulator - Value Determination Pipeline
==================================================

Main entry point for the value determination pipeline. Combines projected stats
and salary data to calculate trade values for MLB players.

Pipeline Steps:
    1. Load prediction and salary data
    2. Calculate pitcher WAR from FIP
    3. Merge prediction data (SP, RP, batters)
    4. Clean and integrate salary data
    5. Normalize contract statuses
    6. Generate contract timeline
    7. Calculate WAR values and surplus
    8. Apply prospect adjustments
    9. Calculate trade values
    10. Export results

Usage:
    python main.py

Input:
    - data/generated/pipeline/: Prediction files
    - data/salary/: Salary data
    - data/prospect_data/: Prospect rankings

Output:
    - data/generated/value_by_year/player_values_complete.csv

Configuration:
    All settings are in config.py - edit there, not here.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from centralized config (SINGLE SOURCE OF TRUTH)
from value_determination.config import (
    Config, logger,
    # Backward compatibility
    OUTPUT_DIR, HITTER_COLUMNS, PITCHER_COLUMNS
)
from value_determination.data_loader import (
    load_prediction_files, merge_prediction_data, load_historical_data
)
from value_determination.salary_processor import (
    clean_salary_data, merge_salary_with_ids
)
from value_determination.contract_processor import (
    normalize_contract_status, check_none_statuses,
    generate_contract_timeline, validate_fa_years, extend_fa_timeline
)
from value_determination.value_calculator import (
    join_predictions_with_timeline, calculate_contract_value,
    calculate_surplus_value, integrate_historical_stats,
    integrate_player_statistics, post_process_export_data
)
from value_determination.trade_value import (
    analyze_contract_options, calculate_trade_values,
    add_trade_ranking_metrics, update_prospect_mlb_status
)
from value_determination.exporter import export_value_data
from value_determination.calculate_war import (
    calculate_war_components, calculate_pitcher_war,
    load_player_orgs, calculate_wrc_plus
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_pitcher_war_for_dataframe(pitcher_df: pd.DataFrame, 
                                        org_data: pd.DataFrame, 
                                        role: str) -> pd.DataFrame:
    """
    Calculate WAR for all pitchers in a DataFrame.
    
    This function properly encapsulates the pitcher WAR calculation that was
    previously inline in main(). Uses calculate_pitcher_war() from calculate_war.py.
    
    Args:
        pitcher_df: DataFrame with pitcher predictions (must have 'FIP', 'IDfg' columns)
        org_data: DataFrame with player team assignments (from load_player_orgs)
        role: 'SP' or 'RP'
        
    Returns:
        DataFrame with 'WAR' and 'IP' columns added
    """
    default_ip = Config.WAR.DEFAULT_SP_IP if role == 'SP' else Config.WAR.DEFAULT_RP_IP
    
    for idx, row in pitcher_df.iterrows():
        # Get team for park factor
        # TODO: Transition to mlbam_id matching
        team = ''
        if row['IDfg'] in org_data['IDfg'].values:
            team = org_data[org_data['IDfg'] == row['IDfg']]['Team'].iloc[0]
        
        # Build rate stats dict
        rate_stats = {
            'ERA': row.get('ERA', 0),
            'K%': row.get('K%', 0),
            'BB%': row.get('BB%', 0)
        }
        
        # Calculate WAR using centralized function
        war, components = calculate_pitcher_war(
            fip=row['FIP'],
            ip=default_ip,
            team=team,
            role=role,
            rate_stats=rate_stats
        )
        
        pitcher_df.at[idx, 'WAR'] = war
        pitcher_df.at[idx, 'IP'] = default_ip
    
    return pitcher_df


def validate_input_data(sp_data, rp_data, batter_data, salary_data) -> bool:
    """
    Validate that all required input data is present and has expected columns.
    
    Args:
        sp_data: Starting pitcher predictions
        rp_data: Relief pitcher predictions  
        batter_data: Batter predictions
        salary_data: Salary/contract data
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails and Config.Pipeline.FAIL_ON_MISSING_DATA is True
    """
    errors = []
    
    # Check pitcher data has FIP
    for name, df in [('SP', sp_data), ('RP', rp_data)]:
        required = Config.Columns.REQUIRED['pitcher_predictions']
        missing = [col for col in required if col not in df.columns]
        if missing:
            errors.append(f"{name} data missing columns: {missing}")
    
    # Check batter data
    required = Config.Columns.REQUIRED['batter_predictions']
    missing = [col for col in required if col not in batter_data.columns]
    if missing:
        errors.append(f"Batter data missing columns: {missing}")
    
    # Check salary data
    required = Config.Columns.REQUIRED['salary']
    # Handle case-insensitive column names
    salary_cols_lower = [c.lower() for c in salary_data.columns]
    for col in required:
        if col.lower() not in salary_cols_lower:
            errors.append(f"Salary data missing column: {col}")
    
    if errors:
        for error in errors:
            logger.error(error)
        if Config.Pipeline.FAIL_ON_MISSING_DATA:
            raise ValueError(f"Input validation failed: {errors}")
        return False
    
    return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Main pipeline execution.
    
    Runs all steps of the value determination pipeline in order.
    Each step is clearly logged and errors are handled gracefully.
    """
    
    logger.info("=" * 60)
    logger.info("MLB Trade Simulator - Value Determination Pipeline")
    logger.info("=" * 60)
    
    # Ensure output directories exist
    Config.Paths.ensure_directories()
    
    try:
        # ============================================================
        # Step 1: Load Data
        # ============================================================
        logger.info("\n[Step 1/10] Loading prediction and salary data...")
        sp_data, rp_data, batter_data, salary_data = load_prediction_files()
        
        # Validate input data
        validate_input_data(sp_data, rp_data, batter_data, salary_data)
        
        # Log unique status values for debugging
        status_col = 'status' if 'status' in salary_data.columns else 'Status'
        if status_col in salary_data.columns:
            logger.debug(f"Unique status values: {salary_data[status_col].unique()}")
        
        # ============================================================
        # Step 2: Calculate Pitcher WAR
        # ============================================================
        logger.info("\n[Step 2/10] Calculating pitcher WAR from FIP...")
        
        # Load team data for park factors
        org_data = load_player_orgs()
        
        # Calculate WAR for SP and RP using proper function
        sp_data = calculate_pitcher_war_for_dataframe(sp_data, org_data, role='SP')
        rp_data = calculate_pitcher_war_for_dataframe(rp_data, org_data, role='RP')
        
        logger.info(f"SP WAR: n={len(sp_data)}, avg={sp_data['WAR'].mean():.2f}")
        logger.info(f"RP WAR: n={len(rp_data)}, avg={rp_data['WAR'].mean():.2f}")
        
        # ============================================================
        # Step 3: Merge Prediction Data
        # ============================================================
        logger.info("\n[Step 3/10] Merging prediction data...")
        player_predictions = merge_prediction_data(sp_data, rp_data, batter_data)
        
        # ============================================================
        # Step 4: Clean Salary Data
        # ============================================================
        logger.info("\n[Step 4/10] Cleaning salary data...")
        salary_data_clean = clean_salary_data(salary_data)
        
        logger.debug(f"Cleaned salary data: {len(salary_data_clean)} records")
        logger.debug(f"Null values: {salary_data_clean.isnull().sum().to_dict()}")
        
        # ============================================================
        # Step 5: Merge Salary with Player IDs
        # ============================================================
        logger.info("\n[Step 5/10] Integrating player IDs with salary data...")
        # TODO: Transition from FG ID matching to MLB ID matching
        salary_data_with_id = merge_salary_with_ids(
            salary_data_clean, sp_data, rp_data, batter_data
        )
        
        # Log unmatched players
        unmatched = salary_data_with_id[salary_data_with_id['IDfg'].isna()]
        if not unmatched.empty and Config.Pipeline.LOG_UNMATCHED_PLAYERS:
            logger.warning(f"{len(unmatched)} players could not be matched to predictions")
        
        # ============================================================
        # Step 6: Normalize Contract Status & Generate Timeline
        # ============================================================
        logger.info("\n[Step 6/10] Processing contracts and generating timeline...")
        contract_data = normalize_contract_status(salary_data_with_id)
        problem_cases = check_none_statuses(contract_data)
        
        contract_timeline = generate_contract_timeline(contract_data)
        logger.info(f"Generated {len(contract_timeline)} timeline records")
        
        missing_fa = validate_fa_years(contract_timeline)
        extended_timeline = extend_fa_timeline(contract_timeline)
        
        # ============================================================
        # Step 7: Calculate Values
        # ============================================================
        logger.info("\n[Step 7/10] Calculating WAR and contract values...")
        timeline_with_war = join_predictions_with_timeline(
            extended_timeline, player_predictions
        )
        timeline_with_values = calculate_contract_value(timeline_with_war)
        timeline_with_values = calculate_surplus_value(timeline_with_values)
        
        # ============================================================
        # Step 8: Integrate Historical Data
        # ============================================================
        logger.info("\n[Step 8/10] Integrating historical data...")
        try:
            batting_history, pitching_history = load_historical_data()
            timeline_with_history = integrate_historical_stats(
                timeline_with_values, batting_history, pitching_history
            )
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
            timeline_with_history = timeline_with_values
        
        export_data = integrate_player_statistics(
            timeline_with_history, batter_data, sp_data, rp_data
        )
        export_data = post_process_export_data(export_data)
        logger.info(f"Processed {len(export_data)} total records")
        
        # ============================================================
        # Step 9: Calculate Trade Values
        # ============================================================
        logger.info("\n[Step 9/10] Calculating trade values with prospect adjustments...")
        export_data = analyze_contract_options(export_data)
        export_data = calculate_trade_values(export_data)
        export_data = add_trade_ranking_metrics(export_data)
        
        # Update prospect MLB status
        try:
            update_prospect_mlb_status(export_data)
        except Exception as e:
            logger.warning(f"Could not update prospect status: {e}")
        
        # ============================================================
        # Step 10: Export Data
        # ============================================================
        logger.info("\n[Step 10/10] Exporting value data...")
        export_value_data(export_data, OUTPUT_DIR)
        
        # Log final summary
        if 'Year' in export_data.columns and 'Status' in export_data.columns:
            year_2025 = export_data[export_data['Year'] == 2025]
            logger.info(f"2025 records: {len(year_2025)}")
            logger.info(f"Unique players: {year_2025['IDfg'].nunique()}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Output saved to: {OUTPUT_DIR / 'player_values_complete.csv'}")
        logger.info("=" * 60)
        
        return export_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
