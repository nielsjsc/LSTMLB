#!/usr/bin/env python
"""
MLB Trade Simulator - Value Determination Pipeline
Main entry point that replicates the determine_value.ipynb notebook.

This script combines projected stats and salary data to calculate trade value.

Usage:
    python main.py

Input data locations:
    - data/generated/pipeline/: Prediction files (batter_predictions.csv, pitcher_predictions.csv)
    - data/historic_mlb/: Historical MLB data
    - data/salary/: Salary data (mlb_salary_data.csv)

Output:
    - data/generated/value_by_year/player_values_complete.csv
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from value_determination.constants import (
    logger, ensure_directories, OUTPUT_DIR, HITTER_COLUMNS, PITCHER_COLUMNS
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


def main():
    """Main pipeline execution."""
    
    logger.info("=" * 60)
    logger.info("MLB Trade Simulator - Value Determination Pipeline")
    logger.info("=" * 60)
    
    # Ensure output directories exist
    ensure_directories()
    
    try:
        # ============================================================
        # Step 1: Load Data
        # ============================================================
        logger.info("\n[Step 1] Loading prediction and salary data...")
        sp_data, rp_data, batter_data, salary_data = load_prediction_files()
        
        # Analyze unique status values
        print("\nUnique status values found:")
        if 'status' in salary_data.columns:
            print(salary_data['status'].unique())
        elif 'Status' in salary_data.columns:
            print(salary_data['Status'].unique())
        
        # ============================================================
        # Step 2: Merge Prediction Data
        # ============================================================
        logger.info("\n[Step 2] Merging prediction data...")
        player_predictions = merge_prediction_data(sp_data, rp_data, batter_data)
        
        # ============================================================
        # Step 3: Clean Salary Data
        # ============================================================
        logger.info("\n[Step 3] Cleaning salary data...")
        salary_data_clean = clean_salary_data(salary_data)
        
        print("\nSample of cleaned salary data:")
        print(salary_data_clean.head())
        print("\nData validation:")
        print(f"Null values:\n{salary_data_clean.isnull().sum()}")
        
        # Print column info for debugging
        print("\nSP columns:", sp_data.columns.tolist())
        print("RP columns:", rp_data.columns.tolist())
        print("Batter columns:", batter_data.columns.tolist())
        
        # ============================================================
        # Step 4: Merge Salary with Player IDs
        # ============================================================
        logger.info("\n[Step 4] Integrating player IDs with salary data...")
        salary_data_with_id = merge_salary_with_ids(
            salary_data_clean, sp_data, rp_data, batter_data
        )
        
        # Display unmatched players
        unmatched = salary_data_with_id[salary_data_with_id['IDfg'].isna()]
        if not unmatched.empty:
            print("\nSample unmatched players:")
            if 'Player Name' in unmatched.columns:
                print(unmatched['Player Name'].unique()[:10])
        
        # ============================================================
        # Step 5: Normalize Contract Status
        # ============================================================
        logger.info("\n[Step 5] Normalizing contract statuses...")
        contract_data = normalize_contract_status(salary_data_with_id)
        
        # Check for None statuses
        problem_cases = check_none_statuses(contract_data)
        
        # ============================================================
        # Step 6: Generate Contract Timeline
        # ============================================================
        logger.info("\n[Step 6] Generating contract timeline...")
        contract_timeline = generate_contract_timeline(contract_data)
        logger.info(f"Generated {len(contract_timeline)} timeline records")
        
        # Validate FA years
        missing_fa = validate_fa_years(contract_timeline)
        
        # ============================================================
        # Step 7: Extend FA Timeline
        # ============================================================
        logger.info("\n[Step 7] Extending FA timeline through 2040...")
        extended_timeline = extend_fa_timeline(contract_timeline)
        
        # ============================================================
        # Step 8: Calculate WAR Values
        # ============================================================
        logger.info("\n[Step 8] Calculating WAR values...")
        timeline_with_war = join_predictions_with_timeline(
            extended_timeline, player_predictions
        )
        
        # ============================================================
        # Step 9: Calculate Contract Values
        # ============================================================
        logger.info("\n[Step 9] Calculating contract values...")
        timeline_with_values = calculate_contract_value(timeline_with_war)
        
        # ============================================================
        # Step 10: Calculate Surplus Values
        # ============================================================
        logger.info("\n[Step 10] Calculating surplus values...")
        timeline_with_values = calculate_surplus_value(timeline_with_values)
        
        # ============================================================
        # Step 11: Integrate Historical Data
        # ============================================================
        logger.info("\n[Step 11] Integrating historical data...")
        try:
            batting_history, pitching_history = load_historical_data()
            timeline_with_history = integrate_historical_stats(
                timeline_with_values, batting_history, pitching_history
            )
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
            timeline_with_history = timeline_with_values
        
        # ============================================================
        # Step 12: Integrate Player Statistics
        # ============================================================
        logger.info("\n[Step 12] Integrating player statistics...")
        export_data = integrate_player_statistics(
            timeline_with_history, batter_data, sp_data, rp_data
        )
        print(f"Records processed: {len(export_data)}")
        print(f"Columns: {export_data.columns.tolist()}")
        
        # ============================================================
        # Step 13: Post-process Export Data
        # ============================================================
        logger.info("\n[Step 13] Post-processing export data...")
        export_data = post_process_export_data(export_data)
        
        # ============================================================
        # Step 14: Analyze Contract Options
        # ============================================================
        logger.info("\n[Step 14] Analyzing contract options...")
        export_data = analyze_contract_options(export_data)
        
        # ============================================================
        # Step 15: Calculate Trade Values
        # ============================================================
        logger.info("\n[Step 15] Calculating trade values...")
        export_data = calculate_trade_values(export_data)
        
        # ============================================================
        # Step 16: Add Trade Ranking Metrics
        # ============================================================
        logger.info("\n[Step 16] Adding trade ranking metrics...")
        export_data = add_trade_ranking_metrics(export_data)
        
        # ============================================================
        # Step 17: Update Prospect MLB Status
        # ============================================================
        logger.info("\n[Step 17] Updating prospect MLB status...")
        try:
            update_prospect_mlb_status(export_data)
        except Exception as e:
            logger.warning(f"Could not update prospect status: {e}")
        
        # ============================================================
        # Step 18: Export Data
        # ============================================================
        logger.info("\n[Step 18] Exporting value data...")
        export_value_data(export_data, OUTPUT_DIR)
        
        # Print final status distribution
        print("\nFinal Status Distribution:")
        if 'Year' in export_data.columns and 'Status' in export_data.columns:
            print(export_data.groupby(['Year', 'Status']).size().unstack(fill_value=0))
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Output saved to: {OUTPUT_DIR / 'player_values_complete.csv'}")
        logger.info("=" * 60)
        
        return export_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
