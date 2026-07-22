#!/usr/bin/env python3
"""
MLB Prediction Pipeline - Unified Interface
============================================

Single script to handle all pipeline operations:
- Generate predictions (Marcel engine)
- Run projection engine (playing time + WAR calculation)
- Calculate trade values

Author: Niels Christoffersen
Date: January 2026
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPTS_DIR = Path(__file__).parent  # auto_train_models/scripts/
AUTO_TRAIN_DIR = SCRIPTS_DIR.parent  # auto_train_models/ (run commands from here)
DATA_DIR = AUTO_TRAIN_DIR.parent / 'data'  # LSTMLB/data/
GENERATED_DIR = DATA_DIR / 'generated'
PIPELINE_DIR = GENERATED_DIR / 'pipeline'

# Python executable - use the same interpreter running this script
PYTHON_EXE = sys.executable

# Ensure directories exist
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    """Print pipeline header"""
    print("\n" + "=" * 70)
    print("MLB PREDICTION PIPELINE (MARCEL ENGINE)")
    print("=" * 70)
    print()


def print_menu():
    """Display main menu"""
    print("\nSelect a pipeline:")
    print()
    print("  PRESEASON")
    print("    1. Generate Predictions (Marcel)")
    print("    2. Run Projections + Trade Values (Playing Time \u2192 WAR \u2192 Surplus)")
    print("    3. Run Full Pipeline (Predict \u2192 Project \u2192 Values)")
    print()
    print("  HISTORICAL")
    print("    4. Generate Historical Predictions")
    print("    5. Calculate Historical Surplus + Timeline")
    print("    6. Run Full Historical Pipeline (Predict \u2192 Surplus \u2192 Timeline)")
    print()
    print("  Q. Exit")
    print()


def get_user_choice(prompt: str, valid_choices: List[str]) -> str:
    """Get validated user input"""
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def get_projection_year() -> int:
    """Get projection year from user"""
    print("\nEnter projection year (default: 2026):")
    year_input = input("> ").strip()
    if year_input and year_input.isdigit():
        return int(year_input)
    return 2026


def run_command(command: str, description: str, timeout: int = 7200) -> bool:
    """Execute a command with progress tracking.
    
    Uses the same Python interpreter that's running this script to ensure
    subprocesses use the correct virtual environment.
    """
    if command.startswith('python '):
        command = f'"{PYTHON_EXE}" ' + command[7:]
    
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    start_time = datetime.now()
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(AUTO_TRAIN_DIR)  # Run from auto_train_models/ so imports work
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait(timeout=timeout)
        
        duration = datetime.now() - start_time
        
        if process.returncode == 0:
            logger.info(f"SUCCESS: {description} completed in {duration}")
            return True
        else:
            logger.error(f"FAILED: {description} failed with code {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: {description} timed out after {timeout}s")
        process.kill()
        return False
    except Exception as e:
        logger.error(f"ERROR: {description} failed: {str(e)}")
        return False


def generate_predictions() -> bool:
    """Generate predictions using Marcel models."""
    print("\nGenerating Marcel predictions...")
    
    command = "python scripts/predict_models.py --model-type all"
    description = "Generating all Marcel predictions"
    
    return run_command(command, description, timeout=3600)


def check_prediction_files() -> bool:
    """Check if all required prediction files exist"""
    required_files = [
        'batter_predictions.csv',
        'pitcher_predictions.csv',
        'baserunning_predictions.csv',
        'fielding_predictions.csv'
    ]
    
    missing = []
    for filename in required_files:
        filepath = PIPELINE_DIR / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        logger.error(f"ERROR: Missing prediction files: {', '.join(missing)}")
        logger.error("Please generate predictions first (option 1) or run full pipeline (option 3)")
        return False
    
    return True


def run_projection_engine(projection_year: int = 2026) -> bool:
    """
    Run the projection engine: allocate playing time, calculate WAR,
    and generate trade values.
    """
    print(f"\nRunning Projection Engine + Trade Values for {projection_year}...")
    
    if not check_prediction_files():
        return False
    
    command = "python -m value_determination.main"
    description = f"Projection engine + trade values for {projection_year}"
    
    success = run_command(command, description, timeout=600)
    
    if success:
        output_file = DATA_DIR / 'generated' / 'value_by_year' / 'player_values_complete.csv'
        if output_file.exists():
            logger.info(f"SUCCESS: Trade values saved to: {output_file}")
        else:
            logger.warning("WARNING: Output file not found at expected location")
    
    return success


def run_full_pipeline() -> bool:
    """Run complete pipeline: predict → project (playing time + WAR) → trade values"""
    print("\nStarting Full Pipeline...")
    print("This will: Generate Marcel predictions → Run Projection Engine → Calculate Trade Values")
    print("\nContinue? (y/n)")
    
    if input("> ").strip().lower() != 'y':
        print("Pipeline cancelled")
        return False
    
    start_time = datetime.now()
    projection_year = get_projection_year()
    
    print("\n" + "="*60)
    print("PHASE 1: GENERATE PREDICTIONS (MARCEL)")
    print("="*60)
    if not generate_predictions():
        logger.error("FAILED: Prediction phase failed")
        return False
    
    print("\n" + "="*60)
    print("PHASE 2: PROJECTIONS & TRADE VALUES")
    print("="*60)
    if not run_projection_engine(projection_year):
        logger.error("FAILED: Projection engine failed")
        return False
    
    duration = datetime.now() - start_time
    
    print(f"\n{'='*70}")
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total duration: {duration}")
    print(f"Projection year: {projection_year}")
    print(f"{'='*70}\n")
    
    return True


def _get_historical_year_range() -> tuple:
    """Prompt for historical year range."""
    print("\nStart year (default: 2014):")
    s = input("> ").strip()
    start = int(s) if s.isdigit() else 2014
    print("End year (default: 2026):")
    e = input("> ").strip()
    end = int(e) if e.isdigit() else 2026
    return start, end


def run_historical_predictions() -> bool:
    """Generate historical Marcel predictions for cutoff years."""
    print("\n" + "=" * 70)
    print("HISTORICAL PREDICTIONS (MARCEL)")
    print("=" * 70)

    start, end = _get_historical_year_range()

    print("\nForce regenerate existing files? (y/n)")
    force = "--force" if input("> ").strip().lower() == 'y' else ""

    command = f"python -m value_determination.pipelines.trade_history predictions --start {start} --end {end} {force}".strip()
    return run_command(command, f"Historical predictions ({start}–{end})", timeout=14400)


def run_historical_surplus() -> bool:
    """Run surplus + timeline on historical predictions."""
    print("\n" + "=" * 70)
    print("HISTORICAL SURPLUS + TIMELINE")
    print("=" * 70)

    start, end = _get_historical_year_range()

    print("\nForce regenerate? (y/n)")
    force = "--force" if input("> ").strip().lower() == 'y' else ""

    command = f"python -m value_determination.pipelines.trade_history surplus timeline --start {start} --end {end} {force}".strip()
    return run_command(command, f"Historical surplus + timeline ({start}–{end})", timeout=7200)


def run_full_historical() -> bool:
    """Run full historical pipeline: predictions → surplus → timeline."""
    print("\n" + "=" * 70)
    print("FULL HISTORICAL PIPELINE")
    print("=" * 70)

    start, end = _get_historical_year_range()

    print("\nForce regenerate existing files? (y/n)")
    force = "--force" if input("> ").strip().lower() == 'y' else ""

    print(f"\nThis will run predictions + surplus + timeline for {start}–{end}.")
    print("Continue? (y/n)")
    if input("> ").strip().lower() != 'y':
        return False

    command = f"python -m value_determination.pipelines.trade_history --start {start} --end {end} {force}".strip()
    return run_command(command, f"Full historical pipeline ({start}–{end})", timeout=14400)


def display_output_files():
    """Display information about generated files"""
    print("\nGenerated Files:")
    
    playing_time_dir = DATA_DIR / 'generated' / 'playing_time'
    value_dir = DATA_DIR / 'generated' / 'value_by_year'
    
    output_files = {
        'Raw Predictions (Rate Stats)': [
            PIPELINE_DIR / 'batter_predictions.csv',
            PIPELINE_DIR / 'pitcher_predictions.csv',
            PIPELINE_DIR / 'baserunning_predictions.csv',
            PIPELINE_DIR / 'fielding_predictions.csv'
        ],
        'Final Projections (with WAR)': [
            playing_time_dir / 'projections_2026.csv',
            playing_time_dir / 'team_summary_2026.csv'
        ],
        'Trade Values': [
            value_dir / 'player_values_complete.csv',
            value_dir / 'trade_value_history.csv',
        ]
    }
    
    for category, files in output_files.items():
        print(f"\n{category}:")
        for filepath in files:
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"  ✓ {filepath.name}: {size:,} bytes ({size/1024/1024:.1f} MB)")
            else:
                print(f"  - {filepath.name}: Not found")


def main():
    """Main pipeline interface"""
    print_header()
    
    print("Welcome to the MLB Prediction Pipeline!")
    print(f"Output directory: {PIPELINE_DIR}")
    
    while True:
        print_menu()
        choice = get_user_choice(
            "Select option (1-6 / Q): ",
            ['1', '2', '3', '4', '5', '6', 'Q', 'q']
        )
        choice = choice.upper()

        if choice == 'Q':
            print("\nExiting pipeline. Goodbye!")
            display_output_files()
            break

        elif choice == '1':
            generate_predictions()
            
        elif choice == '2':
            projection_year = get_projection_year()
            run_projection_engine(projection_year)
            
        elif choice == '3':
            run_full_pipeline()

        elif choice == '4':
            run_historical_predictions()

        elif choice == '5':
            run_historical_surplus()

        elif choice == '6':
            run_full_historical()
        
        # Ask if user wants to continue
        print("\nReturn to main menu? (y/n)")
        if input("> ").strip().lower() != 'y':
            print("\nExiting pipeline. Goodbye!")
            display_output_files()
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWARNING: Pipeline interrupted by user")
        display_output_files()
        sys.exit(0)
    except Exception as e:
        logger.error(f"ERROR: Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
