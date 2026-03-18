#!/usr/bin/env python3
"""
Historical Value Determination Pipeline
----------------------------------------
Generates historical predictions for past snapshot years and computes the historical
surplus files and the final trade_value_history.csv chart data.

Uses pretrained (classical) models for all predictions to ensure consistency and
stability across all time periods:

  Model approach (all cutoff years):
    - Pitcher:     pretrained classical model (13 features)
    - Batter:      pretrained classical model (13 features)
    - Fielding:    classical UZR/150 (non-catchers) + DRS/150 (catchers) projections
    - Baserunning: classical BsR/SB/CS rates with simple aging curves

  Classical projections use traditional metrics available across all eras (2000+):
    - Fielding: UZR/150, DRS/150 with piecewise-linear aging decline
    - Baserunning: FanGraphs BsR rates with age-based speed decay

  The surplus calculator supports these columns as standard fallbacks.

Usage:
  python -m auto_train_models.historical_pipeline.run_historical --start 2014 --end 2026
"""

import sys
import argparse
import subprocess
import logging
import shutil
from pathlib import Path

from auto_train_models.historical_pipeline.classical_projections import (
    generate_classical_fielding,
    generate_classical_baserunning,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("historical_pipeline")

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "auto_train_models" / "scripts"
TRADE_ANALYSIS_DIR = ROOT_DIR / "auto_train_models" / "trade_analysis"
VALUE_DET_DIR = ROOT_DIR / "auto_train_models" / "value_determination"
DATA_PROJECTIONS = ROOT_DIR / "data" / "generated" / "trade_analysis" / "projections"

def run_predictions_for_cutoff(cutoff_year: int):
    """Generate predictions for a single cutoff year using pretrained models."""
    logger.info(f"=== Generating Predictions for Cutoff Year {cutoff_year} ===")
    
    out_dir = DATA_PROJECTIONS / f"cutoff_{cutoff_year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    predict_script = str(SCRIPTS_DIR / "predict_models.py")
    base_cmd = [sys.executable, predict_script, "--cutoff-year", str(cutoff_year), "--output-dir", str(out_dir)]
    
    # ── Pitcher (pretrained classical) ─────────────────────────────────────
    logger.info(f"Pitcher: pretrained classical model (cutoff {cutoff_year})")
    subprocess.run(base_cmd + ["--model-type", "pitcher", "--use-pretrained"], check=True)
    
    # ── Batter (pretrained classical) ──────────────────────────────────────
    logger.info(f"Batter: pretrained classical model (cutoff {cutoff_year})")
    subprocess.run(base_cmd + ["--model-type", "batter", "--use-pretrained"], check=True)
    
    # ── Fielding (classical projections) ───────────────────────────────────
    fielding_file = str(out_dir / "fielding_predictions.csv")
    logger.info(f"Fielding: classical UZR/DRS projections (cutoff {cutoff_year})")
    generate_classical_fielding(cutoff_year, fielding_file)
    
    # ── Baserunning (classical projections) ────────────────────────────────
    baserunning_file = str(out_dir / "baserunning_predictions.csv")
    logger.info(f"Baserunning: classical BsR projections (cutoff {cutoff_year})")
    generate_classical_baserunning(cutoff_year, baserunning_file)

def main():
    parser = argparse.ArgumentParser(description="Run the historical prediction and valuation timeline.")
    parser.add_argument("--start", type=int, default=2014, help="Starting snapshot year (e.g., 2014 means cutoff 2013)")
    parser.add_argument("--end", type=int, default=2025, help="Ending snapshot year (must have Cot's salary data)")
    parser.add_argument("--skip-predict", action="store_true", help="Skip the ML prediction phase and just run surplus & timelines")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(f"Starting Historical Pipeline from {args.start} to {args.end}")
    logger.info("=" * 60)

    # 1. Run Predictions
    if not args.skip_predict:
        for snap_year in range(args.start, args.end + 1):
            cutoff = snap_year - 1
            try:
                run_predictions_for_cutoff(cutoff)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed generating predictions for cutoff {cutoff}: {e}")
                sys.exit(1)
    else:
        logger.info("Skipping ML prediction loop (--skip-predict passed)")

    # 2. Compute Surplus Files
    logger.info("\n=== Computing Surplus Output Files ===")
    surplus_script = str(TRADE_ANALYSIS_DIR / "surplus_calculator.py")
    try:
        subprocess.run([
            sys.executable, surplus_script, 
            "--start", str(args.start), 
            "--end", str(args.end), 
            "--force"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate surplus files: {e}")
        sys.exit(1)

    # 3. Generate Trade Value History (the final output chart mapping)
    logger.info("\n=== Generating Combined Trade Value History ===")
    history_script = str(VALUE_DET_DIR / "generate_trade_value_history.py")
    try:
        subprocess.run([sys.executable, "-m", "auto_train_models.value_determination.generate_trade_value_history"], cwd=str(ROOT_DIR), check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate final trade value history: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Historical Pipeline completed successfully!")

if __name__ == "__main__":
    main()
