#!/usr/bin/env python
"""
Run the value determination pipeline.

This script is a convenience wrapper to run the value determination pipeline
from the project root directory or from auto_train_models.

Usage:
    python run_value_determination.py
    python run_value_determination.py --pipeline-dir ../data/generated/pipeline/preseason --output-file player_values_preseason_2026.csv
"""

import argparse
from value_determination.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Value Determination Pipeline")
    parser.add_argument("--pipeline-dir", default=None,
                        help="Override directory for prediction CSVs")
    parser.add_argument("--output-file", default=None,
                        help="Override output CSV filename")
    args = parser.parse_args()
    main(pipeline_dir=args.pipeline_dir, output_filename=args.output_file)
