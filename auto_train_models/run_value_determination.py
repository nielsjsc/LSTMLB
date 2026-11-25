#!/usr/bin/env python
"""
CLI wrapper for Preseason / Full Value Determination (`valuation_pipeline.py`).

Usage:
    python run_value_determination.py
    python run_value_determination.py --pipeline-dir ../data/generated/pipeline/preseason --output-file player_values_preseason_2026.csv
"""

import argparse
from value_determination.pipelines.valuation_pipeline import run_valuation_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Trade Valuation Pipeline (Preseason)")
    parser.add_argument("--pipeline-dir", default=None, help="Override directory for prediction CSVs")
    parser.add_argument("--output-file", default=None, help="Override output CSV filename")
    args = parser.parse_args()
    run_valuation_pipeline(
        mode="preseason",
        pipeline_dir=args.pipeline_dir,
        output_filename=args.output_file,
    )
