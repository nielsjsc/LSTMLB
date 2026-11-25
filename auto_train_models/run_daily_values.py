#!/usr/bin/env python
"""
CLI wrapper for Daily ROS Value Determination (`valuation_pipeline.py`).

Blends pre-season Marcel projections with current-season actuals, prorates
remaining-season playing time and salary, runs the V2 Monte Carlo valuation
engine, and updates `player_values_complete.csv` + daily snapshots.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from value_determination.pipelines.valuation_pipeline import run_valuation_pipeline


def main():
    return run_valuation_pipeline(mode="ros")


if __name__ == "__main__":
    main()
