#!/usr/bin/env python
"""
Run the value determination pipeline.

This script is a convenience wrapper to run the value determination pipeline
from the project root directory or from auto_train_models.

Usage:
    python run_value_determination.py
"""

from value_determination.main import main

if __name__ == "__main__":
    main()
