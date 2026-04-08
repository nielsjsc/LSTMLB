#!/usr/bin/env python
"""
Backward-compatibility shim.

The production pipeline has moved to ``value_determination.pipelines.current``.
This module re-exports all public names so existing callers
(``run_value_determination.py``, ``python -m value_determination.main``) continue
to work.
"""

from value_determination.pipelines.current import (          # noqa: F401
    main,
    calculate_pitcher_war_for_dataframe,
    validate_input_data,
    _export_fielding_projections,
)

if __name__ == "__main__":
    main()

