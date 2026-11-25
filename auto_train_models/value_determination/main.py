#!/usr/bin/env python
"""
Entry-point shim forwarding to `value_determination.pipelines.valuation_pipeline`.
"""

from value_determination.pipelines.valuation_pipeline import (  # noqa: F401
    _export_fielding_projections,
    main,
    run_valuation_pipeline,
    validate_input_data,
)
from value_determination.war_engine import (  # noqa: F401
    calculate_pitcher_war_dataframe as calculate_pitcher_war_for_dataframe,
)

if __name__ == "__main__":
    main()
