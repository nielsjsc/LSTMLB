"""
MLB Trade Simulator — Value Determination Engine (V2)
=====================================================

Public package exports for the V2 Monte Carlo valuation architecture.
"""

from .config import CURRENT_YEAR, Config, logger
from .contract_timeline import (
    extend_fa_timeline,
    generate_contract_timeline,
    normalize_contract_status,
)
from .data_loader import load_historical_data, load_prediction_files, merge_prediction_data
from .dollar_valuation import (
    calculate_annual_surplus,
    calculate_production_value,
    calculate_salary_cost,
    join_predictions_with_timeline,
)
from .exporter import export_value_data
from .monte_carlo_valuation import (
    add_control_horizon_metrics,
    analyze_contract_options,
    calculate_trade_values,
)
from .prospect_valuation import compute_single_prospect_valuation, generate_prospect_values
from .salary_cleaner import clean_salary_data, complete_years_of_service, merge_salary_with_ids
from .timeline_assembler import assemble_player_timeline, integrate_historical_stats

__all__ = [
    "CURRENT_YEAR",
    "Config",
    "logger",
    "load_prediction_files",
    "merge_prediction_data",
    "load_historical_data",
    "clean_salary_data",
    "merge_salary_with_ids",
    "complete_years_of_service",
    "normalize_contract_status",
    "generate_contract_timeline",
    "extend_fa_timeline",
    "calculate_production_value",
    "join_predictions_with_timeline",
    "calculate_salary_cost",
    "calculate_annual_surplus",
    "integrate_historical_stats",
    "assemble_player_timeline",
    "analyze_contract_options",
    "calculate_trade_values",
    "add_control_horizon_metrics",
    "compute_single_prospect_valuation",
    "generate_prospect_values",
    "export_value_data",
]
