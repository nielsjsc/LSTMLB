"""
Core modules for MLB Marcel projection engine.

Provides data processing utilities and Marcel projection logic
for batter, pitcher, fielding, and baserunning projections.
"""

from .data_processing import calculate_rate_stats, generate_batter_names, generate_pitcher_names
from .marcel_projections import (
    marcel_batter_projections,
    marcel_pitcher_projections,
    marcel_fielding_projections,
    marcel_baserunning_projections,
)

__all__ = [
    'calculate_rate_stats',
    'generate_batter_names',
    'generate_pitcher_names',
    'marcel_batter_projections',
    'marcel_pitcher_projections',
    'marcel_fielding_projections',
    'marcel_baserunning_projections',
]
