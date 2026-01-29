"""
Core modules for LSTM baseball prediction models.

Uses empirical_losses module with survivorship-bias-corrected aging parameters
from aging_parameters.json (25 years of MLB data, 2000-2025).

Corrected parameters account for poor performers exiting at higher rates,
providing more realistic decline expectations for projections.

Example:
    from core.empirical_losses import create_loss, EmpiricalBaseballLoss
    criterion = create_loss('batter', feature_names, strength='moderate')
    
    # For fielding, use position-specific categories:
    criterion = create_loss('fielding_infield', feature_names, strength='moderate')
"""

# =============================================================================
# Empirical losses - based on aging_parameters.json
# =============================================================================
from .empirical_losses import (
    # Main loss class
    EmpiricalBaseballLoss,
    AgingConstraintLoss,
    
    # Configuration
    AgingConstraintConfig,
    EmpiricalAgingParameters,
    
    # Factory functions (RECOMMENDED)
    create_loss,
    create_batter_loss,
    create_pitcher_loss,
    create_baserunning_loss,
    create_fielding_loss,
    create_fielding_infield_loss,
    create_fielding_outfield_loss,
    create_fielding_catcher_loss,
    
    # Utilities
    validate_projection_plausibility,
    
    # Legacy compatibility alias
    DomainAwareMSELoss,
)

__all__ = [
    # Main loss classes
    'EmpiricalBaseballLoss',
    'AgingConstraintLoss',
    
    # Configuration
    'AgingConstraintConfig',
    'EmpiricalAgingParameters',
    
    # Factory functions
    'create_loss',
    'create_batter_loss',
    'create_pitcher_loss',
    'create_baserunning_loss',
    'create_fielding_loss',
    'create_fielding_infield_loss',
    'create_fielding_outfield_loss',
    'create_fielding_catcher_loss',
    
    # Utilities
    'validate_projection_plausibility',
    
    # Legacy alias
    'DomainAwareMSELoss',
]
