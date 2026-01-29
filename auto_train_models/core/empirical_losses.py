"""
Empirical Domain-Constrained Loss Functions for Baseball Projections
=====================================================================

This module provides loss functions grounded in EMPIRICALLY DERIVED aging
parameters from historical MLB data (2000-2025). Unlike approaches that
rely on arbitrary "domain knowledge" parameters, every constraint here
comes from measured data.

KEY INSIGHT: WHY AGING CONSTRAINTS MATTER
-----------------------------------------
The core problem with predicting baseball aging is SURVIVORSHIP BIAS:

    - A 35-year-old with 5.00 ERA in 2025: will he get worse in 2026?
    - In REALITY: Yes, almost certainly
    - In TRAINING DATA: He retires, so we never see him decline
    
This means a pure MSE model trained on historical data will learn:
    "Old players who stick around tend to maintain or improve"
    
But this is selection bias, not reality. The aging constraint corrects
for this by penalizing unrealistic late-career improvements.

EMPIRICAL FOUNDATION
--------------------
All parameters come from aging_parameters.json, derived by:

1. Computing within-player year-over-year deltas (not cross-sectional)
2. Grouping by age bands: 21-25, 26-30, 31-35, 36-40
3. Calculating average annual decline for each age band

Example empirical findings (batters):
    wRC+ decline per year:
    - Ages 21-25: -0.65 (slight improvement during development)
    - Ages 26-30: +3.7  (early decline begins)
    - Ages 31-35: +5.0  (decline accelerates)  
    - Ages 36-40: +9.5  (steep late-career drop)

ARCHITECTURE
------------
We keep it simple:

    L_total = L_mse + λ_aging * L_aging

Where:
    - L_mse: Standard reconstruction loss (required for learning)
    - L_aging: Penalizes year-over-year changes that exceed expected
              decline rates by more than 1-2 standard deviations

We intentionally OMIT complex constraints like:
    - Peak timing loss (we don't know true peaks, they vary by player)
    - Hard bounds loss (scaler handles this; adds complexity)
    - Smoothness loss (MSE + aging already regularizes)

Simpler is better. More constraints = more hyperparameters = more overfitting.

USAGE
-----
    from core.empirical_losses import create_loss, EmpiricalBaseballLoss
    
    # For training (ages extracted from target tensor automatically)
    criterion = create_loss('batter', feature_names=['Age', 'wRC+', 'AVG', ...])
    loss = criterion(predictions, targets)  # Ages extracted from targets[:, 0]
    
    # For inference with explicit ages
    loss = criterion(predictions, targets, ages=player_ages)
    
    # With component breakdown
    loss, components = criterion(predictions, targets, return_components=True)

TRAINING INTEGRATION
--------------------
The loss function is designed to work seamlessly with the existing training loop:

    # In train_models.py
    criterion = create_loss('batter', feature_names, strength='moderate')
    
    # Training loop (no changes needed)
    loss = criterion(outputs, targets)  # Works with standard (pred, target) call

Author: LSTMLB Project
Version: 2.0.0 (Empirically Grounded)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AgingConstraintConfig:
    """
    Configuration for aging constraint strength.
    
    The aging_weight controls how much we penalize improvements beyond
    the expected decline. A value of 0.0 means pure MSE; higher values
    mean stricter aging enforcement.
    
    TUNING GUIDE:
    - 0.0:  Pure MSE, no aging constraint (baseline)
    - 0.05: Light constraint, mostly MSE-driven
    - 0.10: Moderate constraint (RECOMMENDED for most cases)
    - 0.20: Strong constraint, prioritizes aging plausibility
    - 0.30+: Very strict, may hurt prediction accuracy
    
    The tolerance_std controls how many standard deviations of improvement
    we allow before penalizing. Higher = more lenient.
    
    - 1.0: Allow 1 std above expected (fairly strict)
    - 1.5: Allow 1.5 std above expected (RECOMMENDED)
    - 2.0: Allow 2 std above expected (lenient, only penalize outliers)
    """
    aging_weight: float = 0.10
    tolerance_std: float = 1.5  # Allow improvements up to this many std
    
    @classmethod
    def none(cls) -> 'AgingConstraintConfig':
        """No aging constraint - pure MSE."""
        return cls(aging_weight=0.0, tolerance_std=2.0)
    
    @classmethod
    def light(cls) -> 'AgingConstraintConfig':
        """Light aging constraint."""
        return cls(aging_weight=0.05, tolerance_std=2.0)
    
    @classmethod
    def moderate(cls) -> 'AgingConstraintConfig':
        """Moderate aging constraint (RECOMMENDED for batters/pitchers)."""
        return cls(aging_weight=0.15, tolerance_std=1.5)
    
    @classmethod
    def strong(cls) -> 'AgingConstraintConfig':
        """Strong aging constraint (RECOMMENDED for fielding/baserunning)."""
        return cls(aging_weight=0.30, tolerance_std=1.0)
    
    @classmethod
    def aggressive(cls) -> 'AgingConstraintConfig':
        """
        Aggressive aging constraint for models that still show improvement.
        Use this if 'strong' still results in unrealistic late-career projections.
        """
        return cls(aging_weight=0.50, tolerance_std=0.5)


# =============================================================================
# EMPIRICAL PARAMETERS LOADER
# =============================================================================

class EmpiricalAgingParameters:
    """
    Loads and provides access to empirically derived aging parameters.
    
    Parameters are loaded from aging_parameters.json which contains:
    - decline_per_year_corrected by age band (21-25, 26-30, 31-35, 36-40, 41-45)
      with survivorship bias correction (poor performers exit at higher rates)
    - decline_per_year (uncorrected, for reference)
    - standard deviation of year-over-year changes
    - whether the stat is inverted (higher = worse, like ERA)
    
    Example structure for one stat:
        {
            "decline_by_age_band": {
                "21-25": {
                    "decline_per_year": -0.65, 
                    "decline_per_year_corrected": 3.49,
                    "survivorship_correction": 4.14,
                    "std": 27.6
                },
                "26-30": {"decline_per_year_corrected": 8.01, "std": 26.4},
                ...
            },
            "is_inverted": false
        }
    """
    
    # Default path relative to this file's location
    # Use aging_parameters.json which includes position-specific fielding categories
    # and same-position year-over-year comparisons (not mixed positions)
    DEFAULT_PATH = Path(__file__).parent.parent / "analysis" / "aging_parameters.json"
    
    # Age band boundaries
    AGE_BANDS = [
        (21, 25, "21-25"),
        (26, 30, "26-30"),
        (31, 35, "31-35"),
        (36, 40, "36-40"),
        (41, 45, "41-45"),
    ]
    
    def __init__(self, json_path: Optional[Path] = None):
        """
        Load empirical parameters from JSON file.
        
        Args:
            json_path: Path to aging_parameters.json. If None, uses default.
        """
        self.json_path = json_path or self.DEFAULT_PATH
        self._data = self._load_data()
        
    def _load_data(self) -> dict:
        """Load and validate the JSON data."""
        if not self.json_path.exists():
            logger.warning(f"Aging parameters file not found: {self.json_path}")
            return {}
            
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Loaded aging parameters from {self.json_path}")
        return data
    
    def get_decline_rate(
        self, 
        category: str, 
        stat_name: str, 
        age: int
    ) -> Tuple[float, float]:
        """
        Get expected decline rate and std for a stat at a given age.
        
        Args:
            category: 'batter', 'pitcher', 'baserunning', or 'fielding'
            stat_name: Name of the statistic (e.g., 'wRC+', 'ERA')
            age: Player's age
            
        Returns:
            Tuple of (expected_decline_per_year, std_of_change)
            Positive decline = getting worse for normal stats
            For inverted stats (ERA), positive decline = ERA going up = worse
        """
        # Find the age band
        age_band = self._get_age_band(age)
        if age_band is None:
            return 0.0, 1.0  # No constraint for ages outside 21-40
        
        # Look up the stat
        cat_data = self._data.get(category, {})
        stat_data = cat_data.get(stat_name, {})
        band_data = stat_data.get("decline_by_age_band", {}).get(age_band, {})
        
        # Use the survivorship-bias-corrected decline rate
        # Falls back to uncorrected if not available (for old JSON files)
        decline = band_data.get("decline_per_year_corrected")
        if decline is None:
            decline = band_data.get("decline_per_year", 0.0)
        
        std = band_data.get("std", 1.0)
        
        return decline, std
    
    def is_inverted(self, category: str, stat_name: str) -> bool:
        """
        Check if a stat is inverted (higher value = worse performance).
        
        Examples of inverted stats: ERA, FIP, WHIP, K% (for batters)
        """
        cat_data = self._data.get(category, {})
        stat_data = cat_data.get(stat_name, {})
        return stat_data.get("is_inverted", False)
    
    def _get_age_band(self, age: int) -> Optional[str]:
        """Convert age to age band string."""
        for min_age, max_age, band_str in self.AGE_BANDS:
            if min_age <= age <= max_age:
                return band_str
        return None
    
    def get_available_stats(self, category: str) -> List[str]:
        """Get list of stats with empirical parameters for a category."""
        return list(self._data.get(category, {}).keys())


# =============================================================================
# AGING CONSTRAINT LOSS
# =============================================================================

class AgingConstraintLoss(nn.Module):
    """
    Penalizes predictions that violate empirically-derived aging expectations.
    
    This loss compares year-over-year changes in predictions against what
    we expect from historical data (corrected for survivorship bias). 
    If a 36-year-old is predicted to improve significantly when historical 
    data shows 36-40 year olds decline by ~9-15 points of wRC+ per year, 
    we penalize that.
    
    Uses decline_per_year_corrected values which account for the fact that
    poor performers exit at higher rates (survivorship bias), providing
    more realistic decline expectations.
    
    The penalty is asymmetric:
    - No penalty for declining at or faster than expected (that's realistic)
    - Graduated penalty for improving more than expected
    
    Mathematical formulation:
        For consecutive years t and t+1 where player ages from A to A+1:
        
        actual_change = pred[t+1] - pred[t]  
        expected_change = -decline_rate  (decline is positive, change is negative)
        
        # For normal stats (higher = better):
        # Violation if improving beyond expected + tolerance
        improvement = -(actual_change - expected_change)  
        allowed_improvement = tolerance_std * historical_std
        violation = ReLU(improvement - allowed_improvement)
        
        # For inverted stats (higher = worse, like ERA):
        # Violation if getting better (ERA decreasing) beyond expected
        # Logic is flipped accordingly
    """
    
    def __init__(
        self,
        feature_names: List[str],
        category: str,
        params: Optional[EmpiricalAgingParameters] = None,
        config: Optional[AgingConstraintConfig] = None,
        age_feature_name: str = 'Age'
    ):
        """
        Initialize the aging constraint loss.
        
        Args:
            feature_names: List of feature names in order they appear in tensors
            category: 'batter', 'pitcher', 'baserunning', or 'fielding'
            params: EmpiricalAgingParameters instance. Loads default if None.
            config: AgingConstraintConfig instance. Uses moderate if None.
            age_feature_name: Name of the age feature in feature_names
        """
        super().__init__()
        
        self.feature_names = feature_names
        self.category = category
        self.params = params or EmpiricalAgingParameters()
        self.config = config or AgingConstraintConfig.moderate()
        
        # Logging control
        self._batch_count = 0
        self._log_interval = 100  # Log diagnostics every N batches
        
        # Find age index
        try:
            self.age_idx = feature_names.index(age_feature_name)
        except ValueError:
            logger.warning(f"Age feature '{age_feature_name}' not found. Aging loss disabled.")
            self.age_idx = None
        
        # Build feature index mapping for stats we have parameters for
        available_stats = set(self.params.get_available_stats(category))
        self.stat_indices = {}
        for idx, name in enumerate(feature_names):
            if name in available_stats and name != age_feature_name:
                self.stat_indices[name] = idx
        
        if not self.stat_indices:
            logger.warning(f"No matching stats found for category '{category}'. "
                          f"Available: {available_stats}, Features: {feature_names}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        ages: torch.Tensor,
        input_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute aging constraint loss.
        
        Args:
            predictions: Tensor of shape (batch, features) for single-step prediction
                        OR (batch, seq_len, features) for multi-step
                        Contains SCALED predictions (0-1 range from MinMaxScaler)
            ages: Tensor of shape (batch,) for single-step (age of prediction year)
                  OR (batch, seq_len) for multi-step or input sequence ages
                  Contains ACTUAL ages (e.g., 28, 29, 30...)
            input_sequence: Optional tensor (batch, seq_len, features) containing
                           historical data. If provided and predictions are single-step,
                           we concatenate them to get year-over-year comparisons.
                  
        Returns:
            Scalar loss tensor
            
        Note:
            Since predictions are scaled but our empirical parameters are in
            original units, we work with RELATIVE changes rather than absolute
            values. The direction of change (improvement vs decline) is preserved
            under scaling, which is what matters for this constraint.
        """
        if self.age_idx is None or not self.stat_indices:
            return torch.tensor(0.0, device=predictions.device)
        
        # Handle single-step predictions by concatenating with input sequence
        if predictions.dim() == 2:
            if input_sequence is None:
                return torch.tensor(0.0, device=predictions.device)
            
            # Concatenate: [input_sequence (batch, seq_len, features), prediction (batch, 1, features)]
            # This creates (batch, seq_len+1, features) for year-over-year comparison
            predictions_expanded = predictions.unsqueeze(1)  # (batch, 1, features)
            full_sequence = torch.cat([input_sequence, predictions_expanded], dim=1)
            
            # Extract ages from input sequence and concatenate with prediction age
            if ages.dim() == 1:
                # ages is (batch,) - these are ages for the prediction year
                # Extract ages from input sequence
                input_ages = input_sequence[:, :, self.age_idx]  # (batch, seq_len)
                # Concatenate with prediction ages
                full_ages = torch.cat([input_ages, ages.unsqueeze(1)], dim=1)  # (batch, seq_len+1)
            else:
                # ages already has sequence info
                full_ages = ages
            
            predictions = full_sequence
            ages = full_ages
        
        # predictions: (batch, seq_len, features)
        batch_size, seq_len, _ = predictions.shape
        
        if seq_len < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        total_penalty = torch.tensor(0.0, device=predictions.device)
        n_comparisons = 0
        
        # Diagnostic tracking
        max_penalty = 0.0
        max_improvement = 0.0
        total_improvements = 0
        total_excess_improvements = 0
        
        # Compare consecutive years
        for t in range(seq_len - 1):
            # Get ages for this transition
            if ages.dim() == 2:
                current_ages = ages[:, t]  # (batch,)
            else:
                current_ages = ages  # (batch,) - assume same age for all steps
            
            for stat_name, stat_idx in self.stat_indices.items():
                # Get predictions for this stat
                pred_t = predictions[:, t, stat_idx]      # (batch,)
                pred_t1 = predictions[:, t + 1, stat_idx]  # (batch,)
                
                # Compute actual change (in scaled space)
                actual_change = pred_t1 - pred_t  # (batch,)
                
                # Get expected decline for each sample based on age
                # We need to convert this to scaled space, but since we don't
                # have the scaler here, we use a relative approach:
                # - Penalize if change is in the WRONG DIRECTION
                # - Scale by how much of a violation it is
                
                is_inverted = self.params.is_inverted(self.category, stat_name)
                
                # For each sample, compute penalty
                for b in range(batch_size):
                    age = int(current_ages[b].item())
                    decline, std = self.params.get_decline_rate(
                        self.category, stat_name, age
                    )
                    
                    if std == 0:
                        std = 1.0  # Avoid division by zero
                    
                    change = actual_change[b].item()
                    
                    # Interpret the change based on whether stat is inverted
                    # For normal stats: positive change = improvement = suspicious if old
                    # For inverted stats: negative change = improvement = suspicious if old
                    
                    if is_inverted:
                        # ERA, FIP, etc.: lower is better
                        # Decline means ERA goes UP (positive decline)
                        # Improvement means ERA goes DOWN (negative change)
                        improvement = -change  # If change is -0.5, improvement is +0.5
                    else:
                        # wRC+, AVG, etc.: higher is better
                        # Decline means stat goes DOWN (positive decline) 
                        # Improvement means stat goes UP (positive change)
                        improvement = change
                    
                    # Track for diagnostics
                    if improvement > 0:
                        total_improvements += 1
                        max_improvement = max(max_improvement, improvement)
                    
                    # Expected: should decline (or at least not improve much)
                    # decline > 0 means we expect the stat to get worse
                    # 
                    # KEY INSIGHT: We need to penalize improvements MORE AGGRESSIVELY
                    # for older players. The tolerance approach wasn't working because
                    # small gradual improvements each year compound to large unrealistic
                    # improvements over time (e.g., Salvador Perez improving from -6.7 to -0.7)
                    
                    # Allowed improvement: negative decline means development is expected
                    # For older players, NO improvement should be allowed
                    allowed_improvement = max(0, -decline) if age < 26 else 0
                    tolerance = self.config.tolerance_std * std if age < 30 else std * 0.5
                    
                    # Excess improvement beyond what's allowed
                    excess_improvement = max(0, improvement - allowed_improvement - tolerance)
                    
                    if excess_improvement > 0:
                        total_excess_improvements += 1
                        
                        # Penalty scales with:
                        # 1. How much excess improvement (quadratic to penalize big jumps more)
                        # 2. Age factor (exponentially worse for older players)
                        
                        if age >= 36:
                            # Very old: aggressive penalty - NO improvement allowed
                            age_factor = 2.0 + (age - 36) * 0.5  # 2.0 at 36, 4.5 at 41
                        elif age >= 31:
                            # Decline phase: strong penalty
                            age_factor = 1.0 + (age - 31) * 0.2  # 1.0 at 31, 2.0 at 36
                        elif age >= 26:
                            # Prime years: moderate penalty
                            age_factor = 0.5 + (age - 26) * 0.1  # 0.5 at 26, 1.0 at 31
                        else:
                            # Young: minimal penalty
                            age_factor = 0.1
                        
                        # Quadratic penalty with age factor
                        penalty = (excess_improvement ** 2) * age_factor * 10.0
                        total_penalty = total_penalty + penalty
                        n_comparisons += 1
                        max_penalty = max(max_penalty, penalty)
                    
                    # ALSO penalize insufficient decline for very old players
                    # If someone is 38+ and we expect decline of +2.0 but they only decline 0.1,
                    # that's also suspicious (model predicting stable performance at old age)
                    if age >= 36 and decline > 0:
                        # For normal stats: actual_decline = -change (change negative = declining)
                        # For inverted stats: actual_decline = change (change positive = declining)
                        if is_inverted:
                            actual_decline = change
                        else:
                            actual_decline = -change
                        
                        # Expected minimum decline (at least 50% of empirical average)
                        min_expected_decline = decline * 0.5
                        
                        if actual_decline < min_expected_decline:
                            # Not declining fast enough
                            shortfall = min_expected_decline - actual_decline
                            age_factor = 1.0 + (age - 36) * 0.3  # 1.0 at 36, 2.5 at 41
                            insufficient_decline_penalty = (shortfall ** 2) * age_factor * 5.0
                            total_penalty = total_penalty + insufficient_decline_penalty
                            max_penalty = max(max_penalty, insufficient_decline_penalty)
                            n_comparisons += 1
        
        # Average penalty across all valid comparisons
        if n_comparisons > 0:
            avg_penalty = total_penalty / n_comparisons
        else:
            avg_penalty = torch.tensor(0.0, device=predictions.device)
        
        # Log diagnostics periodically (not every batch to reduce noise)
        self._batch_count += 1
        should_log = (self._batch_count % self._log_interval == 1)  # Log first batch and every Nth
        
        if should_log:
            if n_comparisons > 0:
                logger.info(f"Aging Constraint Diagnostics (batch {self._batch_count}):")
                logger.info(f"  Comparisons: {n_comparisons}, Improvements: {total_improvements}, Penalized: {total_excess_improvements}")
                logger.info(f"  Max improvement: {max_improvement:.4f}, Avg penalty: {avg_penalty.item():.6f}")
                logger.info(f"  Weighted loss contribution (weight={self.config.aging_weight}): {(avg_penalty * self.config.aging_weight).item():.6f}")
            else:
                logger.debug(f"Aging Constraint: No valid year-over-year comparisons in batch {self._batch_count}")
        
        return avg_penalty


# =============================================================================
# COMBINED LOSS FUNCTION
# =============================================================================

class EmpiricalBaseballLoss(nn.Module):
    """
    Combined loss function: MSE + empirical aging constraint.
    
    This is the main loss function for training baseball projection models.
    It combines:
    
    1. MSE Loss: Ensures predictions match historical data
    2. Aging Constraint: Penalizes unrealistic late-career improvements
    
    The aging constraint addresses survivorship bias in the training data.
    Without it, models learn that old players who continue playing tend to
    maintain performance - but this is selection bias (the ones who declined
    retired and aren't in the data).
    
    TRAINING INTEGRATION
    --------------------
    This loss is designed to work as a drop-in replacement in the training loop.
    Ages are automatically extracted from the target tensor if 'Age' is in feature_names.
    
    Usage:
        # Setup
        criterion = EmpiricalBaseballLoss(
            feature_names=['Age', 'wRC+', 'AVG', 'OBP', ...],
            category='batter',
            aging_weight=0.10
        )
        
        # Training loop - works with standard (pred, target) signature
        loss = criterion(predictions, targets)
        
        # Or with explicit ages
        loss = criterion(predictions, targets, ages=player_ages)
        
        # With component breakdown
        loss, components = criterion(predictions, targets, return_components=True)
        # components = {'mse': 0.015, 'aging': 0.003, 'total': 0.018}
    """
    
    def __init__(
        self,
        feature_names: List[str],
        category: str,
        aging_weight: float = 0.10,
        tolerance_std: float = 1.5,
        params_path: Optional[Path] = None,
        age_feature_name: str = 'Age'
    ):
        """
        Initialize the combined loss function.
        
        Args:
            feature_names: List of feature names (should include 'Age' for aging constraint)
            category: 'batter', 'pitcher', 'baserunning', or 'fielding'
            aging_weight: Weight for aging constraint (0 = pure MSE)
            tolerance_std: How many std of improvement to allow before penalty
            params_path: Path to aging_parameters.json (uses default if None)
            age_feature_name: Name of age feature in feature_names (for auto-extraction)
        """
        super().__init__()
        
        self.feature_names = feature_names
        self.category = category
        self.aging_weight = aging_weight
        self.age_feature_name = age_feature_name
        
        # Find age index for automatic extraction from targets
        try:
            self.age_idx = feature_names.index(age_feature_name)
            logger.debug(f"Age feature '{age_feature_name}' found at index {self.age_idx}")
        except ValueError:
            self.age_idx = None
            if aging_weight > 0:
                logger.warning(f"Age feature '{age_feature_name}' not in features. "
                             f"Aging constraint will be disabled unless ages are passed explicitly.")
        
        # MSE loss
        self.mse_loss = nn.MSELoss()
        
        # Aging constraint
        params = EmpiricalAgingParameters(params_path) if params_path else EmpiricalAgingParameters()
        config = AgingConstraintConfig(
            aging_weight=aging_weight,
            tolerance_std=tolerance_std
        )
        self.aging_loss = AgingConstraintLoss(
            feature_names=feature_names,
            category=category,
            params=params,
            config=config
        )
        
        self.tolerance_std = tolerance_std
        
        # Logging control
        self._batch_count = 0
        self._log_interval = 100  # Log diagnostics every N batches
        
        logger.info(f"EmpiricalBaseballLoss initialized for '{category}':")
        logger.info(f"  - Aging weight: {aging_weight}")
        logger.info(f"  - Tolerance: {tolerance_std} std")
        logger.info(f"  - Age auto-extract: {'enabled' if self.age_idx is not None else 'disabled'}")
        if aging_weight > 0:
            stats_with_params = list(self.aging_loss.stat_indices.keys())
            logger.info(f"  - Stats with aging params: {stats_with_params[:5]}{'...' if len(stats_with_params) > 5 else ''}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        input_sequence: Optional[torch.Tensor] = None,
        ages: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions, shape (batch, features) or (batch, seq, features)
            targets: Ground truth, same shape as predictions. If 'Age' is a feature,
                    ages will be extracted automatically.
            input_sequence: Optional input sequence (batch, seq_len, features) containing
                           historical data. Required for aging constraint with single-step predictions.
            ages: Player ages (optional). If None and 'Age' is in feature_names,
                  ages will be extracted from targets automatically.
                  Shape (batch,) or (batch, seq)
            return_components: If True, also return dict of individual losses
            
        Returns:
            If return_components is False: total loss tensor
            If return_components is True: (total loss, {'mse': ..., 'aging': ..., 'total': ...})
        """
        # MSE component (always computed)
        mse = self.mse_loss(predictions, targets)
        
        # Aging component
        aging = torch.tensor(0.0, device=predictions.device)
        
        if self.aging_weight > 0:
            # Auto-extract ages from targets if not provided
            if ages is None and self.age_idx is not None:
                if targets.dim() == 2:  # (batch, features)
                    ages = targets[:, self.age_idx]
                elif targets.dim() == 3:  # (batch, seq, features)
                    ages = targets[:, :, self.age_idx]
            
            if ages is not None:
                aging = self.aging_loss(predictions, ages, input_sequence=input_sequence)
        
        # Combined
        total = mse + self.aging_weight * aging
        
        if return_components:
            components = {
                'mse': mse.item(),
                'aging': aging.item() if isinstance(aging, torch.Tensor) else aging,
                'total': total.item()
            }
            return total, components
        
        return total
    
    def get_config_summary(self) -> str:
        """Return human-readable summary of the loss configuration."""
        return (
            f"EmpiricalBaseballLoss Configuration:\n"
            f"  Category: {self.category}\n"
            f"  Aging Weight: {self.aging_weight}\n"
            f"  Tolerance: {self.tolerance_std} std\n"
            f"  Features: {len(self.feature_names)}\n"
            f"  Age auto-extract: {'enabled' if self.age_idx is not None else 'disabled'}\n"
            f"  Stats with aging params: {list(self.aging_loss.stat_indices.keys())}"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_loss(
    category: str,
    feature_names: List[str],
    strength: str = 'moderate',
    **kwargs
) -> EmpiricalBaseballLoss:
    """
    Factory function to create a loss function for a model category.
    
    Args:
        category: One of 'batter', 'pitcher', 'baserunning', 'fielding'
        feature_names: List of feature names used by the model
        strength: Constraint strength - 'none', 'light', 'moderate', 'strong'
        **kwargs: Additional arguments passed to EmpiricalBaseballLoss
        
    Returns:
        Configured EmpiricalBaseballLoss instance
        
    Example:
        criterion = create_loss('batter', ['Age', 'wRC+', 'AVG', ...], 'moderate')
    """
    config_map = {
        'none': AgingConstraintConfig.none(),
        'light': AgingConstraintConfig.light(),
        'moderate': AgingConstraintConfig.moderate(),
        'strong': AgingConstraintConfig.strong(),
    }
    
    config = config_map.get(strength, AgingConstraintConfig.moderate())
    
    return EmpiricalBaseballLoss(
        feature_names=feature_names,
        category=category,
        aging_weight=config.aging_weight,
        tolerance_std=config.tolerance_std,
        **kwargs
    )


def create_batter_loss(feature_names: List[str], **kwargs) -> EmpiricalBaseballLoss:
    """Create loss for batter models."""
    return create_loss('batter', feature_names, **kwargs)


def create_pitcher_loss(feature_names: List[str], **kwargs) -> EmpiricalBaseballLoss:
    """Create loss for pitcher models (SP or RP - uses same aging params)."""
    return create_loss('pitcher', feature_names, **kwargs)


def create_baserunning_loss(feature_names: List[str], **kwargs) -> EmpiricalBaseballLoss:
    """Create loss for baserunning models."""
    return create_loss('baserunning', feature_names, **kwargs)


def create_fielding_loss(
    feature_names: List[str], 
    position_group: str = 'infield',
    **kwargs
) -> EmpiricalBaseballLoss:
    """
    Create loss for fielding models.
    
    Args:
        feature_names: List of feature names
        position_group: One of 'infield', 'outfield', 'catcher'
        **kwargs: Additional args passed to create_loss
        
    Returns:
        Configured EmpiricalBaseballLoss for the position group
    """
    category = f'fielding_{position_group}'
    return create_loss(category, feature_names, **kwargs)


def create_fielding_infield_loss(feature_names: List[str], **kwargs) -> EmpiricalBaseballLoss:
    """Create loss for infield fielding models."""
    return create_loss('fielding_infield', feature_names, **kwargs)


def create_fielding_outfield_loss(feature_names: List[str], **kwargs) -> EmpiricalBaseballLoss:
    """Create loss for outfield fielding models."""
    return create_loss('fielding_outfield', feature_names, **kwargs)


def create_fielding_catcher_loss(feature_names: List[str], **kwargs) -> EmpiricalBaseballLoss:
    """Create loss for catcher fielding models."""
    return create_loss('fielding_catcher', feature_names, **kwargs)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

class DomainAwareMSELoss(nn.Module):
    """
    Drop-in replacement for nn.MSELoss with optional aging constraints.
    
    For code that just does: loss = criterion(pred, target)
    This provides an easy migration path.
    
    Note: Without ages, this is just MSE. For full benefit, use
    EmpiricalBaseballLoss directly.
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        category: str = 'batter'
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.feature_names = feature_names
        self.category = category
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss (aging constraint not applied without ages)."""
        return self.mse(predictions, targets)


# =============================================================================
# UTILITY: PROJECTION VALIDATOR
# =============================================================================

def validate_projection_plausibility(
    projections: List[Dict[str, float]],
    category: str = 'batter',
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate a projection trajectory for plausibility.
    
    Use this AFTER generating projections to sanity-check them.
    
    Args:
        projections: List of dicts, each containing one year's projections
                    Must include 'Age' and stat values
        category: 'batter', 'pitcher', etc.
        verbose: Print validation results
        
    Returns:
        Dict with:
            - 'valid': bool
            - 'warnings': List of warning messages
            - 'severe_warnings': List of serious issues
            
    Example:
        projections = [
            {'Age': 33, 'wRC+': 120, 'AVG': 0.280},
            {'Age': 34, 'wRC+': 125, 'AVG': 0.285},  # <- Unusual improvement
            {'Age': 35, 'wRC+': 130, 'AVG': 0.290},  # <- Very suspicious
        ]
        result = validate_projection_plausibility(projections, 'batter')
    """
    params = EmpiricalAgingParameters()
    warnings = []
    severe_warnings = []
    
    if len(projections) < 2:
        return {'valid': True, 'warnings': [], 'severe_warnings': []}
    
    available_stats = set(params.get_available_stats(category))
    
    for i in range(len(projections) - 1):
        curr = projections[i]
        next_p = projections[i + 1]
        
        age = curr.get('Age', 27)
        
        for stat_name in available_stats:
            if stat_name not in curr or stat_name not in next_p:
                continue
            
            change = next_p[stat_name] - curr[stat_name]
            decline, std = params.get_decline_rate(category, stat_name, age)
            is_inverted = params.is_inverted(category, stat_name)
            
            # Calculate improvement
            if is_inverted:
                improvement = -change  # For ERA, decrease is improvement
            else:
                improvement = change   # For wRC+, increase is improvement
            
            # Flag if improving significantly in late career
            if age >= 32 and improvement > std:
                msg = f"Age {age}->{age+1}: {stat_name} improving by {improvement:.2f} (expected decline: {decline:.2f})"
                if improvement > 2 * std:
                    severe_warnings.append(msg)
                else:
                    warnings.append(msg)
    
    is_valid = len(severe_warnings) == 0
    
    if verbose and (warnings or severe_warnings):
        print("=== Projection Plausibility Check ===")
        if severe_warnings:
            print("\n⚠️  SEVERE WARNINGS:")
            for w in severe_warnings:
                print(f"   - {w}")
        if warnings:
            print("\n⚡ Warnings:")
            for w in warnings:
                print(f"   - {w}")
        print(f"\nValid: {is_valid}")
    
    return {
        'valid': is_valid,
        'warnings': warnings,
        'severe_warnings': severe_warnings
    }
