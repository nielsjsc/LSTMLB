# Custom loss functions for baseball prediction models

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class WeightedPlayerDifferentiationLoss(nn.Module):
    """
    Enhanced loss function specifically designed for batter predictions with:
    - Separate weighting for rate/counting stats
    - L1 regularization for counting stats  
    - Game-weighted importance
    - Feature-specific loss balancing
    """
    def __init__(self, rate_stats_indices: List[int], counting_stats_indices: List[int], alpha: float = 0.6):
        super().__init__()
        self.rate_stats_indices = rate_stats_indices
        self.counting_stats_indices = counting_stats_indices
        self.alpha = alpha  # Balance between rate and counting stats
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, games_weights: torch.Tensor) -> torch.Tensor:
        """
        Enhanced loss function with:
        - Separate weighting for rate/counting stats
        - L1 regularization for counting stats
        - Weighted average based on feature importance
        """
        # Use exponential moving average of games played
        weights = torch.exp(games_weights) / torch.exp(games_weights).mean()
        weights = weights[:, -1]  # Take most recent season weight
        
        # Rate stats loss (MSE works well for rates)
        rate_loss = F.mse_loss(
            pred[:, self.rate_stats_indices], 
            target[:, self.rate_stats_indices], 
            reduction='none'
        )
        
        # Counting stats loss (Combined MSE and L1)
        mse_counting = F.mse_loss(
            pred[:, self.counting_stats_indices],
            target[:, self.counting_stats_indices],
            reduction='none'
        )
        
        l1_counting = F.l1_loss(
            pred[:, self.counting_stats_indices],
            target[:, self.counting_stats_indices],
            reduction='none'
        )
        
        # Combine losses with feature-wise weighting
        rate_loss = (rate_loss * weights.unsqueeze(1)).mean()
        counting_loss = (
            0.8 * (mse_counting * weights.unsqueeze(1)).mean() +
            0.2 * (l1_counting * weights.unsqueeze(1)).mean()
        )
        
        # Combine with learned ratio
        total_loss = self.alpha * rate_loss + (1 - self.alpha) * counting_loss
        
        return total_loss

class InningsWeightedLoss(nn.Module):
    """
    Loss function specifically designed for defense predictions with:
    - Innings-based weighting
    - Feature-specific importance weighting
    - Position-specific optimization
    """
    def __init__(self, feature_weights: Dict[str, float] = None):
        """
        Initialize loss function with feature weights and innings weighting.
        
        Args:
            feature_weights: Dictionary mapping feature names to weights
            Example: {'DRS/150': 2.0, 'UZR/150': 1.5, 'RngR/150': 1.0}
        """
        super().__init__()
        self.feature_weights = feature_weights
        
    def forward(self, pred, target, innings):
        # Normalize innings (last feature in defense models)
        batch_innings = innings[:, -1]
        innings_weights = (batch_innings - batch_innings.min()) / (batch_innings.max() - batch_innings.min() + 1e-8)
        innings_weights = innings_weights + 0.5
        
        # Calculate MSE per feature
        feature_losses = F.mse_loss(pred, target, reduction='none')  # [batch_size, n_features]
        
        if self.feature_weights is not None:
            # Apply feature weights
            feature_weights = torch.ones_like(feature_losses[0])
            for idx, weight in enumerate(self.feature_weights.values()):
                feature_weights[idx] = weight
            feature_losses = feature_losses * feature_weights
        
        # Average across features for each sample
        sample_losses = feature_losses.mean(dim=1)
        
        # Weight by innings
        weighted_loss = (sample_losses * innings_weights).mean()
        
        return weighted_loss

class WeightedMSELoss(nn.Module):
    """
    Simple weighted MSE loss from notebook - numerically stable.
    Used for pitchers and other models.
    """
    def __init__(self, feature_weights=None):
        super().__init__()
        self.feature_weights = feature_weights
        
    def forward(self, pred, target):
        if self.feature_weights is None:
            return F.mse_loss(pred, target)
            
        # Calculate weighted MSE loss
        weighted_loss = torch.mean(self.feature_weights * (pred - target) ** 2)
        return weighted_loss

class PlayerDifferentiationLoss(nn.Module):
    """
    Basic loss function with diversity and consistency penalties.
    Used as fallback when game weights are not available.
    WARNING: Can cause NaN with batch statistics - use WeightedMSELoss instead.
    """
    def __init__(self, alpha: float = 0.1, beta: float = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Base MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Diversity penalty - encourage different predictions within batch
        batch_mean = predictions.mean(dim=0, keepdim=True)
        diversity_loss = -torch.mean(torch.abs(predictions - batch_mean))
        
        # Consistency penalty - predictions should be stable
        pred_std = predictions.std(dim=0).mean()
        consistency_loss = torch.abs(pred_std - targets.std(dim=0).mean())
        
        # Combine losses
        total_loss = mse_loss + self.alpha * diversity_loss + self.beta * consistency_loss
        
        return total_loss
