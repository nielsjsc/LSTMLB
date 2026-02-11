# Custom loss functions for baseball prediction models
#
# This module contains basic loss functions for training.
# For domain-constrained losses that encode baseball knowledge
# (aging curves, physical bounds, etc.), see:
#   - core/domain_losses.py - Main implementation
#   - core/constraint_config.py - Configuration presets
#
# Recommended usage for new models:
#   from core.domain_losses import create_loss_for_model
#   criterion = create_loss_for_model('batter', feature_names, 'medium')
#
# LOSS FUNCTION SELECTION GUIDE:
# ==============================
# - Batters: WeightedPlayerDifferentiationLoss (rate/counting stat aware)
# - Pitchers: IPWeightedMSELoss (innings-weighted for sample importance)
# - Defense: InningsWeightedLoss (defensive innings weighting)
# - Baserunning: WeightedMSELoss (simple feature weighting)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

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
        # Safety checks for input tensors
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise ValueError(f"NaN or Inf in predictions. Range: [{pred.min():.2f}, {pred.max():.2f}]")
        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError(f"NaN or Inf in targets. Range: [{target.min():.2f}, {target.max():.2f}]")
        if torch.isnan(games_weights).any() or torch.isinf(games_weights).any():
            raise ValueError(f"NaN or Inf in weights. Range: [{games_weights.min():.2f}, {games_weights.max():.2f}]")
        
        # Normalize game weights to prevent overflow
        # Take most recent season weight
        weights = games_weights[:, -1]
        
        # Normalize to [0, 1] range, then scale to [0.5, 1.5] to maintain relative importance
        weights_min = weights.min()
        weights_max = weights.max()
        if weights_max > weights_min:
            weights = (weights - weights_min) / (weights_max - weights_min)
            weights = 0.5 + weights  # Scale to [0.5, 1.5]
        else:
            weights = torch.ones_like(weights)
        
        # Rate stats loss (MSE works well for rates)
        if len(self.rate_stats_indices) > 0:
            rate_loss = F.mse_loss(
                pred[:, self.rate_stats_indices], 
                target[:, self.rate_stats_indices], 
                reduction='none'
            )
            rate_loss = (rate_loss * weights.unsqueeze(1)).mean()
        else:
            rate_loss = torch.tensor(0.0, device=pred.device)
        
        # Counting stats loss (Combined MSE and L1)
        if len(self.counting_stats_indices) > 0:
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
            
            counting_loss = (
                0.8 * (mse_counting * weights.unsqueeze(1)).mean() +
                0.2 * (l1_counting * weights.unsqueeze(1)).mean()
            )
        else:
            counting_loss = torch.tensor(0.0, device=pred.device)
        
        # Combine with learned ratio
        # If one type is empty, use the other
        if len(self.rate_stats_indices) > 0 and len(self.counting_stats_indices) > 0:
            total_loss = self.alpha * rate_loss + (1 - self.alpha) * counting_loss
        elif len(self.rate_stats_indices) > 0:
            total_loss = rate_loss
        elif len(self.counting_stats_indices) > 0:
            total_loss = counting_loss
        else:
            # Fallback to basic MSE if indices are misconfigured
            total_loss = F.mse_loss(pred, target)
        
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


class IPWeightedMSELoss(nn.Module):
    """
    Innings Pitched (IP) weighted MSE loss for pitcher models.
    
    RATIONALE:
    ==========
    Pitchers with more IP provide more reliable statistical samples:
    - 200 IP starter: Much more reliable than 50 IP starter
    - This weighting encourages the model to fit high-IP pitchers better
    - Low-IP seasons are still included but with reduced influence
    
    WEIGHTING SCHEME:
    ================
    - Raw IP values are normalized to [0.5, 1.5] range
    - Square root transformation reduces extreme differences
    - Floor of 0.5 ensures all samples contribute
    - Cap of 1.5 prevents any single sample from dominating
    
    USAGE:
    ======
    This loss expects the weights tensor to contain IP values (or proxy)
    for each sample in the batch.
    
    Example:
        criterion = IPWeightedMSELoss(ip_index=5)  # IP is 6th feature
        loss = criterion(pred, target, batch_weights)
    """
    
    def __init__(self, 
                 feature_weights: Optional[Dict[str, float]] = None,
                 ip_index: Optional[int] = None,
                 min_weight: float = 0.5,
                 max_weight: float = 1.5):
        """
        Initialize IP-weighted MSE loss.
        
        Args:
            feature_weights: Optional dict mapping feature indices to importance weights
            ip_index: Index of IP in the feature vector (for extracting IP from targets)
            min_weight: Minimum sample weight (default 0.5)
            max_weight: Maximum sample weight (default 1.5)
        """
        super().__init__()
        self.feature_weights = feature_weights
        self.ip_index = ip_index
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute IP-weighted MSE loss.
        
        Args:
            pred: Model predictions [batch_size, n_features]
            target: Ground truth targets [batch_size, n_features]
            sample_weights: Per-sample weights [batch_size] or [batch_size, seq_len]
                           If seq_len dimension exists, uses last timestep weights
        
        Returns:
            Weighted MSE loss scalar
        """
        # Calculate per-sample, per-feature MSE
        feature_losses = F.mse_loss(pred, target, reduction='none')  # [batch, features]
        
        # Apply optional feature weighting
        if self.feature_weights is not None:
            fw = torch.ones(feature_losses.shape[1], device=pred.device)
            for idx, weight in enumerate(self.feature_weights.values()):
                if idx < len(fw):
                    fw[idx] = weight
            feature_losses = feature_losses * fw
        
        # Average across features for each sample
        sample_losses = feature_losses.mean(dim=1)  # [batch]
        
        # Calculate IP weights
        if sample_weights is not None:
            # Handle different weight tensor shapes
            if sample_weights.dim() == 2:
                # Sequence weights: use last timestep (most recent season)
                ip_weights = sample_weights[:, -1]
            else:
                ip_weights = sample_weights
            
            # Normalize to [min_weight, max_weight] range
            # Use sqrt to reduce extreme differences
            ip_sqrt = torch.sqrt(ip_weights.clamp(min=1.0))
            
            # Min-max normalize
            ip_min = ip_sqrt.min()
            ip_max = ip_sqrt.max()
            if ip_max > ip_min:
                normalized = (ip_sqrt - ip_min) / (ip_max - ip_min + 1e-8)
                final_weights = self.min_weight + normalized * (self.max_weight - self.min_weight)
            else:
                final_weights = torch.ones_like(ip_sqrt)
            
            # Weighted mean
            weighted_loss = (sample_losses * final_weights).sum() / final_weights.sum()
        else:
            # Fallback to simple mean
            weighted_loss = sample_losses.mean()
        
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
