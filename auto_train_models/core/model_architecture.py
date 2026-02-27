# Neural network model architectures

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImprovedLSTM(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = None,
        dropout: float = 0.15,
        bidirectional: bool = True,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size or input_size
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # Learned padding token - initialize to zeros for stability
        self.pad_token = nn.Parameter(torch.zeros(1, 1, input_size))
        
        # Input projection with Layer Normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        # LSTM layers with residual connections and layer normalization
        self.lstm_layers = nn.ModuleList([
            nn.ModuleDict({
                'lstm': nn.LSTM(
                    self.hidden_size * self.directions if i > 0 else self.hidden_size,
                    self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional
                ),
                'norm': nn.LayerNorm(self.hidden_size * self.directions),
                'dropout': nn.Dropout(dropout)
            }) for i in range(self.num_layers)
        ])
        

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size * self.directions, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, self.output_size)
        )
        
        # Learnable output scale for smooth tanh bounding (replaces hard clamp)
        # Initialized to 2.0 so tanh(x)*scale covers roughly [-2, 2] at init
        self.output_scale = nn.Parameter(torch.tensor(2.0))
        
        # Initialize weights for numerical stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization for stability"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Use Xavier initialization for linear layers
                nn.init.xavier_uniform_(param, gain=0.5)  # Reduced gain for stability
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            # LSTM weights are already initialized by PyTorch

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # =====================================================================
        # ANCHOR: Save the last real timestep's raw features for residual skip.
        # This makes the model predict DELTAS (changes) rather than absolute values.
        # Why: MSE-trained absolute predictions regress toward the population mean,
        # which compounds during autoregressive multi-year projections. With a skip
        # connection, the default prediction (delta=0) means "stay the same" instead
        # of "decline toward league average."
        # =====================================================================
        batch_indices = torch.arange(batch_size, device=x.device)
        last_real_input = x[batch_indices, lengths - 1]  # (batch, input_size)
        
        # Replace zero padding with learned padding token (clipped for stability)
        # Clip pad_token to prevent extreme values that cause NaN
        pad_token_clipped = torch.clamp(self.pad_token, -2.0, 2.0)
        padding_mask = (x.sum(dim=-1) == 0).unsqueeze(-1)
        x = torch.where(padding_mask, pad_token_clipped.expand(batch_size, seq_len, -1), x)
        
        # Safety check: ensure no NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"NaN or Inf detected in input after padding replacement")
        
        # Create attention mask
        attention_mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        
        # Input projection
        x = self.input_projection(x)
        
        # Process LSTM layers with residual connections
        for layer in self.lstm_layers:
            # Pack padded sequence
            packed_x = pack_padded_sequence(
                x, lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            # LSTM forward pass
            lstm_out, _ = layer['lstm'](packed_x)
            lstm_out, _ = pad_packed_sequence(
                lstm_out,
                batch_first=True,
                total_length=seq_len
            )
            
            # Apply normalization and dropout
            lstm_out = layer['norm'](lstm_out)
            lstm_out = layer['dropout'](lstm_out)
            
            # Check for NaN after LSTM layer
            if torch.isnan(lstm_out).any() or torch.isinf(lstm_out).any():
                raise ValueError(f"NaN or Inf detected in LSTM output")
            
            # Residual connection if shapes match
            if lstm_out.size(-1) == x.size(-1):
                x = x + lstm_out
            else:
                x = lstm_out
        
        # Mean pooling over valid (non-padded) timesteps
        # Each season is independently meaningful, so equal weighting is appropriate
        # attention_mask: (batch, seq_len) boolean, True for valid positions
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        
        # Sum valid timesteps and divide by count
        pooled = (x * mask_expanded).sum(dim=1)  # (batch, hidden)
        lengths_clamped = lengths.clamp(min=1).unsqueeze(-1).float()  # (batch, 1)
        pooled = pooled / lengths_clamped
        
        # Check for NaN in pooled representation
        if torch.isnan(pooled).any() or torch.isinf(pooled).any():
            raise ValueError(f"NaN or Inf detected in mean-pooled output")
        
        # Project to output size (this learns the DELTA from last season)
        delta = self.output_projection(pooled)
        
        # Smooth bounding on the delta via tanh with learnable scale
        # tanh naturally bounds to [-1, 1], scale expands range smoothly
        # This limits how much any single year-to-year change can be
        delta = torch.tanh(delta) * self.output_scale
        
        # Residual skip: output = last_real_input + bounded_delta
        # The model predicts changes, not absolute values. This prevents
        # autoregressive mean-regression during multi-year projections.
        if self.input_size == self.output_size:
            output = last_real_input + delta
        else:
            # During transfer learning, output may be larger than input.
            # Skip-connect the features that exist, predict the rest as absolute.
            output = delta.clone()
            output[:, :self.input_size] = last_real_input + delta[:, :self.input_size]
        
        # Final check before returning
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError(f"NaN or Inf detected in model output")
        
        return output
    
    # Transfer Learning Methods
    def freeze_lstm_layers(self):
        """Freeze LSTM layers for fine-tuning - preserves learned temporal patterns"""
        for layer in self.lstm_layers:
            for param in layer.parameters():
                param.requires_grad = False
        logger.info(f"Frozen {self.num_layers} LSTM layers")
    
    def unfreeze_lstm_layers(self):
        """Unfreeze LSTM layers"""
        for layer in self.lstm_layers:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"Unfrozen {self.num_layers} LSTM layers")
    
    def freeze_attention(self):
        """No-op: attention was removed. Kept for backward compatibility."""
        logger.info("freeze_attention called (no-op, attention removed)")
    
    def unfreeze_attention(self):
        """No-op: attention was removed. Kept for backward compatibility."""
        logger.info("unfreeze_attention called (no-op, attention removed)")
    
    def freeze_attention_and_output(self):
        """Freeze output projection (attention was removed)"""
        for param in self.output_projection.parameters():
            param.requires_grad = False
        self.output_scale.requires_grad = False
        logger.info("Frozen output layers")
    
    def unfreeze_attention_and_output(self):
        """Unfreeze output projection for fine-tuning (attention was removed)"""
        for param in self.output_projection.parameters():
            param.requires_grad = True
        self.output_scale.requires_grad = True
        logger.info("Unfrozen output layers")
    
    def unfreeze_output_only(self):
        """Unfreeze only output projection - for very limited data scenarios"""
        for param in self.output_projection.parameters():
            param.requires_grad = True
        logger.info("Unfrozen output projection only")
    
    def expand_input_projection(self, new_input_size: int):
        """
        Expand input projection layer to accept additional features (e.g., Statcast).
        Preserves weights for existing features, initializes new feature weights randomly.
        
        Args:
            new_input_size: New input dimension (classical + statcast features)
        """
        old_input_size = self.input_size
        
        if new_input_size <= old_input_size:
            logger.info(f"Input size {new_input_size} <= current size {old_input_size}, no expansion needed")
            return
        
        logger.info(f"Expanding input projection: {old_input_size} → {new_input_size} features")
        
        # Create new input projection with larger input
        new_projection = nn.Sequential(
            nn.Linear(new_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        # Copy weights for existing features (classical stats)
        with torch.no_grad():
            # Copy weights for old features
            new_projection[0].weight[:, :old_input_size] = self.input_projection[0].weight
            new_projection[0].bias[:] = self.input_projection[0].bias
            
            # Initialize new feature weights (Statcast) with Xavier
            nn.init.xavier_uniform_(new_projection[0].weight[:, old_input_size:])
            
            # Copy layer norm parameters
            new_projection[1].weight[:] = self.input_projection[1].weight
            new_projection[1].bias[:] = self.input_projection[1].bias
        
        # Replace input projection
        self.input_projection = new_projection
        self.input_size = new_input_size
        
        # Update pad token size
        old_pad_token = self.pad_token.data
        new_pad_token = torch.zeros(1, 1, new_input_size, device=old_pad_token.device)
        new_pad_token[:, :, :old_input_size] = old_pad_token
        self.pad_token = nn.Parameter(new_pad_token)
        
        logger.info(f"Input projection expanded. New features initialized randomly.")
    
    def expand_output_projection(self, new_output_size: int):
        """
        Expand output projection to predict more features (e.g., from 13 to 18).
        Preserves weights for existing features, initializes new features randomly.
        """
        old_output_size = self.output_size
        
        if new_output_size == old_output_size:
            logger.info(f"Output size already {new_output_size}, no expansion needed")
            return
        
        if new_output_size < old_output_size:
            raise ValueError(f"Cannot shrink output from {old_output_size} to {new_output_size}")
        
        logger.info(f"Expanding output projection: {old_output_size} → {new_output_size} features")
        
        # Get current output projection structure
        # Current: [Linear(lstm_out, H*2), LayerNorm(H*2), GELU(), Dropout(), Linear(H*2, old_output_size)]
        lstm_out_size = self.hidden_size * self.directions
        intermediate_size = self.hidden_size * 2
        
        # Create new output projection with larger output
        new_projection = nn.Sequential(
            nn.Linear(lstm_out_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.GELU(),
            nn.Dropout(self.lstm_layers[0]['dropout'].p),
            nn.Linear(intermediate_size, new_output_size)
        )
        
        # Copy weights for existing layers
        with torch.no_grad():
            # Copy first linear layer (no size change)
            new_projection[0].weight[:] = self.output_projection[0].weight
            new_projection[0].bias[:] = self.output_projection[0].bias
            
            # Copy layer norm
            new_projection[1].weight[:] = self.output_projection[1].weight
            new_projection[1].bias[:] = self.output_projection[1].bias
            
            # Final linear layer: expand output dimension
            # Copy weights for existing outputs
            new_projection[4].weight[:old_output_size, :] = self.output_projection[4].weight
            new_projection[4].bias[:old_output_size] = self.output_projection[4].bias
            
            # Initialize new output weights (for Statcast features) with Xavier
            nn.init.xavier_uniform_(new_projection[4].weight[old_output_size:, :])
            nn.init.zeros_(new_projection[4].bias[old_output_size:])
        
        # Replace output projection
        self.output_projection = new_projection
        self.output_size = new_output_size
        
        logger.info(f"Output projection expanded. New outputs initialized randomly.")
    
    def get_trainable_params_count(self) -> dict:
        """Get count of trainable vs frozen parameters by component"""
        def count_params(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable
        
        # Count output_scale as part of output projection stats
        output_scale_total = self.output_scale.numel()
        output_scale_trainable = self.output_scale.numel() if self.output_scale.requires_grad else 0
        output_proj_stats = count_params(self.output_projection)
        
        stats = {
            'input_projection': count_params(self.input_projection),
            'lstm_layers': count_params(self.lstm_layers),
            'output_projection': (output_proj_stats[0] + output_scale_total, 
                                  output_proj_stats[1] + output_scale_trainable)
        }
        
        total_all = sum(s[0] for s in stats.values())
        total_trainable = sum(s[1] for s in stats.values())
        
        stats['total'] = (total_all, total_trainable)
        
        return stats
    
    def print_trainable_params(self):
        """Print detailed breakdown of trainable parameters"""
        stats = self.get_trainable_params_count()
        
        logger.info("=" * 60)
        logger.info("TRAINABLE PARAMETERS")
        logger.info("=" * 60)
        
        for name, (total, trainable) in stats.items():
            pct = (trainable / total * 100) if total > 0 else 0
            frozen = total - trainable
            status = "OK" if trainable == total else "PARTIAL" if trainable > 0 else "FROZEN"
            logger.info(f"{status} {name:20s}: {trainable:>10,} / {total:>10,} ({pct:>5.1f}%)")
        
        logger.info("=" * 60)