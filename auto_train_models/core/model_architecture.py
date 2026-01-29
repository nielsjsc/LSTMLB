# Neural network model architectures

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# PASTE CELL 3 CONTENT HERE (MultiHeadAttention class)
class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Use Xavier uniform initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Set key and value to query if not provided
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, _ = query.size()
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Get attention output
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, attn_weights
        return attn_output, None
# PASTE CELL 4 CONTENT HERE (ResidualBlock and ImprovedLSTM classes)
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(self.layer_norm(x))

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
        
        # Learned padding token
        self.pad_token = nn.Parameter(torch.randn(1, 1, input_size))
        
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
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            self.hidden_size * self.directions,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size * self.directions, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, self.output_size)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Replace zero padding with learned padding token
        padding_mask = (x.sum(dim=-1) == 0).unsqueeze(-1)
        x = torch.where(padding_mask, self.pad_token.expand(batch_size, seq_len, -1), x)
        
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
            
            # Residual connection if shapes match
            if lstm_out.size(-1) == x.size(-1):
                x = x + lstm_out
            else:
                x = lstm_out
        
        # Apply attention with proper masking
        attended, _ = self.attention(
            x, x, x,
            key_padding_mask=~attention_mask
        )
        
        # Get final states using sequence lengths
        batch_indices = torch.arange(batch_size, device=x.device)
        final_states = attended[batch_indices, lengths - 1]
        
        # Project to output size
        output = self.output_projection(final_states)
        
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
        """Freeze attention layers only (for very limited data scenarios)"""
        for param in self.attention.parameters():
            param.requires_grad = False
        logger.info("Frozen attention layers")
    
    def unfreeze_attention(self):
        """Unfreeze attention layers"""
        for param in self.attention.parameters():
            param.requires_grad = True
        logger.info("Unfrozen attention layers")
    
    def freeze_attention_and_output(self):
        """Freeze attention and output projection"""
        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.output_projection.parameters():
            param.requires_grad = False
        logger.info("Frozen attention and output layers")
    
    def unfreeze_attention_and_output(self):
        """Unfreeze attention and output projection for fine-tuning"""
        for param in self.attention.parameters():
            param.requires_grad = True
        for param in self.output_projection.parameters():
            param.requires_grad = True
        logger.info("Unfrozen attention and output layers")
    
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
        # Current: [Linear(512, 512), LayerNorm(512), GELU(), Dropout(), Linear(512, old_output_size)]
        hidden_size_doubled = self.hidden_size * self.directions
        
        # Create new output projection with larger output
        new_projection = nn.Sequential(
            nn.Linear(hidden_size_doubled, hidden_size_doubled),
            nn.LayerNorm(hidden_size_doubled),
            nn.GELU(),
            nn.Dropout(self.lstm_layers[0]['dropout'].p),
            nn.Linear(hidden_size_doubled, new_output_size)
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
        
        stats = {
            'input_projection': count_params(self.input_projection),
            'lstm_layers': count_params(self.lstm_layers),
            'attention': count_params(self.attention),
            'output_projection': count_params(self.output_projection)
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
            status = "✓" if trainable == total else "⚠" if trainable > 0 else "✗"
            logger.info(f"{status} {name:20s}: {trainable:>10,} / {total:>10,} ({pct:>5.1f}%)")
        
        logger.info("=" * 60)