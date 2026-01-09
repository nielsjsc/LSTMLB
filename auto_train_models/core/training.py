# Training configuration and functions

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import NamedTuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# PASTE CELL 5 CONTENT HERE (DataBatch, Config, PlayerDifferentiationLoss, create_data_loaders)
#Model configuration
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataBatch(NamedTuple):
    """Container for training data batches."""
    train: TensorDataset
    valid: TensorDataset
    test: TensorDataset

@dataclass
class Config:
    """Advanced configuration for LSTM-based baseball statistics prediction."""
    
    # Dynamic sizes from data
    input_size: int = None
    output_size: int = None
    
    # Model Architecture 
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    bidirectional: bool = True
    attention_dropout: float = 0.1
    residual_dropout: float = 0.2
    layer_norm_eps: float = 1e-5
    
    # Training Parameters
    batch_size: int = 16
    dropout: float = 0.3
    learning_rate: float = 1e-4  # Reduced from 1e-3 to prevent gradient explosion
    weight_decay: float = 1e-5
    gradient_clip: float = 0.5   # Reduced from 1.0 for better stability
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # Learning Rate Schedule
    lr_schedule: str = 'cosine'
    min_lr: float = 1e-6
    lr_decay_rate: float = 0.1
    lr_patience: int = 5
    
    # Early Stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Loss Function Parameters
    diversity_alpha: float = 0.1  # Weight for diversity penalty
    consistency_beta: float = 0.05  # Weight for consistency penalty
    
    # Hardware Optimization
    mixed_precision: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    
    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 1
    
    def __init__(self, X_train: torch.Tensor, y_train: torch.Tensor, 
                 hidden_size: int = None, num_layers: int = None, num_heads: int = None,
                 learning_rate: float = None, dropout: float = None, 
                 bidirectional: bool = None, gradient_clip: float = None):
        """Initialize config with data shapes and optional model-specific overrides"""
        self.input_size = X_train.shape[2]
        self.output_size = y_train.shape[1]
        
        # Apply model-specific overrides if provided
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if num_layers is not None:
            self.num_layers = num_layers
        if num_heads is not None:
            self.num_heads = num_heads
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if dropout is not None:
            self.dropout = dropout
        if bidirectional is not None:
            self.bidirectional = bidirectional
        if gradient_clip is not None:
            self.gradient_clip = gradient_clip
            
        self._validate_config()
        self._log_config()
    
    def _validate_config(self) -> None:
        assert self.hidden_size % self.num_heads == 0, \
            "Hidden size must be divisible by number of attention heads"
        assert self.hidden_size >= self.input_size, \
            "Hidden size must be greater than or equal to input size"
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1"
        assert self.num_layers >= 1, "Must have at least one LSTM layer"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.lr_schedule in ['cosine', 'linear', 'exponential'], \
            "Invalid learning rate schedule"
        assert 0 <= self.diversity_alpha <= 1, "Diversity alpha must be between 0 and 1"
        assert 0 <= self.consistency_beta <= 1, "Consistency beta must be between 0 and 1"
    
    def _log_config(self) -> None:
        logger.info("Model Configuration:")
        for key, value in asdict(self).items():
            logger.info(f"{key}: {value}")
    
    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PlayerDifferentiationLoss(nn.Module):
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
    
    @property
    def device(self) -> torch.device:
        """Get appropriate device for training."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_data_loaders(data_tuple: tuple) -> DataBatch:
    """Create DataLoader objects from preprocessed data tuple with game weights."""
    try:
        X_train, y_train, X_valid, y_valid, X_test, y_test, \
        train_masks, valid_masks, test_masks, \
        train_weights, valid_weights, test_weights = data_tuple  # Updated tuple unpacking
        
        # Create datasets with weights
        train_dataset = TensorDataset(X_train, train_masks, train_weights, y_train)
        valid_dataset = TensorDataset(X_valid, valid_masks, valid_weights, y_valid)
        test_dataset = TensorDataset(X_test, test_masks, test_weights, y_test)
        
        return DataBatch(train_dataset, valid_dataset, test_dataset)
    
    except ValueError as e:
        logger.error(f"Error unpacking data: {str(e)}")
        raise


# PASTE CELL 6 CONTENT HERE (train_model function)
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: Config,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    checkpoint_dir: str = './checkpoints',
    checkpoint_filename: str = None,
    training_mode: str = 'pretrain',
    input_features: list = None
) -> dict:
    """
    Train LSTM model with advanced optimizations and monitoring.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        valid_loader: Validation data loader
        config: Training configuration
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        checkpoint_dir: Directory to save checkpoints
        checkpoint_filename: Name of checkpoint file
        training_mode: 'pretrain' or 'finetune'
        input_features: List of input features (for checkpoint metadata)
    
    Returns:
        Dictionary of training metrics
    """
    import os
    
    logger.info(f"Starting training on device: {config.device}")
    logger.info(f"Training mode: {training_mode}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Checkpoint filename: {checkpoint_filename}")
    model = model.to(config.device)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
    
    # Training state tracking
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_metrics = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'best_epoch': 0
    }
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Handle different batch formats (with or without weights)
                    if len(batch) == 4:  # With weights: data, masks, weights, targets
                        data, masks, weights, targets = batch
                    else:  # Without weights: data, masks, targets  
                        data, masks, targets = batch
                        weights = None
                    
                    # Move data to device
                    data = data.to(config.device)
                    masks = masks.to(config.device)
                    targets = targets.to(config.device)
                    if weights is not None:
                        weights = weights.to(config.device)
                    
                    # Calculate sequence lengths from masks
                    lengths = masks.sum(1).clamp(min=1)  # Ensure minimum length of 1
                    
                    # Debug info
                    if batch_idx == 0 and epoch == 0:
                        weight_info = f", Weights: {weights.shape}" if weights is not None else ""
                        logger.info(f"Batch shapes - Data: {data.shape}, Masks: {masks.shape}, "
                                  f"Targets: {targets.shape}, Lengths: {lengths.shape}{weight_info}")
                    
                    # Check for problematic input data BEFORE forward pass
                    if torch.isnan(data).any():
                        logger.error(f"NaN in INPUT data at epoch {epoch}, batch {batch_idx}")
                        logger.error(f"  Data stats: min={data[~torch.isnan(data)].min():.4f}, max={data[~torch.isnan(data)].max():.4f}")
                        logger.error(f"  NaN count: {torch.isnan(data).sum().item()} / {data.numel()}")
                        continue  # Skip this batch
                    
                    if torch.isinf(data).any():
                        logger.error(f"Inf in INPUT data at epoch {epoch}, batch {batch_idx}")
                        logger.error(f"  Data stats: min={data[~torch.isinf(data)].min():.4f}, max={data[~torch.isinf(data)].max():.4f}")
                        logger.error(f"  Inf count: {torch.isinf(data).sum().item()} / {data.numel()}")
                        continue  # Skip this batch
                    
                    # Check for extreme values that might cause overflow
                    data_abs_max = data.abs().max().item()
                    if data_abs_max > 1e6:
                        logger.warning(f"Extreme value in INPUT data at epoch {epoch}, batch {batch_idx}")
                        logger.warning(f"  Max absolute value: {data_abs_max:.2e}")
                        logger.warning(f"  Data range: [{data.min():.2e}, {data.max():.2e}]")
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                        outputs = model(data, lengths)
                        if batch_idx == 0 and epoch == 0:
                            logger.info(f"Output shape: {outputs.shape}")
                        
                        # Check outputs for NaN/Inf BEFORE computing loss
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            logger.error(f"NaN/Inf in MODEL OUTPUT at epoch {epoch}, batch {batch_idx}")
                            logger.error(f"  Input data range: [{data.min():.4f}, {data.max():.4f}]")
                            logger.error(f"  Output NaN count: {torch.isnan(outputs).sum().item()}")
                            logger.error(f"  Output Inf count: {torch.isinf(outputs).sum().item()}")
                            logger.error(f"  Batch indices causing issue - logging first 5 samples:")
                            for i in range(min(5, data.shape[0])):
                                logger.error(f"    Sample {i}: data_range=[{data[i].min():.4f}, {data[i].max():.4f}], "
                                           f"length={lengths[i].item()}, output_has_nan={torch.isnan(outputs[i]).any().item()}")
                            
                            # CHECK MODEL WEIGHTS for NaN
                            logger.error(f"  Checking model weights for NaN...")
                            nan_params = []
                            for name, param in model.named_parameters():
                                if torch.isnan(param).any():
                                    nan_count = torch.isnan(param).sum().item()
                                    total = param.numel()
                                    nan_params.append((name, nan_count, total))
                            
                            if nan_params:
                                logger.error(f"  >>> FOUND {len(nan_params)} PARAMETERS WITH NaN VALUES:")
                                for name, nan_count, total in nan_params[:10]:  # Show first 10
                                    logger.error(f"      {name}: {nan_count}/{total} NaN values ({100*nan_count/total:.1f}%)")
                                logger.error(f"  >>> MODEL WEIGHTS ARE CORRUPTED - STOPPING TRAINING")
                                raise RuntimeError("Model weights corrupted with NaN values - cannot continue training")
                            else:
                                logger.error(f"  Model weights are clean (no NaN) - NaN produced during forward pass")
                            
                            continue  # Skip this batch
                        
                        # Use weighted loss if weights available and criterion supports it
                        # Check if loss function accepts 3 parameters (pred, target, weights)
                        import inspect
                        sig = inspect.signature(criterion.forward)
                        accepts_weights = len(sig.parameters) >= 3  # self not in signature, 3 = pred, target, weights
                        
                        if weights is not None and accepts_weights:
                            loss = criterion(outputs, targets, weights)
                        else:
                            loss = criterion(outputs, targets)
                    
                    # Backward pass with gradient scaling
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping with monitoring
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    
                    # Log gradient norm if it's getting too large
                    if grad_norm > 10.0:  # Threshold for gradient explosion warning
                        logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")
                    
                    # Optimizer step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    
                    # CHECK FOR NaN IN WEIGHTS IMMEDIATELY AFTER UPDATE
                    # This catches the exact batch that corrupts the model
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            logger.error(f">>> WEIGHT CORRUPTION DETECTED AFTER BATCH {batch_idx} <<<")
                            logger.error(f"   Parameter: {name}")
                            logger.error(f"   NaN count: {torch.isnan(param).sum().item()}")
                            logger.error(f"   Inf count: {torch.isinf(param).sum().item()}")
                            logger.error(f"   Loss: {loss.item()}")
                            logger.error(f"   Gradient norm: {grad_norm:.4f}")
                            logger.error(f"   Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                            logger.error(f"   Input data range: [{data.min():.4f}, {data.max():.4f}]")
                            raise RuntimeError(f"Model weight corruption in {name} at batch {batch_idx}")
                    
                    # Update metrics
                    loss_value = loss.item()
                    
                    # Check for NaN loss - skip batch instead of crashing
                    if torch.isnan(loss) or torch.isinf(loss) or not torch.isfinite(loss):
                        logger.warning(f"NaN or infinite loss detected at epoch {epoch}, batch {batch_idx} - SKIPPING BATCH")
                        logger.warning(f"Loss value: {loss_value}")
                        logger.warning(f"Gradient norm: {grad_norm:.4f}")
                        logger.warning(f"Learning rate: {current_lr:.2e}")
                        # Skip this batch and continue training
                        continue
                    
                    epoch_loss += loss_value
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss_value:.3f}',
                        'lr': f'{current_lr:.2e}'
                    })
                    
                except RuntimeError as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    logger.error(f"Data shapes - Input: {data.shape}, Mask: {masks.shape}, "
                               f"Target: {targets.shape}, Lengths: {lengths.shape}")
                    raise
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in valid_loader:
                try:
                    # Handle different batch formats (with or without weights)
                    if len(batch) == 4:  # With weights: data, masks, weights, targets
                        data, masks, weights, targets = batch
                    else:  # Without weights: data, masks, targets  
                        data, masks, targets = batch
                        weights = None
                    
                    data = data.to(config.device)
                    masks = masks.to(config.device)
                    targets = targets.to(config.device)
                    if weights is not None:
                        weights = weights.to(config.device)
                    
                    lengths = masks.sum(1).clamp(min=1)
                    
                    # Check validation data for NaN/Inf
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        logger.error(f"NaN/Inf in VALIDATION input data - skipping batch")
                        continue
                    
                    with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                        outputs = model(data, lengths)
                        
                        # Check validation outputs for NaN/Inf
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            logger.error(f"NaN/Inf in VALIDATION model outputs")
                            logger.error(f"  Input data range: [{data.min():.4f}, {data.max():.4f}]")
                            logger.error(f"  Output NaN count: {torch.isnan(outputs).sum().item()}")
                            logger.error(f"  Checking model weights...")
                            
                            # Check if model weights are corrupted
                            has_nan_weights = False
                            for name, param in model.named_parameters():
                                if torch.isnan(param).any():
                                    logger.error(f"    Parameter {name} has NaN values")
                                    has_nan_weights = True
                            
                            if has_nan_weights:
                                raise RuntimeError("Model weights corrupted during training - validation failed")
                            else:
                                logger.error("  Model weights are clean - NaN produced during forward pass")
                            continue
                        
                        # Use weighted loss if weights available and criterion supports it
                        import inspect
                        sig = inspect.signature(criterion.forward)
                        accepts_weights = len(sig.parameters) >= 3
                        
                        if weights is not None and accepts_weights:
                            loss = criterion(outputs, targets, weights)
                        else:
                            loss = criterion(outputs, targets)
                        
                        # Check loss for NaN
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"NaN/Inf VALIDATION loss detected - skipping batch")
                            continue
                            
                        val_loss += loss.item()
                        
                except RuntimeError as e:
                    logger.error(f"Error in validation: {str(e)}")
                    raise
        
        # Calculate epoch metrics
        epoch_loss /= len(train_loader)
        val_loss /= len(valid_loader)
        
        # Update training metrics
        train_metrics['train_losses'].append(epoch_loss)
        train_metrics['val_losses'].append(val_loss)
        train_metrics['learning_rates'].append(current_lr)
        
        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            train_metrics['best_epoch'] = epoch
            early_stopping_counter = 0
            
            # Save checkpoint
            if checkpoint_filename is None:
                checkpoint_filename = 'model.pth'  # Default fallback
                logger.warning("No checkpoint filename provided, using default 'model.pth'")
            
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            logger.info(f"Saving checkpoint to: {checkpoint_path}")
            
            # Ensure parent directory exists (for subdirectories like sp/, rp/, etc.)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            # Enhanced checkpoint with transfer learning metadata
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': asdict(config),
                'metrics': train_metrics,
                'scaler_state_dict': scaler.state_dict(),
                # Model architecture for reconstruction
                'model_config': {
                    'hidden_size': model.hidden_size,  # Store actual hidden size (no internal modifications)
                    'num_layers': model.num_layers,
                    'num_heads': model.attention.num_heads,  # Access from attention module
                    'dropout': config.dropout,  # Use config dropout value
                    'bidirectional': model.bidirectional
                },
                # Transfer learning metadata
                'training_mode': training_mode,
                'input_features': input_features if input_features else [],
                'num_features': len(input_features) if input_features else config.input_size,
                'timestamp': str(datetime.now())
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            
            logger.info(f'New best model saved with validation loss: {val_loss:.4f}')
            logger.info(f'Training mode: {training_mode}, Features: {len(input_features) if input_features else "unknown"}')
        else:
            early_stopping_counter += 1
        
        # Log epoch metrics
        logger.info(
            f'Epoch {epoch+1}: '
            f'Train Loss = {epoch_loss:.4f}, '
            f'Val Loss = {val_loss:.4f}, '
            f'LR = {current_lr:.2e}'
        )
        
        # Early stopping check
        if early_stopping_counter >= config.early_stopping_patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return train_metrics


def load_checkpoint_for_finetuning(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device,
    finetune_features: list,
    freeze_lstm: bool = True
) -> tuple:
    """
    Load pre-trained checkpoint for fine-tuning with feature expansion.
    
    Args:
        checkpoint_path: Path to pre-trained checkpoint
        model: Model instance (can be None - will be created from checkpoint)
        device: Device to load model on
        finetune_features: List of fine-tuning features (classical + statcast)
        freeze_lstm: Whether to freeze LSTM layers
    
    Returns:
        Tuple of (model, checkpoint_metadata)
    """
    logger.info(f"Loading pre-trained checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Log checkpoint info
    training_mode = checkpoint.get('training_mode', 'unknown')
    pretrain_features = checkpoint.get('input_features', [])
    num_pretrain_features = len(pretrain_features) if pretrain_features else checkpoint.get('num_features', 0)
    
    # If we still don't have the pretrain features count, infer from state_dict
    if num_pretrain_features == 0:
        state_dict = checkpoint['model_state_dict']
        # Input projection: [hidden_size_internal, input_size]
        input_proj_weight_shape = state_dict['input_projection.0.weight'].shape
        num_pretrain_features = input_proj_weight_shape[1]  # Input size is second dimension
        logger.info(f"Inferred pretrain feature count from state_dict: {num_pretrain_features}")
    
    logger.info(f"Checkpoint training mode: {training_mode}")
    logger.info(f"Checkpoint features: {num_pretrain_features}")
    logger.info(f"Fine-tuning features: {len(finetune_features)}")
    
    # Validate we're loading from pre-training
    if training_mode == 'finetune':
        logger.warning("Loading from a fine-tuned checkpoint - this may not be ideal")
    
    # Create model if not provided
    if model is None:
        from core.model_architecture import ImprovedLSTM
        
        # Get model architecture from checkpoint
        model_config = checkpoint.get('model_config', {})
        
        # If model_config not in checkpoint, infer from state_dict
        if not model_config:
            state_dict = checkpoint['model_state_dict']
            # Input projection: [hidden_size_internal, input_size]
            # Note: ImprovedLSTM halves the hidden_size internally, so we need to double it
            input_proj_weight_shape = state_dict['input_projection.0.weight'].shape
            hidden_size_internal = input_proj_weight_shape[0]  # This is already halved
            hidden_size = hidden_size_internal * 2  # Double it for the constructor
            
            # Count LSTM layers
            num_layers = sum(1 for k in state_dict.keys() if k.startswith('lstm_layers.') and '.lstm.weight_ih_l0' in k and '_reverse' not in k)
            
            # Attention heads (harder to infer, use default)
            num_heads = 4
            dropout = 0.2
            bidirectional = True
            
            logger.info(f"Inferred from checkpoint: hidden_size={hidden_size} (internal={hidden_size_internal}), num_layers={num_layers}")
        else:
            hidden_size = model_config.get('hidden_size', 256)
            num_layers = model_config.get('num_layers', 2)
            num_heads = model_config.get('num_heads', 4)
            dropout = model_config.get('dropout', 0.2)
            bidirectional = model_config.get('bidirectional', True)
        
        # Create model with pretrain input size temporarily
        # We'll expand to finetune size after loading the state dict
        model = ImprovedLSTM(
            input_size=num_pretrain_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=num_pretrain_features,  # Same as input for pretrain
            num_heads=num_heads,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(device)
        logger.info(f"Created model from checkpoint architecture: {hidden_size}h, {num_layers}L, {num_heads}heads")
        
        # Load model state dict before expansion
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            logger.info("Loaded pretrained model state dict")
        except Exception as e:
            logger.error(f"Error loading model state: {str(e)}")
            raise
        
        # Now expand projections if needed
        # Input: expand from pretrain features to finetune features (13 → 18)
        if len(finetune_features) > num_pretrain_features:
            logger.info(f"Expanding input projection from {num_pretrain_features} to {len(finetune_features)} features")
            model.expand_input_projection(new_input_size=len(finetune_features))
        
        # Output: expand to match input features for sliding window predictions (13 → 18)
        # We want to predict all features including Statcast for autoregressive forecasting
        if model.output_size < len(finetune_features):
            logger.info(f"Expanding output projection from {model.output_size} to {len(finetune_features)} features")
            model.expand_output_projection(new_output_size=len(finetune_features))
    else:
        # If model is provided, just load the state dict (for compatibility)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("Loaded model state dict to provided model (non-strict)")
        except Exception as e:
            logger.error(f"Error loading model state: {str(e)}")
            raise
    
    # Apply layer freezing if requested
    if freeze_lstm:
        model.freeze_lstm_layers()
        model.unfreeze_attention_and_output()
        logger.info("Applied layer freezing for fine-tuning")
    
    # Print trainable parameters
    model.print_trainable_params()
    
    metadata = {
        'pretrain_epoch': checkpoint.get('epoch', 0),
        'pretrain_val_loss': checkpoint.get('val_loss', 0),
        'pretrain_features': pretrain_features,
        'training_mode': training_mode
    }
    
    return model, metadata

