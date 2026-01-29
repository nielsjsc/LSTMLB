# Main training script

import argparse
import sys
import os
import numpy as np

# Add the auto_train_models directory to the path (parent of scripts/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.utils import setup_logging, set_random_seeds, get_device
from core.data_processing import preprocess_data
from core.training import create_data_loaders, Config, train_model, load_checkpoint_for_finetuning
from core.losses import WeightedPlayerDifferentiationLoss, PlayerDifferentiationLoss, InningsWeightedLoss, WeightedMSELoss, IPWeightedMSELoss

# Empirical losses - data-driven survivorship-bias-corrected aging parameters
from core.empirical_losses import (
    EmpiricalBaseballLoss,
    create_loss as create_empirical_loss,
    AgingConstraintConfig
)

from models.model_registry import ModelFactory
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Train LSTM baseball prediction models with transfer learning support')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['baserunning', 'defense_infield', 'defense_outfield', 'defense_catcher', 
                               'pitcher_sp', 'pitcher_rp', 'batter'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    # Transfer learning arguments
    parser.add_argument('--pretrain', action='store_true',
                       help='Pre-training mode (2000-2024, classical features only)')
    parser.add_argument('--finetune', action='store_true',
                       help='Fine-tuning mode (2015+, classical + Statcast features)')
    parser.add_argument('--from-checkpoint', type=str, default=None,
                       help='Path to pre-trained checkpoint for fine-tuning')
    parser.add_argument('--freeze-lstm', action='store_true', default=False,
                       help='Freeze LSTM layers during fine-tuning (recommended)')
    parser.add_argument('--freeze-attention', action='store_true', default=False,
                       help='Also freeze attention layers (recommended for very limited data)')
    
    # ==========================================================================
    # LOSS FUNCTION SELECTION
    # ==========================================================================
    # NEW: Empirical aging loss (RECOMMENDED) - uses data-derived aging parameters
    parser.add_argument('--empirical-loss', action='store_true',
                       help='[RECOMMENDED] Use empirical aging loss (data-derived parameters from aging_parameters.json)')
    parser.add_argument('--aging-weight', type=float, default=0.10,
                       help='Weight for aging constraint in empirical loss (default: 0.10)')
    parser.add_argument('--aging-tolerance', type=float, default=1.5,
                       help='Std tolerance before penalizing improvement (default: 1.5)')
    parser.add_argument('--empirical-strength', type=str, default='moderate',
                       choices=['none', 'light', 'moderate', 'strong', 'aggressive'],
                       help='Preset strength for empirical loss. Use strong/aggressive for fielding (default: moderate)')
    
    args = parser.parse_args()
    
    # Validate transfer learning arguments
    if args.finetune and not args.from_checkpoint:
        parser.error("--finetune requires --from-checkpoint")
    if args.pretrain and args.finetune:
        parser.error("Cannot use both --pretrain and --finetune")
    if args.from_checkpoint and not args.finetune:
        parser.error("--from-checkpoint requires --finetune")
    if args.freeze_attention and not args.freeze_lstm:
        parser.error("--freeze-attention requires --freeze-lstm")
    
    # Train single model
    return train_single_model(args)


def train_single_model(args):
    """Train a single model (original logic)"""
    
    # Get configuration first to check if model supports two-stage training
    config_class = ModelFactory.get_config(args.model)
    
    # Determine training mode
    if args.finetune:
        training_mode = 'finetune'
    elif args.pretrain:
        training_mode = 'pretrain'
    else:
        # Check if model supports two-stage training (has PRETRAIN_CHECKPOINT_FILE)
        if hasattr(config_class, 'PRETRAIN_CHECKPOINT_FILE'):
            # Default to pretrain for models with two-stage support
            training_mode = 'pretrain'
        else:
            # For models without two-stage training (baserunning, defense), use 'train' mode
            training_mode = 'train'
    
    # Setup
    logger = setup_logging()
    set_random_seeds()
    device = get_device()
    
    # Get data configuration
    if hasattr(config_class, 'get_data_config'):
        # Use mode-specific data config for models with transfer learning support
        # Check if get_data_config accepts a 'mode' parameter
        import inspect
        sig = inspect.signature(config_class.get_data_config)
        if 'mode' in sig.parameters:
            data_config = config_class.get_data_config(mode=training_mode)
        else:
            data_config = config_class.get_data_config()
    else:
        data_config = config_class.DATA_CONFIG
    
    # Get appropriate data file
    if training_mode == 'finetune' and hasattr(config_class, 'FINETUNE_DATA_FILE'):
        data_file = config_class.FINETUNE_DATA_FILE
    elif training_mode == 'pretrain' and hasattr(config_class, 'PRETRAIN_DATA_FILE'):
        data_file = config_class.PRETRAIN_DATA_FILE
    else:
        data_file = ModelFactory.get_data_file(args.model)
    
    # Get scaler path for fine-tuning
    pretrain_scaler_path = None
    if training_mode == 'finetune' and hasattr(config_class, 'PRETRAIN_SCALER_FILE'):
        import os
        models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pretrain_scaler_path = os.path.join(models_dir, 'data', f'{args.model}_scaler.pkl')
    
    logger.info(f"Training {args.model} model in {training_mode} mode")
    logger.info(f"Device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Features: {len(data_config.input_features)}")
    
    if training_mode == 'finetune':
        logger.info(f"Loading from checkpoint: {args.from_checkpoint}")
        logger.info(f"Freeze LSTM: {args.freeze_lstm}")
    
    logger.info(f"Checkpoint dir: {config_class.CHECKPOINT_DIR}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data = preprocess_data(
        data_file, 
        data_config, 
        args.model, 
        mode=training_mode,
        pretrain_scaler_path=pretrain_scaler_path
    )
    
    # Create data loaders
    data_batch = create_data_loaders(data)
    
    # Initialize training config - use model-specific settings from config class
    # Check if config class has model-specific hyperparameters
    if hasattr(config_class, 'HIDDEN_SIZE'):
        # Check if we're in finetune mode and have finetune-specific hyperparameters
        if training_mode == 'finetune' and hasattr(config_class, 'FINETUNE_HIDDEN_SIZE'):
            # Use finetune-specific hyperparameters
            train_config = Config(
                data_batch.train.tensors[0], 
                data_batch.train.tensors[-1],
                hidden_size=config_class.FINETUNE_HIDDEN_SIZE,
                num_layers=config_class.FINETUNE_NUM_LAYERS,
                num_heads=config_class.FINETUNE_NUM_HEADS,
                learning_rate=config_class.FINETUNE_LEARNING_RATE,
                dropout=config_class.FINETUNE_DROPOUT,
                bidirectional=config_class.BIDIRECTIONAL,
                gradient_clip=getattr(config_class, 'GRADIENT_CLIP', 1.0),
                batch_size=config_class.FINETUNE_BATCH_SIZE
            )
            logger.info(f"Using finetune-specific hyperparameters from {args.model} config class")
        else:
            # Use pretrain/default hyperparameters
            train_config = Config(
                data_batch.train.tensors[0], 
                data_batch.train.tensors[-1],
                hidden_size=config_class.HIDDEN_SIZE,
                num_layers=config_class.NUM_LAYERS,
                num_heads=config_class.NUM_HEADS,
                learning_rate=config_class.LEARNING_RATE,
                dropout=config_class.DROPOUT,
                bidirectional=config_class.BIDIRECTIONAL,
                gradient_clip=getattr(config_class, 'GRADIENT_CLIP', 1.0),
                batch_size=getattr(config_class, 'BATCH_SIZE', 16)
            )
            logger.info(f"Using model-specific hyperparameters from {args.model} config class")
    else:
        # Use generic Config defaults
        train_config = Config(data_batch.train.tensors[0], data_batch.train.tensors[-1])
        logger.info(f"Using default Config hyperparameters for {args.model}")
        
    if args.epochs:
        train_config.num_epochs = args.epochs
    
    # Create data loaders
    train_loader = DataLoader(
        data_batch.train,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )
    
    valid_loader = DataLoader(
        data_batch.valid,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )
    
    # Create model (or load from checkpoint for fine-tuning)
    if training_mode == 'finetune' and args.from_checkpoint:
        logger.info(f"Loading pre-trained checkpoint: {args.from_checkpoint}")
        model, checkpoint_metadata = load_checkpoint_for_finetuning(
            args.from_checkpoint,
            None,  # We'll create the model after we know the input size
            device,
            data_config.input_features,
            freeze_lstm=args.freeze_lstm
        )
        logger.info(f"Checkpoint metadata: {checkpoint_metadata}")
        
        # Apply additional attention freezing if requested
        if args.freeze_attention:
            model.freeze_attention()
            logger.info("Additionally frozen attention layers (--freeze-attention)")
            logger.info("Only input_projection and output_projection are trainable")
        
        # Verify checkpoint is from pre-training
        if checkpoint_metadata.get('training_mode') != 'pretrain':
            logger.warning(f"Checkpoint training_mode is '{checkpoint_metadata.get('training_mode')}', expected 'pretrain'")
        
        # Log parameter counts
        param_stats = model.get_trainable_params_count()
        total, trainable = param_stats['total']
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        model.print_trainable_params()
    else:
        # Create fresh model (for pre-training or non-transfer models)
        model = ModelFactory.create_model(
            args.model, 
            train_config.input_size, 
            train_config.output_size, 
            device
        )
        logger.info(f"Created new {args.model} model with {sum(p.numel() for p in model.parameters()):,} parameters")

    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.learning_rate,
        epochs=train_config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=min(train_config.warmup_epochs / train_config.num_epochs, 0.3),  # Cap at 30%
        anneal_strategy='cos',
        final_div_factor=1e4  # More aggressive final lr reduction
    )
    
    # =========================================================================
    # LOSS FUNCTION SELECTION
    # =========================================================================
    # Get output features for loss functions
    output_features = data_config.output_features if hasattr(data_config, 'output_features') else data_config.input_features
    
    # Map model types to categories for empirical loss
    # These must match the keys in aging_parameters.json
    CATEGORY_MAP = {
        'batter': 'batter',
        'pitcher_sp': 'pitcher',
        'pitcher_rp': 'pitcher',
        'baserunning': 'baserunning',
        'defense_infield': 'fielding_infield',
        'defense_outfield': 'fielding_outfield',
        'defense_catcher': 'fielding_catcher',
    }
    
    if args.empirical_loss:
        # =========================================================================
        # EMPIRICAL AGING LOSS (RECOMMENDED)
        # =========================================================================
        # Uses data-derived aging parameters from aging_parameters.json
        logger.info("=" * 70)
        logger.info("USING EMPIRICAL AGING LOSS (v3) - Same-Position Comparisons")
        logger.info("=" * 70)
        
        category = CATEGORY_MAP.get(args.model, 'batter')
        
        # Apply preset or custom settings
        # Note: Stronger presets recommended for fielding/baserunning
        if args.empirical_strength == 'none':
            aging_weight = 0.0
            tolerance_std = 2.0
        elif args.empirical_strength == 'light':
            aging_weight = 0.05
            tolerance_std = 2.0
        elif args.empirical_strength == 'moderate':
            aging_weight = 0.15
            tolerance_std = 1.5
        elif args.empirical_strength == 'strong':
            aging_weight = 0.30
            tolerance_std = 1.0
        elif args.empirical_strength == 'aggressive':
            aging_weight = 0.50
            tolerance_std = 0.5
        else:
            aging_weight = args.aging_weight
            tolerance_std = args.aging_tolerance
        
        # Command-line overrides take precedence
        if args.aging_weight != 0.10:  # User explicitly set it
            aging_weight = args.aging_weight
        if args.aging_tolerance != 1.5:  # User explicitly set it
            tolerance_std = args.aging_tolerance
        
        criterion = EmpiricalBaseballLoss(
            feature_names=output_features,
            category=category,
            aging_weight=aging_weight,
            tolerance_std=tolerance_std
        ).to(device)
        
        logger.info(f"Category: {category}")
        logger.info(f"Aging weight: {aging_weight}")
        logger.info(f"Tolerance: {tolerance_std} std")
        logger.info(f"Features: {output_features[:5]}{'...' if len(output_features) > 5 else ''}")
        logger.info("=" * 70)
        
    elif args.model == 'batter':
        # =========================================================================
        # DEFAULT BATTER LOSS (no aging constraint)
        # =========================================================================
        rate_stats_indices = [i for i, feat in enumerate(output_features) 
                             if not feat.endswith('_rate')]
        counting_stats_indices = [i for i, feat in enumerate(output_features) 
                                if feat.endswith('_rate')]
        
        criterion = WeightedPlayerDifferentiationLoss(
            rate_stats_indices=rate_stats_indices,
            counting_stats_indices=counting_stats_indices
        ).to(device)
        logger.info(f"Using WeightedPlayerDifferentiationLoss for batter model")
        logger.info(f"Rate stats indices: {rate_stats_indices}")
        logger.info(f"Counting stats indices: {counting_stats_indices}")
    elif args.model.startswith('defense'):
        # For defense models, use InningsWeightedLoss with position-specific weights
        criterion = InningsWeightedLoss(feature_weights=config_class.FEATURE_WEIGHTS).to(device)
        logger.info(f"Using InningsWeightedLoss for {args.model} model")
        logger.info(f"Feature weights: {config_class.FEATURE_WEIGHTS}")
    elif args.model.startswith('pitcher'):
        # For pitcher models, use IPWeightedMSELoss (weights samples by innings pitched)
        # This addresses the sample importance issue: 200 IP pitcher is more reliable than 50 IP
        criterion = IPWeightedMSELoss(
            min_weight=0.5,   # Low-IP seasons still contribute
            max_weight=1.5    # High-IP seasons weighted up to 3x more
        ).to(device)
        logger.info(f"Using IPWeightedMSELoss for {args.model} model")
        logger.info(f"  IP weighting range: [0.5, 1.5] (sqrt-normalized)")
    else:
        # For other models, use the enhanced PlayerDifferentiationLoss
        criterion = PlayerDifferentiationLoss().to(device)
        logger.info(f"Using PlayerDifferentiationLoss for {args.model} model")
    
    # Get appropriate checkpoint filename based on training mode
    if training_mode == 'finetune' and hasattr(config_class, 'FINETUNE_CHECKPOINT_FILE'):
        checkpoint_filename = config_class.FINETUNE_CHECKPOINT_FILE
        logger.info(f"Will save finetuned model to: {checkpoint_filename}")
    elif training_mode == 'pretrain' and hasattr(config_class, 'PRETRAIN_CHECKPOINT_FILE'):
        checkpoint_filename = config_class.PRETRAIN_CHECKPOINT_FILE
        logger.info(f"Will save pretrained model to: {checkpoint_filename}")
    else:
        checkpoint_filename = config_class.CHECKPOINT_FILE
        logger.info(f"Will save model to: {checkpoint_filename}")
    
    # Train model
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=train_config,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        checkpoint_dir=config_class.CHECKPOINT_DIR,
        checkpoint_filename=checkpoint_filename,
        training_mode=training_mode,
        input_features=data_config.input_features
    )
    
    logger.info("Training completed!")
    logger.info(f"Best epoch: {metrics['best_epoch']}")
    logger.info(f"Best validation loss: {min(metrics['val_losses']):.4f}")
    
    return metrics

if __name__ == "__main__":
    main()
