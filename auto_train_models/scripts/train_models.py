# Main training script

import argparse
import sys
import os

# Add the auto_train_models directory to the path (parent of scripts/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.utils import setup_logging, set_random_seeds, get_device
from core.data_processing import preprocess_data
from core.training import create_data_loaders, Config, train_model, load_checkpoint_for_finetuning
from core.losses import WeightedPlayerDifferentiationLoss, PlayerDifferentiationLoss, InningsWeightedLoss, WeightedMSELoss
from models.model_registry import ModelFactory
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
    
    args = parser.parse_args()
    
    # Validate transfer learning arguments
    if args.finetune and not args.from_checkpoint:
        parser.error("--finetune requires --from-checkpoint")
    if args.pretrain and args.finetune:
        parser.error("Cannot use both --pretrain and --finetune")
    if args.from_checkpoint and not args.finetune:
        parser.error("--from-checkpoint requires --finetune")
    
    # Determine training mode
    if args.finetune:
        training_mode = 'finetune'
    else:
        training_mode = 'pretrain'  # Default mode
    
    # Setup
    logger = setup_logging()
    set_random_seeds()
    device = get_device()
    
    # Get configuration and data file for the model
    config_class = ModelFactory.get_config(args.model)
    
    # Get mode-specific configurations (only for batter model with transfer learning)
    if args.model == 'batter' and hasattr(config_class, 'get_data_config'):
        data_config = config_class.get_data_config(mode=training_mode)
        training_config_dict = config_class.get_training_config(mode=training_mode)
    else:
        # Non-transfer learning models use default configs
        data_config = config_class.get_data_config() if hasattr(config_class, 'get_data_config') else config_class.DATA_CONFIG
        training_config_dict = config_class.get_training_config() if hasattr(config_class, 'get_training_config') else {}
    
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
        # Model has custom hyperparameters - use them
        train_config = Config(
            data_batch.train.tensors[0], 
            data_batch.train.tensors[-1],
            hidden_size=config_class.HIDDEN_SIZE,
            num_layers=config_class.NUM_LAYERS,
            num_heads=config_class.NUM_HEADS,
            learning_rate=config_class.LEARNING_RATE,
            dropout=config_class.DROPOUT,
            bidirectional=config_class.BIDIRECTIONAL,
            gradient_clip=getattr(config_class, 'GRADIENT_CLIP', 1.0)
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
    
    # Choose appropriate loss function based on model type
    if args.model == 'batter':
        # For batter model, use weighted loss with proper indices
        # Use output features for calculating indices (these are what the model predicts)
        output_features = data_config.output_features
        
        # Only use base feature indices (no vs_career features in training)
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
        # For pitcher models, use simple WeightedMSELoss like notebook (numerically stable)
        criterion = WeightedMSELoss().to(device)
        logger.info(f"Using WeightedMSELoss for {args.model} model")
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

if __name__ == "__main__":
    main()
