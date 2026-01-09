#!/usr/bin/env python3
"""
Prediction script for fine-tuned batter model with Statcast features.

This script loads the fine-tuned model and generates predictions for batters
using both classical stats (2000+) and Statcast data (2015+).
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.batter_config import BatterConfig
from core.training import load_checkpoint_for_finetuning
from core.prediction import predict_all_2024_batters, generate_batter_names
from core.data_processing import calculate_rate_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPTS_DIR = Path(__file__).parent  # auto_train_models/scripts/
AUTO_TRAIN_DIR = SCRIPTS_DIR.parent  # auto_train_models/
DATA_DIR = AUTO_TRAIN_DIR.parent / 'data'  # LSTMLB/data/
GENERATED_DIR = DATA_DIR / 'generated'

# Ensure directories exist
GENERATED_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_finetuned_model():
    """
    Load the fine-tuned model with Statcast features.
    
    Returns:
        model: The loaded fine-tuned model
        scaler: The feature scaler (extended for 18 features)
        data_config: The fine-tuning data configuration
    """
    logger.info("Loading fine-tuned model...")
    
    # Get fine-tuning configuration
    data_config = BatterConfig.get_data_config(mode='finetune')
    
    # Load checkpoint directly
    checkpoint_path = str(MODELS_DIR / 'checkpoints' / 'batter_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get architecture from checkpoint
    training_mode = checkpoint.get('training_mode', 'unknown')
    input_features = checkpoint.get('input_features', [])
    num_input_features = len(input_features) if input_features else checkpoint.get('num_features', 18)
    
    logger.info(f"Checkpoint training mode: {training_mode}")
    logger.info(f"Checkpoint input features: {num_input_features}")
    
    # Get model config
    model_config = checkpoint.get('model_config', {})
    if not model_config:
        # Infer from state dict
        state_dict = checkpoint['model_state_dict']
        input_proj_shape = state_dict['input_projection.0.weight'].shape
        hidden_size_internal = input_proj_shape[0]
        hidden_size = hidden_size_internal * 2  # Double for constructor
        num_layers = 2
        num_heads = 4
        dropout = 0.2
        bidirectional = True
    else:
        hidden_size = model_config.get('hidden_size', 512)
        num_layers = model_config.get('num_layers', 2)
        num_heads = model_config.get('num_heads', 4)
        dropout = model_config.get('dropout', 0.2)
        bidirectional = model_config.get('bidirectional', True)
    
    # Get output size from state dict (more reliable than config)
    state_dict = checkpoint['model_state_dict']
    output_size = state_dict['output_projection.4.weight'].shape[0]  # First dim is output_size
    
    logger.info(f"Model architecture: input={num_input_features}, output={output_size}, hidden={hidden_size}")
    
    # Create model with exact architecture from checkpoint
    from core.model_architecture import ImprovedLSTM
    model = ImprovedLSTM(
        input_size=num_input_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        num_heads=num_heads,
        dropout=dropout,
        bidirectional=bidirectional
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load the fine-tuned scaler (18 features)
    scaler_path = MODELS_DIR / 'data' / 'batter_finetune_scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded fine-tuned scaler from {scaler_path}")
    else:
        logger.warning(f"Fine-tuned scaler not found at {scaler_path}, using pretrain scaler")
        scaler = joblib.load(MODELS_DIR / 'data' / 'batter_scaler.pkl')
    
    metadata = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown'),
        'training_mode': training_mode,
        'input_features': num_input_features,
        'output_features': output_size
    }
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Input features: {len(data_config.input_features)} ({data_config.input_features[:3]}...)")
    logger.info(f"Output features: {len(data_config.output_features)} ({data_config.output_features[:3]}...)")
    logger.info(f"Checkpoint epoch: {metadata['epoch']}")
    logger.info(f"Checkpoint val loss: {metadata['val_loss']}")
    
    return model, scaler, data_config, metadata


def generate_predictions(
    model,
    scaler,
    data_config,
    start_year: int = 2025,
    future_years: int = 10,
    min_seasons: int = 2,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate predictions using the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        scaler: Feature scaler (18 features)
        data_config: Data configuration with input/output features
        start_year: First year to predict (default 2025)
        future_years: Number of years to predict (default 10)
        min_seasons: Minimum seasons required for prediction (default 2)
        output_file: Optional path to save predictions
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Generating predictions from {start_year} for {future_years} years...")
    
    # Load batting data (2000-2025 with Statcast)
    data_file = DATA_DIR / 'mlb_batting_data_2000_2025.csv'
    logger.info(f"Loading data from {data_file}")
    
    raw_df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(raw_df)} rows from {raw_df['Season'].min()} to {raw_df['Season'].max()}")
    
    # Apply rate stats calculation
    raw_df = calculate_rate_stats(raw_df)
    
    # Generate player names
    player_names = generate_batter_names(raw_df)
    logger.info(f"Found {len(player_names)} unique players")
    
    # Calculate cutoff year (the last year with actual data to use as base for predictions)
    cutoff_year = start_year - 1
    logger.info(f"Using cutoff year {cutoff_year} to predict {start_year} onwards")
    
    # Generate predictions
    predictions_df = predict_all_2024_batters(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=data_config.input_features,  # 18 features
        future_years=future_years,
        cutoff_year=cutoff_year  # Pass the cutoff year based on start_year
    )
    
    if predictions_df is not None and len(predictions_df) > 0:
        # Add metadata
        predictions_df['model_type'] = 'finetuned_statcast'
        predictions_df['input_features'] = len(data_config.input_features)
        
        # Rename Year to Season for consistency
        if 'Year' in predictions_df.columns:
            predictions_df = predictions_df.rename(columns={'Year': 'Season'})
        
        # Save predictions in auto_train_models directory
        if output_file is None:
            output_dir = Path(__file__).parent / 'predictions'
            output_dir.mkdir(exist_ok=True)
            output_file = str(output_dir / f'batter_predictions_finetuned_{start_year}.csv')
        
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(predictions_df)} predictions to {output_file}")
        
        # Summary statistics
        unique_players = predictions_df['Name'].nunique()
        seasons_predicted = predictions_df['Season'].nunique()
        logger.info(f"Generated predictions for {unique_players} players across {seasons_predicted} seasons")
        logger.info(f"Season range: {predictions_df['Season'].min()} - {predictions_df['Season'].max()}")
        
        # Show sample predictions
        logger.info("\nSample predictions (first 5 rows):")
        sample_cols = ['Name', 'Season', 'Age', 'wOBA', 'wRC+', 'HR_rate']
        if all(col in predictions_df.columns for col in sample_cols):
            print(predictions_df[sample_cols].head())
        
        # Also show Statcast features if available
        statcast_cols = ['Name', 'Season', 'EV', 'LA', 'Barrel%', 'HardHit%', 'xwOBA']
        if all(col in predictions_df.columns for col in statcast_cols):
            logger.info("\nStatcast predictions (first 5 rows):")
            print(predictions_df[statcast_cols].head())
        
        return predictions_df
    else:
        logger.error("Failed to generate predictions")
        return None


def compare_with_pretrained(
    finetuned_predictions: pd.DataFrame,
    pretrained_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare fine-tuned predictions with pre-trained only predictions.
    
    Args:
        finetuned_predictions: Predictions from fine-tuned model
        pretrained_file: Path to pre-trained predictions (optional)
        
    Returns:
        DataFrame with comparison metrics
    """
    if pretrained_file is None or not Path(pretrained_file).exists():
        logger.warning("No pre-trained predictions file provided or file doesn't exist")
        return None
    
    logger.info(f"Loading pre-trained predictions from {pretrained_file}")
    pretrained_df = pd.read_csv(pretrained_file)
    
    # Merge on player and season
    merged = finetuned_predictions.merge(
        pretrained_df,
        on=['IDfg', 'Season'],
        suffixes=('_finetuned', '_pretrained')
    )
    
    # Calculate differences for key stats
    comparison_stats = ['wOBA', 'wRC+', 'HR_rate']
    for stat in comparison_stats:
        if f'{stat}_finetuned' in merged.columns and f'{stat}_pretrained' in merged.columns:
            merged[f'{stat}_diff'] = merged[f'{stat}_finetuned'] - merged[f'{stat}_pretrained']
            merged[f'{stat}_abs_diff'] = abs(merged[f'{stat}_diff'])
    
    logger.info(f"\nComparison Summary:")
    for stat in comparison_stats:
        if f'{stat}_diff' in merged.columns:
            mean_diff = merged[f'{stat}_diff'].mean()
            mean_abs_diff = merged[f'{stat}_abs_diff'].mean()
            logger.info(f"{stat}: Mean diff={mean_diff:.4f}, Mean abs diff={mean_abs_diff:.4f}")
    
    # Save comparison
    output_file = str(GENERATED_DIR / 'prediction_comparison.csv')
    merged.to_csv(output_file, index=False)
    logger.info(f"Saved comparison to {output_file}")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='Generate predictions using fine-tuned batter model')
    parser.add_argument('--start-year', type=int, default=2025, help='First year to predict (default: 2025)')
    parser.add_argument('--future-years', type=int, default=10, help='Number of years to predict (default: 10)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (optional)')
    parser.add_argument('--compare', type=str, default=None, help='Compare with pre-trained predictions file (optional)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Fine-Tuned Batter Prediction Pipeline")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    
    try:
        # Load fine-tuned model
        model, scaler, data_config, metadata = load_finetuned_model()
        
        # Generate predictions
        predictions = generate_predictions(
            model=model,
            scaler=scaler,
            data_config=data_config,
            start_year=args.start_year,
            future_years=args.future_years,
            output_file=args.output
        )
        
        # Compare with pre-trained if requested
        if args.compare and predictions is not None:
            comparison = compare_with_pretrained(predictions, args.compare)
        
        logger.info("\n" + "="*60)
        logger.info("Prediction pipeline completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
