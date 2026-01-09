#!/usr/bin/env python3
"""
MLB Player Prediction Pipeline
Author: Niels Christoffersen
Version: 1.0
Last Updated: 12/28/2024

This script generates predictions for all MLB player types using trained models.
It matches the exact functionality of the original Jupyter notebooks.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_registry import ModelFactory
from core.data_processing import DataConfig, calculate_rate_stats
from core.prediction import (
    generate_batter_names, load_model_from_checkpoint,
    predict_all_2024_pitchers, predict_all_2024_batters,
    predict_all_2024_fielders, predict_all_2024_baserunners
)

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
PIPELINE_DIR = GENERATED_DIR / 'pipeline'

# Ensure directories exist
GENERATED_DIR.mkdir(exist_ok=True)
PIPELINE_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


def resolve_data_path(config_data_file: str) -> Path:
    """
    Resolve data file path from config.
    Config paths are relative to auto_train_models/ like '../data/historic_mlb/file.csv'
    We need to extract the part after 'data/' and append to DATA_DIR.
    """
    config_path = Path(config_data_file)
    # Find 'data' in the path parts and take everything after it
    parts = config_path.parts
    if 'data' in parts:
        data_idx = parts.index('data')
        relative_path = Path(*parts[data_idx+1:])  # e.g., 'historic_mlb/file.csv'
        return DATA_DIR / relative_path
    else:
        # Fallback to just the filename in DATA_DIR
        return DATA_DIR / config_path.name


def load_pitcher_models_and_scalers() -> tuple:
    """Load pitcher models and scalers for SP and RP"""
    
    # Load SP model
    sp_config = ModelFactory.get_config('pitcher_sp')
    sp_data_config = sp_config.get_data_config()
    
    sp_checkpoint_path = AUTO_TRAIN_DIR / sp_config.CHECKPOINT_DIR / sp_config.CHECKPOINT_FILE
    sp_scaler_path = AUTO_TRAIN_DIR / sp_config.SCALER_FILE
    
    sp_model = load_model_from_checkpoint(
        str(sp_checkpoint_path),
        sp_data_config,
        device
    )
    sp_scaler = joblib.load(sp_scaler_path)
    
    # Load RP model
    rp_config = ModelFactory.get_config('pitcher_rp')
    rp_data_config = rp_config.get_data_config()
    
    rp_checkpoint_path = AUTO_TRAIN_DIR / rp_config.CHECKPOINT_DIR / rp_config.CHECKPOINT_FILE
    rp_scaler_path = AUTO_TRAIN_DIR / rp_config.SCALER_FILE
    
    rp_model = load_model_from_checkpoint(
        str(rp_checkpoint_path),
        rp_data_config,
        device
    )
    rp_scaler = joblib.load(rp_scaler_path)
    
    return sp_model, rp_model, sp_scaler, rp_scaler, sp_config, rp_config


def load_fielding_models_and_scalers() -> tuple:
    """Load fielding models and scalers for all position groups"""
    
    position_models = {}
    position_scalers = {}
    position_configs = {}
    
    # Map position groups to their config keys
    config_map = {
        'infield': 'defense_infield',
        'outfield': 'defense_outfield',
        'catcher': 'defense_catcher'
    }
    
    for pos_group, config_key in config_map.items():
        config = ModelFactory.get_config(config_key)
        data_config = config.get_data_config()
        
        # Get checkpoint and scaler paths from config
        checkpoint_path = AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.CHECKPOINT_FILE
        scaler_path = AUTO_TRAIN_DIR / config.SCALER_FILE
        
        model = load_model_from_checkpoint(
            str(checkpoint_path),
            data_config,
            device
        )
        scaler = joblib.load(scaler_path)
        
        position_models[pos_group] = model
        position_scalers[pos_group] = scaler
        position_configs[pos_group] = config
    
    # Position group mapping
    position_group_map = {
        'C': 'catcher',
        '1B': 'infield', '2B': 'infield', '3B': 'infield', 'SS': 'infield',
        'LF': 'outfield', 'CF': 'outfield', 'RF': 'outfield'
    }
    
    # Input features mapping
    input_features_map = {
        pos_group: config.INPUT_FEATURES 
        for pos_group, config in position_configs.items()
    }
    
    # Sequence length mapping
    seq_length_map = {
        pos_group: config.get_data_config().seq_length
        for pos_group, config in position_configs.items()
    }
    
    return position_models, position_scalers, position_group_map, input_features_map, seq_length_map


def generate_pitcher_predictions(output_file: str = None) -> Optional[pd.DataFrame]:
    """Generate pitcher predictions matching notebook functionality"""
    
    logger.info("Starting pitcher predictions generation...")
    
    # Get data file from config
    sp_config_class = ModelFactory.get_config('pitcher_sp')
    data_file_path = resolve_data_path(sp_config_class.DATA_FILE)
    
    # Load data
    raw_df = pd.read_csv(data_file_path)
    
    # Generate or load pitcher names
    pitcher_names_path = DATA_DIR / 'pitcher_names.csv'
    if not pitcher_names_path.exists():
        player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
        player_names.to_csv(pitcher_names_path, index=False)
    else:
        player_names = pd.read_csv(pitcher_names_path)
    
    # Load models and scalers
    sp_model, rp_model, sp_scaler, rp_scaler, sp_config, rp_config = load_pitcher_models_and_scalers()
    
    # Generate predictions
    predictions_df = predict_all_2024_pitchers(
        raw_df=raw_df,
        player_names=player_names,
        sp_model=sp_model,
        rp_model=rp_model,
        sp_scaler=sp_scaler,
        rp_scaler=rp_scaler,
        input_features=sp_config.INPUT_FEATURES,  # Both use same features
        seq_length=sp_config.get_data_config().seq_length,
        future_years=15,
        cutoff_year=2025  # 2025 season completed, predict 2026+
    )
    
    if predictions_df is not None:
        # Save predictions
        output_path = output_file or str(PIPELINE_DIR / 'pitcher_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions_df)} pitcher predictions to {output_path}")
        
        # Display summary
        logger.info(f"Generated predictions for {predictions_df['Name'].nunique()} unique pitchers")
        logger.info(f"SP predictions: {len(predictions_df[predictions_df['Role'] == 'SP'])}")
        logger.info(f"RP predictions: {len(predictions_df[predictions_df['Role'] == 'RP'])}")
        
        return predictions_df
    else:
        logger.error("Failed to generate pitcher predictions")
        return None


def generate_batter_predictions(output_file: str = None, use_pretrained: bool = False) -> Optional[pd.DataFrame]:
    """Generate batter predictions matching notebook functionality
    
    Args:
        output_file: Path to save predictions
        use_pretrained: If True, use pretrained model (13 features). If False, try finetuned model first.
    """
    
    logger.info("Starting batter predictions generation...")
    
    # Get config
    batter_config_class = ModelFactory.get_config('batter')
    
    # Determine which data file to use based on model type
    # Check if we'll be using finetuned model (has statcast features) or pretrained (classical only)
    finetuned_checkpoint = AUTO_TRAIN_DIR / batter_config_class.CHECKPOINT_DIR / batter_config_class.FINETUNE_CHECKPOINT_FILE if hasattr(batter_config_class, 'FINETUNE_CHECKPOINT_FILE') else None
    
    if not use_pretrained and finetuned_checkpoint and finetuned_checkpoint.exists() and hasattr(batter_config_class, 'FINETUNE_DATA_FILE'):
        # Use statcast-enhanced data for finetuned model
        data_file_path = resolve_data_path(batter_config_class.FINETUNE_DATA_FILE)
        logger.info(f"Using finetuned data file (with statcast features)")
    else:
        # Use classical data for pretrained model
        data_file_path = resolve_data_path(batter_config_class.DATA_FILE)
        logger.info(f"Using pretrained data file (classical features only)")
    
    # Load data
    raw_df = pd.read_csv(data_file_path)
    
    # Apply rate stats calculation if needed
    raw_df = calculate_rate_stats(raw_df)
    
    # Generate player names
    player_names = generate_batter_names(raw_df)
    
    # Load model and scaler
    config = ModelFactory.get_config('batter')
    
    # Try to load finetuned model first (18 features), fall back to pretrained (13 features)
    finetuned_checkpoint = AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.FINETUNE_CHECKPOINT_FILE if hasattr(config, 'FINETUNE_CHECKPOINT_FILE') else None
    finetuned_scaler_path = AUTO_TRAIN_DIR / config.FINETUNE_SCALER_FILE if hasattr(config, 'FINETUNE_SCALER_FILE') else None
    
    if not use_pretrained and finetuned_checkpoint and finetuned_checkpoint.exists():
        logger.info("Loading finetuned model (18 features: classical + Statcast)")
        data_config = config.get_data_config('finetune')
        model = load_model_from_checkpoint(str(finetuned_checkpoint), data_config, device)
        
        # Try finetuned scaler first, fall back to pretrained scaler
        if finetuned_scaler_path and finetuned_scaler_path.exists():
            scaler = joblib.load(finetuned_scaler_path)
            logger.info("Using finetuned scaler")
        else:
            logger.warning(f"Finetuned scaler not found, using pretrained scaler")
            scaler = joblib.load(AUTO_TRAIN_DIR / config.SCALER_FILE)
        
        input_features = config.FINETUNE_FEATURES
    else:
        logger.info("Loading pretrained model (13 classical features)")
        data_config = config.get_data_config()
        
        # Use PRETRAIN_CHECKPOINT_FILE if available, otherwise fall back to CHECKPOINT_FILE
        if hasattr(config, 'PRETRAIN_CHECKPOINT_FILE'):
            checkpoint_path = AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.PRETRAIN_CHECKPOINT_FILE
        else:
            checkpoint_path = AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.CHECKPOINT_FILE
        
        model = load_model_from_checkpoint(
            str(checkpoint_path),
            data_config,
            device
        )
        scaler = joblib.load(AUTO_TRAIN_DIR / config.SCALER_FILE)
        input_features = config.INPUT_FEATURES
    
    # Generate predictions
    predictions_df = predict_all_2024_batters(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=input_features,  # Use features matching the model
        future_years=15,
        cutoff_year=2025  # 2025 season completed, predict 2026+
    )
    
    if predictions_df is not None:
        # Save predictions
        output_path = output_file or str(PIPELINE_DIR / 'batter_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions_df)} batter predictions to {output_path}")
        
        # Display summary
        logger.info(f"Generated predictions for {predictions_df['Name'].nunique()} unique batters")
        
        return predictions_df
    else:
        logger.error("Failed to generate batter predictions")
        return None


def generate_integrated_batter_predictions(output_file: str = None) -> Optional[pd.DataFrame]:
    """
    Generate batter predictions with proper WAR calculation using position-specific fielding data.
    This is the new approach that properly handles defensive values from position-specific predictions.
    """
    from core.prediction import calculate_batter_war_with_fielding
    
    logger.info("Starting integrated batter predictions with position-specific fielding...")
    
    # Get config
    batter_config_class = ModelFactory.get_config('batter')
    
    # Determine which data file to use - check for finetuned model
    finetuned_checkpoint = AUTO_TRAIN_DIR / batter_config_class.CHECKPOINT_DIR / batter_config_class.FINETUNE_CHECKPOINT_FILE if hasattr(batter_config_class, 'FINETUNE_CHECKPOINT_FILE') else None
    
    if finetuned_checkpoint and finetuned_checkpoint.exists() and hasattr(batter_config_class, 'FINETUNE_DATA_FILE'):
        # Use statcast-enhanced data for finetuned model
        data_file_path = resolve_data_path(batter_config_class.FINETUNE_DATA_FILE)
        logger.info(f"Using finetuned data file (with statcast features)")
    else:
        # Use classical data for pretrained model
        data_file_path = resolve_data_path(batter_config_class.DATA_FILE)
        logger.info(f"Using pretrained data file (classical features only)")
    
    # Generate batter predictions (without WAR)
    logger.info("Step 1: Generating base batter predictions...")
    raw_df = pd.read_csv(data_file_path)
    raw_df = calculate_rate_stats(raw_df)
    player_names = generate_batter_names(raw_df)
    
    config = ModelFactory.get_config('batter')
    
    # Try to load finetuned model first (18 features), fall back to pretrained (13 features)
    finetuned_checkpoint = AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.FINETUNE_CHECKPOINT_FILE if hasattr(config, 'FINETUNE_CHECKPOINT_FILE') else None
    finetuned_scaler_path = AUTO_TRAIN_DIR / config.FINETUNE_SCALER_FILE if hasattr(config, 'FINETUNE_SCALER_FILE') else None
    
    if finetuned_checkpoint and finetuned_checkpoint.exists():
        logger.info("Loading finetuned model (18 features: classical + Statcast)")
        data_config = config.get_data_config('finetune')
        model = load_model_from_checkpoint(str(finetuned_checkpoint), data_config, device)
        
        # Try finetuned scaler first, fall back to pretrained scaler
        if finetuned_scaler_path and finetuned_scaler_path.exists():
            scaler = joblib.load(finetuned_scaler_path)
            logger.info("Using finetuned scaler")
        else:
            logger.warning(f"Finetuned scaler not found, using pretrained scaler")
            scaler = joblib.load(AUTO_TRAIN_DIR / config.SCALER_FILE)
        
        input_features = config.FINETUNE_FEATURES
    else:
        logger.info("Loading pretrained model (13 classical features)")
        data_config = config.get_data_config()
        checkpoint_path = AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.CHECKPOINT_FILE
        model = load_model_from_checkpoint(
            str(checkpoint_path),
            data_config,
            device
        )
        scaler = joblib.load(AUTO_TRAIN_DIR / config.SCALER_FILE)
        input_features = config.INPUT_FEATURES
    
    # Get batter predictions without WAR calculation
    from core.prediction import predict_all_2024_batters_no_war
    batter_df = predict_all_2024_batters_no_war(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=input_features,
        future_years=15,
        cutoff_year=2025  # 2025 season completed, predict 2026+
    )
    
    if batter_df is None:
        logger.error("Failed to generate batter predictions")
        return None
    
    # Load or generate fielding predictions  
    logger.info("Step 2: Loading position-specific fielding predictions...")
    fielding_file = PIPELINE_DIR / 'fielding_predictions.csv'
    
    if fielding_file.exists():
        logger.info(f"Loading existing fielding predictions from {fielding_file}")
        fielding_df = pd.read_csv(fielding_file)
    else:
        logger.info("Generating new fielding predictions...")
        fielding_df = generate_fielding_predictions()
        if fielding_df is None:
            logger.error("Failed to generate fielding predictions")
            return None
    
    # Load or generate baserunning predictions
    logger.info("Step 2b: Loading baserunning predictions...")
    baserunning_file = PIPELINE_DIR / 'baserunning_predictions.csv'
    
    if baserunning_file.exists():
        logger.info(f"Loading existing baserunning predictions from {baserunning_file}")
        baserunning_df = pd.read_csv(baserunning_file)
    else:
        logger.info("Generating new baserunning predictions...")
        baserunning_df = generate_baserunning_predictions()
        if baserunning_df is None:
            logger.error("Failed to generate baserunning predictions")
            return None
    
    # Calculate WAR with position-specific fielding data and baserunning
    logger.info("Step 3: Calculating WAR with position-specific defensive and baserunning values...")
    from core.prediction import calculate_batter_war_with_fielding_and_baserunning
    integrated_df = calculate_batter_war_with_fielding_and_baserunning(batter_df, fielding_df, baserunning_df)
    
    # Filter out 2025 predictions (players who played in 2024 but not 2025)
    logger.info("Step 4: Filtering out 2025 predictions...")
    initial_count = len(integrated_df)
    integrated_df = integrated_df[integrated_df['Year'] != 2025].copy()
    filtered_count = len(integrated_df)
    logger.info(f"Removed {initial_count - filtered_count} rows with Year=2025")
    
    # Sort by Year and WAR (descending)
    integrated_df = integrated_df.sort_values(['Year', 'WAR'], ascending=[True, False])
    
    if output_file:
        output_path = output_file
    else:
        output_path = str(PIPELINE_DIR / 'integrated_batter_predictions.csv')
    
    integrated_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(integrated_df)} integrated batter predictions to {output_path}")
    
    # Display summary
    logger.info(f"Generated integrated predictions for {integrated_df['Name'].nunique()} unique batters")
    logger.info(f"Average WAR: {integrated_df['WAR'].mean():.2f}")
    
    return integrated_df


def generate_fielding_predictions(output_file: str = None) -> Optional[pd.DataFrame]:
    """Generate fielding predictions matching notebook functionality"""
    
    logger.info("Starting fielding predictions generation...")
    
    # Get data file from config
    fielding_config_class = ModelFactory.get_config('defense_infield')
    data_file_path = resolve_data_path(fielding_config_class.DATA_FILE)
    
    # Load data (using the same file as notebook)
    raw_df = pd.read_csv(data_file_path)
    
    # Apply rate stats calculation to the entire dataset
    raw_df = calculate_rate_stats(raw_df)
    
    # Generate player names
    player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
    
    # Load models and scalers
    position_models, position_scalers, position_group_map, input_features_map, seq_length_map = load_fielding_models_and_scalers()
    
    # Generate predictions
    predictions_df = predict_all_2024_fielders(
        raw_df=raw_df,
        player_names=player_names,
        position_models=position_models,
        position_scalers=position_scalers,
        position_group_map=position_group_map,
        input_features_map=input_features_map,
        seq_length_map=seq_length_map,
        future_years=15,
        cutoff_year=2025  # 2025 season completed, predict 2026+
    )
    
    if predictions_df is not None:
        # Reorder columns: metadata first, then all predicted features
        metadata_cols = ['Name', 'Age', 'Year', 'IDfg', 'Pos']
        
        # Get all feature columns (everything except metadata)
        feature_cols = [col for col in predictions_df.columns if col not in metadata_cols]
        
        # Reorder: metadata + features
        predictions_df = predictions_df[metadata_cols + feature_cols]
        
        # Save predictions
        output_path = output_file or str(PIPELINE_DIR / 'fielding_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions_df)} fielding predictions to {output_path}")
        
        # Display summary (use original df for Position_Group analysis)
        logger.info(f"Generated predictions for {predictions_df['Name'].nunique()} unique fielders")
        
        # For summary, we need to recreate position group info since we removed that column
        pos_counts = {}
        for pos in predictions_df['Pos'].unique():
            if pos in ['1B', '2B', '3B', 'SS']:
                pos_counts['infield'] = pos_counts.get('infield', 0) + len(predictions_df[predictions_df['Pos'] == pos])
            elif pos in ['LF', 'CF', 'RF']:
                pos_counts['outfield'] = pos_counts.get('outfield', 0) + len(predictions_df[predictions_df['Pos'] == pos])
            elif pos == 'C':
                pos_counts['catcher'] = pos_counts.get('catcher', 0) + len(predictions_df[predictions_df['Pos'] == pos])
        
        for pos_group, count in pos_counts.items():
            logger.info(f"{pos_group.title()} predictions: {count}")
        
        return predictions_df
    else:
        logger.error("Failed to generate fielding predictions")
        return None


def generate_baserunning_predictions(output_file: str = None) -> Optional[pd.DataFrame]:
    """Generate baserunning predictions matching notebook functionality"""
    
    logger.info("Starting baserunning predictions generation...")
    
    # Get data file from config
    baserunning_config_class = ModelFactory.get_config('baserunning')
    data_file_path = resolve_data_path(baserunning_config_class.DATA_FILE)
    
    # Load data
    raw_df = pd.read_csv(data_file_path)
    
    # Apply rate stats calculation
    raw_df = calculate_rate_stats(raw_df)
    
    # Generate player names
    player_names = generate_batter_names(raw_df)
    
    # Load model and scaler
    config = ModelFactory.get_config('baserunning')
    data_config = config.get_data_config()
    
    # Use config paths for checkpoint and scaler
    checkpoint_path = AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.CHECKPOINT_FILE
    scaler_path = AUTO_TRAIN_DIR / config.SCALER_FILE
    
    model = load_model_from_checkpoint(
        str(checkpoint_path),
        data_config,
        device
    )
    scaler = joblib.load(scaler_path)
    
    # Generate predictions
    predictions_df = predict_all_2024_baserunners(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=config.INPUT_FEATURES,
        seq_length=data_config.seq_length,
        future_years=15,
        cutoff_year=2025  # 2025 season completed, predict 2026+
    )
    
    if predictions_df is not None:
        # Save predictions
        output_path = output_file or str(PIPELINE_DIR / 'baserunning_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions_df)} baserunning predictions to {output_path}")
        
        # Display summary
        logger.info(f"Generated predictions for {predictions_df['Name'].nunique()} unique baserunners")
        
        return predictions_df
    else:
        logger.error("Failed to generate baserunning predictions")
        return None


def main():
    """Main entry point for prediction generation"""
    
    parser = argparse.ArgumentParser(description='Generate MLB player predictions')
    parser.add_argument('--model-type', choices=['pitcher', 'batter', 'fielding', 'baserunning', 'integrated-batter', 'all'],
                       default='all', help='Type of predictions to generate')
    parser.add_argument('--output-dir', type=str, default=str(PIPELINE_DIR),
                       help='Output directory for prediction files')
    parser.add_argument('--use-pretrained', action='store_true',
                       help='Use pretrained model only (13 classical features) instead of finetuned model')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting MLB Player Prediction Pipeline")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output directory: {output_dir}")
    
    success = True
    
    try:
        if args.model_type in ['pitcher', 'all']:
            output_file = str(output_dir / 'pitcher_predictions.csv')
            result = generate_pitcher_predictions(output_file)
            if result is None:
                success = False
        
        if args.model_type in ['batter', 'all']:
            output_file = str(output_dir / 'batter_predictions.csv')
            result = generate_batter_predictions(output_file, use_pretrained=args.use_pretrained)
            if result is None:
                success = False
        
        if args.model_type in ['fielding', 'all']:
            output_file = str(output_dir / 'fielding_predictions.csv')
            result = generate_fielding_predictions(output_file)
            if result is None:
                success = False
        
        if args.model_type in ['baserunning', 'all']:
            output_file = str(output_dir / 'baserunning_predictions.csv')
            result = generate_baserunning_predictions(output_file)
            if result is None:
                success = False
        
        # New integrated batter predictions with position-specific fielding
        if args.model_type == 'integrated-batter':
            output_file = str(output_dir / 'integrated_batter_predictions.csv')
            result = generate_integrated_batter_predictions(output_file)
            if result is None:
                success = False
        
        if success:
            logger.info("Prediction pipeline completed successfully!")
            
            # Provide helpful information about the new feature
            if args.model_type == 'integrated-batter':
                logger.info("Generated integrated batter predictions with position-specific defensive values")
                logger.info("This approach properly handles players who play multiple positions")
            
        else:
            logger.error("Some predictions failed to generate")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
