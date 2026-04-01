#!/usr/bin/env python3
"""
MLB Player Prediction Pipeline
Author: Niels Christoffersen
Version: 2.0
Last Updated: January 2026

This script generates predictions for all MLB player types using trained models.
It matches the exact functionality of the original Jupyter notebooks.

Changes in v2.0:
- Added --cutoff-year CLI argument for backtesting support
- Renamed functions to remove hardcoded year references
- Consolidated duplicate batter prediction functions
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
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_registry import ModelFactory
from core.data_processing import DataConfig, calculate_rate_stats
from core.prediction import (
    generate_batter_names, 
    load_model_from_checkpoint,
    predict_all_batters,
    predict_all_fielders, 
    predict_all_baserunners
)
from core.pitcher_prediction import predict_all_pitchers

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
ROSTER_FILE = DATA_DIR / 'active_roster' / 'current_rosters.csv'

# Ensure directories exist
GENERATED_DIR.mkdir(exist_ok=True)
PIPELINE_DIR.mkdir(exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


def load_roster_ids() -> Optional[set]:
    """
    Load active 40-man roster player IDs (FanGraphs IDfg) from current_rosters.csv.
    
    Returns:
        Set of integer IDfg values for all roster players with known FanGraphs IDs,
        or None if the roster file does not exist.
    """
    if not ROSTER_FILE.exists():
        logger.warning(f"Roster file not found at {ROSTER_FILE} — roster recovery disabled")
        return None
    
    roster = pd.read_csv(ROSTER_FILE)
    roster_with_fg = roster.dropna(subset=['fg_id'])
    roster_ids = set(pd.to_numeric(roster_with_fg['fg_id'], errors='coerce').dropna().astype(int))
    logger.info(f"Loaded {len(roster_ids)} roster player IDs from {ROSTER_FILE.name}")
    return roster_ids


def load_pitcher_ids() -> Optional[set]:
    """Load IDfg values for known pitchers from the pitching data CSV."""
    sp_config = ModelFactory.get_config('pitcher_sp')
    data_path = resolve_data_path(sp_config.DATA_FILE)
    if not data_path.exists():
        logger.warning(f"Pitcher data file not found at {data_path}")
        return None
    pitcher_df = pd.read_csv(data_path, usecols=['IDfg'])
    ids = set(pitcher_df['IDfg'].dropna().astype(int).unique())
    logger.info(f"Loaded {len(ids)} pitcher IDfg values for batting filter")
    return ids


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


def load_pitcher_models_and_scalers(use_pretrained: bool = False) -> tuple:
    """Load pitcher models and scalers for SP and RP
    
    Args:
        use_pretrained: If True, use pretrained model (classical features).
                       If False, try finetuned model first.
    
    When UNIFIED_PITCHER_MODEL is True in PitcherSPConfig, a single model/scaler
    is loaded and returned for both SP and RP slots.
    """
    
    # Load SP model
    sp_config = ModelFactory.get_config('pitcher_sp')
    unified = getattr(sp_config, 'UNIFIED_PITCHER_MODEL', False)
    
    # Determine which checkpoint and config to use
    sp_pretrain_checkpoint = AUTO_TRAIN_DIR / sp_config.CHECKPOINT_DIR / sp_config.PRETRAIN_CHECKPOINT_FILE
    sp_finetune_checkpoint = AUTO_TRAIN_DIR / sp_config.CHECKPOINT_DIR / sp_config.FINETUNE_CHECKPOINT_FILE
    
    if not use_pretrained and sp_finetune_checkpoint.exists() and hasattr(sp_config, 'FINETUNE_CHECKPOINT_FILE'):
        logger.info("Using SP finetuned model (classical + PITCHf/x + Statcast)")
        sp_data_config = sp_config.get_data_config(mode='finetune')
        sp_checkpoint_path = sp_finetune_checkpoint
        sp_scaler_path = AUTO_TRAIN_DIR / sp_config.FINETUNE_SCALER_FILE
    else:
        logger.info("Using SP pretrained model (classical features only)")
        sp_data_config = sp_config.get_data_config(mode='pretrain')
        sp_checkpoint_path = sp_pretrain_checkpoint
        sp_scaler_path = AUTO_TRAIN_DIR / sp_config.PRETRAIN_SCALER_FILE
    
    sp_model = load_model_from_checkpoint(
        str(sp_checkpoint_path),
        sp_data_config,
        device
    )
    sp_scaler = joblib.load(sp_scaler_path)
    
    if unified:
        # Unified mode: reuse the SP model/scaler/config for RP as well
        logger.info("UNIFIED_PITCHER_MODEL=True — using SP model for all pitchers (no separate RP model)")
        return sp_model, sp_model, sp_scaler, sp_scaler, sp_config, sp_config, sp_data_config, sp_data_config
    
    # Load RP model (separate mode only)
    rp_config = ModelFactory.get_config('pitcher_rp')
    
    rp_pretrain_checkpoint = AUTO_TRAIN_DIR / rp_config.CHECKPOINT_DIR / rp_config.PRETRAIN_CHECKPOINT_FILE
    rp_finetune_checkpoint = AUTO_TRAIN_DIR / rp_config.CHECKPOINT_DIR / rp_config.FINETUNE_CHECKPOINT_FILE
    
    if not use_pretrained and rp_finetune_checkpoint.exists() and hasattr(rp_config, 'FINETUNE_CHECKPOINT_FILE'):
        logger.info("Using RP finetuned model (classical + PITCHf/x + Statcast)")
        rp_data_config = rp_config.get_data_config(mode='finetune')
        rp_checkpoint_path = rp_finetune_checkpoint
        rp_scaler_path = AUTO_TRAIN_DIR / rp_config.FINETUNE_SCALER_FILE
    else:
        logger.info("Using RP pretrained model (classical features only)")
        rp_data_config = rp_config.get_data_config(mode='pretrain')
        rp_checkpoint_path = rp_pretrain_checkpoint
        rp_scaler_path = AUTO_TRAIN_DIR / rp_config.PRETRAIN_SCALER_FILE
    
    rp_model = load_model_from_checkpoint(
        str(rp_checkpoint_path),
        rp_data_config,
        device
    )
    rp_scaler = joblib.load(rp_scaler_path)
    
    return sp_model, rp_model, sp_scaler, rp_scaler, sp_config, rp_config, sp_data_config, rp_data_config


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


def generate_pitcher_predictions(
    output_file: str = None, 
    use_pretrained: bool = False,
    cutoff_year: int = None,
    roster_ids: set = None
) -> Optional[pd.DataFrame]:
    """Generate pitcher predictions for SP and RP
    
    Args:
        output_file: Path to save predictions
        use_pretrained: If True, use pretrained model (classical features).
                       If False, try finetuned model first.
        cutoff_year: Last year of actual data. Predictions start from cutoff_year + 1.
                    Defaults to current year - 1 if not specified.
    """
    # Default to previous year if not specified
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting pitcher predictions generation (cutoff_year={cutoff_year})...")
    
    # Get data file from config
    sp_config_class = ModelFactory.get_config('pitcher_sp')
    rp_config_class = ModelFactory.get_config('pitcher_rp')
    
    # Check for Marcel projection method (use SP config as primary toggle;
    # if either SP or RP is set to marcel, both use Marcel for consistency)
    prediction_method = getattr(sp_config_class, 'PREDICTION_METHOD', 'lstm').lower()
    if prediction_method != 'marcel':
        prediction_method = getattr(rp_config_class, 'PREDICTION_METHOD', 'lstm').lower()
    
    if prediction_method == 'marcel':
        from core.marcel_projections import marcel_pitcher_projections
        logger.info("Using Marcel projections for pitchers")
        
        data_file_path = resolve_data_path(sp_config_class.DATA_FILE)
        raw_df = pd.read_csv(data_file_path)
        raw_df = calculate_rate_stats(raw_df)
        
        pitcher_names_path = DATA_DIR / 'pitcher_names.csv'
        if not pitcher_names_path.exists():
            player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
            player_names.to_csv(pitcher_names_path, index=False)
        else:
            player_names = pd.read_csv(pitcher_names_path)
        
        predictions_df = marcel_pitcher_projections(
            raw_df=raw_df,
            player_names=player_names,
            future_years=15,
            cutoff_year=cutoff_year,
            roster_ids=roster_ids,
        )
        
        if predictions_df is not None:
            output_path = output_file or str(PIPELINE_DIR / 'pitcher_predictions.csv')
            predictions_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(predictions_df)} Marcel pitcher predictions to {output_path}")
            logger.info(f"Generated predictions for {predictions_df['Name'].nunique()} unique pitchers")
            sp_count = len(predictions_df[predictions_df['Role'] == 'SP'])
            rp_count = len(predictions_df[predictions_df['Role'] == 'RP'])
            logger.info(f"SP predictions: {sp_count}")
            logger.info(f"RP predictions: {rp_count}")
            return predictions_df
        else:
            logger.error("Failed to generate Marcel pitcher predictions")
            return None
    
    data_file_path = resolve_data_path(sp_config_class.DATA_FILE)
    
    # Load data and compute derived rate stats (HR%, HBP% from counting stats)
    raw_df = pd.read_csv(data_file_path)
    raw_df = calculate_rate_stats(raw_df)
    
    # Generate or load pitcher names
    pitcher_names_path = DATA_DIR / 'pitcher_names.csv'
    if not pitcher_names_path.exists():
        player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
        player_names.to_csv(pitcher_names_path, index=False)
    else:
        player_names = pd.read_csv(pitcher_names_path)
    
    # Load models and scalers
    sp_model, rp_model, sp_scaler, rp_scaler, sp_config, rp_config, sp_data_config, rp_data_config = load_pitcher_models_and_scalers(use_pretrained)
    
    # Generate predictions with separate feature lists for SP and RP
    predictions_df = predict_all_pitchers(
        raw_df=raw_df,
        player_names=player_names,
        sp_model=sp_model,
        rp_model=rp_model,
        sp_scaler=sp_scaler,
        rp_scaler=rp_scaler,
        sp_input_features=sp_data_config.input_features,
        rp_input_features=rp_data_config.input_features,
        seq_length=sp_data_config.seq_length,
        future_years=15,
        cutoff_year=cutoff_year,
        sp_config=sp_config,
        rp_config=rp_config,
        roster_ids=roster_ids
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


def generate_batter_predictions(
    output_file: str = None, 
    use_pretrained: bool = False,
    cutoff_year: int = None,
    roster_ids: set = None
) -> Optional[pd.DataFrame]:
    """Generate batter predictions matching notebook functionality
    
    Args:
        output_file: Path to save predictions
        use_pretrained: If True, use pretrained model (13 features). If False, try finetuned model first.
        cutoff_year: Last year of actual data. Predictions start from cutoff_year + 1.
                    Defaults to current year - 1 if not specified.
    """
    # Default to previous year if not specified
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting batter predictions generation (cutoff_year={cutoff_year})...")
    
    # Get config
    batter_config_class = ModelFactory.get_config('batter')
    
    # Check for Marcel projection method
    prediction_method = getattr(batter_config_class, 'PREDICTION_METHOD', 'lstm').lower()
    
    if prediction_method == 'marcel':
        from core.marcel_projections import marcel_batter_projections
        logger.info("Using Marcel projections for batters")
        
        # Load data (use statcast data for x-stats if available)
        if hasattr(batter_config_class, 'FINETUNE_DATA_FILE'):
            data_file_path = resolve_data_path(batter_config_class.FINETUNE_DATA_FILE)
        else:
            data_file_path = resolve_data_path(batter_config_class.DATA_FILE)
        
        raw_df = pd.read_csv(data_file_path)
        raw_df = calculate_rate_stats(raw_df)
        player_names = generate_batter_names(raw_df)
        
        predictions_df = marcel_batter_projections(
            raw_df=raw_df,
            player_names=player_names,
            future_years=15,
            cutoff_year=cutoff_year,
            roster_ids=roster_ids,
            use_xstats=getattr(batter_config_class, 'USE_XWOBA_FOR_PREDICTIONS', True),
        )
        
        if predictions_df is not None:
            output_path = output_file or str(PIPELINE_DIR / 'batter_predictions.csv')
            predictions_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(predictions_df)} Marcel batter predictions to {output_path}")
            logger.info(f"Generated predictions for {predictions_df['Name'].nunique()} unique batters")
            return predictions_df
        else:
            logger.error("Failed to generate Marcel batter predictions")
            return None
    
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
    pitcher_ids = load_pitcher_ids()
    predictions_df = predict_all_batters(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=input_features,  # Use features matching the model
        seq_length=data_config.seq_length,
        future_years=15,
        cutoff_year=cutoff_year,
        min_pa_current=config.MIN_PA_CURRENT if hasattr(config, 'MIN_PA_CURRENT') else 100,
        roster_ids=roster_ids,
        pitcher_ids=pitcher_ids
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


def generate_integrated_batter_predictions(
    output_file: str = None,
    cutoff_year: int = None,
    roster_ids: set = None
) -> Optional[pd.DataFrame]:
    """
    Generate batter predictions with proper WAR calculation using position-specific fielding data.
    This is the new approach that properly handles defensive values from position-specific predictions.
    
    WAR calculation is delegated to evaluation/calculate_war.py which is the authoritative source.
    
    Args:
        output_file: Path to save predictions
        cutoff_year: Last year of actual data. Predictions start from cutoff_year + 1.
                    Defaults to current year - 1 if not specified.
    """
    # Import WAR calculation from the authoritative source
    sys.path.insert(0, str(AUTO_TRAIN_DIR / 'evaluation'))
    from calculate_war import calculate_war_components, calculate_baserunning_value, calculate_defensive_value, load_player_orgs
    
    # Default to previous year if not specified
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting integrated batter predictions with position-specific fielding (cutoff_year={cutoff_year})...")
    
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
    
    # Get batter predictions without WAR calculation (WAR calculated in post-processing via calculate_war.py)
    pitcher_ids = load_pitcher_ids()
    batter_df = predict_all_batters(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=input_features,
        seq_length=data_config.seq_length,
        future_years=15,
        cutoff_year=cutoff_year,
        min_pa_current=config.MIN_PA_CURRENT if hasattr(config, 'MIN_PA_CURRENT') else 100,
        roster_ids=roster_ids,
        pitcher_ids=pitcher_ids
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
    # Use the authoritative WAR calculation from evaluation/calculate_war.py
    logger.info("Step 3: Calculating WAR with position-specific defensive and baserunning values...")
    
    # Load player organizations for park factors
    org_data = load_player_orgs(GENERATED_DIR)
    batter_df = batter_df.merge(org_data, on='IDfg', how='left')
    
    # Calculate WAR components for each batter prediction
    war_components_list = []
    for idx, row in batter_df.iterrows():
        try:
            war, components = calculate_war_components(row, baserunning_df, fielding_df)
            components['IDfg'] = row['IDfg']
            components['Year'] = row['Year']
            war_components_list.append(components)
        except Exception as e:
            logger.warning(f"Error calculating WAR for {row.get('Name', 'Unknown')} ({row['IDfg']}): {e}")
            continue
    
    # Merge WAR components back into batter dataframe
    war_df = pd.DataFrame(war_components_list)
    integrated_df = batter_df.merge(war_df, on=['IDfg', 'Year'], how='left', suffixes=('_old', ''))
    
    # Clean up duplicate columns
    columns_to_remove = [col for col in integrated_df.columns if col.endswith('_old')]
    integrated_df = integrated_df.drop(columns=columns_to_remove, errors='ignore')
    
    # Filter out predictions from cutoff year (players who played in cutoff_year-1 but not cutoff_year)
    # Projections should start from cutoff_year + 1
    filter_year = cutoff_year
    logger.info(f"Step 4: Filtering out {filter_year} predictions...")
    initial_count = len(integrated_df)
    integrated_df = integrated_df[integrated_df['Year'] != filter_year].copy()
    filtered_count = len(integrated_df)
    logger.info(f"Removed {initial_count - filtered_count} rows with Year={filter_year}")
    
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


def generate_fielding_predictions(
    output_file: str = None,
    cutoff_year: int = None,
    use_aging_enforcer: bool = False,
    roster_ids: set = None
) -> Optional[pd.DataFrame]:
    """Generate fielding predictions matching notebook functionality
    
    Args:
        output_file: Path to save predictions
        cutoff_year: Last year of actual data. Predictions start from cutoff_year + 1.
                    Defaults to current year - 1 if not specified.
        use_aging_enforcer: If True, apply aging constraints to prevent unrealistic late-career improvements.
    """
    # Default to previous year if not specified
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting fielding predictions generation (cutoff_year={cutoff_year})...")
    
    # Get data file from config
    fielding_config_class = ModelFactory.get_config('defense_infield')
    data_file_path = resolve_data_path(fielding_config_class.DATA_FILE)
    
    # Load data (using the same file as notebook)
    raw_df = pd.read_csv(data_file_path)
    
    # Apply rate stats calculation to the entire dataset
    raw_df = calculate_rate_stats(raw_df)
    
    # Generate player names
    player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
    
    # Load models and scalers (needed for model groups / position_group_map)
    position_models, position_scalers, position_group_map, input_features_map, seq_length_map = load_fielding_models_and_scalers()
    
    # =========================================================================
    # Check if any position group uses Marcel projections
    # =========================================================================
    marcel_groups = set()
    lstm_groups = set()
    config_map = {
        'infield': 'defense_infield',
        'outfield': 'defense_outfield',
        'catcher': 'defense_catcher',
    }
    for group_name, config_key in config_map.items():
        cfg = ModelFactory.get_config(config_key)
        method = getattr(cfg, 'PREDICTION_METHOD', 'lstm').lower()
        if method == 'marcel':
            marcel_groups.add(group_name)
        else:
            lstm_groups.add(group_name)
    
    # Build position profiles from historical fielding data for multi-position predictions
    from core.position_profiles import build_position_profiles, load_batting_for_games
    batting_for_games = load_batting_for_games()
    # Get all player IDs in the fielding data as candidates
    all_player_ids = raw_df['IDfg'].unique().tolist()
    if roster_ids:
        all_player_ids = list(set(all_player_ids) | roster_ids)
    profiles = build_position_profiles(raw_df, batting_for_games, all_player_ids, cutoff_year=cutoff_year)
    logger.info(f"Built position profiles for {len(profiles)} players")

    predictions_parts = []

    # --- Marcel predictions for groups that use it ---
    if marcel_groups:
        from core.marcel_projections import marcel_fielding_projections
        logger.info(f"Using Marcel projections for: {', '.join(sorted(marcel_groups))}")
        marcel_df = marcel_fielding_projections(
            raw_df=raw_df,
            player_names=player_names,
            position_group_map=position_group_map,
            input_features_map=input_features_map,
            future_years=15,
            cutoff_year=cutoff_year,
            roster_ids=roster_ids,
            position_profiles=profiles,
        )
        if marcel_df is not None:
            marcel_df = marcel_df[marcel_df['Position_Group'].isin(marcel_groups)]
            predictions_parts.append(marcel_df)

    # --- LSTM predictions for groups that use it ---
    if lstm_groups:
        logger.info(f"Using LSTM projections for: {', '.join(sorted(lstm_groups))}")
        # Filter models/scalers/features to only LSTM groups
        lstm_models = {k: v for k, v in position_models.items() if k in lstm_groups}
        lstm_scalers = {k: v for k, v in position_scalers.items() if k in lstm_groups}
        lstm_features = {k: v for k, v in input_features_map.items() if k in lstm_groups}
        lstm_seq = {k: v for k, v in seq_length_map.items() if k in lstm_groups}

        lstm_df = predict_all_fielders(
            raw_df=raw_df,
            player_names=player_names,
            position_models=lstm_models,
            position_scalers=lstm_scalers,
            position_group_map=position_group_map,
            input_features_map=lstm_features,
            seq_length_map=lstm_seq,
            future_years=15,
            cutoff_year=cutoff_year,
            use_aging_enforcer=use_aging_enforcer,
            roster_ids=roster_ids,
            position_profiles=profiles,
        )
        if lstm_df is not None:
            predictions_parts.append(lstm_df)

    # Combine
    if predictions_parts:
        predictions_df = pd.concat(predictions_parts, ignore_index=True)
    else:
        predictions_df = None
    
    if predictions_df is not None:
        # Drop Position_Group if present (used internally for routing only)
        if 'Position_Group' in predictions_df.columns:
            predictions_df = predictions_df.drop(columns=['Position_Group'])
        
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


def generate_baserunning_predictions(
    output_file: str = None,
    cutoff_year: int = None,
    roster_ids: set = None
) -> Optional[pd.DataFrame]:
    """Generate baserunning predictions matching notebook functionality
    
    Args:
        output_file: Path to save predictions
        cutoff_year: Last year of actual data. Predictions start from cutoff_year + 1.
                    Defaults to current year - 1 if not specified.
    """
    # Default to previous year if not specified
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting baserunning predictions generation (cutoff_year={cutoff_year})...")
    
    # Get data file from config
    baserunning_config_class = ModelFactory.get_config('baserunning')
    data_file_path = resolve_data_path(baserunning_config_class.DATA_FILE)
    
    # Load data
    raw_df = pd.read_csv(data_file_path)
    
    # Apply rate stats calculation
    raw_df = calculate_rate_stats(raw_df)
    
    # Generate player names
    player_names = generate_batter_names(raw_df)
    
    config = ModelFactory.get_config('baserunning')
    prediction_method = getattr(config, 'PREDICTION_METHOD', 'lstm').lower()

    if prediction_method == 'marcel':
        from core.marcel_projections import marcel_baserunning_projections
        logger.info("Using Marcel projections for baserunning")
        predictions_df = marcel_baserunning_projections(
            raw_df=raw_df,
            player_names=player_names,
            input_features=config.INPUT_FEATURES,
            future_years=15,
            cutoff_year=cutoff_year,
            roster_ids=roster_ids,
        )
    else:
        # Load model and scaler for LSTM
        data_config = config.get_data_config()
        checkpoint_path = AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.CHECKPOINT_FILE
        scaler_path = AUTO_TRAIN_DIR / config.SCALER_FILE
        model = load_model_from_checkpoint(
            str(checkpoint_path),
            data_config,
            device
        )
        scaler = joblib.load(scaler_path)
        predictions_df = predict_all_baserunners(
            raw_df=raw_df,
            player_names=player_names,
            model=model,
            scaler=scaler,
            input_features=config.INPUT_FEATURES,
            seq_length=data_config.seq_length,
            future_years=15,
            cutoff_year=cutoff_year,
            roster_ids=roster_ids
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
    
    # Calculate default cutoff year (previous year)
    default_cutoff_year = datetime.now().year - 1
    
    parser = argparse.ArgumentParser(
        description='Generate MLB player predictions using trained LSTM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all predictions for current year
  python predict_models.py --model-type all
  
  # Generate batter predictions for backtesting 2023 season
  python predict_models.py --model-type batter --cutoff-year 2022
  
  # Generate predictions using pretrained model only
  python predict_models.py --model-type pitcher --use-pretrained
        """
    )
    parser.add_argument('--model-type', 
                       choices=['pitcher', 'batter', 'fielding', 'baserunning', 'integrated-batter', 'all'],
                       default='all', 
                       help='Type of predictions to generate (default: all)')
    parser.add_argument('--output-dir', type=str, default=str(PIPELINE_DIR),
                       help='Output directory for prediction files')
    parser.add_argument('--cutoff-year', type=int, default=default_cutoff_year,
                       help=f'Last year of actual data. Predictions start from cutoff_year + 1. '
                            f'(default: {default_cutoff_year})')
    parser.add_argument('--use-pretrained', action='store_true',
                       help='Use pretrained model only (classical features) instead of finetuned model')
    parser.add_argument('--use-aging-enforcer', action='store_true',
                       help='Apply aging constraints to fielding predictions (prevents unrealistic late-career improvements)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("MLB Player Prediction Pipeline v2.0")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Cutoff year: {args.cutoff_year} (projections start from {args.cutoff_year + 1})")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using pretrained model: {args.use_pretrained}")
    logger.info(f"Using aging enforcer: {args.use_aging_enforcer}")
    
    # Load active roster IDs for recovery of missing players
    roster_ids = load_roster_ids()
    
    success = True
    
    try:
        if args.model_type in ['pitcher', 'all']:
            output_file = str(output_dir / 'pitcher_predictions.csv')
            result = generate_pitcher_predictions(
                output_file, 
                use_pretrained=args.use_pretrained,
                cutoff_year=args.cutoff_year,
                roster_ids=roster_ids
            )
            if result is None:
                success = False
        
        if args.model_type in ['batter', 'all']:
            output_file = str(output_dir / 'batter_predictions.csv')
            result = generate_batter_predictions(
                output_file, 
                use_pretrained=args.use_pretrained,
                cutoff_year=args.cutoff_year,
                roster_ids=roster_ids
            )
            if result is None:
                success = False
        
        if args.model_type in ['fielding', 'all']:
            output_file = str(output_dir / 'fielding_predictions.csv')
            result = generate_fielding_predictions(
                output_file,
                cutoff_year=args.cutoff_year,
                use_aging_enforcer=args.use_aging_enforcer,
                roster_ids=roster_ids
            )
            if result is None:
                success = False
        
        if args.model_type in ['baserunning', 'all']:
            output_file = str(output_dir / 'baserunning_predictions.csv')
            result = generate_baserunning_predictions(
                output_file,
                cutoff_year=args.cutoff_year,
                roster_ids=roster_ids
            )
            if result is None:
                success = False
        
        # New integrated batter predictions with position-specific fielding
        if args.model_type == 'integrated-batter':
            output_file = str(output_dir / 'integrated_batter_predictions.csv')
            result = generate_integrated_batter_predictions(
                output_file,
                cutoff_year=args.cutoff_year,
                roster_ids=roster_ids
            )
            if result is None:
                success = False
        
        if success:
            logger.info("=" * 60)
            logger.info("Prediction pipeline completed successfully!")
            logger.info("=" * 60)
            
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
