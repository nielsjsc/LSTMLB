# Data processing and preprocessing functions

# Core libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

   
@dataclass
class DataConfig:
    """Configuration for data preprocessing"""
    seq_length: int = 7
    start_season: int = 2016
    min_pa: int = 150
    input_features: List[str] = None
    output_features: List[str] = None  # If None, same as input_features
    train_ratio: float = 0.7
    valid_ratio: float = 0.2
    random_seed: int = 42
    
    def __post_init__(self):
        if self.input_features is None:
            # No default features - must be specified explicitly by each model config
            raise ValueError("input_features must be provided explicitly for each model type")
        # If output_features not specified, default to input_features
        if self.output_features is None:
            self.output_features = self.input_features

class SequenceHandler:
    """Handles creation and padding of sequences for LSTM input"""
    def __init__(self, seq_length: int, feature_dim: int):
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.pad_value = 0
        
    def create_sequence(self, player_data: pd.DataFrame, input_features: List[str]) -> Tuple[np.ndarray, torch.Tensor]:
        available_seasons = len(player_data)
        
        if available_seasons >= self.seq_length:
            # Take most recent seasons
            sequence = player_data.iloc[-self.seq_length:][input_features].values
            mask = torch.ones(self.seq_length, dtype=torch.bool)
        else:
            # Create padding
            padding_size = self.seq_length - available_seasons
            real_data = player_data[input_features].values
            padding = np.full((padding_size, len(input_features)), self.pad_value)
            sequence = np.vstack([padding, real_data])
            mask = torch.zeros(self.seq_length, dtype=torch.bool)
            mask[padding_size:] = 1
            
        return sequence, mask
def prepare_sequences(df: pd.DataFrame, 
                     input_features: List[str],
                     output_features: List[str],
                     seq_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sequences for LSTM input"""
    sequences = []
    masks = []
    handler = SequenceHandler(seq_length, len(input_features))
    
    for _, player_data in df.groupby('IDfg'):
        player_data = player_data.sort_values(by='Season')
        
        for i in range(len(player_data) - 1):
            history = player_data.iloc[:i+1]
            target = player_data.iloc[i+1][output_features].values
            
            sequence, mask = handler.create_sequence(history, input_features)
            sequences.append((sequence, target))
            masks.append(mask)
    
    return sequences, masks

def calculate_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rate statistics for all model types: defense, baserunning, batting, pitching"""
    df = df.copy()
    
    # Configuration for rate stats - easily extensible
    rate_stat_configs = {
        # Defensive stats (per 150 games)
        'defensive_per_150': {
            'condition': lambda df: 'G' in df.columns and any(col in df.columns for col in ['DRS', 'OAA', 'FRM', 'RngR', 'ErrR', 'DPR', 'UZR', 'ARM', 'rSB', 'rCERA', 'sc_total_runs', 'sc_range_runs', 'sc_arm_runs', 'sc_dp_runs', 'sc_framing_runs', 'sc_throwing_runs', 'sc_blocking_runs']),
            'stats': ['DRS', 'OAA', 'FRM', 'RngR', 'ErrR', 'DPR', 'UZR', 'ARM', 'rSB', 'rCERA', 'sc_total_runs', 'sc_range_runs', 'sc_arm_runs', 'sc_dp_runs', 'sc_framing_runs', 'sc_throwing_runs', 'sc_blocking_runs'],
            'denominator': 'G',
            'multiplier': 150,
            'suffix': '/150'
        },
        
        # Baserunning stats (per 150 games)
        'baserunning_per_150': {
            'condition': lambda df: any(col in df.columns for col in ['wSB', 'SB', 'CS', 'sc_baserunning_runner_runs_tot', 'sc_baserunning_runner_runs_XB', 'sc_baserunning_runner_runs_SBX']),
            'stats': ['wSB', 'SB', 'CS', 'sc_baserunning_runner_runs_tot', 'sc_baserunning_runner_runs_XB', 'sc_baserunning_runner_runs_SBX'],
            'denominator': 'G',
            'multiplier': 150,
            'suffix': '_rate'
        },
        
        # Batting stats (per game)
        'batting_per_game': {
            'condition': lambda df: 'G' in df.columns and any(col in df.columns for col in ['HR', '2B', '3B', 'RBI', 'R']),
            'stats': ['HR', '2B', '3B', 'RBI', 'R'],
            'denominator': 'G', 
            'multiplier': 1,
            'suffix': '_rate'
        },
        
        # Add new rate stat types here easily:
        # 'batting_per_pa': {
        #     'condition': lambda df: 'PA' in df.columns and any(col in df.columns for col in ['BB', 'K', 'SF']),
        #     'stats': ['BB', 'K', 'SF'],
        #     'denominator': 'PA',
        #     'multiplier': 1,
        #     'suffix': '_per_pa'
        # },
        
        # 'pitching_per_inning': {
        #     'condition': lambda df: 'IP' in df.columns and any(col in df.columns for col in ['H', 'ER', 'BB', 'K']),
        #     'stats': ['H', 'ER', 'BB', 'K'],
        #     'denominator': 'IP',
        #     'multiplier': 9,  # per 9 innings
        #     'suffix': '_per9'
        # }
    }
    
    # Apply each rate stat configuration
    for config_name, config in rate_stat_configs.items():
        if config['condition'](df):
            for stat in config['stats']:
                rate_col = f"{stat}{config['suffix']}"
                
                # Only calculate if stat exists and rate column doesn't already exist
                if stat in df.columns and rate_col not in df.columns:
                    denominator = config['denominator']
                    multiplier = config['multiplier']
                    
                    # Avoid division by zero
                    df[rate_col] = np.where(
                        df[denominator] > 0, 
                        (df[stat] / df[denominator]) * multiplier, 
                        0
                    )
    
    # Replace infinities and NaN with 0
    # Get all possible rate column suffixes dynamically
    rate_suffixes = ['_rate', '/150', '_per_pa', '_per9', '_pct']
    rate_columns = [col for col in df.columns if any(col.endswith(suffix) for suffix in rate_suffixes)]
    df[rate_columns] = df[rate_columns].replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def validate_features(df: pd.DataFrame, features: List[str]) -> None:
    """Validate that all required features exist in dataframe"""
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

def load_and_validate_data(file_path: str, config: DataConfig) -> pd.DataFrame:
    """Load data and perform initial validation"""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        # Calculate rate stats before validation
        df = calculate_rate_stats(df)
        # Then validate all features including new rate stats
        validate_features(df, config.input_features)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def apply_model_specific_filters(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Apply model-specific filtering logic"""
    logger.info(f"Applying {model_type} specific filters")
    
    if model_type == 'defense_infield':
        # Filter for infield positions
        infield_positions = ['1B', '2B', '3B', 'SS']
        df = df[df['Pos'].isin(infield_positions)]
        logger.info(f"Filtered to {len(df)} infield position records")
        
    elif model_type == 'defense_outfield':
        # Filter for outfield positions
        outfield_positions = ['LF', 'CF', 'RF']
        df = df[df['Pos'].isin(outfield_positions)]
        logger.info(f"Filtered to {len(df)} outfield position records")
        
    elif model_type == 'defense_catcher':
        # Filter for catcher position
        df = df[df['Pos'] == 'C']
        logger.info(f"Filtered to {len(df)} catcher position records")
        
    elif model_type == 'pitcher_sp':
        # Filter for starting pitchers (GS_rate >= 0.8)
        if 'GS' in df.columns and 'G' in df.columns:
            df['GS_rate'] = np.where(df['G'] > 0, df['GS'] / df['G'], 0)
            df = df[df['GS_rate'] >= 0.8]
            logger.info(f"Filtered to {len(df)} starting pitcher records")
        
    elif model_type == 'pitcher_rp':
        # Filter for relief pitchers (GS_rate < 0.8)
        if 'GS' in df.columns and 'G' in df.columns:
            df['GS_rate'] = np.where(df['G'] > 0, df['GS'] / df['G'], 0)
            df = df[df['GS_rate'] < 0.8]
            logger.info(f"Filtered to {len(df)} relief pitcher records")
            
    # For baserunning and batter, no additional filtering needed
    
    return df
def split_data(sequences: List[Tuple], 
               masks: List[torch.Tensor],
               game_weights: List[np.ndarray],
               train_ratio: float = 0.7,
               valid_ratio: float = 0.2) -> Tuple:
    """Split data into train, validation and test sets with game weights"""
    # Validate ratios
    if not 0 < train_ratio + valid_ratio < 1:
        raise ValueError("Train and validation ratios must sum to less than 1")
    
    logger.info("Splitting data into train, validation, and test sets")
    n = len(sequences)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    valid_size = int(n * valid_ratio)
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]
    
    # Split sequences, masks, and game weights
    train_data = ([sequences[i] for i in train_indices], 
                  [masks[i] for i in train_indices],
                  [game_weights[i] for i in train_indices])
    valid_data = ([sequences[i] for i in valid_indices], 
                  [masks[i] for i in valid_indices],
                  [game_weights[i] for i in valid_indices])
    test_data = ([sequences[i] for i in test_indices], 
                 [masks[i] for i in test_indices],
                 [game_weights[i] for i in test_indices])
    
    logger.info(f"Split sizes - Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}")
    
    return train_data, valid_data, test_data
    
    return train_data, valid_data, test_data
def filter_data(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Filter data based on configuration"""
    logger.info("Filtering data...")
    initial_size = len(df)
    
    # Filter by season
    df = df[df['Season'] >= config.start_season]
    
    # Use appropriate minimum threshold based on available columns
    if 'PA' in df.columns:
        # Batting/baserunning data - use plate appearances
        df = df[df['PA'] >= config.min_pa]
        logger.info(f"Filtered by min PA ({config.min_pa})")
    elif 'Inn' in df.columns:
        # Fielding data - use innings (treat min_pa as min_innings)
        df = df[df['Inn'] >= config.min_pa]  # reusing min_pa as min_innings
        logger.info(f"Filtered by min innings ({config.min_pa})")
    elif 'IP' in df.columns:
        # Pitching data - use innings pitched (treat min_pa as min_ip)  
        df = df[df['IP'] >= config.min_pa]  # reusing min_pa as min_ip
        logger.info(f"Filtered by min IP ({config.min_pa})")
    elif 'G' in df.columns:
        # Fallback to games if no other volume stat available
        df = df[df['G'] >= 10]  # minimum 10 games
        logger.info("Filtered by min games (10)")
    
    # Calculate rate statistics
    df = calculate_rate_stats(df)
    
    # Drop NaN values in input features early
    df = df.dropna(subset=config.input_features)
    
    # Log statistics before filtering
    logger.info("NaN counts before filtering:")
    for col in config.input_features:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"{col}: {nan_count} NaN values")
            
    logger.info(f"Filtered from {initial_size} to {len(df)} rows")
    return df

def convert_column_types(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Convert columns to float32 for LSTM compatibility"""
    logger.info("Converting column types...")
    
    for col in features:
        try:
            # Convert percentage strings to floats if needed
            if df[col].dtype == object and df[col].str.contains('%').any():
                df[col] = df[col].str.rstrip('%').astype('float32') / 100
            else:
                df[col] = df[col].astype('float32')
        except Exception as e:
            logger.error(f"Error converting column {col}: {str(e)}")
            raise
            
    return df

from sklearn.preprocessing import MinMaxScaler

def load_and_extend_scaler(
    pretrain_scaler_path: str,
    classical_features: List[str],
    statcast_features: List[str]
) -> MinMaxScaler:
    """
    Load pre-training scaler and extend it for fine-tuning with Statcast features.
    
    Args:
        pretrain_scaler_path: Path to pre-trained scaler
        classical_features: List of classical features (from pre-training)
        statcast_features: List of new Statcast features to add
        
    Returns:
        Extended scaler that can handle classical + Statcast features
    """
    logger.info(f"Loading pre-training scaler from {pretrain_scaler_path}")
    pretrain_scaler = joblib.load(pretrain_scaler_path)
    
    # Get dimensions
    n_classical = len(classical_features)
    n_classical_with_career = n_classical * 2  # Classical + vs_career
    n_statcast = len(statcast_features)
    n_total = n_classical + n_statcast
    n_total_with_career = n_total * 2  # All features + vs_career
    
    logger.info(f"Extending scaler: {n_classical} → {n_total} base features")
    logger.info(f"Total with career features: {n_classical_with_career} → {n_total_with_career}")
    
    # Create new scaler with extended dimensions
    extended_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Initialize with dummy data to set up the scaler structure
    # We'll copy the actual parameters below
    dummy_data = np.zeros((1, n_total_with_career))
    extended_scaler.fit(dummy_data)
    
    # Copy classical feature parameters from pre-trained scaler
    extended_scaler.data_min_[:n_classical_with_career] = pretrain_scaler.data_min_
    extended_scaler.data_max_[:n_classical_with_career] = pretrain_scaler.data_max_
    extended_scaler.data_range_[:n_classical_with_career] = pretrain_scaler.data_range_
    extended_scaler.scale_[:n_classical_with_career] = pretrain_scaler.scale_
    extended_scaler.min_[:n_classical_with_career] = pretrain_scaler.min_
    
    # Initialize Statcast feature parameters (will be updated during fine-tuning)
    # Place Statcast features after classical features but before vs_career features
    statcast_start_idx = n_classical
    statcast_end_idx = n_classical + n_statcast
    
    # Set reasonable defaults for Statcast features (will be refined during fit)
    extended_scaler.data_min_[statcast_start_idx:statcast_end_idx] = 0
    extended_scaler.data_max_[statcast_start_idx:statcast_end_idx] = 1
    extended_scaler.data_range_[statcast_start_idx:statcast_end_idx] = 1
    extended_scaler.scale_[statcast_start_idx:statcast_end_idx] = 2  # For (-1, 1) range
    extended_scaler.min_[statcast_start_idx:statcast_end_idx] = -1
    
    # Set defaults for Statcast vs_career features
    statcast_career_start = n_total + n_classical
    statcast_career_end = n_total + n_classical + n_statcast
    extended_scaler.data_min_[statcast_career_start:statcast_career_end] = -1
    extended_scaler.data_max_[statcast_career_start:statcast_career_end] = 1
    extended_scaler.data_range_[statcast_career_start:statcast_career_end] = 2
    extended_scaler.scale_[statcast_career_start:statcast_career_end] = 1
    extended_scaler.min_[statcast_career_start:statcast_career_end] = -0.5
    
    logger.info("Scaler extended successfully - Statcast features initialized")
    
    return extended_scaler

def scale_features(df: pd.DataFrame, 
                  features: List[str], 
                  scaler: Optional[MinMaxScaler] = None,
                  model_type: str = "baserunning",
                  mode: str = "pretrain",
                  pretrain_scaler_path: Optional[str] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale features using MinMaxScaler and add player-specific normalized features.
    
    Args:
        df: DataFrame with features
        features: List of feature columns to scale
        scaler: Existing scaler (optional)
        model_type: Type of model (for scaler filename)
        mode: 'pretrain' or 'finetune'
        pretrain_scaler_path: Path to pre-trained scaler (required for fine-tuning)
    
    Returns:
        Scaled DataFrame and scaler
    """
    
    # Calculate career averages - position-specific for defense models
    if model_type.startswith('defense'):
        # Defense models: position-specific career averages (like notebook)
        player_stats = df.groupby(['IDfg', 'Pos'])[features].transform('mean')
        logger.info(f"Using position-specific career averages for {model_type}")
    else:
        # Other models: player career averages
        player_stats = df.groupby('IDfg')[features].transform('mean')
        logger.info(f"Using player career averages for {model_type}")
    
    # Create deviation from career average features (like notebook)
    for feature in features:
        df[f'{feature}_vs_career'] = df[feature] - player_stats[feature]
    
    # CRITICAL FIX: Like notebook, create vs_career features but only scale/train on base features
    all_features = features + [f'{feature}_vs_career' for feature in features]
    
    # Handle scaler based on mode
    if scaler is None:
        # For fine-tuning, load and extend pre-trained scaler
        if mode == 'finetune' and pretrain_scaler_path:
            logger.info(f"Fine-tuning mode: Loading pre-trained scaler")
            # For fine-tuning, we need to identify classical vs statcast features
            # Assume features are ordered: [classical..., statcast...]
            # This will be handled by the calling code passing the right scaler
            scaler = joblib.load(pretrain_scaler_path)
            logger.info(f"Loaded pre-trained scaler from {pretrain_scaler_path}")
            
            # Fit on new data to update Statcast statistics
            scaled_data = scaler.fit_transform(df[all_features])
            logger.info("Updated scaler statistics with fine-tuning data")
        else:
            # Pre-training: Create new scaler
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(df[all_features])
            logger.info(f"Created new MinMaxScaler for {model_type} ({mode} mode)")
        
        # Save scaler
        import os
        models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(models_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        if mode == 'finetune':
            scaler_filename = os.path.join(data_dir, f'{model_type}_finetune_scaler.pkl')
        else:
            scaler_filename = os.path.join(data_dir, f'{model_type}_scaler.pkl')
        
        joblib.dump(scaler, scaler_filename)
        logger.info(f"Saved scaler to {scaler_filename}")
        logger.info(f"Scaler data range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")
    else:
        scaled_data = scaler.transform(df[all_features])
    
    # Validate scaled data
    if np.isnan(scaled_data).any():
        raise ValueError("NaN values found after scaling")
    if np.isinf(scaled_data).any():
        raise ValueError("Infinite values found after scaling")
    
    # Update DataFrame with scaled values
    scaled_df = pd.DataFrame(scaled_data, columns=all_features, index=df.index)
    df[all_features] = scaled_df
    
    return df, scaler

def prepare_sequences(df: pd.DataFrame, 
                      input_features: List[str],
                      output_features: List[str],
                      seq_length: int) -> Tuple[List, List, List]:
    """
    Create sequences for LSTM input with padding and game weights.
    Returns sequences, masks, and game weights.
    """
    sequences = []
    masks = []
    game_weights = []
    handler = SequenceHandler(seq_length, len(input_features))
    skipped_sequences = 0

    # Convert types before processing
    df = convert_column_types(df, input_features)

    # Sort by 'IDfg' and 'Season' to ensure correct order
    df = df.sort_values(['IDfg', 'Season'])

    for player_id, player_data in df.groupby('IDfg'):
        player_data = player_data.reset_index(drop=True)
        num_seasons = len(player_data)

        if num_seasons < 2:
            continue

        # Generate sequences starting from the second season
        for i in range(1, num_seasons):
            history = player_data.iloc[:i]
            target = player_data.iloc[i][output_features].values

            if history[input_features].isna().any().any() or pd.isna(target).any():
                skipped_sequences += 1
                continue

            sequence, mask = handler.create_sequence(history, input_features)

            if np.isnan(sequence).any() or np.isinf(sequence).any():
                skipped_sequences += 1
                continue

            # Calculate game weights for the sequence
            games_played = history['G'].values if 'G' in history.columns else np.ones(len(history))
            
            # Take only the last seq_length games if we have more
            if len(games_played) > seq_length:
                games_played = games_played[-seq_length:]
            
            # Pad with zeros to match sequence length
            padded_games = np.zeros(seq_length)
            padded_games[-len(games_played):] = games_played
            
            # Optional: Apply sqrt transformation to reduce extreme differences
            weights = np.sqrt(padded_games)
            # Normalize weights to prevent affecting overall loss magnitude
            weights = weights / np.mean(weights[weights > 0]) if np.any(weights > 0) else weights

            sequences.append((sequence, target))
            masks.append(mask)
            game_weights.append(weights)

    logger.info(f"Created {len(sequences)} valid sequences")
    logger.info(f"Skipped {skipped_sequences} sequences due to invalid values")

    if not sequences:
        raise ValueError("No valid sequences created after filtering")

    return sequences, masks, game_weights

def to_tensor(
    sequences: List[Tuple], 
    masks: List[torch.Tensor],
    game_weights: List[np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sequences, masks, and game weights to PyTorch tensors with validation"""
    sequences_array = np.array([s[0] for s in sequences], dtype=np.float32)
    targets_array = np.array([s[1] for s in sequences], dtype=np.float32)
    weights_array = np.array(game_weights, dtype=np.float32)
    
    # Validate arrays before conversion
    if np.isnan(sequences_array).any():
        raise ValueError("NaN values found in input sequences")
    if np.isnan(targets_array).any():
        raise ValueError("NaN values found in target values")
    if np.isnan(weights_array).any():
        raise ValueError("NaN values found in game weights")
        
    X = torch.FloatTensor(sequences_array)
    y = torch.FloatTensor(targets_array)
    masks = torch.stack(masks)
    weights = torch.FloatTensor(weights_array)
    
    logger.info(f"Tensor shapes - X: {X.shape}, y: {y.shape}, masks: {masks.shape}, weights: {weights.shape}")
    logger.info(f"Value ranges - X: [{X.min():.2f}, {X.max():.2f}], y: [{y.min():.2f}, {y.max():.2f}]")
    logger.info(f"Weight range - [{weights.min():.2f}, {weights.max():.2f}]")
    
    return X, y, masks, weights

def preprocess_data(
    file_path: str, 
    config: DataConfig, 
    model_type: str = None,
    mode: str = "pretrain",
    pretrain_scaler_path: Optional[str] = None
) -> Tuple:
    """
    Main preprocessing function with enhanced validation and game weights.
    Supports both pre-training and fine-tuning modes.
    
    Args:
        file_path: Path to data CSV
        config: DataConfig with features and parameters
        model_type: Type of model (for filtering and scaler naming)
        mode: 'pretrain' or 'finetune'
        pretrain_scaler_path: Path to pre-trained scaler (for fine-tuning)
    
    Returns:
        Tuple of training tensors
    """
    try:
        # Set random seeds
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        logger.info(f"Preprocessing data in {mode} mode")
        logger.info(f"Features: {len(config.input_features)} - {config.input_features[:3]}...")
        logger.info(f"Start season: {config.start_season}")
        
        # Load and validate raw data
        df = load_and_validate_data(file_path, config)
        
        # Apply model-specific filtering if model_type is provided
        if model_type:
            df = apply_model_specific_filters(df, model_type)
        
        # Filter and clean data
        df = filter_data(df, config)
        
        # Scale features before sequence creation
        model_name = model_type or "unknown"
        df, scaler = scale_features(
            df, 
            config.input_features, 
            model_type=model_name,
            mode=mode,
            pretrain_scaler_path=pretrain_scaler_path
        )
        
        # CRITICAL FIX: Like notebook, create vs_career features but only use base features for training
        # This matches the notebook behavior exactly
        training_features = config.input_features  # Input features (may include Statcast)
        output_features = config.output_features  # Output features (classical only for finetuning)
        
        # Create sequences with game weights
        sequences, masks, game_weights = prepare_sequences(df, training_features, output_features, config.seq_length)
        
        # Split data including game weights
        train_data, valid_data, test_data = split_data(
            sequences, masks, game_weights,
            train_ratio=config.train_ratio,
            valid_ratio=config.valid_ratio
        )
        
        # Convert to tensors (updated to handle weights)
        X_train, y_train, train_masks, train_weights = to_tensor(*train_data)
        X_valid, y_valid, valid_masks, valid_weights = to_tensor(*valid_data)
        X_test, y_test, test_masks, test_weights = to_tensor(*test_data)
        
        logger.info(f"Created datasets - Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
        logger.info(f"Input shape: {X_train.shape} (batch, seq_len, features)")
        
        return (X_train, y_train, X_valid, y_valid, X_test, y_test, 
                train_masks, valid_masks, test_masks,
                train_weights, valid_weights, test_weights)
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise
