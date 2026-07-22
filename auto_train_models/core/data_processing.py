# Data processing and preprocessing functions

# Core libraries
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

   
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
    
    # ===== Scaler alignment =====
    # Groups of features that must share the same min/max scaling range.
    # Equal raw values will map to equal scaled values, eliminating phantom
    # gaps caused by different data ranges (e.g. ERA max=11.27 vs FIP max=8.62).
    linked_scale_groups: List[List[str]] = None
    
    # Per-feature percentile clipping applied before scaler fitting.
    # Removes extreme outliers that compress the useful range into a tiny band.
    # Format: {'feature_name': (lower_percentile, upper_percentile)}
    stat_clip_percentiles: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.input_features is None:
            # No default features - must be specified explicitly by each model config
            raise ValueError("input_features must be provided explicitly for each model type")
        # If output_features not specified, default to input_features
        if self.output_features is None:
            self.output_features = self.input_features


def calculate_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rate statistics for all model types: defense, baserunning, batting, pitching"""
    df = df.copy()
    
    # =========================================================================
    # PITCHING COMPONENT RATE STATS (per TBF)
    # =========================================================================
    # HR% and HBP% must be computed FIRST, before the batting per-150 conversion
    # below which overwrites HR and HBP in-place (suffix='').
    # K% and BB% already come pre-computed from FanGraphs.
    #
    # Stabilization (from data analysis):
    #   HR%  stabilizes at ~1108 TBF (~258 IP)  — Y-o-Y r ≈ 0.26
    #   HBP% stabilizes at ~995 TBF  (~231 IP)  — Y-o-Y r ≈ 0.30
    if 'TBF' in df.columns:
        if 'HR' in df.columns and 'HR%' not in df.columns:
            df['HR%'] = np.where(
                df['TBF'] > 0,
                df['HR'] / df['TBF'],
                0.0
            )
        if 'HBP' in df.columns and 'HBP%' not in df.columns:
            df['HBP%'] = np.where(
                df['TBF'] > 0,
                df['HBP'] / df['TBF'],
                0.0
            )
    
    # Import the master list of batting counting stats (single source of truth)
    from core.rate_stats_config import BATTING_COUNTING_STATS as _batting_counting_stats
    
    # Configuration for rate stats - easily extensible
    rate_stat_configs = {
        # Defensive stats (per 1350 innings = per 150 games at 9 inn/game)
        # NOTE: Using Inn as denominator so the rate is consistent regardless of
        # how many innings-per-game a player averaged (pinch-defense, etc.).
        # The multiplier 1350 = 150 games × 9 inn/game, so a full-season player
        # gets a value almost identical to their raw run total, matching the
        # conventional "per 150 games" scale used everywhere else.
        #
        # min_denominator = 100 innings (~11 games): seasons below this threshold
        # are too noisy to include in LSTM sequences and are set to NaN rather
        # than an unreliable scaled value.
        'defensive_per_150': {
            'condition': lambda df: 'Inn' in df.columns and any(col in df.columns for col in ['DRS', 'OAA', 'FRM', 'RngR', 'ErrR', 'DPR', 'UZR', 'ARM', 'rSB', 'rCERA', 'sc_total_runs', 'sc_range_runs', 'sc_arm_runs', 'sc_dp_runs', 'sc_framing_runs', 'sc_throwing_runs', 'sc_blocking_runs']),
            'stats': ['DRS', 'OAA', 'FRM', 'RngR', 'ErrR', 'DPR', 'UZR', 'ARM', 'rSB', 'rCERA', 'sc_total_runs', 'sc_range_runs', 'sc_arm_runs', 'sc_dp_runs', 'sc_framing_runs', 'sc_throwing_runs', 'sc_blocking_runs'],
            'denominator': 'Inn',
            'multiplier': 1350,
            'min_denominator': 100,
            'suffix': '/150'
        },
        
        # Baserunning stats (per 150 games)
        'baserunning_per_150': {
            'condition': lambda df: any(col in df.columns for col in ['wSB', 'SB', 'CS', 'BsR', 'sc_baserunning_runner_runs_tot', 'sc_baserunning_runner_runs_XB', 'sc_baserunning_runner_runs_SBX']),
            'stats': ['wSB', 'SB', 'CS', 'BsR', 'sc_baserunning_runner_runs_tot', 'sc_baserunning_runner_runs_XB', 'sc_baserunning_runner_runs_SBX'],
            'denominator': 'G',
            'multiplier': 150,
            'suffix': '_rate'
        },
        
        # Batting stats (per 150 games) - changed from per game for easier post-processing
        # NOTE: PA is intentionally excluded — it must stay as raw count because
        # the reliability regression module uses it as the sample-size volume
        # in the Bayesian shrinkage formula. Scaling PA to per-150 would make
        # every player look like they had ~400-700 PA regardless of actual playing time.
        'batting_per_150': {
            'condition': lambda df: 'G' in df.columns and any(col in df.columns for col in _batting_counting_stats),
            'stats': _batting_counting_stats,
            'denominator': 'G', 
            'multiplier': 150,
            'suffix': ''
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
                
                # When suffix is '' the rate column name equals the raw stat name,
                # so we always overwrite (in-place per-150 scaling of counting stats).
                # For non-empty suffixes we skip columns that are already computed.
                already_exists = rate_col in df.columns and config['suffix'] != ''
                if stat in df.columns and not already_exists:
                    denominator = config['denominator']
                    multiplier = config['multiplier']

                    min_denom = config.get('min_denominator', 0)

                    if min_denom > 0:
                        # Rows below the minimum threshold get NaN — they are
                        # too noisy to use as training/prediction inputs and will
                        # be excluded from LSTM sequences rather than propagating
                        # unreliable scaled values.
                        df[rate_col] = np.where(
                            df[denominator] >= min_denom,
                            (df[stat] / df[denominator]) * multiplier,
                            np.nan
                        )
                    else:
                        # Avoid division by zero
                        df[rate_col] = np.where(
                            df[denominator] > 0,
                            (df[stat] / df[denominator]) * multiplier,
                            0
                        )
    
    # Replace infinities with 0 for all rate columns.
    # NaN is intentionally preserved for defensive /150 columns where Inn < min_denominator
    # (those are small-sample seasons that should be excluded from sequences).
    # All other rate columns get NaN→0 so downstream code sees clean data.
    rate_suffixes = ['_rate', '/150', '_per_pa', '_per9', '_pct']
    rate_columns = [col for col in df.columns if any(col.endswith(suffix) for suffix in rate_suffixes)]
    defense_rate_cols = [c for c in rate_columns if c.endswith('/150') and any(
        raw in c for raw in ['DRS', 'OAA', 'FRM', 'RngR', 'ErrR', 'DPR', 'UZR', 'ARM',
                             'rSB', 'rCERA', 'sc_total_runs', 'sc_range_runs', 'sc_arm_runs',
                             'sc_dp_runs', 'sc_framing_runs', 'sc_throwing_runs', 'sc_blocking_runs']
    )]
    other_rate_cols = [c for c in rate_columns if c not in defense_rate_cols]
    # Defensive: replace inf but keep NaN (small-sample rows stay NaN)
    df[defense_rate_cols] = df[defense_rate_cols].replace([np.inf, -np.inf], np.nan)
    # All other rate stats: replace inf and fill NaN with 0
    df[other_rate_cols] = df[other_rate_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
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
        # Check for unified pitcher model mode
        from configs.pitcher_sp_config import PitcherSPConfig
        unified = getattr(PitcherSPConfig, 'UNIFIED_PITCHER_MODEL', False)
        if unified:
            logger.info(f"UNIFIED_PITCHER_MODEL=True — using all {len(df)} pitcher records (SP+RP)")
        elif 'GS' in df.columns and 'G' in df.columns:
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
    
    # NOTE: calculate_rate_stats is NOT called here — it was already called in
    # load_and_validate_data. Calling it again would double-convert counting stats
    # that use suffix='' (HR, 2B, 3B, RBI, R) because the guard 'rate_col not in
    # df.columns' is always False when suffix is empty, so every call overwrites.
    
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


def generate_batter_names(raw_df: pd.DataFrame) -> pd.DataFrame:
    batter_names_path = Path(__file__).parent.parent.parent / 'data' / 'batter_names.csv'
    if not batter_names_path.exists():
        player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
        batter_names_path.parent.mkdir(parents=True, exist_ok=True)
        player_names.to_csv(batter_names_path, index=False)
    else:
        player_names = pd.read_csv(batter_names_path)
    return player_names

def generate_pitcher_names(raw_df: pd.DataFrame) -> pd.DataFrame:
    pitcher_names_path = Path(__file__).parent.parent.parent / 'data' / 'pitcher_names.csv'
    if not pitcher_names_path.exists():
        player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
        pitcher_names_path.parent.mkdir(parents=True, exist_ok=True)
        player_names.to_csv(pitcher_names_path, index=False)
    else:
        player_names = pd.read_csv(pitcher_names_path)
    return player_names
