import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional, List
import logging
from tqdm import tqdm
from pathlib import Path
import json
import warnings

# Suppress sklearn feature name warnings during prediction
warnings.filterwarnings('ignore', message='X does not have valid feature names')

from .data_processing import DataConfig
from .model_architecture import ImprovedLSTM

logger = logging.getLogger(__name__)


# =============================================================================
# AGING CONSTRAINT ENFORCEMENT (POST-PREDICTION)
# =============================================================================

class AgingEnforcer:
    """
    Enforces aging constraints on predictions to prevent unrealistic improvements.
    
    This is applied AFTER model prediction to ensure older players don't improve
    on metrics where decline is expected.
    """
    
    # Defensive stats where higher = better (should decline with age)
    DEFENSE_METRICS = [
        'OAA/150', 'DRS/150', 'sc_total_runs/150', 'sc_range_runs/150', 
        'sc_arm_runs/150', 'sc_dp_runs/150', 'sc_framing_runs/150',
        'sc_throwing_runs/150', 'sc_blocking_runs/150'
    ]
    
    def __init__(self, params_path: Optional[Path] = None):
        """Load aging parameters."""
        if params_path is None:
            params_path = Path(__file__).parent.parent / "analysis" / "aging_parameters.json"
        
        self.params = {}
        if params_path.exists():
            with open(params_path) as f:
                self.params = json.load(f)
            logger.info(f"Loaded aging parameters for prediction enforcement")
        else:
            logger.warning(f"Aging parameters not found at {params_path}")
    
    def get_decline_rate(self, category: str, stat: str, age: int) -> float:
        """Get expected decline rate for a stat at given age."""
        # Map category to aging params key
        cat_key = category
        if category in ['fielding_infield', 'fielding_outfield', 'fielding_catcher']:
            cat_key = category
        elif category == 'defense':
            # Try to determine from stat
            if 'framing' in stat.lower() or 'throwing' in stat.lower() or 'blocking' in stat.lower():
                cat_key = 'fielding_catcher'
            elif 'dp_runs' in stat.lower():
                cat_key = 'fielding_infield'
            else:
                cat_key = 'fielding_outfield'  # Default
        
        cat_data = self.params.get(cat_key, {})
        stat_data = cat_data.get(stat, {})
        decline_by_band = stat_data.get('decline_by_age_band', {})
        
        # Find age band
        if 21 <= age <= 25:
            band = '21-25'
        elif 26 <= age <= 30:
            band = '26-30'
        elif 31 <= age <= 35:
            band = '31-35'
        elif 36 <= age <= 40:
            band = '36-40'
        elif 41 <= age <= 45:
            band = '41-45'
        else:
            band = '41-45'  # Use oldest band for very old players
        
        band_data = decline_by_band.get(band, {})
        # Use corrected value
        decline = band_data.get('decline_per_year_corrected')
        if decline is None:
            decline = band_data.get('decline_per_year', 0.0)
        
        return decline if decline is not None else 0.0
    
    def enforce_aging(
        self, 
        predictions: List[Dict], 
        category: str,
        metrics: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Enforce aging constraints on a list of predictions.
        
        For players 30+, prevents improvement on defensive metrics.
        Instead applies expected decline based on empirical aging curves.
        
        Args:
            predictions: List of prediction dicts (must be sorted by year)
            category: 'fielding_infield', 'fielding_outfield', 'fielding_catcher', etc.
            metrics: List of metrics to enforce (defaults to DEFENSE_METRICS)
            
        Returns:
            Adjusted predictions list
        """
        if not predictions or len(predictions) < 2:
            return predictions
        
        if metrics is None:
            metrics = self.DEFENSE_METRICS
        
        # Work on a copy
        adjusted = [p.copy() for p in predictions]
        
        for i in range(1, len(adjusted)):
            prev = adjusted[i - 1]
            curr = adjusted[i]
            age = curr.get('Age', 0)
            
            # Only enforce for players 30+
            if age < 30:
                continue
            
            for metric in metrics:
                if metric not in curr or metric not in prev:
                    continue
                
                prev_val = prev[metric]
                curr_val = curr[metric]
                
                # For defense metrics (higher = better), improvement = increase
                improvement = curr_val - prev_val
                
                if improvement > 0:
                    # Get expected decline
                    expected_decline = self.get_decline_rate(category, metric, age)
                    
                    if expected_decline > 0:
                        # Should be declining, not improving
                        # Apply decline from previous value
                        curr[metric] = prev_val - expected_decline
                    else:
                        # No decline expected (rare), cap improvement at 0
                        curr[metric] = prev_val
                    
                    # Update adjusted
                    adjusted[i] = curr
        
        return adjusted


# Global enforcer instance
_aging_enforcer = None

def get_aging_enforcer() -> AgingEnforcer:
    """Get or create the global aging enforcer."""
    global _aging_enforcer
    if _aging_enforcer is None:
        _aging_enforcer = AgingEnforcer()
    return _aging_enforcer


def generate_batter_names(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Generate unique batter names from the raw dataframe"""
    player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
    return player_names


def load_model_from_checkpoint(checkpoint_path: str, data_config: DataConfig, device: torch.device) -> ImprovedLSTM:
    """Load a model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    model_config = checkpoint['config']
    
    # Check for feature mismatch
    checkpoint_input_size = model_config['input_size']
    config_input_size = len(data_config.input_features)
    if checkpoint_input_size != config_input_size:
        logger.error(f"FEATURE MISMATCH: Checkpoint expects {checkpoint_input_size} features, but config has {config_input_size} features")
        logger.error(f"Config features: {data_config.input_features}")
        raise ValueError(f"Cannot load model trained with {checkpoint_input_size} features using config with {config_input_size} features. "
                        f"Either retrain the model with current config features, or update config to match checkpoint.")
    
    # Detect the actual hidden size from the state dict
    state_dict = checkpoint['model_state_dict']
    
    # Check the input projection layer size to determine the actual hidden size used
    if 'input_projection.0.weight' in state_dict:
        actual_hidden_size = state_dict['input_projection.0.weight'].shape[0]
        logger.info(f"Detected actual hidden size from checkpoint: {actual_hidden_size}")
    else:
        # Fallback to config value
        actual_hidden_size = model_config['hidden_size']
        logger.warning("Could not detect hidden size from state dict, using config value")
    
    # Create model with the actual architecture parameters (no internal modifications anymore)
    model = ImprovedLSTM(
        input_size=model_config['input_size'],
        hidden_size=actual_hidden_size,  # Use actual hidden size directly
        num_layers=model_config['num_layers'],    
        output_size=model_config['output_size'],
        dropout=model_config['dropout'],  # Use actual dropout directly
        bidirectional=model_config.get('bidirectional', True),
        num_heads=model_config.get('num_heads', 4)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


# =============================================================================
# UNIFIED PREDICTION ENGINE
# =============================================================================

def _prepare_player_sequence(
    player_id: str,
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    input_features: List[str],
    seq_length: int,
    cutoff_year: Optional[int] = None
) -> Optional[Dict]:
    """
    Prepare player data and sequence for prediction.
    
    This is the shared preprocessing logic for batter, fielding, and baserunning predictions.
    
    Args:
        player_id: FanGraphs player ID
        raw_df: Historical player data (already filtered to this player if needed)
        player_names: DataFrame mapping IDfg to Name
        input_features: List of feature names for the model
        seq_length: Number of historical seasons to use
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1).
                    If provided, this overrides the player's actual latest_season.
        
    Returns:
        Dict with 'player_name', 'sequence', 'latest_age', 'latest_season', 'n_features'
        or None if player data is invalid
    """
    # Get player data
    player_data = raw_df[raw_df['IDfg'] == player_id].copy()
    if len(player_data) == 0:
        return None
    
    # Get player name
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        logger.warning(f"Player name not found for ID {player_id}")
        return None
    
    # Sort by season
    player_data = player_data.sort_values('Season')
    
    # Handle players with fewer seasons than seq_length by padding
    num_seasons = len(player_data)
    if num_seasons < seq_length:
        recent_data = player_data[input_features].copy()
        while len(recent_data) < seq_length:
            recent_data = pd.concat([recent_data, recent_data.iloc[-1:]], ignore_index=True)
    else:
        recent_data = player_data[input_features].iloc[-seq_length:].copy().reset_index(drop=True)
    
    # STATCAST METRIC SUBSTITUTION: Replace traditional metrics with xStats if configured
    xwoba_substituted = False
    xba_substituted = False
    xslg_substituted = False
    
    try:
        from configs.batter_config import BatterConfig
        
        # xwOBA substitution
        if (BatterConfig.USE_XWOBA_FOR_PREDICTIONS and 
            'wOBA' in input_features and 
            'xwOBA' in player_data.columns):
            original_woba = recent_data['wOBA'].values.copy()
            
            if num_seasons < seq_length:
                xwoba_values = player_data['xwOBA'].copy()
                while len(xwoba_values) < seq_length:
                    xwoba_values = pd.concat([xwoba_values.iloc[:1], xwoba_values], ignore_index=True)
                recent_data['wOBA'] = xwoba_values.values
            else:
                recent_data['wOBA'] = player_data['xwOBA'].iloc[-seq_length:].values
            
            xwoba_substituted = True
            logger.debug(f"Player {player_id}: Substituted xwOBA → wOBA position. "
                        f"Original wOBA: {original_woba}, New xwOBA: {recent_data['wOBA'].values}")
        
        # xBA substitution (AVG → xBA)
        if (BatterConfig.USE_XBA_FOR_PREDICTIONS and 
            'AVG' in input_features and 
            'xBA' in player_data.columns):
            original_avg = recent_data['AVG'].values.copy()
            
            if num_seasons < seq_length:
                xba_values = player_data['xBA'].copy()
                while len(xba_values) < seq_length:
                    xba_values = pd.concat([xba_values.iloc[:1], xba_values], ignore_index=True)
                recent_data['AVG'] = xba_values.values
            else:
                recent_data['AVG'] = player_data['xBA'].iloc[-seq_length:].values
            
            xba_substituted = True
            logger.debug(f"Player {player_id}: Substituted xBA → AVG position. "
                        f"Original AVG: {original_avg}, New xBA: {recent_data['AVG'].values}")
        
        # xSLG substitution (SLG → xSLG)
        if (BatterConfig.USE_XSLG_FOR_PREDICTIONS and 
            'SLG' in input_features and 
            'xSLG' in player_data.columns):
            original_slg = recent_data['SLG'].values.copy()
            
            if num_seasons < seq_length:
                xslg_values = player_data['xSLG'].copy()
                while len(xslg_values) < seq_length:
                    xslg_values = pd.concat([xslg_values.iloc[:1], xslg_values], ignore_index=True)
                recent_data['SLG'] = xslg_values.values
            else:
                recent_data['SLG'] = player_data['xSLG'].iloc[-seq_length:].values
            
            xslg_substituted = True
            logger.debug(f"Player {player_id}: Substituted xSLG → SLG position. "
                        f"Original SLG: {original_slg}, New xSLG: {recent_data['SLG'].values}")
                        
    except (ImportError, AttributeError):
        # Config not available, skip substitution
        pass
    
    # Check for valid data (no NaN)
    if recent_data.isna().any().any():
        return None
    
    sequence = recent_data.values.astype(np.float64)
    latest_age = recent_data['Age'].iloc[-1] if 'Age' in recent_data.columns else player_data['Age'].iloc[-1]
    
    # Use cutoff_year if provided (for consistent projection years),
    # otherwise use player's actual latest season
    if cutoff_year is not None:
        latest_season = cutoff_year
        # Adjust age if cutoff_year is ahead of player's actual latest season
        actual_latest_season = player_data['Season'].max()
        if cutoff_year > actual_latest_season:
            years_ahead = cutoff_year - actual_latest_season
            latest_age = latest_age + years_ahead
    else:
        latest_season = player_data['Season'].max()
    
    return {
        'player_name': player_name,
        'sequence': sequence,
        'latest_age': latest_age,
        'latest_season': latest_season,
        'n_features': len(input_features),
        'xwoba_substituted': xwoba_substituted,
        'xba_substituted': xba_substituted,
        'xslg_substituted': xslg_substituted
    }


def _run_prediction_loop(
    model: ImprovedLSTM,
    scaler: Any,
    sequence: np.ndarray,
    input_features: List[str],
    seq_length: int,
    future_years: int,
    latest_age: float,
    latest_season: int,
    player_name: str,
    player_id: str,
    extra_fields: Optional[Dict] = None,
    non_negative_features: Optional[List[str]] = None
) -> List[Dict]:
    """
    Run the prediction loop for multiple future years.
    
    This is the shared prediction logic for batter, fielding, and baserunning.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted scaler for features
        sequence: Initial sequence array (unscaled)
        input_features: List of feature names
        seq_length: Sequence length
        future_years: Number of years to project
        latest_age: Player's age in their last season
        latest_season: Player's last season year
        player_name: Player's name for output
        player_id: Player's IDfg for output
        extra_fields: Additional fields to add to each prediction dict (e.g., Position_Group)
        non_negative_features: Features that should be clipped to >= 0
        
    Returns:
        List of prediction dictionaries
    """
    # Scale the sequence
    try:
        sequence_scaled = scaler.transform(sequence)
    except Exception as e:
        logger.error(f"Scaling error for player {player_id}: {e}")
        return []
    
    n_features = len(input_features)
    predictions = []
    device = next(model.parameters()).device
    
    for year_offset in range(1, future_years + 1):
        year = latest_season + year_offset
        age = latest_age + year_offset
        
        # Run model prediction
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
            lengths = torch.tensor([seq_length], dtype=torch.int64).to(device)
            output = model(seq_tensor, lengths)
            pred_numpy = output.cpu().numpy()[0]
        
        # Inverse transform to get actual values
        try:
            unscaled_pred = scaler.inverse_transform(pred_numpy.reshape(1, -1))[0]
            
            # Build prediction dictionary
            prediction_dict = {
                'Name': player_name,
                'IDfg': player_id,
                'Year': year,
                'Age': age,
            }
            
            # Add extra fields if provided (e.g., Position_Group, Role)
            if extra_fields:
                prediction_dict.update(extra_fields)
            
            # Add all input features to prediction
            for i, feature in enumerate(input_features):
                if feature == 'Age':
                    prediction_dict[feature] = age
                else:
                    value = unscaled_pred[i]
                    # Apply non-negative constraint if specified
                    if non_negative_features and feature in non_negative_features:
                        value = max(0, value)
                    prediction_dict[feature] = value
            
            predictions.append(prediction_dict)
            
            # Update sequence for next prediction
            # NOTE: pred_numpy is already in scaled space (direct model output)
            # Only the age component needs to be updated with the scaled next year's age
            age_index = input_features.index('Age')
            age_update = np.zeros(n_features)
            age_update[age_index] = age + 1  # Next year's age (unscaled)
            
            # Update just the age in the already-scaled prediction
            pred_numpy[age_index] = scaler.transform(age_update.reshape(1, -1))[0][age_index]
            
            # Slide the sequence window
            sequence_scaled = np.vstack([sequence_scaled[1:], pred_numpy])
            
        except Exception as e:
            logger.error(f"Prediction error for player {player_id}, year {year}: {e}")
            break
    
    return predictions


def predict_future_stats(
    player_id: str,
    input_features: List[str],
    model: ImprovedLSTM,
    scaler: Any,
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    seq_length: int,
    future_years: int = 16,
    extra_fields: Optional[Dict] = None,
    non_negative_features: Optional[List[str]] = None,
    cutoff_year: Optional[int] = None
) -> List[Dict]:
    """
    Unified prediction function for batters, fielders, and baserunners.
    
    This replaces the separate predict_future_stats_batter, predict_future_stats_fielding,
    and predict_future_stats_baserunning functions with a single, maintainable implementation.
    
    Args:
        player_id: FanGraphs player ID
        input_features: List of feature names for the model
        model: Trained LSTM model
        scaler: Fitted scaler for features
        raw_df: Historical player data
        player_names: DataFrame mapping IDfg to Name
        seq_length: Number of historical seasons to use (from config)
        future_years: Number of years to project
        extra_fields: Additional fields to add to each prediction (e.g., {'Position_Group': 'infield'})
        non_negative_features: Features that should be clipped to >= 0 (e.g., ['SB_rate', 'CS_rate'])
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        
    Returns:
        List of prediction dictionaries
    """
    # Prepare player data and sequence
    prep = _prepare_player_sequence(player_id, raw_df, player_names, input_features, seq_length, cutoff_year)
    if prep is None:
        return []
    
    # Run prediction loop
    return _run_prediction_loop(
        model=model,
        scaler=scaler,
        sequence=prep['sequence'],
        input_features=input_features,
        seq_length=seq_length,
        future_years=future_years,
        latest_age=prep['latest_age'],
        latest_season=prep['latest_season'],
        player_name=prep['player_name'],
        player_id=player_id,
        extra_fields=extra_fields,
        non_negative_features=non_negative_features
    )


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================================================================
# These maintain the old function signatures for existing code

def predict_future_stats_batter(player_id: str, input_features: List[str], model: ImprovedLSTM,
                               scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                               seq_length: int = 5, future_years: int = 16, cutoff_year: Optional[int] = None) -> List[Dict]:
    """Predict future stats for a batter. Wrapper around unified predict_future_stats."""
    return predict_future_stats(
        player_id=player_id,
        input_features=input_features,
        model=model,
        scaler=scaler,
        raw_df=raw_df,
        player_names=player_names,
        seq_length=seq_length,
        future_years=future_years,
        cutoff_year=cutoff_year
    )


def predict_future_stats_fielding(player_id: str, input_features: List[str], model: ImprovedLSTM,
                                 scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                 position_group: str, seq_length: int = 5, future_years: int = 16, cutoff_year: Optional[int] = None) -> List[Dict]:
    """Predict future fielding stats. Wrapper around unified predict_future_stats."""
    return predict_future_stats(
        player_id=player_id,
        input_features=input_features,
        model=model,
        scaler=scaler,
        raw_df=raw_df,
        player_names=player_names,
        seq_length=seq_length,
        future_years=future_years,
        extra_fields={'Position_Group': position_group},
        cutoff_year=cutoff_year
    )


def predict_future_stats_baserunning(player_id: str, input_features: List[str], model: ImprovedLSTM,
                                    scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                    seq_length: int = 4, future_years: int = 16, cutoff_year: Optional[int] = None) -> List[Dict]:
    """Predict future baserunning stats. Wrapper around unified predict_future_stats."""
    return predict_future_stats(
        player_id=player_id,
        input_features=input_features,
        model=model,
        scaler=scaler,
        raw_df=raw_df,
        player_names=player_names,
        seq_length=seq_length,
        future_years=future_years,
        non_negative_features=['SB_rate', 'CS_rate'],
        cutoff_year=cutoff_year
    )


# =============================================================================
# PITCHER PREDICTION (kept separate due to additional complexity)
# =============================================================================

def predict_future_stats_pitcher(player_id: str, input_features: List[str], model, 
                                scaler, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                role: str, future_years: int = 16, seq_length: int = 4,
                                target_year: int = None) -> List[Dict]:
    """
    Predict future stats for a pitcher with improved injury handling.
    
    Key improvements over original:
    1. Recency-weighted career averages with IP weighting (prevents peak and small-sample inflation)
    2. Most-recent valid season substitution (not oldest peak year)
    3. Stores player context for post-processing constraints
    4. target_year parameter ensures projections start from correct year
    
    Args:
        model: Trained ImprovedLSTM model
        scaler: Fitted scaler
        target_year: The year projections should start from (e.g., 2026). 
                     If player's last season is before this, projections still start at target_year.
    """
    
    # Get initial player data
    player_data = raw_df[raw_df['IDfg'] == player_id].sort_values('Season')
    if len(player_data) < 1:
        return []
    
    # Check for required features - skip players with too many NaN values
    required_features = [f for f in input_features if f != 'Age']
    last_valid_season = player_data[player_data['IP'] >= 15].tail(1)
    if last_valid_season.empty:
        last_valid_season = player_data.tail(1)
    
    nan_count = last_valid_season[required_features].isna().sum().sum()
    if nan_count > len(required_features) * 0.3:  # More than 30% NaN
        logger.debug(f"Skipping player {player_id} - too many NaN features ({nan_count}/{len(required_features)})")
        return []
        
    # Get player info
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        return []
        
    last_season = player_data['Season'].max()
    last_age = player_data[player_data['Season'] == last_season]['Age'].iloc[0]
    
    # Determine the projection start year
    # If target_year is set, ensure projections start from there even if player missed time
    if target_year is not None and last_season < target_year:
        # Player didn't play in target_year-1 season, adjust age accordingly
        years_missed = target_year - 1 - last_season
        last_age = last_age + years_missed
        last_season = target_year - 1  # So first projection is target_year
    
    # Store player context for post-processing (IP history, velocity history, etc.)
    player_context = {
        'career_high_ip': player_data['IP'].max(),
        'recent_ip': player_data.tail(3)['IP'].mean(),  # Average of last 3 seasons
        'last_fbv': None,  # For velocity constraints
        'last_age': last_age,
        'role': role,
        'recent_surgery': False,  # Will be detected below
        'recent_performance': {},  # Will store ERA/FIP/K% for post-surgery constraints
    }
    
    # Detect recent surgery/injury by checking for large IP drop in last 1-2 seasons
    recent_surgery_detected = False
    if len(player_data) >= 2:
        recent_ip = player_data.tail(2)['IP'].values
        prior_avg = player_data.iloc[:-2]['IP'].mean() if len(player_data) > 2 else player_data.iloc[0]['IP']
        # Check if there was a major injury (< 50 IP) followed by return
        if len(recent_ip) >= 2:
            if (recent_ip[-2] < 50 and prior_avg > 100) or (recent_ip[-1] >= 80 and prior_avg > 130):
                recent_surgery_detected = True
        elif len(recent_ip) > 0 and recent_ip[-1] < 50 and prior_avg > 100:
            recent_surgery_detected = True
    
    player_context['recent_surgery'] = recent_surgery_detected
    
    # If surgery detected, store most recent VALID season's performance for constraints
    if recent_surgery_detected:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            most_recent_valid = valid_seasons.iloc[-1]
            player_context['recent_performance'] = {
                'recent_era': most_recent_valid.get('ERA', 4.5),
                'recent_fip': most_recent_valid.get('FIP', 4.5),
                'recent_k_pct': most_recent_valid.get('K%', 0.22),
                'recent_bb_pct': most_recent_valid.get('BB%', 0.09),
            }
    
    # Extract velocity from most recent VALID season for constraints
    if 'FBv' in player_data.columns:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            player_context['last_fbv'] = valid_seasons.iloc[-1]['FBv']
    if 'Stuff+' in player_data.columns:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            player_context['last_stuff'] = valid_seasons.iloc[-1]['Stuff+']
    
    # Define IP thresholds
    # SEQUENCE_THRESHOLD: Minimum IP for a season to be included in the input sequence
    # This prevents tiny samples (2-10 IP) from polluting predictions
    # QUALIFICATION_THRESHOLD: Minimum recent IP to generate predictions for this player
    sequence_ip_threshold = 50 if role == 'SP' else 30
    qualification_ip_threshold = 45 if role == 'SP' else 15
    
    # Check if player qualifies for predictions (has recent meaningful playing time)
    recent_ip = player_data.tail(2)['IP'].max()  # Best of last 2 seasons
    if recent_ip < qualification_ip_threshold:
        logger.debug(f"Skipping {player_name} - insufficient recent IP ({recent_ip:.1f} < {qualification_ip_threshold})")
        return []
    
    # Build sequence using only seasons that meet sequence threshold
    recent_seasons = player_data.tail(seq_length)
    sequence_data = []
    mask = []
    
    # Process each season
    for idx, season in recent_seasons.iterrows():
        if season['IP'] >= sequence_ip_threshold:
            # Valid season - use actual data (base features only, vs_career removed)
            base_features = season[input_features].values
            sequence_data.append(base_features)
            mask.append(1)
        else:
            # Low-IP season - SKIP it entirely, don't include in sequence
            # This prevents 2-10 IP seasons from influencing predictions
            continue
    
    # Check if we have any valid seasons at all
    if len(sequence_data) == 0:
        logger.debug(f"Skipping pitcher {player_name} - no seasons with IP >= {sequence_ip_threshold}")
        return []
    
    # Pad if not enough seasons (use earliest valid season for padding)
    if len(sequence_data) < seq_length:
        first_valid_year = sequence_data[0]
        # Pad at the beginning with the earliest valid season
        sequence_data = [first_valid_year] * (seq_length - len(sequence_data)) + sequence_data
        mask = [0] * (seq_length - len(mask)) + mask  # Padded seasons marked as invalid
    
    current_sequence = np.array(sequence_data[-seq_length:])
    mask = np.array(mask[-seq_length:], dtype=np.int64)
    
    device = next(model.parameters()).device
    predictions_list = []
    
    # Get number of base features (model was trained on base features only)
    n_features = len(input_features)
    
    # Generate predictions
    for year in range(1, future_years + 1):
        # Scale the sequence
        sequence_scaled = scaler.transform(current_sequence)
        
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
        mask_tensor = torch.LongTensor(mask).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(sequence_tensor, mask_tensor.sum(1))
            prediction = prediction.cpu().numpy()
        
        # Inverse transform to get actual values
        prediction_constrained = scaler.inverse_transform(prediction)[0]
        
        # NOTE: Physical constraints removed for consistency with backtest
        # Previously applied velocity caps, IP limits, and post-surgery constraints
        # These are now handled in post-processing if needed
        
        pred_dict = {
            'Name': player_name,
            'Year': last_season + year,  # Changed from 'Season' to 'Year' for consistency
            'Age': last_age + year,
            'Role': role,
            'IDfg': player_id
        }
        
        # Add predicted stats (except Age)
        for i, feature in enumerate(input_features):
            if feature != 'Age':
                pred_dict[feature] = prediction_constrained[i]
        
        predictions_list.append(pred_dict)
        
        # Update player_context for next year's constraints
        # The constrained velocity becomes the baseline for next year
        if 'FBv' in input_features:
            fbv_idx = input_features.index('FBv')
            player_context['last_fbv'] = prediction_constrained[fbv_idx]
        
        # Update sequence for next prediction - use constrained values for continuity
        next_sequence = prediction_constrained.copy()
        age_index = input_features.index('Age')
        next_sequence[age_index] = last_age + year + 1
        
        # Scale the next prediction
        next_sequence_scaled = scaler.transform(next_sequence.reshape(1, -1))[0]
        
        # Update sequence
        current_sequence = np.vstack([current_sequence[1:], next_sequence])
        mask = np.ones(seq_length, dtype=np.int64)  # All valid for subsequent predictions
    
    return predictions_list


def predict_all_pitchers(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame, 
    sp_model, 
    rp_model,
    sp_scaler, 
    rp_scaler, 
    sp_input_features: List[str],
    rp_input_features: List[str], 
    seq_length: int, 
    future_years: int = 16, 
    cutoff_year: int = 2024,
    sp_config = None,
    rp_config = None
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all qualified pitchers.
    
    Identifies starting and relief pitchers from the cutoff year (and previous year 
    for returning/injured players), then generates multi-year projections for each.
    
    Args:
        raw_df: Historical pitcher data with Season, IDfg, IP, G, GS columns
        player_names: DataFrame mapping IDfg to Name
        sp_model: Trained LSTM model for starting pitchers
        rp_model: Trained LSTM model for relief pitchers
        sp_scaler: Fitted scaler for SP features
        rp_scaler: Fitted scaler for RP features
        sp_input_features: List of input features for SP model
        rp_input_features: List of input features for RP model
        seq_length: Number of historical seasons used as input sequence
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        
    Returns:
        DataFrame with predictions for all pitchers, or None if no predictions generated
    """
    logger.info(f"Starting predictions for pitchers from cutoff year {cutoff_year}")
    
    # Get current year and previous year pitchers
    pitchers_current = raw_df[raw_df['Season'] == cutoff_year].copy()
    pitchers_prev = raw_df[raw_df['Season'] == cutoff_year - 1].copy()
    
    # Calculate GS rates
    pitchers_current['GS_rate'] = pitchers_current['GS'] / pitchers_current['G']
    pitchers_prev['GS_rate'] = pitchers_prev['GS'] / pitchers_prev['G']
    
    # Get minimum IP thresholds from config or use defaults
    sp_min_ip = sp_config.MIN_IP_CURRENT if sp_config and hasattr(sp_config, 'MIN_IP_CURRENT') else 25
    rp_min_ip = rp_config.MIN_IP_CURRENT if rp_config and hasattr(rp_config, 'MIN_IP_CURRENT') else 15
    
    # First determine current year roles by GS rate only
    qualified_current_sp = pitchers_current[
        (pitchers_current['IP'] >= sp_min_ip) & 
        (pitchers_current['G'] >= 6)
    ]
    qualified_current_rp = pitchers_current[
        (pitchers_current['IP'] >= rp_min_ip) & 
        (pitchers_current['G'] >= 15)
    ]
    
    # Use current year role if they appear at all
    sp_ids_current = set(qualified_current_sp[qualified_current_sp['GS_rate'] >= 0.8]['IDfg'])
    rp_ids_current = set(qualified_current_rp[qualified_current_rp['GS_rate'] < 0.8]['IDfg'])
    
    # Only look at previous year for players missing from current year
    missing_current = set(pitchers_prev['IDfg']) - set(pitchers_current['IDfg'])
    sp_ids_prev = set(pitchers_prev[
        (pitchers_prev['IDfg'].isin(missing_current)) &
        (pitchers_prev['IP'] >= sp_min_ip) & 
        (pitchers_prev['G'] >= 6) & 
        (pitchers_prev['GS_rate'] >= 0.8)
    ]['IDfg'])
    
    rp_ids_prev = set(pitchers_prev[
        (pitchers_prev['IDfg'].isin(missing_current)) &
        (pitchers_prev['IP'] >= rp_min_ip) & 
        (pitchers_prev['G'] >= 15) & 
        (pitchers_prev['GS_rate'] < 0.8)
    ]['IDfg'])
    
    # Combine IDs
    sp_ids = sp_ids_current.union(sp_ids_prev)
    rp_ids = rp_ids_current.union(rp_ids_prev)
    
    logger.info(f"Found {len(sp_ids_current)} qualified {cutoff_year} SPs and {len(sp_ids_prev)} returning/recovering SPs")
    logger.info(f"Found {len(rp_ids_current)} qualified {cutoff_year} RPs and {len(rp_ids_prev)} returning/recovering RPs")
    
    # Target year is cutoff_year + 1 (e.g., if cutoff_year=2025, projections start at 2026)
    target_year = cutoff_year + 1
    
    all_predictions = []
    
    # Predict SPs
    logger.info("Generating SP predictions...")
    for player_id in tqdm(sp_ids, desc="Starting Pitchers"):
        # Filter to historical data up to cutoff_year
        player_historical_data = raw_df[
            (raw_df['IDfg'] == player_id) & 
            (raw_df['Season'] <= cutoff_year)
        ].copy()
        
        predictions = predict_future_stats_pitcher(
            player_id=player_id,
            input_features=sp_input_features,
            model=sp_model,
            scaler=sp_scaler,
            raw_df=player_historical_data,
            player_names=player_names,
            role='SP',
            seq_length=seq_length,
            future_years=future_years,
            target_year=target_year
        )
        if predictions:
            all_predictions.extend(predictions)
            
    # Predict RPs
    logger.info("Generating RP predictions...")
    for player_id in tqdm(rp_ids, desc="Relief Pitchers"):
        # Filter to historical data up to cutoff_year
        player_historical_data = raw_df[
            (raw_df['IDfg'] == player_id) & 
            (raw_df['Season'] <= cutoff_year)
        ].copy()
        
        predictions = predict_future_stats_pitcher(
            player_id=player_id,
            input_features=rp_input_features,
            model=rp_model,
            scaler=rp_scaler,
            raw_df=player_historical_data,
            player_names=player_names,
            role='RP',
            seq_length=seq_length,
            future_years=future_years,
            target_year=target_year
        )
        if predictions:
            all_predictions.extend(predictions)
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Sort by Year, Role, and player name
        predictions_df = predictions_df.sort_values(['Year', 'Role', 'Name'], ascending=[True, True, True])
        
        return predictions_df
    else:
        logger.warning("No predictions were generated")
        return None


def predict_all_batters(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame,
    model: ImprovedLSTM, 
    scaler: Any, 
    input_features: List[str],
    seq_length: int = 5,
    future_years: int = 16, 
    cutoff_year: int = 2024,
    min_pa_current: int = 100
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all qualified batters.
    
    Identifies batters with sufficient plate appearances in the cutoff year,
    then generates multi-year projections for each.
    
    Note: If BatterConfig.USE_XWOBA_FOR_PREDICTIONS is True and xwOBA data is available,
    the model will receive xwOBA values in place of wOBA for predictions (while still
    being trained on wOBA for broader historical coverage).
    
    Args:
        raw_df: Historical batter data with Season, IDfg, PA columns
        player_names: DataFrame mapping IDfg to Name
        model: Trained LSTM model for batters
        scaler: Fitted scaler for batter features
        input_features: List of input features for the model
        seq_length: Number of historical seasons to use for predictions (from config)
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        min_pa_current: Minimum PA in cutoff year to qualify for predictions
        
    Returns:
        DataFrame with predictions for all batters, or None if no predictions generated
    """
    # Get only current year players (like fielding/baserunning)
    all_players = set(raw_df[
        (raw_df['Season'] == cutoff_year) & 
        (raw_df['PA'] >= min_pa_current)
    ]['IDfg'])
    
    
    # Check if xStat substitutions are enabled
    try:
        from configs.batter_config import BatterConfig
        if BatterConfig.USE_XWOBA_FOR_PREDICTIONS and 'xwOBA' in raw_df.columns:
            xwoba_available_count = raw_df[raw_df['IDfg'].isin(all_players)]['xwOBA'].notna().sum()
            logger.info(f"xwOBA substitution ENABLED - {xwoba_available_count} player-seasons have xwOBA data available")
        if BatterConfig.USE_XBA_FOR_PREDICTIONS and 'xBA' in raw_df.columns:
            xba_available_count = raw_df[raw_df['IDfg'].isin(all_players)]['xBA'].notna().sum()
            logger.info(f"xBA substitution ENABLED - {xba_available_count} player-seasons have xBA data available")
        if BatterConfig.USE_XSLG_FOR_PREDICTIONS and 'xSLG' in raw_df.columns:
            xslg_available_count = raw_df[raw_df['IDfg'].isin(all_players)]['xSLG'].notna().sum()
            logger.info(f"xSLG substitution ENABLED - {xslg_available_count} player-seasons have xSLG data available")
    except (ImportError, AttributeError):
        pass

    all_predictions = []
    failed_count = 0
    error_sample = []
    xwoba_substitution_count = 0
    xba_substitution_count = 0
    xslg_substitution_count = 0
    
    for player_id in tqdm(all_players, desc="Generating batter predictions"):
        try:
            # Filter to only historical data up to cutoff_year
            player_data = raw_df[
                (raw_df['IDfg'] == player_id) & 
                (raw_df['Season'] <= cutoff_year)
            ].copy()
            # Additional filtering: Remove seasons with fewer than 50 PA (matching notebook)
            player_data = player_data[player_data['PA'] >= 50].reset_index(drop=True)
            
            if len(player_data) < 1:  # Skip if no valid seasons
                continue
                
            # Check if xStat substitutions occurred for this player
            prep = _prepare_player_sequence(
                player_id=player_id,
                raw_df=player_data,
                player_names=player_names,
                input_features=input_features,
                seq_length=seq_length,
                cutoff_year=cutoff_year
            )
            
            if prep:
                if prep.get('xwoba_substituted', False):
                    xwoba_substitution_count += 1
                if prep.get('xba_substituted', False):
                    xba_substitution_count += 1
                if prep.get('xslg_substituted', False):
                    xslg_substitution_count += 1
            
            predictions = predict_future_stats_batter(
                player_id=player_id,
                input_features=input_features,
                model=model,
                scaler=scaler,
                raw_df=player_data,  # Use filtered data (up to cutoff_year)
                player_names=player_names,
                seq_length=seq_length,
                future_years=future_years,
                cutoff_year=cutoff_year  # Ensure predictions start from cutoff_year + 1
            )
            
            if predictions:
                all_predictions.extend(predictions)
                
        except Exception as e:
            failed_count += 1
            # Collect first 5 errors for detailed logging
            if len(error_sample) < 5:
                error_sample.append((player_id, str(e)))
            continue
    
    # Log xStat substitution summary
    total_players = len(all_players)
    if xwoba_substitution_count > 0:
        logger.info(f"✓ xwOBA substitution applied to {xwoba_substitution_count}/{total_players} players ({xwoba_substitution_count/total_players*100:.1f}%)")
    if xba_substitution_count > 0:
        logger.info(f"✓ xBA substitution applied to {xba_substitution_count}/{total_players} players ({xba_substitution_count/total_players*100:.1f}%)")
    if xslg_substitution_count > 0:
        logger.info(f"✓ xSLG substitution applied to {xslg_substitution_count}/{total_players} players ({xslg_substitution_count/total_players*100:.1f}%)")
    if xwoba_substitution_count == 0 and xba_substitution_count == 0 and xslg_substitution_count == 0:
        logger.info("No xStat substitutions applied (all toggles may be False or xStat data not available)")
    
    # Log error details if any failures occurred
    if failed_count > 0:
        logger.error(f"Failed to generate predictions for {failed_count} batters")
        if error_sample:
            logger.error("Sample errors (first 5):")
            for player_id, error in error_sample:
                logger.error(f"  Player {player_id}: {error}")
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df = predictions_df.sort_values(['Year', 'Name'])
        return predictions_df
    else:
        logger.warning("No batter predictions were generated")
        return None


def predict_all_fielders(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame,
    position_models: Dict[str, ImprovedLSTM], 
    position_scalers: Dict[str, Any],
    position_group_map: Dict[str, str],
    input_features_map: Dict[str, List[str]],
    seq_length_map: Dict[str, int],
    future_years: int = 16, 
    cutoff_year: int = 2025,
    use_aging_enforcer: bool = False
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all qualified fielders.
    
    Processes each position group (infield, outfield, catcher) separately using
    position-specific models, then combines results.
    
    Args:
        raw_df: Historical fielding data with Season, IDfg, Pos, Inn columns
        player_names: DataFrame mapping IDfg to Name
        position_models: Dict mapping position group to trained model
        position_scalers: Dict mapping position group to fitted scaler
        position_group_map: Dict mapping specific positions to position groups
        input_features_map: Dict mapping position group to input features
        seq_length_map: Dict mapping position group to sequence length
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        
    Returns:
        DataFrame with predictions for all fielders, or None if no predictions generated
    """
    # Define minimum innings threshold (matches notebook MIN_POSITION_INNINGS)
    MIN_POSITION_INNINGS = 50
    
    all_predictions = []
    
    # Process each position group (like the notebook does)
    for model_key, model in position_models.items():
        scaler = position_scalers[model_key]
        input_features = input_features_map[model_key]
        seq_length = seq_length_map[model_key]
        
        # Get valid positions for this position group
        if model_key == 'infield':
            valid_positions = ['1B', '2B', '3B', 'SS']
        elif model_key == 'outfield':
            valid_positions = ['LF', 'CF', 'RF']
        elif model_key == 'catcher':
            valid_positions = ['C']
        else:
            continue
        
        # Filter data for this position group (matches notebook logic)
        group_df = raw_df[raw_df['Pos'].isin(valid_positions)].copy()
        
        # Get players who played enough innings at any valid position in cutoff year
        # For each player, find their PRIMARY position (most innings played)
        players_current_all = group_df[
            (group_df['Season'] == cutoff_year) & 
            (group_df['Inn'] >= MIN_POSITION_INNINGS) &
            (group_df['Pos'].isin(valid_positions))
        ][['IDfg', 'Pos', 'Inn']].copy()
        
        # Group by player and find position with most innings
        primary_positions = players_current_all.groupby('IDfg').apply(
            lambda x: x.loc[x['Inn'].idxmax(), 'Pos']
        ).reset_index()
        primary_positions.columns = ['IDfg', 'Primary_Pos']
        
        logger.info(f"\nProcessing {model_key} - {len(primary_positions)} players with primary positions")
        
        # Generate predictions for each player at their PRIMARY position only
        for _, row in tqdm(primary_positions.iterrows(), desc=f"{model_key} predictions"):
            try:
                player_id = row['IDfg']
                primary_position = row['Primary_Pos']  # The position they played most
                
                # Filter to historical data up to cutoff_year, PRIMARY POSITION ONLY
                player_historical_data = group_df[
                    (group_df['IDfg'] == player_id) & 
                    (group_df['Season'] <= cutoff_year) &
                    (group_df['Pos'] == primary_position)  # Only this position's data
                ].copy()
                
                # Skip if not enough historical data at this position
                if len(player_historical_data) == 0:
                    continue
                
                predictions = predict_future_stats_fielding(
                    player_id=player_id,
                    input_features=input_features,
                    model=model,
                    scaler=scaler,
                    raw_df=player_historical_data,
                    player_names=player_names,
                    position_group=model_key,
                    seq_length=seq_length,
                    future_years=future_years,
                    cutoff_year=cutoff_year  # Ensure predictions start from cutoff_year + 1
                )
                
                if predictions:
                    # Add the primary position to each prediction
                    for pred in predictions:
                        pred['Pos'] = primary_position  # Use their primary position
                        pred['Position_Group'] = model_key
                    
                    all_predictions.extend(predictions)
                    
            except Exception as e:
                logger.error(f"Error predicting for fielder {player_id} at {specific_position}: {str(e)}")
                continue
    
    if all_predictions:
        # Apply aging enforcement to prevent unrealistic improvements for players 30+
        enforcer = get_aging_enforcer()
        
        # Group predictions by player and position group for enforcement
        from collections import defaultdict
        player_predictions = defaultdict(list)
        for pred in all_predictions:
            key = (pred['IDfg'], pred.get('Position_Group', 'unknown'))
            player_predictions[key].append(pred)
        
        # Enforce aging for each player
        enforced_predictions = []
        for (player_id, pos_group), preds in player_predictions.items():
            # Sort by year
            preds_sorted = sorted(preds, key=lambda x: x['Year'])
            
            # Map position group to category
            if pos_group == 'infield':
                category = 'fielding_infield'
                valid_positions = ['1B', '2B', '3B', 'SS']
            elif pos_group == 'outfield':
                category = 'fielding_outfield'
                valid_positions = ['LF', 'CF', 'RF']
            elif pos_group == 'catcher':
                category = 'fielding_catcher'
                valid_positions = ['C']
            else:
                category = 'fielding_outfield'
                valid_positions = ['LF', 'CF', 'RF']
            
            # Get the player's last actual season as baseline for first prediction
            # This is critical: compare 2025 prediction to 2024 actual, not just 2025 to 2026
            # Use only the position they're being predicted for
            first_pred = preds_sorted[0] if preds_sorted else None
            if first_pred and 'Pos' in first_pred:
                player_position = first_pred['Pos']
                player_last_actual = raw_df[
                    (raw_df['IDfg'] == player_id) & 
                    (raw_df['Season'] == cutoff_year) &
                    (raw_df['Pos'] == player_position)  # Only their primary position
                ]
            else:
                player_last_actual = raw_df[
                    (raw_df['IDfg'] == player_id) & 
                    (raw_df['Season'] == cutoff_year) &
                    (raw_df['Pos'].isin(valid_positions))
                ]
            
            if not player_last_actual.empty:
                # Create a baseline record from actual data
                actual_row = player_last_actual.iloc[0]
                baseline = {
                    'IDfg': player_id,
                    'Year': cutoff_year,  # e.g., 2024
                    'Age': actual_row.get('Age', 0),
                    'Name': preds_sorted[0].get('Name', ''),
                    'Position_Group': pos_group,
                    '_is_baseline': True  # Mark so we can remove later
                }
                # Copy defensive metrics from actual data
                for metric in enforcer.DEFENSE_METRICS:
                    if metric in actual_row.index:
                        baseline[metric] = actual_row[metric]
                
                # Prepend baseline so first prediction is compared against actual
                preds_with_baseline = [baseline] + preds_sorted
            else:
                preds_with_baseline = preds_sorted
            
            # Conditionally enforce aging constraints
            if use_aging_enforcer:
                adjusted = enforcer.enforce_aging(preds_with_baseline, category)
                # Remove baseline row (it was just for comparison)
                adjusted = [p for p in adjusted if not p.get('_is_baseline', False)]
                enforced_predictions.extend(adjusted)
            else:
                # Skip aging enforcement - use raw predictions
                # Still remove baseline row if it exists
                preds_no_baseline = [p for p in preds_with_baseline if not p.get('_is_baseline', False)]
                enforced_predictions.extend(preds_no_baseline)
        
        predictions_df = pd.DataFrame(enforced_predictions)
        
        # Clean up the _is_baseline column if it somehow persisted
        if '_is_baseline' in predictions_df.columns:
            predictions_df = predictions_df.drop(columns=['_is_baseline'])
        
        # Sort by Name, Position, and Year (matches notebook output)
        predictions_df = predictions_df.sort_values(['Name', 'Pos', 'Year'])
        
        return predictions_df
    else:
        logger.warning("No fielding predictions were generated")
        return None


def predict_all_baserunners(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame,
    model, 
    scaler, 
    input_features: List[str],
    seq_length: int = 4, 
    future_years: int = 16, 
    cutoff_year: int = 2024
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all baserunners.
    
    Identifies players from the cutoff year and generates multi-year
    baserunning projections (stolen bases, caught stealing, baserunning runs).
    
    Args:
        raw_df: Historical baserunning data with Season, IDfg columns
        player_names: DataFrame mapping IDfg to Name
        model: Trained LSTM model for baserunning
        scaler: Fitted scaler for baserunning features
        input_features: List of input features for the model
        seq_length: Number of historical seasons used as input sequence
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        
    Returns:
        DataFrame with predictions for all baserunners, or None if no predictions generated
    """
    # Get unique players from cutoff year data
    current_players = raw_df[raw_df['Season'] == cutoff_year]['IDfg'].unique()
    
    all_predictions = []
    
    logger.info(f"Found {len(current_players)} unique players in {cutoff_year} data")
    
    for player_id in tqdm(current_players, desc="Generating baserunning predictions"):
        try:
            # Filter to historical data up to cutoff_year
            player_historical_data = raw_df[
                (raw_df['IDfg'] == player_id) & 
                (raw_df['Season'] <= cutoff_year)
            ].copy()
            
            predictions = predict_future_stats_baserunning(
                player_id=player_id,
                input_features=input_features,
                model=model,
                scaler=scaler,
                raw_df=player_historical_data,
                player_names=player_names,
                seq_length=seq_length,
                future_years=future_years,
                cutoff_year=cutoff_year  # Ensure predictions start from cutoff_year + 1
            )
            
            if predictions:
                all_predictions.extend(predictions)
            else:
                # Log why no predictions were generated for the first few players
                if len(all_predictions) == 0 and len(current_players) < 10:
                    logger.warning(f"No predictions for player {player_id}")
                
        except Exception as e:
            logger.error(f"Error predicting for baserunner {player_id}: {str(e)}")
            continue
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Sort by Year and SB_rate (descending)
        predictions_df = predictions_df.sort_values(['Year', 'SB_rate'], ascending=[True, False])
        
        return predictions_df
    else:
        logger.warning("No baserunning predictions were generated")
        return None


