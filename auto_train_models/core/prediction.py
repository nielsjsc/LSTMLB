import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from tqdm import tqdm

from .data_processing import DataConfig
from .model_architecture import ImprovedLSTM

logger = logging.getLogger(__name__)


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


def is_valid_season(season_data, pitcher_type):
    """Check if season meets IP threshold"""
    ip_threshold = 30 if pitcher_type == 'SP' else 15
    return season_data['IP'] >= ip_threshold

def find_nearest_valid_season(player_data, current_idx, pitcher_type):
    """Find nearest valid season for padding"""
    seasons = player_data.sort_values('Season')
    current_season = seasons.iloc[current_idx]
    
    # Search forward
    for idx in range(current_idx + 1, len(seasons)):
        if is_valid_season(seasons.iloc[idx], pitcher_type):
            return seasons.iloc[idx]
            
    # Search backward
    for idx in range(current_idx - 1, -1, -1):
        if is_valid_season(seasons.iloc[idx], pitcher_type):
            return seasons.iloc[idx]
            
    return current_season  # If no valid season found

def predict_future_stats_pitcher(player_id: str, input_features: List[str], model: ImprovedLSTM, 
                                scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                role: str, future_years: int = 16, seq_length: int = 4) -> List[Dict]:
    """Predict future stats for a pitcher - matches notebook functionality with vs_career features"""
    
    # Get initial player data
    player_data = raw_df[raw_df['IDfg'] == player_id].sort_values('Season')
    if len(player_data) < 1:
        return []
        
    # Get player info
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        return []
        
    last_season = player_data['Season'].max()
    last_age = player_data[player_data['Season'] == last_season]['Age'].iloc[0]
    
    # Calculate career averages from all player data (matching training approach)
    career_stats = player_data[input_features].mean()
    
    # Define IP threshold
    ip_threshold = 30 if role == 'SP' else 15
    
    # Build sequence checking IP thresholds
    recent_seasons = player_data.tail(seq_length)
    sequence_data = []
    mask = []
    
    # Process each season
    for idx, season in recent_seasons.iterrows():
        if season['IP'] >= ip_threshold:
            # Create base features
            base_features = season[input_features].values
            # Create vs_career features (deviation from career average)
            vs_career_features = base_features - career_stats.values
            # Combine: base + vs_career
            combined_features = np.concatenate([base_features, vs_career_features])
            sequence_data.append(combined_features)
            mask.append(1)
        else:
            # Find nearest valid season (forward then backward search)
            valid_seasons = player_data[player_data['IP'] >= ip_threshold]
            if not valid_seasons.empty:
                # Get nearest valid season's stats
                valid_stats = valid_seasons.iloc[0][input_features].values
                # Maintain correct age
                age_idx = input_features.index('Age')
                valid_stats[age_idx] = season['Age']
                # Create vs_career features
                vs_career_features = valid_stats - career_stats.values
                combined_features = np.concatenate([valid_stats, vs_career_features])
                sequence_data.append(combined_features)
            else:
                # If no valid seasons, use current season
                base_features = season[input_features].values
                vs_career_features = base_features - career_stats.values
                combined_features = np.concatenate([base_features, vs_career_features])
                sequence_data.append(combined_features)
            mask.append(0)
    
    # Pad if not enough seasons
    if len(sequence_data) < seq_length:
        first_year = sequence_data[0] if sequence_data else None
        if first_year is None:
            base_features = player_data.iloc[0][input_features].values
            vs_career_features = base_features - career_stats.values
            first_year = np.concatenate([base_features, vs_career_features])
        sequence_data = [first_year] * (seq_length - len(sequence_data)) + sequence_data
        mask = [0] * (seq_length - len(mask)) + mask
    
    current_sequence = np.array(sequence_data[-seq_length:])
    mask = np.array(mask[-seq_length:], dtype=np.int64)
    
    # CRITICAL FIX: Skip players with all invalid seasons (mask sum = 0)
    # This can happen for players who never reached IP threshold
    if mask.sum() == 0:
        logger.warning(f"Skipping pitcher {player_name} - no valid seasons with IP >= {ip_threshold}")
        return []
    
    device = next(model.parameters()).device
    predictions_list = []
    
    # Get number of base features (model was trained on base features only)
    n_features = len(input_features)
    
    # Generate predictions
    for year in range(1, future_years + 1):
        # Scale the full sequence (base + vs_career features)
        sequence_scaled = scaler.transform(current_sequence)
        
        # CRITICAL: Model was trained on base features only, so truncate
        sequence_scaled_base = sequence_scaled[:, :n_features]
        
        sequence_tensor = torch.FloatTensor(sequence_scaled_base).unsqueeze(0).to(device)
        mask_tensor = torch.LongTensor(mask).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(sequence_tensor, mask_tensor.sum(1))
            prediction = prediction.cpu().numpy()
        
        # Reconstruct full feature vector for inverse scaling
        # Create dummy vs_career features (will be replaced)
        full_prediction = np.concatenate([prediction[0], np.zeros(n_features)])
        
        # Inverse transform to get actual values
        prediction_unscaled_full = scaler.inverse_transform(full_prediction.reshape(1, -1))[0]
        
        # Extract base features
        prediction_unscaled = prediction_unscaled_full[:n_features]
        
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
                pred_dict[feature] = prediction_unscaled[i]
        
        predictions_list.append(pred_dict)
        
        # Update sequence for next prediction
        next_sequence_base = prediction_unscaled.copy()
        age_index = input_features.index('Age')
        next_sequence_base[age_index] = last_age + year + 1
        
        # Create vs_career features for next iteration
        next_sequence_vs_career = next_sequence_base - career_stats.values
        next_sequence_full = np.concatenate([next_sequence_base, next_sequence_vs_career])
        
        # Update sequence
        current_sequence = np.vstack([current_sequence[1:], next_sequence_full])
        mask = np.ones(seq_length, dtype=np.int64)  # All valid for subsequent predictions
    
    return predictions_list


def predict_future_stats_batter(player_id: str, input_features: List[str], model: ImprovedLSTM,
                               scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                               future_years: int = 16) -> List[Dict]:
    """Predict future stats for a batter - matches original notebook functionality with enhanced features"""
    
    # Get player data
    player_data = raw_df[raw_df['IDfg'] == player_id].copy()
    if len(player_data) == 0:
        return []
    
    # Get player name
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        logger.warning(f"Player name not found for ID {player_id}")
        return []
    
    # Sort by season
    player_data = player_data.sort_values('Season')
    
    # Handle players with less than 3 seasons (like the notebook)
    num_seasons = len(player_data)
    if num_seasons < 3:
        # For players with 1-2 seasons, pad with their available data
        recent_data = player_data[input_features].copy()
        while len(recent_data) < 3:
            # Duplicate their most recent season
            recent_data = pd.concat([recent_data, recent_data.iloc[-1:]])
    else:
        # Use last 3 seasons for established players
        recent_data = player_data[input_features].iloc[-3:].copy()
    
    # Check for valid data
    if recent_data.isna().any().any():
        return []
    
    # Calculate career stats from available data (matching training approach)
    career_stats = player_data[input_features].mean()
    
    # CRITICAL FIX: Match notebook's enhanced features approach exactly
    # Create enhanced features with career stats (like notebook does)
    enhanced_features = []
    for idx, row in recent_data.iterrows():
        career_dev = row - career_stats  # Direct subtraction like notebook
        combined = np.concatenate([row.values, career_dev.values])  # Simple concatenation like notebook
        enhanced_features.append(combined)

    sequence = np.array(enhanced_features)
    # Scale using the 26-feature scaler (but will only use first 13 for model)
    sequence_scaled = scaler.transform(sequence)
    
    # CRITICAL FIX: Like notebook, truncate to only base features for model input
    n_features = len(input_features)  # 13 base features only
    sequence_scaled = sequence_scaled[:, :n_features]  # Use only first 13 features for model
    
    predictions = []
    latest_age = recent_data['Age'].iloc[-1] if 'Age' in recent_data.columns else player_data['Age'].iloc[-1]
    latest_season = player_data['Season'].max()
    
    # Predict for each future year
    for year_offset in range(1, future_years + 1):
        year = latest_season + year_offset
        age = latest_age + year_offset
        
        # Predict using the model
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)
            # Calculate lengths properly
            lengths = torch.tensor([3], dtype=torch.int64)
            
            # Move tensors to the same device as model
            device = next(model.parameters()).device
            seq_tensor = seq_tensor.to(device)
            lengths = lengths.to(device)
            
            output = model(seq_tensor, lengths)
            pred_numpy = output.cpu().numpy()[0]
        
        # Inverse transform with proper handling (matching training approach)
        try:
            # CRITICAL FIX: Match notebook's inverse transform approach exactly
            n_features = len(input_features)  # 13 base features
            scaler_dim = scaler.n_features_in_  # Should be 26 (13 base + 13 vs_career)
            
            # Pad predictions to match scaler dimensions (like notebook)
            pred_padded = np.pad(pred_numpy, (0, scaler_dim - n_features), 'constant')
            
            # Inverse transform using padded prediction
            unscaled_pred = scaler.inverse_transform([pred_padded])[0][:n_features]
            
            # Create prediction dictionary with all batter features
            prediction_dict = {
                'Name': player_name,
                'IDfg': player_id,
                'Year': year,
                'Age': age,
            }
            
            # Add all input features to prediction
            for i, feature in enumerate(input_features):
                if feature == 'Age':
                    prediction_dict[feature] = age
                else:
                    prediction_dict[feature] = unscaled_pred[i]
            
            predictions.append(prediction_dict)
            
            # CRITICAL FIX: Update sequence for next prediction (matching notebook exactly)
            age_index = input_features.index('Age')
            # Create age update with proper scaling (like notebook)
            age_update = [0] * scaler_dim  # Initialize with zeros
            age_update[age_index] = age    # Set the age
            
            # Scale the age update and use it to update pred_numpy
            pred_numpy[age_index] = scaler.transform([age_update])[0][age_index]
            sequence_scaled = np.vstack([sequence_scaled[1:], pred_numpy])
            
        except Exception as e:
            logger.error(f"Batter prediction error for player {player_id}, year {year}: {e}")
            break
    
    return predictions


def predict_future_stats_fielding(player_id: str, input_features: List[str], model: ImprovedLSTM,
                                 scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                 position_group: str, seq_length: int = 5, future_years: int = 16) -> List[Dict]:
    """Predict future fielding stats - matches notebook functionality with enhanced features"""
    
    # Get player data
    player_data = raw_df[raw_df['IDfg'] == player_id].copy()
    if len(player_data) == 0:
        return []
    
    # Get player name
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        logger.warning(f"Player name not found for fielder ID {player_id}")
        return []
    
    # Sort by season
    player_data = player_data.sort_values('Season')
    
    # Handle players with less than seq_length seasons
    num_seasons = len(player_data)
    if num_seasons < seq_length:
        # For players with insufficient seasons, pad with their available data
        recent_data = player_data[input_features].copy()
        while len(recent_data) < seq_length:
            # Duplicate their most recent season
            recent_data = pd.concat([recent_data, recent_data.iloc[-1:]])
    else:
        # Use last seq_length seasons for established players
        recent_data = player_data[input_features].iloc[-seq_length:].copy()
    
    # Check for valid data
    if recent_data.isna().any().any():
        return []
    
    # Create a temporary DataFrame to match training preprocessing
    temp_df = recent_data.copy()
    temp_df = temp_df.reset_index(drop=True)
    
    # Add IDfg and Pos for groupby (required by scale_features function for defense models)
    temp_df['IDfg'] = player_id
    temp_df['Pos'] = position_group  # Use the actual position for position-specific career averages
    
    # CRITICAL FIX: Calculate position-specific career averages from ALL player data (matching training)
    # Use all historical data to calculate career averages, not just recent 3 seasons
    career_data = player_data[input_features + ['IDfg', 'Pos']].copy()
    
    # Calculate position-specific career averages (this replicates defense model scale_features logic)
    # Group by IDfg and Pos to get position-specific career averages
    player_pos_stats = career_data.groupby(['IDfg', 'Pos'])[input_features].mean()
    
    # For prediction, we need to handle the fact that a player might play multiple positions
    # Use the position_group to determine which positions to average over
    if position_group == 'infield':
        relevant_positions = ['1B', '2B', '3B', 'SS']
    elif position_group == 'outfield':
        relevant_positions = ['LF', 'CF', 'RF']
    elif position_group == 'catcher':
        relevant_positions = ['C']
    else:
        # Default to all positions for this player
        relevant_positions = career_data['Pos'].unique()
    
    # Get career averages for relevant positions
    player_career_means = {}
    for feature in input_features:
        position_values = []
        for pos in relevant_positions:
            if (player_id, pos) in player_pos_stats.index:
                position_values.append(player_pos_stats.loc[(player_id, pos), feature])
        
        if position_values:
            player_career_means[feature] = np.mean(position_values)
        else:
            # Fallback to overall player average
            player_career_means[feature] = career_data[feature].mean()
    
    # Now apply these career averages to the recent 3 seasons
    for feature in input_features:
        temp_df[f'{feature}_vs_career'] = temp_df[feature] - player_career_means[feature]
    
    # Combine original and new features (this replicates scale_features logic)
    all_features = input_features + [f'{feature}_vs_career' for feature in input_features]
    
    # Scale using the same approach as training
    try:
        sequence_scaled = scaler.transform(temp_df[all_features].values)
    except Exception as e:
        logger.error(f"Scaling error for fielder {player_id}: {e}")
        return []
    
    # CRITICAL FIX: The models were trained on ONLY the base features, not vs_career features
    # Extract only the base features for the model (first len(input_features) columns)
    n_features = len(input_features)
    sequence_scaled = sequence_scaled[:, :n_features]  # Keep only base features
    
    predictions = []
    latest_age = recent_data['Age'].iloc[-1] if 'Age' in recent_data.columns else player_data['Age'].iloc[-1]
    latest_season = player_data['Season'].max()
    
    # Predict for each future year
    for year_offset in range(1, future_years + 1):
        year = latest_season + year_offset
        age = latest_age + year_offset
        
        # Predict using the model
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)
            # Calculate lengths properly using seq_length parameter
            lengths = torch.tensor([seq_length], dtype=torch.int64)
            
            # Move tensors to the same device as model
            device = next(model.parameters()).device
            seq_tensor = seq_tensor.to(device)
            lengths = lengths.to(device)
            
            output = model(seq_tensor, lengths)
            pred_numpy = output.cpu().numpy()[0]
        
        # Inverse transform with proper handling (matching training approach)
        try:
            # CRITICAL FIX: Model outputs only base features, but scaler expects all features (base + vs_career)
            # We need to reconstruct the full feature vector for inverse scaling
            n_features = len(input_features)
            
            # Pad the prediction with zeros for vs_career features to match scaler expectations
            full_pred = np.zeros(scaler.n_features_in_)
            full_pred[:n_features] = pred_numpy  # Set base features
            # vs_career features stay as zeros (reasonable for future predictions)
            
            unscaled_pred = scaler.inverse_transform(full_pred.reshape(1, -1))[0][:n_features]
            
            # Create prediction dictionary with all fielding features
            prediction_dict = {
                'Name': player_name,
                'IDfg': player_id,
                'Year': year,
                'Age': age,
                'Position_Group': position_group,
            }
            
            # Add all input features to prediction
            for i, feature in enumerate(input_features):
                if feature == 'Age':
                    prediction_dict[feature] = age
                else:
                    prediction_dict[feature] = unscaled_pred[i]
            
            predictions.append(prediction_dict)
            
            # Update sequence for next prediction (matching notebook approach)
            age_index = input_features.index('Age')
            # Create zero array of correct scaler dimension
            scaler_dim = scaler.n_features_in_
            age_update = np.zeros(scaler_dim)
            age_update[age_index] = age
            pred_numpy[age_index] = scaler.transform([age_update])[0][age_index]
            sequence_scaled = np.vstack([sequence_scaled[1:], pred_numpy])
            
        except Exception as e:
            logger.error(f"Fielding prediction error for player {player_id}, year {year}: {e}")
            break
    
    return predictions


def predict_future_stats_baserunning(player_id: str, input_features: List[str], model: ImprovedLSTM,
                                    scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                    seq_length: int = 4, future_years: int = 16) -> List[Dict]:
    """Predict future baserunning stats - matches original notebook functionality with enhanced features"""
    
    # Get player data
    player_data = raw_df[raw_df['IDfg'] == player_id].copy()
    if len(player_data) == 0:
        return []
    
    # Get player name
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        logger.warning(f"Player name not found for ID {player_id}")
        return []
    
    # Sort by season
    player_data = player_data.sort_values('Season')
    
    # Handle players with less than seq_length seasons
    num_seasons = len(player_data)
    if num_seasons < seq_length:
        # For players with insufficient seasons, pad with their available data
        recent_data = player_data[input_features].copy()
        while len(recent_data) < seq_length:
            # Duplicate their most recent season
            recent_data = pd.concat([recent_data, recent_data.iloc[-1:]], ignore_index=True)
    else:
        # Use last seq_length seasons for established players
        recent_data = player_data[input_features].iloc[-seq_length:].copy().reset_index(drop=True)
    
    # Check for valid data
    if recent_data.isna().any().any():
        return []
    
    # Calculate career stats from available data (matching training approach)
    career_stats = player_data[input_features].mean()
    
    # Create a temporary DataFrame to match training preprocessing
    temp_df = recent_data.copy()
    temp_df = temp_df.reset_index(drop=True)
    
    # Add IDfg for groupby (required by scale_features function)
    temp_df['IDfg'] = player_id
    
    # Calculate player career averages (this replicates scale_features logic)
    player_stats = temp_df.groupby('IDfg')[input_features].transform('mean')
    
    # Create deviation from career average features (this replicates scale_features logic)
    for feature in input_features:
        temp_df[f'{feature}_vs_career'] = temp_df[feature] - player_stats[feature]
    
    # Combine original and new features (this replicates scale_features logic)
    all_features = input_features + [f'{feature}_vs_career' for feature in input_features]
    
    # Scale using the same approach as training
    try:
        sequence_scaled = scaler.transform(temp_df[all_features].values)
    except Exception as e:
        logger.error(f"Scaling error for player {player_id}: {e}")
        return []
    
    # CRITICAL FIX: Model was trained on base features only, but scaler expects all features
    n_features = len(input_features)
    n_extended_features = len(all_features)  # This includes vs_career features
    
    # Extract only the base features for model input (matches training approach)
    sequence_base_features = sequence_scaled[:, :n_features]
    
    predictions = []
    latest_age = recent_data['Age'].iloc[-1] if 'Age' in recent_data.columns else player_data['Age'].iloc[-1]
    latest_season = player_data['Season'].max()
    
    # Predict for each future year
    for year_offset in range(1, future_years + 1):
        year = latest_season + year_offset
        age = latest_age + year_offset
        
        # Predict using the model (using only base features)
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_base_features).unsqueeze(0)
            # Calculate lengths properly
            lengths = torch.tensor([seq_length], dtype=torch.int64)
            
            # Move tensors to the same device as model
            device = next(model.parameters()).device
            seq_tensor = seq_tensor.to(device)
            lengths = lengths.to(device)
            
            output = model(seq_tensor, lengths)
            pred_numpy = output.cpu().numpy()[0]
        
        # Inverse transform with proper handling (matching training approach)
        try:
            # Model outputs base features (6), but scaler expects all features (12)
            # Pad with zeros for vs_career features to match scaler expectations
            if len(pred_numpy) == n_features and scaler.n_features_in_ == n_extended_features:
                # Pad the prediction to match scaler input size
                padded_pred = np.zeros(scaler.n_features_in_)
                padded_pred[:n_features] = pred_numpy
                unscaled_pred = scaler.inverse_transform(padded_pred.reshape(1, -1))[0][:n_features]
            elif len(pred_numpy) == scaler.n_features_in_:
                unscaled_pred = scaler.inverse_transform(pred_numpy.reshape(1, -1))[0][:n_features]
            else:
                logger.error(f"Model output size {len(pred_numpy)} doesn't match expected sizes")
                break
            
            # Create prediction dictionary with all predicted features
            prediction_dict = {
                'Name': player_name,
                'IDfg': player_id,
                'Year': year,
                'Age': age,
            }
            
            # Add all predicted features dynamically from input_features
            for i, feature in enumerate(input_features):
                if feature != 'Age':  # Age is already added above
                    # Apply non-negative constraint for counting stats
                    if feature in ['SB_rate', 'CS_rate']:
                        prediction_dict[feature] = max(0, unscaled_pred[i])
                    else:
                        prediction_dict[feature] = unscaled_pred[i]
            
            predictions.append(prediction_dict)
            

            age_index = input_features.index('Age')
            
            # Create next year's prediction in UNSCALED space
            next_pred_unscaled = unscaled_pred.copy()
            next_pred_unscaled[age_index] = age + 1  # Next year's age
            
            # Calculate vs_career features for the next prediction
            next_pred_vs_career = next_pred_unscaled - career_stats.values
            
            # Combine base + vs_career features
            next_pred_full = np.concatenate([next_pred_unscaled, next_pred_vs_career])
            
            # Scale the full prediction
            next_pred_scaled = scaler.transform(next_pred_full.reshape(1, -1))[0]
            
            # Update sequence_scaled
            sequence_scaled = np.vstack([sequence_scaled[1:], next_pred_scaled.reshape(1, -1)])
            
            # Extract base features for next iteration
            sequence_base_features = sequence_scaled[:, :n_features]
            
        except Exception as e:
            logger.error(f"Baserunning prediction error for player {player_id}, year {year}: {e}")
            break
    
    return predictions


def calculate_pitcher_war(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate WAR for pitchers using correct replacement level baseline"""
    
    def estimate_playing_time(row):
        if row['Role'] == 'SP':
            base_ip = 180
            scaled_ip = base_ip * (4.20/row['FIP'])
            return pd.Series({
                'IP': min(220, max(150, scaled_ip)),
                'G': 32,
                'GS': 32
            })
        else:
            base_ip = 65
            scaled_ip = base_ip * (4.20/row['FIP'])
            return pd.Series({
                'IP': min(80, max(50, scaled_ip)),
                'G': 65,
                'GS': 0
            })
    
    df[['IP', 'G', 'GS']] = df.apply(estimate_playing_time, axis=1)
    
    # Calculate WAR components
    league_fip = 4.20
    replacement_level_fip = 4.95  # Approximately 0.75 runs worse than league average
    
    # Runs above replacement level
    df['RAR'] = (replacement_level_fip - df['FIP']) * (df['IP'] / 9)
    
    # Calculate WAR
    df['WAR'] = df['RAR'] / 9.0
    
    # Cleanup and round
    df = df.drop(columns=['RAR'])
    df['WAR'] = df['WAR'].round(1)
    
    return df


def calculate_war_components(row, pos_games_map={'C': 135}, default_games=150):
    """Calculate WAR components with position-specific playing time - from batter notebook"""
    # Position-based games
    games = pos_games_map.get(row.get('Position', 'OF'), default_games)
    pa = games * 4.2

    # Hitting counting stats
    hitting_stats = {}
    for stat in ['HR', '2B', '3B', 'RBI', 'R']:
        if f'{stat}_rate' in row:
            hitting_stats[stat] = round(row[f'{stat}_rate'] * games, 1)

    # WAR components (from batter notebook)
    wOBA_scale = 1.23
    RPA = 0.117
    lg_wOBA = 0.309
    RPW = 9.8
    team = row.get('Team', None)
    
    # Ballpark factors (simplified - from notebook)
    ballpark_factors = {
        'COL': 104, 'BOS': 103, 'CIN': 102, 'TEX': 101, 'BAL': 101,
        'NYY': 100, 'MIN': 100, 'CHW': 100, 'LAA': 100, 'ATL': 100,
        'HOU': 99, 'WSN': 99, 'TB': 99, 'MIL': 99, 'TOR': 99,
        'KC': 98, 'ARI': 98, 'NYM': 98, 'SF': 98, 'PIT': 98,
        'CLE': 97, 'STL': 97, 'LAD': 97, 'CHC': 97, 'DET': 97,
        'MIA': 96, 'PHI': 96, 'OAK': 95, 'SD': 95, 'SEA': 94
    }
    
    if pd.isnull(team):
        PF = 1.0
    else:
        team_str = str(team).upper().strip()
        PF = ballpark_factors.get(team_str, 100) / 100
    
    lgPA = 186188
    wRAA = ((row['wOBA'] - lg_wOBA) / wOBA_scale) * pa
    batting_runs = wRAA + (RPA - (RPA * PF)) * pa
    
    # Get defensive and baserunning values (default to 0 if not available)
    Def = row.get('def_value', 0)  # From fielding predictions
    BsR = row.get('BsR', 0)        # From baserunning predictions
    
    Off = batting_runs + BsR
    rep_level = 570 * RPW * pa / lgPA
    
    rar = Off + Def + rep_level
    war = rar / RPW

    return war, {
        'Off': Off,
        'BsR': BsR,
        'Def': Def,
        'WAR': war,
        'PA': pa,
        'G': games,
        **hitting_stats,
        'SB': row.get('SB', 0),
        'CS': row.get('CS', 0)
    }


def calculate_batter_war_with_fielding(batter_df: pd.DataFrame, fielding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate WAR for batters using position-specific fielding data.
    
    Args:
        batter_df: Batter predictions DataFrame
        fielding_df: Position-specific fielding predictions DataFrame
        
    Returns:
        DataFrame with WAR calculations including defensive values
    """
    from .defensive_value_calculator import merge_defensive_values_with_batters
    
    # Merge defensive values from position-specific fielding data
    df_with_defense = merge_defensive_values_with_batters(batter_df, fielding_df)
    
    # Position-based games mapping
    pos_games_map = {'C': 135}
    default_games = 150
    
    # Calculate WAR components for each prediction
    for idx, row in df_with_defense.iterrows():
        _, components = calculate_war_components(
            row, 
            pos_games_map=pos_games_map,
            default_games=default_games
        )
        
        # Update the dataframe with WAR components
        for component, value in components.items():
            df_with_defense.at[idx, component] = value
    
    # Remove only intermediate rate statistics (keep the important counting stat rates)
    rate_columns_to_remove = [col for col in df_with_defense.columns 
                              if col.endswith('_rate') and 
                              col not in ['HR_rate', '2B_rate', '3B_rate', 'RBI_rate', 'R_rate']]
    df_with_defense = df_with_defense.drop(columns=rate_columns_to_remove, errors='ignore')
    
    return df_with_defense


def calculate_batter_war_with_fielding_and_baserunning(batter_df: pd.DataFrame, fielding_df: pd.DataFrame, baserunning_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate WAR for batters using position-specific fielding data and baserunning predictions.
    
    Args:
        batter_df: Batter predictions DataFrame
        fielding_df: Position-specific fielding predictions DataFrame
        baserunning_df: Baserunning predictions DataFrame
        
    Returns:
        DataFrame with WAR calculations including defensive and baserunning values
    """
    from .defensive_value_calculator import merge_defensive_values_with_batters
    
    # Merge defensive values from position-specific fielding data
    df_with_defense = merge_defensive_values_with_batters(batter_df, fielding_df)
    
    # Merge baserunning values
    logger.info("Merging baserunning values with batter predictions")
    
    # Calculate BsR from baserunning predictions
    # BsR = wSB + UBR + wGDP (from baserunning notebook methodology)
    baserunning_df = baserunning_df.copy()
    baserunning_df['BsR'] = (baserunning_df['wSB_rate'] + 
                             baserunning_df['UBR_rate'] + 
                             baserunning_df['wGDP_rate']) * 150  # Scale by games
    
    # Merge baserunning data on IDfg and Year
    df_with_baserunning = df_with_defense.merge(
        baserunning_df[['IDfg', 'Year', 'BsR', 'SB_rate', 'CS_rate']],
        on=['IDfg', 'Year'],
        how='left'
    )
    
    # Fill missing baserunning values with 0
    df_with_baserunning['BsR'] = df_with_baserunning['BsR'].fillna(0)
    df_with_baserunning['SB_rate'] = df_with_baserunning['SB_rate'].fillna(0)
    df_with_baserunning['CS_rate'] = df_with_baserunning['CS_rate'].fillna(0)
    
    logger.info(f"Successfully merged baserunning values for {df_with_baserunning['BsR'].notna().sum()} predictions")
    
    # Position-based games mapping
    pos_games_map = {'C': 135}
    default_games = 150
    
    # Calculate WAR components for each prediction
    for idx, row in df_with_baserunning.iterrows():
        _, components = calculate_war_components(
            row, 
            pos_games_map=pos_games_map,
            default_games=default_games
        )
        
        # Update the dataframe with WAR components
        for component, value in components.items():
            df_with_baserunning.at[idx, component] = value
    
    # Remove only intermediate rate statistics (keep the important counting stat rates)
    rate_columns_to_remove = [col for col in df_with_baserunning.columns 
                              if col.endswith('_rate') and 
                              col not in ['HR_rate', '2B_rate', '3B_rate', 'RBI_rate', 'R_rate', 'SB_rate', 'CS_rate']]
    df_with_baserunning = df_with_baserunning.drop(columns=rate_columns_to_remove, errors='ignore')
    
    return df_with_baserunning


def calculate_batter_war(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate WAR for batters using proper methodology from batter notebook (legacy version)"""
    
    # Position-based games mapping
    pos_games_map = {'C': 135}
    default_games = 150
    
    # Calculate WAR components for each prediction
    for idx, row in df.iterrows():
        _, components = calculate_war_components(
            row, 
            pos_games_map=pos_games_map,
            default_games=default_games
        )
        
        # Update the dataframe with WAR components
        for component, value in components.items():
            df.at[idx, component] = value
    
    # Remove only intermediate rate statistics (keep the important counting stat rates)
    rate_columns_to_remove = [col for col in df.columns 
                              if col.endswith('_rate') and 
                              col not in ['HR_rate', '2B_rate', '3B_rate', 'RBI_rate', 'R_rate']]
    df = df.drop(columns=rate_columns_to_remove, errors='ignore')
    
    return df


def predict_all_2024_pitchers(raw_df: pd.DataFrame, player_names: pd.DataFrame, 
                             sp_model: ImprovedLSTM, rp_model: ImprovedLSTM,
                             sp_scaler: Any, rp_scaler: Any, input_features: List[str],
                             seq_length: int, future_years: int = 16, cutoff_year: int = 2024) -> Optional[pd.DataFrame]:
    """Predict future years with role determination - year-agnostic version"""
    logger.info(f"Starting predictions for current and potentially injured/recovering pitchers from {cutoff_year}")
    
    # Get current year and previous year pitchers
    pitchers_current = raw_df[raw_df['Season'] == cutoff_year].copy()
    pitchers_prev = raw_df[raw_df['Season'] == cutoff_year - 1].copy()
    
    # Calculate GS rates
    pitchers_current['GS_rate'] = pitchers_current['GS'] / pitchers_current['G']
    pitchers_prev['GS_rate'] = pitchers_prev['GS'] / pitchers_prev['G']
    
    # First determine current year roles by GS rate only
    qualified_current_sp = pitchers_current[
        (pitchers_current['IP'] >= 25) & 
        (pitchers_current['G'] >= 6)
    ]
    qualified_current_rp = pitchers_current[
        (pitchers_current['IP'] >= 15) & 
        (pitchers_current['G'] >= 15)
    ]
    
    # Use current year role if they appear at all
    sp_ids_current = set(qualified_current_sp[qualified_current_sp['GS_rate'] >= 0.8]['IDfg'])
    rp_ids_current = set(qualified_current_rp[qualified_current_rp['GS_rate'] < 0.8]['IDfg'])
    
    # Only look at previous year for players missing from current year
    missing_current = set(pitchers_prev['IDfg']) - set(pitchers_current['IDfg'])
    sp_ids_prev = set(pitchers_prev[
        (pitchers_prev['IDfg'].isin(missing_current)) &
        (pitchers_prev['IP'] >= 25) & 
        (pitchers_prev['G'] >= 6) & 
        (pitchers_prev['GS_rate'] >= 0.8)
    ]['IDfg'])
    
    rp_ids_prev = set(pitchers_prev[
        (pitchers_prev['IDfg'].isin(missing_current)) &
        (pitchers_prev['IP'] >= 15) & 
        (pitchers_prev['G'] >= 15) & 
        (pitchers_prev['GS_rate'] < 0.8)
    ]['IDfg'])
    
    # Combine IDs
    sp_ids = sp_ids_current.union(sp_ids_prev)
    rp_ids = rp_ids_current.union(rp_ids_prev)
    
    logger.info(f"Found {len(sp_ids_current)} qualified {cutoff_year} SPs and {len(sp_ids_prev)} returning/recovering SPs")
    logger.info(f"Found {len(rp_ids_current)} qualified {cutoff_year} RPs and {len(rp_ids_prev)} returning/recovering RPs")
    
    all_predictions = []
    
    # Predict SPs
    logger.info("Generating SP predictions...")
    for player_id in tqdm(sp_ids, desc="Starting Pitchers"):
        predictions = predict_future_stats_pitcher(
            player_id=player_id,
            input_features=input_features,
            model=sp_model,
            scaler=sp_scaler,
            raw_df=raw_df,
            player_names=player_names,
            role='SP',
            seq_length=seq_length,
            future_years=future_years
        )
        if predictions:
            all_predictions.extend(predictions)
            
    # Predict RPs
    logger.info("Generating RP predictions...")
    for player_id in tqdm(rp_ids, desc="Relief Pitchers"):
        predictions = predict_future_stats_pitcher(
            player_id=player_id,
            input_features=input_features,
            model=rp_model,
            scaler=rp_scaler,
            raw_df=raw_df,
            player_names=player_names,
            role='RP',
            seq_length=seq_length,
            future_years=future_years
        )
        if predictions:
            all_predictions.extend(predictions)
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Calculate WAR
        predictions_df = calculate_pitcher_war(predictions_df)
        
        # Sort by Year, Role, and WAR (descending)
        predictions_df = predictions_df.sort_values(['Year', 'Role', 'WAR'], ascending=[True, True, False])
        
        return predictions_df
    else:
        logger.warning("No predictions were generated")
        return None


def predict_all_2024_batters_no_war(raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                  model: ImprovedLSTM, scaler: Any, input_features: List[str],
                                  future_years: int = 16, cutoff_year: int = 2024) -> Optional[pd.DataFrame]:
    """Generate predictions for all batters without WAR calculation - year-agnostic version"""
    
    # Get only current year players (like fielding/baserunning)
    all_players = set(raw_df[
        (raw_df['Season'] == cutoff_year) & 
        (raw_df['PA'] >= 100)  # Minimum PA threshold
    ]['IDfg'])
    
    logger.info(f"Found {len(all_players)} qualified {cutoff_year} players")
    logger.info(f"Total players to predict: {len(all_players)}")
    
    all_predictions = []
    
    for player_id in tqdm(all_players, desc="Generating batter predictions"):
        try:
            # Additional filtering: Remove seasons with fewer than 50 PA (matching notebook)
            player_data = raw_df[raw_df['IDfg'] == player_id].copy()
            player_data = player_data[player_data['PA'] >= 50].reset_index(drop=True)
            
            if len(player_data) < 1:  # Skip if no valid seasons
                continue
                
            predictions = predict_future_stats_batter(
                player_id=player_id,
                input_features=input_features,
                model=model,
                scaler=scaler,
                raw_df=player_data,  # Use filtered data
                player_names=player_names,
                future_years=future_years
            )
            
            if predictions:
                all_predictions.extend(predictions)
                
        except Exception as e:
            logger.error(f"Error predicting for batter {player_id}: {str(e)}")
            continue
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Sort by Year and player name for consistency (no WAR calculation)
        predictions_df = predictions_df.sort_values(['Year', 'Name'])
        
        return predictions_df
    else:
        logger.warning("No batter predictions were generated")
        return None


def predict_all_2024_batters(raw_df: pd.DataFrame, player_names: pd.DataFrame,
                           model: ImprovedLSTM, scaler: Any, input_features: List[str],
                           future_years: int = 16, cutoff_year: int = 2024) -> Optional[pd.DataFrame]:
    """Generate predictions for all batters - year-agnostic version"""
    
    # Get only current year players (like fielding/baserunning)
    all_players = set(raw_df[
        (raw_df['Season'] == cutoff_year) & 
        (raw_df['PA'] >= 100)  # Minimum PA threshold
    ]['IDfg'])
    
    logger.info(f"Found {len(all_players)} qualified {cutoff_year} players")
    logger.info(f"Total players to predict: {len(all_players)}")
    
    all_predictions = []
    
    for player_id in tqdm(all_players, desc="Generating batter predictions"):
        try:
            # Additional filtering: Remove seasons with fewer than 50 PA (matching notebook)
            player_data = raw_df[raw_df['IDfg'] == player_id].copy()
            player_data = player_data[player_data['PA'] >= 50].reset_index(drop=True)
            
            if len(player_data) < 1:  # Skip if no valid seasons
                continue
                
            predictions = predict_future_stats_batter(
                player_id=player_id,
                input_features=input_features,
                model=model,
                scaler=scaler,
                raw_df=player_data,  # Use filtered data
                player_names=player_names,
                future_years=future_years
            )
            
            if predictions:
                all_predictions.extend(predictions)
                
        except Exception as e:
            logger.error(f"Error predicting for batter {player_id}: {str(e)}")
            continue
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Calculate WAR
        predictions_df = calculate_batter_war(predictions_df)
        
        # Sort by Year and WAR (descending)
        predictions_df = predictions_df.sort_values(['Year', 'WAR'], ascending=[True, False])
        
        return predictions_df
    else:
        logger.warning("No batter predictions were generated")
        return None


def predict_all_2024_fielders(raw_df: pd.DataFrame, player_names: pd.DataFrame,
                            position_models: Dict[str, ImprovedLSTM], 
                            position_scalers: Dict[str, Any],
                            position_group_map: Dict[str, str],
                            input_features_map: Dict[str, List[str]],
                            seq_length_map: Dict[str, int],
                            future_years: int = 16, cutoff_year: int = 2024) -> Optional[pd.DataFrame]:
    """Generate predictions for all fielders - year-agnostic version"""
    
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
        players_current = group_df[
            (group_df['Season'] == cutoff_year) & 
            (group_df['Inn'] >= MIN_POSITION_INNINGS) &
            (group_df['Pos'].isin(valid_positions))
        ][['IDfg', 'Pos']].drop_duplicates()  # Keep player-position combinations
        
        logger.info(f"\nProcessing {model_key} - {len(players_current)} player-position combinations")
        
        # Generate predictions for each player-position combination (matches notebook exactly)
        for _, row in tqdm(players_current.iterrows(), desc=f"{model_key} predictions"):
            try:
                player_id = row['IDfg']
                specific_position = row['Pos']  # Keep the specific position (SS, 2B, etc.)
                
                predictions = predict_future_stats_fielding(
                    player_id=player_id,
                    input_features=input_features,
                    model=model,
                    scaler=scaler,
                    raw_df=group_df,
                    player_names=player_names,
                    position_group=model_key,
                    seq_length=seq_length,
                    future_years=future_years
                )
                
                if predictions:
                    # Add the specific position to each prediction (matches notebook)
                    for pred in predictions:
                        pred['Pos'] = specific_position  # Add specific position column
                        # Keep Position_Group for compatibility but Pos is the key field
                        pred['Position_Group'] = model_key
                    
                    all_predictions.extend(predictions)
                    
            except Exception as e:
                logger.error(f"Error predicting for fielder {player_id} at {specific_position}: {str(e)}")
                continue
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Sort by Name, Position, and Year (matches notebook output)
        predictions_df = predictions_df.sort_values(['Name', 'Pos', 'Year'])
        
        return predictions_df
    else:
        logger.warning("No fielding predictions were generated")
        return None


def predict_all_2024_baserunners(raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                model: ImprovedLSTM, scaler: Any, input_features: List[str],
                                seq_length: int = 4, future_years: int = 16, cutoff_year: int = 2024) -> Optional[pd.DataFrame]:
    """Generate predictions for all baserunners - year-agnostic version"""
    
    # Get unique players from cutoff year data
    current_players = raw_df[raw_df['Season'] == cutoff_year]['IDfg'].unique()
    
    all_predictions = []
    
    logger.info(f"Found {len(current_players)} unique players in {cutoff_year} data")
    
    for player_id in tqdm(current_players, desc="Generating baserunning predictions"):
        try:
            predictions = predict_future_stats_baserunning(
                player_id=player_id,
                input_features=input_features,
                model=model,
                scaler=scaler,
                raw_df=raw_df,
                player_names=player_names,
                seq_length=seq_length,
                future_years=future_years
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

