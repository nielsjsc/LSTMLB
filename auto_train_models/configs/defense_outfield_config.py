# Defense Outfield model configuration

from core.data_processing import DataConfig

class DefenseOutfieldConfig:
    """Configuration for defense outfield model"""
    
    # Data configuration
    DATA_FILE = '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv'
    SCALER_FILE = 'data/defense_outfield_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_FILE = 'OF_fielding_model.pth'
    OUTPUT_FILE = '../data/generated/pipeline/defense_outfield_predictions.csv'
    
    # Position-specific configuration
    POSITION_GROUP = 'OUTFIELD'
    VALID_POSITIONS = ['LF', 'CF', 'RF']
    
    # Model-specific features
    INPUT_FEATURES = [
        'Age', #'Inn',#'OAA/150', 'DRS/150', 
        'sc_total_runs/150',# 'sc_range_runs/150', 'sc_arm_runs/150'
    ]
    
    # Data preprocessing config
    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=DefenseOutfieldConfig.INPUT_FEATURES,
            seq_length=3,  # OF_SEQ_LENGTH from notebook
            start_season=2002,
            min_pa=100,  # OF_MIN_INNINGS from notebook (converted to min_pa for compatibility)
            train_ratio=0.7,  # Fixed - must sum to less than 1.0
            valid_ratio=0.2,
            random_seed=42
        )
    
    # ============================================================================
    # RELIABILITY REGRESSION
    # ============================================================================
    # Bayesian shrinkage of defensive run metrics toward a 0-baseline prior,
    # weighted by innings played (sc_total_runs/150 needs ~1000 Inn to stabilize).
    # The two toggles are independent:
    #
    #   TRAINING   — applied to the historical DataFrame before the LSTM sees it.
    #   PREDICTION — applied to each player's historical sequence at inference time.
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = True
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True
    
    # Model hyperparameters (actual values used internally by notebook's ImprovedLSTM)
    # NOTE: Notebook Config shows 512/6/8 but ImprovedLSTM internally divides by 2 and hardcodes layers
    HIDDEN_SIZE = 256  
    NUM_LAYERS = 2  
    NUM_HEADS = 4  
    DROPOUT = 0.15 
    BIDIRECTIONAL = True
    GRADIENT_CLIP = 1.0
    
    # Training parameters (match notebook)
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Position-specific weights for loss function (must match INPUT_FEATURES)
    FEATURE_WEIGHTS = {
        'Age': 1.0,
        #'Inn': 1.0,
        'sc_total_runs/150': 1.0
    }

