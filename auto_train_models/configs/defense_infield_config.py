# Defense Infield model configuration

from core.data_processing import DataConfig

class DefenseInfieldConfig:
    """Configuration for defense infield model"""
    
    # Data configuration
    DATA_FILE = '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv'
    SCALER_FILE = 'data/defense_infield_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_FILE = 'IF_fielding_model.pth'
    OUTPUT_FILE = '../data/generated/pipeline/defense_infield_predictions.csv'
    
    # Position-specific configuration
    POSITION_GROUP = 'INFIELD'
    VALID_POSITIONS = ['1B', '2B', '3B', 'SS']
    
    # Model-specific features
    INPUT_FEATURES = [
        'Age',# 'OAA/150', 'DRS/150', 'Inn',
        'sc_total_runs/150',# 'sc_range_runs/150', 'sc_arm_runs/150', 'sc_dp_runs/150'
    ]
    
    # Data preprocessing config
    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=DefenseInfieldConfig.INPUT_FEATURES,
            seq_length=3,  # IF_SEQ_LENGTH from notebook
            start_season=2002,
            min_pa=50, 
            train_ratio=0.7, 
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
    
    HIDDEN_SIZE = 256 
    NUM_LAYERS = 2  # Notebook hardcodes this in ImprovedLSTM.__init__
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
    
    # Position-specific weights for loss function
    FEATURE_WEIGHTS = {
        'Age': 1.0,
        #'OAA/150': 1.0,
        #'DRS/150': 1.5,
        #'Inn': 1.0,
        'sc_total_runs/150': 1.0,
        #'sc_range_runs/150': 1.5,
        #'sc_arm_runs/150': 1.0,
        #'sc_dp_runs/150': 1.0
    }
    
    # ============================================================================
    # DOMAIN CONSTRAINT CONFIGURATION
    # ============================================================================
    # Infielders decline more gradually than catchers/outfielders.
    # Range declines fastest, positioning/DP ability more stable.
    
    CONSTRAINT_STRENGTH = 'medium'
    
    DOMAIN_CONSTRAINTS = {
        'mse_weight': 1.0,
        'aging_weight': 0.20,        # Strong aging but less than catcher/OF
        'smoothness_weight': 0.07,   # Defensive metrics are noisy
        'bounds_hard_weight': 0.60,
        'bounds_soft_weight': 0.08,
        'peak_weight': 0.05,
    }
