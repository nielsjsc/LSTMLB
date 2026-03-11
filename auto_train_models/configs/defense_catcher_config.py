# Defense Catcher model configuration

from core.data_processing import DataConfig

class DefenseCatcherConfig:
    """Configuration for defense catcher model"""
    
    # Data configuration
    DATA_FILE = '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv'
    SCALER_FILE = 'data/defense_catcher_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_FILE = 'C_fielding_model.pth'
    OUTPUT_FILE = '../data/generated/pipeline/defense_catcher_predictions.csv'
    
    # Position-specific configuration
    POSITION_GROUP = 'CATCHER'
    VALID_POSITIONS = ['C']
    
    # Model-specific features
    INPUT_FEATURES = [
        'Age', #'Inn',#'DRS/150',  
        'sc_total_runs/150', 'sc_framing_runs/150', 'sc_throwing_runs/150', 'sc_blocking_runs/150'
    ]
    
    # Data preprocessing config
    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=DefenseCatcherConfig.INPUT_FEATURES,
            seq_length=3,  # C_SEQ_LENGTH from notebook
            start_season=2002,
            min_pa=40,  # C_MIN_INNINGS from notebook (converted to min_pa for compatibility)
            train_ratio=0.7, 
            valid_ratio=0.29,
            random_seed=0
        )
    
    # ============================================================================
    # RELIABILITY REGRESSION
    # ============================================================================
    # Bayesian shrinkage of catcher defensive metrics toward a 0-baseline prior,
    # weighted by innings (framing stabilizes at 1200 Inn, throwing/blocking at 1500).
    # The two toggles are independent:
    #
    #   TRAINING   — applied to the historical DataFrame before the LSTM sees it.
    #   PREDICTION — applied to each player's historical sequence at inference time.
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = True
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True
    
    # Model hyperparameters (actual values used internally by notebook's ImprovedLSTM)
    # NOTE: Notebook Config shows 512/6/8 but ImprovedLSTM internally divides by 2 and hardcodes layers
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2 # Notebook hardcodes this in ImprovedLSTM.__init__
    NUM_HEADS = 1 # Notebook hardcodes this in attention layer
    DROPOUT = 0.2  
    BIDIRECTIONAL = False
    GRADIENT_CLIP = 1.0
    
    # Training parameters (match notebook)
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 8
    
    # Position-specific weights for loss function
    FEATURE_WEIGHTS = {
        'Age': 1.0,
        'sc_total_runs/150': 1
    }
    
    # ============================================================================
    # DOMAIN CONSTRAINT CONFIGURATION
    # ============================================================================
    # Catchers have the fastest defensive decline due to physical demands.
    # Framing may be more stable (skill-based), but athleticism drops fast.
    
    CONSTRAINT_STRENGTH = 'medium'
    
    DOMAIN_CONSTRAINTS = {
        'mse_weight': 1.0,
        'aging_weight': 0.25,        # Strongest aging (catchers decline fastest)
        'smoothness_weight': 0.07,   # Defensive metrics are noisy
        'bounds_hard_weight': 0.60,
        'bounds_soft_weight': 0.08,
        'peak_weight': 0.05,
    }
