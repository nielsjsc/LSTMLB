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
        'Age', 'Inn',#'DRS/150',  
        'sc_total_runs/150', 'sc_framing_runs/150', 'sc_throwing_runs/150', 'sc_blocking_runs/150'
    ]
    
    # Data preprocessing config
    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=DefenseCatcherConfig.INPUT_FEATURES,
            seq_length=3,  # C_SEQ_LENGTH from notebook
            start_season=2002,
            min_pa=50,  # C_MIN_INNINGS from notebook (converted to min_pa for compatibility)
            train_ratio=0.7, 
            valid_ratio=0.2,
            random_seed=42
        )
    
    # Model hyperparameters (actual values used internally by notebook's ImprovedLSTM)
    # NOTE: Notebook Config shows 512/6/8 but ImprovedLSTM internally divides by 2 and hardcodes layers
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2  # Notebook hardcodes this in ImprovedLSTM.__init__
    NUM_HEADS = 4 # Notebook hardcodes this in attention layer
    DROPOUT = 0.15  # Notebook uses dropout/2 = 0.3/2 = 0.15
    BIDIRECTIONAL = True
    GRADIENT_CLIP = 1.0
    
    # Training parameters (match notebook)
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Position-specific weights for loss function
    FEATURE_WEIGHTS = {
        'Age': 1.0,
        #'FRM/150': 1.5,
        #'DRS/150': 1.5,
        'Inn': 1.0,
        'sc_total_runs/150': 3.5,
        'sc_framing_runs/150': 1.5,
        'sc_throwing_runs/150': 1.0,
        'sc_blocking_runs/150': 1.0
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
