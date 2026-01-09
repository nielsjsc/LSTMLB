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
        'Age', 'OAA/150', 'DRS/150', 'Inn',
        'sc_total_runs/150', 'sc_range_runs/150', 'sc_arm_runs/150', 'sc_dp_runs/150'
    ]
    
    # Data preprocessing config
    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=DefenseInfieldConfig.INPUT_FEATURES,
            seq_length=5,  # IF_SEQ_LENGTH from notebook
            start_season=2002,
            min_pa=150,  # IF_MIN_INNINGS from notebook (converted to min_pa for compatibility)
            train_ratio=0.7, 
            valid_ratio=0.2,
            random_seed=42
        )
    
    # Model hyperparameters (actual values used internally by notebook's ImprovedLSTM)
    # NOTE: Notebook Config shows 512/6/8 but ImprovedLSTM internally divides by 2 and hardcodes layers
    HIDDEN_SIZE = 128 
    NUM_LAYERS = 2  # Notebook hardcodes this in ImprovedLSTM.__init__
    NUM_HEADS = 4  # Notebook hardcodes this in attention layer
    DROPOUT = 0.15  # Notebook uses dropout/2 = 0.3/2 = 0.15
    BIDIRECTIONAL = True
    GRADIENT_CLIP = 1.0
    
    # Training parameters (match notebook)
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Position-specific weights for loss function
    FEATURE_WEIGHTS = {
        'Age': 2.0,
        'OAA/150': 1.0,
        'DRS/150': 1.5,
        'Inn': 1.0,
        'sc_total_runs/150': 1.5,
        'sc_range_runs/150': 1.5,
        'sc_arm_runs/150': 1.0,
        'sc_dp_runs/150': 1.0
    }
