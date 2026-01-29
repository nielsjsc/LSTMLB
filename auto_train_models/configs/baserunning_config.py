# Baserunning model configuration

from core.data_processing import DataConfig

class BaserunningConfig:
    """Configuration for baserunning model"""
    
    # Data configuration
    DATA_FILE = '../data/historic_mlb/mlb_batting_data_1950_2025_with_statcast.csv'
    SCALER_FILE = 'data/baserunning_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_FILE = 'baserunning_model.pth'
    OUTPUT_FILE = '../data/generated/pipeline/baserunning_predictions.csv'
    
    # Model-specific features
    INPUT_FEATURES = [
        'Age',
        'wSB_rate',
        'sc_baserunning_runner_runs_tot_rate',
        'sc_baserunning_runner_runs_XB_rate',
        'sc_baserunning_runner_runs_SBX_rate',
        'SB_rate',
        'CS_rate',
        'sc_sprint_speed'
    ]
    
    # Data preprocessing config
    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=BaserunningConfig.INPUT_FEATURES,
            seq_length=4,
            start_season=2016,
            min_pa=150,
            train_ratio=0.7,
            valid_ratio=0.2,
            random_seed=42
        )
    
    # Model hyperparameters (actual values used by model)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NUM_HEADS = 2
    DROPOUT = 0.15
    BIDIRECTIONAL = True
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # ============================================================================
    # DOMAIN CONSTRAINT CONFIGURATION
    # ============================================================================
    # Speed/baserunning is the fastest-declining tool in baseball.
    # Peak around age 24-25, with consistent decline thereafter.
    # Sprint speed is very stable year-to-year (high skill component).
    
    CONSTRAINT_STRENGTH = 'medium'
    
    DOMAIN_CONSTRAINTS = {
        'mse_weight': 1.0,
        'aging_weight': 0.22,        # Very strong aging (speed declines fast)
        'smoothness_weight': 0.12,   # More smoothness (speed is consistent)
        'bounds_hard_weight': 0.50,
        'bounds_soft_weight': 0.05,
        'peak_weight': 0.06,
    }
