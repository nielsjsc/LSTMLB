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
        #'wSB_rate',
        'sc_baserunning_runner_runs_tot_rate',
        #'sc_baserunning_runner_runs_XB_rate',
        #'sc_baserunning_runner_runs_SBX_rate',
        'SB_rate',
        'CS_rate',
        #'sc_sprint_speed'
    ]
    
    # Data preprocessing config
    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=BaserunningConfig.INPUT_FEATURES,
            seq_length=3,
            start_season=2016,
            min_pa=100,
            train_ratio=0.7,
            valid_ratio=0.29,
            random_seed=42
        )
    
    # ============================================================================
    # RELIABILITY REGRESSION
    # ============================================================================
    # Bayesian shrinkage of rate stats toward a career/league-average prior,
    # weighted by sample size (G for baserunning).  The two toggles are independent:
    #
    #   TRAINING   — applied to the historical DataFrame before the LSTM sees it.
    #   PREDICTION — applied to each player's historical sequence at inference time.
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = False
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True
    
    # Model hyperparameters (actual values used by model)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NUM_HEADS = 2
    DROPOUT = 0.2
    BIDIRECTIONAL = False
    
    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
