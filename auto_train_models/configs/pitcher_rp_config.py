# Relief Pitcher (RP) Configuration

from core.data_processing import DataConfig

class PitcherRPConfig:
    """Configuration for relief pitcher model"""
    
    # Data configuration
    DATA_FILE = '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv'
    SCALER_FILE = 'data/pitcher_rp_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_FILE = 'rp/pitcher_model.pth'
    OUTPUT_FILE = '../data/generated/pipeline/pitcher_rp_predictions.csv'
    
    # Role-specific configuration
    ROLE = 'RP'
    GS_RATE_THRESHOLD = 0.8
    
    # Model-specific features (exactly from pitcher.ipynb)
    INPUT_FEATURES = [
        'Age','FIP','SIERA', 'ERA','K%', 'BB%','IP'
    ]
    
    
    # Model architecture (actual values used by model)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    BIDIRECTIONAL = True
    DROPOUT = 0.05
    GRADIENT_CLIP = 1.0
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00005
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 15
    
    # Data preprocessing config
    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=PitcherRPConfig.INPUT_FEATURES,
            seq_length=4,  # SEQ_LENGTH from notebook
            start_season=1950,
            min_pa=10,  # MIN_IP threshold for relief pitchers
            train_ratio=0.8,
            valid_ratio=0.19,
            random_seed=42
        )
    
    # Model architecture config
    @staticmethod
    def get_model_config():
        return {
            'input_size': len(PitcherRPConfig.INPUT_FEATURES),
            'output_size': len(PitcherRPConfig.INPUT_FEATURES),
            'hidden_size': 512,
            'num_layers': 6,
            'num_heads': 4,
            'bidirectional': True,
            'attention_dropout': 0.1,
            'residual_dropout': 0.2,
            'layer_norm_eps': 1e-5,
            'dropout': 0.1
        }
    
    # Training configuration
    @staticmethod
    def get_training_config():
        return {
            'batch_size': 32,
            'learning_rate': 0.00005,
            'num_epochs': 50,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'warmup_epochs': 5,
            'lr_schedule': 'cosine',
            'min_lr': 1e-6,
            'lr_decay_rate': 0.1,
            'lr_patience': 5,
            'early_stopping_patience': 15,
            'early_stopping_min_delta': 0.0001,
            'diversity_alpha': 0.1,
            'consistency_beta': 0.05,
            'mixed_precision': True,
            'num_workers': 0,
            'pin_memory': True,
            'log_interval': 100,
            'checkpoint_interval': 1
        }
