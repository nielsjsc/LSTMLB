# Starting Pitcher (SP) Configuration
#
# TRANSFER LEARNING CONFIGURATION
# ================================
# This configuration supports two-stage training:
# 1. Pre-training: Classical features (1950-2024, full history)
# 2. Fine-tuning: Classical + Statcast features (2020+, Stuff+/Location+/Pitching+)
#
# KEY RESEARCH FINDINGS (FanGraphs):
# - Stuff+ stabilizes at ~80 pitches and drives year-over-year stickiness
# - K% stabilizes at 70 BF (skill-based, fast)
# - FIP/ERA have HR components that take 1,320+ BF to stabilize (too slow)
# - Velocity (FBv) is physical and ages predictably
# - xERA is based on contact quality, not luck
#
# FEATURE TIERS:
# Tier 1 (Fast stabilization, skill-based): Stuff+, K%, GB%, FBv
# Tier 2 (Good skill signals): Location+, Pitching+, SwStr%, CSW%, xERA
# Tier 3 (Use with caution): FIP, SIERA (HR component slow to stabilize)
#
# DATA SAMPLE SIZE CONSIDERATIONS (as of Jan 2026):
# ================================================
# - Fine-tuning era: 2020-2025 (6 seasons)
# - SP with 50+ IP per season: ~315 unique pitchers
# - After sequence creation (need 2+ seasons): ~517 sequences
# - After train/valid/test split: ~361 training sequences
#
# SOLUTION: Use aggressive layer freezing to reduce trainable parameters
# - Freeze LSTM layers (learned general patterns from 75 years of data)
# - Only train output projection (~1M params instead of ~5M)
# - This gives ~1:2,800 samples:params ratio (still low, but better)

from core.data_processing import DataConfig

class PitcherSPConfig:
    """Configuration for starting pitcher model with transfer learning support"""
    
    # Data configuration
    DATA_FILE = '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv'
    SCALER_FILE = 'data/pitcher_sp_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_FILE = 'sp/pitcher_model.pth'
    OUTPUT_FILE = '../data/generated/pipeline/pitcher_sp_predictions.csv'
    
    # Role-specific configuration
    ROLE = 'SP'
    GS_RATE_THRESHOLD = 0.8
    MIN_IP_CURRENT = 25  # Minimum IP in current year to generate predictions
    
    # ============================================================================
    # TRANSFER LEARNING FEATURE SETS
    # ============================================================================
    
    # Classical features for pre-training (1950-2024, available all years)
    # Focus on metrics that stabilize quickly and represent true skill
    CLASSICAL_FEATURES = [
        'Age',
        # Skill-based (fast stabilization)
        'K%',           # 70 BF to stabilize - core skill metric
        'BB%',          # 170 BF to stabilize - command metric
        # Traditional metrics (for regression targets, not primary skill)
        'FIP',          # Fielding independent - better than ERA
        'ERA',          # Outcome metric (model learns variance)
    ]
    
    # PITCHf/x era features (2002+) - adds velocity and contact metrics
    PITCHFX_FEATURES = [
        'FBv',          # Fastball velocity - physical, ages predictably
        'SwStr%',       # Swing and miss rate - true skill
        'CSW%',         # Called strikes + whiffs - command + stuff
        'GB%',          # Ground ball rate - 70 BIP to stabilize
        'FB%',          # Fly ball rate - batted ball profile
        'Contact%',     # Contact rate - inverse of strikeout ability
        'xFIP',         # Expected FIP (normalizes HR/FB)
        'SIERA',        # Advanced run estimator
    ]
    
    # Statcast era features (2020+) - the most predictive metrics
    # Research: "Stuff+ becomes reliable 80 pitches into the season"
    # Research: "Stuff+ drives most of the season-to-season stickiness"
    STATCAST_FEATURES = [
        'Stuff+',       # Pitch quality model - stabilizes at 80 pitches!
        'Location+',    # Command model - stabilizes at ~400 pitches
        'Pitching+',    # Combined model - "out-predicts any projection system"       
    ]
    
    # Combined features for fine-tuning (classical + pitchfx + statcast)
    FINETUNE_FEATURES = CLASSICAL_FEATURES + STATCAST_FEATURES
    
    # Legacy compatibility - use classical features by default
    INPUT_FEATURES = CLASSICAL_FEATURES
    
    # ============================================================================
    # PRE-TRAINING CONFIGURATION (1950-2024, Classical only)
    # ============================================================================
    SEQ_LENGTH = 2  # season sequences for pre-training
    PRETRAIN_DATA_FILE = '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv'
    PRETRAIN_SCALER_FILE = 'data/pitcher_sp_pretrain_scaler.pkl'
    PRETRAIN_CHECKPOINT_FILE = 'sp/pitcher_sp_pretrained.pth'
    PRETRAIN_START_SEASON = 1950  # Maximum historical data for classical features
    PRETRAIN_MIN_IP = 60          # Quality threshold for starters
    
    # ============================================================================
    # FINE-TUNING CONFIGURATION (2020+, Full feature set with Stuff+)
    # ============================================================================
    # Note: Stuff+/Location+/Pitching+ only available 2020+
    #
    # LAYER FREEZING STRATEGY:
    # ========================
    # Problem: Only ~360 training sequences vs 5.3M trainable parameters
    # Solution: Aggressive layer freezing
    #
    # Option 1: FREEZE_LSTM=True, FREEZE_ATTENTION=False (current)
    #   - Trainable: input_proj + attention + output_proj = ~5.3M params
    #   - Ratio: 1:14,700 (very underfitting)
    #
    # Option 2: FREEZE_LSTM=True, FREEZE_ATTENTION=True (recommended)
    #   - Trainable: input_proj + output_proj only = ~1.08M params
    #   - Ratio: 1:3,000 (still low but more reasonable)
    #
    # The LSTM layers learned temporal patterns from 75 years of pitcher data.
    # For fine-tuning with limited Statcast data, we should preserve those
    # patterns and only adapt the input/output mappings.
    
    FINETUNE_DATA_FILE = '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv'
    FINETUNE_SCALER_FILE = 'data/pitcher_sp_finetune_scaler.pkl'
    FINETUNE_CHECKPOINT_FILE = 'sp/pitcher_sp_finetuned.pth'
    FINETUNE_START_SEASON = 2020  # Stuff+ era begins
    FINETUNE_MIN_IP = 20          # Lower threshold (less data available)
    # Finetune hyperparameters - TUNED via nested search on Pretrain Trial 25
    # Architecture inherited from pretrain: 512h/3L/2heads
    # Only LR, batch size, and freeze strategy optimized during finetune
    FINETUNE_LEARNING_RATE = 1.00e-04  # Best: 1.00e-04 (10x higher than pretrain)
    FINETUNE_BATCH_SIZE = 32           # Best: 32
    FINETUNE_FREEZE_LSTM = False       # Best: False (train all layers, not frozen)
    
    # Layer freezing configuration (for non-finetune modes)
    FREEZE_LSTM = True             # Default: freeze LSTM
    FREEZE_ATTENTION = True        # Default: freeze attention
    # Note: Nested search found freeze=False optimal for finetune
    
    # Model architecture - TUNED via backtest hyperparameter search (100 trials)
    # Trial #24: 37.90% skill score across 2023-2025 holdout years
    # This configuration directly optimizes for out-of-sample predictive performance
    HIDDEN_SIZE = 128                # Best: 512
    NUM_LAYERS = 1                 # Best: 2 (backtest-optimized)
    NUM_HEADS = 8                 # Best: 4 (backtest-optimized)
    BIDIRECTIONAL = True
    DROPOUT = 0.2603902541101231           # Best: 0.253 (backtest-optimized)
    GRADIENT_CLIP = 1.0
    
    # Training parameters - TUNED via backtest hyperparameter search (100 trials)
    # Trial #24: Achieves 37.90% average skill score (vs naive baseline)
    # Per-year: 24.87% (2023), 48.08% (2024), 40.76% (2025)
    BATCH_SIZE = 16         # Best: 32 (backtest-optimized)
    LEARNING_RATE = 0.001331121608073689       # Best: 4.19e-05 (backtest-optimized)
    WEIGHT_DECAY = 1.1527987128232402e-06        # Best: 3.70e-05 (backtest-optimized)
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 8
    
    # ============================================================================
    # DOMAIN CONSTRAINT CONFIGURATION
    # ============================================================================
    # Pitchers decline faster than hitters and have earlier peaks.
    # These constraints are calibrated for pitcher-specific aging patterns.
    
    CONSTRAINT_STRENGTH = 'medium'
    
    DOMAIN_CONSTRAINTS = {
        'mse_weight': 1.0,
        'aging_weight': 0.18,        # Stronger aging (pitchers decline faster)
        'smoothness_weight': 0.08,   # Less smoothness (pitcher stats more volatile)
        'bounds_hard_weight': 0.50,
        'bounds_soft_weight': 0.05,
        'peak_weight': 0.05,
    }
    
    # Data preprocessing config
    @staticmethod
    def get_data_config(mode='pretrain'):
        """
        Get data config for pre-training or fine-tuning
        
        Args:
            mode: 'pretrain' or 'finetune'
        """
        if mode == 'finetune':
            return DataConfig(
                input_features=PitcherSPConfig.FINETUNE_FEATURES,
                output_features=PitcherSPConfig.FINETUNE_FEATURES,
                seq_length=PitcherSPConfig.SEQ_LENGTH,
                start_season=PitcherSPConfig.FINETUNE_START_SEASON,
                min_pa=PitcherSPConfig.FINETUNE_MIN_IP,
                train_ratio=0.75,
                valid_ratio=0.24,
                random_seed=42
            )
        else:  # pretrain
            return DataConfig(
                input_features=PitcherSPConfig.INPUT_FEATURES,
                seq_length=PitcherSPConfig.SEQ_LENGTH,
                start_season=PitcherSPConfig.PRETRAIN_START_SEASON,
                min_pa=PitcherSPConfig.PRETRAIN_MIN_IP,
                train_ratio=0.8,
                valid_ratio=0.19,
                random_seed=42
            )
    

