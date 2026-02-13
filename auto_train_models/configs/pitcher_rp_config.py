# Relief Pitcher (RP) Configuration
#
# TRANSFER LEARNING CONFIGURATION
# ================================
# This configuration supports two-stage training:
# 1. Pre-training: Classical features (1950-2024, full history)
# 2. Fine-tuning: Classical + Statcast features (2020+, Stuff+/Location+/Pitching+)
#
# KEY RESEARCH FINDINGS FOR RELIEVERS (FanGraphs):
# - "Pitching+ out-predicts any current projection system for relievers"
# - Stuff+ is even MORE important for relievers (pure stuff, less sequencing)
# - Small samples make traditional stats unreliable for relievers
# - Velocity retention is key indicator of reliever longevity
#
# FEATURE TIERS:
# Tier 1 (Fast stabilization, skill-based): Stuff+, K%, SwStr%, FBv
# Tier 2 (Good skill signals): Location+, Pitching+, CSW%, xERA
# Tier 3 (Use with caution): FIP, ERA (very noisy for relievers)
#
# DATA SAMPLE SIZE CONSIDERATIONS (as of Jan 2026):
# ================================================
# - Fine-tuning era: 2020-2025 (6 seasons)
# - Relief pitchers have even MORE limited data than starters
# - Lower IP thresholds help but data is inherently noisier
#
# SOLUTION: Same as SP - aggressive layer freezing
# - Freeze LSTM + attention layers
# - Only train input/output projections

from configs.pitcher_sp_config import PitcherSPConfig
from core.data_processing import DataConfig

class PitcherRPConfig:
    """Configuration for relief pitcher model with transfer learning support"""
    
    # Data configuration
    DATA_FILE = '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv'
    SCALER_FILE = 'data/pitcher_rp_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_FILE = 'rp/pitcher_model.pth'
    OUTPUT_FILE = '../data/generated/pipeline/pitcher_rp_predictions.csv'
    
    # Role-specific configuration
    ROLE = 'RP'
    GS_RATE_THRESHOLD = 0.8
    MIN_IP_CURRENT = 15  # Minimum IP in current year to generate predictions
    
    # ============================================================================
    # RELIABILITY REGRESSION
    # ============================================================================
    # When enabled, applies Bayesian shrinkage to rate stats based on sample size.
    # Each stat is regressed toward the player's career mean (or league average
    # for rookies) proportional to how much exposure they had (BF for pitchers).
    # This reduces noise from small-sample seasons in both training and prediction.
    ENABLE_RELIABILITY_REGRESSION = True
    
    # ============================================================================
    # TRANSFER LEARNING FEATURE SETS
    # ============================================================================
    
    # Classical features for pre-training (1950-2024, available all years)
    # Focus on metrics that stabilize quickly and represent true skill
    CLASSICAL_FEATURES = [
        'Age',
        # Skill-based (fast stabilization)
        'K%',           # 70 BF to stabilize - even more important for RPs
        'BB%',          # 170 BF to stabilize - command metric
        # Traditional metrics (for regression targets)
        'FIP',          # Fielding independent - better than ERA for RPs
        'ERA',          # Highly volatile for RPs but still useful context
        # Workload
        #'IP',           # Innings context (much lower for RPs)
    ]
    
    # PITCHf/x era features (2002+) - adds velocity and contact metrics
    PITCHFX_FEATURES = [
        'FBv',          # Fastball velocity - critical for relievers
        'SwStr%',       # Swing and miss rate - pure stuff indicator
        'CSW%',         # Called strikes + whiffs - command + stuff
        'GB%',          # Ground ball rate
        'FB%',          # Fly ball rate
        'Contact%',     # Contact rate
        'xFIP',         # Expected FIP
        'SIERA',        # Advanced run estimator
    ]
    
    # Statcast era features (2020+) - the most predictive metrics
    # Research: "Pitching+ out-predicts any current projection system for relievers"
    STATCAST_FEATURES = [
        #'Stuff+',       # Pitch quality model - CRITICAL for relievers
        #'Location+',    # Command model
        'Pitching+'    # Combined model - best for relievers per research
    ]
    
    # Combined features for fine-tuning (classical + pitchfx + statcast)
    FINETUNE_FEATURES = CLASSICAL_FEATURES + STATCAST_FEATURES
    
    # Legacy compatibility - use classical features by default
    INPUT_FEATURES = CLASSICAL_FEATURES
    
    # ============================================================================
    # PRE-TRAINING CONFIGURATION (1950-2024, Classical only)
    # ============================================================================
    SEQ_LENGTH = 3  # season sequences for pre-training
    PRETRAIN_DATA_FILE = '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv'
    PRETRAIN_SCALER_FILE = 'data/pitcher_rp_pretrain_scaler.pkl'
    PRETRAIN_CHECKPOINT_FILE = 'rp/pitcher_rp_pretrained.pth'
    PRETRAIN_START_SEASON = 1950  # Maximum historical data for classical features
    PRETRAIN_MIN_IP = 20          # Low threshold for relievers
    
    # ============================================================================
    # FINE-TUNING CONFIGURATION (2020+, Full feature set with Stuff+)
    # ============================================================================
    # Note: Stuff+/Location+/Pitching+ only available 2020+
    #
    # LAYER FREEZING: Same rationale as SP config
    # Limited data requires freezing LSTM + attention
    
    FINETUNE_DATA_FILE = '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv'
    FINETUNE_SCALER_FILE = 'data/pitcher_rp_finetune_scaler.pkl'
    FINETUNE_CHECKPOINT_FILE = 'rp/pitcher_rp_finetuned.pth'
    FINETUNE_START_SEASON = 2020  # Stuff+ era begins
    FINETUNE_MIN_IP = 10          # Keep low for relievers
    FINETUNE_LEARNING_RATE = 5e-6  # Very small - only adapting projections
    
    # Layer freezing configuration
    FREEZE_LSTM = True             # Always freeze LSTM (learned from 75 years)
    FREEZE_ATTENTION = True        # Freeze attention too (limited data)
    
    # Model architecture (smaller than SP - less data, more variance)
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    NUM_HEADS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0
    GRADIENT_CLIP = 1.0
    
    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-04
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 15
    
    # ============================================================================
    # DOMAIN CONSTRAINT CONFIGURATION
    # ============================================================================
    # Relief pitchers have even more volatile stats than starters.
    # Constraints are adjusted accordingly.
    
    CONSTRAINT_STRENGTH = 'medium'
    
    DOMAIN_CONSTRAINTS = {
        'mse_weight': 1.0,
        'aging_weight': 0.18,        # Same as SP
        'smoothness_weight': 0.06,   # Less smoothness (RP very volatile)
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
                input_features=PitcherRPConfig.FINETUNE_FEATURES,
                output_features=PitcherRPConfig.FINETUNE_FEATURES,
                seq_length=4,
                start_season=PitcherRPConfig.FINETUNE_START_SEASON,
                min_pa=PitcherRPConfig.FINETUNE_MIN_IP,
                train_ratio=0.75,
                valid_ratio=0.24,
                random_seed=42
            )
        else:  # pretrain
            return DataConfig(
                input_features=PitcherRPConfig.INPUT_FEATURES,
                seq_length=PitcherRPConfig.SEQ_LENGTH,
                start_season=PitcherRPConfig.PRETRAIN_START_SEASON,
                min_pa=PitcherRPConfig.PRETRAIN_MIN_IP,
                train_ratio=0.8,
                valid_ratio=0.19,
                random_seed=42
            )
    

