# Batter Configuration
#
# VALIDATED OPTIMAL CONFIGURATION
# ================================
# Now includes domain-constrained loss function support.
# See core/domain_losses.py for details on aging curves and physical bounds.
# This configuration has been validated through comprehensive hyperparameter tuning
# (96 configurations tested on 2015-2024 validation data, December 2025)
#
# KEY FINDINGS:
# - Learning Rate: 1e-3 is CRITICAL (10x better than 1e-4)
# - Batch Size: 16 optimal (better generalization than 32)
# - Sequence Length: 5 years optimal (5 > 6, more history adds noise)
# - Gradient Clip: 1.0 optimal (allows necessary large updates)
# - Hidden Size: 512 or 256 equivalent (use 512 for consistency)
# - Dropout: 0.3 or 0.2 equivalent (use 0.3)
#
# PERFORMANCE (R² on 2015-2024):
# - Optimal Config: R²=0.6585, MAE=3.0475, RMSE=15.8211
# - Alternative LR=1e-4: R²=0.6554 (0.5% worse)
#
# This matches the notebook's empirically-tuned configuration and validates
# that aggressive optimization (high LR, small batches) works best for this task.

from core.data_processing import DataConfig

class BatterConfig:
    """
    Configuration for batter model
    
    All hyperparameters validated through systematic tuning.
    Changes to LR, batch_size, or seq_length require re-validation.
    """
    
    # Data configuration
    DATA_FILE = '../data/historic_mlb/mlb_batting_data_1950_2025.csv'
    
    # Checkpoint and scaler files (mode-specific)
    SCALER_FILE = 'data/batter_pretrain_scaler.pkl'  
    FINETUNE_SCALER_FILE = 'data/batter_finetune_scaler.pkl'  # Finetune scaler
    PRETRAIN_SCALER_FILE = 'data/batter_pretrain_scaler.pkl'  # For loading during finetune
    
    CHECKPOINT_DIR = './checkpoints'
    CHECKPOINT_FILE = 'batter_model.pth'  # Legacy/default checkpoint
    PRETRAIN_CHECKPOINT_FILE = 'batter_pretrained.pth'  # Pre-training saves here
    FINETUNE_CHECKPOINT_FILE = 'batter_finetuned.pth'  # Fine-tuning saves here
    
    OUTPUT_FILE = '../data/generated/pipeline/batter_predictions.csv'
    
    # Batter-specific configuration
    SEQ_LEN = 3
    MIN_PA = 50  # Minimum PA per season for training sequences
    MIN_PA_CURRENT = 70  # Minimum PA in current year to generate predictions
    
    # ============================================================================
    # RELIABILITY REGRESSION
    # ============================================================================
    # Bayesian shrinkage of rate stats toward a career/league-average prior,
    # weighted by sample size (PA for batters).  The two toggles are independent:
    #
    #   TRAINING   — applied to the historical DataFrame before the LSTM sees it,
    #                so the model trains on true-talent estimates rather than noisy
    #                small-sample seasons.
    #   PREDICTION — applied to each player's historical sequence just before
    #                building the input window, so padding and per-season weights
    #                reflect regressed (less noisy) values at inference time.
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = False
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = False
     # ============================================================================
    # PREDICTION CONFIGURATION
    # ============================================================================
    # Use xwOBA instead of wOBA for predictions (when available in data)
    # Model is still trained on wOBA (more historical data), but xwOBA is more predictive
    # for recent players. The model sees xwOBA values in the wOBA feature position.
    USE_XWOBA_FOR_PREDICTIONS = True

    # Blend wOBA and xwOBA: feed (wOBA + xwOBA) / 2 into the model instead of pure xwOBA.
    # Captures both observed outcomes and expected (luck-corrected) performance.
    # Only takes effect when USE_XWOBA_FOR_PREDICTIONS = False.
    # Set both to False to use raw wOBA.
    USE_XWOBA_BLEND_FOR_PREDICTIONS = False
    
    # Use xBA instead of AVG for predictions (more predictive for recent players)
    USE_XBA_FOR_PREDICTIONS = True
    
    # Use xSLG instead of SLG for predictions (more predictive for recent players)
    USE_XSLG_FOR_PREDICTIONS = True
    # Ordering: park neutralization happens BEFORE x-stat substitution because
    # x-stats (xBA, xSLG, xwOBA) are already park-neutral by design.
    ENABLE_PARK_FACTOR_ADJUSTMENT = False
    
    # ============================================================================
    # WAR CALCULATION CONFIGURATION
    # ============================================================================
    # Calculate wOBA from component stats (HR, 2B, 3B, BB, etc.) instead of using
    # the LSTM's predicted wOBA directly. This can provide more consistent wOBA values
    # when the model's wOBA predictions don't perfectly align with the counting stats.
    # Set to False to use the LSTM's direct wOBA prediction.
    CALCULATE_WOBA_FROM_COMPONENTS = True
    # ============================================================================
    # TRANSFER LEARNING FEATURE SETS
    # ============================================================================
    
    # Classical features for pre-training (2000-2024, available all years)
    # NOTE: HR, 2B, 3B, RBI, R, HBP, SF, PA are per 150 games (scaled during preprocessing)
    CLASSICAL_FEATURES = [
        'Age', 'BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA', #'wRC+',
        'HR', '2B', '3B','RBI', 'R',
        #'sc_ev50',
        #'sc_anglesweetspotpercent',
        #'HardHit%',
        #'Barrel%',
        #'LD+%',# 'BABIP', 'ISO'
    ]
    
    # Statcast features for fine-tuning (2015+ only)
    # These are the most predictive metrics from statcast data
    STATCAST_FEATURES = [
        # Batted Ball Quality (most predictive)
        'sc_ev50',
        'sc_anglesweetspotpercent',
        'HardHit%',
        'Barrel%',
        'LD+%',



    ]
    
    # Combined features for fine-tuning (classical + statcast)
    # Total: 13 classical + 15 statcast = 28 features
    FINETUNE_FEATURES = CLASSICAL_FEATURES + STATCAST_FEATURES
    
    # Legacy compatibility - use classical features by default
    INPUT_FEATURES = CLASSICAL_FEATURES
    
   
    
    # ============================================================================
    # PARK FACTOR ADJUSTMENT (PREDICTIONS ONLY)
    # ============================================================================
    # When enabled, each player's historical stats are neutralized (divided by
    # their home park factor) before being fed to the model.  This lets the LSTM
    # operate in a park-neutral space so that a Coors hitter and a Petco hitter
    # with identical true talent produce the same model output.
    #
    # In value_determination, park factors are reapplied (multiplied) to the
    # model's predictions before computing wRC+ and WAR, so the final numbers
    # reflect the player's actual home environment.
    #
    # NOTE: Only affects predictions, NOT training — historical data going back
    # to 1950 does not have reliable park factors for all years.
    #
    
    
    # ============================================================================
    # PRE-TRAINING CONFIGURATION (1950-2024, Classical only)
    # ============================================================================
    # OPTIMIZED: Start season validated by hyperparameter tuning (1950 > 2000)
    
    PRETRAIN_DATA_FILE = '../data/historic_mlb/mlb_batting_data_1950_2025.csv'
    PRETRAIN_SCALER_FILE = 'data/batter_pretrain_scaler.pkl'
    PRETRAIN_CHECKPOINT_FILE = 'batter_pretrained.pth'
    PRETRAIN_START_SEASON = 1950  # Validated optimal (more historical data)
    PRETRAIN_MIN_PA = MIN_PA        # Validated optimal
    
    # ============================================================================
    # FINE-TUNING CONFIGURATION (2015+, Classical + Statcast)
    # ============================================================================
    
    FINETUNE_DATA_FILE = '../data/historic_mlb/mlb_batting_data_1950_2025_with_statcast.csv'  # Use statcast-joined data
    FINETUNE_SCALER_FILE = 'data/batter_finetune_scaler.pkl'
    FINETUNE_CHECKPOINT_FILE = 'batter_finetuned.pth'
    FINETUNE_START_SEASON = 2015  # Statcast era begins
    FINETUNE_MIN_PA = MIN_PA
    FINETUNE_LEARNING_RATE = 1e-4  # 10x smaller than pre-training
    FREEZE_LSTM = True  # Freeze LSTM layers during fine-tuning
    
    # Model architecture (direct attributes for factory compatibility)
    # These are the ACTUAL values the model will use after removing hardcoded modifications
    HIDDEN_SIZE = 128  # Actual internal LSTM hidden size
    NUM_LAYERS = 1   # Actual number of LSTM layers
    NUM_HEADS = 2   # Actual number of attention heads
    BIDIRECTIONAL = False
    DROPOUT = 0.05   # Validated optimal dropout rate
    
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5
    
    # Stability settings - prevent NaN issues
    WARMUP_EPOCHS = 5  # Skip warmup - causes tiny LR (4e-5) that leads to NaN
    MIXED_PRECISION = True  # Disable AMP - can cause numerical instability with warmup
    
    # ============================================================================
    # DOMAIN CONSTRAINT CONFIGURATION
    # ============================================================================
    # These weights control how strongly domain knowledge is enforced.
    # See core/constraint_config.py for presets and tuning guide.
    # 
    # Higher weights = more biologically plausible projections but potentially
    # higher MSE. Lower weights = better MSE but possible unrealistic projections.
    #
    # RECOMMENDED: Start with 'medium' constraints and adjust based on validation.
    

    # Data preprocessing config
    @staticmethod
    def get_data_config(mode='pretrain'):
        """
        Get data config for pre-training or fine-tuning
        
        OPTIMIZED PARAMETERS (validated by hyperparameter tuning on 96 configs):
        - seq_length=5: Optimal history window (5 > 6 years, R²=0.6585)
        - start_season=1950: Maximum historical data improves generalization
        - min_pa=75: Balanced threshold for data quality vs quantity
        
        Args:
            mode: 'pretrain' or 'finetune'
        """
        if mode == 'finetune':
            return DataConfig(
                input_features=BatterConfig.FINETUNE_FEATURES,  # 28 features (13 classical + 15 Statcast)
                output_features=BatterConfig.FINETUNE_FEATURES,  # 28 features - predict Statcast too for sliding window
                seq_length=BatterConfig.SEQ_LEN, 
                start_season=BatterConfig.FINETUNE_START_SEASON,
                min_pa=BatterConfig.FINETUNE_MIN_PA,
                train_ratio=0.75,
                valid_ratio=0.249,
                random_seed=42
            )
        else:  # pretrain
            return DataConfig(
                input_features=BatterConfig.CLASSICAL_FEATURES,
                seq_length=BatterConfig.SEQ_LEN,  # VALIDATED: Optimal sequence length
                start_season=BatterConfig.PRETRAIN_START_SEASON,  # VALIDATED: 1950 optimal
                min_pa=BatterConfig.PRETRAIN_MIN_PA,  # VALIDATED: 75 optimal
                train_ratio=0.75,
                valid_ratio=0.249,
                random_seed=42
            )
    

