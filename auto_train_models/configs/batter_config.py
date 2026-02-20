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
    USE_XBA_FOR_PREDICTIONS = False
    
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

    # Calculate OBP from components: (H + BB + HBP) / (AB + BB + HBP + SF)
    # Uses the same AVG, BB%, HR, 2B, 3B predictions as the wOBA calculation.
    # HBP is taken directly from the model's predicted HBP (per-150) value;
    # falls back to 1% of PA only when HBP is absent from predictions.
    # Set to False to use the LSTM's direct OBP prediction.
    CALCULATE_OBP_FROM_COMPONENTS = True

    # Calculate SLG from components: (1B + 2*2B + 3*3B + 4*HR) / AB
    # Uses the same AVG, HR, 2B, 3B predictions as the wOBA calculation.
    # Set to False to use the LSTM's direct SLG prediction.
    CALCULATE_SLG_FROM_COMPONENTS = True

    # ============================================================================
    # STATCAST IMPUTATION FROM xwOBA (PREDICTIONS ONLY)
    # ============================================================================
    # When enabled, players missing statcast features (sc_ev50, sc_anglesweetspot-
    # percent, HardHit%, Barrel%) will have those values estimated from their xwOBA
    # using linear regressions fit on 2015-2024 data (PA >= 100).  This prevents
    # players from being excluded from the finetuned model's predictions solely
    # because the advanced statcast columns are absent in some of their seasons.
    #
    # Only affects PREDICTIONS — training data is not modified.
    # Only fills NaN cells; existing values are never overwritten.
    IMPUTE_STATCAST_FROM_XWOBA = True

    # Regression coefficients: y = slope * xwOBA + intercept
    # Derived from 2015-2024 MLB statcast data (PA >= 100).
    #   sc_ev50                   : R² = 0.41, RMSE = 1.98, n = 4 090
    #   sc_anglesweetspotpercent  : R² = 0.20, RMSE = 3.72, n = 4 090
    #   HardHit%                  : R² = 0.40, RMSE = 0.065, n = 4 836
    #   Barrel%                   : R² = 0.40, RMSE = 0.032, n = 4 760
    STATCAST_IMPUTATION_COEFFICIENTS = {
        'sc_ev50':                  {'slope': 44.036147, 'intercept': 85.274046},
        'sc_anglesweetspotpercent': {'slope': 49.275835, 'intercept': 17.830435},
        'HardHit%':                 {'slope':  1.323041, 'intercept': -0.044321},
        'Barrel%':                  {'slope':  0.667159, 'intercept': -0.135564},
    }

    # ============================================================================
    # COUNTING-STAT ADJUSTMENT FOR X-STAT CONSISTENCY (PREDICTIONS ONLY)
    # ============================================================================
    # When x-stats (xwOBA, xSLG, xBA) are substituted for their actual
    # counterparts, the counting stats (HR, 2B, 3B, etc.) still reflect actual
    # outcomes.  This creates an internal inconsistency: a lucky hitter keeps
    # inflated HR/RBI numbers even though xwOBA says he should be worse.
    #
    # When enabled, each counting stat is adjusted by:
    #   adjustment = β_wOBA * (xwOBA – wOBA) + β_SLG * (xSLG – SLG) + β_AVG * (xBA – AVG)
    #   stat_adjusted = max(0, stat + adjustment)
    #
    # The β coefficients come from OLS regressions of each per-150 counting stat
    # on (wOBA, SLG, AVG) fitted on 2015-2024 statcast data (PA ≥ 100, n = 4 836).
    # Validation: recalculated wOBA MAE vs xwOBA improved 45.5 % (0.0199 → 0.0109)
    # and correlation rose from 0.846 to 0.980.
    #
    # Only affects PREDICTIONS — training data is not modified.
    # Requires at least one of USE_XWOBA/USE_XSLG/USE_XBA to be True, and the
    # corresponding x-stat columns to exist on the player-season row.
    ADJUST_COUNTING_STATS_TO_XSTATS = True

    # OLS coefficients: counting_stat_per150 ~ wOBA + SLG + AVG
    # Keyed by the per-150 column name that appears after calculate_rate_stats().
    XSTAT_COUNTING_ADJUSTMENT_COEFFICIENTS = {
        'HR':  {'wOBA':   +2.575, 'SLG': +159.102, 'AVG': -163.902},  # R²=0.86
        '2B':  {'wOBA':  -39.707, 'SLG':  +56.790, 'AVG':  +93.301},  # R²=0.40
        '3B':  {'wOBA':  -17.878, 'SLG':   +8.405, 'AVG':  +18.513},  # R²=0.05
        'RBI': {'wOBA':  -42.814, 'SLG': +279.536, 'AVG':  -76.336},  # R²=0.64
        'R':   {'wOBA': +273.212, 'SLG':  +56.306, 'AVG':   -7.282},  # R²=0.56
        'HBP': {'wOBA': +127.679, 'SLG':  -41.019, 'AVG':  -48.149},  # R²=0.13
    }

    # ============================================================================
    # TRANSFER LEARNING FEATURE SETS
    # ============================================================================
    #
    # Features are split into RATE stats (already normalised: percentages,
    # averages, indices — no per-150 conversion) and COUNTING stats (raw totals
    # that are converted to per-150-games during preprocessing by
    # calculate_rate_stats in core/data_processing.py).
    #
    # To add a new feature:
    #   Rate stat    (BB%, AVG, wOBA, HardHit%, …)  → add to *_RATE_FEATURES
    #   Counting stat (HR, RBI, HBP, SB, …)         → add to *_COUNTING_FEATURES
    #       AND ensure it also appears in BATTING_COUNTING_STATS in
    #       core/rate_stats_config.py so the per-150 conversion is applied.
    # ============================================================================
    
    # ---------- Classical rate stats (no per-150 conversion) ----------
    CLASSICAL_RATE_FEATURES = [
        'Age',
        'BB%', 'K%',
        'AVG', 'OBP', 'SLG', 'wOBA',
    ]
    
    # ---------- Classical counting stats (converted to per-150 games) ----------
    CLASSICAL_COUNTING_FEATURES = [
        'HR', '2B', '3B', 'RBI', 'R',
        'HBP',  # hit-by-pitch per 150 games; used in OBP and wOBA calculations
    ]
    
    CLASSICAL_FEATURES = CLASSICAL_RATE_FEATURES + CLASSICAL_COUNTING_FEATURES
    
    # ---------- Statcast rate stats (no per-150 conversion) ----------
    STATCAST_RATE_FEATURES = [
        'sc_ev50',
        'sc_anglesweetspotpercent',
        'HardHit%',
        'Barrel%',
        #'Pull%+',
        #'LD+%',
        #'CSW%',
        #'EV',
        #'LA'
    ]
    
    # ---------- Statcast counting stats (converted to per-150 games) ----------
    # (none currently — add counting Statcast features here if needed)
    STATCAST_COUNTING_FEATURES = [
    ]
    
    STATCAST_FEATURES = STATCAST_RATE_FEATURES + STATCAST_COUNTING_FEATURES
    
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
    FINETUNE_LEARNING_RATE = 1e-5  # 10x smaller than pre-training
    FREEZE_LSTM = True  # Freeze LSTM layers during fine-tuning
    
    # Model architecture (direct attributes for factory compatibility)
    # These are the ACTUAL values the model will use after removing hardcoded modifications
    HIDDEN_SIZE = 128  # Actual internal LSTM hidden size
    NUM_LAYERS = 2   # Actual number of LSTM layers
    NUM_HEADS = 2   # Actual number of attention heads
    BIDIRECTIONAL = False
    DROPOUT = 0.1   # Validated optimal dropout rate
    
    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
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
    

