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
    
    # Prediction method: 'lstm' (default) or 'marcel' (weighted avg + aging curves)
    PREDICTION_METHOD = 'marcel'
    
    # Batter-specific configuration
    SEQ_LEN = 3
    MIN_PA = 50  # Minimum PA per season for training sequences
    MIN_PA_CURRENT = 70  # Minimum PA in current year to generate predictions
    
    # ============================================================================
    # SCALER CONFIGURATION
    # ============================================================================
    # Controls how features are scaled before the LSTM sees them.
    #
    # USE_HYBRID_SCALER = True  (recommended)
    #   Rate stats  (BB%, K%, AVG, OBP, SLG, wOBA, Age) → MinMaxScaler[-1, 1]
    #   Counting stats (HR, 2B, 3B, RBI, R, HBP)        → StandardScaler (z-score)
    #   This prevents the right-skewed counting-stat distributions from being
    #   compressed into a narrow band near 0 in MinMax space, which causes the
    #   LSTM to over-regress elite power numbers toward the population mean.
    #
    # USE_HYBRID_SCALER = False
    #   All features use MinMaxScaler[-1, 1] (legacy behaviour).
    #
    # LOG_TRANSFORM_COUNTING_STATS = True  (recommended with hybrid scaler)
    #   Apply log1p to counting stats *before* StandardScaler, and expm1 on
    #   inverse.  Compresses the heavy right tail (e.g. 50 HR/150 → log(51)=3.93
    #   vs mean log(12)=2.48) so the Gaussian assumption of StandardScaler holds
    #   better.  No-op when USE_HYBRID_SCALER = False.
    #
    # IMPORTANT: HybridScaler is INCOMPATIBLE with tanh-bounded outputs.
    # The LSTM uses tanh(x)*1.7079, capping output at ~±1.71.  With StandardScaler,
    # elite HR hitters (30+ HR/150) require z > 1.2, deep in tanh saturation where
    # gradients vanish.  This causes the model to under-predict counting stats for
    # above-average hitters.  MinMaxScaler keeps all counting values in the linear
    # zone of tanh (e.g. 30 HR → -0.2, 50 HR → +0.33).
    USE_HYBRID_SCALER = False
    LOG_TRANSFORM_COUNTING_STATS = True

    # ============================================================================
    # CALCULATE COMPONENTS FROM WOBA  (value_determination pipeline)
    # ============================================================================
    # The LSTM systematically mean-regresses counting stats (HR, 2B, 3B, RBI,
    # R, HBP) toward the training population average, but its rate-stat
    # predictions (wOBA, SLG, OBP, BB%, K%) are well-calibrated.
    #
    # This is the inverse of CALCULATE_WOBA_FROM_COMPONENTS: instead of
    # deriving wOBA from counting stats, we derive counting stats from wOBA.
    #
    # When enabled in value_determination, each counting stat is replaced by:
    #
    #   ratio         = predicted_wOBA / career_wOBA      (clipped 0.5–1.5)
    #   derived_count = career_count_per150 × ratio
    #   final         = blend × derived_count + (1 - blend) × model_prediction
    #   blend         = min(career_PA / COMPONENTS_FROM_WOBA_PA_WEIGHT, 1.0)
    #
    # Each player keeps their OWN HR/2B/3B/RBI/R profile (Raleigh's HR-heavy
    # mix, Witt's doubles+triples, Henderson's power).  The quality ratio
    # scales the entire profile up or down based on the model's projected wOBA.
    #
    # When this is True, all legacy CALCULATE_*_FROM_COMPONENTS flags are
    # ignored — wOBA, OBP, SLG, and AVG are kept as the model predicted them,
    # and only counting stats are derived from career profiles.
    CALCULATE_COMPONENTS_FROM_WOBA = True
    COMPONENTS_FROM_WOBA_PA_WEIGHT = 1500       # ~2.5 full seasons for full trust
    COMPONENTS_FROM_WOBA_RECENT_SEASONS = 3     # seasons for career average

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
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True
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
    # xwOBA-wOBA GAP ADJUSTMENT (Marcel only)
    # ============================================================================
    # Some players consistently over- or under-perform their expected stats due
    # to batted ball direction, speed, or other skills not captured by xwOBA.
    # Example: Jose Ramirez and Cal Raleigh pull the ball in the air, producing
    # better real outcomes than xwOBA predicts.
    #
    # When enabled, the Marcel projection adds back a regressed portion of the
    # player's historical wOBA - xwOBA gap (and analogous gaps for AVG/SLG).
    # The gap is PA-weighted over the last 3 seasons, regressed toward 0 to
    # account for sample noise, then scaled by the skill fraction.
    ENABLE_XWOBA_GAP_ADJUSTMENT = True
    XWOBA_GAP_SKILL_FRACTION = 0.5    # Fraction of regressed gap to add back
    XWOBA_GAP_MIN_SEASONS = 2         # Minimum seasons with x-stat data needed
    XWOBA_GAP_REGRESSION_PA = 800     # PA of regression toward 0 (higher = more conservative)
    
    # ============================================================================
    # WAR CALCULATION CONFIGURATION
    # ============================================================================
    # These legacy toggles only apply when CALCULATE_COMPONENTS_FROM_WOBA = False.
    # When CALCULATE_COMPONENTS_FROM_WOBA = True (Mode A), the model's wOBA/OBP/SLG
    # are kept as-is and counting stats are derived — these flags are ignored.
    CALCULATE_WOBA_FROM_COMPONENTS = True
    CALCULATE_OBP_FROM_COMPONENTS = True
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
    ADJUST_COUNTING_STATS_TO_XSTATS = False

    # ............................................................................
    # PLAYER-SPECIFIC X-STAT COUNTING ADJUSTMENT
    # ............................................................................
    # When enabled, the counting-stat adjustment uses each player's own
    # historical ratio of counting stats to rate stats, weighted by sample size,
    # instead of the population-wide OLS coefficients above.
    #
    # Why: Some players consistently outperform their xSLG (Jose Ramirez — pull
    # hitter, shift exploiter) or consistently under-perform (flyball outs).  The
    # population OLS ignores this and can shift HR by ±10/150 in the wrong
    # direction.  Player-specific ratios, shrunk toward the population mean for
    # small samples, solve both problems:
    #     • A 1-year player gets ~population coefficients (we don't know their
    #       true ratio yet).
    #     • A 6-year veteran with a stable HR/wOBA ratio keeps their own ratio.
    #
    # Implementation:
    #   For each counting stat C and each rate stat R in {wOBA, SLG, AVG}:
    #     player_ratio = career_C / career_R   (across available seasons)
    #     weight       = min(career_PA / XSTAT_PA_FULL_WEIGHT, 1.0)
    #     blended      = weight * player_ratio + (1 - weight) * OLS_β
    #   Then the adjustment is the same formula using blended β's.
    #
    # XSTAT_PA_FULL_WEIGHT: PA threshold at which the player's own ratio is
    #   trusted 100%.  At half this value the blend is 50/50.
    USE_PLAYER_SPECIFIC_XSTAT_ADJUSTMENT = True
    XSTAT_PA_FULL_WEIGHT = 2000  # ~3-4 full seasons

    # OLS coefficients: counting_stat_per150 ~ intercept + wOBA + SLG + AVG
    # Keyed by the per-150 column name that appears after calculate_rate_stats().
    # When USE_PLAYER_SPECIFIC_XSTAT_ADJUSTMENT is enabled these serve as the
    # population priors that small-sample players are shrunk toward.
    #
    # The 'intercept' is used ONLY for computing player-specific residuals so
    # the offset is properly centred.  It does NOT affect the delta-based
    # adjustment itself (Δ_wOBA, Δ_SLG, Δ_AVG deltas already cancel it).
    # Intercepts were derived from 2015-2024+ statcast data (PA >= 100).
    XSTAT_COUNTING_ADJUSTMENT_COEFFICIENTS = {
        'HR':  {'intercept': -8.277, 'wOBA':   +2.575, 'SLG': +159.102, 'AVG': -163.902},  # R²=0.86
        '2B':  {'intercept': -9.829, 'wOBA':  -39.707, 'SLG':  +56.790, 'AVG':  +93.301},  # R²=0.40
        '3B':  {'intercept': -0.178, 'wOBA':  -17.878, 'SLG':   +8.405, 'AVG':  +18.513},  # R²=0.05
        'RBI': {'intercept': -20.086, 'wOBA': -42.814, 'SLG': +279.536, 'AVG':  -76.336},  # R²=0.64
        'R':   {'intercept': -42.640, 'wOBA': +273.212, 'SLG':  +56.306, 'AVG':   -7.282},  # R²=0.56
        'HBP': {'intercept': -5.594, 'wOBA': +127.679, 'SLG':  -41.019, 'AVG':  -48.149},  # R²=0.13
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
        #'HardHit%',
        #'Barrel%',
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
    FINETUNE_LEARNING_RATE = 1e-3  # 10x smaller than pre-training
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
    EARLY_STOPPING_PATIENCE = 15
    
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
    

