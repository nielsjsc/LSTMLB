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
    
    # ============================================================================
    # UNIFIED PITCHER MODEL
    # ============================================================================
    # When True, train a single model on ALL pitcher data (SP + RP combined)
    # and predict for every pitcher regardless of role.  The role column is still
    # set based on GS rate so downstream value determination can apply
    # role-specific playing-time and WAR calculations.
    #
    # Training: `python scripts/train_models.py --model pitcher_sp --pretrain`
    #   → uses ALL pitching data (no GS-rate filter)
    #   → checkpoint saved to PRETRAIN_CHECKPOINT_FILE as usual
    #
    # Prediction: `python scripts/predict_models.py --model-type pitcher --use-pretrained`
    #   → loads only the SP model/scaler (RP model ignored)
    #   → runs every pitcher through the single model
    #
    # Set to False to revert to separate SP/RP models.
    UNIFIED_PITCHER_MODEL = False
    
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
    # RELIABILITY REGRESSION
    # ============================================================================
    # Bayesian shrinkage of rate stats toward a career/league-average prior,
    # weighted by sample size (BF for pitchers).  The two toggles are independent:
    #
    #   TRAINING   — applied to the historical DataFrame before the LSTM sees it.
    #   PREDICTION — applied to each player's historical sequence at inference time.
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = False
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True
    
    # ============================================================================
    # POST-PREDICTION RECONSTRUCTION & CONSTRAINT TOGGLES
    # ============================================================================
    # These control post-prediction transformations applied in the autoregressive
    # prediction loop. Each is independently toggleable.
    #
    # FIP RECONSTRUCTION: Overwrite the model's direct FIP prediction with a
    #   value reconstructed from predicted K%, BB%, HBP% components and HR%
    #   derived on-the-fly from HR/FB × FB% × BIP_rate.
    #   Requires K%, BB%, HBP%, FIP, HR/FB, FB% in INPUT_FEATURES.
    ENABLE_FIP_RECONSTRUCTION = True
    
    # HR% DECOMPOSITION: No longer needed — HR% has been removed as a model
    #   feature. FIP reconstruction now derives HR% internally from HR/FB and
    #   FB% (which are both direct model features).
    ENABLE_HR_DECOMPOSITION = False
    
    # SIERA RECONSTRUCTION: Overwrite the model's direct SIERA prediction with
    #   a value reconstructed from predicted K%, BB%, GB% components using
    #   empirically-derived coefficients (full quadratic model, r=0.907 overall,
    #   r=0.948 on 2020+ data). Requires GB% and SIERA in CLASSICAL_FEATURES
    #   and PRETRAIN_START_SEASON >= 2002 (GB%/SIERA need pitchfx-era data).
    ENABLE_SIERA_RECONSTRUCTION = False
    
    # ERA-FIP ADJUSTMENT: Set ERA = reconstructed_FIP + regressed career ERA-FIP
    #   gap (James-Stein shrinkage, n0=6472 TBF). Captures genuine ERA-FIP skill
    #   for pitchers with sufficient career IP.
    ENABLE_ERA_FIP_ADJUSTMENT = True

    # ERA-SIERA ADJUSTMENT (alternative to ERA-FIP): Set ERA = reconstructed_SIERA
    #   + regressed career ERA-SIERA gap (n0=3103 TBF). SIERA is a better predictor
    #   of future ERA (next-year r=0.403 vs 0.372 for FIP), and the ERA-SIERA gap
    #   has higher ICC (0.124 vs 0.088) and stabilizes faster (~722 IP vs ~1059 IP).
    #   However, FIP-based ERA derivation is tighter for same-season fit (r=0.775
    #   vs 0.618). When enabled, takes precedence over ERA-FIP adjustment.
    ENABLE_ERA_SIERA_ADJUSTMENT = False
    
    # OUTPUT REGRESSION: Apply Bayesian shrinkage to the model's predicted
    #   rate stats (K%, BB%, HBP%, etc.) based on career sample size.
    #   The input sequence is already regressed, but the model's OUTPUT can
    #   still produce extreme values for short-career pitchers because the
    #   skip connection anchors to recent performance.  This post-prediction
    #   regression ensures that K%, BB%, HBP% (which feed into FIP/ERA
    #   reconstruction) are appropriately conservative for low-sample players.
    #   Formula: regressed = (pred × career_tbf + lg_avg × n0) / (career_tbf + n0)
    ENABLE_OUTPUT_REGRESSION = False

    # PITCHER AGING CONSTRAINTS: Prevent unrealistic improvement after peak age
    #   in the autoregressive loop. Enforces physical bounds (BB% >= 0, etc.)
    #   and caps year-over-year improvement rates by age band based on empirical
    #   aging curves derived from 1950-2024 historical data.
    ENABLE_PITCHER_AGING_CONSTRAINTS = True
    
    # ============================================================================
    # STATCAST QUALITY ADJUSTMENT
    # ============================================================================
    # Apply a small multiplier to classical stats based on Stuff+/Location+/
    # Pitching+ to incorporate pitch-quality information without fine-tuning.
    #
    # Motivation: Fine-tuning with only ~360 Statcast-era sequences is unreliable.
    # Instead, we use Statcast grades to nudge classical stats in the direction
    # the pitch-quality models predict — a pitcher with elite Stuff+ (120)
    # should project slightly better K%.
    #
    # Applied AFTER reliability regression, BEFORE sequence construction, so the
    # LSTM sees Statcast-informed inputs without requiring Statcast features in
    # the model architecture.
    #
    # Formula per classical stat:
    #   z = Σ (coefficient × (metric - 100) / 100)  for each mapped Statcast metric
    #   adjusted = regressed × (1 + clamp(z, -CAP, +CAP))
    #
    # Example: Stuff+=120, K% coefficient=+0.12
    #   z = 0.12 × (120-100)/100 = +0.024 → K% boosted by 2.4%
    ENABLE_STATCAST_ADJUSTMENT = True

    # Maximum absolute adjustment (caps the total z across all Statcast metrics).
    # 0.10 = ±10% max change to any single classical stat.
    STATCAST_ADJUSTMENT_CAP = 0.30

    # Maps Statcast metric → {classical stat → signed coefficient}.
    # Positive = Statcast metric positively correlates with the classical stat
    #   (higher Stuff+ → higher K%).
    # Negative = inverse correlation (higher Location+ → lower BB%).
    #
    # Coefficients are intentionally conservative; they represent the marginal
    # information Statcast adds BEYOND what the classical stats already capture.
    STATCAST_ADJUSTMENT_MAP = {
        'Stuff+': {
            'K%':    0.12,   # Strong: elite stuff drives strikeouts (r≈0.73)
            'HR/FB': -0.06,  # Moderate: elite stuff suppresses HR/FB
            'BABIP': -0.04,  # Weak: quality contact reduction
        },
        'Location+': {
            'BB%':  -0.10,   # Strong: elite command reduces walks (r≈-0.50)
            'K%':    0.04,   # Weak: command also contributes to Ks
            'HBP%': -0.04,   # Weak: better command → fewer HBP
        },
        'Pitching+': {
            'ERA':  -0.06,   # Moderate: combined quality → lower ERA
            'FIP':  -0.06,   # Moderate: combined quality → lower FIP
        },
    }

    # RECENCY-WEIGHTED CAREER PRIOR: Exponential decay half-life in seasons.
    #   A season N years ago gets weight 2^(-N / halflife) relative to the most
    #   recent season, applied on top of volume (TBF) weighting. This makes the
    #   career prior more responsive to recent performance trends.
    #   Set to 0 to disable recency weighting (pure TBF-weighted mean).
    PRIOR_RECENCY_HALFLIFE = 3

    # PER-FEATURE LEAGUE-AVERAGE WEIGHT OVERRIDES
    # Maps feature name → minimum fraction of the regression target that comes
    # from the league average (rather than the player's career average).
    #
    # Rationale for HR/FB:
    #   ICC = 0.216 → only ~22% of HR/FB variance is pitcher identity.
    #   Extreme HR/FB seasons (>17%) regress ~83% toward league average.
    #   Optimal blend: 75% league + 25% career (RMSE ≈ 0.039, almost as good
    #   as 100% league while preserving some career signal).
    #
    # Set to {} or None to disable all overrides.
    PRIOR_LEAGUE_WEIGHT_OVERRIDES = {}
    
    # ============================================================================
    # PARK FACTOR ADJUSTMENT (PREDICTIONS ONLY)
    # ============================================================================
    # When enabled, each pitcher's historical stats are neutralized (divided by
    # their home park factor) before being fed to the model.  This lets the LSTM
    # operate in a park-neutral space so that a pitcher in Coors and one in Petco
    # with identical true talent produce the same model output.
    #
    # In value_determination, park factors are reapplied (multiplied) to the
    # model's FIP/ERA predictions before computing pitcher WAR, so the final
    # numbers reflect the pitcher's actual home environment.
    #
    # NOTE: Only affects predictions, NOT training — historical data going back
    # to 1950 does not have reliable park factors for all years.
    ENABLE_PARK_FACTOR_ADJUSTMENT = False
    
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
        # FIP component rates (predicted by model, FIP reconstructed post-prediction)
        # HR% removed — derived on-the-fly from HR/FB × FB% for FIP/HR/9
        'HBP%',         # 995 BF to stabilize - HBP per TBF (data-derived)
        # BABIP: enables per-pitcher BF/IP derivation for FIP reconstruction
        # (instead of a constant 4.25 multiplier for all pitchers)
        'BABIP',        # ~820 BF to stabilize - mostly noise, but needed for BF/IP
        # Traditional metrics (model still learns ERA; FIP reconstructed from components)
        'FIP',          # Fielding independent - better than ERA
        'ERA',          # Outcome metric (model learns variance)
    ]
    
    # Extended classical features for SIERA reconstruction (2002+)
    # When ENABLE_SIERA_RECONSTRUCTION = True, use these features AND set
    # PRETRAIN_START_SEASON = 2002 (GB% requires pitchfx-era data).
    # This trades 52 years of pre-1950 data for SIERA component prediction.
    SIERA_CLASSICAL_FEATURES = [
        'Age',
        'K%', 'BB%', 'HBP%',
        'BABIP',        # Enables per-pitcher BF/IP derivation for FIP
        'GB%',          # 70 BIP to stabilize - ground ball rate (2002+)
        'HR/FB',        # ~170 TBF to stabilize - HR-on-contact skill (2002+)
        'FIP', 'ERA', 'SIERA',
    ]
    
    # PITCHf/x era features (2002+) - adds velocity and contact metrics
    PITCHFX_FEATURES = [
        #'FBv',          # Fastball velocity - physical, ages predictably
        #'SwStr%',       # Swing and miss rate - true skill
        #'CSW%',         # Called strikes + whiffs - command + stuff
        'GB%',          # Ground ball rate - 70 BIP to stabilize
        'FB%',          # Fly ball rate - batted ball profile
        'HR/FB',        # HR per fly ball - isolates HR skill from FB tendency
                        # YoY r=0.17 (noisy) but stabilizes at ~170 TBF
                        # Decomposition: HR% ≈ HR/FB × FB% × BIP_rate
        #'Contact%',     # Contact rate - inverse of strikeout ability
        #'xFIP',         # Expected FIP (normalizes HR/FB)
        'SIERA',        # Advanced run estimator
        'LD%'
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
    FINETUNE_FEATURES = CLASSICAL_FEATURES + PITCHFX_FEATURES + STATCAST_FEATURES

    # Legacy compatibility - use classical features by default
    INPUT_FEATURES = CLASSICAL_FEATURES + PITCHFX_FEATURES
    
    # ============================================================================
    # PRE-TRAINING CONFIGURATION (1950-2024, Classical only)
    # ============================================================================
    SEQ_LENGTH = 3  # season sequences for pre-training
    PRETRAIN_DATA_FILE = '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv'
    PRETRAIN_SCALER_FILE = 'data/pitcher_sp_pretrain_scaler.pkl'
    PRETRAIN_CHECKPOINT_FILE = 'sp/pitcher_sp_pretrained.pth'
    PRETRAIN_START_SEASON = 1950  # Maximum historical data for classical features
    PRETRAIN_MIN_IP = 120          # Quality threshold (lowered from 40 for UNIFIED_PITCHER_MODEL to include RP seasons)
    
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
    NUM_HEADS = 2                 # Best: 4 (backtest-optimized)
    BIDIRECTIONAL = False
    DROPOUT = 0.3          # Best: 0.253 (backtest-optimized)
    GRADIENT_CLIP = 1.0
    
    # Training parameters - TUNED via backtest hyperparameter search (100 trials)
    # Trial #24: Achieves 37.90% average skill score (vs naive baseline)
    # Per-year: 24.87% (2023), 48.08% (2024), 40.76% (2025)
    BATCH_SIZE = 64        # Best: 32 (backtest-optimized)
    LEARNING_RATE = 1e-04      # Best: 4.19e-05 (backtest-optimized)
    WEIGHT_DECAY = 1.1527987128232402e-06        # Best: 3.70e-05 (backtest-optimized)
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # ============================================================================
    # DOMAIN CONSTRAINT CONFIGURATION
    # ============================================================================
    # Pitchers decline faster than hitters and have earlier peaks.
    # These constraints are calibrated for pitcher-specific aging patterns.
    
    # ============================================================================
    # SCALER ALIGNMENT
    # ============================================================================
    # Features in each group share the same MinMax range so that equal raw
    # values (e.g. FIP=4.00 and ERA=4.00) map to the *same* scaled value.
    # Without this, ERA's wider outlier range (max=11.27 vs FIP max=8.62)
    # creates a permanent phantom gap in normalised space — the model sees
    # "FIP ≠ ERA" even when they are identical in reality.
    LINKED_SCALE_GROUPS = [
        ['FIP', 'ERA', 'xFIP', 'SIERA'],
    ]
    
    # Percentile clipping removes extreme outliers before fitting the scaler.
    # A 40-IP pitcher with an 11.0 ERA compresses the 2.5-5.5 range (where
    # 95 % of starters live) into a tiny fraction of [-1, 1].  Clipping at
    # p0.5/p99.5 expands model resolution in the range that matters.
    # HBP% is a small rate (~0.008-0.01) with occasional extreme outliers
    # from low-IP stints — clipping keeps scaler resolution.
    STAT_CLIP_PERCENTILES = {
        'FIP':   (0.5, 99.5),
        'ERA':   (0.5, 99.5),
        'xFIP':  (0.5, 99.5),
        'SIERA': (0.5, 99.5),
        'HBP%':  (0.5, 99.5),
        'HR/FB': (0.5, 99.5),   # Small rate with extreme outliers from low-FB stints
        'BABIP': (0.5, 99.5),  # Narrow range with occasional outliers from low-IP stints
    }
    
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
                random_seed=42,
                linked_scale_groups=PitcherSPConfig.LINKED_SCALE_GROUPS,
                stat_clip_percentiles=PitcherSPConfig.STAT_CLIP_PERCENTILES,
            )
        else:  # pretrain
            return DataConfig(
                input_features=PitcherSPConfig.INPUT_FEATURES,
                seq_length=PitcherSPConfig.SEQ_LENGTH,
                start_season=PitcherSPConfig.PRETRAIN_START_SEASON,
                min_pa=PitcherSPConfig.PRETRAIN_MIN_IP,
                train_ratio=0.8,
                valid_ratio=0.19,
                random_seed=42,
                linked_scale_groups=PitcherSPConfig.LINKED_SCALE_GROUPS,
                stat_clip_percentiles=PitcherSPConfig.STAT_CLIP_PERCENTILES,
            )
    

