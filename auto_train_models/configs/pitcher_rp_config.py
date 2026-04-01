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
    
    # Prediction method: 'lstm' (default) or 'marcel' (weighted avg + aging curves)
    PREDICTION_METHOD = 'marcel'
    
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
    ENABLE_FIP_RECONSTRUCTION = True
    
    # HR% DECOMPOSITION: No longer needed — HR% has been removed as a model
    #   feature. FIP reconstruction now derives HR% internally from HR/FB and
    #   FB% (which are both direct model features).
    ENABLE_HR_DECOMPOSITION = False
    
    # SIERA RECONSTRUCTION: Overwrite the model's direct SIERA prediction with
    #   a value reconstructed from predicted K%, BB%, GB% components.
    #   Requires GB% and SIERA in CLASSICAL_FEATURES and PRETRAIN_START_SEASON >= 2002.
    ENABLE_SIERA_RECONSTRUCTION = True
    
    # ERA-FIP ADJUSTMENT: ERA = reconstructed_FIP + regressed career ERA-FIP gap.
    ENABLE_ERA_FIP_ADJUSTMENT = True

    # ERA-SIERA ADJUSTMENT (alternative to ERA-FIP): ERA = reconstructed_SIERA
    #   + regressed career ERA-SIERA gap (n0=3103 TBF ≈ 722 IP). SIERA better
    #   predicts future ERA (r=0.403 vs 0.372) with higher-ICC gap (0.124).
    #   When enabled, takes precedence over ERA-FIP adjustment.
    ENABLE_ERA_SIERA_ADJUSTMENT = False
    
    # OUTPUT REGRESSION: Apply Bayesian shrinkage to the model's predicted
    #   rate stats based on career sample size. Ensures K%, BB%, HBP%
    #   etc. are appropriately conservative for short-career pitchers.
    ENABLE_OUTPUT_REGRESSION = False

    # PITCHER AGING CONSTRAINTS: Prevent unrealistic improvement after peak age.
    ENABLE_PITCHER_AGING_CONSTRAINTS = True

    # ========================================================================
    # STATCAST QUALITY ADJUSTMENT — see PitcherSPConfig for full documentation.
    # ========================================================================
    ENABLE_STATCAST_ADJUSTMENT = True
    STATCAST_ADJUSTMENT_CAP = 0.30
    STATCAST_ADJUSTMENT_MAP = {
        'Stuff+': {
            'K%':    0.12,
            'HR/FB': -0.06,
            'BABIP': -0.04,
        },
        'Location+': {
            'BB%':  -0.10,
            'K%':    0.04,
            'HBP%': -0.04,
        },
        'Pitching+': {
            'ERA':  -0.06,
            'FIP':  -0.06,
        },
    }
    
    # RECENCY-WEIGHTED CAREER PRIOR: Half-life in seasons (0 = no decay).
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
        'K%',           # 70 BF to stabilize - even more important for RPs
        'BB%',          # 170 BF to stabilize - command metric
        # FIP component rates (predicted by model, FIP reconstructed post-prediction)
        # HR% removed — derived on-the-fly from HR/FB × FB% for FIP/HR/9
        'HBP%',         # 995 BF to stabilize - HBP per TBF (data-derived)
        # BABIP: enables per-pitcher BF/IP derivation for FIP reconstruction
        'BABIP',        # ~820 BF to stabilize - mostly noise, but needed for BF/IP
        # Traditional metrics (model still learns ERA; FIP reconstructed from components)
        'FIP',          # Fielding independent - better than ERA for RPs
        'ERA',          # Highly volatile for RPs but still useful context
        # Workload
        #'IP',           # Innings context (much lower for RPs)
    ]
    
    # Extended classical features for SIERA reconstruction (2002+)
    # When ENABLE_SIERA_RECONSTRUCTION = True, use these AND set
    # PRETRAIN_START_SEASON = 2002 (GB% requires pitchfx-era data).
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
        #'FBv',          # Fastball velocity - critical for relievers
        #'SwStr%',       # Swing and miss rate - pure stuff indicator
        #'CSW%',         # Called strikes + whiffs - command + stuff
        'GB%',          # Ground ball rate
        'FB%',          # Fly ball rate
        'HR/FB',        # HR per fly ball - isolates HR skill from FB tendency
                        # YoY r=0.17 (noisy) but stabilizes at ~170 TBF
                        # Decomposition: HR% ≈ HR/FB × FB% × BIP_rate
        #'Contact%',     # Contact rate
        #'xFIP',         # Expected FIP
        'SIERA',        # Advanced run estimator
        'LD%'
    ]
    
    # Statcast era features (2020+) - the most predictive metrics
    # Research: "Pitching+ out-predicts any current projection system for relievers"
    STATCAST_FEATURES = [
        #'Stuff+',       # Pitch quality model - CRITICAL for relievers
        #'Location+',    # Command model
        'Pitching+'    # Combined model - best for relievers per research
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
    PRETRAIN_SCALER_FILE = 'data/pitcher_rp_pretrain_scaler.pkl'
    PRETRAIN_CHECKPOINT_FILE = 'rp/pitcher_rp_pretrained.pth'
    PRETRAIN_START_SEASON = 1950  # Maximum historical data for classical features
    PRETRAIN_MIN_IP = 25          # Low threshold for relievers
    
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
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    NUM_HEADS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3
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
    
    # ============================================================================
    # SCALER ALIGNMENT
    # ============================================================================
    # Features in each group share the same MinMax range so that equal raw
    # values (e.g. FIP=4.00 and ERA=4.00) map to the *same* scaled value.
    # Without this, ERA's wider outlier range creates a permanent phantom gap
    # in normalised space — the model sees "FIP ≠ ERA" even when they're equal.
    LINKED_SCALE_GROUPS = [
        ['FIP', 'ERA', 'xFIP', 'SIERA'],
    ]
    
    # Percentile clipping removes extreme outliers before fitting the scaler.
    # Relievers are even noisier (low IP), so clipping is especially important.
    # HBP% clipping prevents tiny-sample outliers from dominating scaler.
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
                random_seed=42,
                linked_scale_groups=PitcherRPConfig.LINKED_SCALE_GROUPS,
                stat_clip_percentiles=PitcherRPConfig.STAT_CLIP_PERCENTILES,
            )
        else:  # pretrain
            return DataConfig(
                input_features=PitcherRPConfig.INPUT_FEATURES,
                seq_length=PitcherRPConfig.SEQ_LENGTH,
                start_season=PitcherRPConfig.PRETRAIN_START_SEASON,
                min_pa=PitcherRPConfig.PRETRAIN_MIN_IP,
                train_ratio=0.8,
                valid_ratio=0.19,
                random_seed=42,
                linked_scale_groups=PitcherRPConfig.LINKED_SCALE_GROUPS,
                stat_clip_percentiles=PitcherRPConfig.STAT_CLIP_PERCENTILES,
            )
    

