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
    OUTPUT_FILE = '../data/generated/pipeline/pitcher_rp_predictions.csv'
    
    # Role-specific configuration
    ROLE = 'RP'
    GS_RATE_THRESHOLD = 0.8
    MIN_IP_CURRENT = 15  # Minimum IP in current year to generate predictions
    

    
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
    

    

