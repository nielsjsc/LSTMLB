# Historical Defense Catcher model configuration
#
# Uses DRS/150 (available from 2004) instead of Statcast catcher metrics
# (framing/blocking/throwing, available from 2016).  This allows LSTM-based
# catcher defense projections for all cutoff years in the historical
# trade-value pipeline (2013+).
#
# Train:
#   python scripts/train_models.py --model defense_catcher_historical --pretrain
#
# These models are used exclusively by the historical_values pipeline.

from core.data_processing import DataConfig


class DefenseCatcherHistoricalConfig:
    """LSTM config for historical catcher defense — DRS/150 based."""

    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_FILE = '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv'
    SCALER_FILE = 'data/defense_catcher_historical_pretrain_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints/historical'
    CHECKPOINT_FILE = 'C_fielding_historical.pth'
    PRETRAIN_CHECKPOINT_FILE = 'C_fielding_historical.pth'
    OUTPUT_FILE = '../data/generated/historical_values/pipeline/defense_catcher_predictions.csv'

    # ── Positions ─────────────────────────────────────────────────────────────
    POSITION_GROUP = 'CATCHER'
    VALID_POSITIONS = ['C']

    # ── Features (DRS/150 — available 2004+) ──────────────────────────────────
    INPUT_FEATURES = ['Age', 'DRS/150']

    FEATURE_WEIGHTS = {
        'Age': 1.0,
        'DRS/150': 1.0,
    }

    # ── Reliability regression ────────────────────────────────────────────────
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = True
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True

    # ── Model hyperparameters (simpler for smaller catcher sample) ────────────
    HIDDEN_SIZE   = 128
    NUM_LAYERS    = 2
    NUM_HEADS     = 1
    DROPOUT       = 0.4
    BIDIRECTIONAL = False
    GRADIENT_CLIP = 1.0

    # ── Training hyperparameters ──────────────────────────────────────────────
    BATCH_SIZE              = 64
    LEARNING_RATE           = 1e-3
    WEIGHT_DECAY            = 1e-5
    NUM_EPOCHS              = 30
    EARLY_STOPPING_PATIENCE = 8

    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=DefenseCatcherHistoricalConfig.INPUT_FEATURES,
            seq_length=4,
            start_season=2004,   # DRS available from BIS starting 2004
            min_pa=40,
            train_ratio=0.7,
            valid_ratio=0.29,
            random_seed=0,
        )

    # ── Domain constraints ────────────────────────────────────────────────────
    CONSTRAINT_STRENGTH = 'medium'

    DOMAIN_CONSTRAINTS = {
        'mse_weight': 1.0,
        'aging_weight': 0.25,
        'smoothness_weight': 0.07,
        'bounds_hard_weight': 0.60,
        'bounds_soft_weight': 0.08,
        'peak_weight': 0.05,
    }
