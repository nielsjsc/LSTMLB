# Historical Defense Infield model configuration
#
# Uses UZR/150 (available from 2002) instead of Statcast defensive metrics
# (available from 2016).  This allows LSTM-based fielding projections for
# all cutoff years in the historical trade-value pipeline (2013+).
#
# Train:
#   python scripts/train_models.py --model defense_infield_historical --pretrain
#
# These models are used exclusively by the historical_values pipeline.

from core.data_processing import DataConfig


class DefenseInfieldHistoricalConfig:
    """LSTM config for historical infield defense — UZR/150 based."""

    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_FILE = '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv'
    SCALER_FILE = 'data/defense_infield_historical_pretrain_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints/historical'
    CHECKPOINT_FILE = 'IF_fielding_historical.pth'
    PRETRAIN_CHECKPOINT_FILE = 'IF_fielding_historical.pth'
    OUTPUT_FILE = '../data/generated/historical_values/pipeline/defense_infield_predictions.csv'

    # ── Positions ─────────────────────────────────────────────────────────────
    POSITION_GROUP = 'INFIELD'
    VALID_POSITIONS = ['1B', '2B', '3B', 'SS']

    # ── Features (UZR/150 — available 2002+) ──────────────────────────────────
    INPUT_FEATURES = ['Age', 'UZR/150']

    FEATURE_WEIGHTS = {
        'Age': 1.0,
        'UZR/150': 1.0,
    }

    # ── Reliability regression ────────────────────────────────────────────────
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = True
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True

    # ── Model hyperparameters ─────────────────────────────────────────────────
    HIDDEN_SIZE   = 256
    NUM_LAYERS    = 2
    NUM_HEADS     = 4
    DROPOUT       = 0.15
    BIDIRECTIONAL = True
    GRADIENT_CLIP = 1.0

    # ── Training hyperparameters ──────────────────────────────────────────────
    BATCH_SIZE              = 64
    LEARNING_RATE           = 1e-3
    WEIGHT_DECAY            = 1e-5
    NUM_EPOCHS              = 50
    EARLY_STOPPING_PATIENCE = 10

    @staticmethod
    def get_data_config():
        return DataConfig(
            input_features=DefenseInfieldHistoricalConfig.INPUT_FEATURES,
            seq_length=3,
            start_season=2002,   # UZR/150 available from 2002
            min_pa=50,
            train_ratio=0.7,
            valid_ratio=0.2,
            random_seed=42,
        )

    # ── Domain constraints ────────────────────────────────────────────────────
    CONSTRAINT_STRENGTH = 'medium'

    DOMAIN_CONSTRAINTS = {
        'mse_weight': 1.0,
        'aging_weight': 0.20,
        'smoothness_weight': 0.07,
        'bounds_hard_weight': 0.60,
        'bounds_soft_weight': 0.08,
        'peak_weight': 0.05,
    }
