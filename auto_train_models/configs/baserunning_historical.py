# Historical Baserunning model configuration
#
# Uses FanGraphs BsR_rate, SB_rate, and CS_rate (available from 2002)
# instead of Statcast baserunning metrics (sc_baserunning_runner_runs_tot_rate,
# available from 2016).  This allows LSTM-based baserunning projections for
# all cutoff years in the historical trade-value pipeline (2013+).
#
# BsR_rate is computed by calculate_rate_stats() as BsR / G * 150
# (BsR = FanGraphs Base Running Runs, available from 2002+).
#
# Train:
#   python scripts/train_models.py --model baserunning_historical --pretrain
#
# These models are used exclusively by the historical_values pipeline.

from core.data_processing import DataConfig


class BaserunningHistoricalConfig:
    """LSTM config for historical baserunning — BsR/SB/CS rate based."""

    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_FILE = '../data/historic_mlb/mlb_batting_data_1950_2025_with_statcast.csv'
    SCALER_FILE = 'data/baserunning_historical_pretrain_scaler.pkl'
    CHECKPOINT_DIR = './checkpoints/historical'
    CHECKPOINT_FILE = 'baserunning_historical.pth'
    PRETRAIN_CHECKPOINT_FILE = 'baserunning_historical.pth'
    OUTPUT_FILE = '../data/generated/historical_values/pipeline/baserunning_predictions.csv'

    # ── Features ──────────────────────────────────────────────────────────────
    INPUT_FEATURES = ['Age', 'BsR_rate', 'SB_rate', 'CS_rate']

    FEATURE_WEIGHTS = {
        'Age':      1.0,
        'BsR_rate': 1.0,
        'SB_rate':  0.5,
        'CS_rate':  0.5,
    }

    # ── Reliability regression ────────────────────────────────────────────────
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = False
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True

    # ── Model hyperparameters ─────────────────────────────────────────────────
    HIDDEN_SIZE   = 256
    NUM_LAYERS    = 3
    NUM_HEADS     = 2
    DROPOUT       = 0.5
    BIDIRECTIONAL = False
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
            input_features=BaserunningHistoricalConfig.INPUT_FEATURES,
            seq_length=3,
            start_season=2002,   # FanGraphs BsR available from ~2002
            min_pa=100,
            train_ratio=0.7,
            valid_ratio=0.29,
            random_seed=42,
        )
