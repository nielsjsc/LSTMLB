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
    OUTPUT_FILE = '../data/generated/historical_values/pipeline/baserunning_predictions.csv'

