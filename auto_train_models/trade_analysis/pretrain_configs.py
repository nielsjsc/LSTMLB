"""
Trade Analysis — Pretrained Model Configs
==========================================

These configs are **owned by the trade-analysis workflow** and are completely
separate from the main project configs (configs/defense_*.py, configs/baserunning_config.py).

Why separate configs?
- The main models use Statcast features (available only from 2016+), which means
  they cannot generate predictions for cutoff years before 2016.
- The trade-analysis pipeline needs historical projections back to 2013, so it
  trains its own lightweight models on pre-Statcast metrics:
    • Infield  — UZR/150  (FanGraphs, available 2002+)
    • Outfield — UZR/150  (FanGraphs, available 2002+)
    • Catcher  — DRS/150  (BIS, available 2004+)
    • Baserunning — BsR_rate + SB_rate + CS_rate (FanGraphs, available 2002+)

Checkpoint / scaler storage
- Checkpoints: auto_train_models/trade_analysis/checkpoints/
- Scalers:     auto_train_models/trade_analysis/data/
These are intentionally isolated from the main model artifacts in
auto_train_models/checkpoints/.

Usage:
    from trade_analysis.pretrain_configs import (
        PretrainInfieldConfig,
        PretrainOutfieldConfig,
        PretrainCatcherConfig,
        PretrainBaserunningConfig,
        PRETRAIN_CHECKPOINTS_DIR,
        PRETRAIN_DATA_DIR,
    )
"""

import sys
from pathlib import Path

# Allow importing core modules when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data_processing import DataConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
TRADE_ANALYSIS_DIR   = Path(__file__).resolve().parent          # trade_analysis/
AUTO_TRAIN_DIR       = TRADE_ANALYSIS_DIR.parent                # auto_train_models/

# Checkpoints stored here (self-contained within trade_analysis/)
PRETRAIN_CHECKPOINTS_DIR = TRADE_ANALYSIS_DIR / "checkpoints"  # …/trade_analysis/checkpoints/

# Scalers are saved by core.data_processing.scale_features() to auto_train_models/data/
# using the pattern  {model_type}_pretrain_scaler.pkl.  We point to that location so
# predict_pretrain.py and the training script agree on the path.
PRETRAIN_SCALER_DIR = AUTO_TRAIN_DIR / "data"                  # auto_train_models/data/

# Ensure directories exist
PRETRAIN_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
PRETRAIN_SCALER_DIR.mkdir(parents=True, exist_ok=True)


# ── Shared data-file paths (relative to auto_train_models/ like the main configs) ─
_FIELDING_DATA_FILE   = "../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv"
_BATTING_DATA_FILE    = "../data/historic_mlb/mlb_batting_data_1950_2025_with_statcast.csv"


# =============================================================================
# INFIELD (1B / 2B / 3B / SS) — UZR/150
# =============================================================================

class PretrainInfieldConfig:
    """
    Trade-analysis pretrained infield model.

    Feature: UZR/150 — the standard pre-Statcast infield metric from FanGraphs.
    Available from ~2002 onward, so this model covers every cutoff year in the
    trade-analysis pipeline (2013+).
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_FILE      = _FIELDING_DATA_FILE
    CHECKPOINT_DIR = str(PRETRAIN_CHECKPOINTS_DIR)
    CHECKPOINT_FILE = "IF_fielding_pretrained.pth"
    # Saved by preprocess_data(model_type='defense_infield', mode='pretrain')
    SCALER_FILE    = str(PRETRAIN_SCALER_DIR / "defense_infield_pretrain_scaler.pkl")
    OUTPUT_FILE    = "../data/generated/trade_analysis/pipeline/defense_infield_predictions.csv"

    # ── Positions ─────────────────────────────────────────────────────────────
    POSITION_GROUP  = "INFIELD"
    VALID_POSITIONS = ["1B", "2B", "3B", "SS"]

    # ── Features ──────────────────────────────────────────────────────────────
    INPUT_FEATURES = ["Age", "UZR/150"]
    FEATURE_WEIGHTS = {"Age": 1.0, "UZR/150": 1.0}

    # ── Reliability regression ─────────────────────────────────────────────────
    ENABLE_RELIABILITY_REGRESSION_TRAINING   = True
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True

    # ── Model hyperparameters (same architecture as main infield model) ────────
    HIDDEN_SIZE = 256
    NUM_LAYERS  = 2
    NUM_HEADS   = 4
    DROPOUT     = 0.15
    BIDIRECTIONAL = True
    GRADIENT_CLIP = 1.0

    # ── Training hyperparameters ───────────────────────────────────────────────
    BATCH_SIZE              = 64
    LEARNING_RATE           = 1e-3
    WEIGHT_DECAY            = 1e-5
    NUM_EPOCHS              = 50
    EARLY_STOPPING_PATIENCE = 10

    @staticmethod
    def get_data_config() -> DataConfig:
        return DataConfig(
            input_features=PretrainInfieldConfig.INPUT_FEATURES,
            seq_length=3,
            start_season=2002,   # UZR/150 available from 2002
            min_pa=50,
            train_ratio=0.7,
            valid_ratio=0.2,
            random_seed=42,
        )


# =============================================================================
# OUTFIELD (LF / CF / RF) — UZR/150
# =============================================================================

class PretrainOutfieldConfig:
    """
    Trade-analysis pretrained outfield model.

    Feature: UZR/150 — same rationale as infield; available from ~2002.
    """

    DATA_FILE      = _FIELDING_DATA_FILE
    CHECKPOINT_DIR = str(PRETRAIN_CHECKPOINTS_DIR)
    CHECKPOINT_FILE = "OF_fielding_pretrained.pth"
    # Saved by preprocess_data(model_type='defense_outfield', mode='pretrain')
    SCALER_FILE    = str(PRETRAIN_SCALER_DIR / "defense_outfield_pretrain_scaler.pkl")
    OUTPUT_FILE    = "../data/generated/trade_analysis/pipeline/defense_outfield_predictions.csv"

    POSITION_GROUP  = "OUTFIELD"
    VALID_POSITIONS = ["LF", "CF", "RF"]

    INPUT_FEATURES = ["Age", "UZR/150"]
    FEATURE_WEIGHTS = {"Age": 1.0, "UZR/150": 1.0}

    ENABLE_RELIABILITY_REGRESSION_TRAINING   = True
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True

    HIDDEN_SIZE = 256
    NUM_LAYERS  = 2
    NUM_HEADS   = 4
    DROPOUT     = 0.15
    BIDIRECTIONAL = True
    GRADIENT_CLIP = 1.0

    BATCH_SIZE              = 64
    LEARNING_RATE           = 1e-3
    WEIGHT_DECAY            = 1e-5
    NUM_EPOCHS              = 50
    EARLY_STOPPING_PATIENCE = 10

    @staticmethod
    def get_data_config() -> DataConfig:
        return DataConfig(
            input_features=PretrainOutfieldConfig.INPUT_FEATURES,
            seq_length=3,
            start_season=2002,
            min_pa=100,
            train_ratio=0.7,
            valid_ratio=0.2,
            random_seed=42,
        )


# =============================================================================
# CATCHER (C) — DRS/150
# =============================================================================

class PretrainCatcherConfig:
    """
    Trade-analysis pretrained catcher model.

    Feature: DRS/150 — Defensive Runs Saved (Baseball Info Solutions), the best
    comprehensive catcher defensive metric before Statcast.  Available from ~2004.
    DRS/150 is computed by calculate_rate_stats() from raw DRS / Inn * 150.
    """

    DATA_FILE      = _FIELDING_DATA_FILE
    CHECKPOINT_DIR = str(PRETRAIN_CHECKPOINTS_DIR)
    CHECKPOINT_FILE = "C_fielding_pretrained.pth"
    # Saved by preprocess_data(model_type='defense_catcher', mode='pretrain')
    SCALER_FILE    = str(PRETRAIN_SCALER_DIR / "defense_catcher_pretrain_scaler.pkl")
    OUTPUT_FILE    = "../data/generated/trade_analysis/pipeline/defense_catcher_predictions.csv"

    POSITION_GROUP  = "CATCHER"
    VALID_POSITIONS = ["C"]

    INPUT_FEATURES = ["Age", "DRS/150"]
    FEATURE_WEIGHTS = {"Age": 1.0, "DRS/150": 1.0}

    ENABLE_RELIABILITY_REGRESSION_TRAINING   = True
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = True

    # Simpler model (fewer features, smaller catcher sample)
    HIDDEN_SIZE = 128
    NUM_LAYERS  = 2
    NUM_HEADS   = 1
    DROPOUT     = 0.4
    BIDIRECTIONAL = False
    GRADIENT_CLIP = 1.0

    BATCH_SIZE              = 64
    LEARNING_RATE           = 1e-3
    WEIGHT_DECAY            = 1e-5
    NUM_EPOCHS              = 30
    EARLY_STOPPING_PATIENCE = 8

    @staticmethod
    def get_data_config() -> DataConfig:
        return DataConfig(
            input_features=PretrainCatcherConfig.INPUT_FEATURES,
            seq_length=4,
            start_season=2004,   # DRS available from BIS starting 2004
            min_pa=40,
            train_ratio=0.7,
            valid_ratio=0.29,
            random_seed=0,
        )


# =============================================================================
# BASERUNNING — BsR_rate + SB_rate + CS_rate
# =============================================================================

class PretrainBaserunningConfig:
    """
    Trade-analysis pretrained baserunning model.

    Features:
        BsR_rate — FanGraphs Baserunning Runs per 150 G (available ~2002+).
                   Computed manually as BsR / G * 150 in predict_pretrain.py
                   because BsR is not in the main rate_stats_config list.
        SB_rate  — Stolen bases per 150 G (computed by calculate_rate_stats).
        CS_rate  — Caught stealing per 150 G (computed by calculate_rate_stats).

    Uses the batting data file (same as main baserunning model), since BsR is
    a FanGraphs batting-side stat.
    """

    DATA_FILE      = _BATTING_DATA_FILE
    CHECKPOINT_DIR = str(PRETRAIN_CHECKPOINTS_DIR)
    CHECKPOINT_FILE = "baserunning_pretrained.pth"
    # Saved by preprocess_data(model_type='baserunning', mode='pretrain')
    SCALER_FILE    = str(PRETRAIN_SCALER_DIR / "baserunning_pretrain_scaler.pkl")
    OUTPUT_FILE    = "../data/generated/trade_analysis/pipeline/baserunning_predictions.csv"

    INPUT_FEATURES = ["Age", "BsR_rate", "SB_rate", "CS_rate"]
    FEATURE_WEIGHTS = {
        "Age":      1.0,
        "BsR_rate": 1.0,
        "SB_rate":  0.5,
        "CS_rate":  0.5,
    }

    ENABLE_RELIABILITY_REGRESSION_TRAINING   = False
    ENABLE_RELIABILITY_REGRESSION_PREDICTION = False

    HIDDEN_SIZE = 256
    NUM_LAYERS  = 3
    NUM_HEADS   = 2
    DROPOUT     = 0.5
    BIDIRECTIONAL = False
    GRADIENT_CLIP = 1.0

    BATCH_SIZE              = 64
    LEARNING_RATE           = 1e-3
    WEIGHT_DECAY            = 1e-5
    NUM_EPOCHS              = 50
    EARLY_STOPPING_PATIENCE = 10

    @staticmethod
    def get_data_config() -> DataConfig:
        return DataConfig(
            input_features=PretrainBaserunningConfig.INPUT_FEATURES,
            seq_length=3,
            start_season=2002,   # FanGraphs BsR available from ~2002
            min_pa=100,
            train_ratio=0.7,
            valid_ratio=0.29,
            random_seed=42,
        )


# =============================================================================
# Registry helper — mirrors the ModelFactory pattern without touching model_registry.py
# =============================================================================

_PRETRAIN_CONFIG_MAP = {
    "defense_infield":  PretrainInfieldConfig,
    "defense_outfield": PretrainOutfieldConfig,
    "defense_catcher":  PretrainCatcherConfig,
    "baserunning":      PretrainBaserunningConfig,
}


def get_pretrain_config(model_key: str):
    """Return the pretrain config class for *model_key*.

    Args:
        model_key: One of 'defense_infield', 'defense_outfield',
                   'defense_catcher', 'baserunning'.

    Raises:
        KeyError: If model_key is not known.
    """
    if model_key not in _PRETRAIN_CONFIG_MAP:
        raise KeyError(
            f"Unknown pretrain model key '{model_key}'. "
            f"Valid keys: {list(_PRETRAIN_CONFIG_MAP)}"
        )
    return _PRETRAIN_CONFIG_MAP[model_key]
