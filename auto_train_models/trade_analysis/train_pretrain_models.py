#!/usr/bin/env python3
"""
Trade Analysis — Train Pretrained Fielding & Baserunning Models
===============================================================

Trains four lightweight LSTM models that use pre-Statcast metrics so they work
for any historical cutoff year back to 2002/2004.  These models are owned by
the trade-analysis workflow and are completely separate from the main project's
checkpoint files.

Models trained:
  defense_infield  — UZR/150   (available 2002+)
  defense_outfield — UZR/150   (available 2002+)
  defense_catcher  — DRS/150   (available 2004+)
  baserunning      — BsR_rate + SB_rate + CS_rate (available 2002+)

Checkpoints are saved to:
  auto_train_models/trade_analysis/checkpoints/{filename}

Scalers are saved to:
  auto_train_models/data/{model_type}_pretrain_scaler.pkl
  (the standard location used by core.data_processing.scale_features)

Usage:
    cd auto_train_models
    python -m trade_analysis.train_pretrain_models --model all
    python -m trade_analysis.train_pretrain_models --model defense_infield
    python -m trade_analysis.train_pretrain_models --model baserunning --epochs 30
"""

import sys
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
_TRADE_ANALYSIS_DIR = Path(__file__).resolve().parent
_AUTO_TRAIN_DIR     = _TRADE_ANALYSIS_DIR.parent
_ROOT_DIR           = _AUTO_TRAIN_DIR.parent
_DATA_DIR           = _ROOT_DIR / "data"

sys.path.insert(0, str(_AUTO_TRAIN_DIR))

# ── Core imports ──────────────────────────────────────────────────────────────
from core.utils import setup_logging, set_random_seeds, get_device
from core.data_processing import preprocess_data, calculate_rate_stats
from core.model_architecture import ImprovedLSTM
from core.training import create_data_loaders, Config as TrainConfig, train_model
from core.losses import InningsWeightedLoss, PlayerDifferentiationLoss

# ── Trade-analysis pretrain configs ───────────────────────────────────────────
try:
    from trade_analysis.pretrain_configs import (
        PretrainInfieldConfig,
        PretrainOutfieldConfig,
        PretrainCatcherConfig,
        PretrainBaserunningConfig,
        PRETRAIN_CHECKPOINTS_DIR,
        PRETRAIN_SCALER_DIR,
        get_pretrain_config,
        AUTO_TRAIN_DIR as _TA_AUTO_TRAIN_DIR,
    )
except ModuleNotFoundError:
    # Running as script directly (not as -m module) — pretrain_configs is a sibling file
    sys.path.insert(0, str(_TRADE_ANALYSIS_DIR.parent))  # ensure auto_train_models/ is on path
    from pretrain_configs import (  # type: ignore[import]
        PretrainInfieldConfig,
        PretrainOutfieldConfig,
        PretrainCatcherConfig,
        PretrainBaserunningConfig,
        PRETRAIN_CHECKPOINTS_DIR,
        PRETRAIN_SCALER_DIR,
        get_pretrain_config,
        AUTO_TRAIN_DIR as _TA_AUTO_TRAIN_DIR,
    )

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper: resolve data file path (mirrors predict_models.py's logic)
# =============================================================================

def _resolve_data_path(config_data_file: str) -> Path:
    parts = Path(config_data_file).parts
    if "data" in parts:
        idx = parts.index("data")
        return _DATA_DIR / Path(*parts[idx + 1:])
    return _DATA_DIR / Path(config_data_file).name


# =============================================================================
# Helper: prepare baserunning data file with BsR_rate pre-computed
# =============================================================================

_DERIVED_BASERUNNING_FILE = _TRADE_ANALYSIS_DIR / "data" / "batting_for_baserunning_pretrain.csv"


def _prepare_baserunning_data_file() -> str:
    """
    Create a derived batting CSV that includes the BsR_rate column.

    ``calculate_rate_stats()`` doesn't include BsR in its list, so we add it
    here.  The derived file is saved to trade_analysis/data/ and reused across
    subsequent training runs unless the source batting file changes.

    Returns
    -------
    str: Absolute path to the derived CSV.
    """
    batting_path = _resolve_data_path(PretrainBaserunningConfig.DATA_FILE)

    # Skip regeneration if already up-to-date
    if (_DERIVED_BASERUNNING_FILE.exists() and
            _DERIVED_BASERUNNING_FILE.stat().st_mtime >= batting_path.stat().st_mtime):
        logger.info(f"Derived baserunning data already up-to-date: {_DERIVED_BASERUNNING_FILE}")
        return str(_DERIVED_BASERUNNING_FILE)

    logger.info(f"Preparing derived baserunning data file (adds BsR_rate) …")
    df = pd.read_csv(batting_path)
    df = calculate_rate_stats(df)

    if "BsR" not in df.columns or "G" not in df.columns:
        raise KeyError(
            "Columns 'BsR' and/or 'G' not found in batting data. "
            "Cannot compute BsR_rate."
        )
    df["BsR_rate"] = df["BsR"] / df["G"].replace(0, np.nan) * 150

    _DERIVED_BASERUNNING_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_DERIVED_BASERUNNING_FILE, index=False)
    logger.info(f"Saved derived file → {_DERIVED_BASERUNNING_FILE}")
    return str(_DERIVED_BASERUNNING_FILE)


# =============================================================================
# Core training routine
# =============================================================================

def train_pretrain_model(
    model_key: str,    # 'defense_infield' | 'defense_outfield' | 'defense_catcher' | 'baserunning'
    cfg,               # pretrain config class
    data_file: str,    # path to data CSV (may be derived)
    device,
    num_epochs_override: Optional[int] = None,
) -> dict:
    """Train a single pretrained model and save its checkpoint.

    ``model_key`` must be one of the standard model types that
    ``apply_model_specific_filters`` recognises (e.g. 'defense_infield'),
    because position filtering happens inside ``preprocess_data``.

    Returns the training metrics dict from ``train_model``.
    """
    logger.info("=" * 60)
    logger.info(f"Training pretrained {model_key} model")
    logger.info(f"  Features:   {cfg.INPUT_FEATURES}")
    logger.info(f"  Checkpoint: {PRETRAIN_CHECKPOINTS_DIR / cfg.CHECKPOINT_FILE}")
    logger.info("=" * 60)

    data_config = cfg.get_data_config()

    # ── Preprocess ──────────────────────────────────────────────────────────
    # preprocess_data saves the scaler to:
    #   auto_train_models/data/{model_key}_pretrain_scaler.pkl
    # That path matches cfg.SCALER_FILE (set in pretrain_configs.py).
    data = preprocess_data(
        data_file,
        data_config,
        model_type=model_key,
        mode="pretrain",
    )

    data_batch = create_data_loaders(data)

    # ── Training config ──────────────────────────────────────────────────────
    train_config = TrainConfig(
        data_batch.train.tensors[0],
        data_batch.train.tensors[-1],
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        num_heads=cfg.NUM_HEADS,
        learning_rate=cfg.LEARNING_RATE,
        dropout=cfg.DROPOUT,
        bidirectional=cfg.BIDIRECTIONAL,
        gradient_clip=getattr(cfg, "GRADIENT_CLIP", 1.0),
        batch_size=getattr(cfg, "BATCH_SIZE", 64),
    )
    train_config.num_epochs              = num_epochs_override or cfg.NUM_EPOCHS
    train_config.early_stopping_patience = cfg.EARLY_STOPPING_PATIENCE

    # ── Data loaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        data_batch.train,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    )
    valid_loader = DataLoader(
        data_batch.valid,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = ImprovedLSTM(
        input_size=train_config.input_size,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        output_size=train_config.output_size,
        dropout=cfg.DROPOUT,
        bidirectional=cfg.BIDIRECTIONAL,
        num_heads=cfg.NUM_HEADS,
    ).to(device)

    logger.info(
        f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=getattr(cfg, "WEIGHT_DECAY", 1e-5),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.learning_rate,
        epochs=train_config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=min(
            getattr(train_config, "warmup_epochs", 5) / train_config.num_epochs, 0.3
        ),
        anneal_strategy="cos",
        final_div_factor=1e4,
    )

    # ── Loss function ────────────────────────────────────────────────────────
    if model_key.startswith("defense"):
        criterion = InningsWeightedLoss(
            feature_weights=cfg.FEATURE_WEIGHTS
        ).to(device)
        logger.info(f"  Loss: InningsWeightedLoss  weights={cfg.FEATURE_WEIGHTS}")
    else:
        criterion = PlayerDifferentiationLoss().to(device)
        logger.info("  Loss: PlayerDifferentiationLoss")

    # ── Train ────────────────────────────────────────────────────────────────
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=train_config,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        checkpoint_dir=str(PRETRAIN_CHECKPOINTS_DIR),
        checkpoint_filename=cfg.CHECKPOINT_FILE,
        training_mode="pretrain",
        input_features=data_config.input_features,
    )

    logger.info(f"  Best epoch: {metrics['best_epoch']}")
    logger.info(
        f"  Best val loss: {min(metrics['val_losses']):.4f}"
    )
    return metrics


# =============================================================================
# Public entry point
# =============================================================================

_MODEL_TRAINING_ORDER = [
    "defense_infield",
    "defense_outfield",
    "defense_catcher",
    "baserunning",
]

_FIELDING_DATA_FILE = str(
    _DATA_DIR / "historic_mlb" / "mlb_fielding_data_2000_2025_with_statcast.csv"
)


def train_all_pretrain_models(
    model_keys: list[str] | None = None,
    num_epochs_override: int | None = None,
) -> dict[str, dict]:
    """
    Train all (or a subset of) pretrained trade-analysis models.

    Args:
        model_keys:           Subset of MODEL_TRAINING_ORDER to train.
                              Defaults to all four.
        num_epochs_override:  Override NUM_EPOCHS in the config.

    Returns:
        Mapping of model_key → training metrics dict.
    """
    set_random_seeds()
    device = get_device()

    model_keys = model_keys or _MODEL_TRAINING_ORDER
    results = {}

    for key in model_keys:
        cfg = get_pretrain_config(key)

        if key == "baserunning":
            data_file = _prepare_baserunning_data_file()
        else:
            data_file = _FIELDING_DATA_FILE

        results[key] = train_pretrain_model(
            model_key=key,
            cfg=cfg,
            data_file=data_file,
            device=device,
            num_epochs_override=num_epochs_override,
        )

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train trade-analysis pretrained fielding / baserunning models "
            "(UZR / DRS / BsR-based, covers pre-2016 historical cutoff years)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all four pretrained models
  python -m trade_analysis.train_pretrain_models --model all

  # Train only the infield model with 30 epochs
  python -m trade_analysis.train_pretrain_models --model defense_infield --epochs 30

  # Train baserunning model only
  python -m trade_analysis.train_pretrain_models --model baserunning
        """,
    )
    parser.add_argument(
        "--model",
        choices=_MODEL_TRAINING_ORDER + ["all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override NUM_EPOCHS from config",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    model_keys = _MODEL_TRAINING_ORDER if args.model == "all" else [args.model]

    results = train_all_pretrain_models(
        model_keys=model_keys,
        num_epochs_override=args.epochs,
    )

    logger.info("=" * 60)
    logger.info("Training complete")
    for key, metrics in results.items():
        logger.info(
            f"  {key}: best_epoch={metrics['best_epoch']}, "
            f"best_val_loss={min(metrics['val_losses']):.4f}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
