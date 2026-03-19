"""
Historical Values — Prediction Engine
=======================================

Generates LSTM predictions for all four model types (batter, pitcher,
fielding, baserunning) at each cutoff year.  Calls the same core
prediction functions the production pipeline uses, but always with
pretrained (classical-feature) models for batters/pitchers and the
historical UZR/DRS/BsR configs for fielding/baserunning.

Output per cutoff year::

    data/generated/historical_values/projections/cutoff_{Y}/
        batter_predictions.csv
        pitcher_predictions.csv
        fielding_predictions.csv
        baserunning_predictions.csv

Usage (standalone):
    cd auto_train_models
    python -m historical_values.predictions --start 2013 --end 2025

Usage (from pipeline):
    from historical_values.predictions import generate_all_predictions
    generate_all_predictions(start=2013, end=2025)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib

# ── Path setup ────────────────────────────────────────────────────────────────
_AUTO_TRAIN_DIR = Path(__file__).resolve().parents[1]   # auto_train_models/
_ROOT_DIR       = _AUTO_TRAIN_DIR.parent                # LSTMLB/
_DATA_DIR       = _ROOT_DIR / "data"

if str(_AUTO_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN_DIR))

from historical_values.config import Config, logger, PROJECTIONS_DIR

# Core prediction functions
from core.data_processing import DataConfig, calculate_rate_stats
from core.prediction import (
    load_model_from_checkpoint,
    predict_all_batters,
    predict_all_fielders,
    predict_all_baserunners,
    generate_batter_names,
)
from core.pitcher_prediction import predict_all_pitchers

# Model configs
from models.model_registry import ModelFactory

# Historical configs for fielding and baserunning
from configs.defense_infield_historical import DefenseInfieldHistoricalConfig
from configs.defense_outfield_historical import DefenseOutfieldHistoricalConfig
from configs.defense_catcher_historical import DefenseCatcherHistoricalConfig
from configs.baserunning_historical import BaserunningHistoricalConfig

# ── Device ────────────────────────────────────────────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_data_path(config_data_file: str) -> Path:
    """Resolve a config's relative DATA_FILE path to an absolute Path."""
    parts = Path(config_data_file).parts
    if "data" in parts:
        idx = parts.index("data")
        return _DATA_DIR / Path(*parts[idx + 1 :])
    return _DATA_DIR / Path(config_data_file).name


def _output_dir(cutoff_year: int) -> Path:
    return PROJECTIONS_DIR / f"cutoff_{cutoff_year}"


def _is_complete(cutoff_year: int) -> bool:
    out = _output_dir(cutoff_year)
    expected = [
        "batter_predictions.csv",
        "pitcher_predictions.csv",
        "fielding_predictions.csv",
        "baserunning_predictions.csv",
    ]
    return all((out / f).exists() for f in expected)


# ═══════════════════════════════════════════════════════════════════════════════
# BATTER PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_batter_predictions(
    cutoff_year: int,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """Generate batter predictions using the pretrained (classical) model."""

    config = ModelFactory.get_config("batter")
    data_path = _resolve_data_path(config.DATA_FILE)

    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)
    player_names = generate_batter_names(raw_df)

    # Always use pretrained model for historical consistency
    data_config = config.get_data_config()
    if hasattr(config, "PRETRAIN_CHECKPOINT_FILE"):
        ckpt = _AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.PRETRAIN_CHECKPOINT_FILE
    else:
        ckpt = _AUTO_TRAIN_DIR / config.CHECKPOINT_DIR / config.CHECKPOINT_FILE
    scaler_path = _AUTO_TRAIN_DIR / config.SCALER_FILE

    if not ckpt.exists():
        logger.error(f"Batter checkpoint not found: {ckpt}")
        return None

    model = load_model_from_checkpoint(str(ckpt), data_config, _device)
    scaler = joblib.load(scaler_path)

    predictions = predict_all_batters(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=config.INPUT_FEATURES,
        seq_length=data_config.seq_length,
        future_years=Config.PROJECTION_HORIZON,
        cutoff_year=cutoff_year,
        min_pa_current=getattr(config, "MIN_PA_CURRENT", 100),
    )

    if predictions is None:
        logger.error(f"  batter cutoff={cutoff_year}: predict_all_batters returned None")
        return None

    out_path = out_dir / "batter_predictions.csv"
    predictions.to_csv(out_path, index=False)
    logger.info(
        f"  batter cutoff={cutoff_year}: {len(predictions)} rows, "
        f"{predictions['Name'].nunique()} players → {out_path.name}"
    )
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# PITCHER PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_pitcher_predictions(
    cutoff_year: int,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """Generate pitcher predictions using pretrained SP + RP models."""

    sp_config = ModelFactory.get_config("pitcher_sp")
    data_path = _resolve_data_path(sp_config.DATA_FILE)

    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)

    # Player names
    pitcher_names_path = _DATA_DIR / "pitcher_names.csv"
    if pitcher_names_path.exists():
        player_names = pd.read_csv(pitcher_names_path)
    else:
        player_names = pd.DataFrame(
            raw_df[["Name", "IDfg"]].drop_duplicates()
        ).sort_values("Name")

    # Always use pretrained for both SP and RP
    sp_data_config = sp_config.get_data_config(mode="pretrain")
    sp_ckpt = _AUTO_TRAIN_DIR / sp_config.CHECKPOINT_DIR / sp_config.PRETRAIN_CHECKPOINT_FILE
    sp_scaler_path = _AUTO_TRAIN_DIR / sp_config.PRETRAIN_SCALER_FILE

    rp_config = ModelFactory.get_config("pitcher_rp")
    rp_data_config = rp_config.get_data_config(mode="pretrain")
    rp_ckpt = _AUTO_TRAIN_DIR / rp_config.CHECKPOINT_DIR / rp_config.PRETRAIN_CHECKPOINT_FILE
    rp_scaler_path = _AUTO_TRAIN_DIR / rp_config.PRETRAIN_SCALER_FILE

    unified = getattr(sp_config, "UNIFIED_PITCHER_MODEL", False)

    for path, label in [(sp_ckpt, "SP"), (rp_ckpt, "RP")]:
        if not path.exists() and not (unified and label == "RP"):
            logger.error(f"Pitcher {label} checkpoint not found: {path}")
            return None

    sp_model = load_model_from_checkpoint(str(sp_ckpt), sp_data_config, _device)
    sp_scaler = joblib.load(sp_scaler_path)

    if unified:
        rp_model, rp_scaler = sp_model, sp_scaler
    else:
        rp_model = load_model_from_checkpoint(str(rp_ckpt), rp_data_config, _device)
        rp_scaler = joblib.load(rp_scaler_path)

    predictions = predict_all_pitchers(
        raw_df=raw_df,
        player_names=player_names,
        sp_model=sp_model,
        rp_model=rp_model,
        sp_scaler=sp_scaler,
        rp_scaler=rp_scaler,
        sp_input_features=sp_data_config.input_features,
        rp_input_features=rp_data_config.input_features,
        seq_length=sp_data_config.seq_length,
        future_years=Config.PROJECTION_HORIZON,
        cutoff_year=cutoff_year,
        sp_config=sp_config,
        rp_config=rp_config,
    )

    if predictions is None:
        logger.error(f"  pitcher cutoff={cutoff_year}: predict_all_pitchers returned None")
        return None

    out_path = out_dir / "pitcher_predictions.csv"
    predictions.to_csv(out_path, index=False)
    logger.info(
        f"  pitcher cutoff={cutoff_year}: {len(predictions)} rows, "
        f"{predictions['Name'].nunique()} players → {out_path.name}"
    )
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# FIELDING PREDICTIONS  (historical UZR/DRS configs)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_fielding_predictions(
    cutoff_year: int,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """Generate fielding predictions with historical LSTM models."""

    config_map = {
        "infield":  DefenseInfieldHistoricalConfig,
        "outfield": DefenseOutfieldHistoricalConfig,
        "catcher":  DefenseCatcherHistoricalConfig,
    }

    data_path = _resolve_data_path(DefenseInfieldHistoricalConfig.DATA_FILE)
    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)

    player_names = pd.DataFrame(
        raw_df[["Name", "IDfg"]].drop_duplicates()
    ).sort_values("Name")

    position_models  = {}
    position_scalers = {}
    input_features_map = {}
    seq_length_map = {}

    for pos_group, cfg in config_map.items():
        data_config = cfg.get_data_config()
        ckpt = Path(cfg.CHECKPOINT_DIR) / cfg.CHECKPOINT_FILE
        scaler_path = Path(cfg.SCALER_FILE)

        if not ckpt.exists():
            logger.error(
                f"Historical {pos_group} checkpoint not found: {ckpt}\n"
                f"Train with: python scripts/train_models.py "
                f"--model defense_{pos_group}_historical --pretrain"
            )
            return None
        if not scaler_path.exists():
            logger.error(f"Historical {pos_group} scaler not found: {scaler_path}")
            return None

        model = load_model_from_checkpoint(str(ckpt), data_config, _device)
        scaler = joblib.load(scaler_path)

        position_models[pos_group]    = model
        position_scalers[pos_group]   = scaler
        input_features_map[pos_group] = cfg.INPUT_FEATURES
        seq_length_map[pos_group]     = data_config.seq_length

    position_group_map = {
        "C": "catcher",
        "1B": "infield", "2B": "infield", "3B": "infield", "SS": "infield",
        "LF": "outfield", "CF": "outfield", "RF": "outfield",
    }

    predictions = predict_all_fielders(
        raw_df=raw_df,
        player_names=player_names,
        position_models=position_models,
        position_scalers=position_scalers,
        position_group_map=position_group_map,
        input_features_map=input_features_map,
        seq_length_map=seq_length_map,
        future_years=Config.PROJECTION_HORIZON,
        cutoff_year=cutoff_year,
    )

    if predictions is None:
        logger.error(f"  fielding cutoff={cutoff_year}: predict_all_fielders returned None")
        return None

    # Ensure metadata columns are first
    meta = ["Name", "Age", "Year", "IDfg", "Pos"]
    feat = [c for c in predictions.columns if c not in meta]
    predictions = predictions[meta + feat]

    out_path = out_dir / "fielding_predictions.csv"
    predictions.to_csv(out_path, index=False)
    logger.info(
        f"  fielding cutoff={cutoff_year}: {len(predictions)} rows, "
        f"{predictions['Name'].nunique()} players → {out_path.name}"
    )
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# BASERUNNING PREDICTIONS  (historical BsR config)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_baserunning_predictions(
    cutoff_year: int,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """Generate baserunning predictions with the historical LSTM model."""

    cfg = BaserunningHistoricalConfig
    data_path = _resolve_data_path(cfg.DATA_FILE)
    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)

    # BsR_rate is now produced by calculate_rate_stats() (BsR / G * 150)
    if "BsR_rate" not in raw_df.columns:
        logger.error("BsR_rate missing after calculate_rate_stats() — check that BsR exists in the data")
        return None

    player_names = generate_batter_names(raw_df)

    data_config = cfg.get_data_config()
    ckpt = Path(cfg.CHECKPOINT_DIR) / cfg.CHECKPOINT_FILE
    scaler_path = Path(cfg.SCALER_FILE)

    if not ckpt.exists():
        logger.error(
            f"Historical baserunning checkpoint not found: {ckpt}\n"
            "Train with: python scripts/train_models.py --model baserunning_historical --pretrain"
        )
        return None
    if not scaler_path.exists():
        logger.error(f"Historical baserunning scaler not found: {scaler_path}")
        return None

    model = load_model_from_checkpoint(str(ckpt), data_config, _device)
    scaler = joblib.load(scaler_path)

    predictions = predict_all_baserunners(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=cfg.INPUT_FEATURES,
        seq_length=data_config.seq_length,
        future_years=Config.PROJECTION_HORIZON,
        cutoff_year=cutoff_year,
    )

    if predictions is None:
        logger.error(f"  baserunning cutoff={cutoff_year}: predict_all_baserunners returned None")
        return None

    out_path = out_dir / "baserunning_predictions.csv"
    predictions.to_csv(out_path, index=False)
    logger.info(
        f"  baserunning cutoff={cutoff_year}: {len(predictions)} rows, "
        f"{predictions['Name'].nunique()} players → {out_path.name}"
    )
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_TYPES = ["batter", "pitcher", "fielding", "baserunning"]

_GENERATORS = {
    "batter":      _generate_batter_predictions,
    "pitcher":     _generate_pitcher_predictions,
    "fielding":    _generate_fielding_predictions,
    "baserunning": _generate_baserunning_predictions,
}


def generate_predictions_for_year(
    cutoff_year: int,
    force: bool = False,
    model_types: list[str] | None = None,
) -> bool:
    """Generate predictions for a single cutoff year.

    Returns True if all requested model types succeeded.
    """
    model_types = model_types or MODEL_TYPES
    out_dir = _output_dir(cutoff_year)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for mt in model_types:
        out_file = out_dir / f"{mt}_predictions.csv"
        if out_file.exists() and not force:
            logger.info(f"  {mt} cutoff={cutoff_year}: already exists, skipping")
            continue

        gen = _GENERATORS.get(mt)
        if gen is None:
            logger.error(f"  Unknown model type: {mt}")
            all_ok = False
            continue

        result = gen(cutoff_year, out_dir)
        if result is None:
            all_ok = False

    return all_ok


def generate_all_predictions(
    start: int | None = None,
    end: int | None = None,
    force: bool = False,
    model_types: list[str] | None = None,
) -> None:
    """Generate predictions for every cutoff year in the configured range."""
    start = start or Config.CUTOFF_START
    end   = end   or Config.CUTOFF_END

    Config.ensure_directories()

    logger.info("=" * 60)
    logger.info("Historical Values — Generate Predictions")
    logger.info(f"Cutoff years {start} → {end}  (horizon={Config.PROJECTION_HORIZON})")
    logger.info("=" * 60)

    for cutoff_year in range(start, end + 1):
        if _is_complete(cutoff_year) and not force:
            logger.info(f"[cutoff={cutoff_year}] complete — skipping")
            continue

        logger.info(f"[cutoff={cutoff_year}] generating predictions …")
        ok = generate_predictions_for_year(cutoff_year, force=force, model_types=model_types)
        if not ok:
            logger.warning(f"[cutoff={cutoff_year}] some model types failed")

    logger.info("Prediction generation complete.")
