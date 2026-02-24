#!/usr/bin/env python3
"""
Trade Analysis — Pretrained Fielding & Baserunning Predictions
==============================================================

Generates fielding and baserunning predictions using the trade-analysis-owned
pretrained models (UZR/DRS/BsR-based).  These work for any cutoff year back to
2002/2004, covering the full trade-analysis historical range (2013+).

This script is intentionally isolated from the main ``predict_models.py`` so that
the main project configs and models are not modified.

CLI Usage:
    # Called automatically by generate_projections.py — rarely needed directly
    cd auto_train_models
    python -m trade_analysis.predict_pretrain --model-type fielding   --cutoff-year 2022 --output-dir /path/to/out
    python -m trade_analysis.predict_pretrain --model-type baserunning --cutoff-year 2022 --output-dir /path/to/out
    python -m trade_analysis.predict_pretrain --model-type all         --cutoff-year 2022 --output-dir /path/to/out
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import torch
import joblib

# ── Path setup ────────────────────────────────────────────────────────────────
_TRADE_ANALYSIS_DIR = Path(__file__).resolve().parent          # trade_analysis/
_AUTO_TRAIN_DIR     = _TRADE_ANALYSIS_DIR.parent               # auto_train_models/
_ROOT_DIR           = _AUTO_TRAIN_DIR.parent                   # LSTMLB/
_DATA_DIR           = _ROOT_DIR / "data"

sys.path.insert(0, str(_AUTO_TRAIN_DIR))

# ── Core imports (from main project) ─────────────────────────────────────────
from core.data_processing import calculate_rate_stats
from core.prediction import (
    load_model_from_checkpoint,
    predict_all_fielders,
    predict_all_baserunners,
    generate_batter_names,
)

# ── Trade-analysis pretrain configs ────────────────────────────────────────────
try:
    from trade_analysis.pretrain_configs import (
        PretrainInfieldConfig,
        PretrainOutfieldConfig,
        PretrainCatcherConfig,
        PretrainBaserunningConfig,
        get_pretrain_config,
    )
except ModuleNotFoundError:
    # Running as script directly (not as -m module) — pretrain_configs is a sibling file
    _TRADE_ANALYSIS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(_TRADE_ANALYSIS_DIR.parent))  # ensure auto_train_models/ is on path
    from pretrain_configs import (  # type: ignore[import]
        PretrainInfieldConfig,
        PretrainOutfieldConfig,
        PretrainCatcherConfig,
        PretrainBaserunningConfig,
        get_pretrain_config,
    )

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ── Path helpers ──────────────────────────────────────────────────────────────

def _resolve_data_path(config_data_file: str) -> Path:
    """
    Resolve a config's DATA_FILE path to an absolute ``Path``.

    Config paths look like ``'../data/historic_mlb/file.csv'`` relative to
    ``auto_train_models/``.  We extract the part after ``data/`` and join
    with the workspace DATA_DIR.
    """
    parts = Path(config_data_file).parts
    if "data" in parts:
        idx = parts.index("data")
        return _DATA_DIR / Path(*parts[idx + 1:])
    return _DATA_DIR / Path(config_data_file).name


# =============================================================================
# FIELDING PREDICTIONS
# =============================================================================

def _load_pretrain_fielding_models() -> tuple:
    """
    Load all three pretrained fielding models (infield / outfield / catcher).

    Returns
    -------
    position_models    : dict  {group → model}
    position_scalers   : dict  {group → scaler}
    position_group_map : dict  {pos  → group}
    input_features_map : dict  {group → feature list}
    seq_length_map     : dict  {group → seq_length}
    """
    config_map = {
        "infield":  PretrainInfieldConfig,
        "outfield": PretrainOutfieldConfig,
        "catcher":  PretrainCatcherConfig,
    }

    position_models    = {}
    position_scalers   = {}
    input_features_map = {}
    seq_length_map     = {}

    for pos_group, cfg in config_map.items():
        data_config = cfg.get_data_config()

        checkpoint_path = Path(cfg.CHECKPOINT_DIR) / cfg.CHECKPOINT_FILE
        scaler_path     = Path(cfg.SCALER_FILE)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained {pos_group} checkpoint not found: {checkpoint_path}\n"
                f"Run: python -m trade_analysis.train_pretrain_models --model defense_{pos_group}"
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Pretrained {pos_group} scaler not found: {scaler_path}\n"
                f"Run: python -m trade_analysis.train_pretrain_models --model defense_{pos_group}"
            )

        logger.info(f"  Loading pretrained {pos_group} model ({checkpoint_path.name})")
        model  = load_model_from_checkpoint(str(checkpoint_path), data_config, device)
        scaler = joblib.load(scaler_path)

        position_models[pos_group]    = model
        position_scalers[pos_group]   = scaler
        input_features_map[pos_group] = cfg.INPUT_FEATURES
        seq_length_map[pos_group]     = data_config.seq_length

    position_group_map = {
        "C":  "catcher",
        "1B": "infield", "2B": "infield", "3B": "infield", "SS": "infield",
        "LF": "outfield", "CF": "outfield", "RF": "outfield",
    }

    return position_models, position_scalers, position_group_map, input_features_map, seq_length_map


def generate_pretrain_fielding_predictions(
    output_file: str = None,
    cutoff_year: int = None,
    use_aging_enforcer: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Generate fielding predictions with pretrained (UZR/DRS-based) models.

    Args:
        output_file:        Path for the output CSV.
        cutoff_year:        Last year of actual data (predictions start cutoff_year + 1).
        use_aging_enforcer: Apply aging constraints at prediction time.
    """
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1

    logger.info(f"[pretrain fielding] cutoff_year={cutoff_year}")

    # Load data
    data_file = _resolve_data_path(PretrainInfieldConfig.DATA_FILE)
    logger.info(f"  Data file: {data_file}")
    raw_df = pd.read_csv(data_file)

    # Compute rate stats (derives DRS/150 from DRS/Inn*150; UZR/150 already present)
    raw_df = calculate_rate_stats(raw_df)

    player_names = pd.DataFrame(
        raw_df[["Name", "IDfg"]].drop_duplicates()
    ).sort_values("Name")

    # Load models
    logger.info("  Loading pretrained fielding models…")
    (
        position_models,
        position_scalers,
        position_group_map,
        input_features_map,
        seq_length_map,
    ) = _load_pretrain_fielding_models()

    # Generate predictions
    predictions_df = predict_all_fielders(
        raw_df=raw_df,
        player_names=player_names,
        position_models=position_models,
        position_scalers=position_scalers,
        position_group_map=position_group_map,
        input_features_map=input_features_map,
        seq_length_map=seq_length_map,
        future_years=15,
        cutoff_year=cutoff_year,
        use_aging_enforcer=use_aging_enforcer,
    )

    if predictions_df is None:
        logger.error("[pretrain fielding] predict_all_fielders returned None")
        return None

    # Reorder columns: metadata first
    metadata_cols = ["Name", "Age", "Year", "IDfg", "Pos"]
    feature_cols  = [c for c in predictions_df.columns if c not in metadata_cols]
    predictions_df = predictions_df[metadata_cols + feature_cols]

    out_path = output_file or str(
        _DATA_DIR / "generated" / "trade_analysis" / "pipeline" / "fielding_predictions.csv"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(out_path, index=False)
    logger.info(
        f"  Saved {len(predictions_df)} fielding predictions "
        f"({predictions_df['Name'].nunique()} players) → {out_path}"
    )
    return predictions_df


# =============================================================================
# BASERUNNING PREDICTIONS
# =============================================================================

def generate_pretrain_baserunning_predictions(
    output_file: str = None,
    cutoff_year: int = None,
) -> Optional[pd.DataFrame]:
    """
    Generate baserunning predictions with the pretrained (BsR-based) model.

    BsR_rate is computed here (BsR / G * 150) because BsR is not in the main
    project's rate_stats_config.  SB_rate and CS_rate are computed by the
    standard calculate_rate_stats().

    Args:
        output_file:  Path for the output CSV.
        cutoff_year:  Last year of actual data (predictions start cutoff_year + 1).
    """
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1

    logger.info(f"[pretrain baserunning] cutoff_year={cutoff_year}")

    cfg = PretrainBaserunningConfig

    # Load data
    data_file = _resolve_data_path(cfg.DATA_FILE)
    logger.info(f"  Data file: {data_file}")
    raw_df = pd.read_csv(data_file)

    # Compute standard rate stats (gives SB_rate, CS_rate, etc.)
    raw_df = calculate_rate_stats(raw_df)

    # Compute BsR_rate manually (not in main rate_stats_config)
    if "BsR" in raw_df.columns and "G" in raw_df.columns:
        raw_df["BsR_rate"] = (
            raw_df["BsR"] / raw_df["G"].replace(0, np.nan) * 150
        )
        logger.info("  Computed BsR_rate = BsR / G * 150")
    else:
        missing = [c for c in ("BsR", "G") if c not in raw_df.columns]
        raise KeyError(
            f"Required columns missing from batting data: {missing}. "
            f"Cannot compute BsR_rate for pretrained baserunning model."
        )

    player_names = generate_batter_names(raw_df)

    # Load model
    data_config     = cfg.get_data_config()
    checkpoint_path = Path(cfg.CHECKPOINT_DIR) / cfg.CHECKPOINT_FILE
    scaler_path     = Path(cfg.SCALER_FILE)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained baserunning checkpoint not found: {checkpoint_path}\n"
            "Run: python -m trade_analysis.train_pretrain_models --model baserunning"
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Pretrained baserunning scaler not found: {scaler_path}\n"
            "Run: python -m trade_analysis.train_pretrain_models --model baserunning"
        )

    logger.info(f"  Loading pretrained baserunning model ({checkpoint_path.name})")
    model  = load_model_from_checkpoint(str(checkpoint_path), data_config, device)
    scaler = joblib.load(scaler_path)

    predictions_df = predict_all_baserunners(
        raw_df=raw_df,
        player_names=player_names,
        model=model,
        scaler=scaler,
        input_features=cfg.INPUT_FEATURES,
        seq_length=data_config.seq_length,
        future_years=15,
        cutoff_year=cutoff_year,
    )

    if predictions_df is None:
        logger.error("[pretrain baserunning] predict_all_baserunners returned None")
        return None

    out_path = output_file or str(
        _DATA_DIR / "generated" / "trade_analysis" / "pipeline" / "baserunning_predictions.csv"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(out_path, index=False)
    logger.info(
        f"  Saved {len(predictions_df)} baserunning predictions "
        f"({predictions_df['Name'].nunique()} players) → {out_path}"
    )
    return predictions_df


# =============================================================================
# CLI
# =============================================================================

def main():
    default_cutoff = datetime.now().year - 1

    parser = argparse.ArgumentParser(
        description=(
            "Generate fielding / baserunning predictions using trade-analysis "
            "pretrained (UZR / DRS / BsR) models."
        )
    )
    parser.add_argument(
        "--model-type",
        choices=["fielding", "baserunning", "all"],
        default="all",
        help="Which predictions to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write prediction CSVs (default: data/generated/trade_analysis/pipeline/)",
    )
    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=default_cutoff,
        help=f"Last year of actual data (default: {default_cutoff})",
    )
    parser.add_argument(
        "--use-aging-enforcer",
        action="store_true",
        help="Apply aging constraints to fielding predictions",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    out_dir = Path(args.output_dir) if args.output_dir else None

    success = True

    if args.model_type in ("fielding", "all"):
        out_file = str(out_dir / "fielding_predictions.csv") if out_dir else None
        result = generate_pretrain_fielding_predictions(
            output_file=out_file,
            cutoff_year=args.cutoff_year,
            use_aging_enforcer=args.use_aging_enforcer,
        )
        if result is None:
            success = False

    if args.model_type in ("baserunning", "all"):
        out_file = str(out_dir / "baserunning_predictions.csv") if out_dir else None
        result = generate_pretrain_baserunning_predictions(
            output_file=out_file,
            cutoff_year=args.cutoff_year,
        )
        if result is None:
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
