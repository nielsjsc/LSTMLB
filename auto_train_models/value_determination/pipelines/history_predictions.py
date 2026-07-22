"""
Historical Values — Prediction Engine
=======================================

Generates Marcel predictions for all four model types (batter, pitcher,
fielding, baserunning) at each cutoff year.

Output per cutoff year::

    data/generated/historical_values/projections/cutoff_{Y}/
        batter_predictions.csv
        pitcher_predictions.csv
        fielding_predictions.csv
        baserunning_predictions.csv

Usage (standalone):
    cd auto_train_models
    python -m value_determination.pipelines.history_predictions --start 2013 --end 2025

Usage (from pipeline):
    from value_determination.pipelines.history_predictions import generate_all_predictions
    generate_all_predictions(start=2013, end=2025)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_AUTO_TRAIN_DIR = Path(__file__).resolve().parents[3]   # auto_train_models/
_ROOT_DIR       = _AUTO_TRAIN_DIR.parent                # LSTMLB/
_DATA_DIR       = _ROOT_DIR / "data"

if str(_AUTO_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN_DIR))

from value_determination.config import Config, logger, CURRENT_YEAR
PROJECTIONS_DIR = Config.Paths.PROJECTIONS_DIR

# Core prediction functions
from core.data_processing import calculate_rate_stats, generate_batter_names, generate_pitcher_names
from core.marcel_projections import (
    marcel_fielding_projections,
    marcel_baserunning_projections,
    marcel_batter_projections,
    marcel_pitcher_projections,
)
from core.position_profiles import build_position_profiles, load_batting_for_games

# Model configs
from models.model_registry import ModelFactory

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
    """Generate batter predictions using Marcel projections."""

    config = ModelFactory.get_config("batter")
    
    # Use statcast data for x-stats when available
    if hasattr(config, "FINETUNE_DATA_FILE"):
        data_path = _resolve_data_path(config.FINETUNE_DATA_FILE)
    else:
        data_path = _resolve_data_path(config.DATA_FILE)

    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)
    player_names = generate_batter_names(raw_df)

    predictions = marcel_batter_projections(
        raw_df=raw_df,
        player_names=player_names,
        future_years=Config.History.PROJECTION_HORIZON,
        cutoff_year=cutoff_year,
        use_xstats=getattr(config, "USE_XWOBA_FOR_PREDICTIONS", True),
    )

    if predictions is None:
        logger.error(f"  batter cutoff={cutoff_year}: marcel_batter_projections returned None")
        return None

    out_path = out_dir / "batter_predictions.csv"
    predictions.to_csv(out_path, index=False)
    logger.info(
        f"  batter cutoff={cutoff_year} [marcel]: {len(predictions)} rows, "
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
    """Generate pitcher predictions using Marcel projections."""

    sp_config = ModelFactory.get_config("pitcher_sp")

    data_path = _resolve_data_path(sp_config.DATA_FILE)
    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)

    player_names = generate_pitcher_names(raw_df)

    predictions = marcel_pitcher_projections(
        raw_df=raw_df,
        player_names=player_names,
        future_years=Config.History.PROJECTION_HORIZON,
        cutoff_year=cutoff_year,
    )

    if predictions is None:
        logger.error(f"  pitcher cutoff={cutoff_year}: marcel_pitcher_projections returned None")
        return None

    out_path = out_dir / "pitcher_predictions.csv"
    predictions.to_csv(out_path, index=False)
    logger.info(
        f"  pitcher cutoff={cutoff_year} [marcel]: {len(predictions)} rows, "
        f"{predictions['Name'].nunique()} players → {out_path.name}"
    )
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# FIELDING PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_fielding_predictions(
    cutoff_year: int,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """Generate fielding predictions with Marcel projections."""

    config = ModelFactory.get_config("defense_infield")
    data_path = _resolve_data_path(config.DATA_FILE)
    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)

    player_names = pd.DataFrame(
        raw_df[["Name", "IDfg"]].drop_duplicates()
    ).sort_values("Name")

    position_group_map = {
        "C": "catcher",
        "1B": "infield", "2B": "infield", "3B": "infield", "SS": "infield",
        "LF": "outfield", "CF": "outfield", "RF": "outfield",
    }

    config_map = {
        "infield":  ModelFactory.get_config("defense_infield"),
        "outfield": ModelFactory.get_config("defense_outfield"),
        "catcher":  ModelFactory.get_config("defense_catcher"),
    }
    input_features_map = {
        group: cfg.INPUT_FEATURES for group, cfg in config_map.items()
    }

    # Build position profiles
    batting_for_games = load_batting_for_games()
    all_player_ids = raw_df["IDfg"].unique().tolist()
    profiles = build_position_profiles(
        raw_df, batting_for_games, all_player_ids, cutoff_year=cutoff_year,
    )

    predictions = marcel_fielding_projections(
        raw_df=raw_df,
        player_names=player_names,
        position_group_map=position_group_map,
        input_features_map=input_features_map,
        future_years=Config.History.PROJECTION_HORIZON,
        cutoff_year=cutoff_year,
        position_profiles=profiles,
    )

    if predictions is None:
        logger.error(f"  fielding cutoff={cutoff_year}: marcel_fielding_projections returned None")
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
# BASERUNNING PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_baserunning_predictions(
    cutoff_year: int,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """Generate baserunning predictions with Marcel projections."""

    cfg = ModelFactory.get_config("baserunning")
    data_path = _resolve_data_path(cfg.DATA_FILE)
    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)

    player_names = generate_batter_names(raw_df)

    predictions = marcel_baserunning_projections(
        raw_df=raw_df,
        player_names=player_names,
        input_features=cfg.INPUT_FEATURES,
        future_years=Config.History.PROJECTION_HORIZON,
        cutoff_year=cutoff_year,
    )

    if predictions is None:
        logger.error(f"  baserunning cutoff={cutoff_year}: marcel_baserunning_projections returned None")
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
    start = start or Config.History.CUTOFF_START
    end   = end   or Config.History.CUTOFF_END

    Config.Paths.ensure_directories()

    logger.info("=" * 60)
    logger.info("Historical Values — Generate Predictions")
    logger.info(f"Cutoff years {start} → {end}  (horizon={Config.History.PROJECTION_HORIZON})")
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
