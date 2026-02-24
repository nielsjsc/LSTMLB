#!/usr/bin/env python3
"""
Generate Historical Projections
================================

For each cutoff year Y in [CUTOFF_START … CUTOFF_END]:
  - Calls the existing ``predict_models.py`` with ``--cutoff-year Y``
  - Saves outputs per model type to::

        data/generated/trade_analysis/projections/cutoff_{Y}/
            batter_predictions.csv
            pitcher_predictions.csv
            fielding_predictions.csv
            baserunning_predictions.csv

These are the same LSTM projections the production pipeline generates,
just rolled back to each historical data cutoff.

Already-generated cutoff years are skipped (delete the folder to re-run).

Usage (standalone):
    cd auto_train_models
    python -m trade_analysis.generate_projections          # all cutoff years
    python -m trade_analysis.generate_projections --start 2018 --end 2020

Usage (from pipeline):
    from trade_analysis.generate_projections import generate_all_projections
    generate_all_projections()
"""

import sys
import argparse
import subprocess
from pathlib import Path

from .config import (
    Config,
    logger,
    PROJECTIONS_DIR,
    SCRIPTS_DIR,
    AUTO_TRAIN_DIR,
)

# Model types to generate per cutoff year
MODEL_TYPES = ["pitcher", "batter", "fielding", "baserunning"]


def _output_dir(cutoff_year: int) -> Path:
    """Return the per-cutoff-year projection directory."""
    return PROJECTIONS_DIR / f"cutoff_{cutoff_year}"


def _is_complete(cutoff_year: int) -> bool:
    """Check whether all expected prediction files exist for a cutoff year."""
    out = _output_dir(cutoff_year)
    expected = [f"{mt}_predictions.csv" for mt in MODEL_TYPES]
    return all((out / f).exists() for f in expected)


def generate_projections_for_year(
    cutoff_year: int,
    model_types: list[str] | None = None,
    force: bool = False,
) -> bool:
    """
    Run ``predict_models.py`` for a single cutoff year.

    Args:
        cutoff_year:  Last year of actual data (predictions start cutoff_year + 1).
        model_types:  Subset of MODEL_TYPES to generate (default: all four).
        force:        If True, regenerate even if outputs already exist.

    Returns:
        True if all requested model types succeeded.
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

        # Fielding and baserunning use trade-analysis-owned pretrained models
        # (UZR/DRS/BsR-based) so they work for any historical cutoff year,
        # including pre-2016 seasons where Statcast data is unavailable.
        if mt in ("fielding", "baserunning"):
            cmd = [
                sys.executable,
                "-m", "trade_analysis.predict_pretrain",
                "--model-type", mt,
                "--cutoff-year", str(cutoff_year),
                "--output-dir", str(out_dir),
            ]
        else:
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "predict_models.py"),
                "--model-type", mt,
                "--cutoff-year", str(cutoff_year),
                "--output-dir", str(out_dir),
            ]
            if Config.USE_PRETRAINED and mt in ("batter", "pitcher"):
                cmd.append("--use-pretrained")

        logger.info(f"  {mt} cutoff={cutoff_year}: generating …")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(AUTO_TRAIN_DIR),
            )
            logger.info(f"  {mt} cutoff={cutoff_year}: OK")
        except subprocess.CalledProcessError as e:
            logger.error(f"  {mt} cutoff={cutoff_year}: FAILED")
            logger.error(f"    stderr: {e.stderr[-500:]}")
            all_ok = False

    return all_ok


def generate_all_projections(
    start: int | None = None,
    end: int | None = None,
    force: bool = False,
    model_types: list[str] | None = None,
) -> None:
    """
    Generate projections for every cutoff year in the configured range.

    Args:
        start:       First cutoff year (default: Config.CUTOFF_START).
        end:         Last cutoff year  (default: Config.CUTOFF_END).
        force:       If True, regenerate even if outputs already exist.
        model_types: Subset of model types to generate (default: all four).
    """
    start = start or Config.CUTOFF_START
    end   = end   or Config.CUTOFF_END

    Config.ensure_directories()

    logger.info("=" * 60)
    logger.info("Trade Analysis — Generate Historical Projections")
    logger.info(f"Cutoff years {start} → {end}  (horizon={Config.PROJECTION_HORIZON})")
    logger.info("=" * 60)

    for cutoff_year in range(start, end + 1):
        if _is_complete(cutoff_year) and not force:
            logger.info(f"[cutoff={cutoff_year}] complete — skipping")
            continue

        logger.info(f"[cutoff={cutoff_year}] generating projections …")
        ok = generate_projections_for_year(cutoff_year, model_types=model_types, force=force)
        if not ok:
            logger.warning(f"[cutoff={cutoff_year}] some model types failed")

    logger.info("Projection generation complete.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate historical LSTM projections for trade-value calibration",
    )
    parser.add_argument("--start", type=int, default=Config.CUTOFF_START,
                        help=f"First cutoff year (default: {Config.CUTOFF_START})")
    parser.add_argument("--end",   type=int, default=Config.CUTOFF_END,
                        help=f"Last cutoff year (default: {Config.CUTOFF_END})")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if outputs already exist")
    parser.add_argument("--model-type", dest="model_types", nargs="+",
                        choices=MODEL_TYPES, default=None,
                        help="Only generate these model types (default: all four)")
    args = parser.parse_args()

    generate_all_projections(
        start=args.start, end=args.end, force=args.force, model_types=args.model_types
    )


if __name__ == "__main__":
    main()
