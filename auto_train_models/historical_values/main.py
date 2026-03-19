#!/usr/bin/env python
"""
Historical Values — Pipeline Orchestrator
===========================================

End-to-end CLI that runs all three pipeline stages:

  1. **Predictions**  — generate LSTM projections for each cutoff year
  2. **Surplus**      — compute surplus value per player per snapshot year
  3. **Timeline**     — combine prospect + MLB surplus + Spotrac into
                        the final ``trade_value_history.csv``

Usage:
    cd auto_train_models
    python -m historical_values.main                          # run everything
    python -m historical_values.main --skip-predictions       # skip predictions
    python -m historical_values.main --start 2020 --end 2025  # year range
    python -m historical_values.main --force                  # overwrite existing
    python -m historical_values.main --predictions-only       # only predictions
    python -m historical_values.main --surplus-only            # only surplus
    python -m historical_values.main --timeline-only           # only timeline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_AUTO_TRAIN = Path(__file__).resolve().parents[1]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from historical_values.config import Config, logger


def _run_predictions(
    start: int, end: int, force: bool, model_types: list[str] | None,
) -> None:
    """Phase 1 — generate LSTM predictions for each cutoff year."""
    from historical_values.predictions import generate_all_predictions

    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info("║  Phase 1 / 3 — Generate Predictions              ║")
    logger.info("╚═══════════════════════════════════════════════════╝")

    generate_all_predictions(
        start=start,
        end=end,
        force=force,
        model_types=model_types,
    )


def _run_surplus(start: int, end: int, force: bool) -> None:
    """Phase 2 — compute per-player surplus for each snapshot year."""
    from historical_values.surplus import compute_all_surpluses

    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info("║  Phase 2 / 3 — Compute Surplus Values            ║")
    logger.info("╚═══════════════════════════════════════════════════╝")

    snap_start = start + Config.SNAPSHOT_LAG
    snap_end   = end   + Config.SNAPSHOT_LAG

    compute_all_surpluses(start=snap_start, end=snap_end, force=force)


def _run_timeline() -> None:
    """Phase 3 — build the final trade_value_history.csv."""
    from historical_values.timeline import generate_timeline

    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info("║  Phase 3 / 3 — Generate Timeline                 ║")
    logger.info("╚═══════════════════════════════════════════════════╝")

    generate_timeline()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Historical Trade Value Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Year range
    parser.add_argument(
        "--start", type=int, default=Config.CUTOFF_START,
        help=f"First cutoff year (default: {Config.CUTOFF_START})",
    )
    parser.add_argument(
        "--end", type=int, default=Config.CUTOFF_END,
        help=f"Last cutoff year (default: {Config.CUTOFF_END})",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-predictions", action="store_true",
        help="Skip phase 1 (predictions) — use existing CSVs",
    )
    parser.add_argument(
        "--skip-surplus", action="store_true",
        help="Skip phase 2 (surplus) — use existing CSVs",
    )
    parser.add_argument(
        "--skip-timeline", action="store_true",
        help="Skip phase 3 (timeline generation)",
    )

    # Shortcut flags for running only one phase
    parser.add_argument(
        "--predictions-only", action="store_true",
        help="Run only phase 1 (predictions)",
    )
    parser.add_argument(
        "--surplus-only", action="store_true",
        help="Run only phase 2 (surplus)",
    )
    parser.add_argument(
        "--timeline-only", action="store_true",
        help="Run only phase 3 (timeline)",
    )

    # Model types for predictions
    parser.add_argument(
        "--model-types", nargs="+",
        choices=["batter", "pitcher", "fielding", "baserunning"],
        help="Only generate specific prediction types (default: all)",
    )

    # Force regeneration
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files instead of skipping them",
    )

    args = parser.parse_args(argv)

    # Resolve --*-only flags into skip flags
    if args.predictions_only:
        args.skip_surplus = True
        args.skip_timeline = True
    elif args.surplus_only:
        args.skip_predictions = True
        args.skip_timeline = True
    elif args.timeline_only:
        args.skip_predictions = True
        args.skip_surplus = True

    logger.info("=" * 60)
    logger.info("  Historical Trade Value Pipeline")
    logger.info(f"  Cutoff years: {args.start} → {args.end}")
    logger.info(f"  Force: {args.force}")
    logger.info("=" * 60)

    Config.ensure_directories()
    t0 = time.perf_counter()

    # ── Phase 1: Predictions ─────────────────────────────────────────────
    if not args.skip_predictions:
        _run_predictions(args.start, args.end, args.force, args.model_types)
    else:
        logger.info("Phase 1 (predictions) — SKIPPED")

    # ── Phase 2: Surplus ─────────────────────────────────────────────────
    if not args.skip_surplus:
        _run_surplus(args.start, args.end, args.force)
    else:
        logger.info("Phase 2 (surplus) — SKIPPED")

    # ── Phase 3: Timeline ────────────────────────────────────────────────
    if not args.skip_timeline:
        _run_timeline()
    else:
        logger.info("Phase 3 (timeline) — SKIPPED")

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(f"  Pipeline complete  ({elapsed:.1f}s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
