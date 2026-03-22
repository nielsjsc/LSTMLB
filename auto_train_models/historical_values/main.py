#!/usr/bin/env python
"""
Historical Values — Pipeline Orchestrator
===========================================

End-to-end CLI that runs all three pipeline stages:

  1. Predictions  — generate LSTM / Marcel projections for each cutoff year
  2. Surplus      — compute surplus value per player per snapshot year
  3. Timeline     — combine prospect + MLB surplus + Spotrac into
                    the final trade_value_history.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quick reference  (run from auto_train_models/):

  # Full pipeline (all years, all phases)
  python -m historical_values.main

  # Rerun only fielding predictions (all years)
  python -m historical_values.main predictions --models fielding --force

  # Rerun fielding + baserunning predictions, 2018-2025
  python -m historical_values.main predictions --models fielding baserunning --start 2018 --force

  # Rerun only surplus (all years)
  python -m historical_values.main surplus --force

  # Rerun only timeline
  python -m historical_values.main timeline

  # Predictions then surplus, skip timeline
  python -m historical_values.main predictions surplus --start 2020

  # Everything for a single year
  python -m historical_values.main --start 2022 --end 2022 --force
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

ALL_PHASES = ["predictions", "surplus", "timeline"]
ALL_MODELS = ["batter", "pitcher", "fielding", "baserunning"]


def _run_predictions(
    start: int, end: int, force: bool, model_types: list[str] | None,
) -> None:
    """Phase 1 — generate LSTM predictions for each cutoff year."""
    from historical_values.predictions import generate_all_predictions

    label = ", ".join(model_types) if model_types else "all"
    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info(f"║  Phase 1 — Predictions  ({label})")
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
    logger.info("║  Phase 2 — Surplus Values                        ║")
    logger.info("╚═══════════════════════════════════════════════════╝")

    snap_start = start + Config.SNAPSHOT_LAG
    snap_end   = end   + Config.SNAPSHOT_LAG

    compute_all_surpluses(start=snap_start, end=snap_end, force=force)


def _run_timeline() -> None:
    """Phase 3 — build the final trade_value_history.csv."""
    from historical_values.timeline import generate_timeline

    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info("║  Phase 3 — Timeline                              ║")
    logger.info("╚═══════════════════════════════════════════════════╝")

    generate_timeline()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m historical_values.main",
        description="Historical Trade Value Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Positional: which phases to run ---
    parser.add_argument(
        "phases", nargs="*", default=[], metavar="PHASE",
        help=(
            "Which phases to run: predictions, surplus, timeline. "
            "Omit to run all three. Combine to run specific phases in order."
        ),
    )

    # --- Year range ---
    parser.add_argument(
        "--start", type=int, default=Config.CUTOFF_START,
        help=f"First cutoff year (default: {Config.CUTOFF_START})",
    )
    parser.add_argument(
        "--end", type=int, default=Config.CUTOFF_END,
        help=f"Last cutoff year (default: {Config.CUTOFF_END})",
    )

    # --- Model types for predictions ---
    parser.add_argument(
        "--models", nargs="+", dest="model_types",
        choices=ALL_MODELS, metavar="MODEL",
        help="Prediction types to generate: batter, pitcher, fielding, baserunning (default: all)",
    )

    # --- Force regeneration ---
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Overwrite existing files instead of skipping them",
    )

    # --- Legacy flags (still supported for backward compat) ---
    parser.add_argument("--predictions-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--surplus-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeline-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-predictions", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-surplus", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-timeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-types", nargs="+", dest="model_types_legacy",
                        choices=ALL_MODELS, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    # ── Resolve which phases to run ──────────────────────────────────────
    phases = list(args.phases) if args.phases else []

    # Validate phase names
    for p in phases:
        if p not in ALL_PHASES:
            parser.error(f"Unknown phase '{p}'. Choose from: {', '.join(ALL_PHASES)}")

    # Legacy --*-only flags
    if args.predictions_only:
        phases = ["predictions"]
    elif args.surplus_only:
        phases = ["surplus"]
    elif args.timeline_only:
        phases = ["timeline"]

    # If no phases specified, run all (unless legacy --skip flags)
    if not phases:
        phases = list(ALL_PHASES)
    if args.skip_predictions and "predictions" in phases:
        phases.remove("predictions")
    if args.skip_surplus and "surplus" in phases:
        phases.remove("surplus")
    if args.skip_timeline and "timeline" in phases:
        phases.remove("timeline")

    # Merge legacy --model-types with --models
    model_types = args.model_types or args.model_types_legacy

    # ── Print plan ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Historical Trade Value Pipeline")
    logger.info(f"  Phases:      {', '.join(phases)}")
    logger.info(f"  Cutoff years: {args.start} → {args.end}")
    if model_types:
        logger.info(f"  Model types:  {', '.join(model_types)}")
    logger.info(f"  Force:        {args.force}")
    logger.info("=" * 60)

    Config.ensure_directories()
    t0 = time.perf_counter()

    # ── Phase 1: Predictions ─────────────────────────────────────────────
    if "predictions" in phases:
        _run_predictions(args.start, args.end, args.force, model_types)
    else:
        logger.info("Phase 1 (predictions) — SKIPPED")

    # ── Phase 2: Surplus ─────────────────────────────────────────────────
    if "surplus" in phases:
        _run_surplus(args.start, args.end, args.force)
    else:
        logger.info("Phase 2 (surplus) — SKIPPED")

    # ── Phase 3: Timeline ────────────────────────────────────────────────
    if "timeline" in phases:
        _run_timeline()
    else:
        logger.info("Phase 3 (timeline) — SKIPPED")

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(f"  Done  ({elapsed:.1f}s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
