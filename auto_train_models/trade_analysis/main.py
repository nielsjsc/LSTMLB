#!/usr/bin/env python
"""
Trade Analysis Pipeline — Main Orchestrator
=============================================

Runs the full pipeline end-to-end:

    Phase 1 — generate_projections
        For each cutoff year (2013–2024), shell out to predict_models.py to
        produce batter/pitcher/fielding/baserunning prediction CSVs.

    Phase 2 — surplus_calculator
        For each snapshot year (2014–2025), compute WAR from the raw
        predictions, estimate salary from service-time, and produce
        per-player surplus values.

    Phase 3 — analyze_trades
        Link actual trades (2014–2024) to surplus, fit the non-linear β
        parameter, and export results.

Usage:
    # Full pipeline
    python -m trade_analysis.main

    # Individual phases
    python -m trade_analysis.main --phase projections
    python -m trade_analysis.main --phase surplus
    python -m trade_analysis.main --phase trades

    # Custom year range
    python -m trade_analysis.main --start 2018 --end 2020

    # Force re-run
    python -m trade_analysis.main --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_AUTO_TRAIN = Path(__file__).resolve().parents[1]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from trade_analysis.config import Config, logger


PHASES = ("projections", "surplus", "trades")


def run_pipeline(
    phases: tuple[str, ...] = PHASES,
    start: int | None = None,
    end: int | None = None,
    force: bool = False,
    beta: float | None = None,
):
    """
    Execute requested pipeline phases.

    Args:
        phases: Which phases to run (default: all three).
        start:  Override earliest snapshot year.
        end:    Override latest snapshot year.
        force:  Re-run even if outputs already exist.
        beta:   Fixed β for trade analysis (None = optimise).
    """
    Config.ensure_directories()

    cutoff_start = start - Config.SNAPSHOT_LAG if start else Config.CUTOFF_START
    cutoff_end   = end   - Config.SNAPSHOT_LAG if end   else Config.CUTOFF_END
    snap_start   = cutoff_start + Config.SNAPSHOT_LAG
    snap_end     = cutoff_end   + Config.SNAPSHOT_LAG

    t0 = time.time()
    logger.info("=" * 65)
    logger.info("  Trade Analysis Pipeline")
    logger.info(f"  Phases   : {', '.join(phases)}")
    logger.info(f"  Cutoffs  : {cutoff_start}–{cutoff_end}")
    logger.info(f"  Snapshots: {snap_start}–{snap_end}")
    logger.info(f"  Force    : {force}")
    logger.info("=" * 65)

    # ── Phase 1: projections ──────────────────────────────────────────────
    if "projections" in phases:
        logger.info("\n▸ Phase 1 — Generating projections")
        from trade_analysis.generate_projections import generate_all_projections
        generate_all_projections(
            start=cutoff_start,
            end=cutoff_end,
            force=force,
        )

    # ── Phase 2: surplus ──────────────────────────────────────────────────
    if "surplus" in phases:
        logger.info("\n▸ Phase 2 — Computing surplus values")
        from trade_analysis.surplus_calculator import compute_all_surpluses
        compute_all_surpluses(
            start=snap_start,
            end=snap_end,
            force=force,
        )

    # ── Phase 3: trade analysis ───────────────────────────────────────────
    if "trades" in phases:
        logger.info("\n▸ Phase 3 — Analyzing trades")
        from trade_analysis.analyze_trades import run_analysis
        run_analysis(beta=beta)

    elapsed = time.time() - t0
    logger.info(f"\nPipeline finished in {elapsed / 60:.1f} minutes.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run the historical trade-analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--phase", choices=PHASES, action="append", default=None,
        help="Run only this phase (can be repeated). Default: all.",
    )
    p.add_argument("--start", type=int, help="First snapshot year.")
    p.add_argument("--end", type=int, help="Last snapshot year.")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--beta", type=float, default=None,
                   help="Fixed β for trade analysis (skip optimisation).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    phases = tuple(args.phase) if args.phase else PHASES
    run_pipeline(
        phases=phases,
        start=args.start,
        end=args.end,
        force=args.force,
        beta=args.beta,
    )
