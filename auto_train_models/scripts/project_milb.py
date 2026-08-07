import sys
import argparse
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'auto_train_models'))

from core.milb_projections.batter_regression import save_milb_priors

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def _print_top(projections, projection_year: int, min_pa: int = 150, n: int = 50):
    top = projections[projections['PA'] >= min_pa].head(n)

    print(f"===========================================================")
    print(f" TOP {n} MiLB HITTER PROJECTIONS ({projection_year})")
    print(f" (Min {min_pa} PA over past 5 seasons)")
    print(f"===========================================================")
    print(f"{'Rank':<5} | {'Name':<22} | {'wRC+':<5} | {'wOBA':<5} | {'AVG':<5} | {'OBP':<5} | {'SLG':<5} | {'OPS':<5} | {'HR':<3}")
    print("-" * 80)

    for idx, (_, row) in enumerate(top.iterrows(), 1):
        name = str(row['Name'])[:21]
        wrc = f"{row['Target_wRC+']:.0f}"
        woba = f"{row['Proj_wOBA']:.3f}".lstrip('0')
        avg = f"{row['Proj_AVG']:.3f}".lstrip('0')
        obp = f"{row['Proj_OBP']:.3f}".lstrip('0')
        slg = f"{row['Proj_SLG']:.3f}".lstrip('0')
        ops = f"{row['Proj_OPS']:.3f}".lstrip('0')
        hr = f"{row['Proj_HR']:.0f}"

        print(f"{idx:<5} | {name:<22} | {wrc:<5} | {woba:<5} | {avg:<5} | {obp:<5} | {slg:<5} | {ops:<5} | {hr:<3}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MiLB batter priors and cache them for the daily pipeline."
    )
    parser.add_argument(
        '--projection-year', type=int, default=2026,
        help="Year being projected. Only affects the --rookies-only exclusion filter "
             "(MiLB priors themselves are anchored per-player, not to this year)."
    )
    parser.add_argument(
        '--rookies-only', action='store_true',
        help="Restrict to rookie-eligible prospects with no meaningful MLB experience. "
             "Off by default — marcel_projections.py expects priors for EVERY qualifying "
             "player (including those with MLB track records) so it can blend them in."
    )
    args = parser.parse_args()

    # This writes to data/generated/pipeline/MiLB_priors/milb_batter_priors.csv —
    # the single location marcel_projections.py reads cached MiLB priors from.
    # It intentionally does NOT write to /outputs; nothing in the pipeline reads
    # from there, and that directory previously held a copy nothing consumed.
    projections = save_milb_priors(
        args.projection_year,
        exclude_mlb_experienced=args.rookies_only,
    )

    if projections.empty:
        logger.warning("No projections generated.")
        sys.exit(1)

    projections = projections.sort_values('Target_wRC+', ascending=False)
    _print_top(projections, args.projection_year)


if __name__ == '__main__':
    main()