#!/usr/bin/env python3
"""
MLB Player Prediction Pipeline
Author: Niels Christoffersen
Version: 2.1
Last Updated: January 2026

This script generates predictions for all MLB player types using Marcel projections.
It has been simplified to remove legacy LSTM logic.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from typing import Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_registry import ModelFactory
from core.data_processing import calculate_rate_stats, generate_batter_names
from core.marcel_projections import (
    marcel_pitcher_projections,
    marcel_batter_projections,
    marcel_baserunning_projections,
    marcel_fielding_projections,
    BATTER_BASE_COMPONENTS,
)
from core.park_factors import neutralize_park_factors, get_woba_residual_factor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPTS_DIR = Path(__file__).parent
AUTO_TRAIN_DIR = SCRIPTS_DIR.parent
DATA_DIR = AUTO_TRAIN_DIR.parent / 'data'
GENERATED_DIR = DATA_DIR / 'generated'
PIPELINE_DIR = GENERATED_DIR / 'pipeline'
ROSTER_FILE = DATA_DIR / 'active_roster' / 'current_rosters.csv'

# Ensure directories exist
GENERATED_DIR.mkdir(exist_ok=True)
PIPELINE_DIR.mkdir(exist_ok=True)


def load_roster_ids() -> Optional[set]:
    if not ROSTER_FILE.exists():
        logger.warning(f"Roster file not found at {ROSTER_FILE} — roster recovery disabled")
        return None
    
    roster = pd.read_csv(ROSTER_FILE)
    roster_with_fg = roster.dropna(subset=['fg_id'])
    roster_ids = set(pd.to_numeric(roster_with_fg['fg_id'], errors='coerce').dropna().astype(int))
    logger.info(f"Loaded {len(roster_ids)} roster player IDs from {ROSTER_FILE.name}")
    return roster_ids


def resolve_data_path(config_data_file: str) -> Path:
    config_path = Path(config_data_file)
    parts = config_path.parts
    if 'data' in parts:
        data_idx = parts.index('data')
        relative_path = Path(*parts[data_idx+1:])
        return DATA_DIR / relative_path
    else:
        return DATA_DIR / config_path.name


def get_pitcher_names(raw_df: pd.DataFrame) -> pd.DataFrame:
    pitcher_names_path = DATA_DIR / 'pitcher_names.csv'
    if not pitcher_names_path.exists():
        player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
        player_names.to_csv(pitcher_names_path, index=False)
    else:
        player_names = pd.read_csv(pitcher_names_path)
    return player_names


def generate_pitcher_predictions(
    output_file: str = None, 
    cutoff_year: int = None,
    roster_ids: set = None,
    data_file_path: str | None = None,
) -> Optional[pd.DataFrame]:
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting pitcher predictions generation (cutoff_year={cutoff_year})...")
    
    sp_config_class = ModelFactory.get_config('pitcher_sp')
    resolved_data_file = Path(data_file_path) if data_file_path else resolve_data_path(sp_config_class.DATA_FILE)
    
    raw_df = pd.read_csv(resolved_data_file)
    raw_df = calculate_rate_stats(raw_df)
    player_names = get_pitcher_names(raw_df)
    
    predictions_df = marcel_pitcher_projections(
        raw_df=raw_df,
        player_names=player_names,
        future_years=15,
        cutoff_year=cutoff_year,
        roster_ids=roster_ids,
    )
    
    if predictions_df is not None:
        output_path = output_file or str(PIPELINE_DIR / 'pitcher_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions_df)} Marcel pitcher predictions to {output_path}")
        return predictions_df
    return None


def generate_batter_predictions(
    output_file: str = None, 
    cutoff_year: int = None,
    roster_ids: set = None,
    data_file_path: str | None = None,
) -> Optional[pd.DataFrame]:
    if cutoff_year is None:
        # Batters default to the CURRENT (in-progress) season, not the
        # last completed one. marcel_batter_projections now emits the
        # current season directly (year_offset=0) using year1_base built
        # from in-season-to-date stats + the MiLB-prior blend — this is
        # the single source of truth for the current year's projection,
        # replacing the old ros.py blend-in-a-stale-preseason-number step.
        # This requires the batter DATA_FILE to already carry the current
        # season's stats-to-date as each player's most recent row.
        cutoff_year = datetime.now().year
    
    logger.info(f"Starting batter predictions generation (cutoff_year={cutoff_year}, "
                f"current-season row included via year_offset=0)...")
    batter_config_class = ModelFactory.get_config('batter')
    
    if data_file_path is not None:
        resolved_data_file = Path(data_file_path)
    else:
        resolved_data_file = resolve_data_path(batter_config_class.DATA_FILE)
    
    raw_df = pd.read_csv(resolved_data_file)
    raw_df = calculate_rate_stats(raw_df)

    # ---- Park factor neutralization (pre-Marcel) ----------------------------
    # Neutralize each HISTORICAL season's park-sensitive stats by that
    # season's own team (raw_df's 'Team' column already varies per row/season)
    # before Marcel's weighted-average/regression step ever sees them.
    #
    # IMPORTANT — two different scales, two different feature sets:
    #
    # 1. wOBA/AVG/SLG/xBA/xSLG/xwOBA are NOT neutralized here anymore.
    #    marcel_projections.py substitutes xwOBA/xBA/xSLG directly into
    #    wOBA/AVG/SLG for any season that has them — and Statcast x-stats
    #    are already close to park-neutral by construction (they're modeled
    #    from exit velo/launch angle, not from the actual outcome in this
    #    specific park). Dividing an already-near-neutral x-stat by the
    #    runs-scale PARK_FACTORS_5YR overstates the correction (e.g. it
    #    inflates a Mariner's already-neutral xwOBA by ~6%, when the real
    #    leftover park effect on wOBA is closer to a residual PF of 94, not
    #    the runs-scale 94 used against a not-yet-neutral number). Leaving
    #    these columns alone here means Marcel receives data that's already
    #    correctly (and only once) neutral for these stats, matching the
    #    single reapply that happens downstream in
    #    calculate_war._apply_park_factors_to_batter_predictions /
    #    calculate_wrc_plus / calculate_war_components (which now use
    #    get_woba_residual_factor, not get_park_factor — see calculate_war.py).
    #
    # 2. ISO and HR/FB ARE still neutralized here, using the wOBA-scale
    #    RESIDUAL factor (get_woba_residual_factor), not the runs-scale one.
    #    These two components are the ones a park mechanically affects
    #    (carry distance, altitude, wall height/distance) and aren't fully
    #    covered by the xwOBA/xBA/xSLG substitution above, so they still
    #    need an explicit correction. K%/BB%/HBP%/GB%/LD% are intentionally
    #    left out — a park doesn't change plate discipline or swing plane,
    #    so adjusting those was never justified. BABIP is excluded
    #    automatically by park_factors.get_adjustable_features (a flat
    #    park factor overcorrects BABIP; see that module's EXCLUDED_STATS
    #    comment).
    #
    # Gated behind ENABLE_PARK_FACTOR_ADJUSTMENT (default False — see
    # batter_config.py) since park factors are only reliable for recent/
    # current teams; historical seasons for relocated/defunct franchises
    # fall back to neutral automatically (see park_factors.TEAM_CODE_ALIASES).
    if getattr(batter_config_class, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False):
        park_neutral_features = [
            f for f in ['ISO', 'HR/FB']
            if f in raw_df.columns
        ]
        logger.info(
            f"Park factor neutralization ENABLED — neutralizing "
            f"{len(park_neutral_features)} features (ISO, HR/FB; wOBA-scale "
            f"residual factor) per-season by each row's own Team before "
            f"Marcel. wOBA/AVG/SLG/xBA/xSLG/xwOBA are left as-is (already "
            f"near-neutral via x-stat substitution downstream)."
        )
        raw_df = neutralize_park_factors(
            raw_df, park_neutral_features, team_column='Team',
            factor_fn=get_woba_residual_factor,
        )
    else:
        logger.info("Park factor neutralization DISABLED (ENABLE_PARK_FACTOR_ADJUSTMENT=False)")

    player_names = generate_batter_names(raw_df)
    
    predictions_df = marcel_batter_projections(
        raw_df=raw_df,
        player_names=player_names,
        future_years=15,
        cutoff_year=cutoff_year,
        roster_ids=roster_ids,
        use_xstats=getattr(batter_config_class, 'USE_XWOBA_FOR_PREDICTIONS', True),
    )
    
    if predictions_df is not None:
        output_path = output_file or str(PIPELINE_DIR / 'batter_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions_df)} Marcel batter predictions to {output_path}")
        return predictions_df
    return None


def generate_fielding_predictions(
    output_file: str = None,
    cutoff_year: int = None,
    roster_ids: set = None,
    data_file_path: str = None
) -> Optional[pd.DataFrame]:
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting fielding predictions generation (cutoff_year={cutoff_year})...")
    fielding_config_class = ModelFactory.get_config('defense_infield')
    
    if data_file_path is None:
        data_file_path = resolve_data_path(fielding_config_class.DATA_FILE)
    else:
        logger.info(f"Using provided fielding data file: {data_file_path}")
    
    raw_df = pd.read_csv(data_file_path)
    raw_df = calculate_rate_stats(raw_df)
    player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
    
    config_map = {
        'infield': 'defense_infield',
        'outfield': 'defense_outfield',
        'catcher': 'defense_catcher',
    }
    
    position_group_map = {
        'C': 'catcher',
        '1B': 'infield', '2B': 'infield', '3B': 'infield', 'SS': 'infield',
        'LF': 'outfield', 'CF': 'outfield', 'RF': 'outfield'
    }
    
    input_features_map = {}
    for group_name, config_key in config_map.items():
        cfg = ModelFactory.get_config(config_key)
        input_features_map[group_name] = cfg.INPUT_FEATURES
        
    from core.position_profiles import build_position_profiles, load_batting_for_games
    batting_for_games = load_batting_for_games()
    all_player_ids = raw_df['IDfg'].unique().tolist()
    if roster_ids:
        all_player_ids = list(set(all_player_ids) | roster_ids)
    profiles = build_position_profiles(raw_df, batting_for_games, all_player_ids, cutoff_year=cutoff_year)
    
    predictions_df = marcel_fielding_projections(
        raw_df=raw_df,
        player_names=player_names,
        position_group_map=position_group_map,
        input_features_map=input_features_map,
        future_years=15,
        cutoff_year=cutoff_year,
        roster_ids=roster_ids,
        position_profiles=profiles,
    )
    
    if predictions_df is not None:
        if 'Position_Group' in predictions_df.columns:
            predictions_df = predictions_df.drop(columns=['Position_Group'])
        
        metadata_cols = ['Name', 'Age', 'Year', 'IDfg', 'Pos']
        feature_cols = [col for col in predictions_df.columns if col not in metadata_cols]
        predictions_df = predictions_df[metadata_cols + feature_cols]
        
        output_path = output_file or str(PIPELINE_DIR / 'fielding_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions_df)} fielding predictions to {output_path}")
        return predictions_df
    return None


def generate_baserunning_predictions(
    output_file: str = None,
    cutoff_year: int = None,
    roster_ids: set = None
) -> Optional[pd.DataFrame]:
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting baserunning predictions generation (cutoff_year={cutoff_year})...")
    baserunning_config_class = ModelFactory.get_config('baserunning')
    data_file_path = resolve_data_path(baserunning_config_class.DATA_FILE)
    
    raw_df = pd.read_csv(data_file_path)
    raw_df = calculate_rate_stats(raw_df)
    player_names = generate_batter_names(raw_df)
    
    predictions_df = marcel_baserunning_projections(
        raw_df=raw_df,
        player_names=player_names,
        input_features=baserunning_config_class.INPUT_FEATURES,
        future_years=15,
        cutoff_year=cutoff_year,
        roster_ids=roster_ids,
    )
    
    if predictions_df is not None:
        output_path = output_file or str(PIPELINE_DIR / 'baserunning_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions_df)} baserunning predictions to {output_path}")
        return predictions_df
    return None


def generate_integrated_batter_predictions(
    output_file: str = None,
    cutoff_year: int = None,
    roster_ids: set = None
) -> Optional[pd.DataFrame]:
    from value_determination.calculate_war import calculate_war_components, calculate_baserunning_value, calculate_defensive_value, load_player_orgs
    
    if cutoff_year is None:
        cutoff_year = datetime.now().year - 1
    
    logger.info(f"Starting integrated batter predictions with position-specific fielding (cutoff_year={cutoff_year})...")
    
    batter_df = generate_batter_predictions(cutoff_year=cutoff_year, roster_ids=roster_ids)
    if batter_df is None: return None
    
    fielding_file = PIPELINE_DIR / 'fielding_predictions.csv'
    if fielding_file.exists():
        fielding_df = pd.read_csv(fielding_file)
    else:
        fielding_df = generate_fielding_predictions(cutoff_year=cutoff_year, roster_ids=roster_ids)
        if fielding_df is None: return None
        
    baserunning_file = PIPELINE_DIR / 'baserunning_predictions.csv'
    if baserunning_file.exists():
        baserunning_df = pd.read_csv(baserunning_file)
    else:
        baserunning_df = generate_baserunning_predictions(cutoff_year=cutoff_year, roster_ids=roster_ids)
        if baserunning_df is None: return None
        
    org_data = load_player_orgs(GENERATED_DIR)
    batter_df = batter_df.merge(org_data, on='IDfg', how='left')
    
    war_components_list = []
    for idx, row in batter_df.iterrows():
        try:
            war, components = calculate_war_components(row, baserunning_df, fielding_df)
            components['IDfg'] = row['IDfg']
            components['Year'] = row['Year']
            war_components_list.append(components)
        except Exception as e:
            continue
            
    war_df = pd.DataFrame(war_components_list)
    integrated_df = batter_df.merge(war_df, on=['IDfg', 'Year'], how='left', suffixes=('_old', ''))
    
    columns_to_remove = [col for col in integrated_df.columns if col.endswith('_old')]
    integrated_df = integrated_df.drop(columns=columns_to_remove, errors='ignore')
    
    integrated_df = integrated_df[integrated_df['Year'] != cutoff_year].copy()
    integrated_df = integrated_df.sort_values(['Year', 'WAR'], ascending=[True, False])
    
    output_path = output_file or str(PIPELINE_DIR / 'integrated_batter_predictions.csv')
    integrated_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(integrated_df)} integrated batter predictions to {output_path}")
    
    return integrated_df


def main():
    default_cutoff_year = datetime.now().year - 1
    
    parser = argparse.ArgumentParser(description='Generate MLB player predictions using Marcel models')
    parser.add_argument('--model-type', 
                       choices=['pitcher', 'batter', 'fielding', 'baserunning', 'integrated-batter', 'all'],
                       default='all')
    parser.add_argument('--output-dir', type=str, default=str(PIPELINE_DIR))
    parser.add_argument('--cutoff-year', type=int, default=default_cutoff_year,
                       help='Last completed season, used by pitcher/fielding/baserunning.')
    parser.add_argument('--batter-cutoff-year', type=int, default=None,
                       help='Season to anchor batter Marcel projections on. Defaults to the '
                            'CURRENT (in-progress) season — batters now emit their current-year '
                            'row directly from year1_base (year_offset=0), so this should stay '
                            'the in-progress season, not the last completed one, as long as the '
                            'batter data file carries current-season stats-to-date. Override only '
                            'for backfills/testing against a specific historical cutoff.')
    parser.add_argument('--batter-data-file', type=str, default=None)
    parser.add_argument('--pitcher-data-file', type=str, default=None)
    parser.add_argument('--fielding-data-file', type=str, default=None)
    parser.add_argument('--use-ros-blending', action='store_true', help='Use ROS blending (handled inherently by data files)')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    roster_ids = load_roster_ids()
    
    if args.model_type in ['pitcher', 'all']:
        generate_pitcher_predictions(
            output_file=str(output_dir / 'pitcher_predictions.csv'),
            cutoff_year=args.cutoff_year,
            roster_ids=roster_ids,
            data_file_path=args.pitcher_data_file
        )
        
    if args.model_type in ['batter', 'all']:
        generate_batter_predictions(
            output_file=str(output_dir / 'batter_predictions.csv'),
            cutoff_year=args.batter_cutoff_year,  # None → defaults to CURRENT_YEAR inside
            roster_ids=roster_ids,
            data_file_path=args.batter_data_file
        )
        
    if args.model_type in ['fielding', 'all']:
        generate_fielding_predictions(
            output_file=str(output_dir / 'fielding_predictions.csv'),
            cutoff_year=args.cutoff_year,
            roster_ids=roster_ids,
            data_file_path=args.fielding_data_file
        )
        
    if args.model_type in ['baserunning', 'all']:
        generate_baserunning_predictions(
            output_file=str(output_dir / 'baserunning_predictions.csv'),
            cutoff_year=args.cutoff_year,
            roster_ids=roster_ids
        )
        
    if args.model_type in ['integrated-batter', 'all']:
        generate_integrated_batter_predictions(
            output_file=str(output_dir / 'integrated_batter_predictions.csv'),
            cutoff_year=args.cutoff_year,
            roster_ids=roster_ids
        )

if __name__ == "__main__":
    main()