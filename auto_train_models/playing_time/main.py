#!/usr/bin/env python3
"""
Projection Engine - Playing Time & WAR Calculation
===================================================

Orchestrates the full projection post-processing pipeline:
1. Load all data sources (rate stat predictions)
2. Process injury data
3. Build team rosters with depth charts
4. Allocate playing time by projected value (wOBA/FIP)
5. Calculate WAR based on allocated playing time
6. Export final projections

This is the final step that converts rate stats into counting stats and WAR,
based on realistic playing time allocations.

Usage:
    python -m playing_time.main [--year YEAR] [--output-dir DIR]

Author: Niels Christoffersen
Date: January 2026
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

from .config import Config
from .data_loader import DataLoader
from .injury_processor import InjuryProcessor
from .roster_builder import RosterBuilder
from .allocator import PlayingTimeAllocator, TeamAllocation
from .value_calculator import ValueCalculator, get_calculator
from .confidence import ConfidenceCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(projection_year: int = None,
                 output_dir: Path = None,
                 config: Config = None) -> pd.DataFrame:
    """
    Run the full playing time projection pipeline.
    
    Args:
        projection_year: Year to project (default: current year from config)
        output_dir: Output directory (default: from config)
        config: Configuration object
        
    Returns:
        DataFrame with playing time allocations
    """
    config = config or Config()
    projection_year = projection_year or config.CURRENT_YEAR
    output_dir = output_dir or config.OUTPUT_DIR
    
    logger.info(f"Starting playing time projection for {projection_year}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Step 1: Load Data
    # ==========================================================================
    logger.info("Step 1: Loading data...")
    loader = DataLoader(config)
    
    batters, pitchers, fielding, roster, _, injuries = loader.load_all()  # Ignore prospects
    
    # ==========================================================================
    # Step 2: Process Injuries
    # ==========================================================================
    logger.info("Step 2: Processing injury data...")
    injury_processor = InjuryProcessor(injuries, config)
    injury_processor.process()
    
    # Get all player IDs from predictions
    all_batter_ids = batters['IDfg'].unique().tolist()
    all_pitcher_ids = pitchers['IDfg'].unique().tolist()
    all_ids = list(set(all_batter_ids + all_pitcher_ids))
    
    # Build injury adjustment lookup
    injury_adjustments = injury_processor.build_adjustment_lookup(
        all_ids, projection_year
    )
    
    logger.info(f"Computed injury adjustments for {len(injury_adjustments)} players")
    
    # Log players with significant adjustments
    significant = {k: v for k, v in injury_adjustments.items() if v.multiplier < 0.9}
    if significant:
        logger.info(f"Players with injury adjustments: {len(significant)}")
        for fg_id, info in list(significant.items())[:10]:
            logger.info(f"  IDfg {fg_id}: {info.multiplier:.2f}x - {info.injury_name or 'Unknown'}")
    
    # ==========================================================================
    # Step 3: Build Team Rosters with Confidence Scores
    # ==========================================================================
    logger.info("Step 3: Building team rosters with confidence scores...")
    
    # Initialize confidence calculator with historical data
    # Use correct file paths for historical MLB data
    confidence_calculator = ConfidenceCalculator(
        historical_batting_path=config.DATA_DIR / 'historic_mlb' / 'mlb_batting_data_1950_2025.csv',
        historical_pitching_path=config.DATA_DIR / 'historic_mlb' / 'mlb_pitching_data_1950_2025.csv',
        prospects_path=config.DATA_DIR / 'prospect_data' / 'prospects_2014_2025_complete.csv'
    )
    
    builder = RosterBuilder(config, confidence_calculator=confidence_calculator)
    
    team_rosters = builder.build_team_rosters(
        roster_df=roster,
        batter_preds=batters,
        pitcher_preds=pitchers,
        fielding_df=fielding,
        injury_adjustments=injury_adjustments,
        projection_year=projection_year
    )
    
    logger.info(f"Built rosters for {len(team_rosters)} teams")
    
    # Log confidence tier distribution
    all_players = []
    for team_roster in team_rosters.values():
        for group in team_roster.values():
            all_players.extend(group)
    
    tier_counts = {}
    for player in all_players:
        tier = player.confidence_tier
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    logger.info(f"Confidence tier distribution: {tier_counts}")
    
    # ==========================================================================
    # Step 4: Allocate Playing Time
    # ==========================================================================
    logger.info("Step 4: Allocating playing time...")
    allocator = PlayingTimeAllocator(config)
    
    allocations = allocator.allocate_all_teams(team_rosters, projection_year)
    
    logger.info(f"Allocated playing time for {sum(len(a.players) for a in allocations.values())} players")
    
    # ==========================================================================
    # Step 5: Calculate WAR
    # ==========================================================================
    logger.info("Step 5: Calculating WAR based on allocated playing time...")
    
    # Load baserunning data for WAR calculation
    baserunning_df = loader.load_baserunning_predictions()
    
    allocations = calculate_war_for_allocations(
        allocations=allocations,
        baserunning_df=baserunning_df,
        projection_year=projection_year
    )
    
    # Convert to DataFrame
    results_df = allocator.to_dataframe(allocations)
    summary_df = allocator.summarize(allocations)
    
    # ==========================================================================
    # Step 6: Export Results
    # ==========================================================================
    logger.info("Step 6: Exporting results...")
    
    # Main allocations with WAR
    output_file = output_dir / f'projections_{projection_year}.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved projections to {output_file}")
    
    # Team summary
    summary_file = output_dir / f'team_summary_{projection_year}.csv'
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved team summary to {summary_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"MLB PROJECTIONS - {projection_year}")
    print("=" * 70)
    print(f"\nTotal players: {len(results_df)}")
    print(f"Teams: {len(allocations)}")
    
    # Top batters by WAR
    batters = results_df[results_df['Allocated_Games'].notna()].copy()
    if not batters.empty and 'WAR' in batters.columns:
        print(f"\nTop 10 Position Players by Projected WAR:")
        top_batters = batters.nlargest(10, 'WAR')[['Name', 'Team', 'Position', 'Allocated_Games', 'WAR']]
        print(top_batters.to_string(index=False))
    
    # Top pitchers by WAR
    pitchers = results_df[results_df['Allocated_IP'].notna()].copy()
    if not pitchers.empty and 'WAR' in pitchers.columns:
        print(f"\nTop 10 Pitchers by Projected WAR:")
        top_pitchers = pitchers.nlargest(10, 'WAR')[['Name', 'Team', 'Role', 'Allocated_IP', 'WAR']]
        print(top_pitchers.to_string(index=False))
    
    print("\n" + "=" * 70)
    
    return results_df


def calculate_war_for_allocations(
    allocations: Dict[str, TeamAllocation],
    baserunning_df: pd.DataFrame,
    projection_year: int
) -> Dict[str, TeamAllocation]:
    """
    Calculate WAR for all allocated players based on their playing time.
    
    This is the key step that converts rate stats to actual WAR using
    the allocated games/IP.
    """
    calculator = get_calculator()
    
    for team, team_alloc in allocations.items():
        for result in team_alloc.players:
            pred_data = result.prediction_data or {}
            
            if result.allocated_games > 0:
                # Position player - calculate batter WAR
                woba = pred_data.get('wOBA', 0.300)
                
                # Get baserunning data
                bsr_row = baserunning_df[
                    (baserunning_df['IDfg'] == result.id) & 
                    (baserunning_df['Year'] == projection_year)
                ]
                baserunning_data = bsr_row.iloc[0].to_dict() if not bsr_row.empty else None
                
                # Get fielding data from prediction
                fielding_data = pred_data.get('fielding', {})
                
                # Rate stats for counting stats
                rate_stats = {
                    'HR_rate': pred_data.get('HR_rate', 0),
                    '2B_rate': pred_data.get('2B_rate', 0),
                    'RBI_rate': pred_data.get('RBI_rate', 0),
                    'R_rate': pred_data.get('R_rate', 0),
                }
                
                war_result = calculator.calculate_batter_war(
                    woba=woba,
                    games=result.allocated_games,
                    team=team,
                    position=result.position,
                    baserunning_data=baserunning_data,
                    fielding_data=fielding_data,
                    rate_stats=rate_stats
                )
                
                result.war = round(war_result.war, 1)
                result.wrc_plus = round(war_result.wrc_plus, 0)
                result.batting_runs = round(war_result.batting_runs, 1)
                result.baserunning_runs = round(war_result.baserunning_runs, 1)
                result.fielding_runs = round(war_result.fielding_runs, 1)
                result.positional_adj = round(war_result.positional_adj, 1)
                
            elif result.allocated_ip > 0:
                # Pitcher - calculate pitcher WAR
                # projected_value is -FIP, so negate to get actual FIP
                fip = -result.projected_value if result.projected_value else pred_data.get('FIP', 4.50)
                
                rate_stats = {
                    'K%': pred_data.get('K%', 0.20),
                    'BB%': pred_data.get('BB%', 0.08),
                    'SV%': pred_data.get('SV%', 0),
                    'ERA': pred_data.get('ERA', 4.50),
                }
                
                war_result = calculator.calculate_pitcher_war(
                    fip=fip,
                    ip=result.allocated_ip,
                    team=team,
                    role=result.role,
                    rate_stats=rate_stats
                )
                
                result.war = round(war_result.war, 1)
                result.fip_runs = round(war_result.fip_runs, 1)
    
    return allocations


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Project MLB playing time allocations'
    )
    parser.add_argument(
        '--year', 
        type=int, 
        default=None,
        help='Projection year (default: 2026)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        run_pipeline(
            projection_year=args.year,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
