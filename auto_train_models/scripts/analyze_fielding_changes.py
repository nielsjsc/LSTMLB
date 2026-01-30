#!/usr/bin/env python3
"""
Analyze Year-Over-Year Fielding Changes
========================================

Compare actual 2025 fielding data with 2026 predictions to identify
the biggest projected changes in defensive performance.

Usage:
    python analyze_fielding_changes.py [--position-group POSITION]
    
Examples:
    python analyze_fielding_changes.py --position-group outfield
    python analyze_fielding_changes.py --position-group infield
    python analyze_fielding_changes.py --position-group catcher
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Setup paths
SCRIPTS_DIR = Path(__file__).parent
AUTO_TRAIN_DIR = SCRIPTS_DIR.parent
DATA_DIR = AUTO_TRAIN_DIR.parent / 'data'
GENERATED_DIR = DATA_DIR / 'generated'
PIPELINE_DIR = GENERATED_DIR / 'pipeline'
HISTORIC_MLB_DIR = DATA_DIR / 'historic_mlb'

# File paths
PREDICTIONS_FILE = PIPELINE_DIR / 'fielding_predictions.csv'
HISTORICAL_FILE = HISTORIC_MLB_DIR / 'mlb_fielding_data_2000_2025_with_statcast.csv'


def load_data():
    """Load predictions and historical data"""
    print("Loading data...")
    
    # Load predictions
    predictions_df = pd.read_csv(PREDICTIONS_FILE)
    print(f"  Loaded {len(predictions_df)} prediction records")
    
    # Load historical data
    historical_df = pd.read_csv(HISTORICAL_FILE)
    print(f"  Loaded {len(historical_df)} historical records")
    
    return predictions_df, historical_df


def filter_by_position_group(df, position_group):
    """Filter dataframe by position group"""
    position_map = {
        'outfield': ['LF', 'CF', 'RF'],
        'infield': ['1B', '2B', '3B', 'SS'],
        'catcher': ['C']
    }
    
    if position_group not in position_map:
        raise ValueError(f"Invalid position group. Choose from: {list(position_map.keys())}")
    
    valid_positions = position_map[position_group]
    return df[df['Pos'].isin(valid_positions)].copy()


def calculate_changes(historical_df, predictions_df, position_group='outfield', min_innings=400):
    """
    Calculate year-over-year changes in fielding metrics.
    
    Args:
        historical_df: Historical fielding data (raw sc_total_runs)
        predictions_df: Predicted fielding data (sc_total_runs already scaled to /150)
        position_group: 'outfield', 'infield', or 'catcher'
        min_innings: Minimum innings to qualify
    """
    print(f"\nAnalyzing {position_group} fielding changes (2025 → 2026)...")
    print(f"Minimum innings threshold: {min_innings}")
    
    # Filter by position group
    hist_filtered = filter_by_position_group(historical_df, position_group)
    pred_filtered = filter_by_position_group(predictions_df, position_group)
    
    # Get 2025 actual data
    actual_2025 = hist_filtered[hist_filtered['Season'] == 2025].copy()
    actual_2025 = actual_2025[actual_2025['Inn'] >= min_innings]
    
    # Scale 2025 sc_total_runs to per 150 games (1350 innings)
    # Formula: (sc_total_runs / Inn) * 1350
    if 'sc_total_runs' in actual_2025.columns:
        actual_2025['sc_total_runs_scaled'] = (actual_2025['sc_total_runs'] / actual_2025['Inn']) * 1350
    
    print(f"  Found {len(actual_2025)} qualified {position_group} players in 2025")
    
    # Get 2026 predictions (already scaled to /150)
    pred_2026 = pred_filtered[pred_filtered['Year'] == 2026].copy()
    
    # Rename prediction column to match - handle both possible column names
    if 'sc_total_runs/150' in pred_2026.columns:
        pred_2026['sc_total_runs_scaled'] = pred_2026['sc_total_runs/150']
    elif 'sc_total_runs' in pred_2026.columns:
        # Already in the right format
        pred_2026['sc_total_runs_scaled'] = pred_2026['sc_total_runs']
    
    print(f"  Found {len(pred_2026)} {position_group} players predicted for 2026")
    
    # Merge on player ID
    comparison = actual_2025.merge(
        pred_2026,
        left_on='IDfg',
        right_on='IDfg',
        suffixes=('_2025', '_2026')
    )
    
    if len(comparison) == 0:
        print("  No matching players found!")
        return None
    
    print(f"  Comparing {len(comparison)} players with data in both years")
    
    # Metric name
    metrics = {
        'sc_total_runs': 'Total Defensive Runs (per 150 games)',
    }
    
    # Calculate changes
    changes = pd.DataFrame()
    changes['Name'] = comparison['Name_2025']
    changes['Pos'] = comparison['Pos_2025']
    changes['Age_2025'] = comparison['Age_2025']
    changes['Age_2026'] = comparison['Age_2026']
    changes['Inn_2025'] = comparison['Inn_2025']
    
    # Use scaled values for comparison (after merge, they have suffixes)
    changes['sc_total_runs_2025'] = comparison['sc_total_runs_scaled_2025']
    changes['sc_total_runs_2026'] = comparison['sc_total_runs_scaled_2026']
    changes['sc_total_runs_change'] = comparison['sc_total_runs_scaled_2026'] - comparison['sc_total_runs_scaled_2025']
    
    return changes, metrics


def display_top_changes(changes, metrics, n=20):
    """Display top improvers and decliners for each metric"""
    
    for metric, description in metrics.items():
        change_col = f'{metric}_change'
        
        if change_col not in changes.columns:
            print(f"\n⚠️  {description} data not available")
            continue
        
        # Filter out players with NaN changes
        valid_changes = changes[changes[change_col].notna()].copy()
        
        if len(valid_changes) == 0:
            print(f"\n⚠️  No valid data for {description}")
            continue
        
        print(f"\n{'='*80}")
        print(f"{description.upper()} ({metric})")
        print(f"{'='*80}")
        
        # Top improvers
        print(f"\n🔥 TOP {min(n, len(valid_changes))} IMPROVERS:")
        print("-" * 80)
        top_improvers = valid_changes.nlargest(n, change_col)
        
        for idx, row in top_improvers.iterrows():
            print(f"{row['Name']:25s} {row['Pos']:3s} (Age {int(row['Age_2025'])}→{int(row['Age_2026'])})")
            print(f"  2025: {row[f'{metric}_2025']:7.1f} | 2026: {row[f'{metric}_2026']:7.1f} | Change: +{row[change_col]:6.1f}")
            print(f"  2025 Innings: {row['Inn_2025']:.0f}")
            print()
        
        # Top decliners
        print(f"\n📉 TOP {min(n, len(valid_changes))} DECLINERS:")
        print("-" * 80)
        top_decliners = valid_changes.nsmallest(n, change_col)
        
        for idx, row in top_decliners.iterrows():
            print(f"{row['Name']:25s} {row['Pos']:3s} (Age {int(row['Age_2025'])}→{int(row['Age_2026'])})")
            print(f"  2025: {row[f'{metric}_2025']:7.1f} | 2026: {row[f'{metric}_2026']:7.1f} | Change: {row[change_col]:6.1f}")
            print(f"  2025 Innings: {row['Inn_2025']:.0f}")
            print()


def export_to_csv(changes, position_group):
    """Export full comparison to CSV"""
    output_file = PIPELINE_DIR / f'fielding_changes_2025_to_2026_{position_group}.csv'
    changes.to_csv(output_file, index=False)
    print(f"\n💾 Full comparison exported to: {output_file}")
    print(f"   ({len(changes)} players)")


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(
        description='Analyze year-over-year fielding changes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--position-group',
        choices=['outfield', 'infield', 'catcher', 'all'],
        default='outfield',
        help='Position group to analyze (default: outfield)'
    )
    parser.add_argument(
        '--min-innings',
        type=int,
        default=100,
        help='Minimum innings to qualify (default: 100)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top changes to display (default: 20)'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export full comparison to CSV'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not PREDICTIONS_FILE.exists():
        print(f"❌ Predictions file not found: {PREDICTIONS_FILE}")
        print("   Run predictions first: python scripts/predict_models.py --model-type fielding")
        return 1
    
    if not HISTORICAL_FILE.exists():
        print(f"❌ Historical data file not found: {HISTORICAL_FILE}")
        return 1
    
    # Load data
    predictions_df, historical_df = load_data()
    
    # Determine which position groups to analyze
    position_groups = ['outfield', 'infield', 'catcher'] if args.position_group == 'all' else [args.position_group]
    
    for pos_group in position_groups:
        # Calculate changes
        result = calculate_changes(
            historical_df,
            predictions_df,
            position_group=pos_group,
            min_innings=args.min_innings
        )
        
        if result is None:
            continue
        
        changes, metrics = result
        
        # Display top changes
        display_top_changes(changes, metrics, n=args.top_n)
        
        # Export if requested
        if args.export:
            export_to_csv(changes, pos_group)
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
