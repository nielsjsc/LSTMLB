#!/usr/bin/env python3
"""
Backtesting and Confidence Analysis for MLB Predictions

This script:
1. Generates historical predictions for each year (2000-2025)
2. Compares predictions to actual performance
3. Calculates prediction errors grouped by sample size
4. Fits error curves to derive empirical confidence intervals

Author: Niels Christoffersen
Date: January 2026
"""

import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
BACKTEST_DIR = DATA_DIR / 'backtest'
RESULTS_DIR = BACKTEST_DIR / 'results'

# Ensure directories exist
BACKTEST_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def generate_historical_predictions(
    model_type: str,
    start_year: int = 2000,
    end_year: int = 2024,
    use_pretrained: bool = True
):
    """
    Generate predictions for each historical year.
    
    Args:
        model_type: 'batter', 'pitcher', or 'fielding'
        start_year: First cutoff year to generate predictions from
        end_year: Last cutoff year to generate predictions from
        use_pretrained: Whether to use pretrained models
    """
    logger.info(f"=" * 80)
    logger.info(f"Generating historical predictions for {model_type}")
    logger.info(f"Years: {start_year} to {end_year}")
    logger.info(f"=" * 80)
    
    for cutoff_year in range(start_year, end_year + 1):
        # Create year-specific output directory
        year_dir = BACKTEST_DIR / model_type / f"cutoff_{cutoff_year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        
        # predict_models.py saves as {model_type}_predictions.csv in the output dir
        output_file = year_dir / f"{model_type}_predictions.csv"
        
        # Skip if already exists
        if output_file.exists():
            logger.info(f"Predictions for {cutoff_year} already exist, skipping")
            continue
        
        logger.info(f"Generating predictions for cutoff year {cutoff_year}...")
        
        # Build command
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / 'predict_models.py'),
            '--model-type', model_type,
            '--cutoff-year', str(cutoff_year),
            '--output-dir', str(year_dir)
        ]
        
        # Only add --use-pretrained for batter/pitcher (fielding doesn't use it)
        if use_pretrained and model_type in ['batter', 'pitcher']:
            cmd.append('--use-pretrained')
        
        # Run prediction
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Generated predictions for {cutoff_year}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate predictions for {cutoff_year}")
            logger.error(f"Error: {e.stderr}")
            continue


def load_actual_data(model_type: str) -> pd.DataFrame:
    """Load actual historical performance data."""
    if model_type == 'batter':
        data_file = DATA_DIR / 'historic_mlb' / 'mlb_batting_data_1950_2025.csv'
    elif model_type == 'pitcher':
        data_file = DATA_DIR / 'historic_mlb' / 'mlb_pitching_data_1950_2025.csv'
    elif model_type == 'fielding':
        data_file = DATA_DIR / 'historic_mlb' / 'mlb_fielding_data_2000_2025_with_statcast.csv'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Loading actual data from {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} rows of actual data")
    return df


def calculate_sample_size(
    player_data: pd.DataFrame,
    cutoff_year: int,
    seq_length: int,
    sample_col: str
) -> float:
    """
    Calculate sample size as sum of last n seasons where n = seq_length.
    
    Args:
        player_data: Historical data for a single player
        cutoff_year: Last year of available data
        seq_length: Number of seasons to look back
        sample_col: Column name for sample size (e.g., 'PA', 'IP')
    
    Returns:
        Sum of sample_col over last seq_length seasons
    """
    # Get seasons up to and including cutoff_year
    historical = player_data[player_data['Season'] <= cutoff_year].copy()
    
    # Sort by season and take last seq_length seasons
    historical = historical.sort_values('Season').tail(seq_length)
    
    # Sum the sample size column
    return historical[sample_col].sum()


def analyze_batter_predictions(
    cutoff_year: int,
    actual_df: pd.DataFrame,
    seq_length: int = 5,
    min_actual_pa: int = 50
) -> pd.DataFrame:
    """
    Analyze batter predictions vs actual performance.
    
    Args:
        cutoff_year: Year predictions were made from
        actual_df: DataFrame with actual performance data
        seq_length: Number of seasons used for prediction sequence
        min_actual_pa: Minimum PA in target year to include (default 100)
    
    Returns:
        DataFrame with prediction errors and sample sizes
    """
    # Load predictions
    pred_file = BACKTEST_DIR / 'batter' / f'cutoff_{cutoff_year}' / 'batter_predictions.csv'
    
    if not pred_file.exists():
        logger.warning(f"Prediction file not found: {pred_file}")
        return None
    
    pred_df = pd.read_csv(pred_file)
    
    # Target year is cutoff_year + 1 (first prediction year)
    target_year = cutoff_year + 1
    
    # Filter to only first year predictions
    pred_df = pred_df[pred_df['Year'] == target_year].copy()
    
    # Get actual performance for target year
    actual_target = actual_df[actual_df['Season'] == target_year].copy()
    
    # Metrics to analyze
    metrics = ['wOBA', 'K%', 'BB%', 'AVG', 'OBP', 'SLG', 'wRC+']
    
    errors = []
    
    for _, pred_row in pred_df.iterrows():
        player_id = pred_row['IDfg']
        
        # Get actual performance for this player in target year
        actual_row = actual_target[actual_target['IDfg'] == player_id]
        
        if len(actual_row) == 0:
            continue  # Player didn't play in target year
        
        actual_row = actual_row.iloc[0]
        
        # Filter by minimum PA
        if actual_row.get('PA', 0) < min_actual_pa:
            continue
        
        # Calculate sample size (sum of PA over last seq_length seasons)
        player_historical = actual_df[actual_df['IDfg'] == player_id]
        sample_size = calculate_sample_size(
            player_historical,
            cutoff_year,
            seq_length,
            'PA'
        )
        
        # Calculate errors for each metric
        error_dict = {
            'IDfg': player_id,
            'Name': pred_row.get('Name', ''),
            'cutoff_year': cutoff_year,
            'target_year': target_year,
            'sample_size': sample_size,
            'actual_PA': actual_row.get('PA', 0)
        }
        
        for metric in metrics:
            if metric in pred_row and metric in actual_row:
                pred_val = pred_row[metric]
                actual_val = actual_row[metric]
                
                # Skip if either is NaN
                if pd.notna(pred_val) and pd.notna(actual_val):
                    error = pred_val - actual_val
                    error_dict[f'{metric}_error'] = error
                    error_dict[f'{metric}_abs_error'] = abs(error)
                    error_dict[f'{metric}_pred'] = pred_val
                    error_dict[f'{metric}_actual'] = actual_val
        
        errors.append(error_dict)
    
    if errors:
        errors_df = pd.DataFrame(errors)
        logger.info(f"Analyzed {len(errors_df)} predictions for {cutoff_year}")
        return errors_df
    else:
        logger.warning(f"No valid predictions found for {cutoff_year}")
        return None


def analyze_pitcher_predictions(
    cutoff_year: int,
    actual_df: pd.DataFrame,
    seq_length: int = 3,
    min_actual_ip: int = 20,
    role_filter: str = None
) -> pd.DataFrame:
    """
    Analyze pitcher predictions vs actual performance.
    
    Args:
        cutoff_year: Year predictions were made from
        actual_df: DataFrame with actual performance data
        seq_length: Number of seasons used for prediction sequence
        min_actual_ip: Minimum IP in target year to include (default 20)
        role_filter: Filter by role ('SP', 'RP', or None for all)
    
    Returns:
        DataFrame with prediction errors and sample sizes
    """
    # Load predictions
    pred_file = BACKTEST_DIR / 'pitcher' / f'cutoff_{cutoff_year}' / 'pitcher_predictions.csv'
    
    if not pred_file.exists():
        logger.warning(f"Prediction file not found: {pred_file}")
        return None
    
    pred_df = pd.read_csv(pred_file)
    
    # Target year is cutoff_year + 1
    target_year = cutoff_year + 1
    
    # Filter to only first year predictions
    pred_df = pred_df[pred_df['Year'] == target_year].copy()
    
    # Filter by role if specified
    if role_filter:
        pred_df = pred_df[pred_df['Role'] == role_filter].copy()
    
    # Get actual performance for target year
    actual_target = actual_df[actual_df['Season'] == target_year].copy()
    
    # Metrics to analyze
    metrics = ['ERA', 'FIP', 'K%', 'BB%', 'WHIP', 'K/9', 'BB/9']
    
    errors = []
    
    for _, pred_row in pred_df.iterrows():
        player_id = pred_row['IDfg']
        
        # Get actual performance for this player in target year
        actual_row = actual_target[actual_target['IDfg'] == player_id]
        
        if len(actual_row) == 0:
            continue  # Player didn't pitch in target year
        
        actual_row = actual_row.iloc[0]
        
        # Filter by minimum IP
        if actual_row.get('IP', 0) < min_actual_ip:
            continue
        
        # Calculate sample size (sum of IP over last seq_length seasons)
        player_historical = actual_df[actual_df['IDfg'] == player_id]
        sample_size = calculate_sample_size(
            player_historical,
            cutoff_year,
            seq_length,
            'IP'
        )
        
        # Calculate errors for each metric
        error_dict = {
            'IDfg': player_id,
            'Name': pred_row.get('Name', ''),
            'Role': pred_row.get('Role', ''),
            'cutoff_year': cutoff_year,
            'target_year': target_year,
            'sample_size': sample_size,
            'actual_IP': actual_row.get('IP', 0)
        }
        
        for metric in metrics:
            if metric in pred_row and metric in actual_row:
                pred_val = pred_row[metric]
                actual_val = actual_row[metric]
                
                # Skip if either is NaN
                if pd.notna(pred_val) and pd.notna(actual_val):
                    error = pred_val - actual_val
                    error_dict[f'{metric}_error'] = error
                    error_dict[f'{metric}_abs_error'] = abs(error)
                    error_dict[f'{metric}_pred'] = pred_val
                    error_dict[f'{metric}_actual'] = actual_val
        
        errors.append(error_dict)
    
    if errors:
        errors_df = pd.DataFrame(errors)
        logger.info(f"Analyzed {len(errors_df)} predictions for {cutoff_year}")
        return errors_df
    else:
        logger.warning(f"No valid predictions found for {cutoff_year}")
        return None


# =============================================================================
# FIELDING ANALYSIS FUNCTIONS
# =============================================================================

# Position group definitions
FIELDING_POSITION_GROUPS = {
    'infield': ['1B', '2B', '3B', 'SS'],
    'outfield': ['LF', 'CF', 'RF'],
    'catcher': ['C']
}

# Metrics by position group (based on config files)
FIELDING_METRICS = {
    'infield': ['OAA/150', 'DRS/150', 'sc_total_runs/150', 'sc_range_runs/150', 'sc_arm_runs/150', 'sc_dp_runs/150'],
    'outfield': ['OAA/150', 'DRS/150', 'sc_total_runs/150', 'sc_range_runs/150', 'sc_arm_runs/150'],
    'catcher': ['sc_total_runs/150', 'sc_framing_runs/150', 'sc_throwing_runs/150', 'sc_blocking_runs/150']
}


def get_position_group(pos: str) -> Optional[str]:
    """Map a specific position to its position group."""
    for group, positions in FIELDING_POSITION_GROUPS.items():
        if pos in positions:
            return group
    return None


def analyze_fielding_predictions(
    cutoff_year: int,
    actual_df: pd.DataFrame,
    seq_length: int = 5,
    min_actual_inn: int = 50,
    position_group_filter: str = None
) -> pd.DataFrame:
    """
    Analyze fielding predictions vs actual performance.
    
    This compares position-to-position across seasons. A player's prediction
    at SS is compared to their actual performance at SS in the target year.
    
    Args:
        cutoff_year: Year predictions were made from
        actual_df: DataFrame with actual fielding data
        seq_length: Number of seasons used for prediction sequence
        min_actual_inn: Minimum innings in target year to include
        position_group_filter: Filter by position group ('infield', 'outfield', 'catcher')
    
    Returns:
        DataFrame with prediction errors and sample sizes
    """
    # Load predictions
    pred_file = BACKTEST_DIR / 'fielding' / f'cutoff_{cutoff_year}' / 'fielding_predictions.csv'
    
    if not pred_file.exists():
        logger.warning(f"Prediction file not found: {pred_file}")
        return None
    
    pred_df = pd.read_csv(pred_file)
    
    # Target year is cutoff_year + 1
    target_year = cutoff_year + 1
    
    # Filter to only first year predictions
    pred_df = pred_df[pred_df['Year'] == target_year].copy()
    
    # Filter by position group if specified
    if position_group_filter:
        valid_positions = FIELDING_POSITION_GROUPS.get(position_group_filter, [])
        pred_df = pred_df[pred_df['Pos'].isin(valid_positions)].copy()
    
    # Get actual performance for target year
    actual_target = actual_df[actual_df['Season'] == target_year].copy()
    
    errors = []
    
    for _, pred_row in pred_df.iterrows():
        player_id = pred_row['IDfg']
        position = pred_row['Pos']
        position_group = get_position_group(position)
        
        if position_group is None:
            continue
            
        # Get metrics for this position group
        metrics = FIELDING_METRICS.get(position_group, [])
        
        # Get actual performance for this player at this position in target year
        actual_row = actual_target[
            (actual_target['IDfg'] == player_id) & 
            (actual_target['Pos'] == position)
        ]
        
        if len(actual_row) == 0:
            continue  # Player didn't play this position in target year
        
        actual_row = actual_row.iloc[0]
        
        # Filter by minimum innings
        if actual_row.get('Inn', 0) < min_actual_inn:
            continue
        
        # Calculate sample size (sum of Inn at this position over last seq_length seasons)
        player_historical = actual_df[
            (actual_df['IDfg'] == player_id) & 
            (actual_df['Pos'] == position)
        ]
        sample_size = calculate_sample_size(
            player_historical,
            cutoff_year,
            seq_length,
            'Inn'
        )
        
        # Calculate errors for each metric
        error_dict = {
            'IDfg': player_id,
            'Name': pred_row.get('Name', ''),
            'Pos': position,
            'Position_Group': position_group,
            'cutoff_year': cutoff_year,
            'target_year': target_year,
            'sample_size': sample_size,
            'actual_Inn': actual_row.get('Inn', 0)
        }
        
        for metric in metrics:
            if metric in pred_row and metric in actual_row:
                pred_val = pred_row[metric]
                actual_val = actual_row[metric]
                
                # Skip if either is NaN
                if pd.notna(pred_val) and pd.notna(actual_val):
                    error = pred_val - actual_val
                    error_dict[f'{metric}_error'] = error
                    error_dict[f'{metric}_abs_error'] = abs(error)
                    error_dict[f'{metric}_pred'] = pred_val
                    error_dict[f'{metric}_actual'] = actual_val
        
        errors.append(error_dict)
    
    if errors:
        errors_df = pd.DataFrame(errors)
        logger.info(f"Analyzed {len(errors_df)} fielding predictions for {cutoff_year}")
        return errors_df
    else:
        logger.warning(f"No valid fielding predictions found for {cutoff_year}")
        return None


def combine_all_errors(
    model_type: str,
    start_year: int,
    end_year: int,
    actual_df: pd.DataFrame,
    seq_length: int,
    min_actual_sample: int = None,
    role_filter: str = None,
    position_group_filter: str = None
) -> pd.DataFrame:
    """Combine error analysis across all years."""
    
    all_errors = []
    
    if position_group_filter:
        desc = f"Analyzing {position_group_filter} errors"
    elif role_filter:
        desc = f"Analyzing {role_filter} errors"
    else:
        desc = f"Analyzing {model_type} errors"
        
    for cutoff_year in tqdm(range(start_year, end_year + 1), desc=desc):
        if model_type == 'batter':
            min_pa = min_actual_sample if min_actual_sample else 50
            errors_df = analyze_batter_predictions(cutoff_year, actual_df, seq_length, min_actual_pa=min_pa)
        elif model_type == 'pitcher':
            min_ip = min_actual_sample if min_actual_sample else 20
            errors_df = analyze_pitcher_predictions(cutoff_year, actual_df, seq_length, min_actual_ip=min_ip, role_filter=role_filter)
        elif model_type == 'fielding':
            min_inn = min_actual_sample if min_actual_sample else 50
            errors_df = analyze_fielding_predictions(cutoff_year, actual_df, seq_length, min_actual_inn=min_inn, position_group_filter=position_group_filter)
        else:
            continue
        
        if errors_df is not None:
            all_errors.append(errors_df)
    
    if all_errors:
        combined = pd.concat(all_errors, ignore_index=True)
        logger.info(f"Combined {len(combined)} total predictions")
        return combined
    else:
        logger.error("No errors to combine")
        return None


def create_sample_size_bins(
    errors_df: pd.DataFrame,
    sample_col: str = 'sample_size',
    bin_width: int = 100
) -> pd.DataFrame:
    """
    Group errors by sample size bins.
    
    Args:
        errors_df: DataFrame with prediction errors
        sample_col: Column name for sample size
        bin_width: Width of each bin (e.g., 100 PA)
    
    Returns:
        DataFrame with binned statistics
    """
    # Create bins
    max_sample = errors_df[sample_col].max()
    bins = list(range(0, int(max_sample) + bin_width, bin_width))
    
    # Add bin labels
    errors_df['sample_bin'] = pd.cut(
        errors_df[sample_col],
        bins=bins,
        labels=[f"{b}-{b+bin_width}" for b in bins[:-1]],
        include_lowest=True
    )
    
    return errors_df


def calculate_binned_metrics(
    errors_df: pd.DataFrame,
    metric: str,
    model_type: str
) -> pd.DataFrame:
    """
    Calculate RMSE and MAE for each sample size bin.
    
    Args:
        errors_df: DataFrame with binned errors
        metric: Metric to analyze (e.g., 'wOBA', 'ERA')
        model_type: 'batter' or 'pitcher'
    
    Returns:
        DataFrame with bin statistics
    """
    error_col = f'{metric}_error'
    abs_error_col = f'{metric}_abs_error'
    
    # Filter to rows with this metric
    metric_df = errors_df.dropna(subset=[error_col])
    
    # Group by bin
    binned = metric_df.groupby('sample_bin').agg({
        error_col: ['count', lambda x: np.sqrt(np.mean(x**2))],  # RMSE
        abs_error_col: 'mean',  # MAE
        'sample_size': 'mean'
    }).reset_index()
    
    # Flatten column names
    binned.columns = ['sample_bin', 'count', 'RMSE', 'MAE', 'avg_sample_size']
    binned['metric'] = metric
    
    return binned


def fit_error_curve(
    binned_df: pd.DataFrame,
    error_type: str = 'RMSE'
) -> Tuple[float, float, np.ndarray]:
    """
    Fit error curve: error = a/sqrt(sample_size) + b
    
    Args:
        binned_df: DataFrame with binned error statistics
        error_type: 'RMSE' or 'MAE'
    
    Returns:
        (a, b, fitted_values) where fitted_values are predictions on the data points
    """
    # Define the curve function
    def error_curve(sample_size, a, b):
        return a / np.sqrt(sample_size) + b
    
    # Filter out bins with too few samples
    fit_df = binned_df[binned_df['count'] >= 10].copy()
    
    if len(fit_df) < 3:
        logger.warning(f"Not enough data points to fit curve (only {len(fit_df)} bins)")
        return None, None, None
    
    x = fit_df['avg_sample_size'].values
    y = fit_df[error_type].values
    
    try:
        # Fit the curve
        popt, _ = curve_fit(error_curve, x, y, p0=[1.0, 0.01], maxfev=10000)
        a, b = popt
        
        # Calculate fitted values
        fitted = error_curve(x, a, b)
        
        # Calculate R²
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        logger.info(f"Fitted curve: error = {a:.4f}/sqrt(sample_size) + {b:.4f}")
        logger.info(f"R² = {r_squared:.4f}")
        
        return a, b, fitted
    except Exception as e:
        logger.error(f"Failed to fit curve: {e}")
        return None, None, None


def plot_error_curves(
    binned_df: pd.DataFrame,
    metric: str,
    model_type: str,
    sample_col_name: str = 'PA'
):
    """Plot error curves with fitted models."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE plot
    ax1.scatter(binned_df['avg_sample_size'], binned_df['RMSE'], 
                alpha=0.6, s=binned_df['count']*2, label='Actual RMSE')
    
    # Fit and plot RMSE curve
    a_rmse, b_rmse, fitted_rmse = fit_error_curve(binned_df, 'RMSE')
    if a_rmse is not None:
        x_smooth = np.linspace(binned_df['avg_sample_size'].min(), 
                               binned_df['avg_sample_size'].max(), 100)
        y_smooth = a_rmse / np.sqrt(x_smooth) + b_rmse
        ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                label=f'Fit: {a_rmse:.3f}/√{sample_col_name} + {b_rmse:.3f}')
    
    ax1.set_xlabel(f'Average {sample_col_name}')
    ax1.set_ylabel('RMSE')
    ax1.set_title(f'{model_type.title()} - {metric} RMSE vs Sample Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.scatter(binned_df['avg_sample_size'], binned_df['MAE'], 
                alpha=0.6, s=binned_df['count']*2, label='Actual MAE')
    
    # Fit and plot MAE curve
    a_mae, b_mae, fitted_mae = fit_error_curve(binned_df, 'MAE')
    if a_mae is not None:
        x_smooth = np.linspace(binned_df['avg_sample_size'].min(), 
                               binned_df['avg_sample_size'].max(), 100)
        y_smooth = a_mae / np.sqrt(x_smooth) + b_mae
        ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2,
                label=f'Fit: {a_mae:.3f}/√{sample_col_name} + {b_mae:.3f}')
    
    ax2.set_xlabel(f'Average {sample_col_name}')
    ax2.set_ylabel('MAE')
    ax2.set_title(f'{model_type.title()} - {metric} MAE vs Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = RESULTS_DIR / f'{model_type}_{metric}_error_curves.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_file}")
    plt.close()


def save_error_curve_summary(all_curve_params: List[Dict], output_file: Path):
    """Save fitted error curve parameters to CSV."""
    summary_df = pd.DataFrame(all_curve_params)
    summary_df.to_csv(output_file, index=False)
    logger.info(f"Saved error curve summary to {output_file}")
    
    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("ERROR CURVE SUMMARY")
    logger.info("=" * 80)
    for _, row in summary_df.iterrows():
        logger.info(f"{row['model_type']:8s} | {row['metric']:6s} | {row['error_type']:4s} | "
                   f"error = {row['a']:.4f}/√sample + {row['b']:.4f} | R² = {row['r_squared']:.4f}")
    logger.info("=" * 80)


def main():
    """Main execution function."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Backtest predictions and analyze confidence')
    parser.add_argument('--model-type', choices=['batter', 'pitcher', 'fielding', 'all'], 
                       default='all', help='Model type to backtest')
    parser.add_argument('--start-year', type=int, default=2000,
                       help='First cutoff year for predictions (default: 2000, use 2016 for fielding)')
    parser.add_argument('--end-year', type=int, default=2024,
                       help='Last cutoff year for predictions')
    parser.add_argument('--generate-only', action='store_true',
                       help='Only generate predictions, skip analysis')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing predictions, skip generation')
    parser.add_argument('--min-pa', type=int, default=100,
                       help='Minimum PA in target year for batters (default 100, matches pipeline)')
    parser.add_argument('--min-ip', type=int, default=None,
                       help='Minimum IP in target year for pitchers (default: 25 for SP, 15 for RP)')
    parser.add_argument('--min-inn', type=int, default=50,
                       help='Minimum innings in target year for fielders (default: 50)')
    
    args = parser.parse_args()
    
    # Determine model types to process
    if args.model_type == 'all':
        model_types = ['batter', 'pitcher', 'fielding']
    else:
        model_types = [args.model_type]
    
    # Step 1: Generate predictions
    if not args.analyze_only:
        for model_type in model_types:
            # Use appropriate start year (fielding only has Statcast data from 2016)
            if model_type == 'fielding':
                start_year = max(args.start_year, 2016)  # Statcast fielding starts 2016
                logger.info(f"Using start_year={start_year} for fielding (Statcast era)")
            else:
                start_year = args.start_year
            
            generate_historical_predictions(
                model_type=model_type,
                start_year=start_year,
                end_year=args.end_year,
                use_pretrained=True
            )
    
    # Step 2: Analyze predictions
    if not args.generate_only:
        all_curve_params = []  # Store all fitted curve parameters
        
        for model_type in model_types:
            # Determine sub-groups to analyze
            if model_type == 'pitcher':
                sub_groups = [('SP', None), ('RP', None)]  # (role_filter, position_group_filter)
            elif model_type == 'fielding':
                sub_groups = [(None, 'infield'), (None, 'outfield'), (None, 'catcher')]
            else:
                sub_groups = [(None, None)]
            
            for role_filter, position_group_filter in sub_groups:
                # Determine display name
                if position_group_filter:
                    display_type = position_group_filter
                elif role_filter:
                    display_type = role_filter
                else:
                    display_type = model_type
                    
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Analyzing {display_type} predictions")
                logger.info(f"{'=' * 80}")
                
                # Load actual data
                actual_df = load_actual_data(model_type)
                
                # Model-specific configuration
                if model_type == 'batter':
                    seq_length = 5
                    min_actual_sample = args.min_pa
                    sample_col_name = 'PA'
                    metrics = ['wOBA', 'K%', 'BB%', 'wRC+']
                    bin_width = 100
                    start_year = args.start_year
                elif model_type == 'pitcher':
                    seq_length = 3
                    if role_filter == 'SP':
                        min_actual_sample = args.min_ip if args.min_ip else 25
                    elif role_filter == 'RP':
                        min_actual_sample = args.min_ip if args.min_ip else 15
                    else:
                        min_actual_sample = args.min_ip if args.min_ip else 20
                    sample_col_name = 'IP'
                    metrics = ['ERA', 'FIP', 'K%', 'BB%']
                    bin_width = 50
                    start_year = args.start_year
                elif model_type == 'fielding':
                    # Position group specific config
                    if position_group_filter == 'catcher':
                        seq_length = 3  # Catchers use shorter sequences
                    else:
                        seq_length = 5
                    min_actual_sample = args.min_inn
                    sample_col_name = 'Inn'
                    metrics = FIELDING_METRICS.get(position_group_filter, ['sc_total_runs/150'])
                    bin_width = 100  # Innings bins
                    start_year = max(args.start_year, 2016)  # Statcast era
                else:
                    continue
                
                # Combine all errors
                errors_df = combine_all_errors(
                    model_type=model_type,
                    start_year=start_year,
                    end_year=args.end_year,
                    actual_df=actual_df,
                    seq_length=seq_length,
                    min_actual_sample=min_actual_sample,
                    role_filter=role_filter,
                    position_group_filter=position_group_filter
                )
            
                if errors_df is None:
                    logger.error(f"No errors to analyze for {display_type}")
                    continue
                
                # Save raw errors
                errors_file = RESULTS_DIR / f'{display_type}_prediction_errors.csv'
                errors_df.to_csv(errors_file, index=False)
                logger.info(f"Saved raw errors to {errors_file}")
                
                # Create sample size bins
                errors_df = create_sample_size_bins(errors_df, 'sample_size', bin_width)
                
                all_binned = []
                for metric in metrics:
                    logger.info(f"\nAnalyzing {metric}...")
                    binned = calculate_binned_metrics(errors_df, metric, display_type)
                    
                    if len(binned) > 0:
                        all_binned.append(binned)
                        
                        # Fit curves and save parameters
                        for error_type in ['RMSE', 'MAE']:
                            logger.info(f"  {error_type} curve:")
                            a, b, fitted = fit_error_curve(binned, error_type)
                            
                            if a is not None:
                                # Calculate R²
                                y = binned[binned['count'] >= 10][error_type].values
                                ss_res = np.sum((y - fitted) ** 2)
                                ss_tot = np.sum((y - np.mean(y)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot)
                                
                                all_curve_params.append({
                                    'model_type': display_type,
                                    'metric': metric,
                                    'error_type': error_type,
                                    'a': a,
                                    'b': b,
                                    'r_squared': r_squared,
                                    'n_bins': len(binned[binned['count'] >= 10])
                                })
                        
                        # Plot
                        plot_error_curves(binned, metric, display_type, sample_col_name)
                
                # Save binned statistics
                if all_binned:
                    combined_binned = pd.concat(all_binned, ignore_index=True)
                    binned_file = RESULTS_DIR / f'{display_type}_binned_errors.csv'
                    combined_binned.to_csv(binned_file, index=False)
                    logger.info(f"Saved binned statistics to {binned_file}")
    
        # Save error curve summary
        if all_curve_params:
            summary_file = RESULTS_DIR / 'error_curve_summary.csv'
            save_error_curve_summary(all_curve_params, summary_file)
    
    logger.info(f"\n{'=' * 80}")
    logger.info("Backtesting complete!")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'=' * 80}")


if __name__ == '__main__':
    main()
