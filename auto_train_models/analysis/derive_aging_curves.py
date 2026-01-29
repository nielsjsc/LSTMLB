"""
Empirical Aging Curve Derivation v2
====================================
Derives age-band specific decline rates from historical MLB data.

Instead of a misleading single "peak age", outputs decline rates by age band:
- 21-25 (development)
- 26-30 (prime)  
- 31-35 (early decline)
- 36-40 (late career)

Output: aging_parameters.json

Usage:
    python derive_aging_curves_v2.py [--min-year 2000]
"""

import pandas as pd
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historic_mlb"

# Minimum thresholds
MIN_IP_PITCHER = 40
MIN_PA_BATTER = 150
MIN_INN_FIELDER = 200

# Age bands for analysis
AGE_BANDS = [
    ('21-25', 21, 25),
    ('26-30', 26, 30),
    ('31-35', 31, 35),
    ('36-40', 36, 40),
    ('41-45', 41, 45),
]

# Stats to analyze
BATTER_STATS = {
    'wRC+': {'inverted': False, 'description': 'Weighted Runs Created Plus'},
    'AVG': {'inverted': False, 'description': 'Batting Average'},
    'OBP': {'inverted': False, 'description': 'On-Base Percentage'},
    'SLG': {'inverted': False, 'description': 'Slugging Percentage'},
    'BB%': {'inverted': False, 'description': 'Walk Rate'},
    'K%': {'inverted': True, 'description': 'Strikeout Rate (higher=worse)'},
    'ISO': {'inverted': False, 'description': 'Isolated Power'},
    'wOBA': {'inverted': False, 'description': 'Weighted On-Base Average'},
}

PITCHER_STATS = {
    'ERA': {'inverted': True, 'description': 'Earned Run Average (higher=worse)'},
    'FIP': {'inverted': True, 'description': 'Fielding Independent Pitching'},
    'K%': {'inverted': False, 'description': 'Strikeout Rate'},
    'BB%': {'inverted': True, 'description': 'Walk Rate (higher=worse)'},
    'WHIP': {'inverted': True, 'description': 'Walks+Hits per IP'},
    'K/9': {'inverted': False, 'description': 'Strikeouts per 9 IP'},
    'xFIP': {'inverted': True, 'description': 'Expected FIP'},
}

BASERUNNING_STATS = {
    # Traditional stats
    'BsR': {'inverted': False, 'description': 'Baserunning Runs'},
    'Spd': {'inverted': False, 'description': 'Speed Score'},
    'wSB': {'inverted': False, 'description': 'Weighted Stolen Base Runs'},
    # Statcast baserunning metrics (used by model)
    'sc_sprint_speed': {'inverted': False, 'description': 'Sprint Speed (ft/sec)'},
    'sc_baserunning_runner_runs_tot': {'inverted': False, 'description': 'Total Baserunning Runs (Statcast)'},
    'sc_baserunning_runner_runs_XB': {'inverted': False, 'description': 'Extra Base Taking Runs'},
    'sc_baserunning_runner_runs_SBX': {'inverted': False, 'description': 'Stolen Base Runs (Statcast)'},
}

# Fielding stats - organized by position group (matching config INPUT_FEATURES)
# Note: Using /150 normalized versions as used in actual models

FIELDING_STATS_INFIELD = {
    # From defense_infield_config.py INPUT_FEATURES
    'OAA/150': {'inverted': False, 'description': 'Outs Above Average per 150 games'},
    'DRS/150': {'inverted': False, 'description': 'Defensive Runs Saved per 150 games'},
    'sc_total_runs/150': {'inverted': False, 'description': 'Total Fielding Runs (Statcast) per 150'},
    'sc_range_runs/150': {'inverted': False, 'description': 'Range Runs per 150'},
    'sc_arm_runs/150': {'inverted': False, 'description': 'Arm Runs per 150'},
    'sc_dp_runs/150': {'inverted': False, 'description': 'Double Play Runs per 150'},
}

FIELDING_STATS_OUTFIELD = {
    # From defense_outfield_config.py INPUT_FEATURES
    'OAA/150': {'inverted': False, 'description': 'Outs Above Average per 150 games'},
    'DRS/150': {'inverted': False, 'description': 'Defensive Runs Saved per 150 games'},
    'sc_total_runs/150': {'inverted': False, 'description': 'Total Fielding Runs (Statcast) per 150'},
    'sc_range_runs/150': {'inverted': False, 'description': 'Range Runs per 150'},
    'sc_arm_runs/150': {'inverted': False, 'description': 'Arm Runs per 150'},
}

FIELDING_STATS_CATCHER = {
    # From defense_catcher_config.py INPUT_FEATURES
    'sc_total_runs/150': {'inverted': False, 'description': 'Total Catching Runs (Statcast) per 150'},
    'sc_framing_runs/150': {'inverted': False, 'description': 'Framing Runs per 150'},
    'sc_throwing_runs/150': {'inverted': False, 'description': 'Throwing Runs per 150'},
    'sc_blocking_runs/150': {'inverted': False, 'description': 'Blocking Runs per 150'},
}


def calculate_fielding_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-150 game rate stats for fielding metrics.
    Matches the calculation in core/data_processing.py calculate_rate_stats()
    """
    df = df.copy()
    
    # Fielding rate stats (per 150 games = ~1350 innings)
    INNINGS_PER_150_GAMES = 1350.0
    
    # Raw stats to normalize
    fielding_raw_stats = [
        'DRS', 'OAA', 
        'sc_total_runs', 'sc_range_runs', 'sc_arm_runs', 'sc_dp_runs',
        'sc_framing_runs', 'sc_throwing_runs', 'sc_blocking_runs'
    ]
    
    for stat in fielding_raw_stats:
        if stat in df.columns:
            rate_col = f'{stat}/150'
            # Avoid division by zero
            df[rate_col] = np.where(
                df['Inn'] > 0,
                df[stat] * INNINGS_PER_150_GAMES / df['Inn'],
                0
            )
    
    return df


def load_data(data_type: str, min_year: int, position_filter: List[str] = None) -> pd.DataFrame:
    """Load and filter data by type."""
    if data_type == 'batting':
        path = DATA_DIR / "mlb_batting_data_1950_2025_with_statcast.csv"
        df = pd.read_csv(path, low_memory=False)
        df = df[(df['Season'] >= min_year) & (df['PA'] >= MIN_PA_BATTER)]
        logger.info(f"Loaded {len(df):,} batter-seasons")
    elif data_type == 'pitching':
        path = DATA_DIR / "mlb_pitching_data_1950_2025_with_statcast.csv"
        df = pd.read_csv(path, low_memory=False)
        df = df[(df['Season'] >= min_year) & (df['IP'] >= MIN_IP_PITCHER)]
        logger.info(f"Loaded {len(df):,} pitcher-seasons")
    elif data_type == 'fielding':
        path = DATA_DIR / "mlb_fielding_data_2000_2025_with_statcast.csv"
        df = pd.read_csv(path, low_memory=False)
        df = df[(df['Season'] >= min_year) & (df['Inn'] >= MIN_INN_FIELDER)]
        
        # Filter by position if specified
        if position_filter:
            df = df[df['Pos'].isin(position_filter)]
            logger.info(f"Loaded {len(df):,} fielder-seasons for positions: {position_filter}")
        else:
            logger.info(f"Loaded {len(df):,} fielder-seasons")
        
        # Calculate rate stats for fielding
        df = calculate_fielding_rate_stats(df)
    return df


def compute_deltas(df: pd.DataFrame, stat: str, group_by_position: bool = False) -> pd.DataFrame:
    """
    Compute within-player year-over-year changes.
    
    Args:
        df: DataFrame with player data
        stat: Stat column to analyze
        group_by_position: If True, only compare same position year-over-year
                          (required for fielding data where positions matter)
    """
    results = []
    
    if group_by_position and 'Pos' in df.columns:
        # For fielding: group by player AND position
        # Only compare SS 2023 to SS 2024, not SS 2023 to 2B 2024
        df = df.sort_values(['IDfg', 'Pos', 'Season'])
        
        for (player_id, pos), player_pos_df in df.groupby(['IDfg', 'Pos']):
            player_pos_df = player_pos_df.sort_values('Season')
            for i in range(len(player_pos_df) - 1):
                r1, r2 = player_pos_df.iloc[i], player_pos_df.iloc[i + 1]
                if r2['Season'] - r1['Season'] != 1:
                    continue
                if pd.isna(r1[stat]) or pd.isna(r2[stat]):
                    continue
                results.append({
                    'age': r1['Age'],
                    'delta': r2[stat] - r1[stat],
                    'value': r1[stat],
                    'position': pos,
                    'player_id': player_id
                })
    else:
        # For batting/pitching: just group by player
        df = df.sort_values(['IDfg', 'Season'])
        
        for player_id, player_df in df.groupby('IDfg'):
            player_df = player_df.sort_values('Season')
            for i in range(len(player_df) - 1):
                r1, r2 = player_df.iloc[i], player_df.iloc[i + 1]
                if r2['Season'] - r1['Season'] != 1:
                    continue
                if pd.isna(r1[stat]) or pd.isna(r2[stat]):
                    continue
                results.append({
                    'age': r1['Age'],
                    'delta': r2[stat] - r1[stat],
                    'value': r1[stat]
                })
    
    return pd.DataFrame(results)


def compute_age_band_decline(deltas: pd.DataFrame, is_inverted: bool, 
                              retirement_rates: Dict = None) -> Dict:
    """
    Compute decline rates by age band with survivorship bias correction.
    
    For normal stats (higher=better): decline = negative delta
    For inverted stats (higher=worse): decline = positive delta
    
    If retirement_rates are provided, applies correction for the fact that
    poor performers are more likely to exit (making observed decline appear smaller).
    
    Returns dict with decline rates per age band.
    """
    results = {}
    
    for band_name, age_min, age_max in AGE_BANDS:
        band_data = deltas[(deltas['age'] >= age_min) & (deltas['age'] <= age_max)]
        
        if len(band_data) < 30:
            results[band_name] = {
                'decline_per_year': None,
                'decline_per_year_corrected': None,
                'std': None,
                'n_transitions': len(band_data),
                'note': 'insufficient data'
            }
            continue
        
        mean_delta = band_data['delta'].mean()
        std_delta = band_data['delta'].std()
        
        # For inverted stats, positive delta = decline (getting worse)
        # For normal stats, negative delta = decline
        if is_inverted:
            decline = mean_delta  # Positive delta means getting worse
        else:
            decline = -mean_delta  # Negative delta means getting worse
        
        # Calculate survivorship bias correction
        correction = 0.0
        if retirement_rates and band_name in retirement_rates:
            rates = retirement_rates[band_name]
            good_rate = rates.get('good_performers_continue_rate')
            poor_rate = rates.get('poor_performers_continue_rate')
            
            if good_rate is not None and poor_rate is not None and poor_rate > 0:
                # The correction factor represents how much we underestimate decline
                # because poor performers exit at higher rates than good performers.
                # 
                # If good performers continue at 80% and poor at 50%, then
                # for every 100 poor performers, only 50 show up in year 2,
                # while 80 good performers remain. This biases our sample toward
                # players who didn't decline as much.
                #
                # We estimate the "missing" decline from the performance gap
                # between those who stay vs leave, weighted by exit probability.
                survival_ratio = good_rate / poor_rate if poor_rate > 0 else 1.0
                
                # The bias is proportional to how much more likely good performers
                # are to survive. A ratio of 1.5 means good performers are 50% more
                # likely to continue, suggesting significant bias.
                if survival_ratio > 1.0:
                    # Estimate additional decline from the "missing" poor performers
                    # Use the std dev as a proxy for the performance spread
                    # The correction adds decline for the missing poor performers
                    exit_rate_gap = good_rate - poor_rate
                    # Scale correction by how many are "missing" and their likely decline
                    correction = exit_rate_gap * std_delta * 0.5  # 0.5 = ~0.67 std below mean
        
        decline_corrected = decline + correction
        
        results[band_name] = {
            'decline_per_year': float(decline),
            'decline_per_year_corrected': float(decline_corrected),
            'survivorship_correction': float(correction),
            'std': float(std_delta),
            'n_transitions': int(len(band_data)),
            'avg_starting_value': float(band_data['value'].mean())
        }
    
    # Extrapolate for age bands with insufficient data (36-40, 41-45)
    # Use the trend from earlier bands to project forward
    _extrapolate_older_bands(results)
    
    return results


def _extrapolate_older_bands(results: Dict) -> None:
    """
    Extrapolate decline rates for older age bands with insufficient data.
    Uses the acceleration pattern from younger bands.
    
    Modifies results dict in-place.
    """
    bands_to_check = ['36-40', '41-45']
    bands_with_data = ['21-25', '26-30', '31-35']
    
    # Check if we have data for the base bands
    valid_bands = [b for b in bands_with_data if b in results and results[b].get('decline_per_year_corrected') is not None]
    
    if len(valid_bands) < 2:
        # Not enough data to extrapolate
        return
    
    # Calculate the acceleration rate (how much decline increases per age band)
    declines = [results[b]['decline_per_year_corrected'] for b in valid_bands]
    
    # Fit a trend: are we accelerating linearly, or accelerating at an increasing rate?
    if len(declines) >= 3:
        # Use quadratic acceleration if we have 3+ points
        # Compute second derivative (acceleration of decline)
        d1 = declines[1] - declines[0]  # Change from band 1 to 2
        d2 = declines[2] - declines[1]  # Change from band 2 to 3
        acceleration = d2 - d1  # How much faster decline is increasing
    else:
        # Linear extrapolation
        d1 = declines[1] - declines[0]
        acceleration = 0  # No acceleration
    
    # Last known band and its decline
    last_band = valid_bands[-1]
    last_decline = results[last_band]['decline_per_year_corrected']
    last_std = results[last_band]['std']
    
    # Extrapolate forward
    current_decline = last_decline
    current_acceleration = d2 if len(declines) >= 3 else d1
    
    for band in bands_to_check:
        if band in results:
            # Skip if we already have real data
            if results[band].get('n_transitions', 0) >= 30:
                continue
            
            # Project next band's decline
            current_decline += current_acceleration
            current_acceleration += acceleration * 0.5  # Acceleration grows but dampened
            
            # Ensure decline is non-negative and reasonable (cap at 3x last std dev)
            current_decline = max(0, min(current_decline, last_std * 3))
            
            results[band] = {
                'decline_per_year': None,
                'decline_per_year_corrected': float(current_decline),
                'survivorship_correction': None,
                'std': float(last_std * 1.1),  # Slightly higher uncertainty
                'n_transitions': results[band].get('n_transitions', 0),
                'note': 'extrapolated from younger age bands',
                'extrapolated': True
            }


def analyze_retirement_bias(df: pd.DataFrame, stat: str, is_inverted: bool) -> Dict:
    """
    Analyze how performance affects likelihood of continuing to next year.
    This quantifies survivorship bias.
    """
    max_season = df['Season'].max()
    
    # Only look at seasons where we can check next year
    df_check = df[df['Season'] < max_season].copy()
    
    # Check if player appeared next year
    next_year_players = df[['IDfg', 'Season']].copy()
    next_year_players['Season'] = next_year_players['Season'] - 1
    next_year_players['played_next'] = True
    
    df_check = df_check.merge(
        next_year_players[['IDfg', 'Season', 'played_next']], 
        on=['IDfg', 'Season'], 
        how='left'
    )
    df_check['played_next'] = df_check['played_next'].fillna(False)
    
    # Analyze by age band and performance tier
    results = {}
    
    for band_name, age_min, age_max in AGE_BANDS:
        band_data = df_check[(df_check['Age'] >= age_min) & (df_check['Age'] <= age_max)]
        
        if len(band_data) < 50:
            continue
        
        # Split into performance tiers
        stat_values = band_data[stat].dropna()
        if len(stat_values) < 50:
            continue
            
        q33 = stat_values.quantile(0.33)
        q67 = stat_values.quantile(0.67)
        
        # For inverted stats, "good" is below median
        if is_inverted:
            good = band_data[band_data[stat] <= q33]
            avg = band_data[(band_data[stat] > q33) & (band_data[stat] <= q67)]
            poor = band_data[band_data[stat] > q67]
        else:
            good = band_data[band_data[stat] >= q67]
            avg = band_data[(band_data[stat] >= q33) & (band_data[stat] < q67)]
            poor = band_data[band_data[stat] < q33]
        
        results[band_name] = {
            'good_performers_continue_rate': float(good['played_next'].mean()) if len(good) > 10 else None,
            'avg_performers_continue_rate': float(avg['played_next'].mean()) if len(avg) > 10 else None,
            'poor_performers_continue_rate': float(poor['played_next'].mean()) if len(poor) > 10 else None,
            'n_good': int(len(good)),
            'n_avg': int(len(avg)),
            'n_poor': int(len(poor)),
        }
    
    return results


def analyze_stat(df: pd.DataFrame, stat: str, stat_config: Dict, group_by_position: bool = False) -> Optional[Dict]:
    """
    Analyze a single stat and return age-band decline rates.
    
    Args:
        df: DataFrame with player data
        stat: Stat column name to analyze
        stat_config: Dict with 'inverted' and 'description' keys
        group_by_position: If True, only compare same position year-over-year
                          (required for fielding data)
    """
    if stat not in df.columns:
        logger.warning(f"Stat '{stat}' not found")
        return None
    
    deltas = compute_deltas(df, stat, group_by_position=group_by_position)
    if len(deltas) < 100:
        logger.warning(f"Insufficient data for {stat} ({len(deltas)} transitions)")
        return None
    
    is_inverted = stat_config['inverted']
    
    # First compute retirement bias so we can use it for correction
    retirement_bias = analyze_retirement_bias(df, stat, is_inverted)
    
    # Pass retirement rates to compute corrected decline
    decline_by_band = compute_age_band_decline(deltas, is_inverted, retirement_rates=retirement_bias)
    
    return {
        'stat_name': stat,
        'description': stat_config['description'],
        'is_inverted': is_inverted,
        'decline_by_age_band': decline_by_band,
        'retirement_bias': retirement_bias,
        'total_transitions': int(len(deltas))
    }


def run_analysis(min_year: int) -> Dict:
    """Run full analysis on all model types."""
    results = {
        'metadata': {
            'min_year': min_year,
            'generated': pd.Timestamp.now().isoformat(),
            'age_bands': {b[0]: {'min': b[1], 'max': b[2]} for b in AGE_BANDS},
            'note': 'decline_per_year is the observed average annual performance loss (positive = getting worse). '
                    'decline_per_year_corrected adjusts for survivorship bias - poor performers exit at higher rates, '
                    'so observed decline underestimates true decline. Use the corrected value for projections. '
                    'Age bands 36-40 and 41-45 are extrapolated from younger bands due to insufficient data.'
        },
        'batter': {},
        'pitcher': {},
        'baserunning': {},
        'fielding_infield': {},
        'fielding_outfield': {},
        'fielding_catcher': {},
    }
    
    # Batters
    logger.info("\n" + "="*60)
    logger.info("BATTERS")
    logger.info("="*60)
    batting_df = load_data('batting', min_year)
    for stat, config in BATTER_STATS.items():
        analysis = analyze_stat(batting_df, stat, config)
        if analysis:
            results['batter'][stat] = analysis
            logger.info(f"  {stat}: analyzed")
    
    # Baserunning (from batting data, but needs Statcast era for sc_ columns)
    logger.info("\n" + "="*60)
    logger.info("BASERUNNING")
    logger.info("="*60)
    # For Statcast baserunning stats, filter to 2015+
    baserunning_df = batting_df[batting_df['Season'] >= 2015].copy()
    logger.info(f"  Using {len(baserunning_df):,} seasons (2015+ for Statcast)")
    for stat, config in BASERUNNING_STATS.items():
        analysis = analyze_stat(baserunning_df, stat, config)
        if analysis:
            results['baserunning'][stat] = analysis
            logger.info(f"  {stat}: analyzed")
    
    # Pitchers
    logger.info("\n" + "="*60)
    logger.info("PITCHERS")
    logger.info("="*60)
    pitching_df = load_data('pitching', min_year)
    for stat, config in PITCHER_STATS.items():
        analysis = analyze_stat(pitching_df, stat, config)
        if analysis:
            results['pitcher'][stat] = analysis
            logger.info(f"  {stat}: analyzed")
    
    # Fielding - Infield (2016+ for Statcast metrics)
    # IMPORTANT: group_by_position=True ensures we only compare same position year-over-year
    # (e.g., SS 2023 to SS 2024, not SS 2023 to 2B 2024)
    logger.info("\n" + "="*60)
    logger.info("FIELDING - INFIELD (1B, 2B, 3B, SS)")
    logger.info("="*60)
    infield_positions = ['1B', '2B', '3B', 'SS']
    infield_df = load_data('fielding', max(min_year, 2016), position_filter=infield_positions)
    for stat, config in FIELDING_STATS_INFIELD.items():
        analysis = analyze_stat(infield_df, stat, config, group_by_position=True)
        if analysis:
            results['fielding_infield'][stat] = analysis
            logger.info(f"  {stat}: analyzed ({analysis['total_transitions']} same-position transitions)")
    
    # Fielding - Outfield (2016+ for Statcast metrics)
    logger.info("\n" + "="*60)
    logger.info("FIELDING - OUTFIELD (LF, CF, RF)")
    logger.info("="*60)
    outfield_positions = ['LF', 'CF', 'RF']
    outfield_df = load_data('fielding', max(min_year, 2016), position_filter=outfield_positions)
    for stat, config in FIELDING_STATS_OUTFIELD.items():
        analysis = analyze_stat(outfield_df, stat, config, group_by_position=True)
        if analysis:
            results['fielding_outfield'][stat] = analysis
            logger.info(f"  {stat}: analyzed ({analysis['total_transitions']} same-position transitions)")
    
    # Fielding - Catcher (2016+ for Statcast metrics)
    logger.info("\n" + "="*60)
    logger.info("FIELDING - CATCHER")
    logger.info("="*60)
    catcher_df = load_data('fielding', max(min_year, 2016), position_filter=['C'])
    for stat, config in FIELDING_STATS_CATCHER.items():
        analysis = analyze_stat(catcher_df, stat, config, group_by_position=True)
        if analysis:
            results['fielding_catcher'][stat] = analysis
            logger.info(f"  {stat}: analyzed ({analysis['total_transitions']} same-position transitions)")
    
    return results


def print_summary(results: Dict):
    """Print a readable summary table."""
    print("\n" + "="*85)
    print("EMPIRICAL DECLINE RATES BY AGE BAND")
    print("="*85)
    print(f"{'Model':<18} {'Stat':<30} {'21-25':>8} {'26-30':>8} {'31-35':>8} {'36-40':>8}")
    print("-"*85)
    
    model_types = ['batter', 'pitcher', 'baserunning', 
                   'fielding_infield', 'fielding_outfield', 'fielding_catcher']
    
    for model_type in model_types:
        for stat, data in results.get(model_type, {}).items():
            bands = data['decline_by_age_band']
            values = []
            for band_name in ['21-25', '26-30', '31-35', '36-40']:
                if band_name in bands and bands[band_name]['decline_per_year'] is not None:
                    val = bands[band_name]['decline_per_year']
                    values.append(f"{val:+.3f}")
                else:
                    values.append("N/A")
            
            # Shorten model name for display
            display_model = model_type.replace('fielding_', 'fld_')
            print(f"{display_model:<18} {stat:<30} {values[0]:>8} {values[1]:>8} {values[2]:>8} {values[3]:>8}")
        
        if results.get(model_type):
            print()
    
    print("="*85)
    print("Note: Positive values = decline (performance getting worse)")
    print("      For inverted stats (ERA, K% for batters), higher raw numbers = worse")
    print("="*85)
    
    # Print retirement bias summary
    print("\n" + "="*85)
    print("SURVIVORSHIP BIAS: Continuation Rates by Performance Tier")
    print("="*85)
    print(f"{'Model':<18} {'Stat':<20} {'Band':<8} {'Good%':>8} {'Avg%':>8} {'Poor%':>8}")
    print("-"*85)
    
    # Show key stats for survivorship analysis
    bias_stats = [
        ('batter', 'wRC+'),
        ('pitcher', 'ERA'),
        ('baserunning', 'sc_sprint_speed'),
        ('fielding_infield', 'sc_total_runs/150'),
        ('fielding_outfield', 'sc_total_runs/150'),
        ('fielding_catcher', 'sc_total_runs/150'),
    ]
    
    for model_type, stat in bias_stats:
        if stat not in results.get(model_type, {}):
            continue
        
        retirement = results[model_type][stat].get('retirement_bias', {})
        for band in ['26-30', '31-35', '36-40']:
            if band not in retirement:
                continue
            r = retirement[band]
            good = f"{r['good_performers_continue_rate']*100:.0f}%" if r['good_performers_continue_rate'] else "N/A"
            avg = f"{r['avg_performers_continue_rate']*100:.0f}%" if r['avg_performers_continue_rate'] else "N/A"
            poor = f"{r['poor_performers_continue_rate']*100:.0f}%" if r['poor_performers_continue_rate'] else "N/A"
            display_model = model_type.replace('fielding_', 'fld_')
            print(f"{display_model:<18} {stat:<20} {band:<8} {good:>8} {avg:>8} {poor:>8}")
    
    print("="*85)


def main():
    parser = argparse.ArgumentParser(description='Derive empirical aging curves v2')
    parser.add_argument('--min-year', type=int, default=2000)
    parser.add_argument('--output', type=str, default='aging_parameters.json')
    args = parser.parse_args()
    
    results = run_analysis(args.min_year)
    
    # Save
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to: {output_path}")
    
    # Print summary
    print_summary(results)
    
    return results


if __name__ == '__main__':
    main()
