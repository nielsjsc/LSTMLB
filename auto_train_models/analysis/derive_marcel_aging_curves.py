"""
Derive aging curves for fielding and baserunning stats using the delta method.

For each stat, we compute the average year-over-year change at each age
using paired seasons from the same player at the same position (for fielding).
This is the standard "delta method" used in sabermetrics.

Outputs a JSON file with per-stat aging curves.
"""
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, '.')

from core.data_processing import calculate_rate_stats


def derive_fielding_aging_curves(hist_df: pd.DataFrame, min_inn: float = 200) -> dict:
    """
    Derive aging curves for fielding stats using the delta method.
    
    For each position group, compute the average year-over-year change
    at each age for each stat, using paired seasons from the same player
    at the same position.
    
    Args:
        hist_df: Historical fielding data with rate stats already computed
        min_inn: Minimum innings in BOTH seasons for a pair to qualify
        
    Returns:
        Dict of {position_group: {stat: {age: delta}}}
    """
    # Position groups
    groups = {
        'outfield': ['LF', 'CF', 'RF'],
        'infield': ['1B', '2B', '3B', 'SS'],
        'catcher': ['C'],
    }
    
    # Stats per group
    stats_map = {
        'outfield': ['sc_total_runs/150', 'sc_range_runs/150', 'sc_arm_runs/150'],
        'infield': ['sc_total_runs/150', 'sc_range_runs/150', 'sc_arm_runs/150', 'sc_dp_runs/150'],
        'catcher': ['sc_total_runs/150', 'sc_framing_runs/150', 'sc_throwing_runs/150', 'sc_blocking_runs/150'],
    }
    
    results = {}
    
    for group_name, positions in groups.items():
        group_df = hist_df[
            (hist_df['Pos'].isin(positions)) & 
            (hist_df['Inn'] >= min_inn)
        ].copy()
        
        # Drop rows with NaN in key stats
        group_stats = stats_map[group_name]
        group_df = group_df.dropna(subset=group_stats + ['Age'])
        
        # For fielding, we aggregate across positions within a group
        # (e.g., all OF positions together) since players move between LF/CF/RF
        # Group by player-season, taking the innings-weighted average
        player_seasons = group_df.groupby(['IDfg', 'Season']).apply(
            lambda g: pd.Series({
                'Age': g['Age'].iloc[0],
                'Inn': g['Inn'].sum(),
                **{stat: np.average(g[stat].values, weights=g['Inn'].values) 
                   for stat in group_stats}
            })
        ).reset_index()
        
        group_results = {}
        for stat in group_stats:
            # Create paired seasons: same player, consecutive years
            df_sorted = player_seasons.sort_values(['IDfg', 'Season'])
            
            pairs = []
            for pid, player_data in df_sorted.groupby('IDfg'):
                player_data = player_data.sort_values('Season')
                seasons = player_data['Season'].values
                for i in range(len(seasons) - 1):
                    if seasons[i+1] - seasons[i] == 1:  # consecutive
                        age_in_later_year = player_data.iloc[i+1]['Age']
                        val_before = player_data.iloc[i][stat]
                        val_after = player_data.iloc[i+1][stat]
                        inn_before = player_data.iloc[i]['Inn']
                        inn_after = player_data.iloc[i+1]['Inn']
                        
                        pairs.append({
                            'age': int(age_in_later_year),
                            'delta': val_after - val_before,
                            'weight': min(inn_before, inn_after),  # weight by smaller innings
                        })
            
            pairs_df = pd.DataFrame(pairs)
            if pairs_df.empty:
                group_results[stat] = {}
                continue
            
            # Compute weighted average delta at each age
            age_deltas = {}
            for age in range(20, 45):
                age_pairs = pairs_df[pairs_df['age'] == age]
                if len(age_pairs) >= 10:  # Need enough pairs for stability
                    weighted_delta = np.average(
                        age_pairs['delta'].values,
                        weights=age_pairs['weight'].values
                    )
                    age_deltas[str(age)] = round(float(weighted_delta), 4)
            
            group_results[stat] = age_deltas
        
        results[group_name] = group_results
        
        print(f"\n=== {group_name.upper()} ===")
        for stat, deltas in group_results.items():
            print(f"\n  {stat}:")
            for age in sorted(deltas.keys(), key=int):
                print(f"    Age {age}: {deltas[age]:+.4f} runs/150")
    
    return results


def derive_baserunning_aging_curves(batting_df: pd.DataFrame, min_games: int = 50) -> dict:
    """
    Derive aging curves for baserunning stats using the delta method.
    
    Args:
        batting_df: Historical batting data with rate stats computed
        min_games: Minimum games in BOTH seasons for a pair to qualify
        
    Returns:
        Dict of {stat: {age: delta}}
    """
    stats = ['sc_baserunning_runner_runs_tot_rate', 'SB_rate', 'CS_rate']
    
    # Filter to valid rows
    df = batting_df[batting_df['G'] >= min_games].copy()
    df = df.dropna(subset=stats + ['Age'])
    
    # Group by player-season (should already be unique per player-season in batting data)
    df_sorted = df.sort_values(['IDfg', 'Season'])
    
    results = {}
    for stat in stats:
        pairs = []
        for pid, player_data in df_sorted.groupby('IDfg'):
            player_data = player_data.sort_values('Season')
            seasons = player_data['Season'].values
            for i in range(len(seasons) - 1):
                if seasons[i+1] - seasons[i] == 1:
                    age_in_later_year = player_data.iloc[i+1]['Age']
                    val_before = player_data.iloc[i][stat]
                    val_after = player_data.iloc[i+1][stat]
                    games_before = player_data.iloc[i]['G']
                    games_after = player_data.iloc[i+1]['G']
                    
                    pairs.append({
                        'age': int(age_in_later_year),
                        'delta': val_after - val_before,
                        'weight': min(games_before, games_after),
                    })
        
        pairs_df = pd.DataFrame(pairs)
        if pairs_df.empty:
            results[stat] = {}
            continue
        
        age_deltas = {}
        for age in range(20, 45):
            age_pairs = pairs_df[pairs_df['age'] == age]
            if len(age_pairs) >= 15:  # Need enough pairs
                weighted_delta = np.average(
                    age_pairs['delta'].values,
                    weights=age_pairs['weight'].values
                )
                age_deltas[str(age)] = round(float(weighted_delta), 4)
        
        results[stat] = age_deltas
    
    print(f"\n=== BASERUNNING ===")
    for stat, deltas in results.items():
        print(f"\n  {stat}:")
        for age in sorted(deltas.keys(), key=int):
            print(f"    Age {age}: {deltas[age]:+.4f} per 150G")
    
    return results


def derive_batting_aging_curves(batting_df: pd.DataFrame, min_pa: int = 200) -> dict:
    """
    Derive aging curves for batting stats using the delta method.

    Computes weighted average year-over-year deltas at each age for the
    rate stats that drive batter Marcel projections.

    Args:
        batting_df: Historical batting data (output of calculate_rate_stats)
        min_pa: Minimum PA in BOTH seasons for a pair to qualify

    Returns:
        Dict of {stat: {age_str: delta}}
    """
    # Rate stats only — counting stats are derived from wOBA in the projection
    stats = ['BB%', 'K%', 'AVG', 'OBP', 'SLG', 'wOBA']

    df = batting_df[(batting_df['PA'] >= min_pa)].copy()
    df = df.dropna(subset=stats + ['Age'])

    df_sorted = df.sort_values(['IDfg', 'Season'])

    results = {}
    for stat in stats:
        pairs = []
        for pid, player_data in df_sorted.groupby('IDfg'):
            player_data = player_data.sort_values('Season')
            seasons = player_data['Season'].values
            for i in range(len(seasons) - 1):
                if seasons[i + 1] - seasons[i] == 1:
                    age_in_later_year = player_data.iloc[i + 1]['Age']
                    val_before = player_data.iloc[i][stat]
                    val_after = player_data.iloc[i + 1][stat]
                    pa_before = player_data.iloc[i]['PA']
                    pa_after = player_data.iloc[i + 1]['PA']

                    if pd.notna(val_before) and pd.notna(val_after):
                        pairs.append({
                            'age': int(age_in_later_year),
                            'delta': val_after - val_before,
                            'weight': min(pa_before, pa_after),
                        })

        pairs_df = pd.DataFrame(pairs)
        if pairs_df.empty:
            results[stat] = {}
            continue

        age_deltas = {}
        for age in range(20, 45):
            age_pairs = pairs_df[pairs_df['age'] == age]
            if len(age_pairs) >= 30:
                weighted_delta = np.average(
                    age_pairs['delta'].values,
                    weights=age_pairs['weight'].values
                )
                age_deltas[str(age)] = round(float(weighted_delta), 6)

        results[stat] = age_deltas

    print(f"\n=== BATTING ===")
    for stat, deltas in results.items():
        print(f"\n  {stat}:")
        for age in sorted(deltas.keys(), key=int):
            print(f"    Age {age}: {deltas[age]:+.6f}")

    return results


def derive_pitching_aging_curves(pitching_df: pd.DataFrame, min_ip: float = 50) -> dict:
    """
    Derive aging curves for pitching stats using the delta method.

    Computes weighted average year-over-year deltas at each age for the
    component rate stats that drive pitcher Marcel projections.  Composite
    stats like FIP and ERA are reconstructed from components, so they
    don't need their own aging curves.

    Args:
        pitching_df: Historical pitching data (output of calculate_rate_stats)
        min_ip: Minimum IP in BOTH seasons for a pair to qualify

    Returns:
        Dict of {stat: {age_str: delta}}
    """
    stats = ['K%', 'BB%', 'HBP%', 'BABIP', 'HR/FB', 'GB%', 'FB%', 'LD%']

    df = pitching_df[(pitching_df['IP'] >= min_ip)].copy()
    df = df.dropna(subset=['Age', 'K%', 'BB%'])

    df_sorted = df.sort_values(['IDfg', 'Season'])

    results = {}
    for stat in stats:
        stat_df = df_sorted.dropna(subset=[stat])
        pairs = []
        for pid, player_data in stat_df.groupby('IDfg'):
            player_data = player_data.sort_values('Season')
            seasons = player_data['Season'].values
            for i in range(len(seasons) - 1):
                if seasons[i + 1] - seasons[i] == 1:
                    age_in_later_year = player_data.iloc[i + 1]['Age']
                    val_before = player_data.iloc[i][stat]
                    val_after = player_data.iloc[i + 1][stat]
                    ip_before = player_data.iloc[i]['IP']
                    ip_after = player_data.iloc[i + 1]['IP']

                    if pd.notna(val_before) and pd.notna(val_after):
                        pairs.append({
                            'age': int(age_in_later_year),
                            'delta': val_after - val_before,
                            'weight': min(ip_before, ip_after),
                        })

        pairs_df = pd.DataFrame(pairs)
        if pairs_df.empty:
            results[stat] = {}
            continue

        age_deltas = {}
        for age in range(20, 45):
            age_pairs = pairs_df[pairs_df['age'] == age]
            if len(age_pairs) >= 20:
                weighted_delta = np.average(
                    age_pairs['delta'].values,
                    weights=age_pairs['weight'].values
                )
                age_deltas[str(age)] = round(float(weighted_delta), 6)

        results[stat] = age_deltas

    print(f"\n=== PITCHING ===")
    for stat, deltas in results.items():
        print(f"\n  {stat}:")
        for age in sorted(deltas.keys(), key=int):
            print(f"    Age {age}: {deltas[age]:+.6f}")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("DERIVING FIELDING AGING CURVES")
    print("=" * 60)
    
    fielding_df = pd.read_csv('../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv')
    fielding_df = calculate_rate_stats(fielding_df)
    fielding_curves = derive_fielding_aging_curves(fielding_df, min_inn=200)
    
    print("\n" + "=" * 60)
    print("DERIVING BASERUNNING AGING CURVES")
    print("=" * 60)
    
    batting_df = pd.read_csv('../data/historic_mlb/mlb_batting_data_1950_2025_with_statcast.csv')
    batting_df = calculate_rate_stats(batting_df)
    # Only use statcast era for baserunning (sc_ columns)
    batting_statcast = batting_df[batting_df['Season'] >= 2016].copy()
    baserunning_curves = derive_baserunning_aging_curves(batting_statcast, min_games=50)

    print("\n" + "=" * 60)
    print("DERIVING BATTING AGING CURVES")
    print("=" * 60)

    batting_curves = derive_batting_aging_curves(batting_df, min_pa=200)

    print("\n" + "=" * 60)
    print("DERIVING PITCHING AGING CURVES")
    print("=" * 60)

    pitching_df = pd.read_csv('../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv')
    pitching_df = calculate_rate_stats(pitching_df)
    pitching_curves = derive_pitching_aging_curves(pitching_df, min_ip=50)
    
    # Combine into one output
    all_curves = {
        'fielding': fielding_curves,
        'baserunning': baserunning_curves,
        'batting': batting_curves,
        'pitching': pitching_curves,
        'metadata': {
            'method': 'delta_method',
            'fielding_min_innings': 200,
            'baserunning_min_games': 50,
            'batting_min_pa': 200,
            'pitching_min_ip': 50,
            'fielding_min_pairs_per_age': 10,
            'baserunning_min_pairs_per_age': 15,
            'batting_min_pairs_per_age': 30,
            'pitching_min_pairs_per_age': 20,
            'fielding_data_source': 'mlb_fielding_data_2000_2025_with_statcast.csv',
            'baserunning_data_source': 'mlb_batting_data_1950_2025_with_statcast.csv (2016+)',
            'batting_data_source': 'mlb_batting_data_1950_2025_with_statcast.csv',
            'pitching_data_source': 'mlb_pitching_data_1950_2025_with_statcast.csv',
        }
    }
    
    output_path = 'analysis/marcel_aging_curves.json'
    with open(output_path, 'w') as f:
        json.dump(all_curves, f, indent=2)
    
    print(f"\n\nSaved aging curves to {output_path}")
