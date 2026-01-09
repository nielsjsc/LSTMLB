"""
Data Joiner Script - Joins Statcast data to historical MLB data files
Uses player_id_mapping.csv to match FanGraphs IDs (IDfg) with MLB/Statcast IDs (mlbam_id)
Falls back to name-based matching for unmapped players
Adds mlbam_id column to all datasets for cross-referencing
Avoids duplicate columns by only adding new statcast metrics

This script replaces moveage.py by automatically adding Age to fielding data
from the batting data file during the join process.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import glob
import unicodedata

# Paths
DATA_DIR = Path(__file__).parent.parent
HISTORIC_DIR = DATA_DIR / 'historic_mlb'
STATCAST_DIR = DATA_DIR / 'statcast'
ACTIVE_ROSTER_DIR = DATA_DIR / 'active_roster'

# ID mapping file
ID_MAPPING_FILE = ACTIVE_ROSTER_DIR / 'player_id_mapping.csv'

# Historical data files
BATTING_FILE = HISTORIC_DIR / 'mlb_batting_data_1950_2025.csv'
PITCHING_FILE = HISTORIC_DIR / 'mlb_pitching_data_1950_2025.csv'
FIELDING_FILE = HISTORIC_DIR / 'mlb_fielding_data_2000_2025.csv'

# Statcast files - Batters
# Note: percentile ranks excluded - only available for qualified hitters
STATCAST_BATTER_FILES = {
    'exitvelo': STATCAST_DIR / 'statcast_batter_exitvelo_barrels_2015_2025.csv',
    'expected': STATCAST_DIR / 'statcast_batter_expected_stats_2015_2025.csv',
    'sprint': STATCAST_DIR / 'statcast_sprint_speed_2015_2025.csv',
    'baserunning': STATCAST_DIR / 'statcast_baserunning_run_value.csv'
}

# Statcast files - Pitchers
# Note: percentile ranks excluded - only available for qualified pitchers
STATCAST_PITCHER_FILES = {
    'arsenal_speed': STATCAST_DIR / 'statcast_pitcher_arsenal_speed_2015_2025.csv',
    'arsenal_spin': STATCAST_DIR / 'statcast_pitcher_arsenal_spin_2015_2025.csv',
    'exitvelo': STATCAST_DIR / 'statcast_pitcher_exitvelo_barrels_2015_2025.csv',
    'expected': STATCAST_DIR / 'statcast_pitcher_expected_stats_2015_2025.csv'
}

# Statcast files - Fielders
STATCAST_FIELDER_FILES = {
    'fielding': STATCAST_DIR / 'statcast_fielding_run_value_2015_2025.csv'
}

def load_id_mapping():
    """Load player ID mapping (FanGraphs ID <-> MLB/Statcast ID)"""
    print("Loading player ID mapping from player_id_mapping.csv...")
    
    # Load the ID mapping file
    id_df = pd.read_csv(ID_MAPPING_FILE)
    
    # Create lookup dictionary: mlbam_id -> fg_id
    # Filter out rows where either ID is missing OR fg_id is -1 (invalid)
    valid_mappings = id_df.dropna(subset=['mlbam_id', 'fg_id'])
    valid_mappings = valid_mappings[valid_mappings['fg_id'] != -1]
    valid_mappings['mlbam_id'] = valid_mappings['mlbam_id'].astype(int)
    valid_mappings['fg_id'] = valid_mappings['fg_id'].astype(int)
    
    id_dict = dict(zip(valid_mappings['mlbam_id'], valid_mappings['fg_id']))
    print(f"  Loaded {len(id_dict):,} player ID mappings")
    return id_dict

def add_name_based_mapping(statcast_df, historical_df, id_col='IDfg', name_col='Name'):
    """
    Add name-based fallback mapping for players without ID mapping.
    Handles name format differences robustly.
    
    Args:
        statcast_df: Statcast dataframe with 'player_name' and id_col columns
        historical_df: Historical dataframe with id_col and name_col columns
        id_col: Name of the ID column (default 'IDfg')
        name_col: Name of the name column (default 'Name')
    
    Returns:
        Number of new mappings added
    """
    # Find unmapped players
    unmapped = statcast_df[statcast_df[id_col].isna()].copy()
    if len(unmapped) == 0:
        return 0
    
    # Check if player_name column exists
    if 'player_name' not in statcast_df.columns:
        print(f"  WARNING: 'player_name' column not found in statcast data, skipping name-based matching")
        print(f"  Available columns: {statcast_df.columns.tolist()}")
        return 0
    
    print(f"  Attempting name-based matching for {len(unmapped):,} unmapped players...")
    
    def normalize_name(name):
        """Normalize name: 'Last, First' -> 'first last', strip accents"""
        if pd.isna(name):
            return ''
        name_str = str(name).strip()
        # Split on comma: "Holliday, Jackson" -> ["Holliday", " Jackson"]
        if ',' in name_str:
            parts = [p.strip() for p in name_str.split(',')]
            if len(parts) >= 2:
                # Reverse to "Jackson Holliday"
                name_str = f"{parts[1]} {parts[0]}"
        
        # Remove accents: "Narváez" -> "Narvaez"
        name_str = name_str.lower()
        name_str = unicodedata.normalize('NFD', name_str)
        name_str = ''.join(char for char in name_str if unicodedata.category(char) != 'Mn')
        return name_str
    
    # Normalize statcast names
    unmapped['clean_name'] = unmapped['player_name'].apply(normalize_name)
    
    # Build historical name lookup (with accent removal)
    hist_names = historical_df[[id_col, name_col]].drop_duplicates()
    hist_names['clean_name'] = hist_names[name_col].apply(lambda x: normalize_name(x) if pd.notna(x) else '')
    name_to_id = dict(zip(hist_names['clean_name'], hist_names[id_col]))
    
    # Build historical name lookup
    hist_names = historical_df[[id_col, name_col]].drop_duplicates()
    hist_names['clean_name'] = hist_names[name_col].str.strip().str.lower()
    name_to_id = dict(zip(hist_names['clean_name'], hist_names[id_col]))
    
    # Match by name
    unmapped['mapped_id'] = unmapped['clean_name'].map(name_to_id)
    
    # Update original dataframe
    matched_count = 0
    for idx in unmapped.index:
        mapped_id = unmapped.at[idx, 'mapped_id']
        if pd.notna(mapped_id):
            statcast_df.at[idx, id_col] = mapped_id
            matched_count += 1
    
    if matched_count > 0:
        print(f"  Name-based matching added {matched_count:,} more mappings")
    
    return matched_count

def identify_duplicate_columns(historical_cols, statcast_cols, exclude_keys=['player_id', 'year', 'pa', 'bip']):
    """
    Identify columns that exist in both historical and statcast data
    Returns: set of columns to exclude from statcast data
    """
    # Normalize column names for comparison
    hist_lower = {col.lower() for col in historical_cols}
    statcast_lower = {col.lower(): col for col in statcast_cols}
    
    duplicates = set()
    
    for sc_col_lower, sc_col_original in statcast_lower.items():
        # Skip key columns used for joining
        if sc_col_lower in [k.lower() for k in exclude_keys]:
            continue
            
        # Check for exact matches
        if sc_col_lower in hist_lower:
            duplicates.add(sc_col_original)
            continue
        
        # Check for similar columns (common duplicates)
        # AVG vs ba, SLG vs slg, wOBA vs woba, etc.
        similar_mappings = {
            'ba': ['avg'],
            'slg': ['slg'],
            'woba': ['woba'],
            'era': ['era']
        }
        
        for sc_key, hist_keys in similar_mappings.items():
            if sc_col_lower == sc_key:
                if any(h in hist_lower for h in hist_keys):
                    duplicates.add(sc_col_original)
    
    return duplicates

def load_and_merge_statcast_files(file_dict, id_column='player_id'):
    """
    Load multiple statcast files and merge them on player_id and year
    Returns: merged dataframe with all statcast metrics
    """
    dfs = []
    
    for name, filepath in file_dict.items():
        if not filepath.exists():
            print(f"  WARNING: {filepath.name} not found, skipping...")
            continue
            
        df = pd.read_csv(filepath)
        
        # Standardize column names
        # Handle "last_name, first_name" vs player_name vs name
        if 'last_name, first_name' in df.columns:
            df = df.rename(columns={'last_name, first_name': 'player_name'})
        elif '"last_name, first_name"' in df.columns:
            df = df.rename(columns={'"last_name, first_name"': 'player_name'})
        elif 'name' in df.columns:
            df = df.rename(columns={'name': 'player_name'})
        elif 'entity_name' in df.columns:
            df = df.rename(columns={'entity_name': 'player_name'})
        
        # Handle pitcher vs player_id
        if 'pitcher' in df.columns:
            df = df.rename(columns={'pitcher': 'player_id'})
        elif 'id' in df.columns:
            df = df.rename(columns={'id': 'player_id'})
        
        # Handle start_year/end_year (baserunning file uses these)
        if 'start_year' in df.columns and 'year' not in df.columns:
            df = df.rename(columns={'start_year': 'year'})
            # Drop end_year as we only need one year column
            if 'end_year' in df.columns:
                df = df.drop(columns=['end_year'])
        
        # Add source prefix to avoid column collisions during merge
        rename_dict = {}
        for col in df.columns:
            if col not in ['player_id', 'year', 'player_name', 'position']:
                rename_dict[col] = f"{name}_{col}"
        
        df = df.rename(columns=rename_dict)
        dfs.append(df)
        print(f"  Loaded {name}: {len(df):,} rows, {len(df.columns)} columns")
    
    if not dfs:
        return None
    
    # Merge all dataframes
    merged = dfs[0]
    for df in dfs[1:]:
        # Merge on player_id, year, and player_name to preserve names
        merge_cols = ['player_id', 'year']
        if 'player_name' in merged.columns and 'player_name' in df.columns:
            merge_cols.append('player_name')
        
        merged = merged.merge(df, on=merge_cols, how='outer', suffixes=('', '_dup'))
        
        # Remove duplicate columns from merge
        dup_cols = [col for col in merged.columns if col.endswith('_dup')]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)
    
    # Keep player_name for name-based matching (will be dropped later)
    
    return merged

def remove_source_prefixes(df):
    """
    Remove source prefixes from column names (e.g., 'exitvelo_barrels' -> 'barrels')
    But keep descriptive prefixes for clarity
    """
    rename_dict = {}
    for col in df.columns:
        # Remove prefixes but keep semantic meaning
        new_col = col
        
        # Remove source file prefixes
        prefixes = ['exitvelo_', 'expected_', 'percentile_', 'sprint_', 
                   'arsenal_speed_', 'arsenal_spin_', 'fielding_']
        
        for prefix in prefixes:
            if col.startswith(prefix):
                new_col = col.replace(prefix, '', 1)
                break
        
        # Add 'sc_' prefix to all statcast columns to distinguish them
        if new_col not in ['IDfg', 'Season', 'player_id', 'year', 'Pos'] and not new_col.startswith('sc_'):
            new_col = f'sc_{new_col}'
        
        rename_dict[col] = new_col
    
    return df.rename(columns=rename_dict)

def join_statcast_to_batting(id_mapping):
    """Join statcast batter data to historical batting data"""
    print("\n" + "="*80)
    print("JOINING STATCAST BATTER DATA")
    print("="*80)
    
    # Load historical batting data
    print("\nLoading historical batting data...")
    batting = pd.read_csv(BATTING_FILE)
    print(f"  Loaded {len(batting):,} rows, {len(batting.columns)} columns")
    original_cols = set(batting.columns)
    
    # Load and merge all statcast batter files
    print("\nLoading statcast batter files...")
    statcast = load_and_merge_statcast_files(STATCAST_BATTER_FILES)
    
    if statcast is None:
        print("  No statcast batter data found!")
        return
    
    print(f"\nMerged statcast data: {len(statcast):,} rows, {len(statcast.columns)} columns")
    
    # Map statcast player_id to FanGraphs IDfg
    print("\nMapping player IDs...")
    statcast['IDfg'] = statcast['player_id'].map(id_mapping)
    
    # Keep mlbam_id for reference
    statcast['mlbam_id'] = statcast['player_id']
    
    # Remove rows where mapping failed
    before_filter = len(statcast)
    statcast = statcast.dropna(subset=['IDfg'])
    statcast['IDfg'] = statcast['IDfg'].astype(int)
    print(f"  Mapped {len(statcast):,} / {before_filter:,} rows successfully ({len(statcast)/before_filter*100:.1f}%)")
    
    # Drop player_name as it's no longer needed
    if 'player_name' in statcast.columns:
        statcast = statcast.drop(columns=['player_name'])
    
    # Rename year to Season for matching
    statcast = statcast.rename(columns={'year': 'Season'})
    
    # Identify and remove duplicate columns (but keep join keys)
    print("\nIdentifying duplicate columns...")
    duplicates = identify_duplicate_columns(batting.columns, statcast.columns)
    # Don't remove join keys
    join_keys = {'IDfg', 'Season'}
    duplicates = duplicates - join_keys
    if duplicates:
        print(f"  Removing {len(duplicates)} duplicate columns: {sorted(duplicates)}")
        statcast = statcast.drop(columns=list(duplicates))
    else:
        print("  No duplicate columns found")
    
    # Remove source prefixes and add 'sc_' prefix
    statcast = remove_source_prefixes(statcast)
    
    # Drop player_id as we use IDfg
    if 'player_id' in statcast.columns:
        statcast = statcast.drop(columns=['player_id'])
    
    # Merge with historical data
    print("\nMerging with historical batting data...")
    print(f"  Before merge: {len(batting):,} rows")
    batting_merged = batting.merge(
        statcast, 
        on=['IDfg', 'Season'], 
        how='left',
        suffixes=('', '_statcast')
    )
    print(f"  After merge: {len(batting_merged):,} rows")
    
    # Count new columns added
    new_cols = sorted(set(batting_merged.columns) - original_cols)
    print(f"\n  Added {len(new_cols)} new statcast columns:")
    display_cols = new_cols[:20] if len(new_cols) > 20 else new_cols
    for col in display_cols:
        try:
            non_null = batting_merged[col].notna().sum()
            pct = (non_null / len(batting_merged)) * 100
            print(f"    {col}: {non_null:,} values ({pct:.1f}%)")
        except:
            pass
    if len(new_cols) > 20:
        print(f"    ... and {len(new_cols) - 20} more columns")
    
    # Save merged data
    output_file = HISTORIC_DIR / 'mlb_batting_data_1950_2025_with_statcast.csv'
    print(f"\nSaving to {output_file.name}...")
    batting_merged.to_csv(output_file, index=False)
    print(f"  Saved {len(batting_merged):,} rows, {len(batting_merged.columns)} columns")

def join_statcast_to_pitching(id_mapping):
    """Join statcast pitcher data to historical pitching data"""
    print("\n" + "="*80)
    print("JOINING STATCAST PITCHER DATA")
    print("="*80)
    
    # Load historical pitching data
    print("\nLoading historical pitching data...")
    pitching = pd.read_csv(PITCHING_FILE)
    print(f"  Loaded {len(pitching):,} rows, {len(pitching.columns)} columns")
    original_cols = set(pitching.columns)
    
    # Load and merge all statcast pitcher files
    print("\nLoading statcast pitcher files...")
    statcast = load_and_merge_statcast_files(STATCAST_PITCHER_FILES)
    
    if statcast is None:
        print("  No statcast pitcher data found!")
        return
    
    print(f"\nMerged statcast data: {len(statcast):,} rows, {len(statcast.columns)} columns")
    
    # Map statcast player_id to FanGraphs IDfg
    print("\nMapping player IDs...")
    statcast['IDfg'] = statcast['player_id'].map(id_mapping)
    
    # Try name-based fallback for unmapped players
    add_name_based_mapping(statcast, pitching, id_col='IDfg', name_col='Name')
    
    # Keep mlbam_id for reference
    statcast['mlbam_id'] = statcast['player_id']
    
    # Remove rows where mapping failed
    before_filter = len(statcast)
    statcast = statcast.dropna(subset=['IDfg'])
    statcast['IDfg'] = statcast['IDfg'].astype(int)
    print(f"  Mapped {len(statcast):,} / {before_filter:,} rows successfully ({len(statcast)/before_filter*100:.1f}%)")
    
    # Drop player_name as it's no longer needed
    if 'player_name' in statcast.columns:
        statcast = statcast.drop(columns=['player_name'])
    
    # Rename year to Season for matching
    statcast = statcast.rename(columns={'year': 'Season'})
    
    # Identify and remove duplicate columns (but keep join keys)
    print("\nIdentifying duplicate columns...")
    duplicates = identify_duplicate_columns(pitching.columns, statcast.columns)
    # Don't remove join keys
    join_keys = {'IDfg', 'Season'}
    duplicates = duplicates - join_keys
    if duplicates:
        print(f"  Removing {len(duplicates)} duplicate columns: {sorted(duplicates)}")
        statcast = statcast.drop(columns=list(duplicates))
    else:
        print("  No duplicate columns found")
    
    # Remove source prefixes and add 'sc_' prefix
    statcast = remove_source_prefixes(statcast)
    
    # Drop player_id as we use IDfg
    if 'player_id' in statcast.columns:
        statcast = statcast.drop(columns=['player_id'])
    
    # Merge with historical data
    print("\nMerging with historical pitching data...")
    print(f"  Before merge: {len(pitching):,} rows")
    pitching_merged = pitching.merge(
        statcast,
        on=['IDfg', 'Season'],
        how='left',
        suffixes=('', '_statcast')
    )
    print(f"  After merge: {len(pitching_merged):,} rows")
    
    # Count new columns added
    new_cols = sorted(set(pitching_merged.columns) - original_cols)
    print(f"\n  Added {len(new_cols)} new statcast columns:")
    display_cols = new_cols[:20] if len(new_cols) > 20 else new_cols
    for col in display_cols:
        try:
            non_null = pitching_merged[col].notna().sum()
            pct = (non_null / len(pitching_merged)) * 100
            print(f"    {col}: {non_null:,} values ({pct:.1f}%)")
        except:
            pass
    if len(new_cols) > 20:
        print(f"    ... and {len(new_cols) - 20} more columns")
    
    # Save merged data
    output_file = HISTORIC_DIR / 'mlb_pitching_data_1950_2025_with_statcast.csv'
    print(f"\nSaving to {output_file.name}...")
    pitching_merged.to_csv(output_file, index=False)
    print(f"  Saved {len(pitching_merged):,} rows, {len(pitching_merged.columns)} columns")

def join_statcast_to_fielding(id_mapping):
    """Join statcast fielding data to historical fielding data"""
    print("\n" + "="*80)
    print("JOINING STATCAST FIELDING DATA")
    print("="*80)
    
    # Load historical fielding data
    print("\nLoading historical fielding data...")
    fielding = pd.read_csv(FIELDING_FILE)
    print(f"  Loaded {len(fielding):,} rows, {len(fielding.columns)} columns")
    original_cols = set(fielding.columns)
    
    # Check if Pos column exists
    if 'Pos' not in fielding.columns:
        print("  ERROR: 'Pos' column not found in fielding data!")
        return
    
    # Add Age from batting data if not already present
    if 'Age' not in fielding.columns:
        print("\nAdding Age column from batting data...")
        batting = pd.read_csv(BATTING_FILE, usecols=['IDfg', 'Season', 'Age'])
        age_before = fielding['Age'].notna().sum() if 'Age' in fielding.columns else 0
        fielding = fielding.merge(
            batting[['IDfg', 'Season', 'Age']],
            on=['IDfg', 'Season'],
            how='left',
            suffixes=('', '_batting')
        )
        age_added = fielding['Age'].notna().sum() - age_before
        print(f"  Added Age for {age_added:,} records")
        original_cols.add('Age')  # Track as part of original columns
    
    # Load statcast fielding file
    print("\nLoading statcast fielding file...")
    statcast = load_and_merge_statcast_files(STATCAST_FIELDER_FILES)
    
    if statcast is None:
        print("  No statcast fielding data found!")
        return
    
    print(f"  Loaded statcast data: {len(statcast):,} rows, {len(statcast.columns)} columns")
    
    # Check if position column exists in statcast data
    if 'position' not in statcast.columns:
        print("  ERROR: 'position' column not found in statcast data!")
        print("  Make sure you've downloaded FRV data with the updated script that includes position.")
        return
    
    # Map statcast player_id to FanGraphs IDfg
    print("\nMapping player IDs...")
    statcast['IDfg'] = statcast['player_id'].map(id_mapping)
    
    # Try name-based fallback for unmapped players
    add_name_based_mapping(statcast, fielding, id_col='IDfg', name_col='Name')
    
    # Keep mlbam_id for reference
    statcast['mlbam_id'] = statcast['player_id']
    
    # Remove rows where mapping failed
    before_filter = len(statcast)
    statcast = statcast.dropna(subset=['IDfg'])
    statcast['IDfg'] = statcast['IDfg'].astype(int)
    print(f"  Mapped {len(statcast):,} / {before_filter:,} rows successfully ({len(statcast)/before_filter*100:.1f}%)")
    
    # Drop player_name as it's no longer needed
    if 'player_name' in statcast.columns:
        statcast = statcast.drop(columns=['player_name'])
    
    # Rename year to Season and position to Pos for matching
    statcast = statcast.rename(columns={'year': 'Season', 'position': 'Pos'})
    
    # Identify and remove duplicate columns (but keep join keys)
    print("\nIdentifying duplicate columns...")
    duplicates = identify_duplicate_columns(fielding.columns, statcast.columns)
    # Don't remove join keys
    join_keys = {'IDfg', 'Season', 'Pos'}
    duplicates = duplicates - join_keys
    if duplicates:
        print(f"  Removing {len(duplicates)} duplicate columns: {sorted(duplicates)}")
        statcast = statcast.drop(columns=list(duplicates))
    else:
        print("  No duplicate columns found")
    
    # Remove source prefixes and add 'sc_' prefix
    statcast = remove_source_prefixes(statcast)
    
    # Drop player_id as we use IDfg
    if 'player_id' in statcast.columns:
        statcast = statcast.drop(columns=['player_id'])
    
    # Merge with historical data on IDfg, Season, AND Pos
    print("\nMerging with historical fielding data (by player, season, and position)...")
    print(f"  Before merge: {len(fielding):,} rows")
    fielding_merged = fielding.merge(
        statcast,
        on=['IDfg', 'Season', 'Pos'],
        how='left',
        suffixes=('', '_statcast')
    )
    print(f"  After merge: {len(fielding_merged):,} rows")
    
    # Count new columns added
    new_cols = sorted(set(fielding_merged.columns) - original_cols)
    print(f"\n  Added {len(new_cols)} new statcast columns:")
    for col in new_cols:
        try:
            non_null = fielding_merged[col].notna().sum()
            pct = (non_null / len(fielding_merged)) * 100
            print(f"    {col}: {non_null:,} values ({pct:.1f}%)")
        except:
            pass
    
    # Save merged data
    output_file = HISTORIC_DIR / 'mlb_fielding_data_2000_2025_with_statcast.csv'
    print(f"\nSaving to {output_file.name}...")
    fielding_merged.to_csv(output_file, index=False)
    print(f"  Saved {len(fielding_merged):,} rows, {len(fielding_merged.columns)} columns")

def main():
    print("\n" + "="*80)
    print("STATCAST DATA JOINER")
    print("="*80)
    print("\nThis script joins Statcast data to historical MLB data files")
    print("  - Batting: exitvelo, expected stats, sprint speed, baserunning run values")
    print("  - Pitching: arsenal speed/spin, exitvelo, expected stats")
    print("  - Fielding: fielding run values + Age from batting data")
    print("\nNote: Percentile ranks excluded (only available for qualified players)")
    print("\nUsing player_id_mapping.csv for ID mappings (with name-based fallback)")
    print("Adds mlbam_id column to all datasets for cross-referencing")
    print("Avoids duplicate columns (e.g., won't add 'ba' if 'AVG' exists)")
    print("\nNote: This script replaces moveage.py functionality")
    
    # Load ID mapping
    id_mapping = load_id_mapping()
    
    # Join data for each category
    try:
        join_statcast_to_batting(id_mapping)
    except Exception as e:
        print(f"\nERROR joining batting data: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        join_statcast_to_pitching(id_mapping)
    except Exception as e:
        print(f"\nERROR joining pitching data: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        join_statcast_to_fielding(id_mapping)
    except Exception as e:
        print(f"\nERROR joining fielding data: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nOutput files created:")
    print("  - mlb_batting_data_1950_2025_with_statcast.csv")
    print("  - mlb_pitching_data_1950_2025_with_statcast.csv")
    print("  - mlb_fielding_data_2000_2025_with_statcast.csv")
    print("\n")

if __name__ == '__main__':
    main()
