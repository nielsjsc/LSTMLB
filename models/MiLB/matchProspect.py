import pandas as pd
import os
import numpy as np
from pathlib import Path

# Set up absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up two levels to LSTMLB root
SAVE_DIR = PROJECT_ROOT / 'data' / 'generated' / 'MiLB'
DATA_DIR = PROJECT_ROOT / 'models' / 'MiLB' / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# Create output directory if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f"Script directory: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")

def calculate_true_average(df):
    """
    Calculate true average rankings by only including unique values for each source
    """
    # Columns that contain rankings (exclude metadata columns)
    ranking_columns = [col for col in df.columns if col not in ['Player Name', 'Name', 'Team', 'Pos', 'RANK', 'AVG']]
    
    # Create new column for true average
    df['True_AVG'] = df.apply(lambda row: calculate_row_average(row, df, ranking_columns), axis=1)
    
    return df

def calculate_row_average(row, df, ranking_columns):
    """Calculate average for a single row using only unique rankings"""
    valid_rankings = []
    
    for col in ranking_columns:
        value = row[col]
        # Check if value is numeric and not placeholder (150)
        if pd.notna(value) and value != 'Paywall':
            try:
                value = float(value)
                # Check if this value is unique in this column
                column_values = df[col].dropna()
                value_counts = column_values.value_counts()
                if value in value_counts and value_counts[value] == 1:
                    valid_rankings.append(value)
            except ValueError:
                continue
    
    return np.mean(valid_rankings) if valid_rankings else None

def load_composite_rankings():
    """Load all composite rankings from 2022-2025"""
    composite_rankings = {}
    
    for year in range(2022, 2026):
        file = DATA_DIR / f'{year}-composite.csv'
        if file.exists():
            df = pd.read_csv(file)
            name_column = next((col for col in ['Player Name', 'PlayerName'] if col in df.columns), None)
            
            if name_column is None:
                print(f"Warning: Could not find name column in {year} composite file")
                print(f"Available columns: {df.columns.tolist()}")
                continue
            
            # Standardize name format
            df['Name'] = df[name_column].apply(lambda x: 
                ' '.join(reversed(x.split(', '))) if ', ' in str(x) else x)
            
            # Calculate true average rankings
            df = calculate_true_average(df)
            
            # Replace original AVG with True_AVG
            df['AVG'] = df['True_AVG']
            df = df.drop('True_AVG', axis=1)
            
            composite_rankings[year] = df
            print(f"Loaded {year} composite rankings with {len(df)} players")
    
    return composite_rankings
def load_historical_data(historical_years):
    """Load all historical player data into a single DataFrame"""
    historical_dfs = []
    
    for year in historical_years:
        # Load both hitters and pitchers data using RAW_DATA_DIR
        hitter_path = RAW_DATA_DIR / 'hitters' / f'{year}.csv'
        pitcher_path = RAW_DATA_DIR / 'pitchers' / f'{year}.csv'
        
        for path in [hitter_path, pitcher_path]:
            if path.exists():
                df = pd.read_csv(path)
                df['Year'] = year
                

                historical_dfs.append(df)
                print(f"Loaded {year} {'hitter' if 'hitter' in str(path) else 'pitcher'} data with {len(df)} players")
    
    final_df = pd.concat(historical_dfs, ignore_index=True) if historical_dfs else pd.DataFrame()
    print(f"\nTotal historical players loaded: {len(final_df)}")
    return final_df

def process_player_history(player_data, composite_rankings):
    """Process historical data for a single player"""
    histories = []
    player_name = player_data['Name'].iloc[0]
    IDfg = player_data['IDfg'].iloc[0]
    
    # Process each year's data
    for year, year_data in player_data.groupby('Year'):
        row = year_data.iloc[0]
        
        # Base info for all players
        year_info = {
            'Name': player_name,
            'IDfg': IDfg,
            'Year': year,
            'Team': row['Org'],
            'Level': row['Current Level'],
            'Age': row['Age'],
            'FV': row['FV']
        }
        
        # Add all grade columns if they exist
        grade_columns = ['Hit', 'Game', 'Raw', 'Spd', 'FB', 'SL', 'CB', 'CH', 'CMD']
        for col in grade_columns:
            if col in row:
                year_info[col] = row[col]
        
        # Add composite ranking if available
        if year in composite_rankings:
            year_composite = composite_rankings[year]
            year_match = year_composite[year_composite['Name'] == player_name]
            if not year_match.empty:
                year_info[f'{year}_Composite'] = year_match.iloc[0]['AVG']
                year_info[f'{year}_Value'] = calculate_prospect_value(
                    year_info['FV'],
                    year_match.iloc[0]['AVG']
                )
        
        histories.append(year_info)
    
    return histories





def create_player_mapping(composite_rankings, historical_df):
    """Create a mapping of names to IDfgs using only name matching"""
    player_mapping = {}
    duplicates = []
    
    # Process historical data first
    for _, row in historical_df.iterrows():
        std_name = standardize_name(row['Name'])
        if std_name not in player_mapping:
            player_mapping[std_name] = {
                'IDfg': row['IDfg'],
                'original_names': {row['Name']},
                'sources': {'historical'}
            }
        else:
            # Track potential duplicates
            if player_mapping[std_name]['IDfg'] != row['IDfg']:
                duplicates.append({
                    'standardized_name': std_name,
                    'name1': row['Name'],
                    'name2': list(player_mapping[std_name]['original_names'])[0],
                    'id1': row['IDfg'],
                    'id2': player_mapping[std_name]['IDfg']
                })
    
    # Process composite rankings
    for year, df in composite_rankings.items():
        for _, row in df.iterrows():
            std_name = standardize_name(row['Name'])
            if std_name not in player_mapping:
                player_mapping[std_name] = {
                    'IDfg': None,  # Will need manual matching
                    'original_names': {row['Name']},
                    'sources': {f'composite_{year}'}
                }
            else:
                player_mapping[std_name]['original_names'].add(row['Name'])
                player_mapping[std_name]['sources'].add(f'composite_{year}')
    
    # Save duplicates for manual review
    if duplicates:
        pd.DataFrame(duplicates).to_csv('duplicate_players.csv', index=False)
        print(f"Found {len(duplicates)} potential duplicate players. See duplicate_players.csv")
    
    return player_mapping

def standardize_name(name):
    """Standardize name format by removing special characters and suffixes"""
    import unicodedata
    import re
    
    # Convert to standard form and remove accents
    name = unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove common suffixes and extra spaces
    name = re.sub(r'\s+(Jr\.?|Sr\.?|I{2,}|IV)$', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name.lower()


def match_player(name, player_mapping, historical_df=None):
    """Match a player name using the mapping and handle edge cases"""
    std_name = standardize_name(name)
    
    # Direct match in mapping
    if std_name in player_mapping:
        return player_mapping[std_name]['IDfg']
    
    # Try to find match in historical data if provided
    if historical_df is not None:
        matches = historical_df[historical_df['Name'].apply(standardize_name) == std_name]
        if not matches.empty:
            player_id = matches.iloc[0]['IDfg']
            # Add to mapping for future use
            player_mapping[std_name] = {
                'IDfg': player_id,
                'original_names': {name}
            }
            return player_id
    
    return None

def process_players(historical_df, composite_rankings):
    """Process all players from 2022-2025"""
    print("\nStarting IDfg value checks:")
    print("\nSample of historical_df IDfg values:")
    print(historical_df[['Name', 'IDfg', 'Year']].head())
    
    histories = []
    unmatched = []
    # Process historical data (2022-2025)
    if not historical_df.empty:
        for year, year_data in historical_df.groupby('Year'):
            print(f"\nProcessing year {year}")
            for _, player in year_data.iterrows():
                # Debug player IDfg
                if isinstance(player['IDfg'], pd.Series):
                    print(f"WARNING: Series IDfg found for {player['Name']}:")
                    print(player['IDfg'])
                
                std_name = standardize_name(player['Name'])
                
                history_entry = {
                    'Name': player['Name'],
                    'Team': player['Org'],
                    'Position': player['Pos'],
                    'Year': year,
                    'Level': player['Current Level'],
                    'Age': player['Age'],
                    'FV': player['FV'],
                    'IDfg': player['IDfg'] if not isinstance(player['IDfg'], pd.Series) else player['IDfg'].iloc[0]
                }
                
                # Debug history entry
                if isinstance(history_entry['IDfg'], pd.Series):
                    print(f"WARNING: Series IDfg in history_entry for {history_entry['Name']}")
                
                # Add grades and composite rankings
                grade_columns = ['Hit', 'Game', 'Raw', 'Spd', 'FB', 'SL', 'CB', 'CH', 'CMD']
                for col in grade_columns:
                    if col in player and pd.notna(player[col]):
                        history_entry[col] = player[col]
                
                if year in composite_rankings:
                    year_composite = composite_rankings[year]
                    composite_match = year_composite[
                        year_composite['Name'].apply(standardize_name) == std_name
                    ]
                    if not composite_match.empty:
                        history_entry[f'{year}_Composite'] = composite_match.iloc[0]['AVG']
                        history_entry[f'{year}_Rank'] = composite_match.iloc[0]['RANK']
                        history_entry[f'{year}_Value'] = calculate_prospect_value(
                            history_entry['FV'],
                            composite_match.iloc[0]['RANK']
                        )
                
                histories.append(history_entry)
    
    # Process any remaining 2025 composite rankings that weren't in historical data
    composite_2025 = composite_rankings[2025]
    for _, player in composite_2025.iterrows():
        std_name = standardize_name(player['Name'])
        
        # Skip if we already have 2025 data for this player
        if any(h['Year'] == 2025 and standardize_name(h['Name']) == std_name for h in histories):
            continue
            
        # Try to find 2025 data first
        player_2025 = historical_df[
            (historical_df['Year'] == 2025) & 
            (historical_df['Name'].apply(standardize_name) == std_name)
        ]
        
        # If no 2025 data, try 2024 data
        player_2024 = historical_df[
            (historical_df['Year'] == 2024) & 
            (historical_df['Name'].apply(standardize_name) == std_name)
        ] if player_2025.empty else pd.DataFrame()
        
        history_entry = {
            'Name': player['Name'],
            'Team': player['Team'],
            'Position': player['Pos'],
            'Year': 2025,
            '2025_Composite': player['AVG'],
            '2025_Rank': player['RANK']
        }
        
        # Use 2025 data if available, otherwise fall back to 2024
        source_data = player_2025 if not player_2025.empty else player_2024
        if not source_data.empty:
            latest = source_data.iloc[0]
            history_entry['FV'] = latest['FV']
            if 'IDfg' in latest:
                history_entry['IDfg'] = latest['IDfg']
            for col in grade_columns:
                if col in latest:
                    history_entry[col] = latest[col]
        
        history_entry['2025_Value'] = calculate_prospect_value(
            history_entry.get('FV'),
            player['RANK']
        )
        
        histories.append(history_entry)
    
    # Create DataFrame and interpolate missing values
    histories_df = pd.DataFrame(histories)
    histories_df = interpolate_missing_2025_values(histories_df)
    
    return histories_df, pd.DataFrame(unmatched)



def calculate_prospect_value(fv_str, rank):
    """Calculate prospect value based on FV and rank"""
    
    if pd.notna(fv_str):
        try:
            # Handle FV with plus grades
            if '+' in str(fv_str):
                fv = float(str(fv_str).replace('+', '')) + 2.5
            else:
                fv = float(fv_str)
            
            # Base values for each FV tier
            fv_values = {
                70: 180000000,
                65: 100000000,
                60: 80000000,
                55: 75000000,
                50: 55000000,
                45: 40000000, 
                40: 15000000,
                35: 5000000,
                30: 2000000
            }
            
            # Find closest FV tier and get base value
            base_fv = max(k for k in fv_values.keys() if k <= fv)
            base_value = fv_values[base_fv]
            
            if pd.notna(rank):
                # New rank adjustment formula
                rank_float = float(rank)
                if rank_float <= 100:
                    rank_adj = 0.9 - (rank_float - 1) * 0.4 / 100  # Top 100 gradual decline from 0.9 to 0.5
                else:
                    rank_adj = 0.5 - (min(rank_float - 100, 400) * 0.2 / 400)  # Ranks 101-500 decline from 0.5 to 0.3
                
                rank_adj = max(0.3, rank_adj)  # Minimum adjustment of 0.3
                return base_value * rank_adj
            return base_value * 0.3  # Default adjustment if no rank
            
        except Exception as e:
            print(f"Error calculating value for FV {fv_str}: {str(e)}")
            return None
    
    return None

def interpolate_missing_2025_values(histories_df):
    """Fill in missing 2025 values by averaging the four closest ranked players' values"""
    
    # Create a copy to avoid warnings
    histories_df = histories_df.copy()
    
    # Filter to 2025 players and sort by composite rank
    df_2025 = histories_df[histories_df['Year'] == 2025].sort_values('2025_Composite')
    
    # For each player with missing value
    for idx in df_2025[df_2025['2025_Value'].isna()].index:
        current_composite = df_2025.loc[idx, '2025_Composite']
        
        # Get players with valid values
        valid_players = df_2025[
            (df_2025['2025_Value'].notna()) & 
            (df_2025.index != idx)
        ]
        
        # Calculate composite score distance for all valid players
        valid_players['composite_diff'] = (valid_players['2025_Composite'] - current_composite).abs()
        
        # Get 4 closest players by composite score
        closest = valid_players.nsmallest(4, 'composite_diff')
        
        # Calculate average if we have any values
        if not closest.empty:
            histories_df.loc[idx, '2025_Value'] = closest['2025_Value'].mean()
    
    return histories_df

def main():
    """Main processing function using path constants"""
    historical_years = range(2017, 2026)
    
    # Load all data using new path constants
    composite_rankings = load_composite_rankings()
    historical_df = load_historical_data(historical_years)
    #rename playerId to IDfg
    historical_df.rename(columns={'playerId': 'IDfg'}, inplace=True)
    # Process players
    print("\nProcessing players...")
    player_histories, unmatched_players = process_players(historical_df, composite_rankings)
    
    # Filter to only keep 2022-2025 entries
    player_histories = player_histories[player_histories['Year'] >= 2022]
    
    # Remove players with no rankings in any year 2022-2025
    ranking_columns = [f'{year}_Composite' for year in range(2022, 2026)]
    has_ranking = player_histories[ranking_columns].notna().any(axis=1)
    player_histories = player_histories[has_ranking]
    
    # Save filtered results using PROCESSED_DIR
    player_histories.to_csv(SAVE_DIR / 'player_histories.csv', index=False)
    unmatched_players.to_csv(PROCESSED_DIR / 'unmatched_players.csv', index=False)
    
    print(f"\nProcessed {len(player_histories['Name'].unique())} unique players")
    print(f"Found {len(unmatched_players)} unmatched players")
    print(f"Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()