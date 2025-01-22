import pandas as pd
import os
from unidecode import unidecode

def load_data(prospect_file, mlb_hitters_file, mlb_pitchers_file):
    # Load data with encoding to handle Latin names properly
    prospects = pd.read_csv(prospect_file, encoding='utf-8-sig')
    mlb_hitters = pd.read_csv(mlb_hitters_file, encoding='utf-8-sig')
    mlb_pitchers = pd.read_csv(mlb_pitchers_file, encoding='utf-8-sig')
    return prospects, mlb_hitters, mlb_pitchers

def normalize_name(name):
    return unidecode(name)

def match_prospects_with_mlb(prospects, mlb_stats, position):
    # Normalize MLB names for matching
    mlb_stats['Name'] = mlb_stats['Name'].apply(normalize_name)
    
    # Initialize list for matched data
    matched_data = []
    
    for idx, prospect in prospects.iterrows():
        prospect_name = normalize_name(prospect['Name'])
        prospect_year = prospect['Year']
        prospect_team = prospect['Org']
        
        # Find MLB stats for players with the same name
        mlb_records = mlb_stats[mlb_stats['Name'] == prospect_name]
        
        if mlb_records.empty:
            # Player has not yet appeared in the MLB
            matched_data.append({**prospect, 'MLB_Debut': None, 'Years_Played': 0, 'Games_Played': 0, 'WAR_Sum': 0})
            continue
        
        # Handle multiple matches with additional filters
        matched_mlb = []
        for _, mlb_record in mlb_records.iterrows():
            mlb_team = mlb_record['Team']
            mlb_year = mlb_record['Season']

            # Check if MLB debut year makes sense
            if mlb_year >= prospect_year:
                matched_mlb.append(mlb_record)

        if not matched_mlb:
            # No suitable MLB records found
            matched_data.append({**prospect, 'MLB_Debut': None, 'Years_Played': 0, 'Games_Played': 0, 'WAR_Sum': 0})
            continue
        
        # Sum MLB WAR and games played
        if position == 'hitter':
            l_war_sum = sum(record['L-WAR'] for record in matched_mlb)
        else:
            l_war_sum = sum(record['WAR'] for record in matched_mlb)
        
        games_played_sum = sum(record['G'] for record in matched_mlb)
        mlb_debut = matched_mlb[0]['Season']
        num_years_played = len(set(record['Season'] for record in matched_mlb))

        # Combine prospect and MLB data
        combined_record = {**prospect, 'MLB_Debut': mlb_debut, 'Years_Played': num_years_played, 
                           'Games_Played': games_played_sum, 'WAR_Sum': l_war_sum}
        matched_data.append(combined_record)
    
    return pd.DataFrame(matched_data)

def save_matched_data(hitters_data, pitchers_data, output_hitters_file, output_pitchers_file):
    # Save matched data for hitters and pitchers separately
    hitters_data.to_csv(output_hitters_file, index=False, encoding='utf-8-sig')
    pitchers_data.to_csv(output_pitchers_file, index=False, encoding='utf-8-sig')

def main():
    # File paths
    base_dir = './data'
    hitter_prospects_file = os.path.join(base_dir, 'hitters_prospects.csv')
    pitcher_prospects_file = os.path.join(base_dir, 'pitchers_prospects.csv')
    mlb_hitters_file = os.path.join(base_dir, 'mlb_batting_data_2010_2024.csv')
    mlb_pitchers_file = os.path.join(base_dir, 'mlb_pitching_data_2010_2024.csv')
    output_hitters_file = os.path.join(base_dir, 'matched_hitters.csv')
    output_pitchers_file = os.path.join(base_dir, 'matched_pitchers.csv')
    
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Load data
    hitter_prospects, mlb_hitters, mlb_pitchers = load_data(hitter_prospects_file, mlb_hitters_file, mlb_pitchers_file)
    pitcher_prospects, _, _ = load_data(pitcher_prospects_file, mlb_hitters_file, mlb_pitchers_file)
    
    # Match hitters and pitchers with their MLB data
    matched_hitters = match_prospects_with_mlb(hitter_prospects, mlb_hitters, 'hitter')
    matched_pitchers = match_prospects_with_mlb(pitcher_prospects, mlb_pitchers, 'pitcher')
    
    # Save the matched data to CSV files
    save_matched_data(matched_hitters, matched_pitchers, output_hitters_file, output_pitchers_file)

if __name__ == "__main__":
    main()