import pandas as pd

# Read the CSV files
fielding_df = pd.read_csv('mlb_fielding_data_2000_2024.csv')
batting_df = pd.read_csv('mlb_batting_data_2000_2024.csv')

# Create mapping dictionary from batting data
# Combine IDfg and Season as key for matching
age_mapping = dict(zip(batting_df['IDfg'].astype(str) + batting_df['Season'].astype(str), 
                      batting_df['Age']))

# Create matching key in fielding data
fielding_df['matching_key'] = fielding_df['IDfg'].astype(str) + fielding_df['Season'].astype(str)

# Add Age column using the mapping
fielding_df['Age'] = fielding_df['matching_key'].map(age_mapping)

# Remove the temporary matching key column
fielding_df = fielding_df.drop('matching_key', axis=1)

# Save the updated fielding data
fielding_df.to_csv('mlb_fielding_data_2000_2024_with_age.csv', index=False)

print("Processing complete. New file saved with Age column added.")