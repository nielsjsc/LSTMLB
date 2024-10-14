import pandas as pd
from pybaseball import batting_stats, pitching_stats
import os

def fetch_data_in_chunks(start_year, end_year, data_func):
    all_data = []
    for year in range(start_year, end_year + 1):
        try:
            data = data_func(year, year, qual=0)
            all_data.append(data)
            print(f"Fetched data for year {year}")
        except requests.exceptions.HTTPError as e:
            print(f"Failed to fetch data for year {year}: {e}")
    return pd.concat(all_data, ignore_index=True)

# Get user input for the start and end years
start_year = int(input("Enter the start year: "))
end_year = int(input("Enter the end year: "))

# Fetch batting data in smaller chunks
batting_data = fetch_data_in_chunks(start_year, end_year, batting_stats)

# Fetch pitching data in smaller chunks
pitching_data = fetch_data_in_chunks(start_year, end_year, pitching_stats)

# Define the base directory for saving files
base_dir = './data'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Save the data to CSV files with the correct names
batting_file_name = os.path.join(base_dir, f'mlb_batting_data_{start_year}_{end_year}.csv')
pitching_file_name = os.path.join(base_dir, f'mlb_pitching_data_{start_year}_{end_year}.csv')
batting_data.to_csv(batting_file_name, index=False)
pitching_data.to_csv(pitching_file_name, index=False)
