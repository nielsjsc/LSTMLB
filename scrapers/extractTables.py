import os
import pandas as pd
from bs4 import BeautifulSoup
import re

# Define the specific columns for hitters and pitchers for 2017 and 2018
HITTER_COLUMNS_2017_2018 = {'Hit', 'Game', 'Raw', 'Spd', 'Fld', 'Arm'}
PITCHER_COLUMNS_2017_2018 = {'FB', 'SL', 'CB', 'CH', 'CMD'}

# Define columns to exclude from hitter and pitcher dataframes
EXCLUDE_FROM_HITTERS = {'CB', 'CH', 'CMD', 'FB', 'SL'}
EXCLUDE_FROM_PITCHERS = {'Game', 'Hit', 'Arm', 'Raw', 'Spd'}

def parse_html_table(file_path):
    """Parse an HTML table and return a DataFrame."""
    # Use 'utf-8-sig' to handle accented characters correctly
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        soup = BeautifulSoup(file, 'html.parser')
        table = soup.find('table')
        
        # Extract headers from data-stat attribute
        headers = [th['data-stat'] for th in table.find('thead').find_all('th')]
        rows = []
        for tr in table.find('tbody').find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all('td')]
            rows.append(cells)
        df = pd.DataFrame(rows, columns=headers)
        
        # Extract year from filename
        year = re.search(r'\d{4}', file_path).group()
        df['Year'] = year
        
        return df

def merge_dataframes(dfs):
    """Merge multiple DataFrames, ensuring all columns are preserved."""
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    all_columns = sorted(all_columns)
    merged_df = pd.DataFrame(columns=all_columns)
    for df in dfs:
        merged_df = pd.concat([merged_df, df.reindex(columns=all_columns)], ignore_index=True)
    return merged_df

def filter_players_2017_2018(df):
    """Filter players for 2017 and 2018 based on the presence of specific columns."""
    hitters_df = df[df['Hit'].notna()]
    pitchers_df = df[df['FB'].notna()]
    return hitters_df, pitchers_df

def process_scouting_tables(html_files):
    """Process scouting tables and sort players into hitters and pitchers."""
    hitters_dfs = []
    pitchers_dfs = []

    for file in html_files:
        df = parse_html_table(file)
        year = int(re.search(r'\d{4}', file).group())
        
        if year in [2017, 2018]:
            hitters_df, pitchers_df = filter_players_2017_2018(df)
            hitters_dfs.append(hitters_df)
            pitchers_dfs.append(pitchers_df)
        else:
            if 'hitters' in file.lower():
                hitters_dfs.append(df)
            elif 'pitchers' in file.lower():
                pitchers_dfs.append(df)

    hitters_df = merge_dataframes(hitters_dfs)
    pitchers_df = merge_dataframes(pitchers_dfs)

    return hitters_df, pitchers_df

def process_summary_tables(html_files, hitters_df, pitchers_df):
    """Process summary tables and merge with scouting data."""
    summary_dfs = [parse_html_table(file) for file in html_files if 'summary' in file.lower()]
    summary_df = merge_dataframes(summary_dfs)

    # Debugging: Print columns of DataFrames before merging
    print("Hitters DataFrame columns:", hitters_df.columns)
    print("Pitchers DataFrame columns:", pitchers_df.columns)
    print("Summary DataFrame columns:", summary_df.columns)

    # Ensure column names match between scouting and summary tables
    common_columns = ['Name', 'Org', 'Pos']
    for col in common_columns:
        if col not in summary_df.columns:
            raise KeyError(f"Column '{col}' not found in summary DataFrame")

    hitters_df = pd.merge(hitters_df, summary_df, on=common_columns, how='left')
    pitchers_df = pd.merge(pitchers_df, summary_df, on=common_columns, how='left')

    return hitters_df, pitchers_df

def clean_and_filter_dataframe(df, role):
    """Remove '_x' columns, drop empty columns, clean formats, and filter rows based on role."""
    # Drop '_x' columns and keep '_y' columns
    df = df.drop([col for col in df.columns if col.endswith('_x')], axis=1)
    df.columns = df.columns.str.replace('_y$', '', regex=True)

    # Drop columns with all rows empty
    df = df.dropna(axis=1, how='all')

    # Clean formats like '40/50' to extract numeric values
    def clean_format(value):
        if isinstance(value, str) and '/' in value:
            parts = value.split('/')
            try:
                return (int(parts[0]) + int(parts[1])) / 2
            except ValueError:
                return value
        return value
    
    df = df.applymap(clean_format)

    # Filter out players based on the role and ensure the column exists
    if role == 'hitters' and 'Hit' in df.columns:
        df = df[df['Hit'].notna() & (df['Hit'] != '')]  # Keep only rows where 'Hit' is not NaN or empty
    elif role == 'pitchers' and 'FB' in df.columns:
        df = df[df['FB'].notna() & (df['FB'] != '')]  # Keep only rows where 'FB' is not NaN or empty

    return df

def remove_irrelevant_columns(df, exclude_columns):
    """Remove columns that are not relevant to the dataset."""
    return df[[col for col in df.columns if col not in exclude_columns]]

def main():
    base_dir = "./data/tables/"
    
    # Generate scouting files dynamically
    scouting_files = [f"{base_dir}prospects_{year}_{role}.html" for year in range(2019, 2025) for role in ['hitters', 'pitchers']]
    scouting_files += [f"{base_dir}prospects_{year}.html" for year in [2017, 2018]]

    # Generate summary files dynamically
    summary_files = [f"{base_dir}prospects_{year}_summary.html" for year in range(2017, 2025)]

    # Process scouting tables
    hitters_df, pitchers_df = process_scouting_tables(scouting_files)

    # Process summary tables and merge with scouting data
    hitters_df, pitchers_df = process_summary_tables(summary_files, hitters_df, pitchers_df)

    # Clean and filter hitters and pitchers
    hitters_df = clean_and_filter_dataframe(hitters_df, 'hitters')
    pitchers_df = clean_and_filter_dataframe(pitchers_df, 'pitchers')

    # Remove irrelevant columns
    hitters_df = remove_irrelevant_columns(hitters_df, EXCLUDE_FROM_HITTERS)
    pitchers_df = remove_irrelevant_columns(pitchers_df, EXCLUDE_FROM_PITCHERS)

    # Save to CSV using 'utf-8-sig' encoding to preserve special characters correctly
    hitters_df.to_csv('./data/hitters_prospects.csv', index=False, encoding='utf-8-sig')
    pitchers_df.to_csv('./data/pitchers_prospects.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()