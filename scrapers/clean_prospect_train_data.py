import pandas as pd
import numpy as np
import re
import os

# Define file paths for matched hitters and pitchers CSVs
base_dir = './data'
matched_hitters_path = os.path.join(base_dir, 'matched_hitters.csv')
matched_pitchers_path = os.path.join(base_dir, 'matched_pitchers.csv')

# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)

# Load data
matched_hitters = pd.read_csv(matched_hitters_path, encoding='utf-8')
matched_pitchers = pd.read_csv(matched_pitchers_path, encoding='utf-8')

# Function to convert height from format like "6' 2\"" to inches
def convert_height_to_inches(height_str):
    if pd.isna(height_str):
        return np.nan
    match = re.match(r"(\d+)' (\d+)\"", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 12 + inches
    return np.nan

# Function to calculate years before MLB debut
def calculate_years_before_debut(row):
    try:
        return int(row['MLB_Debut']) - int(row['Sign Yr'])
    except:
        return np.nan

# Function to extract signing information based on the rules provided
def extract_signing_info(row):
    year = int(row['Year'])
    if year in [2017, 2018]:
        signed_info = row['Signed']
        if pd.notna(signed_info):
            if signed_info.startswith("J2"):
                row['Sign Mkt'] = "J2"
                row['Sign Yr'] = signed_info[-4:]  # The last 4 characters are the year
            else:
                row['Sign Mkt'] = "Draft"
                row['Sign Yr'] = signed_info[-4:]  # The last 4 characters are the year
    elif year == 2019:
        signed_info = row['Signed']
        if pd.notna(signed_info):
            sign_year = signed_info[:4]
            row['Sign Yr'] = sign_year
            if "J2" in signed_info:
                row['Sign Mkt'] = "J2"
            else:
                row['Sign Mkt'] = "Draft"
    elif year == 2020:
        signed_from = row['Signed From']
        if pd.notna(signed_from):
            if signed_from in ['Dominican Republic', 'Venezuela', 'Cuba', 'Mexico', 'Panama', 'Puerto Rico', 
                               'Japan', 'South Korea', 'Taiwan', 'Colombia', 'Curacao', 'Aruba', 'Nicaragua', 
                               'Brazil', 'Australia', 'Canada', 'United States']:
                row['Sign Mkt'] = "J2"
            else:
                row['Sign Mkt'] = "Draft"
            # No 'Sign Yr' available for 2020 based on provided data    
    return row

# Function to convert FV values like "35+" to a numeric average
def convert_fv(fv_str):
    if pd.isna(fv_str):
        return np.nan
    match = re.match(r"(\d+)\+", fv_str)
    if match:
        base_value = int(match.group(1))
        return base_value + 2.5
    try:
        return float(fv_str)
    except:
        return np.nan

# Function to calculate average fastball velocity from the "Sits" column
def calculate_avg_fb_velocity(sits_str):
    if pd.isna(sits_str):
        return np.nan
    match = re.match(r"(\d+)-(\d+)", sits_str)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (low + high) / 2
    return np.nan

# Function to parse the "Bonus" column and convert to millions
def parse_bonus(bonus_str):
    if pd.isna(bonus_str) or bonus_str == '':
        return np.nan
    # Remove the dollar sign if present
    bonus_str = bonus_str.replace('$', '')
    if 'M' in bonus_str:
        return float(bonus_str.replace('M', ''))
    if 'k' in bonus_str:
        return float(bonus_str.replace('k', '')) / 1000
    return np.nan


def calculate_earliest_debut(df):
    # Ensure all entries are retained, calculate earliest debuts
    earliest_debut = df.groupby(['Name', 'Org'])['MLB_Debut'].min().reset_index()
    earliest_debut.rename(columns={'MLB_Debut': 'Earliest_Debut'}, inplace=True)

    # Merge the earliest debut back into the original DataFrame
    df = df.merge(earliest_debut, on=['Name', 'Org'], how='left')

    # Update the MLB_Debut to reflect the earliest debut year
    df['MLB_Debut'] = df['Earliest_Debut']
    
    # Drop the auxiliary column if no longer needed
    df.drop(columns=['Earliest_Debut'], inplace=True)

    return df

# Function to fill missing signing bonuses for the same player across multiple years
def fill_signing_bonus(df):
    df['Bonus'] = df.groupby(['Name'])['Bonus'].transform(lambda x: x.ffill().bfill())
    return df

# Function to remove duplicate rows for the same player, organization, position, and year
def remove_duplicates(df):
    df = df.drop_duplicates(subset=['Name', 'Org', 'Pos', 'Year'], keep='first')
    return df

def preprocess_hitters_data(df):
    # Convert height to inches
    df['Ht'] = df['Ht'].apply(convert_height_to_inches)
    
    # Apply signing information extraction
    df = df.apply(extract_signing_info, axis=1)
    
    # Calculate years before MLB debut
    df['Years_Before_Debut'] = df.apply(calculate_years_before_debut, axis=1)
    
    # Convert FV to numeric values
    df['FV'] = df['FV'].apply(convert_fv)
    
    # Parse the "Bonus" column
    df['Bonus'] = df['Bonus'].apply(parse_bonus)
    
    # Fill missing signing bonuses
    df = fill_signing_bonus(df)
    
    # Remove duplicate rows
    df = remove_duplicates(df)
    df = calculate_earliest_debut(df)

    # Preprocess the "Hard Hit%" column by removing the '%' and converting to numeric
    if 'Hard Hit%' in df.columns:
        df['Hard Hit%'] = df['Hard Hit%'].str.rstrip('%').astype(float)
    
    # Convert other numeric columns that might contain non-numeric values
    numeric_columns = ['Avg EV', 'Max EV', 'Hard Hit%', 'Bat Ctrl', 'Fld', 'Game', 
                       'Hit', 'Pitch Sel', 'Raw', 'Spd', 'Risk', 'Org Rk', 'Top 100']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop unnecessary columns based on availability of key data points
    columns_to_drop = ['Signed', 'Signed From', 'Sign Org', 'Video', 'Report', 'Current Level', 'Arm', 'Trend']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df

def preprocess_pitchers_data(df):
    # Convert height to inches
    df['Ht'] = df['Ht'].apply(convert_height_to_inches)
    
    # Apply signing information extraction
    df = df.apply(extract_signing_info, axis=1)
    
    # Calculate years before MLB debut
    df['Years_Before_Debut'] = df.apply(calculate_years_before_debut, axis=1)
    
    # Convert FV to numeric values
    df['FV'] = df['FV'].apply(convert_fv)
    
    # Parse the "Bonus" column
    df['Bonus'] = df['Bonus'].apply(parse_bonus)
    
    # Fill missing signing bonuses
    df = fill_signing_bonus(df)
    
    # Remove duplicate rows
    df = remove_duplicates(df)
    df = calculate_earliest_debut(df)

    # Calculate average fastball velocity
    df['Avg FB Velo'] = df['Sits'].apply(calculate_avg_fb_velocity)
    
    # Convert numeric columns that might contain non-numeric values
    numeric_columns = ['CB', 'CH', 'CMD', 'FB', 'Fld', 'RPM Break', 'RPM FB', 
                       'SL', 'Avg FB Velo', 'Org Rk', 'Top 100', 'Wt', 'WAR_Sum']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop unnecessary columns based on availability of key data points
    columns_to_drop = ['Signed', 'Signed From', 'Sign Org', 'Video', 'Report', 'Current Level', 'Trend', 'Sits']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    return df

# Preprocess hitters and pitchers data separately
processed_hitters = preprocess_hitters_data(matched_hitters)
processed_pitchers = preprocess_pitchers_data(matched_pitchers)

# Save the cleaned and processed data to new CSV files
processed_hitters.to_csv(os.path.join(base_dir, 'cleaned_matched_hitters.csv'), index=False, encoding='utf-8-sig')
processed_pitchers.to_csv(os.path.join(base_dir, 'cleaned_matched_pitchers.csv'), index=False, encoding='utf-8-sig')

print("Data preprocessing completed and saved as cleaned_matched_hitters.csv and cleaned_matched_pitchers.csv")