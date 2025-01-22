import pandas as pd
import numpy as np
from pybaseball import (
    batting_stats, pitching_stats, fielding_stats,
    statcast_batter, statcast_pitcher,
    team_batting, team_fielding
)
import os
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data_with_retry(func, *args, max_retries=3, **kwargs):
    """Fetch data with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            data = func(*args, **kwargs)
            return data
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
    raise Exception(f"Failed after {max_retries} attempts")

def fetch_data_in_chunks(start_year, end_year, data_func):
    """Fetch data year by year with retry logic"""
    all_data = []
    for year in range(start_year, end_year + 1):
        try:
            data = fetch_data_with_retry(data_func, year, year, qual=0)
            all_data.append(data)
            logger.info(f"Fetched data for year {year}")
            time.sleep(2)  # Rate limiting
        except Exception as e:
            logger.error(f"Failed to fetch data for year {year}: {e}")
    return pd.concat(all_data, ignore_index=True)

def fetch_statcast_data(start_year, end_year):
    """Fetch Statcast data in small date ranges"""
    all_data = []
    start_date = datetime(start_year, 3, 1)  # Start of season
    end_date = datetime(end_year, 11, 1)    # End of season
    current_date = start_date
    
    while current_date < end_date:
        next_date = current_date + timedelta(days=7)
        try:
            data = fetch_data_with_retry(
                statcast_batter,
                str(current_date.date()),
                str(next_date.date())
            )
            if data is not None and not data.empty:
                all_data.append(data)
                logger.info(f"Fetched Statcast data from {current_date.date()} to {next_date.date()}")
            time.sleep(3)  # Rate limiting
        except Exception as e:
            logger.error(f"Failed to fetch Statcast data: {e}")
        current_date = next_date
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def main():
    # Get user input
    start_year = int(input("Enter the start year: "))
    end_year = int(input("Enter the end year: "))
    
    # Create data directory
    base_dir = './data'
    os.makedirs(base_dir, exist_ok=True)
    
    # Fetch different types of data
    data_types = {
        'batting': (batting_stats, 'mlb_batting_data'),
        'pitching': (pitching_stats, 'mlb_pitching_data'),
        'fielding': (fielding_stats, 'mlb_fielding_data')
    }
    
    # Fetch and save regular stats
    for data_type, (func, filename) in data_types.items():
        logger.info(f"Fetching {data_type} data...")
        try:
            data = fetch_data_in_chunks(start_year, end_year, func)
            file_path = os.path.join(base_dir, f'{filename}_{start_year}_{end_year}.csv')
            data.to_csv(file_path, index=False)
            logger.info(f"Saved {data_type} data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to fetch {data_type} data: {e}")
    
    # Fetch and save Statcast data
    logger.info("Fetching Statcast data...")
    try:
        statcast_data = fetch_statcast_data(start_year, end_year)
        if not statcast_data.empty:
            statcast_file = os.path.join(base_dir, f'mlb_statcast_data_{start_year}_{end_year}.csv')
            statcast_data.to_csv(statcast_file, index=False)
            logger.info(f"Saved Statcast data to {statcast_file}")
    except Exception as e:
        logger.error(f"Failed to fetch Statcast data: {e}")

if __name__ == "__main__":
    main()