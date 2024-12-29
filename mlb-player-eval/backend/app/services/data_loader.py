import pandas as pd
from pathlib import Path
import os
from ..models.player import Player
from ..database import SessionLocal, engine

def load_player_data(year: float, file_path: str):
    """Load player data from CSV for specific year"""
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()  # Remove duplicates from DataFrame
    print(f"Loading {len(df)} rows for year {year}")
    
    db = SessionLocal()
    try:
        # Delete existing data for this year
        db.query(Player).filter(Player.year == year).delete()
        
        for _, row in df.iterrows():
            player = Player(
                name=row['Player Name'],
                team=row['Team'],
                status=row['Status'],
                year=year,
                war=row['WAR'],
                base_value=row['Base_Value'],
                contract_value=row['Contract_Value'],
                surplus_value=row['Surplus_Value']
            )
            db.add(player)
        db.commit()
    except Exception as e:
        print(f"Error loading data: {e}")
        db.rollback()
        raise e
    finally:
        db.close()

def init_db():
    """Initialize database with all player data"""
    Player.__table__.create(bind=engine, checkfirst=True)
    
    # Get absolute path to data directory
    current_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = current_dir.parent / "data" / "generated" / "value_by_year"
    
    print(f"Looking for CSV files in: {data_dir}")
    
    # Load data for each year
    for csv_file in data_dir.glob("PLAYER_VALUES_*.csv"):
        year = float(csv_file.stem.split("_")[-1])
        print(f"Loading data from {csv_file} for year {year}")
        load_player_data(year, str(csv_file))

if __name__ == "__main__":
    init_db()