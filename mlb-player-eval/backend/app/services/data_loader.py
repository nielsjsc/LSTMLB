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
                id=row['IDfg'],
                team=row['Team'],
                age=row['Age'],
                status=row['Status'],
                year=year,
                position=row['Position'],
                war=row['WAR'],
                war_bat=row.get('WAR_batter'),
                war_pit=row.get('WAR_pitcher'),
                base_value=row['Base_Value'],
                contract_value=row['Contract_Value'],
                surplus_value=row['Surplus_Value'],
                bb_pct_bat=row.get('BB%_bat'),
                bb_pct_pit=row.get('BB%_pit'),
                k_pct_bat=row.get('K%_bat'),
                k_pct_pit=row.get('K%_pit'),
                avg=row.get('AVG'),
                obp=row.get('OBP'),
                slg=row.get('SLG'),
                woba=row.get('wOBA'),
                wrc_plus=row.get('wRC+'),
                ev=row.get('EV'),
                off=row.get('Off'),
                bsr=row.get('BsR'),
                def_value=row.get('Def'),
                fip=row.get('FIP'),
                siera=row.get('SIERA'),
                gb_pct=row.get('GB%'),
                fb_pct=row.get('FB%'),
                stuff_plus=row.get('Stuff+'),
                location_plus=row.get('Location+'),
                pitching_plus=row.get('Pitching+'),
                fbv=row.get('FBv')
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