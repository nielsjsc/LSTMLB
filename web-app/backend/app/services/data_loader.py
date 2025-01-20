from sqlalchemy.orm import Session
from ..models.player import Player
import pandas as pd
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, db: Session):
        self.db = db

    def validate_player_data(self, data: Dict[str, Any]) -> bool:
        required_fields = ['name', 'team', 'position', 'year', 'real_id']
        return all(field in data for field in required_fields)

    def transform_player_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'real_id': row['IDfg'],
            'name': row['Player_Name'],
            'team': row['Team'],
            'position': row['Position'], 
            'status': row.get('Status', ''),
            'age': row['Age'],
            'year': row['Year'],
            # Hitting stats - match CSV case
            'war_bat': row.get('WAR_batter'),  # Changed from war_bat
            'bb_pct_bat': row.get('BB%_bat'),  # Changed from bb_pct_bat
            'k_pct_bat': row.get('K%_bat'),    # Changed from k_pct_bat
            'avg': row.get('AVG'),             # Changed from avg
            'obp': row.get('OBP'),             # Changed from obp
            'slg': row.get('SLG'),             # Changed from slg
            'ops': row.get('OPS'),             # Changed from ops
            'woba': row.get('wOBA'),           # Changed from woba
            'wrc_plus': row.get('wRC+'),       # Changed from wrc_plus
            'ev': row.get('EV'),               # Changed from ev
            'off': row.get('Off'),             # Changed from off
            'bsr': row.get('BsR'),             # Changed from bsr
            'def_val': row.get('Def'),         # Changed from def_val
            
            # Pitching stats - match CSV case
            'war_pit': row.get('WAR_pitcher'),  # Changed from war_pit
            'era': row.get('ERA'),              # Changed from era
            'fip': row.get('FIP'),              # Changed from fip
            'siera': row.get('SIERA'),          # Changed from siera
            'k_pct_pit': row.get('K%_pit'),     # Changed from k_pct_pit
            'bb_pct_pit': row.get('BB%_pit'),   # Changed from bb_pct_pit

            # Value metrics
            'base_value': row.get('Base_Value'),
            'contract_value': row.get('contract_value'),
            'surplus_value': row.get('surplus_value'),
            'fa_year': row.get('FA_Year'),
            'probable_fa_year': row.get('probable_fa_year'),
            'earliest_fa_year': row.get('earliest_fa_year'),

            # Additional stats
            'hr': row.get('HR'),
            'doubles': row.get('2B'),
            'triples': row.get('3B'),
            'r': row.get('R'),
            'rbi': row.get('RBI'),
            'sb': row.get('SB'),
            'cs': row.get('CS'),
        }

    def load_data(self, csv_path: str) -> None:
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} players from {csv_path}")
            
            # Change real_ID to IDfg to match CSV column name
            for _, group in df.groupby('IDfg'):
                for _, row in group.iterrows():
                    data = self.transform_player_data(row.to_dict())
                    if self.validate_player_data(data):
                        player = Player(**data)
                        self.db.add(player)
                    else:
                        logger.warning(f"Skipping invalid player data: {row['Player_Name']}")

            self.db.commit()
            logger.info("Data loading completed successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.db.rollback()
            raise

from pathlib import Path
from ..database import SessionLocal, engine

def init_db():
    """Initialize database with player data"""
    Player.__table__.create(bind=engine, checkfirst=True)
    
    # Construct path to data file
    current_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    data_file = current_dir / "data" / "generated"/"value_by_year"/"player_values_complete.csv"
    
    print(f"Loading data from: {data_file}")
    
    # Create DB session and loader
    db = SessionLocal()
    try:
        loader = DataLoader(db)
        loader.load_data(str(data_file))
        print("Data loading completed successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_db()