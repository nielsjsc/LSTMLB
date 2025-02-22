from sqlalchemy.orm import Session
from ..models.player import Player
from ..models.prospect import Prospect
import pandas as pd
import logging
from typing import Dict, Any
from pathlib import Path
from ..database import SessionLocal, engine
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
            'status': row.get('Status'),
            'age': row['Age'],
            'year': row['Year'],
            # Hitting stats - match CSV case
            'war_bat': row.get('WAR_batter'),  # Changed from war_bat
            'bb_pct_bat': row.get('BB%_bat'),  # Changed from bb_pct_bat
            'k_pct_bat': row.get('K%_bat'),    # Changed from k_pct_bat
            'g_bat': row['G_bat'],      # Changed from G_bat
            'avg': row.get('AVG'),             # Changed from avg
            'obp': row.get('OBP'),             # Changed from obp
            'slg': row.get('SLG'),             # Changed from slg
            'ops': row.get('OPS'),             # Changed from ops
            'woba': row.get('wOBA'),           # Changed from woba
            'wrc_plus': row.get('wRC+'),       # Changed from wrc_plus
            'ev': row.get('EV'),               # Changed from ev
            'off': row.get('Off'),             # Changed from off
            'bsr': row.get('BsR'),             # Changed from bsr
            'def_value': row.get('Def'),         # Changed from def_val

            # Value metrics
            
            # Pitching stats - match CSV case
            'war_pit': row.get('WAR_pitcher'),  # Changed from war_pit
            'g_pit': row['G_pit'],      # Changed from G_pit
            'gs': row['GS'],        # Changed from GS
            'era': row.get('ERA'),              # Changed from era
            'fip': row.get('FIP'),              # Changed from fip
            'siera': row.get('SIERA'),          # Changed from siera
            'k_pct_pit': row.get('K%_pit'),     # Changed from k_pct_pit
            'bb_pct_pit': row.get('BB%_pit'),   # Changed from bb_pct_pit

            # Value metrics
            'base_value': row.get('Base_Value'),
            'contract_value': row.get('contract_value'),
            'surplus_value': row.get('surplus_value'),
            'trade_value': row.get('trade_value'),
            'fa_year': row.get('FA_Year'),
            'probable_fa_year': row.get('probable_fa_year'),
            'earliest_fa_year': row.get('earliest_fa_year'),
            'contract_war': row.get('contract_war'),
            'avg_war': row.get('avg_war'),
            'avg_contract': row.get('avg_contract'),
            'years_control': row.get('years_control'),
            'control_through': row.get('control_through'),
            'total_future_war': row.get('total_future_war'),
            'total_future_value': row.get('total_future_value'),
            'total_value': row.get('total_value'),
            'total_war': row.get('total_war'),
            'historical_value': row.get('historical_value'),
            'historical_war': row.get('historical_war'),
            'total_contract': row.get('total_contract'),
            'contract_base_value': row.get('contract_base_value'),


            # Additional stats
            'hr': row.get('HR'),
            'doubles': row.get('2B'),
            'triples': row.get('3B'),
            'r': row.get('R'),
            'rbi': row.get('RBI'),
            'sb': row.get('SB'),
            'cs': row.get('CS'),
        }
    def transform_prospect_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Determine if pitcher based on position
        is_pitcher = 'p' in str(row['Position']).lower() if row['Position'] else False
        
        # Only include IDfg if it's a valid integer
        id_fg = None
        if pd.notna(row['IDfg']):
            try:
                id_fg = int(row['IDfg'])
            except ValueError:
                id_fg = None
        
        base_data = {
            'IDfg': id_fg,  # Will be None for string IDs or NaN values
            'name': row['Name'],
            'org': row['Team'],
            'position': row['Position'],
            'year': row['Year'],
            'age': row['Age'],
            'fv': row['FV'],
            'has_mlb': row['has_mlb'],
            
            # Values and composites for all years
            'value_2022': row.get('2022_Value'),
            'value_2023': row.get('2023_Value'),
            'value_2024': row.get('2024_Value'),
            'value_2025': row.get('2025_Value'),
            
            'composite_2022': row.get('2022_Composite'),
            'composite_2023': row.get('2023_Composite'),
            'composite_2024': row.get('2024_Composite'),
            'composite_2025': row.get('2025_Composite'),

            # Tool grades based on player type
            'hit': None if is_pitcher else row.get('Hit'),
            'game_power': None if is_pitcher else row.get('Game'),
            'raw_power': None if is_pitcher else row.get('Raw'),
            'speed': None if is_pitcher else row.get('Spd'),
            'fastball': row.get('FB') if is_pitcher else None,
            'slider': row.get('SL') if is_pitcher else None,
            'curve': row.get('CB') if is_pitcher else None,
            'changeup': row.get('CH') if is_pitcher else None,
            'command': row.get('CMD') if is_pitcher else None
        }
        
        return base_data

    def load_prospect_data(self, prospects_csv: str) -> None:
        try:
            df = pd.read_csv(prospects_csv)
            logger.info(f"Loading {len(df)} prospects from {prospects_csv}")
            
            for _, row in df.iterrows():
                data = self.transform_prospect_data(row.to_dict())
                prospect = Prospect(**data)
                self.db.add(prospect)
                    
            self.db.commit()
            logger.info("Prospect data loading completed successfully")
                
        except Exception as e:
            logger.error(f"Error loading prospect data: {str(e)}")
            self.db.rollback()
            raise



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


def init_db():
    """Initialize database with player and prospect data"""
    Player.__table__.create(bind=engine, checkfirst=True)
    Prospect.__table__.create(bind=engine, checkfirst=True)
    
    current_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    player_data = current_dir / "data" / "generated" / "value_by_year" / "player_values_complete.csv"
    prospects_data = current_dir / "data" / "generated" / "MiLB" / "player_histories.csv"
    
    db = SessionLocal()
    try:
        loader = DataLoader(db)
        loader.load_data(str(player_data))
        loader.load_prospect_data(str(prospects_data))
        print("Data loading completed successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_db()