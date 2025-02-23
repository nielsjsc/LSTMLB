from sqlalchemy.orm import Session
from ..models.player import Player
from ..models.prospect import Prospect
import pandas as pd
import logging
from typing import Dict, Any
from pathlib import Path
from ..database import SessionLocal, engine, Base
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, db: Session):
        self.db = db

    def validate_player_data(self, data: Dict[str, Any]) -> bool:
        required_fields = ['name', 'team', 'position', 'year', 'real_id']
        return all(field in data for field in required_fields)

    def transform_player_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform CSV row data into correct types for database model"""
        return {
            # Integer fields
            'real_id': int(row['IDfg']) if pd.notna(row['IDfg']) else None,
            'age': int(row['Age']) if pd.notna(row['Age']) else None,
            'year': int(row['Year']) if pd.notna(row['Year']) else None,
            'years_control': int(row.get('years_control')) if pd.notna(row.get('years_control')) else None,
            'fa_year': int(row.get('FA_Year')) if pd.notna(row.get('FA_Year')) else None,
            'probable_fa_year': int(row.get('probable_fa_year')) if pd.notna(row.get('probable_fa_year')) else None,
            'earliest_fa_year': int(row.get('earliest_fa_year')) if pd.notna(row.get('earliest_fa_year')) else None,
            'control_through': int(row.get('control_through')) if pd.notna(row.get('control_through')) else None,
            
            # Integer stats
            'g_bat': int(row['G_bat']) if pd.notna(row['G_bat']) else None,
            'g_pit': int(row['G_pit']) if pd.notna(row['G_pit']) else None,
            'gs': int(row['GS']) if pd.notna(row['GS']) else None,
            'hr': int(row.get('HR')) if pd.notna(row.get('HR')) else None,
            'doubles': int(row.get('2B')) if pd.notna(row.get('2B')) else None,
            'triples': int(row.get('3B')) if pd.notna(row.get('3B')) else None,
            'r': int(row.get('R')) if pd.notna(row.get('R')) else None,
            'rbi': int(row.get('RBI')) if pd.notna(row.get('RBI')) else None,
            'sb': int(row.get('SB')) if pd.notna(row.get('SB')) else None,
            'cs': int(row.get('CS')) if pd.notna(row.get('CS')) else None,

            # String fields
            'name': str(row['Player_Name']) if pd.notna(row['Player_Name']) else None,
            'team': str(row['Team']) if pd.notna(row['Team']) else None,
            'position': str(row['Position']) if pd.notna(row['Position']) else None,
            'status': str(row.get('Status')) if pd.notna(row.get('Status')) else None,

            # Float fields - Hitting stats
            'war_bat': float(row.get('WAR_batter')) if pd.notna(row.get('WAR_batter')) else None,
            'bb_pct_bat': float(row.get('BB%_bat')) if pd.notna(row.get('BB%_bat')) else None,
            'k_pct_bat': float(row.get('K%_bat')) if pd.notna(row.get('K%_bat')) else None,
            'avg': float(row.get('AVG')) if pd.notna(row.get('AVG')) else None,
            'obp': float(row.get('OBP')) if pd.notna(row.get('OBP')) else None,
            'slg': float(row.get('SLG')) if pd.notna(row.get('SLG')) else None,
            'ops': float(row.get('OPS')) if pd.notna(row.get('OPS')) else None,
            'woba': float(row.get('wOBA')) if pd.notna(row.get('wOBA')) else None,
            'wrc_plus': float(row.get('wRC+')) if pd.notna(row.get('wRC+')) else None,
            'ev': float(row.get('EV')) if pd.notna(row.get('EV')) else None,
            'off': float(row.get('Off')) if pd.notna(row.get('Off')) else None,
            'bsr': float(row.get('BsR')) if pd.notna(row.get('BsR')) else None,
            'def_value': float(row.get('Def')) if pd.notna(row.get('Def')) else None,

            # Float fields - Pitching stats
            'war_pit': float(row.get('WAR_pitcher')) if pd.notna(row.get('WAR_pitcher')) else None,
            'era': float(row.get('ERA')) if pd.notna(row.get('ERA')) else None,
            'fip': float(row.get('FIP')) if pd.notna(row.get('FIP')) else None,
            'siera': float(row.get('SIERA')) if pd.notna(row.get('SIERA')) else None,
            'k_pct_pit': float(row.get('K%_pit')) if pd.notna(row.get('K%_pit')) else None,
            'bb_pct_pit': float(row.get('BB%_pit')) if pd.notna(row.get('BB%_pit')) else None,

            # Float fields - Value metrics
            'base_value': float(row.get('Base_Value')) if pd.notna(row.get('Base_Value')) else None,
            'contract_value': float(row.get('contract_value')) if pd.notna(row.get('contract_value')) else None,
            'surplus_value': float(row.get('surplus_value')) if pd.notna(row.get('surplus_value')) else None,
            'trade_value': float(row.get('trade_value')) if pd.notna(row.get('trade_value')) else None,
            'contract_war': float(row.get('contract_war')) if pd.notna(row.get('contract_war')) else None,
            'avg_war': float(row.get('avg_war')) if pd.notna(row.get('avg_war')) else None,
            'total_contract': float(row.get('total_contract')) if pd.notna(row.get('total_contract')) else None,
            'avg_contract': float(row.get('avg_contract')) if pd.notna(row.get('avg_contract')) else None,
            'total_future_war': float(row.get('total_future_war')) if pd.notna(row.get('total_future_war')) else None,
            'total_future_value': float(row.get('total_future_value')) if pd.notna(row.get('total_future_value')) else None,
            'total_value': float(row.get('total_value')) if pd.notna(row.get('total_value')) else None,
            'total_war': float(row.get('total_war')) if pd.notna(row.get('total_war')) else None,
            'historical_value': float(row.get('historical_value')) if pd.notna(row.get('historical_value')) else None,
            'historical_war': float(row.get('historical_war')) if pd.notna(row.get('historical_war')) else None,
            'contract_base_value': float(row.get('contract_base_value')) if pd.notna(row.get('contract_base_value')) else None,
        }
    def transform_prospect_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform CSV row data into correct types for Prospect model"""
        # Handle IDfg as Integer
        id_fg = None
        if pd.notna(row['IDfg']):
            try:
                id_fg = int(row['IDfg'])
            except ValueError:
                id_fg = None
                
        return {
            # Integer fields
            'IDfg': id_fg,
            'year': int(row['Year']) if pd.notna(row['Year']) else None,
            
            # String fields
            'name': str(row['Name']) if pd.notna(row['Name']) else None,
            'org': str(row['Team']) if pd.notna(row['Team']) else None,
            'position': str(row['Position']) if pd.notna(row['Position']) else None,
            'fv': str(row['FV']) if pd.notna(row['FV']) else None,
            
            # Boolean fields
            'has_mlb': bool(row['has_mlb']) if pd.notna(row['has_mlb']) else False,
            
            # Float fields
            'age': float(row['Age']) if pd.notna(row['Age']) else None,
            
            # Value metrics (Float)
            'value_2022': float(row.get('2022_Value')) if pd.notna(row.get('2022_Value')) else None,
            'value_2023': float(row.get('2023_Value')) if pd.notna(row.get('2023_Value')) else None,
            'value_2024': float(row.get('2024_Value')) if pd.notna(row.get('2024_Value')) else None,
            'value_2025': float(row.get('2025_Value')) if pd.notna(row.get('2025_Value')) else None,
            
            # Composite metrics (Float)
            'composite_2022': float(row.get('2022_Composite')) if pd.notna(row.get('2022_Composite')) else None,
            'composite_2023': float(row.get('2023_Composite')) if pd.notna(row.get('2023_Composite')) else None,
            'composite_2024': float(row.get('2024_Composite')) if pd.notna(row.get('2024_Composite')) else None,
            'composite_2025': float(row.get('2025_Composite')) if pd.notna(row.get('2025_Composite')) else None,
            
            # Tool grades (String)
            'hit': str(row.get('Hit')) if pd.notna(row.get('Hit')) else None,
            'game_power': str(row.get('Game')) if pd.notna(row.get('Game')) else None,
            'raw_power': str(row.get('Raw')) if pd.notna(row.get('Raw')) else None,
            'speed': str(row.get('Spd')) if pd.notna(row.get('Spd')) else None,
            'fastball': str(row.get('FB')) if pd.notna(row.get('FB')) else None,
            'slider': str(row.get('SL')) if pd.notna(row.get('SL')) else None,
            'curve': str(row.get('CB')) if pd.notna(row.get('CB')) else None,
            'changeup': str(row.get('CH')) if pd.notna(row.get('CH')) else None,
            'command': str(row.get('CMD')) if pd.notna(row.get('CMD')) else None
        }
        

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
    try:
        logger.info("Starting database initialization...")
        
        # Create tables using SQLAlchemy
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Get the base path for data files
        base_path = Path(__file__).resolve().parent.parent.parent.parent.parent
        
        # Define paths to data files
        player_data = base_path / "data" / "generated" / "value_by_year" / "player_values_complete.csv"
        prospects_data = base_path / "data" / "generated" / "MiLB" / "player_histories.csv"
        
        logger.info(f"Looking for data files in: {base_path}")
        
        if not player_data.exists() or not prospects_data.exists():
            logger.error(f"Data files not found! Checked path: {base_path}")
            return
        
        db = SessionLocal()
        try:
            loader = DataLoader(db)
            
            # Load player data
            logger.info("Loading player data...")
            loader.load_data(str(player_data))
            
            # Load prospect data
            logger.info("Loading prospect data...")
            loader.load_prospect_data(str(prospects_data))
            
            logger.info("Data loading completed successfully!")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

if __name__ == "__main__":
    init_db()