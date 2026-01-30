#######
#To reload data to local db and then server, run this script, 
#ensure db is present in /backend, then ctrl+c in backend cmd, and run:
#python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
#######

import sys
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

load_dotenv(backend_dir / '.env')
from app.models.player import Player
from app.models.prospect import Prospect
from app.database import SessionLocal, engine, Base

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
            'probable_fa_year': int(row.get('Probable_FA_Year')) if pd.notna(row.get('Probable_FA_Year')) else None,
            'earliest_fa_year': int(row.get('Earliest_FA_Year')) if pd.notna(row.get('Earliest_FA_Year')) else None,
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
            'contract_value': float(row.get('Contract_Value')) if pd.notna(row.get('Contract_Value')) else None,
            'surplus_value': float(row.get('Surplus_Value')) if pd.notna(row.get('Surplus_Value')) else None,
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
            
            # Composite metrics (Float) - top 100 rank for top prospects, None otherwise
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
        

    def load_player_data(self, players_csv: str) -> None:
        try:
            df = pd.read_csv(players_csv)
            logger.info(f"Loading {len(df)} players from {players_csv}")
            
            for _, row in df.iterrows():
                data = self.transform_player_data(row.to_dict())
                player = Player(**data)
                self.db.add(player)
                    
            self.db.commit()
            logger.info("Player data loading completed successfully")
                
        except Exception as e:
            logger.error(f"Error loading player data: {str(e)}")
            self.db.rollback()
            raise

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

    def reset_and_load_data(self, players_csv: str, prospects_csv: str = None):
        try:
            # Clear existing data - different syntax for SQLite vs PostgreSQL
            logger.info("Clearing existing data...")
            
            # Check if we're using SQLite or PostgreSQL
            db_url = os.getenv("DATABASE_URL", "")
            if db_url.startswith("sqlite"):
                # SQLite syntax
                self.db.execute(text("DELETE FROM players;"))
                self.db.execute(text("DELETE FROM prospects;"))
            else:
                # PostgreSQL syntax
                self.db.execute(text("TRUNCATE TABLE players CASCADE;"))
                self.db.execute(text("TRUNCATE TABLE prospects CASCADE;"))
                
            self.db.commit()
            logger.info("Tables cleared successfully")

            # Load fresh data
            logger.info("Loading fresh data...")
            self.load_player_data(players_csv)
            if prospects_csv:
                self.load_prospect_data(prospects_csv)
            logger.info("Data reload complete")

        except Exception as e:
            logger.error(f"Error during data reset and load: {e}")
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
        prospects_data = base_path / "data" / "generated" / "MiLB" / "prospect_histories.csv"
        
        logger.info(f"Looking for data files in: {base_path}")
        
        if not player_data.exists() or not prospects_data.exists():
            logger.error(f"Data files not found! Checked path: {base_path}")
            return
        
        db = SessionLocal()
        try:
            loader = DataLoader(db)
            loader.reset_and_load_data(str(player_data), str(prospects_data))
            logger.info("Data loading completed successfully!")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

if __name__ == "__main__":
    init_db()