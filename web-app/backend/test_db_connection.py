from app.database import SessionLocal, DATABASE_URL
from app.models.player import Player
from app.models.prospect import Prospect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_local_data():
    db = SessionLocal()
    try:
        logger.info(f"Database URL: {DATABASE_URL}")
        
        player_count = db.query(Player).count()
        prospect_count = db.query(Prospect).count()
        
        logger.info(f"Players in local database: {player_count}")
        logger.info(f"Prospects in local database: {prospect_count}")
        
        if player_count > 0:
            # Get a sample player
            sample_player = db.query(Player).first()
            logger.info(f"Sample player: {sample_player.name} ({sample_player.position})")
        else:
            logger.info("No players found - database is empty")
        
    except Exception as e:
        logger.error(f"Error querying database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    verify_local_data()