from app.database import SessionLocal
from app.models.player import Player
from app.models.prospect import Prospect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_railway_data():
    db = SessionLocal()
    try:
        player_count = db.query(Player).count()
        prospect_count = db.query(Prospect).count()
        
        logger.info(f"Database URL: {db.get_bind().url.render_as_string(hide_password=True)}")
        logger.info(f"Players in Railway database: {player_count}")
        logger.info(f"Prospects in Railway database: {prospect_count}")
        
        # Get a sample player
        sample_player = db.query(Player).first()
        logger.info(f"Sample player: {sample_player.name} ({sample_player.position})")
        
    finally:
        db.close()

if __name__ == "__main__":
    verify_railway_data()