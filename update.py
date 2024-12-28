import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_data_collection():
    """Run all data collection scripts"""
    try:
        # Run MLB stats collection
        from scrape.MLB_data_script import main as mlb_scraper
        # Run salary data collection  
        from scrape.sportracScraper import main as salary_scraper
        # Run prospect data collection
        from scrape.fangraphs_MiLB_scraper import main as prospect_scraper
        
        mlb_scraper()
        salary_scraper()
        prospect_scraper()
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        raise

def run_data_processing():
    """Run data cleaning and matching"""
    try:
        from scrape.matchProspect import main as match_prospects
        from scrape.clean_prospect_train_data import main as clean_prospects
        
        match_prospects()
        clean_prospects()
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise

def run_model_training():
    """Run all model training and prediction scripts"""
    try:
        from models.MiLB import main as train_milb
        from models.batter import main as train_batter
        from models.fielder import main as train_fielder
        
        # Train models
        train_milb()
        train_batter() 
        train_fielder()
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

def main():
    """Main orchestration function"""
    start_time = datetime.now()
    logger.info(f"Starting MLBTradeSim update at {start_time}")
    
    try:
        # Run pipeline steps
        run_data_collection()
        run_data_processing()
        run_model_training()
        
        logger.info(f"Successfully completed update in {datetime.now() - start_time}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()