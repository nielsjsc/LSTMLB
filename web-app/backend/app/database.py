import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Get DATABASE_URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./longball_local.db")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with different settings for SQLite vs PostgreSQL
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL debugging
    )
else:
    # PostgreSQL settings
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,
        pool_recycle=300
    )

try:
    # Test connection with text() wrapper
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        if DATABASE_URL.startswith("sqlite"):
            logger.info("Successfully connected to local SQLite database")
        else:
            logger.info("Successfully connected to PostgreSQL database")
except Exception as e:
    logger.error(f"Failed to connect to database: {e}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for getting database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        logger.info("Successfully connected to the database")
        db.close()
        return True
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        return False