"""DB model for Statcast expected stats (batters + pitchers combined)."""

from sqlalchemy import Column, Integer, Float, Index
from app.database import Base


class StatcastExpected(Base):
    __tablename__ = "statcast_expected"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False, index=True)   # mlbam_id
    year = Column(Integer, nullable=False)
    # Batter stats (null for pitcher-only rows)
    xba = Column(Float, nullable=True)
    xslg = Column(Float, nullable=True)
    xwoba = Column(Float, nullable=True)
    # Pitcher stats (null for batter-only rows)
    xera = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_sc_pid_year", "player_id", "year", unique=True),
    )
