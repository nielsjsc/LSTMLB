"""DB model for historical MLB players (1950–present).

Replaces the 34 MB in-memory JSON cache with a proper SQL table.
Career-level columns are indexed for search; per-season data is stored
as JSON arrays so the API response is returned directly without transformation.
"""

from sqlalchemy import Column, Integer, Float, String, Boolean, JSON, Text
from app.database import Base


class HistoricalPlayer(Base):
    __tablename__ = "historical_players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    idfg = Column(Integer, unique=True, index=True, nullable=False)
    mlbam = Column(Integer, index=True, nullable=True)
    bbref = Column(String, nullable=True)
    name = Column(String, nullable=False)
    name_lower = Column(String, index=True, nullable=False)
    birth_year = Column(Integer, nullable=True)
    death_year = Column(Integer, nullable=True)
    first_year = Column(Integer, nullable=True, index=True)
    last_year = Column(Integer, nullable=True, index=True)
    teams = Column(JSON, default=list)
    career_war = Column(Float, default=0, index=True)
    career_bat_war = Column(Float, default=0)
    career_pit_war = Column(Float, default=0)
    career_salary = Column(Float, default=0)
    career_war_value = Column(Float, default=0)
    career_surplus = Column(Float, default=0)
    is_pitcher = Column(Boolean, default=False)

    # Per-season data stored as JSON arrays (same structure the API returns)
    batting = Column(JSON, default=list)
    pitching = Column(JSON, default=list)

    def __repr__(self) -> str:
        return f"<HistoricalPlayer {self.name} (idfg={self.idfg})>"

    def to_search_dict(self) -> dict:
        """Lightweight dict for the search endpoint (no season arrays)."""
        return {
            "idfg": self.idfg,
            "mlbam": self.mlbam,
            "name": self.name,
            "name_lower": self.name_lower,
            "teams": self.teams or [],
            "first_year": self.first_year,
            "last_year": self.last_year,
            "career_war": self.career_war or 0,
            "is_pitcher": self.is_pitcher,
        }

    def to_full_dict(self) -> dict:
        """Complete dict matching the original JSON structure for the detail endpoint."""
        return {
            "idfg": self.idfg,
            "mlbam": self.mlbam,
            "bbref": self.bbref,
            "name": self.name,
            "birth_year": self.birth_year,
            "death_year": self.death_year,
            "first_year": self.first_year,
            "last_year": self.last_year,
            "teams": self.teams or [],
            "career_war": self.career_war or 0,
            "career_bat_war": self.career_bat_war or 0,
            "career_pit_war": self.career_pit_war or 0,
            "career_salary": self.career_salary or 0,
            "career_war_value": self.career_war_value or 0,
            "career_surplus": self.career_surplus or 0,
            "is_pitcher": self.is_pitcher,
            "batting": self.batting or [],
            "pitching": self.pitching or [],
        }
