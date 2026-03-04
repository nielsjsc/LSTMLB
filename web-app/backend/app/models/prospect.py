import sys
from pathlib import Path
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.database import Base

class Prospect(Base):
    __tablename__ = "prospects"

    id = Column(Integer, primary_key=True, index=True)
    IDfg = Column(String, index=True)
    mlbam_id = Column(Integer, index=True, nullable=True)
    name = Column(String, index=True)
    has_mlb = Column(Boolean, default=False)
    org = Column(String)
    position = Column(String)
    year = Column(Integer, index=True)
    age = Column(Float)
    fv = Column(String)

    # Rankings
    top_100 = Column(Integer, nullable=True)    # MLB-wide top-100 rank
    org_rank = Column(Integer, nullable=True)    # Organization rank

    # Per-row value & composite (one record per player-year)
    value = Column(Float, nullable=True)
    composite = Column(Float, nullable=True)

    # Tool grades
    hit = Column(String)
    game_power = Column(String)
    raw_power = Column(String)
    speed = Column(String)
    fastball = Column(String)
    slider = Column(String)
    curve = Column(String)
    changeup = Column(String)
    command = Column(String)

    # Legacy JSON columns — kept for backward compat with trade routes
    values_by_year = Column(JSON, default=dict)
    composites_by_year = Column(JSON, default=dict)

    # ── Helpers ──────────────────────────────────────────────────────────
    def get_value(self, year: int = None) -> float | None:
        """Return the prospect value for *year*, or this record's value."""
        if year is not None and year == self.year:
            return self.value
        if year is not None and self.values_by_year:
            return self.values_by_year.get(str(year))
        return self.value

    def get_composite(self, year: int = None) -> float | None:
        """Return the composite ranking for *year*, or this record's composite."""
        if year is not None and year == self.year:
            return self.composite
        if year is not None and self.composites_by_year:
            return self.composites_by_year.get(str(year))
        return self.composite

    # Type is now determined by position, not stored separately
    @property
    def type(self) -> str:
        return 'pitcher' if 'p' in self.position.lower() else 'hitter'