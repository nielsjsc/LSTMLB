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
    name = Column(String)
    has_mlb = Column(Boolean, default=False)
    org = Column(String)
    position = Column(String)
    year = Column(Integer)
    age = Column(Float)
    fv = Column(String)
    
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
    
    # Year-keyed value & composite maps — no schema change needed when a new
    # season is added.  Stored as ``{"2022": 1.5, "2023": 2.0, ...}``.
    values_by_year = Column(JSON, default=dict)
    composites_by_year = Column(JSON, default=dict)

    # ── Helpers ──────────────────────────────────────────────────────────
    def get_value(self, year: int) -> float | None:
        """Return the prospect value for *year*, or ``None``."""
        if not self.values_by_year:
            return None
        return self.values_by_year.get(str(year))

    def get_composite(self, year: int) -> float | None:
        """Return the composite ranking for *year*, or ``None``."""
        if not self.composites_by_year:
            return None
        return self.composites_by_year.get(str(year))

    # Type is now determined by position, not stored separately
    @property
    def type(self) -> str:
        return 'pitcher' if 'p' in self.position.lower() else 'hitter'