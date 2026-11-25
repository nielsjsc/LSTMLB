import sys
from pathlib import Path
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.database import Base


class Prospect(Base):
    __tablename__ = "prospects"

    id = Column(Integer, primary_key=True, index=True)
    fg_id = Column(String, index=True)
    mlbam_id = Column(Integer, index=True, nullable=True)
    name = Column(String, index=True)
    has_mlb = Column(Boolean, default=False)
    org = Column(String, index=True)
    position = Column(String)
    player_role = Column(String)
    level = Column(String, nullable=True)
    eta = Column(Integer, nullable=True)
    year = Column(Integer, index=True)
    age = Column(Float)
    fv = Column(String)

    # Rankings
    top_100 = Column(Integer, nullable=True)
    org_rank = Column(Integer, nullable=True)

    # Unified V2 Prospect Valuation & Uncertainty Quantiles
    trade_value = Column(Float, nullable=True)
    trade_value_p10 = Column(Float, nullable=True)
    trade_value_p50 = Column(Float, nullable=True)
    trade_value_p90 = Column(Float, nullable=True)
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

    values_by_year = Column(JSON, default=dict)
    composites_by_year = Column(JSON, default=dict)

    def get_value(self, year: int = None) -> float | None:
        """Return the prospect trade_value for *year*, or this record's trade_value."""
        if year is not None and year == self.year:
            return self.trade_value
        if year is not None and self.values_by_year:
            return self.values_by_year.get(str(year))
        return self.trade_value

    def get_composite(self, year: int = None) -> float | None:
        if year is not None and year == self.year:
            return self.composite
        if year is not None and self.composites_by_year:
            return self.composites_by_year.get(str(year))
        return self.composite

    @property
    def type(self) -> str:
        if self.player_role in ("SP", "RP"):
            return "pitcher"
        return "pitcher" if "p" in (self.position or "").lower() else "hitter"
