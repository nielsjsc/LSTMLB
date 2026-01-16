import sys
from pathlib import Path
from sqlalchemy import Column, Integer, String, Float, Boolean

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.database import Base

class Prospect(Base):
    __tablename__ = "prospects"

    id = Column(Integer, primary_key=True, index=True)
    IDfg = Column(Integer, index=True)
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
    
    # Dynamic value columns

    value_2022 = Column(Float)
    value_2023 = Column(Float)
    value_2024 = Column(Float)
    value_2025 = Column(Float)
    
    # Dynamic composite columns
    composite_2022 = Column(Float)
    composite_2023 = Column(Float)
    composite_2024 = Column(Float)
    composite_2025 = Column(Float)

    # Type is now determined by position, not stored separately
    @property
    def type(self) -> str:
        return 'pitcher' if 'p' in self.position.lower() else 'hitter'