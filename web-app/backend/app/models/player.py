import sys
from pathlib import Path
from sqlalchemy import Column, Integer, String, Float, Index, ForeignKey
from sqlalchemy.orm import relationship

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.database import Base

class Player(Base):
    __tablename__ = "players"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    real_id = Column(Integer, index=True)
    mlb_id = Column(Integer, index=True)
    
    # Basic Info
    name = Column(String, index=True)
    team = Column(String, index=True)
    position = Column(String, index=True)
    status = Column(String)
    age = Column(Integer)
    year = Column(Integer, index=True)
    projection_type = Column(String, default='ros', index=True)
    
    # Contract/Value Info
    base_value = Column(Float)
    contract_value = Column(Float)
    surplus_value = Column(Float)
    trade_value = Column(Float)
    fa_year = Column(Integer)
    probable_fa_year = Column(Integer)
    earliest_fa_year = Column(Integer)
    contract_war = Column(Float)
    avg_war = Column(Float)
    total_contract = Column(Float)
    avg_contract = Column(Float)
    years_control = Column(Integer)
    control_through = Column(Integer)
    total_future_war = Column(Float)
    total_future_value = Column(Float)
    total_value = Column(Float)
    total_war = Column(Float)
    historical_value = Column(Float)
    historical_war = Column(Float)
    contract_base_value = Column(Float)
    
    
    # Hitting Stats
    g_bat = Column(Integer)
    war_bat = Column(Float)
    bb_pct_bat = Column(Float)
    k_pct_bat = Column(Float)
    avg = Column(Float)
    obp = Column(Float)
    slg = Column(Float)
    ops = Column(Float)
    woba = Column(Float)
    wrc_plus = Column(Float)
    ev = Column(Float)
    bat = Column(Float)
    off = Column(Float)
    bsr = Column(Float)
    def_value = Column(Float)
    hr = Column(Integer)
    doubles = Column(Integer)
    triples = Column(Integer)
    r = Column(Integer)
    rbi = Column(Integer)
    sb = Column(Integer)
    cs = Column(Integer)
    
    # Pitching Stats
    g_pit = Column(Integer)
    gs = Column(Integer)
    ip = Column(Float)
    war_pit = Column(Float)
    era = Column(Float)
    fip = Column(Float)
    k_pct_pit = Column(Float)
    bb_pct_pit = Column(Float)
    gb_pct = Column(Float)
    fb_pct = Column(Float)
    hr_fb = Column(Float)
    hr_9 = Column(Float)
    
    # Composite Indices
    __table_args__ = (
        Index('idx_year_team', 'year', 'team'),
        Index('idx_year_position', 'year', 'position'),
        Index('idx_year_war_bat', 'year', 'war_bat'),
        Index('idx_year_war_pit', 'year', 'war_pit'),
        Index('idx_year_projection_type', 'year', 'projection_type'),
    )

    def __repr__(self):
        return f"<Player {self.name}>"