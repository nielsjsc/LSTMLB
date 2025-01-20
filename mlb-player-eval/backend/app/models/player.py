from sqlalchemy import Column, Integer, String, Float, Index, ForeignKey
from sqlalchemy.orm import relationship
from ..database import Base

class Player(Base):
    __tablename__ = "players"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    real_id = Column(Integer, index=True)
    
    # Basic Info
    name = Column(String, index=True)
    team = Column(String, index=True)
    position = Column(String, index=True)
    status = Column(String)
    age = Column(Integer)
    year = Column(Integer, index=True)
    
    # Contract/Value Info
    base_value = Column(Float)
    contract_value = Column(Float)
    surplus_value = Column(Float)
    fa_year = Column(Integer)
    probable_fa_year = Column(Integer)
    earliest_fa_year = Column(Integer)
    
    # Hitting Stats
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
    off = Column(Float)
    bsr = Column(Float)
    def_val = Column(Float)
    hr = Column(Integer)
    doubles = Column(Integer)
    triples = Column(Integer)
    r = Column(Integer)
    rbi = Column(Integer)
    sb = Column(Integer)
    cs = Column(Integer)
    
    # Pitching Stats
    war_pit = Column(Float)
    era = Column(Float)
    fip = Column(Float)
    siera = Column(Float)
    k_pct_pit = Column(Float)
    bb_pct_pit = Column(Float)
    
    # Composite Indices
    __table_args__ = (
        Index('idx_year_team', 'year', 'team'),
        Index('idx_year_position', 'year', 'position'),
        Index('idx_year_war_bat', 'year', 'war_bat'),
        Index('idx_year_war_pit', 'year', 'war_pit'),
    )

    def __repr__(self):
        return f"<Player {self.name}>"