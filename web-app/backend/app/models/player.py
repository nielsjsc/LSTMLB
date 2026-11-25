import sys
from pathlib import Path
from sqlalchemy import Column, Integer, String, Float, Index

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.database import Base


class Player(Base):
    __tablename__ = "players"

    # Primary Key & Canonical IDs
    id = Column(Integer, primary_key=True, index=True)
    fg_id = Column(Integer, index=True)
    mlbam_id = Column(Integer, index=True)

    # Basic Info
    name = Column(String, index=True)
    team = Column(String, index=True)
    position = Column(String, index=True)
    player_role = Column(String, index=True)
    status = Column(String)
    age = Column(Integer)
    year = Column(Integer, index=True)
    projection_type = Column(String, default="ros", index=True)

    # Single-Season & Monte Carlo Surplus Valuation
    production_value = Column(Float)
    salary_cost = Column(Float)
    annual_surplus = Column(Float)
    trade_value = Column(Float)
    trade_value_p10 = Column(Float)
    trade_value_p50 = Column(Float)
    trade_value_p90 = Column(Float)

    # Control Window & Horizon Metrics
    fa_year = Column(Integer)
    probable_fa_year = Column(Integer)
    earliest_fa_year = Column(Integer)
    years_control = Column(Integer)
    control_through = Column(Integer)
    control_war = Column(Float)
    avg_war = Column(Float)
    control_salary_cost = Column(Float)
    avg_salary_cost = Column(Float)
    control_production_value = Column(Float)
    control_surplus = Column(Float)

    # 15-Year & Full-Career Metrics
    projected_15yr_war = Column(Float)
    projected_15yr_value = Column(Float)
    career_war = Column(Float)
    career_production_value = Column(Float)
    past_career_war = Column(Float)
    past_career_value = Column(Float)

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
    xba = Column(Float)
    xslg = Column(Float)
    xwoba = Column(Float)
    sc_ba = Column(Float)
    sc_slg = Column(Float)
    sc_woba = Column(Float)

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

    __table_args__ = (
        Index("idx_year_team", "year", "team"),
        Index("idx_year_position", "year", "position"),
        Index("idx_year_war_bat", "year", "war_bat"),
        Index("idx_year_war_pit", "year", "war_pit"),
        Index("idx_year_projection_type", "year", "projection_type"),
        Index("idx_fg_year_proj", "fg_id", "year", "projection_type"),
        Index("idx_mlbam_year_proj", "mlbam_id", "year", "projection_type"),
    )

    def __repr__(self):
        return f"<Player {self.name} (fg_id={self.fg_id})>"
