"""DB models for Minor League (MiLB) performance statistics.

Two tables — one for hitters, one for pitchers — mirroring the FanGraphs
MiLB stats CSVs.  Each row is one player-season-team-level combination.
Cross-referenced with the Prospect table via ``IDfg`` (FanGraphs player ID).
"""

from sqlalchemy import Column, Integer, Float, String, Index
from app.database import Base


class MiLBHittingStats(Base):
    __tablename__ = "milb_hitting_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    IDfg = Column(String, index=True, nullable=False)
    season = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    team = Column(String)
    level = Column(String)       # R, A, A+, AA, AAA, CPX, etc.
    age = Column(Integer)
    pa = Column(Integer)
    bb_pct = Column(Float)       # BB%
    k_pct = Column(Float)        # K%
    bb_k = Column(Float)         # BB/K
    avg = Column(Float)
    obp = Column(Float)
    slg = Column(Float)
    ops = Column(Float)
    iso = Column(Float)
    spd = Column(Float)
    babip = Column(Float)
    wsb = Column(Float)          # wSB
    wrc = Column(Float)          # wRC
    wraa = Column(Float)         # wRAA
    woba = Column(Float)         # wOBA
    wrc_plus = Column(Float)     # wRC+

    __table_args__ = (
        Index("ix_milb_hitting_idfg_season", "IDfg", "season"),
    )


class MiLBPitchingStats(Base):
    __tablename__ = "milb_pitching_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    IDfg = Column(String, index=True, nullable=False)
    season = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    team = Column(String)
    level = Column(String)
    age = Column(Integer)
    ip = Column(Float)
    k_9 = Column(Float)         # K/9
    bb_9 = Column(Float)        # BB/9
    k_bb = Column(Float)        # K/BB
    hr_9 = Column(Float)        # HR/9
    k_pct = Column(Float)       # K%
    bb_pct = Column(Float)      # BB%
    k_bb_pct = Column(Float)    # K-BB%
    avg = Column(Float)
    whip = Column(Float)
    babip = Column(Float)
    lob_pct = Column(Float)     # LOB%
    era = Column(Float)
    fip = Column(Float)
    e_f = Column(Float)         # ERA-FIP (E-F)
    xfip = Column(Float)

    __table_args__ = (
        Index("ix_milb_pitching_idfg_season", "IDfg", "season"),
    )
