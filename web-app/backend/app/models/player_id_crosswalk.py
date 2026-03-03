"""Player ID Crosswalk — maps MLBAM IDs ↔ FanGraphs IDs.

Populated from ``data/generated/player_id_crosswalk.csv`` (built by
``scrapers/build_id_crosswalk.py``) and used at startup to resolve
``IDfg`` for prospects and any other cross-system lookups.
"""

from sqlalchemy import Column, Integer, String, Index

import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.database import Base


class PlayerIdCrosswalk(Base):
    __tablename__ = "player_id_crosswalk"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlbam_id = Column(Integer, nullable=False, index=True, unique=True)
    fg_id = Column(String, index=True, nullable=True)
    name = Column(String, nullable=True)
    source = Column(String, nullable=True)  # 'mlb_api'

    __table_args__ = (
        Index("ix_crosswalk_mlbam_fg", "mlbam_id", "fg_id"),
    )
