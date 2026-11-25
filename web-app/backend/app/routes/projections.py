import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, desc, asc, or_
from sqlalchemy.orm import Session
from typing import Optional, Literal
import logging

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.database import get_db
from app.models.player import Player

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def get_projections(
    year: int,
    player_type: Literal["hitter", "pitcher"],
    team: Optional[str] = None,
    position: Optional[str] = None,
    projection_type: Optional[Literal["ros", "preseason"]] = "ros",
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    sort_by: Optional[str] = None,
    sort_direction: Optional[Literal["asc", "desc"]] = "desc",
    db: Session = Depends(get_db),
):
    try:
        query = db.query(Player).filter(
            Player.year == year,
            Player.projection_type == projection_type,
        )

        if player_type == "hitter":
            query = query.filter(Player.war_bat.isnot(None))
        else:
            query = query.filter(Player.war_pit.isnot(None))

        if team:
            if team.upper() == "FA":
                query = query.filter(Player.team == "FA")
            else:
                query = query.filter(func.upper(Player.team) == team.upper())

        if position:
            if position == "OF":
                query = query.filter(or_(
                    Player.position.in_(["LF", "CF", "RF"]),
                    Player.position.like("%/LF%"),
                    Player.position.like("%/CF%"),
                    Player.position.like("%/RF%"),
                    Player.position.like("LF/%"),
                    Player.position.like("CF/%"),
                    Player.position.like("RF/%"),
                ))
            else:
                query = query.filter(or_(
                    Player.position == position,
                    Player.position.like(f"{position}/%"),
                    Player.position.like(f"%/{position}"),
                ))

        total_count = query.count()

        if sort_by:
            sort_column = getattr(Player, sort_by, None)
            if sort_column is not None:
                query = query.order_by(desc(sort_column) if sort_direction == "desc" else asc(sort_column))

        offset = (page - 1) * page_size
        players = query.offset(offset).limit(page_size).all()
        total_pages = (total_count + page_size - 1) // page_size

        return {
            "total_count": total_count,
            "total_pages": total_pages,
            "page": page,
            "page_size": page_size,
            "players": [
                {
                    "fg_id": p.fg_id,
                    "mlbam_id": p.mlbam_id,
                    "name": p.name,
                    "team": p.team,
                    "position": p.position,
                    "player_role": p.player_role,
                    "status": p.status,
                    "age": p.age,
                    "fa_year": p.fa_year,
                    "probable_fa_year": p.probable_fa_year,
                    "earliest_fa_year": p.earliest_fa_year,
                    "value": {
                        "production_value": p.production_value,
                        "salary_cost": p.salary_cost,
                        "annual_surplus": p.annual_surplus,
                        "trade_value": p.trade_value,
                        "trade_value_p10": p.trade_value_p10,
                        "trade_value_p90": p.trade_value_p90,
                    },
                    **(
                        {
                            "hitting": {
                                "g_bat": p.g_bat,
                                "war_bat": p.war_bat,
                                "bb_pct_bat": p.bb_pct_bat,
                                "k_pct_bat": p.k_pct_bat,
                                "avg": p.avg,
                                "obp": p.obp,
                                "slg": p.slg,
                                "ops": p.ops,
                                "woba": p.woba,
                                "wrc_plus": p.wrc_plus,
                                "bat": p.bat,
                                "bsr": p.bsr,
                                "def_value": p.def_value,
                                "hr": p.hr,
                                "doubles": p.doubles,
                                "triples": p.triples,
                                "r": p.r,
                                "rbi": p.rbi,
                                "sb": p.sb,
                                "cs": p.cs,
                            }
                        }
                        if player_type == "hitter"
                        else {
                            "pitching": {
                                "g_pit": p.g_pit,
                                "gs": p.gs,
                                "ip": p.ip,
                                "war_pit": p.war_pit,
                                "era": p.era,
                                "fip": p.fip,
                                "k_pct_pit": p.k_pct_pit,
                                "bb_pct_pit": p.bb_pct_pit,
                                "gb_pct": p.gb_pct,
                                "fb_pct": p.fb_pct,
                                "hr_fb": p.hr_fb,
                                "hr_9": p.hr_9,
                            }
                        }
                    ),
                }
                for p in players
            ],
        }
    except Exception as e:
        logger.error(f"Error in get_projections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
