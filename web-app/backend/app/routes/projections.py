import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, desc, asc, or_
from sqlalchemy.orm import Session
from typing import Optional, Literal
import logging

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Change to absolute imports
from app.database import get_db
from app.models.player import Player

logger = logging.getLogger(__name__)
router = APIRouter()


def position_in_string(position: str, position_string: str) -> bool:
    """Check if position exists as a distinct position in a position string (separated by /)"""
    positions = position_string.split('/')
    return position in positions
@router.get("/")
async def get_projections(
    year: int,
    player_type: Literal["hitter", "pitcher"],
    team: Optional[str] = None,
    position: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    sort_by: Optional[str] = None,
    sort_direction: Optional[Literal["asc", "desc"]] = "desc",
    db: Session = Depends(get_db)
):
    try:
        query = db.query(Player)
        
        # Apply all filters first
        query = query.filter(Player.year == year)
        
        if player_type == "hitter":
            query = query.filter(Player.war_bat.isnot(None))
        else:
            query = query.filter(Player.war_pit.isnot(None))
            
        if team:
            if team.upper() == 'FA':
                query = query.filter(Player.team == 'FA')
            else:
                query = query.filter(func.upper(Player.team) == team.upper())
        
        if position:
            if position == 'OF':
                query = query.filter(or_(
                    Player.position.in_(['LF', 'CF', 'RF']),
                    Player.position.like('%/LF%'),
                    Player.position.like('%/CF%'),
                    Player.position.like('%/RF%'),
                    Player.position.like('LF/%'),
                    Player.position.like('CF/%'),
                    Player.position.like('RF/%')
                ))
            else:
                query = query.filter(or_(
                    Player.position == position,  # Exact match
                    Player.position.like(f'{position}/%'),  # Position at start
                    Player.position.like(f'%/{position}')  # Position at end
                ))

        # Get total count before pagination
        total_count = query.count()
        
        # Apply sorting
        if sort_by:
            sort_column = getattr(Player, sort_by, None)
            if sort_column is not None:
                query = query.order_by(desc(sort_column) if sort_direction == "desc" else asc(sort_column))
        
        # Apply pagination last
        offset = (page - 1) * page_size
        players = query.offset(offset).limit(page_size).all()
        
        # Calculate total pages
        total_pages = (total_count + page_size - 1) // page_size


        
        


        return {
            "total_count": total_count,
            "total_pages": total_pages,
            "page": page,
            "page_size": page_size,
            "players": [{
                "real_id": p.real_id,
                "mlb_id": p.mlb_id,
                "name": p.name,
                "team": p.team,
                "position": p.position,
                "status": p.status,
                "age": p.age,
                "fa_year": p.fa_year,
                "probable_fa_year": p.probable_fa_year,
                "earliest_fa_year": p.earliest_fa_year,
                "value": {
                    "base_value": p.base_value,
                    "contract_value": p.contract_value,
                    "surplus_value": p.surplus_value,
                    "trade_value": p.trade_value
                },
                **({"hitting": {
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
                    "off": p.off,
                    "bsr": p.bsr,
                    "def_value": p.def_value,
                    "hr": p.hr,
                    "doubles": p.doubles,
                    "triples": p.triples,
                    "r": p.r,
                    "rbi": p.rbi,
                    "sb": p.sb,
                    "cs": p.cs
                }} if player_type == "hitter" else {"pitching": {
                    "g_pit": p.g_pit,
                    "gs": p.gs,
                    "war_pit": p.war_pit,
                    "era": p.era,
                    "fip": p.fip,
                    "siera": p.siera,
                    "k_pct_pit": p.k_pct_pit,
                    "bb_pct_pit": p.bb_pct_pit
                }})
            } for p in players]
        }
    except Exception as e:
        logger.error(f"Error in get_projections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))