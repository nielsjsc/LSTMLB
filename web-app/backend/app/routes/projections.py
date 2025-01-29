from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Optional, Literal
from ..database import get_db
from ..models.player import Player
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/projections")
async def get_projections(
    year: int,
    player_type: Literal["hitter", "pitcher"],
    team: Optional[str] = None,
    position: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        query = db.query(Player)
        
        # Year filter
        query = query.filter(Player.year == year)
        
        # Player type filter
        if player_type == "hitter":
            query = query.filter(Player.war_bat.isnot(None))
        else:
            query = query.filter(Player.war_pit.isnot(None))
            
        print(f"Filtering with team: {team}, position: {position}")
        print(f"Query before filters: {query.count()}")
        if team:
            if team.lower() == 'fa':
                query = query.filter(Player.team == 'FA')
            else:
                query = query.filter(Player.team == team.lower())
            print(f"After team filter: {query.count()}")
        if position:
            query = query.filter(Player.position == position)
            print(f"After position filter: {query.count()}")
            
        players = query.all()


        return {
            "count": len(players),
            "players": [{
                "id": p.id,
                "name": p.name,
                "team": p.team,
                "position": p.position,
                "age": p.age,
                "fa_year": p.fa_year,
                "probable_fa_year": p.probable_fa_year,
                "earliest_fa_year": p.earliest_fa_year,
                "value": {
                    "base_value": p.base_value,
                    "contract_value": p.contract_value,
                    "surplus_value": p.surplus_value
                },
                **({"hitting": {
                    "war_bat": p.war_bat,
                    "bb_pct_bat": p.bb_pct_bat,
                    "k_pct_bat": p.k_pct_bat,
                    "avg": p.avg,
                    "obp": p.obp,
                    "slg": p.slg,
                    "ops": p.ops,
                    "woba": p.woba,
                    "wrc_plus": p.wrc_plus,
                    "ev": p.ev,
                    "off": p.off,
                    "bsr": p.bsr,
                    "def_val": p.def_val,
                    "hr": p.hr,
                    "doubles": p.doubles,
                    "triples": p.triples,
                    "r": p.r,
                    "rbi": p.rbi,
                    "sb": p.sb,
                    "cs": p.cs
                }} if player_type == "hitter" else {"pitching": {
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