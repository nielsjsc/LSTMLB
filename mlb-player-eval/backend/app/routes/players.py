from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from typing import Optional
from ..database import get_db
from ..models.player import Player
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/players")
async def get_players(
    year: int = 2025,
    team: Optional[str] = None,
    position: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "war",
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Getting players: year={year}, team={team}, position={position}, search={search}")
        
        # Base query
        query = db.query(Player).filter(Player.year == year)
        
        # Apply filters
        if search:
            query = query.filter(Player.name.ilike(f"%{search}%"))
        if team:
            query = query.filter(Player.team == team.upper())
        if position:
            query = query.filter(
                or_(
                    Player.position.ilike(f"%{position}%"),
                    Player.status == position
                )
            )
            
        # Apply sorting
        if sort_by == "war":
            # Combined WAR sorting
            query = query.order_by(
                func.coalesce(Player.war_bat, 0) + 
                func.coalesce(Player.war_pit, 0).desc()
            )
        elif sort_by == "value":
            query = query.order_by(Player.surplus_value.desc())
            
        players = query.all()
        logger.info(f"Found {len(players)} players")
        
        return {
            "count": len(players),
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "team": p.team,
                    "position": p.position,
                    "status": p.status,
                    "age": p.age,
                    "war_bat": p.war_bat,
                    "war_pit": p.war_pit,
                    "value": {
                        "base_value": p.base_value,
                        "contract_value": p.contract_value,
                        "surplus_value": p.surplus_value
                    }
                } for p in players
            ]
        }
    except Exception as e:
        logger.error(f"Error in get_players: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/players/{player_id}/details")
async def get_player_details(player_id: int, db: Session = Depends(get_db)):
    try:
        # First get player to find real_id
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            raise HTTPException(status_code=404, detail="Player not found")
            
        # Get all years using real_id
        player_years = (
            db.query(Player)
            .filter(Player.real_id == player.real_id)  # Use real_id instead of id
            .order_by(Player.year)
            .all()
        )
        
        logger.info(f"Found {len(player_years)} years for player {player_id} (real_id: {player.real_id})")
        
        if not player_years:
            raise HTTPException(status_code=404, detail="Player not found")
            
        year_2025 = next((p for p in player_years if p.year == 2025), player_years[0])
        
        
        response = {
            "name": year_2025.name,
            "team": year_2025.team,
            "position": year_2025.position,
            "projections": [{
                "year": p.year,
                "age": p.age,
                "team": p.team,
                "position": p.position,  # Add position to each projection
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
                    "def_value": p.def_val,
                    "hr": p.hr,
                    "doubles": p.doubles,
                    "triples": p.triples,
                    "r": p.r,
                    "rbi": p.rbi,
                    "sb": p.sb,
                    "cs": p.cs
                }} if p.war_bat is not None else {}),
                **({"pitching": {
                    "war_pit": p.war_pit,
                    "era": p.era,
                    "fip": p.fip,
                    "siera": p.siera,
                    "k_pct_pit": p.k_pct_pit,
                    "bb_pct_pit": p.bb_pct_pit
                }} if p.war_pit is not None else {})
            } for p in player_years]  # Include all years
        }
        return response
    except Exception as e:
        logger.error(f"Error in get_player_details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))