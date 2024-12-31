from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional, List
from ..database import get_db
from ..models.player import Player

router = APIRouter()

@router.get("/players")
async def get_players(
    year: float = 2025,
    team: Optional[str] = None,
    position: Optional[str] = None,
    sort_by: Optional[str] = "war",
    db: Session = Depends(get_db)
):
    try:
        query = db.query(Player).filter(Player.year == year)
        
        if team:
            query = query.filter(Player.team == team.lower())
        
        if position:
            query = query.filter(Player.status == position)
        
        # Sorting
        if sort_by == "war":
            query = query.order_by(Player.war.desc())
        elif sort_by == "value":
            query = query.order_by(Player.surplus_value.desc())
        
        players = query.all()
        print(f"Found {len(players)} players for year {year}")
        
        return {
            "count": len(players),
            "players": [{
                "name": p.name,
                "team": p.team,
                "position": p.status,
                "war": p.war,
                "base_value": p.base_value,
                "contract_value": p.contract_value,
                "surplus_value": p.surplus_value
            } for p in players]
        }
    except Exception as e:
        print(f"Error in get_players: {str(e)}")
        raise

@router.get("/players/{player_name}")
async def get_player(player_name: str, db: Session = Depends(get_db)):
    player = db.query(Player).filter(Player.name == player_name).first()
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return player