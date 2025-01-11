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
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        query = db.query(Player).filter(Player.year == year)
        
        if search:
            query = query.filter(Player.name.ilike(f"%{search}%"))
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
                "id": p.id,
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

@router.get("/players/{player_id}/details")
async def get_player_details(
    player_id: int,
    db: Session = Depends(get_db)
):
    try:
        print(f"Fetching details for player ID: {player_id}")  # Debug log
        player_years = db.query(Player).filter(
            Player.id == player_id
        ).order_by(Player.year).all()
        
        print(f"Found {len(player_years)} years for player ID {player_id}")  # Debug log
        
        if not player_years:
            raise HTTPException(status_code=404, detail="Player not found")
        
        response = {
            "name": player_years[0].name,
            "team": player_years[0].team,
            "position": player_years[0].position,
            "projections": [{
                "year": p.year,
                "war": p.war,
                "age": p.age,
                "value": {
                    "base": p.base_value,
                    "contract": p.contract_value,
                    "surplus": p.surplus_value
                },
                **({"pitching": {
                    "fip": p.fip,
                    "siera": p.siera,
                    "k_pct": p.k_pct_pit,
                    "bb_pct": p.bb_pct_pit,
                    "gb_pct": p.gb_pct,
                    "fb_pct": p.fb_pct,
                    "stuff_plus": p.stuff_plus,
                    "location_plus": p.location_plus,
                    "pitching_plus": p.pitching_plus,
                    "fbv": p.fbv
                }} if p.position in ['SP', 'RP'] else {"hitting": {
                    "bb_pct": p.bb_pct_bat,
                    "k_pct": p.k_pct_bat,
                    "avg": p.avg,
                    "obp": p.obp,
                    "slg": p.slg,
                    "woba": p.woba,
                    "wrc_plus": p.wrc_plus,
                    "ev": p.ev,
                    "off": p.off,
                    "bsr": p.bsr,
                    "def": p.def_value
                }})
            } for p in player_years]
        }
        print(f"Response: {response}")  # Debug log
        return response
    except Exception as e:
        print(f"Error in get_player_details: {str(e)}")
        raise