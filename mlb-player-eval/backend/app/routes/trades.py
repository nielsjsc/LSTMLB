from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from ..database import get_db
from ..models.player import Player

class TradeRequest(BaseModel):
    team1_players: List[str]
    team2_players: List[str]

router = APIRouter()

@router.post("/analyze")
def analyze_trade(
    trade: TradeRequest,
    db: Session = Depends(get_db)
):
    def get_player_values(player_name: str):
    # Use distinct to prevent duplicates
        player_years = (
            db.query(Player)
            .filter(Player.name == player_name)
            .distinct(Player.year)  # Add distinct clause
            .order_by(Player.year)
            .all()
        )
        
        if not player_years:
            raise HTTPException(status_code=404, detail=f"Player {player_name} not found")
        
        return {
            "name": player_name,
            "team": player_years[0].team,
            "status": player_years[0].status,
            "total_surplus": sum(p.surplus_value for p in player_years),
            "total_contract": sum(p.contract_value for p in player_years),
            "yearly_projections": [
                {
                    "year": p.year,
                    "war": p.war,
                    "surplus_value": p.surplus_value,
                    "contract_value": p.contract_value,
                    "status": p.status
                } for p in player_years
            ],
            "years_remaining": len(player_years)
        }

    team1_analysis = [get_player_values(name) for name in trade.team1_players]
    team2_analysis = [get_player_values(name) for name in trade.team2_players]

    return {
        "team1": {
            "total_value": sum(p["total_surplus"] for p in team1_analysis),
            "players": team1_analysis
        },
        "team2": {
            "total_value": sum(p["total_surplus"] for p in team2_analysis),
            "players": team2_analysis
        }
    }