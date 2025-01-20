from math import isnan
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from ..database import get_db
from ..models.player import Player

class TradeRequest(BaseModel):
    team1_players: List[str]
    team2_players: List[str]

router = APIRouter(
    prefix="/trades",
    tags=["trades"]
)

def get_player_values(player_name: str, db: Session):  # Add db parameter
    player_years = (
        db.query(Player)
        .filter(Player.name == player_name)
        .distinct(Player.year)
        .order_by(Player.year)
        .all()
    )
    
    if not player_years:
        raise HTTPException(status_code=404, detail=f"Player {player_name} not found")
    
    
    def safe_sum(values):
        return sum(v for v in values if v is not None and not isnan(v))
    
    valid_years = [p for p in player_years if p.contract_value is not None and not isnan(p.contract_value)]
    
    total_surplus = safe_sum(p.surplus_value for p in valid_years)
    total_contract = safe_sum(p.contract_value for p in valid_years)
    total_production = safe_sum(p.base_value for p in valid_years)
    
    return {
        "name": player_name,
        "team": valid_years[0].team,
        "status": valid_years[0].status,
        "total_surplus": total_surplus,
        "total_contract": total_contract,
        "total_production": total_production,
        "yearly_projections": [
            {
                "year": p.year,
                "war": p.war_bat if p.war_bat is not None else (p.war_pit if p.war_pit is not None else 0),
                "base_value": p.base_value or 0,
                "contract_value": p.contract_value or 0,
                "surplus_value": p.surplus_value or 0,
                "status": p.status
            } for p in valid_years
        ]
    }

@router.post("/analyze")
def analyze_trade(trade: TradeRequest, db: Session = Depends(get_db)):
    try:
        # Pass db to get_player_values
        team1_analysis = [get_player_values(name, db) for name in trade.team1_players]
        team2_analysis = [get_player_values(name, db) for name in trade.team2_players]
        
        return {
            "team1": {
                "total_surplus": sum(p["total_surplus"] for p in team1_analysis),
                "total_contract": sum(p["total_contract"] for p in team1_analysis),
                "total_production": sum(p["total_production"] for p in team1_analysis),
                "players": team1_analysis
            },
            "team2": {
                "total_surplus": sum(p["total_surplus"] for p in team2_analysis),
                "total_contract": sum(p["total_contract"] for p in team2_analysis), 
                "total_production": sum(p["total_production"] for p in team2_analysis),
                "players": team2_analysis
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))