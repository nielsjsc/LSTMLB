import sys
from math import isnan
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from typing import List, Optional
from pydantic import BaseModel
import logging

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Change relative imports to absolute
from app.database import get_db
from app.models.player import Player
from app.models.prospect import Prospect

# Initialize router and logger
router = APIRouter()
logger = logging.getLogger(__name__)



def get_player_values(player_name: str, db: Session):
    # Get only the 2025 player entry since it has all the values we need
    base_player = (
        db.query(Player)
        .filter(Player.name == player_name)
        .filter(Player.year == 2025)
        .first()
    )
    
    if not base_player:
        raise HTTPException(status_code=404, detail=f"Player {player_name} not found")
    
    return {
        "name": base_player.name,
        "team": base_player.team,
        "position": base_player.position,
        "war": base_player.contract_war or 0,  # Changed from calculated war
        "total_surplus": base_player.trade_value or 0,  # Already correct
        "total_contract": base_player.total_contract or 0,  # Changed from calculated total_contract
        "total_production": base_player.contract_base_value or 0,  # Changed from base_value
        "years": [year for year in range(2025, (base_player.control_through or 2025) + 1)]  # Using control_through
    }




class TradeAsset(BaseModel):
    name: str
    isProspect: bool
    team: str

class TradeRequest(BaseModel):
    team1_assets: List[TradeAsset]
    team2_assets: List[TradeAsset]

def get_prospect_values(prospect_name: str, db: Session):
    prospect = (
        db.query(Prospect)
        .filter(Prospect.name == prospect_name)
        .filter(Prospect.year == 2025)  # Add this filter
        .first()
    )
    
    if not prospect:
        raise HTTPException(status_code=404, detail=f"Prospect {prospect_name} not found")
    
    value = getattr(prospect, 'value_2025', 0) or 0
    

    
    return {
        "name": prospect.name,
        "team": prospect.org,
        "position": prospect.position,
        "fv": prospect.fv,
        "value": value,
        "total_surplus": value,
        "total_contract": 0,
        "total_production": value
    }

@router.post("/analyze")
def analyze_trade(trade: TradeRequest, db: Session = Depends(get_db)):
    try:
        # Process team 1 assets
        team1_analysis = []
        for asset in trade.team1_assets:
            if asset.isProspect:
                analysis = get_prospect_values(asset.name, db)
            else:
                analysis = get_player_values(asset.name, db)
            team1_analysis.append(analysis)

        # Process team 2 assets
        team2_analysis = []
        for asset in trade.team2_assets:
            if asset.isProspect:
                analysis = get_prospect_values(asset.name, db)
            else:
                analysis = get_player_values(asset.name, db)
            team2_analysis.append(analysis)
        
        # Safe summation function
        def safe_sum(items, key):
            return sum(item[key] or 0 for item in items)
        
        return {
            "team1": {
                "total_surplus": safe_sum(team1_analysis, "total_surplus"),
                "total_contract": safe_sum(team1_analysis, "total_contract"),
                "total_production": safe_sum(team1_analysis, "total_production"),
                "assets": team1_analysis
            },
            "team2": {
                "total_surplus": safe_sum(team2_analysis, "total_surplus"),
                "total_contract": safe_sum(team2_analysis, "total_contract"),
                "total_production": safe_sum(team2_analysis, "total_production"),
                "assets": team2_analysis
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prospects")
def get_all_prospects(
    player_type: str = Query(..., description="Either 'hitter' or 'pitcher'"),
    year: int = Query(2025, ge=2022, le=2025),
    db: Session = Depends(get_db)
):
    try:
        # Start with base query and add year filter
        query = db.query(Prospect).filter(Prospect.year == year)
        
        # Filter by player type based on position
        if player_type == 'pitcher':
            query = query.filter(Prospect.position.ilike('%p%'))
        else:
            query = query.filter(~Prospect.position.ilike('%p%'))
            
        prospects = query.all()
        
        response_data = []
        for p in prospects:

            
            response_data.append({
                "name": p.name,
                "org": p.org,
                "position": p.position,
                "fv": p.fv,
                "value": getattr(p, f"value_2025", None)
            })
        
        return {"players": response_data}

    except Exception as e:
        logger.error(f"Error in get_all_prospects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Add this new endpoint after existing endpoints
@router.get("/trade-val-rank")
def get_trade_value_rankings(
    db: Session = Depends(get_db),
    team: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    sort_by: str = Query(
        "trade_value",
        regex="^(trade_value|contract_war|avg_war|total_contract|avg_contract|control_through|years_control|total_future_war|total_future_value|historical_war|historical_value|contract_base_value)$"
    ),
    sort_direction: str = Query("desc", regex="^(asc|desc)$")
):
    try:
        # Start with 2025 players
        query = db.query(Player).filter(Player.year == 2025)
        
        # Add filter for non-NaN trade values
        query = query.filter(Player.trade_value.isnot(None))  # Filter out NULL values
        query = query.filter(Player.trade_value != float('nan'))  # Filter out NaN values
        
        # Add team filter if specified
        if team:
            if team.lower() == 'fa':
                query = query.filter(Player.team == 'FA')
            else:
                query = query.filter(Player.team == team.lower())
        
        
        sort_column = getattr(Player, sort_by)
        query = query.order_by(desc(sort_column) if sort_direction == "desc" else asc(sort_column))
        
        total_count = query.count()
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        players = query.all()
        results = [{
            "real_id": p.real_id,
            "name": p.name,
            "team": p.team,
            "position": p.position,
            "contract_war": p.contract_war,
            "avg_war": p.avg_war,
            "total_contract": p.total_contract,
            "avg_contract": p.avg_contract,
            "trade_value": p.trade_value,
            "control_through": p.control_through,
            "years_control": p.years_control,
            "total_future_war": p.total_future_war,
            "total_future_value": p.total_future_value,
            "historical_war": p.historical_war,
            "historical_value": p.historical_value,
            "contract_base_value": p.contract_base_value,
        } for p in players]
        
        return {
            "players": results,
            "total_count": total_count,
            "total_pages": (total_count + page_size - 1) // page_size,
            "current_page": page
        }
        
    except Exception as e:
        logger.error(f"Error getting trade value rankings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))