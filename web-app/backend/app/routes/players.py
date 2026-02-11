import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from typing import Optional
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

def normalize_team_abbreviation(team: str) -> str:
    """Normalize team abbreviations to match prospect data format (3-letter codes)"""
    team_mapping = {
        'SF': 'SFG',
        'SD': 'SDP',
        'KC': 'KCR',
        'ATH': 'OAK',
        'TB': 'TBR',
        # Also handle reverse mapping for queries
        'SFG': 'SFG',
        'SDP': 'SDP',
        'KCR': 'KCR',
        'OAK': 'OAK',
        'TBR': 'TBR'
    }
    normalized = team_mapping.get(team.upper(), team.upper())
    return normalized

@router.get("/")
async def get_players(
    year: int = 2026,
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
            # Convert search string for SQL LIKE comparison
            normalized_search = search.lower().replace('.', '').replace(' ', '')
            # Use SQL function to normalize player names for comparison
            normalized_name = func.replace(
                func.replace(
                    func.lower(Player.name),
                    '.', ''
                ),
                ' ', ''
            )
            query = query.filter(normalized_name.like(f"%{normalized_search}%"))
            
        if team:
            # Normalize team abbreviation and check both formats
            normalized_team = normalize_team_abbreviation(team)
            # Handle historical variations: check if player.team matches either format
            # E.g., for KCR, accept both 'KC' and 'KCR' in database
            alt_teams = []
            if normalized_team == 'KCR':
                alt_teams = ['KC', 'KCR']
            elif normalized_team == 'OAK':
                alt_teams = ['ATH', 'OAK']
            elif normalized_team == 'TBR':
                alt_teams = ['TB', 'TBR']
            elif normalized_team == 'SDP':
                alt_teams = ['SD', 'SDP']
            elif normalized_team == 'SFG':
                alt_teams = ['SF', 'SFG']
            else:
                alt_teams = [normalized_team]
            query = query.filter(Player.team.in_(alt_teams))
            
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
                    "real_id": p.real_id,
                    "mlb_id": p.mlb_id,
                    "name": p.name,
                    "team": normalize_team_abbreviation(p.team) if p.team else p.team,
                    "position": p.position,
                    "status": p.status,
                    "age": p.age,
                    "war_bat": p.war_bat,
                    "war_pit": p.war_pit,
                    "value": {
                        "base_value": p.base_value,
                        "contract_value": p.contract_value,
                        "surplus_value": p.surplus_value,
                        "trade_value": p.trade_value
                    }
                } for p in players
            ]
        }
    except Exception as e:
        logger.error(f"Error in get_players: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{player_id}/details")
async def get_player_details(player_id: int, db: Session = Depends(get_db)):
    logger.info(f"Received request for player_id: {player_id}")  # Debug log
    
    try:
        # Try lookup by mlb_id first, then fall back to real_id (IDfg)
        logger.info(f"Trying mlb_id lookup for: {player_id}")
        query = db.query(Player).filter(Player.mlb_id == player_id)
        player_years = query.order_by(Player.year).all()
        
        if not player_years:
            logger.info(f"No mlb_id match, trying real_id lookup for: {player_id}")
            query = db.query(Player).filter(Player.real_id == player_id)
            player_years = query.order_by(Player.year).all()
        
        logger.info(f"Found {len(player_years)} years for player with ID: {player_id}")
        
        if not player_years:
            raise HTTPException(
                status_code=404, 
                detail=f"Player not found with ID: {player_id}"
            )
        
        # Get current year data (2026) or fallback to first available
        current_year_data = next((p for p in player_years if p.year == 2026), player_years[0])
        
        response = {
            "name": current_year_data.name,
            "team": normalize_team_abbreviation(current_year_data.team) if current_year_data.team else current_year_data.team,
            "position": current_year_data.position,
            "mlb_id": current_year_data.mlb_id,
            "projections": [{
                "year": p.year,
                "age": p.age,
                "team": normalize_team_abbreviation(p.team) if p.team else p.team,
                "position": p.position,
                "status": p.status,
                "fa_year": p.fa_year,
                "probable_fa_year": p.probable_fa_year,
                "earliest_fa_year": p.earliest_fa_year,
                "value": {
                    "base_value": p.base_value,
                    "contract_value": p.contract_value,
                    "surplus_value": p.surplus_value,
                    "trade_value": p.trade_value,
                    "contract_war": p.contract_war,
                    "avg_war": p.avg_war,
                    "total_contract": p.total_contract,
                    "avg_contract": p.avg_contract,
                    "years_control": p.years_control,
                    "control_through": p.control_through,
                    "total_future_war": p.total_future_war,
                    "total_future_value": p.total_future_value,
                    "total_war": p.total_war,
                    "total_value": p.total_value,
                    "historical_war": p.historical_war,
                    "historical_value": p.historical_value,
                    "contract_base_value": p.contract_base_value,
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
                }} if p.war_bat is not None else {}),
                **({"pitching": {
                    "g_pit": p.g_pit,
                    "gs": p.gs,
                    "war_pit": p.war_pit,
                    "era": p.era,
                    "fip": p.fip,
                    "siera": p.siera,
                    "k_pct_pit": p.k_pct_pit,
                    "bb_pct_pit": p.bb_pct_pit
                }} if p.war_pit is not None else {})
            } for p in player_years]
        }
        return response
        
    except Exception as e:
        logger.error(f"Error in get_player_details: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")