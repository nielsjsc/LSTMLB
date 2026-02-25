import sys
from pathlib import Path
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, nullslast, case, or_
import logging

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Change to absolute imports
from app.database import get_db
from app.models.prospect import Prospect
from typing import Optional, List

router = APIRouter()
logger = logging.getLogger(__name__)

def map_ui_to_prospect_team(team: str) -> str:
    """Map UI team codes (2-letter) to prospect database codes (3-letter)"""
    ui_to_prospect = {
        'SF': 'SFG',
        'SD': 'SDP',
        'KC': 'KCR',
        'ATH': 'OAK',
        'TB': 'TBR'
    }
    return ui_to_prospect.get(team.upper(), team.upper())

def is_pitcher(position: str) -> bool:
    """Determine if a player is a pitcher based on position"""
    return 'p' in position.lower() if position else False
def position_in_string(position: str, position_string: str) -> bool:
    """Check if position exists as a distinct position in a position string (separated by /)"""
    positions = position_string.split('/')
    return position in positions
@router.get("/")
async def get_prospects(
    player_type: str = Query(..., description="Either 'hitter' or 'pitcher'"),
    year: int = Query(2025, ge=2022, le=2025),
    team: Optional[str] = None,
    position: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    sort_by: str = Query(None),
    sort_direction: str = Query('asc'),
    db: Session = Depends(get_db)
) -> dict:
    try:
        query = db.query(Prospect)
        
        # Filter by player type based on position
        if player_type == 'pitcher':
            query = query.filter(Prospect.position.ilike('%p%'))
        else:
            query = query.filter(~Prospect.position.ilike('%p%'))
            
        # Filter by year
        query = query.filter(Prospect.year == year)
        
        # Apply other filters
        if team:
            prospect_team = map_ui_to_prospect_team(team)
            query = query.filter(Prospect.org == prospect_team)
        if position:
            if position == 'OF':
                query = query.filter(or_(
                    Prospect.position.in_(['LF', 'CF', 'RF']),
                    Prospect.position.like('%/LF%'),
                    Prospect.position.like('%/CF%'),
                    Prospect.position.like('%/RF%'),
                    Prospect.position.like('LF/%'),
                    Prospect.position.like('CF/%'),
                    Prospect.position.like('RF/%')
                ))
            else:
                query = query.filter(or_(
                    Prospect.position == position,  # Exact match
                    Prospect.position.like(f'{position}/%'),  # Position at start
                    Prospect.position.like(f'%/{position}')  # Position at end
                ))


        # Handle sorting with proper null handling
        if sort_by:
            # Special handling for dynamic year-based columns
            if sort_by == 'value':
                value_col = getattr(Prospect, f'value_{year}')
                sort_attr = value_col
            elif sort_by == 'composite':
                composite_col = getattr(Prospect, f'composite_{year}')
                sort_attr = composite_col
            else:
                # Regular columns mapping
                sort_map = {
                    'name': Prospect.name,
                    'IDfg': Prospect.IDfg,
                    'has_mlb': Prospect.has_mlb,
                    'org': Prospect.org,
                    'position': Prospect.position,
                    'age': Prospect.age,
                    'fv': Prospect.fv,
                    'hit': Prospect.hit,
                    'game': Prospect.game_power,
                    'raw': Prospect.raw_power,
                    'speed': Prospect.speed,
                    'fastball': Prospect.fastball,
                    'slider': Prospect.slider,
                    'curve': Prospect.curve,
                    'change': Prospect.changeup,
                    'command': Prospect.command
                }
                sort_attr = sort_map.get(sort_by)

            if sort_attr is not None:
                if sort_direction == 'desc':
                    query = query.order_by(nullslast(sort_attr.desc()))
                else:
                    query = query.order_by(nullslast(sort_attr.asc()))


        # Get total count before pagination
        total_count = query.count()
        
        # Apply pagination after sorting
        prospects = query.offset((page - 1) * page_size).limit(page_size).all()

        
        
        return {
            "count": total_count,
            "page": page,
            "pages": (total_count + page_size - 1) // page_size,
            "players": [{
                "id": p.id,
                "IDfg": p.IDfg,
                "name": p.name,
                "org": p.org,
                "position": p.position,
                "age": p.age,
                "fv": p.fv,
                "has_mlb": p.has_mlb,
                "value": getattr(p, f"value_{year}", None),
                "composite": getattr(p, f"composite_{year}", None),
                # Tool grades based on player type
                **({"hit": p.hit,
                    "game": p.game_power,
                    "raw": p.raw_power,
                    "speed": p.speed} if not is_pitcher(p.position) else
                   {"fastball": p.fastball,
                    "slider": p.slider,
                    "curve": p.curve,
                    "change": p.changeup,
                    "command": p.command})
            } for p in prospects]
        }
    except Exception as e:
        logger.error(f"Error in get_prospects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{prospect_id}")
async def get_prospect_detail(
    prospect_id: int,
    db: Session = Depends(get_db)
) -> dict:
    """Get detailed prospect data across all available years."""
    try:
        # First look up the prospect by DB primary key
        anchor = db.query(Prospect).filter(Prospect.id == prospect_id).first()

        if not anchor:
            raise HTTPException(status_code=404, detail=f"Prospect not found: {prospect_id}")

        # Get all records for this prospect across years (by name match)
        records = (
            db.query(Prospect)
            .filter(Prospect.name == anchor.name)
            .order_by(Prospect.year.desc())
            .all()
        )

        latest = records[0]
        is_pitcher = 'p' in latest.position.lower() if latest.position else False

        # Build tool grades from latest record
        if is_pitcher:
            tools = {
                "fastball": latest.fastball,
                "slider": latest.slider,
                "curve": latest.curve,
                "changeup": latest.changeup,
                "command": latest.command,
            }
        else:
            tools = {
                "hit": latest.hit,
                "game_power": latest.game_power,
                "raw_power": latest.raw_power,
                "speed": latest.speed,
            }

        # Build year-by-year rankings
        history = []
        for r in records:
            entry = {
                "year": r.year,
                "age": r.age,
                "org": r.org,
                "position": r.position,
                "fv": r.fv,
                "value": getattr(r, f"value_{r.year}", None),
                "composite": getattr(r, f"composite_{r.year}", None),
            }
            # Add tool grades per year
            if is_pitcher:
                entry.update({
                    "fastball": r.fastball, "slider": r.slider,
                    "curve": r.curve, "changeup": r.changeup, "command": r.command,
                })
            else:
                entry.update({
                    "hit": r.hit, "game_power": r.game_power,
                    "raw_power": r.raw_power, "speed": r.speed,
                })
            history.append(entry)

        # Try to get MLB stats/headshot info via MLB API ID
        mlb_info = None
        if latest.has_mlb:
            # Try to look up the player in the main players table
            try:
                from app.models.player import Player
                player_record = db.query(Player).filter(
                    Player.name == latest.name,
                    Player.year == 2026
                ).first()
                if player_record and player_record.mlb_id:
                    mlb_info = {
                        "mlb_id": player_record.mlb_id,
                        "headshot_url": f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{player_record.mlb_id}/headshot/67/current",
                    }
            except Exception:
                pass

        return {
            "id": anchor.id,
            "IDfg": latest.IDfg,
            "name": latest.name,
            "org": latest.org,
            "position": latest.position,
            "age": latest.age,
            "fv": latest.fv,
            "has_mlb": latest.has_mlb,
            "is_pitcher": is_pitcher,
            "tools": tools,
            "history": history,
            "mlb_info": mlb_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_prospect_detail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))