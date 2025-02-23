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

def is_pitcher(position: str) -> bool:
    """Determine if a player is a pitcher based on position"""
    return 'p' in position.lower() if position else False
def position_in_string(position: str, position_string: str) -> bool:
    """Check if position exists as a distinct position in a position string (separated by /)"""
    positions = position_string.split('/')
    return position in positions
@router.get("/prospects")
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
            query = query.filter(Prospect.org == team.upper())
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