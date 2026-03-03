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
from app.models.milb_stats import MiLBHittingStats, MiLBPitchingStats
from app.config import PROSPECT_YEARS, PROSPECT_DEFAULT_YEAR, PROSPECT_YEAR_START, PROSPECT_YEAR_END
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
    year: int = Query(PROSPECT_DEFAULT_YEAR, ge=PROSPECT_YEAR_START, le=PROSPECT_YEAR_END),
    team: Optional[str] = None,
    position: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=500),
    sort_by: str = Query(None),
    sort_direction: str = Query('asc'),
    slim: bool = Query(False, description="Return minimal fields (name, org, position, fv, value)"),
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
            # Special handling for dynamic year-based columns (now JSON)
            if sort_by in ('value', 'composite'):
                # JSON column sorting: extract the year key from the JSON dict.
                # SQLAlchemy can sort on a Python-side expression via case(),
                # but for JSON extraction we fall back to fetching all rows and
                # sorting in-memory (fine for ~500 prospects per year).
                pass  # handled below after fetch
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

        # If sorting by a JSON-derived value/composite, fetch all then sort in Python
        if sort_by in ('value', 'composite'):
            all_prospects = query.all()
            year_key = str(year)
            reverse = sort_direction == 'desc'

            def _json_sort_key(p: Prospect) -> float:
                blob = p.values_by_year if sort_by == 'value' else p.composites_by_year
                val = (blob or {}).get(year_key)
                # None → sort to end regardless of direction
                if val is None:
                    return float('-inf') if reverse else float('inf')
                return val

            all_prospects.sort(key=_json_sort_key, reverse=reverse)
            prospects = all_prospects[(page - 1) * page_size: (page - 1) * page_size + page_size]
        else:
            # Apply pagination after sorting
            prospects = query.offset((page - 1) * page_size).limit(page_size).all()

        
        
        return {
            "count": total_count,
            "page": page,
            "pages": (total_count + page_size - 1) // page_size,
            "players": [{
                "name": p.name,
                "org": p.org,
                "position": p.position,
                "fv": p.fv,
                "value": p.get_value(year),
            } for p in prospects] if slim else [{
                "id": p.id,
                "IDfg": p.IDfg,
                "name": p.name,
                "org": p.org,
                "position": p.position,
                "age": p.age,
                "fv": p.fv,
                "has_mlb": p.has_mlb,
                "value": p.get_value(year),
                "composite": p.get_composite(year),
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
                "value": r.get_value(r.year),
                "composite": r.get_composite(r.year),
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
                from app.config import CURRENT_YEAR
                player_record = db.query(Player).filter(
                    Player.name == latest.name,
                    Player.year == CURRENT_YEAR
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


@router.get("/{prospect_id}/milb-stats")
async def get_prospect_milb_stats(
    prospect_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Return all MiLB hitting & pitching stats for a prospect.

    Looks up the prospect by DB primary key to get their FanGraphs ID
    (``IDfg``), then queries the MiLB stats tables.  Results are grouped
    by season and sorted most-recent-first.
    """
    try:
        prospect = db.query(Prospect).filter(Prospect.id == prospect_id).first()
        if not prospect:
            raise HTTPException(status_code=404, detail="Prospect not found")

        idfg = prospect.IDfg
        if not idfg:
            return {"hitting": [], "pitching": []}

        # ── Hitting stats ─────────────────────────────────────────────────
        hitting_rows = (
            db.query(MiLBHittingStats)
            .filter(MiLBHittingStats.IDfg == idfg)
            .order_by(MiLBHittingStats.season.desc(), MiLBHittingStats.level)
            .all()
        )

        hitting = []
        for h in hitting_rows:
            hitting.append({
                "season": h.season,
                "team": h.team,
                "level": h.level,
                "age": h.age,
                "pa": h.pa,
                "bb_pct": round(h.bb_pct * 100, 1) if h.bb_pct is not None else None,
                "k_pct": round(h.k_pct * 100, 1) if h.k_pct is not None else None,
                "avg": round(h.avg, 3) if h.avg is not None else None,
                "obp": round(h.obp, 3) if h.obp is not None else None,
                "slg": round(h.slg, 3) if h.slg is not None else None,
                "ops": round(h.ops, 3) if h.ops is not None else None,
                "iso": round(h.iso, 3) if h.iso is not None else None,
                "babip": round(h.babip, 3) if h.babip is not None else None,
                "woba": round(h.woba, 3) if h.woba is not None else None,
                "wrc_plus": round(h.wrc_plus, 1) if h.wrc_plus is not None else None,
                "spd": round(h.spd, 1) if h.spd is not None else None,
            })

        # ── Pitching stats ────────────────────────────────────────────────
        pitching_rows = (
            db.query(MiLBPitchingStats)
            .filter(MiLBPitchingStats.IDfg == idfg)
            .order_by(MiLBPitchingStats.season.desc(), MiLBPitchingStats.level)
            .all()
        )

        pitching = []
        for p in pitching_rows:
            pitching.append({
                "season": p.season,
                "team": p.team,
                "level": p.level,
                "age": p.age,
                "ip": round(p.ip, 1) if p.ip is not None else None,
                "k_9": round(p.k_9, 2) if p.k_9 is not None else None,
                "bb_9": round(p.bb_9, 2) if p.bb_9 is not None else None,
                "k_bb": round(p.k_bb, 2) if p.k_bb is not None else None,
                "hr_9": round(p.hr_9, 2) if p.hr_9 is not None else None,
                "k_pct": round(p.k_pct * 100, 1) if p.k_pct is not None else None,
                "bb_pct": round(p.bb_pct * 100, 1) if p.bb_pct is not None else None,
                "avg": round(p.avg, 3) if p.avg is not None else None,
                "whip": round(p.whip, 2) if p.whip is not None else None,
                "babip": round(p.babip, 3) if p.babip is not None else None,
                "era": round(p.era, 2) if p.era is not None else None,
                "fip": round(p.fip, 2) if p.fip is not None else None,
                "xfip": round(p.xfip, 2) if p.xfip is not None else None,
            })

        return {"hitting": hitting, "pitching": pitching}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_prospect_milb_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-idfg/{idfg}")
async def get_prospect_by_idfg(
    idfg: int,
    db: Session = Depends(get_db),
) -> dict:
    """Look up a prospect's DB primary key by FanGraphs ID.

    Useful for linking trade players (who have mlb_id / IDfg) to their
    prospect detail pages.
    """
    prospect = (
        db.query(Prospect)
        .filter(Prospect.IDfg == idfg)
        .order_by(Prospect.year.desc())
        .first()
    )
    if not prospect:
        raise HTTPException(status_code=404, detail="Prospect not found")
    return {"id": prospect.id, "name": prospect.name}