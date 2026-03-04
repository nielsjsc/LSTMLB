import sys
from pathlib import Path
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, nullslast, case, or_, func
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
    page_size: int = Query(25, ge=1, le=1000),
    sort_by: str = Query(None),
    sort_direction: str = Query('asc'),
    slim: bool = Query(False, description="Return minimal fields (name, org, position, fv, value)"),
    view: str = Query('grades', description="View mode: grades, stats, all_stats"),
    min_pa: Optional[int] = Query(None, description="Minimum plate appearances (hitters)"),
    min_ip: Optional[float] = Query(None, description="Minimum innings pitched (pitchers)"),
    min_g: Optional[int] = Query(None, description="Minimum games"),
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
                    Prospect.position == position,
                    Prospect.position.like(f'{position}/%'),
                    Prospect.position.like(f'%/{position}')
                ))

        # Handle sorting — value & composite are now direct columns
        sort_map = {
            'name': Prospect.name,
            'IDfg': Prospect.IDfg,
            'has_mlb': Prospect.has_mlb,
            'org': Prospect.org,
            'position': Prospect.position,
            'age': Prospect.age,
            'fv': Prospect.fv,
            'value': Prospect.value,
            'composite': Prospect.composite,
            'top_100': Prospect.top_100,
            'org_rank': Prospect.org_rank,
            'hit': Prospect.hit,
            'game': Prospect.game_power,
            'raw': Prospect.raw_power,
            'speed': Prospect.speed,
            'fastball': Prospect.fastball,
            'slider': Prospect.slider,
            'curve': Prospect.curve,
            'change': Prospect.changeup,
            'command': Prospect.command,
        }

        if sort_by and sort_by in sort_map:
            sort_attr = sort_map[sort_by]
            if sort_direction == 'desc':
                query = query.order_by(nullslast(sort_attr.desc()))
            else:
                query = query.order_by(nullslast(sort_attr.asc()))

        # Get total count before pagination
        total_count = query.count()

        # Apply pagination
        prospects = query.offset((page - 1) * page_size).limit(page_size).all()

        # ── Build MiLB stats lookup if stats view requested ──────────
        stats_by_idfg = {}
        if view in ('stats', 'all_stats') and not slim:
            idfg_list = [p.IDfg for p in prospects if p.IDfg]
            if idfg_list:
                if player_type == 'hitter':
                    # Get aggregated hitting stats per prospect for the year
                    hitting_rows = (
                        db.query(MiLBHittingStats)
                        .filter(MiLBHittingStats.IDfg.in_(idfg_list))
                        .all()
                    )
                    for h in hitting_rows:
                        if h.IDfg not in stats_by_idfg:
                            stats_by_idfg[h.IDfg] = []
                        stats_by_idfg[h.IDfg].append({
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
                else:
                    pitching_rows = (
                        db.query(MiLBPitchingStats)
                        .filter(MiLBPitchingStats.IDfg.in_(idfg_list))
                        .all()
                    )
                    for pit in pitching_rows:
                        if pit.IDfg not in stats_by_idfg:
                            stats_by_idfg[pit.IDfg] = []
                        stats_by_idfg[pit.IDfg].append({
                            "season": pit.season,
                            "team": pit.team,
                            "level": pit.level,
                            "age": pit.age,
                            "ip": round(pit.ip, 1) if pit.ip is not None else None,
                            "k_9": round(pit.k_9, 2) if pit.k_9 is not None else None,
                            "bb_9": round(pit.bb_9, 2) if pit.bb_9 is not None else None,
                            "k_bb": round(pit.k_bb, 2) if pit.k_bb is not None else None,
                            "hr_9": round(pit.hr_9, 2) if pit.hr_9 is not None else None,
                            "k_pct": round(pit.k_pct * 100, 1) if pit.k_pct is not None else None,
                            "bb_pct": round(pit.bb_pct * 100, 1) if pit.bb_pct is not None else None,
                            "avg": round(pit.avg, 3) if pit.avg is not None else None,
                            "whip": round(pit.whip, 2) if pit.whip is not None else None,
                            "babip": round(pit.babip, 3) if pit.babip is not None else None,
                            "era": round(pit.era, 2) if pit.era is not None else None,
                            "fip": round(pit.fip, 2) if pit.fip is not None else None,
                            "xfip": round(pit.xfip, 2) if pit.xfip is not None else None,
                        })

        # ── Filter by min PA/IP/G if stats requested ─────────────────
        if view in ('stats', 'all_stats') and (min_pa or min_ip or min_g):
            filtered_prospects = []
            for p in prospects:
                if not p.IDfg or p.IDfg not in stats_by_idfg:
                    filtered_prospects.append(p)
                    continue
                rows = stats_by_idfg[p.IDfg]
                # Check if any season row meets the minimum threshold
                passes = False
                for row in rows:
                    if min_pa and player_type == 'hitter' and row.get('pa', 0) and row['pa'] >= min_pa:
                        passes = True
                        break
                    if min_ip and player_type == 'pitcher' and row.get('ip', 0) and row['ip'] >= min_ip:
                        passes = True
                        break
                    if min_g and row.get('g', 0) and row['g'] >= min_g:
                        passes = True
                        break
                    if not min_pa and not min_ip and not min_g:
                        passes = True
                        break
                if passes:
                    filtered_prospects.append(p)
            prospects = filtered_prospects
            total_count = len(prospects)

        # ── Build response ───────────────────────────────────────────
        def _build_latest_stats(p):
            """Get the most recent season's stats for a prospect."""
            if not p.IDfg or p.IDfg not in stats_by_idfg:
                return {}
            rows = stats_by_idfg[p.IDfg]
            if not rows:
                return {}
            # Sort by season desc, return most recent
            rows.sort(key=lambda r: r.get('season', 0), reverse=True)
            return rows[0]

        if slim:
            players_list = [{
                "name": p.name,
                "org": p.org,
                "position": p.position,
                "fv": p.fv,
                "value": p.value,
            } for p in prospects]
        else:
            players_list = []
            for p in prospects:
                entry = {
                    "id": p.id,
                    "IDfg": p.IDfg,
                    "name": p.name,
                    "org": p.org,
                    "position": p.position,
                    "age": p.age,
                    "fv": p.fv,
                    "has_mlb": p.has_mlb,
                    "value": p.value,
                    "composite": p.composite,
                    "top_100": p.top_100,
                    "org_rank": p.org_rank,
                }
                # Tool grades based on player type
                if not is_pitcher(p.position):
                    entry.update({
                        "hit": p.hit,
                        "game": p.game_power,
                        "raw": p.raw_power,
                        "speed": p.speed,
                    })
                else:
                    entry.update({
                        "fastball": p.fastball,
                        "slider": p.slider,
                        "curve": p.curve,
                        "change": p.changeup,
                        "command": p.command,
                    })
                # Add stats if requested
                if view in ('stats', 'all_stats'):
                    latest = _build_latest_stats(p)
                    entry["latest_stats"] = latest
                    if view == 'all_stats':
                        entry["all_stats"] = stats_by_idfg.get(p.IDfg, []) if p.IDfg else []
                players_list.append(entry)

        return {
            "count": total_count,
            "page": page,
            "pages": (total_count + page_size - 1) // page_size,
            "players": players_list,
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
                "value": r.value,
                "composite": r.composite,
                "top_100": r.top_100,
                "org_rank": r.org_rank,
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

        # Fallback: try resolving via crosswalk if IDfg is missing
        if not idfg and prospect.mlbam_id:
            from app.models.player_id_crosswalk import PlayerIdCrosswalk
            xref = (
                db.query(PlayerIdCrosswalk)
                .filter(PlayerIdCrosswalk.mlbam_id == prospect.mlbam_id)
                .first()
            )
            if xref and xref.fg_id:
                idfg = xref.fg_id

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
    idfg: str,
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