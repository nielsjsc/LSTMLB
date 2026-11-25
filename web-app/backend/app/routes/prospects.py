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


def is_pitcher(position: str) -> bool:
    """Determine if a player is a pitcher based on position."""
    return "p" in position.lower() if position else False


@router.get("/")
async def get_prospects(
    player_type: str = Query(..., description="Either 'hitter' or 'pitcher'"),
    year: int = Query(PROSPECT_DEFAULT_YEAR, ge=PROSPECT_YEAR_START, le=PROSPECT_YEAR_END),
    team: Optional[str] = None,
    position: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=1000),
    sort_by: str = Query(None),
    sort_direction: str = Query("asc"),
    slim: bool = Query(False, description="Return pre-debut Trade Simulator fields only"),
    view: str = Query("grades", description="View mode: grades, stats, all_stats"),
    min_pa: Optional[int] = Query(None, ge=0),
    min_ip: Optional[float] = Query(None, ge=0),
    min_g: Optional[int] = Query(None, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    try:
        query = db.query(Prospect)

        if player_type == "pitcher":
            query = query.filter(Prospect.position.ilike("%p%"))
        else:
            query = query.filter(~Prospect.position.ilike("%p%"))

        query = query.filter(Prospect.year == year)

        # Hard MLB Debut Boundary: Trade Simulator (slim=true) only includes
        # pre-debut prospects (0 MLB games) so debuted rookies never appear twice.
        if slim:
            query = query.filter(Prospect.has_mlb == False)  # noqa: E712

        if team:
            query = query.filter(func.upper(Prospect.org) == team.upper())
        if position:
            if position == "OF":
                query = query.filter(or_(
                    Prospect.position.in_(["LF", "CF", "RF"]),
                    Prospect.position.like("%/LF%"),
                    Prospect.position.like("%/CF%"),
                    Prospect.position.like("%/RF%"),
                    Prospect.position.like("LF/%"),
                    Prospect.position.like("CF/%"),
                    Prospect.position.like("RF/%"),
                ))
            else:
                query = query.filter(or_(
                    Prospect.position == position,
                    Prospect.position.like(f"{position}/%"),
                    Prospect.position.like(f"%/{position}"),
                ))

        sort_map = {
            "name": Prospect.name,
            "fg_id": Prospect.fg_id,
            "has_mlb": Prospect.has_mlb,
            "org": Prospect.org,
            "position": Prospect.position,
            "level": Prospect.level,
            "eta": Prospect.eta,
            "age": Prospect.age,
            "fv": Prospect.fv,
            "trade_value": Prospect.trade_value,
            "trade_value_p10": Prospect.trade_value_p10,
            "trade_value_p90": Prospect.trade_value_p90,
            "composite": Prospect.composite,
            "top_100": Prospect.top_100,
            "org_rank": Prospect.org_rank,
            "hit": Prospect.hit,
            "game": Prospect.game_power,
            "raw": Prospect.raw_power,
            "speed": Prospect.speed,
            "fastball": Prospect.fastball,
            "slider": Prospect.slider,
            "curve": Prospect.curve,
            "change": Prospect.changeup,
            "command": Prospect.command,
        }

        if sort_by and sort_by in sort_map:
            sort_attr = sort_map[sort_by]
            if sort_direction == "desc":
                query = query.order_by(nullslast(sort_attr.desc()))
            else:
                query = query.order_by(nullslast(sort_attr.asc()))
        else:
            query = query.order_by(nullslast(Prospect.trade_value.desc()))

        total_count = query.count()
        prospects = query.offset((page - 1) * page_size).limit(page_size).all()

        stats_by_fg = {}
        if view in ("stats", "all_stats") and not slim:
            fg_list = [p.fg_id for p in prospects if p.fg_id]
            if fg_list:
                if player_type == "hitter":
                    hitting_rows = (
                        db.query(MiLBHittingStats)
                        .filter(MiLBHittingStats.IDfg.in_(fg_list))
                        .all()
                    )
                    for h in hitting_rows:
                        stats_by_fg.setdefault(h.IDfg, []).append({
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
                        .filter(MiLBPitchingStats.IDfg.in_(fg_list))
                        .all()
                    )
                    for pit in pitching_rows:
                        stats_by_fg.setdefault(pit.IDfg, []).append({
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

        has_stat_filter = (
            (min_pa is not None and min_pa > 0)
            or (min_ip is not None and min_ip > 0)
            or (min_g is not None and min_g > 0)
        )
        if view in ("stats", "all_stats") and has_stat_filter:
            filtered_prospects = []
            for p in prospects:
                if not p.fg_id or p.fg_id not in stats_by_fg:
                    continue
                rows = stats_by_fg[p.fg_id]
                passes = False
                for row in rows:
                    if min_pa and min_pa > 0 and player_type == "hitter":
                        if (row.get("pa") or 0) >= min_pa:
                            passes = True
                            break
                    elif min_ip and min_ip > 0 and player_type == "pitcher":
                        if (row.get("ip") or 0) >= min_ip:
                            passes = True
                            break
                    elif min_g and min_g > 0:
                        if (row.get("g") or 0) >= min_g:
                            passes = True
                            break
                if passes:
                    filtered_prospects.append(p)
            prospects = filtered_prospects
            total_count = len(prospects)

        def _build_latest_stats(p):
            if not p.fg_id or p.fg_id not in stats_by_fg:
                return {}
            rows = stats_by_fg[p.fg_id]
            if not rows:
                return {}
            rows.sort(key=lambda r: r.get("season", 0), reverse=True)
            return rows[0]

        if slim:
            players_list = [
                {
                    "id": p.id,
                    "fg_id": p.fg_id,
                    "mlbam_id": p.mlbam_id,
                    "name": p.name,
                    "org": p.org,
                    "position": p.position,
                    "level": p.level,
                    "eta": p.eta,
                    "fv": p.fv,
                    "trade_value": p.trade_value,
                    "trade_value_p10": p.trade_value_p10,
                    "trade_value_p90": p.trade_value_p90,
                }
                for p in prospects
            ]
        else:
            players_list = []
            for p in prospects:
                entry = {
                    "id": p.id,
                    "fg_id": p.fg_id,
                    "mlbam_id": p.mlbam_id,
                    "name": p.name,
                    "org": p.org,
                    "position": p.position,
                    "player_role": p.player_role,
                    "level": p.level,
                    "eta": p.eta,
                    "age": p.age,
                    "fv": p.fv,
                    "has_mlb": p.has_mlb,
                    "trade_value": p.trade_value,
                    "trade_value_p10": p.trade_value_p10,
                    "trade_value_p50": p.trade_value_p50,
                    "trade_value_p90": p.trade_value_p90,
                    "composite": p.composite,
                    "top_100": p.top_100,
                    "org_rank": p.org_rank,
                }
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
                if view in ("stats", "all_stats"):
                    entry["latest_stats"] = _build_latest_stats(p)
                    if view == "all_stats":
                        entry["all_stats"] = stats_by_fg.get(p.fg_id, []) if p.fg_id else []
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
                "level": r.level,
                "eta": r.eta,
                "fv": r.fv,
                "trade_value": r.trade_value,
                "trade_value_p10": r.trade_value_p10,
                "trade_value_p90": r.trade_value_p90,
                "composite": r.composite,
                "top_100": r.top_100,
                "org_rank": r.org_rank,
            }
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

        mlb_info = None
        mlbam_id = latest.mlbam_id
        if mlbam_id:
            mlb_info = {
                "mlbam_id": mlbam_id,
                "headshot_url": f"https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam_id}/headshot/67/current",
            }
            if latest.has_mlb:
                try:
                    from app.models.player import Player
                    from app.config import CURRENT_YEAR
                    player_record = db.query(Player).filter(
                        Player.mlbam_id == mlbam_id,
                        Player.year == CURRENT_YEAR,
                        Player.projection_type == "ros",
                    ).first()
                    if player_record and player_record.mlbam_id:
                        mlb_info["mlbam_id"] = player_record.mlbam_id
                except Exception:
                    pass

        return {
            "id": anchor.id,
            "fg_id": latest.fg_id,
            "mlbam_id": mlbam_id,
            "name": latest.name,
            "org": latest.org,
            "position": latest.position,
            "level": latest.level,
            "eta": latest.eta,
            "age": latest.age,
            "fv": latest.fv,
            "trade_value": latest.trade_value,
            "trade_value_p10": latest.trade_value_p10,
            "trade_value_p90": latest.trade_value_p90,
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
    try:
        prospect = db.query(Prospect).filter(Prospect.id == prospect_id).first()
        if not prospect:
            raise HTTPException(status_code=404, detail="Prospect not found")

        fg_id = prospect.fg_id
        if not fg_id and prospect.mlbam_id:
            from app.models.player_id_crosswalk import PlayerIdCrosswalk
            xref = (
                db.query(PlayerIdCrosswalk)
                .filter(PlayerIdCrosswalk.mlbam_id == prospect.mlbam_id)
                .first()
            )
            if xref and xref.fg_id:
                fg_id = xref.fg_id

        if not fg_id:
            return {"hitting": [], "pitching": []}

        hitting_rows = (
            db.query(MiLBHittingStats)
            .filter(MiLBHittingStats.IDfg == fg_id)
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

        pitching_rows = (
            db.query(MiLBPitchingStats)
            .filter(MiLBPitchingStats.IDfg == fg_id)
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


@router.get("/by-idfg/{fg_id}")
async def get_prospect_by_idfg(
    fg_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Look up a prospect's DB primary key by FanGraphs ID."""
    prospect = (
        db.query(Prospect)
        .filter(Prospect.fg_id == fg_id)
        .order_by(Prospect.year.desc())
        .first()
    )
    if not prospect:
        raise HTTPException(status_code=404, detail="Prospect not found")
    return {"id": prospect.id, "name": prospect.name}