import sys
import re
import time
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List, Tuple
import logging
import pandas as pd

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore
    logging.getLogger(__name__).warning(
        "httpx not installed - /transactions endpoint will return empty. "
        "Install with: pip install httpx"
    )

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Change to absolute imports
from app.database import get_db
from app.models.player import Player
from app.models.prospect import Prospect
from app.config import CURRENT_YEAR

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Prospect data helper ──────────────────────────────────────────────────
def _build_prospect_data(db: Session, *, mlbam_id: int = None, name: str = None) -> Optional[Dict[str, Any]]:
    """Look up prospect records and build a ProspectDetail-style dict.

    Tries to match by *mlbam_id* first (most reliable), then falls back
    to a name match.  Returns ``None`` when no prospect data is found.
    """
    anchor: Optional[Prospect] = None
    if mlbam_id:
        anchor = db.query(Prospect).filter(
            Prospect.mlbam_id == mlbam_id
        ).order_by(Prospect.year.desc()).first()
    if anchor is None and name:
        anchor = db.query(Prospect).filter(
            Prospect.name == name
        ).order_by(Prospect.year.desc()).first()
    if anchor is None:
        return None

    records = (
        db.query(Prospect)
        .filter(Prospect.name == anchor.name)
        .order_by(Prospect.year.desc())
        .all()
    )
    if not records:
        return None

    latest = records[0]
    is_pitcher_flag = "p" in (latest.position or "").lower()

    if is_pitcher_flag:
        tools = {
            "fastball": latest.fastball, "slider": latest.slider,
            "curve": latest.curve, "changeup": latest.changeup,
            "command": latest.command,
        }
    else:
        tools = {
            "hit": latest.hit, "game_power": latest.game_power,
            "raw_power": latest.raw_power, "speed": latest.speed,
        }

    history: List[Dict[str, Any]] = []
    for r in records:
        entry: Dict[str, Any] = {
            "year": r.year, "age": r.age, "org": r.org,
            "position": r.position, "fv": r.fv, "value": r.value,
            "composite": r.composite, "top_100": r.top_100,
            "org_rank": r.org_rank,
        }
        if is_pitcher_flag:
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

    return {
        "prospect_id": anchor.id,
        "IDfg": latest.IDfg,
        "mlbam_id": latest.mlbam_id,
        "name": latest.name,
        "org": latest.org,
        "position": latest.position,
        "age": latest.age,
        "fv": latest.fv,
        "has_mlb": latest.has_mlb,
        "is_pitcher": is_pitcher_flag,
        "tools": tools,
        "history": history,
    }


# ── Trade Value History (CSV-backed, loaded once at startup) ──────────────
_TRADE_HISTORY_CSV = (
    Path(__file__).resolve().parents[4]  # project root (LSTMLB)
    / "data" / "generated" / "value_by_year" / "trade_value_history.csv"
)
_trade_history_df: Optional[pd.DataFrame] = None


def _get_trade_history() -> pd.DataFrame:
    """Lazy-load and cache the trade value history CSV."""
    global _trade_history_df
    if _trade_history_df is None:
        if not _TRADE_HISTORY_CSV.exists():
            logger.warning(f"Trade value history not found: {_TRADE_HISTORY_CSV}")
            _trade_history_df = pd.DataFrame()
        else:
            _trade_history_df = pd.read_csv(_TRADE_HISTORY_CSV)
            logger.info(
                f"Loaded trade value history: {len(_trade_history_df)} entries, "
                f"{_trade_history_df['mlb_id'].nunique()} players"
            )
    return _trade_history_df

# ── Statcast expected stats (CSV-backed, loaded once) ─────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_STATCAST_BATTER_CSV = _PROJECT_ROOT / "data" / "statcast" / "statcast_batter_expected_stats_2015_2025.csv"
_STATCAST_PITCHER_CSV = _PROJECT_ROOT / "data" / "statcast" / "statcast_pitcher_expected_stats_2015_2025.csv"
_statcast_batter: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None
_statcast_pitcher: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None


def _get_statcast_batter() -> Dict[Tuple[int, int], Dict[str, float]]:
    """Lazy-load batter expected stats keyed by (mlbam_id, year)."""
    global _statcast_batter
    if _statcast_batter is None:
        _statcast_batter = {}
        if _STATCAST_BATTER_CSV.exists():
            df = pd.read_csv(_STATCAST_BATTER_CSV)
            for _, r in df.iterrows():
                key = (int(r["player_id"]), int(r["year"]))
                _statcast_batter[key] = {
                    "xba": r.get("est_ba"),
                    "xslg": r.get("est_slg"),
                    "xwoba": r.get("est_woba"),
                }
            logger.info(f"Loaded statcast batter expected stats: {len(_statcast_batter)} entries")
        else:
            logger.warning(f"Statcast batter CSV not found: {_STATCAST_BATTER_CSV}")
    return _statcast_batter


def _get_statcast_pitcher() -> Dict[Tuple[int, int], Dict[str, float]]:
    """Lazy-load pitcher expected stats keyed by (mlbam_id, year)."""
    global _statcast_pitcher
    if _statcast_pitcher is None:
        _statcast_pitcher = {}
        if _STATCAST_PITCHER_CSV.exists():
            df = pd.read_csv(_STATCAST_PITCHER_CSV)
            for _, r in df.iterrows():
                key = (int(r["player_id"]), int(r["year"]))
                _statcast_pitcher[key] = {
                    "xera": r.get("xera"),
                }
            logger.info(f"Loaded statcast pitcher expected stats: {len(_statcast_pitcher)} entries")
        else:
            logger.warning(f"Statcast pitcher CSV not found: {_STATCAST_PITCHER_CSV}")
    return _statcast_pitcher


def normalize_team_abbreviation(team: str) -> str:
    """Normalize team abbreviations - players use 2-letter codes, prospects use 3-letter codes"""
    # Map from prospect 3-letter codes to player 2-letter codes for output consistency
    from_prospect_to_player = {
        'SFG': 'SF',
        'SDP': 'SD',
        'KCR': 'KC',
        'OAK': 'ATH',
        'TBR': 'TB'
    }
    # Map from player 2-letter codes to themselves (identity)
    player_codes = {'SF', 'SD', 'KC', 'ATH', 'TB'}
    
    team_upper = team.upper()
    # If it's a 3-letter prospect code, convert to 2-letter player code
    if team_upper in from_prospect_to_player:
        return from_prospect_to_player[team_upper]
    # Otherwise return as-is (uppercase)
    return team_upper

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
            # Players use 2-letter codes (TB, SD, SF, ATH, KC) directly
            query = query.filter(Player.team == team.upper())
            
        if position:
            query = query.filter(
                or_(
                    Player.position.ilike(f"%{position}%"),
                    Player.status == position
                )
            )
            
        # Apply sorting
        if sort_by == "war":
            # Combined WAR sorting (parenthesized so .desc() applies to the sum)
            query = query.order_by(
                (func.coalesce(Player.war_bat, 0) + 
                 func.coalesce(Player.war_pit, 0)).desc()
            )
        elif sort_by == "value":
            query = query.order_by(Player.surplus_value.desc())
            
        players = query.all()
        logger.info(f"Found {len(players)} players")

        active_results = [
            {
                "real_id": p.real_id,
                "mlb_id": p.mlb_id,
                "name": p.name,
                "team": p.team,
                "position": p.position,
                "status": p.status,
                "age": p.age,
                "war_bat": p.war_bat,
                "war_pit": p.war_pit,
                "is_historical": False,
                "value": {
                    "base_value": p.base_value,
                    "contract_value": p.contract_value,
                    "surplus_value": p.surplus_value,
                    "trade_value": p.trade_value
                }
            } for p in players
        ]

        # For searches, also include historical player matches (up to a limit)
        historical_results = []
        if search and len(search) >= 2:
            try:
                from app.models.historical import HistoricalPlayer
                q_lower = search.strip().lower()
                # Collect active mlb_ids to avoid duplicates
                active_mlb_ids = {p.get("mlb_id") for p in active_results if p.get("mlb_id")}
                active_names_lower = {p["name"].lower() for p in active_results}
                hist_rows = (
                    db.query(HistoricalPlayer)
                    .filter(HistoricalPlayer.name_lower.contains(q_lower))
                    .order_by(HistoricalPlayer.career_war.desc())
                    .limit(50)
                    .all()
                )
                for hp in hist_rows:
                    mlbam = hp.mlbam
                    if mlbam and mlbam in active_mlb_ids:
                        continue
                    if hp.name_lower in active_names_lower:
                        continue
                    teams_str = ", ".join((hp.teams or [])[:3])
                    historical_results.append({
                        "real_id": hp.idfg,
                        "mlb_id": mlbam,
                        "name": hp.name,
                        "team": teams_str or "---",
                        "position": "P" if hp.is_pitcher else "Pos",
                        "status": "historical",
                        "age": None,
                        "war_bat": None if hp.is_pitcher else (hp.career_war or 0),
                        "war_pit": (hp.career_war or 0) if hp.is_pitcher else None,
                        "is_historical": True,
                        "career_war": hp.career_war or 0,
                        "first_year": hp.first_year,
                        "last_year": hp.last_year,
                        "value": None
                    })
                    if len(historical_results) >= 15:
                        break
            except Exception as he:
                logger.warning(f"Historical search fallback failed: {he}")

        all_results = active_results + historical_results

        return {
            "count": len(all_results),
            "players": all_results
        }
    except Exception as e:
        logger.error(f"Error in get_players: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _build_historical_response(hist: dict) -> dict:
    """Transform historical player JSON into a response compatible with PlayerStats.
    Adds isHistorical=True and historical batting/pitching seasons."""
    # Build per-year projections-like objects from historical batting/pitching data
    # Merge batting + pitching for same year
    by_year: dict = {}

    for s in hist.get("batting", []):
        yr = s.get("year")
        if yr is None:
            continue
        if yr not in by_year:
            by_year[yr] = {"year": yr, "team": s.get("team", ""), "age": s.get("age")}
        entry = by_year[yr]
        entry["hitting"] = {
            "g_bat": s.get("g"), "war_bat": s.get("war"),
            "bb_pct_bat": s.get("bb_pct"), "k_pct_bat": s.get("k_pct"),
            "avg": s.get("avg"), "obp": s.get("obp"), "slg": s.get("slg"),
            "ops": s.get("ops"), "woba": s.get("woba"), "wrc_plus": s.get("wrc_plus"),
            "off": s.get("off"), "bat": s.get("bat", s.get("off")), "bsr": s.get("bsr"), "def_value": s.get("def_value"),
            "hr": s.get("hr"), "doubles": s.get("doubles") if "doubles" in s else None,
            "triples": s.get("triples") if "triples" in s else None,
            "r": s.get("r"), "rbi": s.get("rbi"), "sb": s.get("sb"), "cs": s.get("cs"),
        }
        entry["salary"] = s.get("salary")
        entry["war_value"] = s.get("war_value")
        entry["surplus"] = s.get("surplus")

    for s in hist.get("pitching", []):
        yr = s.get("year")
        if yr is None:
            continue
        if yr not in by_year:
            by_year[yr] = {"year": yr, "team": s.get("team", ""), "age": s.get("age")}
        entry = by_year[yr]
        entry["pitching"] = {
            "g_pit": s.get("g"), "gs": s.get("gs"), "war_pit": s.get("war"),
            "era": s.get("era"), "fip": s.get("fip"),
            "ip": s.get("ip"),
            "k_pct_pit": s.get("k_pct"), "bb_pct_pit": s.get("bb_pct"),
            "w": s.get("w"), "l": s.get("l"), "sv": s.get("sv"),
            "ip": s.get("ip"), "whip": s.get("whip"),
            "so": s.get("so"), "bb": s.get("bb"),
            "k_9": s.get("k_9"), "bb_9": s.get("bb_9"), "hr_9": s.get("hr_9"),
        }
        # Accumulate pitching war_value into the combined entry.
        # Salary is only set once (from whichever section provided it
        # first) to avoid double-counting.
        pit_wv = s.get("war_value")
        if pit_wv is not None:
            entry["war_value"] = (entry.get("war_value") or 0) + pit_wv
        if entry.get("salary") is None:
            entry["salary"] = s.get("salary")
        # Recalculate surplus from combined war_value minus the single
        # salary.  Do NOT sum individual surplus values — each component's
        # surplus was computed independently with the full salary, so
        # summing them would subtract the contract twice.
        wv = entry.get("war_value") or 0
        sal = entry.get("salary")
        entry["surplus"] = (wv - sal) if sal else None

    # Augment with statcast expected stats
    mlbam = hist.get("mlbam")
    if mlbam:
        sc_bat = _get_statcast_batter()
        sc_pit = _get_statcast_pitcher()
        for yr, entry in by_year.items():
            key = (int(mlbam), int(yr))
            if "hitting" in entry:
                sc = sc_bat.get(key)
                if sc:
                    entry["hitting"]["xba"] = sc.get("xba")
                    entry["hitting"]["xslg"] = sc.get("xslg")
                    entry["hitting"]["xwoba"] = sc.get("xwoba")
            if "pitching" in entry:
                sc = sc_pit.get(key)
                if sc:
                    entry["pitching"]["xera"] = sc.get("xera")

    projections = []
    for yr in sorted(by_year.keys()):
        e = by_year[yr]
        proj = {
            "year": yr,
            "age": e.get("age"),
            "team": e.get("team", ""),
            "position": "",
            "status": "",
            "fa_year": None,
            "probable_fa_year": None,
            "earliest_fa_year": None,
            "value": {
                "base_value": e.get("war_value") or 0,
                "contract_value": e.get("salary") or 0,
                "surplus_value": e.get("surplus") or 0,
                "trade_value": 0, "contract_war": 0, "avg_war": 0,
                "total_contract": hist.get("career_salary") or 0,
                "avg_contract": 0,
                "years_control": 0, "control_through": 0,
                "total_future_war": 0, "total_future_value": 0,
                "total_war": hist.get("career_war", 0),
                "total_value": hist.get("career_war_value", 0),
                "historical_war": hist.get("career_war", 0),
                "historical_value": hist.get("career_war_value", 0),
                "contract_base_value": 0,
            },
            "salary": e.get("salary"),
            "war_value": e.get("war_value"),
        }
        if "hitting" in e:
            proj["hitting"] = e["hitting"]
        if "pitching" in e:
            proj["pitching"] = e["pitching"]
        projections.append(proj)

    # Determine position and team from last season
    last_team = hist.get("teams", [""])[0] if hist.get("teams") else ""
    if projections:
        last_team = projections[-1].get("team", last_team)

    return {
        "name": hist["name"],
        "team": last_team,
        "position": "P" if hist.get("is_pitcher") else "Pos",
        "mlb_id": hist.get("mlbam"),
        "isHistorical": True,
        "historicalMeta": {
            "idfg": hist.get("idfg"),
            "bbref": hist.get("bbref"),
            "birth_year": hist.get("birth_year"),
            "death_year": hist.get("death_year"),
            "first_year": hist.get("first_year"),
            "last_year": hist.get("last_year"),
            "teams": hist.get("teams", []),
            "career_war": hist.get("career_war", 0),
            "career_bat_war": hist.get("career_bat_war", 0),
            "career_pit_war": hist.get("career_pit_war", 0),
            "career_salary": hist.get("career_salary"),
            "career_war_value": hist.get("career_war_value", 0),
            "career_surplus": hist.get("career_surplus"),
        },
        "projections": projections,
    }

@router.get("/{player_id}/details")
async def get_player_details(player_id: int, db: Session = Depends(get_db)):
    logger.debug(f"Received request for player_id: {player_id}")
    
    try:
        # Try lookup by mlb_id first, then fall back to real_id (IDfg)
        logger.debug(f"Trying mlb_id lookup for: {player_id}")
        query = db.query(Player).filter(Player.mlb_id == player_id)
        player_years = query.order_by(Player.year).all()
        
        if not player_years:
            logger.info(f"No mlb_id match, trying real_id lookup for: {player_id}")
            query = db.query(Player).filter(Player.real_id == player_id)
            player_years = query.order_by(Player.year).all()
        
        logger.info(f"Found {len(player_years)} years for player with ID: {player_id}")
        
        if not player_years:
            # Fall back to historical player data
            from app.routes.historical import get_historical_player
            hist = get_historical_player(player_id)
            if hist is not None:
                logger.info(f"Found historical player: {hist['name']} (IDfg={hist['idfg']})")
                return _build_historical_response(hist)

            # Fall back to prospect-only data (player has mlbam_id but no
            # Player rows — e.g. a prospect who hasn't accumulated MLB stats)
            prospect_data = _build_prospect_data(db, mlbam_id=player_id)
            if prospect_data is not None:
                logger.info(f"Found prospect-only player: {prospect_data['name']} (mlbam={player_id})")
                headshot_url = (
                    f"https://img.mlbstatic.com/mlb-photos/image/upload/"
                    f"w_213,d_people:generic:headshot:silo:current.png,"
                    f"q_auto:best,f_auto/v1/people/{player_id}/headshot/67/current"
                )
                return {
                    "name": prospect_data["name"],
                    "team": prospect_data["org"],
                    "position": prospect_data["position"],
                    "mlb_id": player_id,
                    "isProspectOnly": True,
                    "prospectData": prospect_data,
                    "headshot_url": headshot_url,
                    "projections": [],
                }

            raise HTTPException(
                status_code=404, 
                detail=f"Player not found with ID: {player_id}"
            )
        
        # Get current year data or fallback to first available
        current_year_data = next((p for p in player_years if p.year == CURRENT_YEAR), player_years[0])
        
        response = {
            "name": current_year_data.name,
            "team": current_year_data.team,
            "position": current_year_data.position,
            "mlb_id": current_year_data.mlb_id,
            "projections": [{
                "year": p.year,
                "age": p.age,
                "team": p.team,
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
                    "bat": p.bat,
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
                    "ip": p.ip,
                    "war_pit": p.war_pit,
                    "era": p.era,
                    "fip": p.fip,
                    "k_pct_pit": p.k_pct_pit,
                    "bb_pct_pit": p.bb_pct_pit,
                    "gb_pct": p.gb_pct,
                    "fb_pct": p.fb_pct,
                    "hr_fb": p.hr_fb,
                    "hr_9": p.hr_9
                }} if p.war_pit is not None else {})
            } for p in player_years]
        }

        # Augment historical seasons with statcast expected stats
        mlbam = current_year_data.mlb_id
        if mlbam:
            sc_bat = _get_statcast_batter()
            sc_pit = _get_statcast_pitcher()
            for proj in response["projections"]:
                key = (int(mlbam), int(proj["year"]))
                if "hitting" in proj:
                    sc = sc_bat.get(key)
                    if sc:
                        proj["hitting"]["xba"] = sc.get("xba")
                        proj["hitting"]["xslg"] = sc.get("xslg")
                        proj["hitting"]["xwoba"] = sc.get("xwoba")
                if "pitching" in proj:
                    sc = sc_pit.get(key)
                    if sc:
                        proj["pitching"]["xera"] = sc.get("xera")

        # Attach prospect data if available (looked up by mlbam_id or name)
        prospect_data = _build_prospect_data(
            db,
            mlbam_id=current_year_data.mlb_id,
            name=current_year_data.name,
        )
        if prospect_data is not None:
            response["prospectData"] = prospect_data

        return response
        
    except Exception as e:
        logger.error(f"Error in get_player_details: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Trade-value history endpoint ──────────────────────────────────────────
@router.get("/{player_id}/trade-value-history")
async def get_trade_value_history(player_id: str, db: Session = Depends(get_db)):
    """Return year-by-year trade value timeline for a player."""
    df = _get_trade_history()
    if df.empty:
        return JSONResponse([])

    # Resolve the player so we can match on mlb_id
    try:
        pid = int(player_id)
        player = db.query(Player).filter(
            or_(Player.mlb_id == pid, Player.real_id == pid)
        ).first()
    except (ValueError, TypeError):
        player = None

    if player is None:
        return JSONResponse([])

    mlb_id = player.mlb_id
    rows = df[df["mlb_id"] == mlb_id]

    if rows.empty:
        # Try matching by IDfg (real_id) as fallback
        idfg = player.real_id
        if idfg is not None:
            rows = df[df["IDfg"] == idfg]

    if rows.empty:
        return JSONResponse([])

    rows = rows.sort_values("year")
    result = [
        {
            "year": int(r["year"]),
            "value": round(float(r["value"]), 2),
            "valueType": r["value_type"],
            "label": r["label"],
        }
        for _, r in rows.iterrows()
    ]
    return JSONResponse(result)


# ── Player Bio / Awards / Draft (MLB Stats API, cached) ──────────────────

_player_info_cache: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_PLAYER_INFO_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days — bio data changes rarely

# Award names we consider "major" (filter out weekly/monthly/minor league awards)
_MAJOR_AWARDS = {
    "MVP", "Cy Young", "Rookie of the Year", "Silver Slugger",
    "Gold Glove", "All-Star", "Hank Aaron Award",
    "World Series Championship", "World Series MVP", "NLCS MVP", "ALCS MVP",
    "Platinum Glove", "Roberto Clemente Award",
    "Edgar Martinez Outstanding DH", "Reliever of the Year",
    "All-MLB First Team", "All-MLB Second Team",
    "Home Run Derby Winner",
}

# Keywords that indicate a non-MLB (minor league / amateur) award
_MINOR_LEAGUE_KEYWORDS = (
    "minor league", "milb", "triple-a", "double-a", "single-a",
    "aaa", "aa all-star", "futures game", "arizona fall",
    "carolina league", "texas league", "eastern league",
    "southern league", "international league",
    "pacific coast", "midwest league", "south atlantic",
    "pioneer league", "appalachian", "northwest league",
    "florida state", "california league",
    "college", "ncaa", "high school",
)


def _is_major_award(award_name: str) -> bool:
    """Check if an award name matches a major award at the MLB level (partial match)."""
    name_lower = award_name.lower()
    # Exclude any award with minor league / amateur indicators
    for kw in _MINOR_LEAGUE_KEYWORDS:
        if kw in name_lower:
            return False
    for major in _MAJOR_AWARDS:
        if major.lower() in name_lower:
            return True
    return False


async def _fetch_player_info(mlb_id: int) -> Dict[str, Any]:
    """Fetch player bio/awards/draft from MLB Stats API with caching."""
    if httpx is None:
        return {}

    now = time.time()
    if mlb_id in _player_info_cache:
        cached_time, cached_data = _player_info_cache[mlb_id]
        if now - cached_time < _PLAYER_INFO_CACHE_TTL:
            return cached_data

    url = f"https://statsapi.mlb.com/api/v1/people/{mlb_id}?hydrate=awards,draft"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch player info for mlb_id={mlb_id}: {e}")
        return {}

    people = data.get("people", [])
    if not people:
        return {}

    p = people[0]

    # Parse bio
    bio = {
        "height": p.get("height"),
        "weight": p.get("weight"),
        "birthDate": p.get("birthDate"),
        "birthCity": p.get("birthCity"),
        "birthStateProvince": p.get("birthStateProvince"),
        "birthCountry": p.get("birthCountry"),
        "batSide": p.get("batSide", {}).get("description") if p.get("batSide") else None,
        "pitchHand": p.get("pitchHand", {}).get("description") if p.get("pitchHand") else None,
        "mlbDebutDate": p.get("mlbDebutDate"),
        "primaryNumber": p.get("primaryNumber"),
        "nickName": p.get("nickName"),
    }

    # Parse awards (filter to major awards, deduplicate by name+season)
    raw_awards = p.get("awards", [])
    major_awards = []
    seen_awards = set()
    for a in raw_awards:
        name = a.get("name", "")
        season = a.get("season")
        if not _is_major_award(name):
            continue
        key = (name, season)
        if key in seen_awards:
            continue
        seen_awards.add(key)
        major_awards.append({
            "name": name,
            "season": season,
        })
    # Sort by season desc
    major_awards.sort(key=lambda x: x.get("season", ""), reverse=True)

    # Parse draft info
    drafts_raw = p.get("drafts", [])
    draft_info = None
    if drafts_raw:
        # Take the most recent draft (Rule 4 / June Amateur Draft)
        for d in reversed(drafts_raw):
            draft_info = {
                "year": d.get("year"),
                "round": d.get("pickRound"),
                "pickNumber": d.get("pickNumber"),
                "school": d.get("school", {}).get("name") if d.get("school") else None,
                "team": d.get("team", {}).get("name") if d.get("team") else None,
            }
            break

    result = {
        "bio": bio,
        "awards": major_awards,
        "draft": draft_info,
    }
    _player_info_cache[mlb_id] = (now, result)
    return result


@router.get("/{player_id}/info")
async def get_player_info(player_id: str, db: Session = Depends(get_db)):
    """Return bio, awards, and draft info for a player from MLB Stats API."""
    try:
        pid = int(player_id)
        player = db.query(Player).filter(
            or_(Player.mlb_id == pid, Player.real_id == pid)
        ).first()
    except (ValueError, TypeError):
        player = None

    mlb_id = None
    if player is not None and player.mlb_id is not None:
        mlb_id = player.mlb_id
    else:
        # Fall back to historical data for MLBAM resolution
        from app.routes.historical import get_historical_player
        try:
            hist = get_historical_player(int(player_id))
            if hist and hist.get("mlbam"):
                mlb_id = hist["mlbam"]
        except (ValueError, TypeError):
            pass

    if mlb_id is None:
        # Last resort: try the raw player_id as an mlbam_id directly.
        # This covers prospect-only players who have an mlbam_id but no
        # Player or HistoricalPlayer rows — the MLB Stats API will still
        # return bio data for them (draft info, height, bats/throws, etc.).
        try:
            mlb_id = int(player_id)
        except (ValueError, TypeError):
            return JSONResponse({})

    info = await _fetch_player_info(mlb_id)
    return JSONResponse(info)


# ── Transaction History (MLB Stats API, cached) ──────────────────────────

# Simple in-memory cache: mlb_id -> (timestamp, data)
_transaction_cache: Dict[int, Tuple[float, List[Dict[str, Any]]]] = {}
_TRANSACTION_CACHE_TTL = 60 * 60 * 24  # 24 hours

# Transaction types we want to display (filter out noise)
_IMPORTANT_TYPE_CODES = {
    "TR",   # Trade
    "SGN",  # Signed
    "SFA",  # Signed as Free Agent
    "DFA",  # Declared Free Agency (player elects FA — NOT "Designated for Assignment")
    "FA",   # Free Agency declared
    "CL",   # Claimed (waivers)
    "WV",   # Waiver
    "SC",   # Status Change (filtered below to exclude IL stints)
    "RET",  # Retirement
    "REL",  # Released
    "SE",   # Selected (to 40-man roster, Rule 5, etc.)
}

# IL / disabled-list keywords to filter OUT of Status Change transactions
_IL_KEYWORDS = (
    "injured list", "disabled list", "paternity", "bereavement",
    "restricted list", "suspended", "concussion", "covid",
)

# MLB full-name → our team abbreviation mapping
_MLB_TEAM_TO_ABBREV: Dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Florida Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
    "Oakland Athletics": "ATH", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Tampa Bay Devil Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Montreal Expos": "WSH",
}


def _build_name_index(db: Session) -> Dict[str, List[Dict[str, Any]]]:
    """Build a name→[{mlb_id, real_id, name}] index for matching players in trade descriptions."""
    players = db.query(
        Player.name, Player.mlb_id, Player.real_id
    ).filter(Player.year == CURRENT_YEAR).all()

    index: Dict[str, List[Dict[str, Any]]] = {}
    for p in players:
        if not p.name or not p.mlb_id:
            continue
        key = p.name.lower().strip()
        if key not in index:
            index[key] = []
        index[key].append({
            "name": p.name,
            "mlb_id": p.mlb_id,
            "real_id": p.real_id,
        })
    return index


def _find_players_in_description(
    description: str, name_index: Dict[str, List[Dict[str, Any]]], exclude_mlb_id: int
) -> List[Dict[str, Any]]:
    """Parse a trade description and find referenced players in our database."""
    found = []
    seen_ids = {exclude_mlb_id}  # Don't link the player to themselves

    for name_key, player_list in name_index.items():
        for player in player_list:
            pname = player["name"]
            if pname in description:
                if player["mlb_id"] not in seen_ids:
                    found.append({
                        "name": pname,
                        "mlbId": player["mlb_id"],
                        "realId": player["real_id"],
                    })
                    seen_ids.add(player["mlb_id"])

    return found


async def _fetch_transactions(mlb_id: int) -> List[Dict[str, Any]]:
    """Fetch transactions from MLB Stats API with caching."""
    if httpx is None:
        return []

    now = time.time()
    if mlb_id in _transaction_cache:
        cached_time, cached_data = _transaction_cache[mlb_id]
        if now - cached_time < _TRANSACTION_CACHE_TTL:
            return cached_data

    url = f"https://statsapi.mlb.com/api/v1/transactions?playerId={mlb_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch transactions for mlb_id={mlb_id}: {e}")
        return []

    raw = data.get("transactions", [])
    _transaction_cache[mlb_id] = (now, raw)
    return raw


@router.get("/{player_id}/milb-stats")
async def get_player_milb_stats(player_id: str, db: Session = Depends(get_db)):
    """Return all MiLB hitting & pitching stats for a player.

    Resolves the player's FanGraphs ID via the Player model (real_id) or
    the PlayerIdCrosswalk (mlbam_id → fg_id), then queries MiLB stat tables.
    """
    from app.models.milb_stats import MiLBHittingStats, MiLBPitchingStats
    from app.models.player_id_crosswalk import PlayerIdCrosswalk

    try:
        pid = int(player_id)
        player = db.query(Player).filter(
            or_(Player.mlb_id == pid, Player.real_id == pid)
        ).first()
    except (ValueError, TypeError):
        player = None

    # Collect candidate FanGraphs IDs to try (as strings)
    candidate_fgids: list[str] = []

    if player:
        # real_id is the numeric FanGraphs ID for active players
        if player.real_id is not None:
            candidate_fgids.append(str(player.real_id))
        # Try crosswalk via mlbam_id
        if player.mlb_id is not None:
            xref = (
                db.query(PlayerIdCrosswalk)
                .filter(PlayerIdCrosswalk.mlbam_id == player.mlb_id)
                .first()
            )
            if xref and xref.fg_id and xref.fg_id not in candidate_fgids:
                candidate_fgids.append(xref.fg_id)
    else:
        # No Player record — try crosswalk directly (player_id may be mlbam_id)
        try:
            xref = (
                db.query(PlayerIdCrosswalk)
                .filter(PlayerIdCrosswalk.mlbam_id == int(player_id))
                .first()
            )
            if xref and xref.fg_id:
                candidate_fgids.append(xref.fg_id)
        except (ValueError, TypeError):
            pass
        # Also try raw player_id as IDfg
        candidate_fgids.append(player_id)

    if not candidate_fgids:
        return {"hitting": [], "pitching": []}

    # Try each candidate until we find stats
    hitting: list[dict] = []
    pitching: list[dict] = []

    for fgid in candidate_fgids:
        if hitting or pitching:
            break

        hitting_rows = (
            db.query(MiLBHittingStats)
            .filter(MiLBHittingStats.IDfg == fgid)
            .order_by(MiLBHittingStats.season.desc(), MiLBHittingStats.level)
            .all()
        )
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
            .filter(MiLBPitchingStats.IDfg == fgid)
            .order_by(MiLBPitchingStats.season.desc(), MiLBPitchingStats.level)
            .all()
        )
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


@router.get("/{player_id}/transactions")
async def get_player_transactions(player_id: str, db: Session = Depends(get_db)):
    """Return transaction history for a player, with linked players for trades."""
    try:
        pid = int(player_id)
        player = db.query(Player).filter(
            or_(Player.mlb_id == pid, Player.real_id == pid)
        ).first()
    except (ValueError, TypeError):
        player = None

    mlb_id = None
    if player is not None and player.mlb_id is not None:
        mlb_id = player.mlb_id
    else:
        # Fall back to historical data for MLBAM resolution
        from app.routes.historical import get_historical_player
        try:
            hist = get_historical_player(int(player_id))
            if hist and hist.get("mlbam"):
                mlb_id = hist["mlbam"]
        except (ValueError, TypeError):
            pass

    if mlb_id is None:
        return JSONResponse([])

    raw_transactions = await _fetch_transactions(mlb_id)

    if not raw_transactions:
        return JSONResponse([])

    # Build name index for matching trade participants
    name_index = _build_name_index(db)

    results = []
    for txn in raw_transactions:
        type_code = txn.get("typeCode", "")
        if type_code not in _IMPORTANT_TYPE_CODES:
            continue

        description = txn.get("description", "")
        date = txn.get("date") or txn.get("effectiveDate", "")

        # Filter out IL stints and minor status changes
        if type_code == "SC":
            desc_lower = description.lower()
            if any(kw in desc_lower for kw in _IL_KEYWORDS):
                continue

        # Fix DFA: MLB API uses typeCode="DFA" for "Declared Free Agency"
        # (player elects FA). Remap to "FA" for correct display.
        display_type_code = type_code
        if type_code == "DFA" and "elected free agency" in description.lower():
            display_type_code = "FA"

        from_team_name = txn.get("fromTeam", {}).get("name", "") if txn.get("fromTeam") else ""
        to_team_name = txn.get("toTeam", {}).get("name", "") if txn.get("toTeam") else ""

        from_team = _MLB_TEAM_TO_ABBREV.get(from_team_name, "")
        to_team = _MLB_TEAM_TO_ABBREV.get(to_team_name, "")

        # For trades, find linked players in description
        linked_players = []
        if type_code == "TR" and description:
            linked_players = _find_players_in_description(description, name_index, mlb_id)

        results.append({
            "id": txn.get("id"),
            "date": date,
            "typeCode": display_type_code,
            "typeDesc": txn.get("typeDesc", ""),
            "description": description,
            "fromTeam": from_team,
            "fromTeamName": from_team_name,
            "toTeam": to_team,
            "toTeamName": to_team_name,
            "linkedPlayers": linked_players,
        })

    # Sort by date descending (most recent first)
    results.sort(key=lambda x: x["date"], reverse=True)

    return JSONResponse(results)