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

logger = logging.getLogger(__name__)
router = APIRouter()

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
                    "team": p.team,
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


def _is_major_award(award_name: str) -> bool:
    """Check if an award name matches a major award (partial match)."""
    name_lower = award_name.lower()
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

    if player is None or player.mlb_id is None:
        return JSONResponse({})

    info = await _fetch_player_info(player.mlb_id)
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
    ).filter(Player.year == 2026).all()

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

    if player is None or player.mlb_id is None:
        return JSONResponse([])

    mlb_id = player.mlb_id
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