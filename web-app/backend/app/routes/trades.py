import csv
import json
import sys
from math import isnan
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func
from typing import Any, Dict, List, Optional, Tuple
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

def map_team_abbreviation(team: str) -> str:
    """Map UI team abbreviations to database team abbreviations"""
    team_mapping = {
        'SF': 'SFG',
        'SD': 'SDP',
        'KC': 'KCR',
        'TB': 'TBR'
    }
    return team_mapping.get(team.upper(), team.upper())

def get_player_values(player_name: str, db: Session):
    # Get only the 2026 player entry since it has all the values we need
    base_player = (
        db.query(Player)
        .filter(Player.name == player_name)
        .filter(Player.year == 2026)
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
        "years": [year for year in range(2026, (base_player.control_through or 2026) + 1)]  # Using control_through
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
        .filter(Prospect.year == 2025)  # Use 2025 since that's what exists in DB
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
                "value": getattr(p, f"value_{year}", None)
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
        pattern="^(trade_value|contract_war|avg_war|total_contract|avg_contract|control_through|years_control|total_future_war|total_future_value|historical_war|historical_value|contract_base_value)$"
    ),
    sort_direction: str = Query("desc", pattern="^(asc|desc)$")
):
    try:
        # Start with 2026 players
        query = db.query(Player).filter(Player.year == 2026)
        
        # Add filter for non-NaN trade values
        query = query.filter(Player.trade_value.isnot(None))  # Filter out NULL values
        
        # Add team filter if specified
        if team:
            if team.upper() == 'FA':
                query = query.filter(Player.team == 'FA')
            else:
                query = query.filter(func.upper(Player.team) == team.upper())
        
        
        sort_column = getattr(Player, sort_by)
        query = query.order_by(desc(sort_column) if sort_direction == "desc" else asc(sort_column))
        
        total_count = query.count()
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        players = query.all()
        results = [{
            "real_id": p.real_id,
            "mlb_id": p.mlb_id,
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


# ═══════════════════════════════════════════════════════════════════════════════
#  PAST TRADES — Pre-computed trade evaluations
# ═══════════════════════════════════════════════════════════════════════════════

_PAST_TRADES_FILE = (
    Path(__file__).resolve().parents[4] / "data" / "generated" / "past_trades" / "trades.json"
)

_past_trades_cache: Optional[List[Dict[str, Any]]] = None

# ── Surplus projection data ──────────────────────────────────────────────────

_SURPLUS_FILE = (
    Path(__file__).resolve().parents[4] / "data" / "generated" / "trade_analysis" / "surplus" / "surplus_2025.csv"
)

_PROJECTION_CUTOFF = "2024-10-01"  # trades after this may not have actual WAR yet

_DOLLAR_PER_WAR = {
    2025: 8_500_000, 2026: 8_500_000, 2027: 8_500_000, 2028: 8_500_000,
    2029: 8_500_000, 2030: 8_500_000, 2031: 8_500_000, 2032: 8_500_000,
    2033: 8_500_000, 2034: 8_500_000, 2035: 8_500_000, 2036: 8_500_000,
    2037: 8_500_000, 2038: 8_500_000, 2039: 8_500_000,
}


def _load_surplus_projections() -> Dict[int, Dict[str, Any]]:
    """Load surplus_2025.csv → {mlbam_id: projection_dict}."""
    if not _SURPLUS_FILE.exists():
        logger.warning(f"Surplus file not found: {_SURPLUS_FILE}")
        return {}

    projections: Dict[int, Dict[str, Any]] = {}
    with open(_SURPLUS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mlbam_raw = row.get("mlbam_id", "")
                if not mlbam_raw or mlbam_raw == "nan":
                    continue
                mlbam_id = int(float(mlbam_raw))
            except (ValueError, TypeError):
                continue

            # Extract year-by-year projected WAR
            yearly_war = []
            for yr in range(2025, 2040):
                col = f"WAR_{yr}"
                val = row.get(col, "")
                if val and val != "nan":
                    try:
                        yearly_war.append({"year": yr, "war": round(float(val), 1)})
                    except ValueError:
                        pass

            # Total future values
            try:
                total_war = float(row.get("total_future_WAR", 0) or 0)
                total_war_value = float(row.get("total_future_WAR_value", 0) or 0)
                total_salary = float(row.get("total_future_salary", 0) or 0)
                surplus = float(row.get("surplus", 0) or 0)
            except (ValueError, TypeError):
                total_war = total_war_value = total_salary = surplus = 0.0

            projections[mlbam_id] = {
                "name": row.get("Name", ""),
                "projected_war": round(total_war, 1),
                "projected_war_value": int(total_war_value),
                "projected_salary": int(total_salary),
                "projected_surplus": int(surplus),
                "projected_yearly_war": yearly_war,
            }

    logger.info(f"Loaded {len(projections)} surplus projections for trade augmentation")
    return projections


def _augment_with_projections(trades: List[Dict[str, Any]]) -> None:
    """
    Augment recent trades (post-cutoff, zero WAR) with projected future values.
    Modifies trades in-place.
    """
    projections = _load_surplus_projections()
    if not projections:
        for t in trades:
            t["evaluation_type"] = "actual"
        return

    augmented_count = 0

    for trade in trades:
        trade_date = trade.get("date", "")
        total_war = trade.get("total_trade_war", 0)

        # Classify the trade
        if trade_date >= _PROJECTION_CUTOFF and abs(total_war) < 0.5:
            trade["evaluation_type"] = "projected"
        else:
            trade["evaluation_type"] = "actual"
            continue  # no augmentation needed

        # Augment each side with projected values
        for side in trade.get("sides", []):
            side_proj_war = 0.0
            side_proj_war_value = 0.0
            side_proj_salary = 0.0
            side_proj_surplus = 0.0
            has_any_projection = False

            for player in side.get("players_received", []):
                mlb_id = player.get("mlb_id")
                proj = projections.get(mlb_id) if mlb_id else None

                if proj:
                    has_any_projection = True
                    player["projected_war"] = proj["projected_war"]
                    player["projected_war_value"] = proj["projected_war_value"]
                    player["projected_salary"] = proj["projected_salary"]
                    player["projected_surplus"] = proj["projected_surplus"]
                    player["projected_yearly_war"] = proj["projected_yearly_war"]
                    player["has_projection"] = True

                    side_proj_war += proj["projected_war"]
                    side_proj_war_value += proj["projected_war_value"]
                    side_proj_salary += proj["projected_salary"]
                    side_proj_surplus += proj["projected_surplus"]
                else:
                    player["projected_war"] = None
                    player["projected_war_value"] = None
                    player["projected_salary"] = None
                    player["projected_surplus"] = None
                    player["projected_yearly_war"] = []
                    player["has_projection"] = False

            side["projected_total_war"] = round(side_proj_war, 1)
            side["projected_total_war_value"] = int(side_proj_war_value)
            side["projected_total_salary"] = int(side_proj_salary)
            side["projected_total_surplus"] = int(side_proj_surplus)

        # Re-determine winner/loser for projected trades using projected surplus
        sides = trade.get("sides", [])
        if len(sides) >= 2:
            sorted_sides = sorted(sides, key=lambda s: s.get("projected_total_surplus", 0), reverse=True)
            trade["projected_winner"] = sorted_sides[0]["team"]
            trade["projected_winner_name"] = sorted_sides[0]["team_name"]
            trade["projected_loser"] = sorted_sides[-1]["team"]
            trade["projected_loser_name"] = sorted_sides[-1]["team_name"]
            trade["projected_surplus_diff"] = (
                sorted_sides[0].get("projected_total_surplus", 0) -
                sorted_sides[-1].get("projected_total_surplus", 0)
            )
            trade["projected_total_war"] = round(
                sum(s.get("projected_total_war", 0) for s in sides), 1
            )

            # Override the "winner" fields to use projected for display
            trade["winner"] = trade["projected_winner"]
            trade["winner_name"] = trade["projected_winner_name"]
            trade["loser"] = trade["projected_loser"]
            trade["loser_name"] = trade["projected_loser_name"]
            trade["surplus_diff"] = trade["projected_surplus_diff"]

        augmented_count += 1

    logger.info(f"Augmented {augmented_count} recent trades with projected values")


_HIST_TEAM_ALIASES: Dict[str, str] = {
    "ANA": "LAA", "CAL": "LAA", "FLA": "MIA", "MON": "WSN", "TBD": "TBR",
}


def _augment_with_historical_war(trades: List[Dict[str, Any]]) -> None:
    """
    Fill in missing WAR data for trade players using the 13k-player historical
    dataset.  The pre-computed trades.json has gaps because the offline pipeline
    uses a limited crosswalk; the historical JSON has 12,999 MLBAM → IDfg
    mappings which covers most of them.

    Modifies trades in-place.
    """
    from app.routes.historical import _load_historical, _mlbam_to_idfg, _players

    _load_historical()
    if not _mlbam_to_idfg or not _players:
        return

    augmented = 0
    for trade in trades:
        trade_year = trade.get("year", 0)
        sides_changed = False

        for side in trade.get("sides", []):
            for player in side.get("players_received", []):
                # Skip players that already have WAR data
                if player.get("yearly_war") or abs(player.get("war_with_team", 0)) > 0.01:
                    continue

                mlb_id = player.get("mlb_id")
                if not mlb_id:
                    continue

                # Look up historical data via MLBAM crosswalk
                idfg = _mlbam_to_idfg.get(str(mlb_id))
                if idfg is None:
                    continue
                hp = _players.get(str(idfg))
                if hp is None:
                    continue

                to_team = player.get("to_team", "")

                # Aggregate batting + pitching WAR by year for matching team
                war_by_year: Dict[int, float] = {}
                salary_by_year: Dict[int, int] = {}
                for season in hp.get("batting", []) + hp.get("pitching", []):
                    st = _HIST_TEAM_ALIASES.get(season["team"], season["team"])
                    if st == to_team and season["year"] >= trade_year:
                        yr = season["year"]
                        war_by_year[yr] = war_by_year.get(yr, 0) + (season.get("war") or 0)
                        salary_by_year[yr] = salary_by_year.get(yr, 0) + int(season.get("salary") or 0)

                if not war_by_year:
                    continue

                # Build yearly_war list sorted by year
                yearly = sorted(
                    [{"year": yr, "war": round(w, 1)} for yr, w in war_by_year.items()],
                    key=lambda x: x["year"],
                )
                total_war = round(sum(w for w in war_by_year.values()), 1)
                total_salary = sum(salary_by_year.values())

                player["war_with_team"] = total_war
                player["yearly_war"] = yearly
                player["salary_with_team"] = total_salary
                player["seasons_with_team"] = len(yearly)

                # Recalculate individual WAR value and surplus
                war_value = 0
                for yr, w in war_by_year.items():
                    dpw = _DOLLAR_PER_WAR.get(yr, 8_500_000)
                    war_value += w * dpw
                player["war_value"] = int(war_value)
                player["surplus"] = int(war_value - total_salary)

                augmented += 1
                sides_changed = True

        if sides_changed:
            # Recalculate side totals
            for side in trade.get("sides", []):
                players_list = side.get("players_received", [])
                side["total_war"] = round(sum(p.get("war_with_team", 0) for p in players_list), 1)
                side["total_salary"] = sum(p.get("salary_with_team", 0) for p in players_list)
                side["total_war_value"] = sum(p.get("war_value", 0) for p in players_list)
                side["total_surplus"] = sum(p.get("surplus", 0) for p in players_list)

            # Recalculate winner/loser for actual (non-projected) trades
            if trade.get("evaluation_type") != "projected":
                sides = trade.get("sides", [])
                if len(sides) >= 2:
                    sorted_sides = sorted(sides, key=lambda s: s.get("total_surplus", 0), reverse=True)
                    trade["winner"] = sorted_sides[0]["team"]
                    trade["winner_name"] = sorted_sides[0]["team_name"]
                    trade["loser"] = sorted_sides[-1]["team"]
                    trade["loser_name"] = sorted_sides[-1]["team_name"]
                    trade["surplus_diff"] = (
                        sorted_sides[0].get("total_surplus", 0) - sorted_sides[-1].get("total_surplus", 0)
                    )
                trade["total_trade_war"] = round(
                    sum(s.get("total_war", 0) for s in trade.get("sides", [])), 1
                )

    logger.info(f"Augmented {augmented} trade players with historical WAR data")


def _load_past_trades() -> List[Dict[str, Any]]:
    """Load & cache the pre-computed trade evaluations, augmented with projections."""
    global _past_trades_cache
    if _past_trades_cache is not None:
        return _past_trades_cache

    if not _PAST_TRADES_FILE.exists():
        logger.warning(f"Past trades file not found: {_PAST_TRADES_FILE}")
        return []

    with open(_PAST_TRADES_FILE, "r") as f:
        _past_trades_cache = json.load(f)

    # Augment recent trades with projected values
    _augment_with_projections(_past_trades_cache)

    # Fill in missing WAR from historical player data
    _augment_with_historical_war(_past_trades_cache)

    logger.info(f"Loaded {len(_past_trades_cache)} past trades")
    return _past_trades_cache


# ── Index helpers (built lazily) ─────────────────────────────────────────────

_trade_by_id: Optional[Dict[int, Dict]] = None
_trades_by_player: Optional[Dict[int, List[int]]] = None  # mlb_id → [trade_id]
_trades_by_team: Optional[Dict[str, List[int]]] = None     # team abbrev → [trade_id]


def _build_indexes():
    """Build lookup indexes from the trades list."""
    global _trade_by_id, _trades_by_player, _trades_by_team
    trades = _load_past_trades()

    _trade_by_id = {}
    _trades_by_player = {}
    _trades_by_team = {}

    for t in trades:
        tid = t["trade_id"]
        _trade_by_id[tid] = t

        for side in t.get("sides", []):
            team = side["team"]
            if team not in _trades_by_team:
                _trades_by_team[team] = []
            _trades_by_team[team].append(tid)

            for p in side.get("players_received", []):
                mid = p.get("mlb_id")
                if mid:
                    if mid not in _trades_by_player:
                        _trades_by_player[mid] = []
                    _trades_by_player[mid].append(tid)


def _ensure_indexes():
    if _trade_by_id is None:
        _build_indexes()


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/past-trades")
def get_past_trades(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    sort_by: str = Query("date", pattern="^(date|surplus_diff|total_trade_war|max_prospect_fv|n_players)$"),
    sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
    team: Optional[str] = None,
    year: Optional[int] = None,
    min_war: Optional[float] = None,
    search: Optional[str] = None,
):
    """List all evaluated past trades with sorting, filtering, and pagination."""
    _ensure_indexes()
    trades = _load_past_trades()

    # ── Filtering ────────────────────────────────────────────────────────
    filtered = trades

    if team:
        team_upper = team.upper()
        filtered = [
            t for t in filtered
            if any(s["team"] == team_upper for s in t.get("sides", []))
        ]

    if year:
        filtered = [t for t in filtered if t["year"] == year]

    if min_war is not None:
        filtered = [t for t in filtered if t["total_trade_war"] >= min_war]

    if search:
        search_lower = search.lower()
        filtered = [
            t for t in filtered
            if search_lower in t["description"].lower()
            or any(
                search_lower in p["name"].lower()
                for s in t.get("sides", [])
                for p in s.get("players_received", [])
            )
        ]

    # ── Sorting ──────────────────────────────────────────────────────────
    reverse = sort_dir == "desc"

    def sort_key(t):
        val = t.get(sort_by)
        if val is None:
            return -999999 if reverse else 999999
        return val

    filtered.sort(key=sort_key, reverse=reverse)

    # ── Pagination ───────────────────────────────────────────────────────
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = filtered[start:end]

    # Return lightweight summaries (no yearly_war per player for list view)
    summaries = []
    for t in page_items:
        sides_summary = []
        for s in t.get("sides", []):
            players_summary = []
            for p in s.get("players_received", []):
                players_summary.append({
                    "mlb_id": p["mlb_id"],
                    "name": p["name"],
                    "war_with_team": p["war_with_team"],
                    "surplus": p["surplus"],
                    "prospect_fv": p.get("prospect_fv"),
                    "from_team": p["from_team"],
                    "from_team_name": p["from_team_name"],
                    # Projected fields (present for "projected" trades)
                    "projected_war": p.get("projected_war"),
                    "projected_surplus": p.get("projected_surplus"),
                    "has_projection": p.get("has_projection"),
                })
            side_data = {
                "team": s["team"],
                "team_name": s["team_name"],
                "total_war": s["total_war"],
                "total_salary": s["total_salary"],
                "total_war_value": s["total_war_value"],
                "total_surplus": s["total_surplus"],
                "players_received": players_summary,
            }
            # Add projected side totals for projected trades
            if "projected_total_war" in s:
                side_data["projected_total_war"] = s["projected_total_war"]
                side_data["projected_total_surplus"] = s["projected_total_surplus"]
            sides_summary.append(side_data)

        summary = {
            "trade_id": t["trade_id"],
            "date": t["date"],
            "year": t["year"],
            "description": t["description"],
            "has_cash": t["has_cash"],
            "has_ptbnl": t["has_ptbnl"],
            "n_teams": t["n_teams"],
            "n_players": t["n_players"],
            "winner": t["winner"],
            "winner_name": t["winner_name"],
            "loser": t["loser"],
            "loser_name": t["loser_name"],
            "surplus_diff": t["surplus_diff"],
            "total_trade_war": t["total_trade_war"],
            "max_prospect_fv": t["max_prospect_fv"],
            "evaluation_type": t.get("evaluation_type", "actual"),
            "sides": sides_summary,
        }
        # Add projected trade-level fields for projected trades
        if t.get("evaluation_type") == "projected":
            summary["projected_total_war"] = t.get("projected_total_war", 0)
            summary["projected_surplus_diff"] = t.get("projected_surplus_diff", 0)

        summaries.append(summary)

    return {
        "trades": summaries,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/past-trades/{trade_id}")
def get_past_trade_detail(trade_id: int):
    """Get full details for a single past trade, including yearly WAR."""
    _ensure_indexes()
    trade = _trade_by_id.get(trade_id)  # type: ignore
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade


@router.get("/player-trades/{mlb_id}")
def get_player_past_trades(mlb_id: int):
    """Get all past trades involving a specific player (by mlb_id)."""
    _ensure_indexes()
    trade_ids = _trades_by_player.get(mlb_id, [])  # type: ignore
    if not trade_ids:
        return {"trades": []}

    unique_ids = list(set(trade_ids))
    trades = [_trade_by_id[tid] for tid in unique_ids if tid in _trade_by_id]  # type: ignore
    trades.sort(key=lambda t: t["date"], reverse=True)
    return {"trades": trades}