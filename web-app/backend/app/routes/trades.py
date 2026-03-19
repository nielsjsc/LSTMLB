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
from app.config import CURRENT_YEAR, PROSPECT_DEFAULT_YEAR, PROSPECT_YEAR_START, PROSPECT_YEAR_END

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
    # Get only the current-year player entry since it has all the values we need
    base_player = (
        db.query(Player)
        .filter(Player.name == player_name)
        .filter(Player.year == CURRENT_YEAR)
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
        "years": [year for year in range(CURRENT_YEAR, (base_player.control_through or CURRENT_YEAR) + 1)]  # Using control_through
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
        .filter(Prospect.year == PROSPECT_DEFAULT_YEAR)
        .first()
    )
    
    if not prospect:
        raise HTTPException(status_code=404, detail=f"Prospect {prospect_name} not found")
    
    value = prospect.get_value(PROSPECT_DEFAULT_YEAR) or 0
    

    
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
async def get_all_prospects(
    player_type: str = Query(..., description="Either 'hitter' or 'pitcher'"),
    year: int = Query(PROSPECT_DEFAULT_YEAR, ge=PROSPECT_YEAR_START, le=PROSPECT_YEAR_END),
    db: Session = Depends(get_db)
):
    """Deprecated: use GET /prospects/?slim=true instead.
    Kept for backward compatibility — delegates to the unified prospects endpoint."""
    from app.routes.prospects import get_prospects
    result = await get_prospects(
        player_type=player_type, year=year,
        page=1, page_size=500, slim=True, db=db
    )
    return {"players": result["players"]}


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
        # Start with current-year players
        query = db.query(Player).filter(Player.year == CURRENT_YEAR)
        
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

_SALARY_BY_YEAR_DIR = (
    Path(__file__).resolve().parents[4] / "data" / "salary" / "by_year"
)

_past_trades_cache: Optional[List[Dict[str, Any]]] = None

# ── Surplus projection data ──────────────────────────────────────────────────

_SURPLUS_FILE = (
    Path(__file__).resolve().parents[4] / "data" / "generated" / "historical_values" / "surplus" / f"surplus_{CURRENT_YEAR - 1}.csv"
)

_PROJECTION_CUTOFF = f"{CURRENT_YEAR - 2}-10-01"  # trades after this may not have actual WAR yet

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

    Also fixes departure_year and still_on_team which the offline pipeline
    computes incorrectly for off-season trades.

    Modifies trades in-place.
    """
    from app.routes import historical as _hist_mod

    _hist_mod._load_historical()
    if not _hist_mod._mlbam_to_idfg or not _hist_mod._players:
        return

    mlbam_to_idfg = _hist_mod._mlbam_to_idfg
    players_db = _hist_mod._players

    CURRENT_YEAR = 2025  # latest year with full historical data

    augmented = 0
    for trade in trades:
        trade_year = trade.get("year", 0)
        sides_changed = False

        for side in trade.get("sides", []):
            for player in side.get("players_received", []):
                mlb_id = player.get("mlb_id")
                if not mlb_id:
                    continue

                # Look up historical data via MLBAM crosswalk
                idfg = mlbam_to_idfg.get(str(mlb_id))
                if idfg is None:
                    continue
                hp = players_db.get(str(idfg))
                if hp is None:
                    continue

                to_team = player.get("to_team", "")

                # Aggregate batting + pitching WAR by year for matching team
                # Include "- - -" (multi-team split) seasons as potential matches
                war_by_year: Dict[int, float] = {}
                salary_by_year: Dict[int, int] = {}
                for season in hp.get("batting", []) + hp.get("pitching", []):
                    raw_team = season.get("team", "")
                    st = _HIST_TEAM_ALIASES.get(raw_team, raw_team)
                    is_match = (st == to_team) or (raw_team == "- - -")
                    if is_match and season["year"] >= trade_year:
                        yr = season["year"]
                        war_by_year[yr] = war_by_year.get(yr, 0) + (season.get("war") or 0)
                        salary_by_year[yr] = salary_by_year.get(yr, 0) + int(season.get("salary") or 0)

                # Always try to fix departure_year / still_on_team even if
                # the player already had WAR data from the offline pipeline
                has_existing_war = bool(player.get("yearly_war")) or abs(player.get("war_with_team", 0)) > 0.01

                if war_by_year:
                    max_year = max(war_by_year.keys())
                    still_on = max_year >= CURRENT_YEAR
                    departure = None if still_on else max_year + 1

                    # Update departure_year and still_on_team regardless
                    player["still_on_team"] = still_on
                    player["departure_year"] = departure

                    if not has_existing_war:
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
                    else:
                        # Even if WAR was already correct, check if side totals
                        # need recalculation (departure fix can change display)
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


def _attach_future_projections(trades: List[Dict[str, Any]]) -> None:
    """
    For actual (non-projected) trades, attach projected future WAR to players
    who are still under team control.  This must run AFTER
    _augment_with_historical_war so that still_on_team is already set.
    """
    projections = _load_surplus_projections()
    if not projections:
        return

    attached = 0
    for trade in trades:
        if trade.get("evaluation_type") == "projected":
            continue  # already handled by _augment_with_projections
        for side in trade.get("sides", []):
            for player in side.get("players_received", []):
                if not player.get("still_on_team"):
                    continue
                mlb_id = player.get("mlb_id")
                proj = projections.get(mlb_id) if mlb_id else None
                if not proj or not proj.get("projected_yearly_war"):
                    continue
                # Only include projected years beyond the player's last actual year
                actual_years = {yw["year"] for yw in player.get("yearly_war", [])}
                future = [
                    yw for yw in proj["projected_yearly_war"]
                    if yw["year"] not in actual_years
                ]
                if future:
                    player["projected_yearly_war"] = future
                    player["has_projection"] = True
                    attached += 1

    if attached:
        logger.info(f"Attached future projections to {attached} still-under-control players")


# ── Contract remaining at time of trade (from Cot's by-year CSVs) ────────────

_FG_TO_COTS_TEAM: Dict[str, str] = {
    "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH",
}

_cots_salary_cache: Optional[Dict[tuple, int]] = None
_cots_by_name_year: Optional[Dict[tuple, int]] = None


def _load_cots_salary() -> Dict[tuple, int]:
    """Build (name_lower, team_cots, year) → total_future_salary lookup.

    For duplicate entries (same player/team/year), take the MAX value which
    represents the full contract obligation.
    Also builds a secondary index: (name_lower, year) → max total_future_salary
    for team-agnostic fallback.
    """
    global _cots_salary_cache, _cots_by_name_year
    if _cots_salary_cache is not None:
        return _cots_salary_cache

    lookup: Dict[tuple, int] = {}
    name_year: Dict[tuple, int] = {}
    if not _SALARY_BY_YEAR_DIR.exists():
        logger.warning(f"Cot's salary directory not found: {_SALARY_BY_YEAR_DIR}")
        _cots_salary_cache = lookup
        _cots_by_name_year = name_year
        return lookup

    for year in range(2014, CURRENT_YEAR):
        fpath = _SALARY_BY_YEAR_DIR / f"{year}.csv"
        if not fpath.exists():
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("player") or "").strip()
                    team = (row.get("team") or "").strip()
                    tfs_raw = row.get("total_future_salary", "")
                    if not name or not team or not tfs_raw:
                        continue
                    # Skip aggregate rows
                    if "payroll" in name.lower() or "projected" in name.lower():
                        continue
                    try:
                        tfs = int(float(tfs_raw))
                    except (ValueError, TypeError):
                        continue
                    if tfs <= 0:
                        continue
                    name_lc = name.lower()
                    key = (name_lc, team, year)
                    # Keep the maximum (main contract, not deferred splits)
                    if key not in lookup or tfs > lookup[key]:
                        lookup[key] = tfs
                    # Secondary index: max across all teams
                    ny_key = (name_lc, year)
                    if ny_key not in name_year or tfs > name_year[ny_key]:
                        name_year[ny_key] = tfs
        except Exception as e:
            logger.warning(f"Could not load Cot's {year}: {e}")

    logger.info(f"Loaded {len(lookup)} Cot's salary entries for contract lookup")
    _cots_salary_cache = lookup
    _cots_by_name_year = name_year
    return lookup


def _attach_contract_remaining(trades: List[Dict[str, Any]]) -> None:
    """
    Attach ``contract_remaining`` to each non-prospect player in a trade.
    Uses the Cot's by-year data to find the total remaining salary obligation
    at the time the player was traded.  The lookup checks the sending team
    (``from_team``) in the trade year, falling back to next year for
    off-season trades (Nov/Dec).
    """
    cots = _load_cots_salary()
    if not cots:
        return
    name_year_idx = _cots_by_name_year or {}

    attached = 0
    for trade in trades:
        trade_year = trade.get("year")
        trade_date = trade.get("date", "")
        if not trade_year:
            continue
        # For off-season trades, the contract data may be under next year
        trade_month = 0
        try:
            trade_month = int(trade_date.split("-")[1]) if "-" in trade_date else 0
        except (IndexError, ValueError):
            pass
        is_offseason = trade_month >= 11 or trade_month <= 2

        for side in trade.get("sides", []):
            for player in side.get("players_received", []):
                name = (player.get("name") or "").strip().lower()
                from_team_fg = player.get("from_team", "")
                if not name or not from_team_fg:
                    continue
                # Convert FG team abbreviation to Cot's format
                from_team_cots = _FG_TO_COTS_TEAM.get(from_team_fg, from_team_fg)
                # Also handle OAK→ATH for 2025+
                if from_team_fg == "OAK" and trade_year >= 2025:
                    from_team_cots = "ATH"

                # Primary lookup: trade year, from-team
                key = (name, from_team_cots, trade_year)
                val = cots.get(key)

                # Fallback: next year for off-season trades
                if val is None and is_offseason:
                    key2 = (name, from_team_cots, trade_year + 1)
                    val = cots.get(key2)

                # Fallback: any team for this name+year (covers team abbreviation mismatches)
                if val is None:
                    val = name_year_idx.get((name, trade_year))
                if val is None and is_offseason:
                    val = name_year_idx.get((name, trade_year + 1))

                if val is not None and val > 0:
                    player["contract_remaining"] = val
                    attached += 1

    if attached:
        logger.info(f"Attached contract_remaining to {attached} trade players")


# ── FV → dollar value fallback (median by FV grade from prospect model) ──────

_FV_VALUE_MAP: Dict[int, int] = {
    35: 3_000_000,
    40: 7_000_000,
    45: 15_000_000,
    50: 30_000_000,
    55: 60_000_000,
    60: 100_000_000,
    65: 135_000_000,
    70: 200_000_000,
    75: 275_000_000,
    80: 350_000_000,
}


def _estimate_prospect_value(fv: int) -> int:
    """Return a dollar value estimate for a prospect based on FV grade."""
    if fv in _FV_VALUE_MAP:
        return _FV_VALUE_MAP[fv]
    # Interpolate between nearest known grades
    grades = sorted(_FV_VALUE_MAP.keys())
    if fv <= grades[0]:
        return _FV_VALUE_MAP[grades[0]]
    if fv >= grades[-1]:
        return _FV_VALUE_MAP[grades[-1]]
    for lo, hi in zip(grades, grades[1:]):
        if lo <= fv <= hi:
            frac = (fv - lo) / (hi - lo)
            return int(_FV_VALUE_MAP[lo] + frac * (_FV_VALUE_MAP[hi] - _FV_VALUE_MAP[lo]))
    return _FV_VALUE_MAP.get(50, 30_000_000)


def _augment_with_prospect_values(trades: List[Dict[str, Any]]) -> None:
    """
    Enrich trade players with prospect dollar values and fix prospect_top_100.
    
    For players with prospect_fv:
      - Attempt DB lookup by name + trade year for exact value
      - Fall back to FV-based estimate
      - Fix prospect_top_100 (offline pipeline bug: bool(NaN) == True)
      - Clear salary for prospects with 0 MLB seasons (pre-arb, not rostered)
    """
    from app.database import SessionLocal

    # Build a (name_lower, year) → value lookup from the prospect DB
    db = SessionLocal()
    try:
        all_prospects = db.query(Prospect).all()
    finally:
        db.close()

    prospect_db: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for p in all_prospects:
        name_lower = p.name.strip().lower() if p.name else ""
        year = p.year
        val = p.get_value(year)
        fv_str = p.fv
        # Also store the DB FV for cross-reference
        prospect_db[(name_lower, year)] = {
            "value": val,
            "fv": fv_str,
        }

    enriched = 0
    t100_fixed = 0

    for trade in trades:
        trade_year = trade.get("year", 0)
        for side in trade.get("sides", []):
            for player in side.get("players_received", []):
                prospect_fv = player.get("prospect_fv")
                if not prospect_fv:
                    continue

                # ------- Fix prospect_top_100 -------
                # The offline pipeline has a bug: bool(NaN) == True
                # If prospect_top_100 is True but prospect has no top-100 rank,
                # it's a false positive. We reset it.
                # A true top-100 prospect would have a top_100_rank in the CSV.
                # Since we can't access the CSV at runtime, use FV as heuristic:
                # Generally only FV >= 50 are top-100 candidates, and even then
                # not guaranteed. But the key indicator is: the pipeline stored
                # top_100=True for ALL prospects because of the NaN bug.
                # Safe fix: treat prospect_top_100 as False unless we see
                # prospect_rank ≤ 10 AND fv >= 50 (org top-3 with solid FV).
                # Actually better: just set it to False. The pipeline fix will
                # generate correct data on the next run.
                if player.get("prospect_top_100") is True:
                    # Heuristic: a real top-100 prospect typically has FV >= 50
                    # and the rank field from the CSV doesn't discriminate
                    # between org rank and national rank. Safest to clear.
                    player["prospect_top_100"] = False
                    t100_fixed += 1

                # ------- Add prospect_value -------
                name_lower = player.get("name", "").strip().lower()

                # Try exact year, then adjacent years
                db_entry = None
                for yr_offset in [0, -1, 1, -2, 2]:
                    lookup_year = trade_year + yr_offset
                    db_entry = prospect_db.get((name_lower, lookup_year))
                    if db_entry and db_entry["value"]:
                        break

                if db_entry and db_entry["value"]:
                    player["prospect_value"] = int(db_entry["value"])
                else:
                    # Fallback to FV-based estimate
                    player["prospect_value"] = _estimate_prospect_value(int(prospect_fv))

                # ------- Clear salary for non-rostered prospects -------
                if player.get("seasons_with_team", 0) == 0 and player.get("war_with_team", 0) == 0:
                    player["salary_with_team"] = 0
                    player["war_value"] = 0
                    # Surplus for a pure prospect = prospect_value
                    player["surplus"] = player["prospect_value"]

                enriched += 1

    # Recalculate side totals after prospect enrichment
    for trade in trades:
        for side in trade.get("sides", []):
            players_list = side.get("players_received", [])
            side["total_salary"] = sum(p.get("salary_with_team", 0) for p in players_list)
            side["total_war_value"] = sum(p.get("war_value", 0) for p in players_list)
            side["total_surplus"] = sum(p.get("surplus", 0) for p in players_list)

        # Recalculate winner/loser
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

    logger.info(
        f"Enriched {enriched} trade players with prospect values, "
        f"fixed {t100_fixed} false top-100 flags"
    )


def _compute_confidence_and_featured(trades: List[Dict[str, Any]]) -> None:
    """
    Add evaluation_confidence tiers and is_featured flags to trades.
    Modifies trades in-place.

    Confidence tiers:
      - "definitive" — trade year ≤ CURRENT_YEAR-4 (4+ years of data)
      - "maturing"   — trade year ≤ CURRENT_YEAR-2 (2-3 years)
      - "early"      — recent actual trades (1 year of data)
      - "projected"  — evaluation_type == "projected" (no actual data)

    Featured flag — set for notable/blockbuster trades based on heuristics.
    """
    for trade in trades:
        trade_year = trade.get("year", 0)
        eval_type = trade.get("evaluation_type", "actual")

        # ── Confidence tier ───────────────────────────────────────────────
        if eval_type == "projected":
            trade["evaluation_confidence"] = "projected"
        elif trade_year <= CURRENT_YEAR - 4:
            trade["evaluation_confidence"] = "definitive"
        elif trade_year <= CURRENT_YEAR - 2:
            trade["evaluation_confidence"] = "maturing"
        else:
            trade["evaluation_confidence"] = "early"

    # ── Featured flag: top trades by absolute surplus differential ───────
    sorted_by_surplus = sorted(
        trades,
        key=lambda t: abs(t.get("surplus_diff", 0)),
        reverse=True,
    )
    featured_threshold = max(50, len(trades) // 10)
    featured_set = set()
    for t in sorted_by_surplus[:featured_threshold]:
        featured_set.add(t.get("trade_id"))

    # Also include any trade with very large surplus, many players,
    # or a top prospect (FV >= 60)
    for t in trades:
        surplus = abs(t.get("surplus_diff", 0))
        max_fv = t.get("max_prospect_fv") or 0
        n_players = t.get("n_players", 0)
        total_war = abs(t.get("total_trade_war", 0))
        if surplus >= 30_000_000 or max_fv >= 60 or total_war >= 10 or n_players >= 5:
            featured_set.add(t.get("trade_id"))

    for t in trades:
        t["is_featured"] = t.get("trade_id") in featured_set

    featured_count = sum(1 for t in trades if t.get("is_featured"))
    logger.info(
        f"Computed confidence tiers and featured flags "
        f"({featured_count} featured out of {len(trades)} trades)"
    )


def _add_has_data_flags(trades: List[Dict[str, Any]]) -> None:
    """
    Mark trade players with has_data=False when we have no meaningful
    information about their value — no WAR data, no projection, no prospect
    value.  This lets the frontend show "No Data" instead of misleading zeros.

    Modifies trades in-place.
    """
    flagged = 0
    for trade in trades:
        for side in trade.get("sides", []):
            for player in side.get("players_received", []):
                has_war = abs(player.get("war_with_team", 0)) > 0.01
                has_projection = player.get("has_projection", False)
                has_prospect_val = player.get("prospect_value") is not None and player.get("prospect_value", 0) > 0
                has_seasons = player.get("seasons_with_team", 0) > 0

                if has_war or has_projection or has_prospect_val or has_seasons:
                    player["has_data"] = True
                else:
                    player["has_data"] = False
                    flagged += 1

    logger.info(f"Flagged {flagged} trade players with has_data=False")


def _link_prospect_ids(trades: List[Dict[str, Any]]) -> None:
    """
    Cross-reference trade players against the Prospect table by MLB ID → IDfg
    mapping and by name.  Attaches ``prospect_id`` (the Prospect table PK) so
    the frontend can link pure prospects to their detail pages.

    Also attempts to enrich players who have no prospect_fv in raw data but
    ARE in the prospect DB — fixes the gap where the offline pipeline didn't
    tag them as prospects.

    Modifies trades in-place.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        all_prospects = db.query(Prospect).all()
    finally:
        db.close()

    # Build lookups: (name_lower, year) → prospect row, and idfg → prospect row
    name_year_lookup: Dict[Tuple[str, int], Any] = {}
    idfg_lookup: Dict[int, Any] = {}
    for p in all_prospects:
        nl = p.name.strip().lower() if p.name else ""
        yr = p.year
        key = (nl, yr)
        # Keep the most recent year entry per name
        if key not in name_year_lookup or p.year > name_year_lookup[key].year:
            name_year_lookup[key] = p
        if p.IDfg:
            # Keep the most recent entry per IDfg
            if p.IDfg not in idfg_lookup or p.year > idfg_lookup[p.IDfg].year:
                idfg_lookup[p.IDfg] = p

    # Also need MLBAM → IDfg from historical data for cross-referencing
    try:
        from app.routes import historical as _hist_mod
        _hist_mod._load_historical()
        mlbam_to_idfg = _hist_mod._mlbam_to_idfg or {}
    except Exception:
        mlbam_to_idfg = {}

    linked = 0
    enriched = 0

    for trade in trades:
        trade_year = trade.get("year", 0)
        for side in trade.get("sides", []):
            for player in side.get("players_received", []):
                name_lower = player.get("name", "").strip().lower()
                mlb_id = player.get("mlb_id")

                # Try to find prospect by IDfg (via MLBAM crosswalk)
                prospect = None
                if mlb_id:
                    idfg_str = mlbam_to_idfg.get(str(mlb_id))
                    if idfg_str:
                        prospect = idfg_lookup.get(int(idfg_str))

                # Fallback: name + year matching (try exact year, then adjacent)
                if not prospect:
                    for yr_offset in [0, -1, 1, -2, 2]:
                        prospect = name_year_lookup.get((name_lower, trade_year + yr_offset))
                        if prospect:
                            break

                if prospect:
                    player["prospect_id"] = prospect.id
                    linked += 1

                    # If the raw data didn't tag this player as a prospect but
                    # they ARE in the prospect DB, enrich them now
                    if not player.get("prospect_fv") and prospect.fv:
                        try:
                            fv_int = int(prospect.fv)
                            player["prospect_fv"] = fv_int
                            # Calculate prospect value
                            val = prospect.get_value(prospect.year)
                            if val:
                                player["prospect_value"] = int(val)
                            else:
                                player["prospect_value"] = _estimate_prospect_value(fv_int)
                            # If they're a pure prospect (0 WAR, 0 seasons), fix surplus
                            if player.get("seasons_with_team", 0) == 0 and player.get("war_with_team", 0) == 0:
                                player["salary_with_team"] = 0
                                player["war_value"] = 0
                                player["surplus"] = player["prospect_value"]
                            enriched += 1
                        except (ValueError, TypeError):
                            pass

    # Recalculate side totals after prospect enrichment
    if enriched > 0:
        for trade in trades:
            for side in trade.get("sides", []):
                players_list = side.get("players_received", [])
                side["total_salary"] = sum(p.get("salary_with_team", 0) for p in players_list)
                side["total_war_value"] = sum(p.get("war_value", 0) for p in players_list)
                side["total_surplus"] = sum(p.get("surplus", 0) for p in players_list)

            # Recalculate winner/loser
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

    logger.info(
        f"Linked {linked} prospect IDs to trade players, "
        f"enriched {enriched} previously-untagged prospects"
    )


def _load_past_trades() -> List[Dict[str, Any]]:
    """Load & augment the trade evaluations from JSON.

    Kept for backward-compatibility — called by ``data_loader`` during
    ingestion.  Runtime endpoints now read from the DB.
    """
    global _past_trades_cache
    if _past_trades_cache is not None:
        return _past_trades_cache

    if not _PAST_TRADES_FILE.exists():
        logger.warning(f"Past trades file not found: {_PAST_TRADES_FILE}")
        return []

    with open(_PAST_TRADES_FILE, "r") as f:
        _past_trades_cache = json.load(f)

    _augment_with_projections(_past_trades_cache)
    _augment_with_historical_war(_past_trades_cache)
    _attach_future_projections(_past_trades_cache)
    _attach_contract_remaining(_past_trades_cache)
    _augment_with_prospect_values(_past_trades_cache)
    _link_prospect_ids(_past_trades_cache)
    _add_has_data_flags(_past_trades_cache)
    _compute_confidence_and_featured(_past_trades_cache)

    logger.info(f"Loaded {len(_past_trades_cache)} past trades (for ingestion)")
    return _past_trades_cache


# ═══════════════════════════════════════════════════════════════════════════════
#  DB-backed past-trade endpoints
# ═══════════════════════════════════════════════════════════════════════════════

from app.models.past_trade import PastTrade

_SORT_COLUMN_MAP = {
    "date": PastTrade.date,
    "surplus_diff": PastTrade.surplus_diff,
    "total_trade_war": PastTrade.total_trade_war,
    "max_prospect_fv": PastTrade.max_prospect_fv,
    "n_players": PastTrade.n_players,
}


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
    featured: Optional[bool] = None,
    confidence: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all evaluated past trades with sorting, filtering, and pagination."""
    query = db.query(PastTrade)

    # ── Filtering ────────────────────────────────────────────────────────
    if team:
        team_upper = team.upper()
        query = query.filter(PastTrade.teams_csv.contains(team_upper))

    if year:
        query = query.filter(PastTrade.year == year)

    if min_war is not None:
        query = query.filter(PastTrade.total_trade_war >= min_war)

    if featured is not None:
        query = query.filter(PastTrade.is_featured == featured)

    if confidence:
        query = query.filter(PastTrade.evaluation_confidence == confidence)

    if search:
        search_lower = search.lower()
        query = query.filter(
            PastTrade.player_names_lower.contains(search_lower)
            | PastTrade.description.ilike(f"%{search_lower}%")
        )

    # ── Sorting ──────────────────────────────────────────────────────────
    col = _SORT_COLUMN_MAP.get(sort_by, PastTrade.date)
    order_fn = desc if sort_dir == "desc" else asc
    query = query.order_by(order_fn(col))

    # ── Pagination ───────────────────────────────────────────────────────
    total = query.count()
    offset = (page - 1) * page_size
    page_items = query.offset(offset).limit(page_size).all()

    return {
        "trades": [t.to_summary_dict() for t in page_items],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/past-trades/{trade_id}")
def get_past_trade_detail(trade_id: int, db: Session = Depends(get_db)):
    """Get full details for a single past trade, including yearly WAR."""
    trade = db.query(PastTrade).filter(PastTrade.trade_id == trade_id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade.to_full_dict()


@router.get("/player-trades/{mlb_id}")
def get_player_past_trades(mlb_id: int, db: Session = Depends(get_db)):
    """Get all past trades involving a specific player (by mlb_id)."""
    mlb_str = str(mlb_id)
    trades = (
        db.query(PastTrade)
        .filter(PastTrade.player_mlb_ids_csv.contains(mlb_str))
        .order_by(PastTrade.date.desc())
        .all()
    )
    return {"trades": [t.to_full_dict() for t in trades]}