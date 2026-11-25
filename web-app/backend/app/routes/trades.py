import csv
import json
import math
import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, or_
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, model_validator
import logging

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.database import get_db
from app.models.player import Player
from app.models.prospect import Prospect
from app.config import CURRENT_YEAR, PROSPECT_DEFAULT_YEAR, PROSPECT_YEAR_START, PROSPECT_YEAR_END

router = APIRouter()
logger = logging.getLogger(__name__)

_CALIBRATION_FILE = (
    Path(__file__).resolve().parents[4] / "data" / "generated" / "valuation_calibration.json"
)


def _get_package_gamma() -> float:
    """Load empirically calibrated package consolidation exponent (default 1.211)."""
    if _CALIBRATION_FILE.exists():
        try:
            with open(_CALIBRATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("package_gamma", 1.211))
        except Exception:
            pass
    return 1.211


def _consolidate_package(asset_values: List[float], gamma: float) -> Tuple[float, float, float]:
    """Sub-additive L_gamma norm for positive-surplus assets; linear sum for salary dumps."""
    clean = [float(v) for v in asset_values if v is not None and not math.isnan(float(v))]
    if not clean:
        return 0.0, 0.0, 0.0
    raw_sum = sum(clean)
    pos = [v for v in clean if v > 0.0]
    neg = [v for v in clean if v <= 0.0]
    if len(pos) <= 1:
        pos_c = sum(pos)
    else:
        g = max(1.0, float(gamma))
        pos_c = sum(v ** g for v in pos) ** (1.0 / g)
    consolidated = pos_c + sum(neg)
    penalty = consolidated - raw_sum
    return consolidated, raw_sum, penalty


class TradeAsset(BaseModel):
    name: str
    isProspect: bool
    team: Optional[str] = None
    fg_id: Optional[int] = None
    mlbam_id: Optional[int] = None
    prospect_id: Optional[int] = None


class TradeRequest(BaseModel):
    side_a_assets: List[TradeAsset] = []
    side_b_assets: List[TradeAsset] = []

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "side_a_assets" not in data and "team1_assets" in data:
                data = {**data, "side_a_assets": data["team1_assets"]}
            if "side_b_assets" not in data and "team2_assets" in data:
                data = {**data, "side_b_assets": data["team2_assets"]}
        return data


def get_player_values(asset: TradeAsset, db: Session) -> Dict[str, Any]:
    q = db.query(Player).filter(
        Player.year == CURRENT_YEAR,
        Player.projection_type == "ros",
    )
    base_player = None
    if asset.fg_id is not None:
        base_player = q.filter(Player.fg_id == asset.fg_id).first()
    if base_player is None and asset.mlbam_id is not None:
        base_player = q.filter(Player.mlbam_id == asset.mlbam_id).first()
    if base_player is None:
        matches = q.filter(Player.name == asset.name).all()
        if len(matches) == 1:
            base_player = matches[0]
        elif len(matches) > 1:
            if asset.team:
                t_up = asset.team.upper()
                base_player = next((m for m in matches if (m.team or "").upper() == t_up), matches[0])
            else:
                base_player = matches[0]

    if not base_player:
        raise HTTPException(status_code=404, detail=f"Player {asset.name} not found")

    timeline_rows = (
        db.query(Player.year, Player.status)
        .filter(
            Player.fg_id == base_player.fg_id,
            Player.year >= CURRENT_YEAR,
            Player.projection_type == "ros",
        )
        .order_by(Player.year.asc())
        .all()
    )
    projections_data = [{"year": r.year, "status": r.status or ""} for r in timeline_rows]
    fa_year = base_player.fa_year or ((base_player.control_through + 1) if base_player.control_through else None)

    return {
        "type": "player",
        "id": base_player.mlbam_id or base_player.fg_id,
        "fg_id": base_player.fg_id,
        "mlbam_id": base_player.mlbam_id,
        "name": base_player.name,
        "team": base_player.team,
        "position": base_player.position,
        "control_war": base_player.control_war or 0,
        "trade_value": base_player.trade_value or 0,
        "trade_value_p10": base_player.trade_value_p10,
        "trade_value_p90": base_player.trade_value_p90,
        "control_salary_cost": base_player.control_salary_cost or 0,
        "control_production_value": base_player.control_production_value or 0,
        "years_control": base_player.years_control or 0,
        "probable_fa_year": base_player.probable_fa_year or fa_year,
        "fa_year": fa_year,
        "projections": projections_data,
    }


def get_prospect_values(asset: TradeAsset, db: Session) -> Dict[str, Any]:
    q = db.query(Prospect).filter(Prospect.year == PROSPECT_DEFAULT_YEAR)
    prospect = None
    if asset.prospect_id is not None:
        prospect = q.filter(Prospect.id == asset.prospect_id).first()
    if prospect is None and asset.mlbam_id is not None:
        prospect = q.filter(Prospect.mlbam_id == asset.mlbam_id).first()
    if prospect is None:
        matches = q.filter(Prospect.name == asset.name).all()
        if len(matches) == 1:
            prospect = matches[0]
        elif len(matches) > 1:
            if asset.team:
                t_up = asset.team.upper()
                prospect = next((m for m in matches if (m.org or "").upper() == t_up), matches[0])
            else:
                prospect = matches[0]

    if not prospect:
        raise HTTPException(status_code=404, detail=f"Prospect {asset.name} not found")

    val = prospect.get_value(PROSPECT_DEFAULT_YEAR) or 0

    return {
        "type": "prospect",
        "id": prospect.mlbam_id or prospect.id,
        "prospect_id": prospect.id,
        "fg_id": prospect.fg_id,
        "mlbam_id": prospect.mlbam_id,
        "name": prospect.name,
        "team": prospect.org,
        "position": prospect.position,
        "level": prospect.level,
        "eta": prospect.eta,
        "fv": prospect.fv,
        "trade_value": val,
        "trade_value_p10": prospect.trade_value_p10,
        "trade_value_p90": prospect.trade_value_p90,
        "control_salary_cost": 0,
        "control_production_value": val,
        "years_control": 6,
        "probable_fa_year": None,
        "fa_year": None,
        "projections": [],
    }


def _summarize_trade_side(assets: List[Dict[str, Any]], gamma: float) -> Dict[str, Any]:
    surpluses = [a["trade_value"] or 0.0 for a in assets]
    consolidated, raw_sum, penalty = _consolidate_package(surpluses, gamma)

    p10_vals = [a["trade_value_p10"] if a.get("trade_value_p10") is not None else (a["trade_value"] or 0.0) for a in assets]
    p90_vals = [a["trade_value_p90"] if a.get("trade_value_p90") is not None else (a["trade_value"] or 0.0) for a in assets]
    c_p10, _, _ = _consolidate_package(p10_vals, gamma)
    c_p90, _, _ = _consolidate_package(p90_vals, gamma)

    return {
        "consolidated_surplus": round(consolidated),
        "raw_surplus_sum": round(raw_sum),
        "consolidation_penalty": round(penalty),
        "surplus_p10": round(c_p10),
        "surplus_p90": round(c_p90),
        "control_salary_cost": round(sum(a["control_salary_cost"] or 0 for a in assets)),
        "control_production_value": round(sum(a["control_production_value"] or 0 for a in assets)),
        "assets": assets,
    }


@router.post("/analyze")
def analyze_trade(trade: TradeRequest, db: Session = Depends(get_db)):
    try:
        gamma = _get_package_gamma()

        side_a_assets = [
            get_prospect_values(a, db) if a.isProspect else get_player_values(a, db)
            for a in trade.side_a_assets
        ]
        side_b_assets = [
            get_prospect_values(a, db) if a.isProspect else get_player_values(a, db)
            for a in trade.side_b_assets
        ]

        return {
            "side_a": _summarize_trade_side(side_a_assets, gamma),
            "side_b": _summarize_trade_side(side_b_assets, gamma),
            "package_gamma": gamma,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prospects")
async def get_all_prospects(
    player_type: str = Query(..., description="Either 'hitter' or 'pitcher'"),
    year: int = Query(PROSPECT_DEFAULT_YEAR, ge=PROSPECT_YEAR_START, le=PROSPECT_YEAR_END),
    db: Session = Depends(get_db),
):
    from app.routes.prospects import get_prospects
    result = await get_prospects(
        player_type=player_type, year=year,
        page=1, page_size=500, slim=True, db=db,
    )
    return {"players": result["players"]}


@router.get("/trade-val-rank")
def get_trade_value_rankings(
    db: Session = Depends(get_db),
    team: Optional[str] = None,
    position: Optional[str] = None,
    age_operator: Optional[str] = None,
    age_value: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    sort_by: str = Query(
        "trade_value",
        pattern="^(trade_value|trade_value_p10|trade_value_p90|control_war|avg_war|control_salary_cost|avg_salary_cost|control_through|years_control|projected_15yr_war|projected_15yr_value|past_career_war|past_career_value|control_production_value)$",
    ),
    sort_direction: str = Query("desc", pattern="^(asc|desc)$"),
):
    try:
        query = db.query(Player).filter(
            Player.year == CURRENT_YEAR,
            Player.projection_type == "ros",
            Player.trade_value.isnot(None),
        )

        if team:
            if team.upper() == "FA":
                query = query.filter(Player.team == "FA")
            else:
                query = query.filter(func.upper(Player.team) == team.upper())

        if position:
            if position == "OF":
                query = query.filter(or_(
                    Player.position.in_(["LF", "CF", "RF"]),
                    Player.position.like("%/LF%"),
                    Player.position.like("%/CF%"),
                    Player.position.like("%/RF%"),
                    Player.position.like("LF/%"),
                    Player.position.like("CF/%"),
                    Player.position.like("RF/%"),
                ))
            else:
                query = query.filter(or_(
                    Player.position == position,
                    Player.position.like(f"{position}/%"),
                    Player.position.like(f"%/{position}"),
                ))

        if age_value is not None and age_operator:
            if age_operator == ">=":
                query = query.filter(Player.age >= age_value)
            elif age_operator == "<=":
                query = query.filter(Player.age <= age_value)
            elif age_operator == "=":
                query = query.filter(Player.age == age_value)

        sort_column = getattr(Player, sort_by)
        query = query.order_by(desc(sort_column) if sort_direction == "desc" else asc(sort_column))

        total_count = query.count()
        players = query.offset((page - 1) * page_size).limit(page_size).all()

        results = [
            {
                "fg_id": p.fg_id,
                "mlbam_id": p.mlbam_id,
                "name": p.name,
                "team": p.team,
                "position": p.position,
                "player_role": p.player_role,
                "age": p.age,
                "trade_value": p.trade_value,
                "trade_value_p10": p.trade_value_p10,
                "trade_value_p50": p.trade_value_p50,
                "trade_value_p90": p.trade_value_p90,
                "control_war": p.control_war,
                "avg_war": p.avg_war,
                "control_salary_cost": p.control_salary_cost,
                "avg_salary_cost": p.avg_salary_cost,
                "control_production_value": p.control_production_value,
                "control_surplus": p.control_surplus,
                "control_through": p.control_through,
                "years_control": p.years_control,
                "projected_15yr_war": p.projected_15yr_war,
                "projected_15yr_value": p.projected_15yr_value,
                "past_career_war": p.past_career_war,
                "past_career_value": p.past_career_value,
            }
            for p in players
        ]

        return {
            "players": results,
            "total_count": total_count,
            "total_pages": (total_count + page_size - 1) // page_size,
            "current_page": page,
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

            # Total future values (canonical V2 surplus_*.csv columns)
            try:
                total_war = float(row.get("control_war", row.get("total_future_WAR", 0)) or 0)
                total_war_value = float(row.get("control_production_value", row.get("total_future_WAR_value", 0)) or 0)
                total_salary = float(row.get("control_salary_cost", row.get("total_future_salary", 0)) or 0)
                surplus = float(row.get("trade_value", row.get("surplus", 0)) or 0)
                years_control = int(round(float(row.get("years_control", 0) or 0)))
                snap_yr = int(float(row.get("snapshot_year", CURRENT_YEAR - 1) or (CURRENT_YEAR - 1)))
                control_through = (snap_yr + years_control - 1) if years_control > 0 else 0
            except (ValueError, TypeError):
                total_war = total_war_value = total_salary = surplus = 0.0
                years_control = 0
                control_through = 0

            projections[mlbam_id] = {
                "name": row.get("player_name", row.get("Name", "")),
                "projected_war": round(total_war, 1),
                "projected_war_value": int(total_war_value),
                "projected_salary": int(total_salary),
                "projected_surplus": int(surplus),
                "projected_yearly_war": yearly_war,
                "years_control": years_control,
                "control_through": control_through or None,
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
                    player["years_control"] = proj.get("years_control")
                    player["control_through"] = proj.get("control_through")
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

    IMPORTANT: This function ONLY fills missing data. It does NOT override
    correctly calculated fields from the offline pipeline (trades.json).

    Modifies trades in-place.
    """
    from app.routes import historical as _hist_mod
    from app.config import CURRENT_YEAR

    _hist_mod._load_historical()
    if not _hist_mod._mlbam_to_idfg or not _hist_mod._players:
        return

    mlbam_to_idfg = _hist_mod._mlbam_to_idfg
    players_db = _hist_mod._players

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

                # Check if player already has correct data from offline pipeline
                has_existing_war = bool(player.get("yearly_war")) or abs(player.get("war_with_team", 0)) > 0.01
                has_existing_seasons = (player.get("seasons_with_team", 0) or 0) > 0

                if war_by_year:
                    if not has_existing_war:
                        # Only fill if data was missing
                        # Build yearly_war list sorted by year with each year rounded to 1 decimal
                        yearly = sorted(
                            [{"year": yr, "war": round(w, 1)} for yr, w in war_by_year.items()],
                            key=lambda x: x["year"],
                        )
                        # Calculate total from rounded yearly values (not raw values)
                        # This ensures war_with_team matches sum of yearly_war display
                        total_war = round(sum(yw["war"] for yw in yearly), 1)
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
                        # Player has correct war_with_team from offline pipeline
                        # Only need to check and fix still_on_team/departure_year if needed
                        # NEVER override seasons_with_team if it's already set
                        sides_changed = False

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
                
                # Use control_through from trades.json (loaded from FA year data)
                # If missing, don't show projections
                control_through = player.get("control_through")
                if not control_through:
                    continue
                
                max_proj_year = int(control_through)
                
                future = [
                    yw for yw in proj["projected_yearly_war"]
                    if yw["year"] not in actual_years
                    and yw["year"] <= max_proj_year
                ]
                if future:
                    player["projected_yearly_war"] = future
                    player["years_control"] = proj.get("years_control")
                    # control_through already set from trades.json, keep it
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
    30: 150_000,
    35: 400_000,
    40: 1_200_000,
    45: 3_500_000,
    50: 8_500_000,
    55: 19_000_000,
    60: 38_000_000,
    65: 56_000_000,
    70: 78_000_000,
    75: 98_000_000,
    80: 120_000_000,
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


def _augment_with_prospect_values(trades: List[Dict[str, Any]], db=None) -> None:
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
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        all_prospects = db.query(Prospect).all()
    finally:
        if close_db:
            db.close()

    prospect_db: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for p in all_prospects:
        name_lower = p.name.strip().lower() if p.name else ""
        year = p.year
        val = p.get_value(year)
        fv_str = p.fv
        prospect_db[(name_lower, year)] = {
            "value": val,
            "fv": fv_str,
            "top_100": p.top_100 is not None,
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

                name_lower = player.get("name", "").strip().lower()
                db_entry = None
                for yr_offset in [0, -1, 1, -2, 2]:
                    lookup_year = trade_year + yr_offset
                    db_entry = prospect_db.get((name_lower, lookup_year))
                    if db_entry and db_entry["value"]:
                        break

                if db_entry is not None:
                    player["prospect_top_100"] = bool(db_entry.get("top_100"))
                    t100_fixed += 1

                if db_entry and db_entry["value"]:
                    player["prospect_value"] = int(db_entry["value"])
                else:
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


def _link_prospect_ids(trades: List[Dict[str, Any]], db=None) -> None:
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

    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        all_prospects = db.query(Prospect).all()
    finally:
        if close_db:
            db.close()

    # Build lookups: (name_lower, year) → prospect row, and fg_id → prospect row
    name_year_lookup: Dict[Tuple[str, int], Any] = {}
    fg_lookup: Dict[str, Any] = {}
    mlbam_lookup: Dict[int, Any] = {}
    for p in all_prospects:
        nl = p.name.strip().lower() if p.name else ""
        yr = p.year
        key = (nl, yr)
        if key not in name_year_lookup or p.year > name_year_lookup[key].year:
            name_year_lookup[key] = p
        if p.fg_id:
            fg_str = str(p.fg_id)
            if fg_str not in fg_lookup or p.year > fg_lookup[fg_str].year:
                fg_lookup[fg_str] = p
        if p.mlbam_id:
            if p.mlbam_id not in mlbam_lookup or p.year > mlbam_lookup[p.mlbam_id].year:
                mlbam_lookup[p.mlbam_id] = p

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

                prospect = None
                if mlb_id:
                    prospect = mlbam_lookup.get(int(mlb_id))
                    if not prospect:
                        idfg_str = mlbam_to_idfg.get(str(mlb_id))
                        if idfg_str:
                            prospect = fg_lookup.get(str(idfg_str))

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