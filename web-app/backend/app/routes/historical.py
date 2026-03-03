"""
Historical player endpoints.

Serves career data for all 13,000+ players in FanGraphs data (1950-2025).
Data is loaded from the *historical_players* DB table (populated by
``data_loader.init_db``).  The old in-memory JSON cache is no longer used
at runtime — salary-augmented JSON is ingested once into the DB.

NOTE: The ``_load_historical`` and salary-loading helpers below are kept
for use by the **ingestion script** (``data_loader.py``) and by the
trade-WAR augmentation.  They are NOT called by any endpoint.
"""

import csv as _csv
import json
import time
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional
import logging

from app.database import get_db
from app.models.historical import HistoricalPlayer

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Legacy file paths (used by ingestion helpers only) ───────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # LSTMLB
_HIST_FILE = _PROJECT_ROOT / "data" / "generated" / "historical_players" / "historical_players.json"
_SALARY_BY_YEAR_DIR = _PROJECT_ROOT / "data" / "salary" / "by_year"


# ═══════════════════════════════════════════════════════════════════════════
# Ingestion helpers (called by data_loader and trade augmentation only)
# ═══════════════════════════════════════════════════════════════════════════

# In-memory caches — populated ONLY during ingestion / trade augmentation.
_hist_data: Optional[Dict[str, Any]] = None
_players: Dict[str, Any] = {}
_mlbam_to_idfg: Dict[str, int] = {}
_name_index: List[Dict[str, Any]] = []


def _load_salary_supplement() -> Dict[tuple, int]:
    """Load by_year salary CSVs → {(name_lower, team, year): salary_int}."""
    lookup: Dict[tuple, int] = {}
    if not _SALARY_BY_YEAR_DIR.exists():
        return lookup
    for fname in sorted(_SALARY_BY_YEAR_DIR.iterdir()):
        if not fname.suffix == ".csv":
            continue
        try:
            yr = int(fname.stem)
        except ValueError:
            continue
        with open(fname, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                name = (row.get("player") or "").replace("*", "").strip().lower()
                team = (row.get("team") or "").strip()
                sal_str = row.get("salary", "")
                if not name or not team or not sal_str:
                    continue
                try:
                    sal = int(float(sal_str))
                except (ValueError, TypeError):
                    continue
                if sal > 0:
                    lookup[(name, team, yr)] = sal
    return lookup


_LAHMAN_FILE = _PROJECT_ROOT / "data" / "salary" / "lahman_salaries.csv"
_LAHMAN_TEAM_MAP: Dict[str, str] = {
    "ANA": "LAA", "CAL": "LAA", "CHA": "CHW", "CHN": "CHC",
    "FLO": "FLA", "KCA": "KCR", "LAN": "LAD", "ML4": "MIL",
    "NYA": "NYY", "NYN": "NYM", "SDN": "SDP", "SFN": "SFG",
    "SLN": "STL", "TBA": "TBR", "WAS": "WSN", "MON": "MON",
}


def _load_lahman_salaries() -> Dict[tuple, int]:
    """Load Lahman salary data → {(bbref_id, year): salary_int}."""
    lookup: Dict[tuple, int] = {}
    if not _LAHMAN_FILE.exists():
        return lookup
    with open(_LAHMAN_FILE, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            bbref = (row.get("playerID") or "").strip()
            yr_str = row.get("yearID", "")
            sal_str = row.get("salary", "")
            if not bbref or not yr_str or not sal_str:
                continue
            try:
                yr = int(yr_str)
                sal = int(float(sal_str))
            except (ValueError, TypeError):
                continue
            if sal > 0:
                lookup[(bbref, yr)] = sal
    return lookup


_SPOTRAC_FILE = _PROJECT_ROOT / "data" / "salary" / "mlb_salary_data.csv"
_SPOTRAC_TEAM_MAP: Dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Florida Marlins": "FLA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Tampa Bay Devil Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN", "Montreal Expos": "MON",
}


def _load_spotrac_salaries() -> Dict[tuple, int]:
    """Load Spotrac salary data → {(name_lower, team_abbrev, year): salary_int}."""
    lookup: Dict[tuple, int] = {}
    if not _SPOTRAC_FILE.exists():
        return lookup
    with open(_SPOTRAC_FILE, "r", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            name = ""
            for key in row:
                if "player_name" in key:
                    name = (row[key] or "").strip().lower()
                    break
            team_full = (row.get("team") or "").strip()
            yr_str = row.get("year", "")
            sal_str = row.get("payroll_annual", "") or row.get("cash_annual", "")
            if not name or not team_full or not yr_str or not sal_str:
                continue
            team_abbrev = _SPOTRAC_TEAM_MAP.get(team_full, "")
            if not team_abbrev:
                continue
            try:
                yr = int(yr_str)
                sal = int(float(sal_str.replace("$", "").replace(",", "")))
            except (ValueError, TypeError):
                continue
            if sal > 0:
                key = (name, team_abbrev, yr)
                if key not in lookup:
                    lookup[key] = sal
    return lookup


def _load_historical():
    """Load historical player data into module-level globals.

    Used by the trade-WAR augmentation during ingestion.  NOT called by any
    endpoint at runtime.
    """
    global _hist_data, _players, _mlbam_to_idfg, _name_index

    if _hist_data is not None:
        return

    if not _HIST_FILE.exists():
        logger.warning(f"Historical players file not found: {_HIST_FILE}")
        _hist_data = {}
        return

    t0 = time.time()
    with open(_HIST_FILE, "r") as f:
        _hist_data = json.load(f)

    _players = _hist_data.get("players", {})
    _mlbam_to_idfg = _hist_data.get("mlbam_to_idfg", {})

    _name_index.clear()
    for idfg_str, p in _players.items():
        _name_index.append({
            "idfg": p["idfg"],
            "mlbam": p.get("mlbam"),
            "name": p["name"],
            "name_lower": p["name"].lower(),
            "teams": p.get("teams", []),
            "first_year": p.get("first_year"),
            "last_year": p.get("last_year"),
            "career_war": p.get("career_war", 0),
            "is_pitcher": p.get("is_pitcher", False),
        })
    _name_index.sort(key=lambda x: x["career_war"], reverse=True)

    elapsed = time.time() - t0
    logger.info(
        f"Loaded historical players (for ingestion): {len(_players)} players ({elapsed:.1f}s)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public helper — used by other route modules (e.g. player detail page)
# ═══════════════════════════════════════════════════════════════════════════

def get_historical_player(player_id: int, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
    """Look up a historical player by IDfg or MLBAM ID.

    Queries the DB.  If no *db* session is provided, creates a short-lived one.
    """
    from app.database import SessionLocal

    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        hp = db.query(HistoricalPlayer).filter(HistoricalPlayer.idfg == player_id).first()
        if hp:
            return hp.to_full_dict()
        hp = db.query(HistoricalPlayer).filter(HistoricalPlayer.mlbam == player_id).first()
        if hp:
            return hp.to_full_dict()
        return None
    finally:
        if close_db:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints (DB-backed)
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/search")
async def search_historical_players(
    q: str = Query("", description="Search query (name)"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    min_war: Optional[float] = Query(None, description="Minimum career WAR"),
    decade: Optional[int] = Query(None, description="Active during this decade (e.g. 1990)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Search historical players by name, team, WAR, decade."""
    query = db.query(HistoricalPlayer)

    q_lower = q.strip().lower()
    if q_lower:
        query = query.filter(HistoricalPlayer.name_lower.contains(q_lower))
    if min_war is not None:
        query = query.filter(HistoricalPlayer.career_war >= min_war)
    if decade is not None:
        query = query.filter(
            HistoricalPlayer.first_year <= decade + 9,
            HistoricalPlayer.last_year >= decade,
        )

    query = query.order_by(HistoricalPlayer.career_war.desc())
    rows = query.all()

    # Team filter in Python (teams is a JSON array)
    if team:
        team_upper = team.upper()
        rows = [r for r in rows if team_upper in (r.teams or [])]

    total = len(rows)
    page = rows[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "players": [r.to_search_dict() for r in page],
    }


@router.get("/{player_id}")
async def get_historical_player_detail(
    player_id: int,
    db: Session = Depends(get_db),
):
    """Get full historical player data by IDfg or MLBAM ID."""
    player = get_historical_player(player_id, db=db)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Historical player not found: {player_id}")
    return JSONResponse(player)
