"""
Historical player endpoints.

Serves career data for all 13,000+ players in FanGraphs data (1950-2025).
Data comes from pre-computed JSON built by build_historical_players.py.
"""

import json
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Data file ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # LSTMLB
_HIST_FILE = _PROJECT_ROOT / "data" / "generated" / "historical_players" / "historical_players.json"
_SALARY_BY_YEAR_DIR = _PROJECT_ROOT / "data" / "salary" / "by_year"

_hist_data: Optional[Dict[str, Any]] = None
_players: Dict[str, Any] = {}
_mlbam_to_idfg: Dict[str, int] = {}
_name_index: List[Dict[str, Any]] = []  # For search


def _load_salary_supplement() -> Dict[tuple, int]:
    """Load by_year salary CSVs → {(name_lower, team, year): salary_int}."""
    import csv as _csv

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


# ── Lahman salary data (1985–2016) ──────────────────────────────────────
_LAHMAN_FILE = _PROJECT_ROOT / "data" / "salary" / "lahman_salaries.csv"

# Lahman teamID → FanGraphs abbreviation
_LAHMAN_TEAM_MAP: Dict[str, str] = {
    "ANA": "LAA", "CAL": "LAA", "CHA": "CHW", "CHN": "CHC",
    "FLO": "FLA", "KCA": "KCR", "LAN": "LAD", "ML4": "MIL",
    "NYA": "NYY", "NYN": "NYM", "SDN": "SDP", "SFN": "SFG",
    "SLN": "STL", "TBA": "TBR", "WAS": "WSN", "MON": "MON",
}


def _load_lahman_salaries() -> Dict[tuple, int]:
    """Load Lahman salary data → {(bbref_id, year): salary_int}."""
    import csv as _csv

    lookup: Dict[tuple, int] = {}
    if not _LAHMAN_FILE.exists():
        logger.warning(f"Lahman salary file not found: {_LAHMAN_FILE}")
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

    logger.info(f"Loaded {len(lookup)} Lahman salary entries")
    return lookup


# ── Spotrac salary data ─────────────────────────────────────────────────
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
    import csv as _csv

    lookup: Dict[tuple, int] = {}
    if not _SPOTRAC_FILE.exists():
        logger.warning(f"Spotrac salary file not found: {_SPOTRAC_FILE}")
        return lookup

    with open(_SPOTRAC_FILE, "r", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            # Handle BOM in first column name
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

            # Parse dollar-formatted salary: "$3,100,000" → 3100000
            try:
                yr = int(yr_str)
                sal = int(float(sal_str.replace("$", "").replace(",", "")))
            except (ValueError, TypeError):
                continue
            if sal > 0:
                # Only set if not already present (by_year CSVs take priority)
                key = (name, team_abbrev, yr)
                if key not in lookup:
                    lookup[key] = sal

    logger.info(f"Loaded {len(lookup)} Spotrac salary entries")
    return lookup


def _augment_salaries():
    """Fill null salary fields using by_year CSVs, Spotrac, and Lahman data."""
    # Load all three salary sources (priority: by_year > Spotrac > Lahman)
    name_team_year_lookup = _load_salary_supplement()  # by_year CSVs
    spotrac_lookup = _load_spotrac_salaries()          # Spotrac (name, team, yr)
    lahman_lookup = _load_lahman_salaries()             # Lahman (bbref, yr)

    # Merge Spotrac into name_team_year lookup (by_year takes priority)
    for key, sal in spotrac_lookup.items():
        if key not in name_team_year_lookup:
            name_team_year_lookup[key] = sal

    filled = 0
    lahman_filled = 0
    for _idfg_str, p in _players.items():
        name_lower = p["name"].lower().strip()
        bbref_id = p.get("bbref", "")
        career_salary_add = 0

        for season_list_key in ("batting", "pitching"):
            for s in p.get(season_list_key, []):
                if s.get("salary"):
                    continue  # already has salary

                team = s.get("team", "")
                yr = s.get("year", 0)

                # Try name+team+year lookup first (by_year + Spotrac)
                sal = name_team_year_lookup.get((name_lower, team, yr))

                # Fall back to Lahman (bbref+year)
                if sal is None and bbref_id:
                    sal = lahman_lookup.get((bbref_id, yr))
                    if sal:
                        lahman_filled += 1

                if sal:
                    s["salary"] = sal
                    # Recalculate surplus for this season
                    war_val = s.get("war_value") or 0
                    s["surplus"] = war_val - sal
                    career_salary_add += sal
                    filled += 1

        # Update career totals if we added salary
        if career_salary_add > 0:
            old = p.get("career_salary") or 0
            p["career_salary"] = old + career_salary_add
            old_surplus = p.get("career_surplus")
            if old_surplus is not None:
                p["career_surplus"] = old_surplus - career_salary_add
            else:
                war_value = p.get("career_war_value") or 0
                p["career_surplus"] = war_value - p["career_salary"]

    logger.info(
        f"Salary augmentation: filled {filled} entries "
        f"(Lahman: {lahman_filled}, by_year+Spotrac: {filled - lahman_filled})"
    )


def _load_historical():
    """Lazy-load the historical player data."""
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

    # Build lightweight name index for search
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
    # Sort by career WAR descending for default ordering
    _name_index.sort(key=lambda x: x["career_war"], reverse=True)

    # Supplement missing salary data from by_year CSVs (covers 2014–2025)
    _augment_salaries()

    elapsed = time.time() - t0
    logger.info(
        f"Loaded historical players: {len(_players)} players, "
        f"{len(_mlbam_to_idfg)} MLBAM mappings ({elapsed:.1f}s)"
    )


def get_historical_player(player_id: int) -> Optional[Dict[str, Any]]:
    """Look up a historical player by IDfg or MLBAM ID. Used by other routes."""
    _load_historical()

    # Try IDfg first
    idfg_str = str(player_id)
    if idfg_str in _players:
        return _players[idfg_str]

    # Try MLBAM lookup
    mlbam_str = str(player_id)
    if mlbam_str in _mlbam_to_idfg:
        mapped_idfg = str(_mlbam_to_idfg[mlbam_str])
        return _players.get(mapped_idfg)

    return None


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get("/search")
async def search_historical_players(
    q: str = Query("", description="Search query (name)"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    min_war: Optional[float] = Query(None, description="Minimum career WAR"),
    decade: Optional[int] = Query(None, description="Active during this decade (e.g. 1990)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Search historical players by name, team, WAR, decade."""
    _load_historical()

    results = _name_index
    q_lower = q.strip().lower()

    if q_lower:
        results = [p for p in results if q_lower in p["name_lower"]]

    if team:
        team_upper = team.upper()
        results = [p for p in results if team_upper in p["teams"]]

    if min_war is not None:
        results = [p for p in results if p["career_war"] >= min_war]

    if decade is not None:
        results = [
            p for p in results
            if p.get("first_year") and p.get("last_year")
            and p["first_year"] <= decade + 9 and p["last_year"] >= decade
        ]

    total = len(results)
    page = results[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "players": page,
    }


@router.get("/{player_id}")
async def get_historical_player_detail(player_id: int):
    """Get full historical player data by IDfg or MLBAM ID."""
    _load_historical()

    player = get_historical_player(player_id)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Historical player not found: {player_id}")

    return JSONResponse(player)
