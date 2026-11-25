"""
Canonical Player ID Crosswalk & Team Code Normalization
=======================================================

Single source of truth across all pipelines for:
  1. Bidirectional ``mlbam_id <-> fg_id`` integer resolution
     (combining ``player_id_crosswalk.csv``, ``current_rosters.csv``,
     Statcast ``sc_mlbam_id`` columns, and Chadwick register fallback).
  2. Canonical 3-letter MLB team abbreviation normalization
     (handling full team names, MLB.com slugs, FanGraphs codes, and Cot's codes).
  3. Recent scouting FV lookup keyed by ``fg_id`` for rookie talent priors.
"""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from core.name_utils import name_key

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_CROSSWALK_CSV = _DATA_DIR / "generated" / "player_id_crosswalk.csv"
_ROSTER_CSV = _DATA_DIR / "active_roster" / "current_rosters.csv"
_PROSPECT_CSV = _DATA_DIR / "prospect_data" / "prospects_2014_2026_with_top100.csv"
_REGISTER_DIR = _DATA_DIR / "register" / "data"
_HIST_BAT_STATCAST = _DATA_DIR / "historic_mlb" / "mlb_batting_data_1950_2025_with_statcast.csv"
_HIST_PIT_STATCAST = _DATA_DIR / "historic_mlb" / "mlb_pitching_data_1950_2025_with_statcast.csv"

# ---------------------------------------------------------------------------
# Canonical 30 MLB team codes (+ 'FA')
# ---------------------------------------------------------------------------
CANONICAL_TEAM_CODES: Dict[str, str] = {
    # Canonical 2/3-letter codes & historical/source aliases
    "ARI": "ARI", "AZ": "ARI", "DIAMONDBACKS": "ARI", "DBACKS": "ARI", "ARIZONA DIAMONDBACKS": "ARI",
    "ATL": "ATL", "BRAVES": "ATL", "ATLANTA BRAVES": "ATL",
    "ATH": "ATH", "OAK": "ATH", "ATHLETICS": "ATH", "OAKLAND ATHLETICS": "ATH",
    "BAL": "BAL", "ORIOLES": "BAL", "BALTIMORE ORIOLES": "BAL",
    "BOS": "BOS", "RED SOX": "BOS", "RED-SOX": "BOS", "REDSOX": "BOS", "BOSTON RED SOX": "BOS",
    "CHC": "CHC", "CUBS": "CHC", "CHICAGO CUBS": "CHC",
    "CHW": "CHW", "CWS": "CHW", "WHITE SOX": "CHW", "WHITE-SOX": "CHW", "WHITESOX": "CHW", "CHICAGO WHITE SOX": "CHW",
    "CIN": "CIN", "REDS": "CIN", "CINCINNATI REDS": "CIN",
    "CLE": "CLE", "GUARDIANS": "CLE", "INDIANS": "CLE", "CLEVELAND GUARDIANS": "CLE", "CLEVELAND INDIANS": "CLE",
    "COL": "COL", "ROCKIES": "COL", "COLORADO ROCKIES": "COL",
    "DET": "DET", "TIGERS": "DET", "DETROIT TIGERS": "DET",
    "HOU": "HOU", "ASTROS": "HOU", "HOUSTON ASTROS": "HOU",
    "KC": "KC", "KCR": "KC", "ROYALS": "KC", "KCROYALS": "KC", "KANSAS CITY ROYALS": "KC",
    "LAA": "LAA", "ANA": "LAA", "CAL": "LAA", "ANGELS": "LAA", "LOS ANGELES ANGELS": "LAA",
    "LAD": "LAD", "LA": "LAD", "DODGERS": "LAD", "LOS ANGELES DODGERS": "LAD",
    "MIA": "MIA", "FLA": "MIA", "MARLINS": "MIA", "MIAMI MARLINS": "MIA", "FLORIDA MARLINS": "MIA",
    "MIL": "MIL", "BREWERS": "MIL", "MILWAUKEE BREWERS": "MIL",
    "MIN": "MIN", "TWINS": "MIN", "MINNESOTA TWINS": "MIN",
    "NYM": "NYM", "METS": "NYM", "NEW YORK METS": "NYM",
    "NYY": "NYY", "YANKEES": "NYY", "NEW YORK YANKEES": "NYY",
    "PHI": "PHI", "PHILLIES": "PHI", "PHILADELPHIA PHILLIES": "PHI",
    "PIT": "PIT", "PIRATES": "PIT", "PITTSBURGH PIRATES": "PIT",
    "SD": "SD", "SDP": "SD", "PADRES": "SD", "SDPADRES": "SD", "SAN DIEGO PADRES": "SD",
    "SEA": "SEA", "MARINERS": "SEA", "SEATTLE MARINERS": "SEA",
    "SF": "SF", "SFG": "SF", "GIANTS": "SF", "SFGIANTS": "SF", "SAN FRANCISCO GIANTS": "SF",
    "STL": "STL", "CARDINALS": "STL", "ST. LOUIS CARDINALS": "STL", "ST LOUIS CARDINALS": "STL",
    "TB": "TB", "TBR": "TB", "TBD": "TB", "RAYS": "TB", "TAMPA BAY RAYS": "TB",
    "TEX": "TEX", "RANGERS": "TEX", "TEXAS RANGERS": "TEX",
    "TOR": "TOR", "BLUE JAYS": "TOR", "BLUE-JAYS": "TOR", "BLUEJAYS": "TOR", "TORONTO BLUE JAYS": "TOR",
    "WSH": "WSH", "WSN": "WSH", "WAS": "WSH", "MON": "WSH", "NATIONALS": "WSH", "WASHINGTON NATIONALS": "WSH",
    # Free agent / multi-team markers
    "FA": "FA", "UFA": "FA", "FREE AGENT": "FA", "- - -": "FA", "---": "FA", "": "FA",
}

_URL_MLBAM_RE = re.compile(r"-(\d+)$")

_mlbam_to_fg_cache: Optional[Dict[int, int]] = None
_fg_to_mlbam_cache: Optional[Dict[int, int]] = None


def normalize_team_code(team: object, default: str = "FA") -> str:
    """Normalize any team name, slug, or abbreviation to a canonical 2/3-letter MLB code."""
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return default
    cleaned = str(team).strip().upper()
    if not cleaned or cleaned in ("NAN", "NONE"):
        return default
    return CANONICAL_TEAM_CODES.get(cleaned, cleaned[:3] if len(cleaned) >= 2 else default)


def extract_mlbam_from_url(url: object) -> Optional[int]:
    """Extract integer ``mlbam_id`` from an MLB.com prospect URL."""
    if url is None or not isinstance(url, str) or pd.isna(url):
        return None
    match = _URL_MLBAM_RE.search(url.strip())
    return int(match.group(1)) if match else None


def parse_fv_grade(fv_raw: object) -> Optional[float]:
    """Parse an FV grade string/number (handling ``'55+' -> 57.5``)."""
    if fv_raw is None or (isinstance(fv_raw, float) and pd.isna(fv_raw)):
        return None
    text = str(fv_raw).strip()
    if not text or text.lower() == "nan":
        return None
    has_plus = "+" in text
    try:
        base = float(text.replace("+", ""))
        return base + (2.5 if has_plus else 0.0)
    except ValueError:
        return None


def _build_crosswalk_caches() -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build and cache bidirectional integer ``mlbam_id <-> fg_id`` maps.

    Priority order (later sources override earlier ones):
      1. Chadwick Bureau ``people-*.csv`` (broad historical retired baseline)
      2. Statcast ``sc_mlbam_id`` from historic batting/pitching CSVs
      3. ``player_id_crosswalk.csv`` (enriched pipeline crosswalk)
      4. ``current_rosters.csv`` (authoritative active 40-man mappings)
    """
    global _mlbam_to_fg_cache, _fg_to_mlbam_cache
    if _mlbam_to_fg_cache is not None and _fg_to_mlbam_cache is not None:
        return _mlbam_to_fg_cache, _fg_to_mlbam_cache

    mlbam_to_fg: Dict[int, int] = {}
    fg_to_mlbam: Dict[int, int] = {}

    def _ingest_pairs(mlbam_series: pd.Series, fg_series: pd.Series) -> None:
        m_num = pd.to_numeric(mlbam_series, errors="coerce")
        f_num = pd.to_numeric(fg_series, errors="coerce")
        valid = pd.DataFrame({"m": m_num, "f": f_num}).dropna()
        valid = valid[(valid["m"] > 0) & (valid["f"] > 0)]
        for m_val, f_val in zip(valid["m"].astype(int), valid["f"].astype(int)):
            mlbam_to_fg[m_val] = f_val
            fg_to_mlbam[f_val] = m_val

    # 1. Chadwick register fallback
    for path_str in glob.glob(str(_REGISTER_DIR / "people-*.csv")):
        try:
            df = pd.read_csv(path_str, usecols=["key_mlbam", "key_fangraphs"], low_memory=False)
            _ingest_pairs(df["key_mlbam"], df["key_fangraphs"])
        except Exception:
            pass

    # 2. Statcast historic CSVs
    for sc_path in (_HIST_BAT_STATCAST, _HIST_PIT_STATCAST):
        if sc_path.exists():
            try:
                df = pd.read_csv(sc_path, usecols=["sc_mlbam_id", "IDfg"], low_memory=False)
                _ingest_pairs(df["sc_mlbam_id"], df["IDfg"])
            except Exception:
                pass

    # 3. Unified player_id_crosswalk.csv
    if _CROSSWALK_CSV.exists():
        try:
            df = pd.read_csv(_CROSSWALK_CSV, encoding="utf-8-sig", low_memory=False)
            df.columns = df.columns.str.strip()
            if {"mlbam_id", "fg_id"}.issubset(df.columns):
                _ingest_pairs(df["mlbam_id"], df["fg_id"])
        except Exception as exc:
            logger.warning("Failed reading %s: %s", _CROSSWALK_CSV, exc)

    # 4. Active roster file
    if _ROSTER_CSV.exists():
        try:
            df = pd.read_csv(_ROSTER_CSV, usecols=["mlbam_id", "fg_id"], low_memory=False)
            _ingest_pairs(df["mlbam_id"], df["fg_id"])
        except Exception as exc:
            logger.warning("Failed reading %s: %s", _ROSTER_CSV, exc)

    _mlbam_to_fg_cache = mlbam_to_fg
    _fg_to_mlbam_cache = fg_to_mlbam
    logger.info(
        "Built canonical ID crosswalk: %d mlbam->fg, %d fg->mlbam mappings",
        len(mlbam_to_fg),
        len(fg_to_mlbam),
    )
    return mlbam_to_fg, fg_to_mlbam


def get_crosswalk() -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return cached ``(mlbam_to_fg, fg_to_mlbam)`` integer mapping dicts."""
    return _build_crosswalk_caches()


def get_mlbam_to_fg_map() -> Dict[int, int]:
    """Return cached ``{mlbam_id: fg_id}`` mapping."""
    return _build_crosswalk_caches()[0]


def get_fg_to_mlbam_map() -> Dict[int, int]:
    """Return cached ``{fg_id: mlbam_id}`` mapping."""
    return _build_crosswalk_caches()[1]


def mlbam_to_fg(mlbam_id: object) -> Optional[int]:
    """Resolve a single ``mlbam_id`` to integer ``fg_id`` (or ``None``)."""
    if mlbam_id is None or (isinstance(mlbam_id, float) and pd.isna(mlbam_id)):
        return None
    try:
        return get_mlbam_to_fg_map().get(int(float(mlbam_id)))
    except (TypeError, ValueError):
        return None


def fg_to_mlbam(fg_id: object) -> Optional[int]:
    """Resolve a single ``fg_id`` to integer ``mlbam_id`` (or ``None``)."""
    if fg_id is None or (isinstance(fg_id, float) and pd.isna(fg_id)):
        return None
    try:
        return get_fg_to_mlbam_map().get(int(float(fg_id)))
    except (TypeError, ValueError):
        return None


def load_recent_prospect_info(
    as_of_year: int,
    recency_years: int = 3,
    name_to_fg_fallback: Optional[Dict[str, int]] = None,
) -> Dict[int, dict]:
    """Load each recent prospect's latest scouting record keyed by ``fg_id``.

    Only includes rankings published in ``[as_of_year - recency_years, as_of_year]``
    (preventing future leakage during historical backtests).
    """
    if not _PROSPECT_CSV.exists():
        return {}

    df = pd.read_csv(_PROSPECT_CSV, low_memory=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    min_year = as_of_year - recency_years
    df = df[(df["year"] >= min_year) & (df["year"] <= as_of_year)].copy()
    if df.empty:
        return {}

    df["mlbam_id"] = df["prospect_url"].apply(extract_mlbam_from_url)
    mlbam_map = get_mlbam_to_fg_map()

    def _resolve_fg(row: pd.Series) -> Optional[int]:
        mid = row.get("mlbam_id")
        if pd.notna(mid):
            mapped = mlbam_map.get(int(mid))
            if mapped is not None:
                return mapped
        if name_to_fg_fallback:
            nk = name_key(row.get("name", ""))
            if nk in name_to_fg_fallback:
                return name_to_fg_fallback[nk]
        return None

    df["fg_id"] = df.apply(_resolve_fg, axis=1)
    df = df.dropna(subset=["fg_id"]).copy()
    df["fg_id"] = df["fg_id"].astype(int)

    # Keep most recent scouting report per player
    df = df.sort_values("year", ascending=False).drop_duplicates(subset=["fg_id"], keep="first")

    lookup: Dict[int, dict] = {}
    for _, row in df.iterrows():
        fv = parse_fv_grade(row.get("grade_overall"))
        if fv is None:
            continue
        top_100 = row.get("top_100")
        lookup[int(row["fg_id"])] = {
            "fg_id": int(row["fg_id"]),
            "mlbam_id": int(row["mlbam_id"]) if pd.notna(row.get("mlbam_id")) else None,
            "name": str(row.get("name", "")),
            "fv": fv,
            "top_100": float(top_100) if pd.notna(top_100) else None,
            "org_rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
            "position": str(row.get("position", "")),
            "level": str(row.get("level", "")) if pd.notna(row.get("level")) else "",
            "team": normalize_team_code(row.get("team_slug")),
            "year": int(row["year"]),
            "grade_hit": float(row["grade_hit"]) if pd.notna(row.get("grade_hit")) else None,
            "grade_power": float(row["grade_power"]) if pd.notna(row.get("grade_power")) else None,
        }

    return lookup
