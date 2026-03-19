"""
Universal Player Name Matching
==============================

Single source of truth for normalizing and matching player names across
all data sources: FanGraphs, Spotrac, Cot's, MLB API, prospect lists,
and the player-ID crosswalk.

Data-source characteristics
---------------------------
- **Crosswalk** (player_id_crosswalk.csv): accented names (``Javier Báez``)
- **FanGraphs**: ASCII-only (``Javier Baez``), used in batting/pitching CSVs
- **Cot's by-year**: ASCII-only (``Javier Baez``)
- **Spotrac transactions**: mixed — mostly ASCII, sometimes periods
  (``A.J. Puk``, ``J.P. Crawford``)
- **Prospect data**: accented (``Javier Báez``), same source as crosswalk

Normalization pipeline:  name → unidecode → uppercase → strip periods/hyphens
→ collapse spaces → remove FA suffix → apply aliases → optionally strip suffixes.

Usage
-----
    from core.name_utils import normalize_name, name_key, normalize_team

    # Standard normalization (preserves suffix like Jr.)
    normalize_name("Yadier Molina Jr.")  # → "YADIER MOLINA JR"

    # Aggressive key for matching (strips suffix, letters only)
    name_key("J.P. Crawford")           # → "jp crawford"
    name_key("Yadier Molina Jr.")       # → "yadier molina"

    # Team normalization
    normalize_team("CWS")              # → "CHW"
"""

import re
import unicodedata

import pandas as pd
import unidecode as _unidecode


# ═══════════════════════════════════════════════════════════════════════════════
# Suffix pattern: Jr., Sr., II, III, IV, V — anchored at end
# ═══════════════════════════════════════════════════════════════════════════════
_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|iii|iv|ii|v)\s*$", re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════════
# Name aliases: variant → canonical  (both stored as name_key form)
# Add entries when a data source consistently uses a different spelling.
# ═══════════════════════════════════════════════════════════════════════════════
_NAME_ALIASES: dict[str, str] = {
    "jake junis":        "jakob junis",
    "cam schlittler":    "cameron schlittler",
    "bo naylor":         "beau naylor",
    "manny machado":     "manuel machado",
    "gio urshela":       "giovanny urshela",
    "j d martinez":      "jd martinez",
    "hyun jin ryu":      "hyunjin ryu",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Team alias map: every known abbreviation → canonical 3-letter code
# ═══════════════════════════════════════════════════════════════════════════════
_TEAM_ALIAS: dict[str, str] = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CWS": "CHW",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU",
    "KC":  "KC",  "KCR": "KC",
    "LAA": "LAA", "ANA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA", "FLA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "OAK": "ATH", "ATH": "ATH",
    "PHI": "PHI", "PIT": "PIT",
    "SD":  "SD",  "SDP": "SD",
    "SF":  "SF",  "SFG": "SF",
    "SEA": "SEA", "STL": "STL",
    "TB":  "TB",  "TBR": "TB",
    "TEX": "TEX", "TOR": "TOR",
    "WSH": "WSH", "WSN": "WSH", "WAS": "WSH",
    "FA":  "FA",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def strip_accents(name: str) -> str:
    """Remove diacritics: ``Báez`` → ``Baez``.

    Uses ``unidecode`` for robust transliteration of all Latin scripts,
    with a NFKD fallback for edge cases.
    """
    if not isinstance(name, str):
        return str(name) if name is not None else ""
    return _unidecode.unidecode(name)


def normalize_name(name: str) -> str:
    """Normalize a player name for display/comparison.

    Pipeline: accent removal → uppercase → strip periods/hyphens →
    collapse whitespace → remove trailing ``FA`` tag → apply aliases.

    Preserves suffixes (Jr., III, etc.).

    Returns
    -------
    str  Uppercase ASCII name, e.g. ``"YADIER MOLINA JR"``
    """
    if pd.isna(name):
        return name
    s = strip_accents(str(name)).upper().strip()
    s = s.replace(".", "").replace("-", " ")
    s = " ".join(s.split())
    if s.endswith(" FA"):
        s = s[:-3].strip()
    # Check alias on the lowered form, return uppercased
    low = s.lower()
    if low in _NAME_ALIASES:
        s = _NAME_ALIASES[low].upper()
    return s


def name_key(name: str) -> str:
    """Aggressive name key for cross-source matching.

    Pipeline: ``normalize_name`` → lowercase → strip suffixes →
    remove all non-alpha/space → collapse whitespace.

    Returns
    -------
    str  Lowercase ASCII key, e.g. ``"jp crawford"``
    """
    if pd.isna(name):
        return ""
    s = normalize_name(name)
    if pd.isna(s):
        return ""
    s = s.lower()
    s = _SUFFIX_RE.sub("", s).strip()
    s = re.sub(r"[^a-z\s]", "", s)
    return " ".join(s.split())


def name_key_alpha_only(name: str) -> str:
    """Most aggressive key — letters only, no spaces.

    ``"J.P. Crawford"`` → ``"jpcrawford"``

    Useful as a last-resort fallback when spaces might differ.
    """
    return re.sub(r"[^a-z]", "", name_key(name))


def normalize_team(team: str) -> str:
    """Normalize a team abbreviation to its canonical form."""
    if pd.isna(team):
        return ""
    return _TEAM_ALIAS.get(str(team).upper().strip(), str(team).upper().strip())


def match_players_by_name(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name_col: str = "Name",
    right_name_col: str = "name",
    left_team_col: str | None = None,
    right_team_col: str | None = None,
) -> dict[str, list]:
    """Build a name_key → list of right-side indices mapping.

    When *team_col* is provided on both sides, ties are broken by
    matching team.  Returns a dict of {name_key: [right_indices]}.
    """
    lookup: dict[str, list[int]] = {}
    for idx, row in right.iterrows():
        key = name_key(row[right_name_col])
        if key:
            lookup.setdefault(key, []).append(idx)
    return lookup
