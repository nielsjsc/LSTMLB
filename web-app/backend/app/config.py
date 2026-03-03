"""
Central Season Configuration
=============================
Single source of truth for year constants in the web-app backend.
Update these values once per season — every downstream reference derives from here.

Usage:
    from app.config import CURRENT_YEAR, PROSPECT_YEARS, PROJECTION_RANGE
"""

# ── Core season constant ─────────────────────────────────────────────────
CURRENT_YEAR: int = 2026

# ── Prospect data availability ───────────────────────────────────────────
# Years for which prospect valuations exist (oldest → newest).
PROSPECT_YEAR_START: int = 2022
PROSPECT_YEAR_END: int = 2025
PROSPECT_YEARS: list[int] = list(range(PROSPECT_YEAR_START, PROSPECT_YEAR_END + 1))
PROSPECT_DEFAULT_YEAR: int = PROSPECT_YEAR_END  # year shown by default in UI

# ── Projection range ─────────────────────────────────────────────────────
MAX_PROJECTION_YEARS: int = 5  # how many future seasons to display
PROJECTION_RANGE: list[int] = list(range(CURRENT_YEAR, CURRENT_YEAR + MAX_PROJECTION_YEARS))
