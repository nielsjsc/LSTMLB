"""Database initialisation & ETL pipeline.
==========================================

Populates the application database from CSV / JSON source files.
Run standalone to perform a full reload::

    cd web-app/backend
    python -m app.services.data_loader

Architecture
------------
1. **Schema reset** — tables are dropped and recreated so column-level
   schema changes are picked up automatically.
2. **Bulk loading** — each dataset is read with pandas, transformed via
   vectorised column renames / type coercions, then written through
   SQLAlchemy Core ``insert()`` with batched ``executemany`` (bypasses
   ORM identity-map overhead → 10-100x faster than ``bulk_save_objects``).
3. **Cross-reference resolution** — a single SQL ``UPDATE`` resolves
   prospect ``IDfg`` from the pre-enriched crosswalk.  The crosswalk is
   enriched *offline* by ``scrapers/enrich_crosswalk.py`` (name-matching
   prospects → MiLB FanGraphs IDs).  ``has_mlb`` is resolved via an
   indexed integer join (``mlbam_id IN players.mlb_id``).

Performance
-----------
Full reload (all tables, ~250 K total rows): **< 20 s** on a typical
laptop with SQLite.  Previously ~8 min due to LOWER(name) matching.
"""

import sys
import time
import json as _json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import logging
import os

from sqlalchemy import text, insert
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

load_dotenv(backend_dir / '.env')

from app.models.player import Player
from app.models.prospect import Prospect
from app.models.historical import HistoricalPlayer
from app.models.past_trade import PastTrade
from app.models.milb_stats import MiLBHittingStats, MiLBPitchingStats
from app.models.player_id_crosswalk import PlayerIdCrosswalk
from app.models.trade_value_history import TradeValueHistory
from app.models.statcast_expected import StatcastExpected
from app.models.spotrac_transaction import SpotracTransaction
from app.models.fielding_stats import FieldingStats
from app.config import PROSPECT_YEARS, PROSPECT_DEFAULT_YEAR
from app.database import SessionLocal, engine, Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _find_latest_preseason(directory: Path) -> Optional[Path]:
    """Find the most recent player_values_preseason_YYYY.csv in *directory*.
    Falls back to the legacy un-dated filename if no dated file exists."""
    if not directory.is_dir():
        return None
    dated = sorted(directory.glob("player_values_preseason_*.csv"), reverse=True)
    if dated:
        return dated[0]
    legacy = directory / "player_values_preseason.csv"
    return legacy if legacy.exists() else None


# ══════════════════════════════════════════════════════════════════════════
# Data-file manifest
# ══════════════════════════════════════════════════════════════════════════
# Single source of truth for every data file the loader touches.
# All paths relative to the project root (LSTMLB/).

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

DATA_PATHS: Dict[str, Path] = {
    # ── Generated / processed ─────────────────────────────────────────
    "players":       PROJECT_ROOT / "data" / "generated" / "value_by_year" / "player_values_complete.csv",
    "players_preseason": PROJECT_ROOT / "data" / "generated" / "value_by_year",  # directory; resolved dynamically
    "prospects":     PROJECT_ROOT / "data" / "prospect_data" / "prospects_2014_2026_with_top100.csv",
    "historical":    PROJECT_ROOT / "data" / "generated" / "historical_players" / "historical_players.json",
    "trades":        PROJECT_ROOT / "data" / "generated" / "past_trades" / "trades.json",
    "crosswalk":     PROJECT_ROOT / "data" / "generated" / "player_id_crosswalk.csv",
    # ── Raw / scraped ─────────────────────────────────────────────────
    "milb_hitters":  PROJECT_ROOT / "data" / "MiLB" / "MiLB_Hitters.csv",
    "milb_pitchers": PROJECT_ROOT / "data" / "MiLB" / "MiLB_Pitchers.csv",
    # ── Supplementary (migrated from lazy-loaded CSVs) ────────────────
    "trade_value_history": PROJECT_ROOT / "data" / "generated" / "value_by_year" / "trade_value_history.csv",
    "statcast_batter":     PROJECT_ROOT / "data" / "statcast" / "statcast_batter_expected_stats_2015_2025.csv",
    "statcast_pitcher":    PROJECT_ROOT / "data" / "statcast" / "statcast_pitcher_expected_stats_2015_2025.csv",
    "spotrac_transactions": PROJECT_ROOT / "data" / "salary" / "spotrac_transactions.csv",
    # ── Fielding ───────────────────────────────────────────────────
    "fielding_historical": PROJECT_ROOT / "data" / "historic_mlb" / "mlb_fielding_data_2000_2025_with_statcast.csv",
    "fielding_projections": PROJECT_ROOT / "data" / "generated" / "pipeline" / "fielding_projections_complete.csv",
    "crosswalk_for_fielding": PROJECT_ROOT / "data" / "generated" / "player_id_crosswalk.csv",
}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _clean_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame → list[dict], replacing NaN/NaT with ``None``.

    Uses vectorised ``astype(object) + where`` — much faster than
    row-level ``iterrows`` with ``pd.notna`` checks.
    """
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _coerce_int_cols(records: list[dict], int_cols: set[str]) -> None:
    """In-place: convert float → int for integer DB columns."""
    for rec in records:
        for col in int_cols:
            v = rec.get(col)
            if v is not None:
                try:
                    rec[col] = int(v)
                except (ValueError, TypeError):
                    rec[col] = None


class _Timer:
    """Context manager that logs elapsed wall-clock time."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0
        logger.info(f"  >> {self.label} — {self.elapsed:.1f}s")


BATCH_SIZE = 5_000  # rows per Core executemany chunk


def _bulk_insert(session: Session, model, records: list[dict]) -> int:
    """Batch-insert *records* via Core ``executemany``.  Returns row count."""
    total = len(records)
    if total == 0:
        return 0
    for i in range(0, total, BATCH_SIZE):
        session.execute(insert(model), records[i : i + BATCH_SIZE])
    session.commit()
    return total


# ══════════════════════════════════════════════════════════════════════════
# Column-mapping configs
# ══════════════════════════════════════════════════════════════════════════

# CSV column name → Player model column name
# NOTE: column names must match the ACTUAL CSV headers exactly.
# player_values_complete.csv uses: Player_Name, WAR_batter, WAR_pitcher,
# Status (capitalised), Base_Value, Contract_Value, Surplus_Value, etc.
PLAYER_COL_MAP = {
    "IDfg": "real_id", "mlb_id": "mlb_id",
    "Player_Name": "name", "Team": "team", "Position": "position", "Status": "status",
    "Age": "age", "Year": "year",
    "years_control": "years_control", "FA_Year": "fa_year",
    "Probable_FA_Year": "probable_fa_year", "Earliest_FA_Year": "earliest_fa_year",
    "control_through": "control_through",
    "Base_Value": "base_value", "Contract_Value": "contract_value",
    "Surplus_Value": "surplus_value", "trade_value": "trade_value",
    "contract_war": "contract_war",
    # Hitting
    "G_bat": "g_bat", "WAR_batter": "war_bat",
    "BB%_bat": "bb_pct_bat", "K%_bat": "k_pct_bat",
    "AVG": "avg", "OBP": "obp", "SLG": "slg", "OPS": "ops",
    "wOBA": "woba", "wRC+": "wrc_plus",
    "Bat": "bat", "Off": "off", "BsR": "bsr", "Def": "def_value",
    "HR": "hr", "2B": "doubles", "3B": "triples",
    "R": "r", "RBI": "rbi", "SB": "sb", "CS": "cs",
    # Pitching
    "G_pit": "g_pit", "GS": "gs", "IP": "ip", "WAR_pitcher": "war_pit",
    "ERA": "era", "FIP": "fip", "K%_pit": "k_pct_pit", "BB%_pit": "bb_pct_pit",
    "GB%_pit": "gb_pct", "FB%_pit": "fb_pct", "HR/FB_pit": "hr_fb", "HR/9": "hr_9",
    # Aggregate values
    "avg_war": "avg_war", "total_contract": "total_contract",
    "avg_contract": "avg_contract",
    "total_future_war": "total_future_war", "total_future_value": "total_future_value",
    "total_value": "total_value", "total_war": "total_war",
    "historical_value": "historical_value", "historical_war": "historical_war",
    "contract_base_value": "contract_base_value",
}

PLAYER_INT_COLS = {
    "real_id", "mlb_id", "age", "year", "fa_year",
    "probable_fa_year", "earliest_fa_year", "control_through",
    "g_bat", "hr", "doubles", "triples", "r", "rbi", "sb", "cs",
    "g_pit", "gs",
}

PLAYER_STR_COLS = {"name", "team", "position", "status", "projection_type"}

# ── Prospect source CSV (prospects_2014_2026_with_top100.csv) ──────────
# This CSV has one row per player per year with lowercase column names.
# We map them to DB model column names.

PROSPECT_TEAM_SLUG_MAP = {
    'diamondbacks': 'ARI', 'dbacks': 'ARI',
    'braves': 'ATL',
    'orioles': 'BAL',
    'red-sox': 'BOS', 'redsox': 'BOS',
    'cubs': 'CHC',
    'white-sox': 'CHW', 'whitesox': 'CHW',
    'reds': 'CIN',
    'guardians': 'CLE', 'indians': 'CLE',
    'rockies': 'COL',
    'tigers': 'DET',
    'astros': 'HOU',
    'royals': 'KCR', 'kcroyals': 'KCR',
    'angels': 'LAA',
    'dodgers': 'LAD',
    'marlins': 'MIA',
    'brewers': 'MIL',
    'twins': 'MIN',
    'mets': 'NYM',
    'yankees': 'NYY',
    'athletics': 'ATH',
    'phillies': 'PHI',
    'pirates': 'PIT',
    'padres': 'SDP', 'sdpadres': 'SDP',
    'giants': 'SFG', 'sfgiants': 'SFG',
    'mariners': 'SEA',
    'cardinals': 'STL',
    'rays': 'TBR', 'tbr': 'TBR',
    'rangers': 'TEX',
    'blue-jays': 'TOR', 'bluejays': 'TOR',
    'nationals': 'WSH', 'wsh': 'WSH',
}

# FV grade → base value in dollars (must match the prospect model pipeline)
_FV_BASE_VALUES = {
    80: 300_000_000,
    75: 250_000_000,
    70: 200_000_000,
    65: 135_000_000,
    60: 90_000_000,
    55: 50_000_000,
    50: 25_000_000,
    47: 15_000_000,
    45: 10_000_000,
    40: 4_000_000,
    35: 1_000_000,
    30: 250_000,
}

import re as _re

def _extract_mlbam_from_url(url) -> int | None:
    """Extract MLBAM ID from prospect URL like ``...-605151``."""
    if not url or not isinstance(url, str) or pd.isna(url):
        return None
    m = _re.search(r"-(\d+)$", str(url))
    return int(m.group(1)) if m else None


def _prospect_rank_adj(rank: float) -> float:
    """Top-100 rank → multiplicative adjustment (1.0 = no change)."""
    if rank <= 5:
        return 1.35
    if rank <= 10:
        return 1.25
    if rank <= 25:
        return 1.15
    if rank <= 50:
        return 1.05
    if rank <= 75:
        return 0.95
    return 0.90


def _calc_prospect_value(fv: float, top_100: float | None) -> float:
    """Calculate prospect value from FV grade + optional top-100 rank."""
    if pd.isna(fv):
        return 0.0
    grade = max(30, min(80, round(fv / 5) * 5))
    base = _FV_BASE_VALUES.get(int(grade), 0)
    if base == 0:
        return 0.0
    mult = _prospect_rank_adj(top_100) if pd.notna(top_100) else 1.0
    return base * mult

# MiLB hitting CSV → model
MILB_HIT_COL_MAP = {
    "Season": "season", "Name": "name", "Team": "team", "Level": "level",
    "Age": "age", "PA": "pa",
    "BB%": "bb_pct", "K%": "k_pct", "BB/K": "bb_k",
    "AVG": "avg", "OBP": "obp", "SLG": "slg", "OPS": "ops",
    "ISO": "iso", "Spd": "spd", "BABIP": "babip", "wSB": "wsb",
    "wRC": "wrc", "wRAA": "wraa", "wOBA": "woba", "wRC+": "wrc_plus",
}

# MiLB pitching CSV → model
MILB_PIT_COL_MAP = {
    "Season": "season", "Name": "name", "Team": "team", "Level": "level",
    "Age": "age", "IP": "ip",
    "K/9": "k_9", "BB/9": "bb_9", "K/BB": "k_bb", "HR/9": "hr_9",
    "K%": "k_pct", "BB%": "bb_pct", "K-BB%": "k_bb_pct",
    "AVG": "avg", "WHIP": "whip", "BABIP": "babip", "LOB%": "lob_pct",
    "ERA": "era", "FIP": "fip", "E-F": "e_f", "xFIP": "xfip",
}


# ══════════════════════════════════════════════════════════════════════════
# DataLoader
# ══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """ETL pipeline: CSV/JSON → transformed records → bulk SQL insert."""

    def __init__(self, db: Session):
        self.db = db
        self._is_sqlite = os.getenv("DATABASE_URL", "sqlite").startswith("sqlite")

    # ── Players ───────────────────────────────────────────────────────────

    def load_players(self, csv_path: Path, projection_type: str = "ros") -> None:
        """Vectorised CSV read → Core bulk insert for the *players* table."""
        with _Timer(f"Players ({projection_type})"):
            df = pd.read_csv(csv_path, low_memory=False)
            initial = len(df)

            # Rename CSV headers → DB column names
            df = df.rename(columns=PLAYER_COL_MAP)

            # Keep only columns that exist in the model
            model_cols = {c.key for c in Player.__table__.columns if c.key != "id"}
            df = df[[c for c in df.columns if c in model_cols]]

            # Tag projection type
            df["projection_type"] = projection_type

            # Drop rows without mlb_id (can't link them to anything)
            df = df.dropna(subset=["mlb_id"])

            # Numeric coercion (vectorised)
            for col in df.columns:
                if col not in PLAYER_STR_COLS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # ── Fill missing salary from Cot's by-year data ──────────────
            # Historical rows in the pipeline may have NaN contract_value
            # even though Cot's by-year CSVs have the actual salary.
            self._augment_player_salary(df)

            records = _clean_records(df)
            _coerce_int_cols(records, PLAYER_INT_COLS)

            n = _bulk_insert(self.db, Player, records)
            logger.info(f"    {n:,} players loaded ({initial - n:,} skipped — no mlb_id)")

    # ── Salary augmentation for players ───────────────────────────────────

    @staticmethod
    def _augment_player_salary(df: pd.DataFrame) -> None:
        """Fill NaN ``contract_value`` from the universal salary CSV (in-place).

        Also recalculates ``surplus_value`` for rows that gained a salary.
        Uses (name_lower, year) as the primary key, with a (name_lower,
        team, year) check first for specificity.
        """
        if "contract_value" not in df.columns:
            return

        mask = df["contract_value"].isna()
        if not mask.any():
            return

        # Lazy import to avoid circular deps at module level
        from app.routes.historical import _load_universal_salary

        salary_lookup = _load_universal_salary()  # {(name, yr): int, (name, team, yr): int}
        if not salary_lookup:
            return

        filled = 0
        idxs = df.index[mask]
        for idx in idxs:
            row = df.loc[idx]
            nm = str(row.get("name", "")).lower().strip()
            team = str(row.get("team", "")).strip()
            yr = row.get("year")
            if pd.isna(yr) or not nm:
                continue
            yr = int(yr)

            # Try team-specific first, then name+year
            sal = salary_lookup.get((nm, team, yr))
            if sal is None:
                sal = salary_lookup.get((nm, yr))
            if sal is not None:
                df.at[idx, "contract_value"] = float(sal)
                bv = row.get("base_value")
                if pd.notna(bv):
                    df.at[idx, "surplus_value"] = float(bv) - float(sal)
                filled += 1

        if filled:
            logger.info(f"    Salary augmentation: filled {filled:,}/{mask.sum():,} missing contract_value rows")

    # ── Prospects ─────────────────────────────────────────────────────────

    def load_prospects(self, csv_path: Path) -> None:
        """Load prospect data from the raw source CSV
        (``prospects_2014_2026_with_top100.csv``).

        One DB row per player-year.  Extracts mlbam_id from prospect_url,
        maps team slugs → 3-letter abbreviations, maps tool grades,
        calculates per-row value & composite, and builds JSON
        ``values_by_year`` / ``composites_by_year`` for backward compat.
        """
        with _Timer("Prospects"):
            df = pd.read_csv(csv_path, low_memory=False)
            logger.info(f"    Source CSV: {len(df):,} rows, years {df['year'].min():.0f}–{df['year'].max():.0f}")

            # ── Column mapping & derivation ────────────────────────────
            # Name → name (already lowercase in source)
            df = df.rename(columns={"name": "name", "position": "position",
                                     "age": "age", "rank": "org_rank"})

            # Team slug → 3-letter abbreviation
            df["org"] = df["team_slug"].map(
                lambda s: PROSPECT_TEAM_SLUG_MAP.get(str(s).lower().strip(), str(s).upper()[:3])
                if pd.notna(s) else "FA"
            )

            # Year (float → int)
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)

            # FV (grade_overall → fv, stored as string to match model)
            df["fv"] = pd.to_numeric(df["grade_overall"], errors="coerce")
            # Round to nearest 5 for display
            df["fv_num"] = df["fv"].copy()  # keep numeric copy
            df["fv"] = df["fv"].apply(
                lambda v: str(int(round(v / 5) * 5)) if pd.notna(v) else None
            )

            # Top 100 rank
            df["top_100"] = pd.to_numeric(df["top_100"], errors="coerce")
            df.loc[df["top_100"].isna(), "top_100"] = None

            # Org rank
            df["org_rank"] = pd.to_numeric(df["org_rank"], errors="coerce")

            # ── Tool grades (stored as "grade" strings) ────────────────
            # Hitter tools
            df["hit"] = df["grade_hit"].apply(lambda v: str(int(v)) if pd.notna(v) else None)
            df["game_power"] = df["grade_power"].apply(lambda v: str(int(v)) if pd.notna(v) else None)
            df["raw_power"] = df["grade_power"].apply(lambda v: str(int(v)) if pd.notna(v) else None)
            df["speed"] = df["grade_run"].apply(lambda v: str(int(v)) if pd.notna(v) else None)
            # Pitcher tools
            df["fastball"] = df["grade_fastball"].apply(lambda v: str(int(v)) if pd.notna(v) else None)
            df["slider"] = df["grade_slider"].apply(lambda v: str(int(v)) if pd.notna(v) else None)
            df["curve"] = df["grade_curveball"].apply(lambda v: str(int(v)) if pd.notna(v) else None)
            df["changeup"] = df["grade_changeup"].apply(lambda v: str(int(v)) if pd.notna(v) else None)
            df["command"] = df["grade_control"].apply(lambda v: str(int(v)) if pd.notna(v) else None)

            # ── mlbam_id from prospect_url ─────────────────────────────
            df["mlbam_id"] = df["prospect_url"].apply(_extract_mlbam_from_url)

            # ── Value & Composite per row ──────────────────────────────
            df["value"] = df.apply(
                lambda r: _calc_prospect_value(r["fv_num"], r["top_100"]),
                axis=1,
            )
            # Composite = top_100 rank if available, else None
            df["composite"] = df["top_100"]

            # ── Build legacy JSON dicts per player ─────────────────────
            # Group by name → aggregate all years' values/composites
            val_lookup: dict[str, dict] = {}
            comp_lookup: dict[str, dict] = {}
            for name_key, grp in df.groupby("name"):
                vd, cd = {}, {}
                for _, r in grp.iterrows():
                    yr_str = str(int(r["year"]))
                    if pd.notna(r["value"]) and r["value"] > 0:
                        vd[yr_str] = float(r["value"])
                    if pd.notna(r["top_100"]):
                        cd[yr_str] = float(r["top_100"])
                val_lookup[name_key] = vd
                comp_lookup[name_key] = cd

            df["values_by_year"] = df["name"].map(val_lookup)
            df["composites_by_year"] = df["name"].map(comp_lookup)

            # ── has_mlb default False ──────────────────────────────────
            df["has_mlb"] = False

            # ── IDfg placeholder (resolved later via crosswalk) ────────
            df["IDfg"] = None

            # ── Keep only model columns ────────────────────────────────
            model_cols = {c.key for c in Prospect.__table__.columns if c.key != "id"}
            df = df[[c for c in df.columns if c in model_cols]]

            records = _clean_records(df)

            # Fix int types for mlbam_id / year / org_rank / top_100
            for rec in records:
                for col in ("mlbam_id", "year", "org_rank", "top_100"):
                    if rec.get(col) is not None:
                        try:
                            rec[col] = int(rec[col])
                        except (ValueError, TypeError):
                            rec[col] = None

            n = _bulk_insert(self.db, Prospect, records)
            logger.info(f"    {n:,} prospects loaded")

    # ── Crosswalk ─────────────────────────────────────────────────────────

    def load_crosswalk(self, csv_path: Path) -> None:
        """Bulk-insert the MLBAM <-> FanGraphs crosswalk table."""
        with _Timer("Crosswalk"):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.columns = df.columns.str.strip()  # remove BOM artefacts

            df = df.dropna(subset=["mlbam_id"])
            df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")
            df = df.dropna(subset=["mlbam_id"])
            df["mlbam_id"] = df["mlbam_id"].astype(int)

            # Clean fg_id
            df["fg_id"] = df["fg_id"].astype(str).str.strip()
            df.loc[df["fg_id"].isin(["nan", "", "None", "NaN"]), "fg_id"] = None

            df["name"] = df.get("name", pd.Series(dtype=str)).fillna("")
            df["source"] = df.get("source", pd.Series(dtype=str)).fillna("")

            records = _clean_records(df[["mlbam_id", "fg_id", "name", "source"]])
            # Ensure mlbam_id stays int after clean_records
            for rec in records:
                rec["mlbam_id"] = int(rec["mlbam_id"])

            n = _bulk_insert(self.db, PlayerIdCrosswalk, records)
            resolved = sum(1 for r in records if r.get("fg_id"))
            logger.info(f"    {n:,} crosswalk entries ({resolved:,} with FanGraphs IDs)")

    # ── Resolve prospect IDfg via SQL ─────────────────────────────────────

    def resolve_prospect_idfg(self) -> None:
        """Single SQL UPDATE: ``prospects.IDfg = crosswalk.fg_id``
        where the prospect has ``mlbam_id`` but no ``IDfg``.
        Replaces the old Python loop over every prospect row.
        """
        with _Timer("Resolve prospect IDfg"):
            if self._is_sqlite:
                sql = text("""
                    UPDATE prospects
                    SET "IDfg" = (
                        SELECT pc.fg_id
                        FROM   player_id_crosswalk pc
                        WHERE  pc.mlbam_id = prospects.mlbam_id
                          AND  pc.fg_id IS NOT NULL
                        LIMIT 1
                    )
                    WHERE "IDfg" IS NULL
                      AND mlbam_id IS NOT NULL
                """)
            else:
                # PostgreSQL supports UPDATE ... FROM
                sql = text("""
                    UPDATE prospects
                    SET    "IDfg" = pc.fg_id
                    FROM   player_id_crosswalk pc
                    WHERE  pc.mlbam_id = prospects.mlbam_id
                      AND  pc.fg_id IS NOT NULL
                      AND  prospects."IDfg" IS NULL
                      AND  prospects.mlbam_id IS NOT NULL
                """)

            result = self.db.execute(sql)
            self.db.commit()
            logger.info(f"    {result.rowcount:,} prospect IDfg values resolved")

    def resolve_prospect_has_mlb(self) -> None:
        """Set ``has_mlb = True`` for prospects whose ``mlbam_id``
        appears in the MLB *players* table (``players.mlb_id``).

        Uses an indexed integer join — nearly instant compared to the
        old LOWER(name) approach which took ~140 s.
        """
        with _Timer("Resolve prospect has_mlb"):
            if self._is_sqlite:
                sql = text("""
                    UPDATE prospects
                    SET has_mlb = 1
                    WHERE has_mlb = 0
                      AND mlbam_id IS NOT NULL
                      AND mlbam_id IN (
                          SELECT DISTINCT mlb_id FROM players
                          WHERE mlb_id IS NOT NULL
                      )
                """)
            else:
                sql = text("""
                    UPDATE prospects
                    SET    has_mlb = TRUE
                    WHERE  has_mlb = FALSE
                      AND  mlbam_id IS NOT NULL
                      AND  mlbam_id IN (
                          SELECT DISTINCT mlb_id FROM players
                          WHERE mlb_id IS NOT NULL
                      )
                """)
            result = self.db.execute(sql)
            self.db.commit()
            logger.info(f"    {result.rowcount:,} prospects marked as has_mlb")

    # ── MiLB statistics ───────────────────────────────────────────────────

    def _load_milb_csv(
        self, csv_path: Path, col_map: dict, model, int_cols: set[str]
    ) -> int:
        """Generic MiLB stat loader: pandas read -> rename -> Core insert."""
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        df.columns = df.columns.str.strip()

        # Normalise the PlayerId column name
        pid_col = None
        for candidate in ("PlayerId", "playerid", "playerID", "PLAYERID"):
            if candidate in df.columns:
                pid_col = candidate
                break
        if pid_col is None:
            for col in df.columns:
                if col.strip().lower() == "playerid":
                    pid_col = col
                    break
        if pid_col is None:
            logger.warning(f"  No PlayerId column in {csv_path}")
            return 0

        # Rename: PlayerId → IDfg, plus stat columns
        df = df.rename(columns={pid_col: "IDfg", **col_map})

        # Clean IDfg — ensure string, drop invalid
        df["IDfg"] = df["IDfg"].astype(str).str.strip().str.strip('"')
        df = df[df["IDfg"].notna() & (df["IDfg"] != "") & (df["IDfg"] != "nan")]

        # Keep only columns that exist in the model
        model_cols = {c.key for c in model.__table__.columns if c.key != "id"}
        df = df[[c for c in df.columns if c in model_cols]]

        # Coerce numerics
        str_cols = {"IDfg", "name", "team", "level"}
        for col in df.columns:
            if col not in str_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill NOT NULL string columns (name is required by schema)
        df["name"] = df["name"].fillna("")

        records = _clean_records(df)
        _coerce_int_cols(records, int_cols)
        return _bulk_insert(self.db, model, records)

    def load_milb_stats(self, hitters_csv: Path, pitchers_csv: Path) -> None:
        """Load both MiLB hitting and pitching stat tables."""
        with _Timer("MiLB hitting stats"):
            h_n = 0
            if hitters_csv.exists():
                h_n = self._load_milb_csv(
                    hitters_csv, MILB_HIT_COL_MAP, MiLBHittingStats,
                    int_cols={"season", "age", "pa"},
                )
                logger.info(f"    {h_n:,} hitting rows")
            else:
                logger.warning(f"  MiLB hitters CSV not found: {hitters_csv}")

        with _Timer("MiLB pitching stats"):
            p_n = 0
            if pitchers_csv.exists():
                p_n = self._load_milb_csv(
                    pitchers_csv, MILB_PIT_COL_MAP, MiLBPitchingStats,
                    int_cols={"season", "age"},
                )
                logger.info(f"    {p_n:,} pitching rows")
            else:
                logger.warning(f"  MiLB pitchers CSV not found: {pitchers_csv}")

    # ── Historical players ────────────────────────────────────────────────

    def load_historical(self, json_path: Path) -> None:
        """Ingest ``historical_players.json`` with salary augmentation."""
        with _Timer("Historical players"):
            if not json_path.exists():
                logger.warning(f"  File not found: {json_path}")
                return

            from app.routes.historical import _load_universal_salary

            with open(json_path, "r") as f:
                data = _json.load(f)

            players = data.get("players", {})
            logger.info(f"    {len(players):,} players read from JSON")

            # ── Salary augmentation (single canonical source) ─────────────
            salary = _load_universal_salary()  # {(name,yr): int, (name,team,yr): int}

            filled = 0
            for pl in players.values():
                name_lower = pl["name"].lower().strip()
                career_salary_add = 0

                for section in ("batting", "pitching"):
                    for s in pl.get(section, []):
                        if s.get("salary"):
                            continue
                        yr = s.get("year", 0)
                        team = s.get("team", "")
                        sal = salary.get((name_lower, team, yr))
                        if sal is None:
                            sal = salary.get((name_lower, yr))
                        if sal:
                            s["salary"] = sal
                            s["surplus"] = (s.get("war_value") or 0) - sal
                            career_salary_add += sal
                            filled += 1

                if career_salary_add > 0:
                    old = pl.get("career_salary") or 0
                    pl["career_salary"] = old + career_salary_add
                    old_surplus = pl.get("career_surplus")
                    if old_surplus is not None:
                        pl["career_surplus"] = old_surplus - career_salary_add
                    else:
                        pl["career_surplus"] = (pl.get("career_war_value") or 0) - pl["career_salary"]

            logger.info(f"    Salary augmentation: filled {filled:,} season entries")

            # ── Bulk write ────────────────────────────────────────────────
            records = [
                {
                    "idfg": int(idfg_str),
                    "mlbam": pl.get("mlbam"),
                    "bbref": pl.get("bbref"),
                    "name": pl["name"],
                    "name_lower": pl["name"].lower().strip(),
                    "birth_year": pl.get("birth_year"),
                    "death_year": pl.get("death_year"),
                    "first_year": pl.get("first_year"),
                    "last_year": pl.get("last_year"),
                    "teams": pl.get("teams", []),
                    "career_war": pl.get("career_war", 0),
                    "career_bat_war": pl.get("career_bat_war", 0),
                    "career_pit_war": pl.get("career_pit_war", 0),
                    "career_salary": pl.get("career_salary", 0),
                    "career_war_value": pl.get("career_war_value", 0),
                    "career_surplus": pl.get("career_surplus", 0),
                    "is_pitcher": pl.get("is_pitcher", False),
                    "batting": pl.get("batting", []),
                    "pitching": pl.get("pitching", []),
                }
                for idfg_str, pl in players.items()
            ]

            n = _bulk_insert(self.db, HistoricalPlayer, records)
            logger.info(f"    {n:,} historical players loaded")

            # ── Inject current-season stats from CSV ──────────────────────
            # The historical_players.json only covers through last year.
            # Append current-year actual stats from the daily pipeline CSVs
            # so the DB-only backend (remote deployment) can serve them.
            self._inject_current_season_stats(records)

    def _inject_current_season_stats(self, records: list[dict]) -> None:
        """Append current-season batting/pitching stats to HistoricalPlayer rows."""
        from app.config import CURRENT_YEAR

        current_season_dir = PROJECT_ROOT / "data" / "current_season"
        bat_file = current_season_dir / f"mlb_batting_data_{CURRENT_YEAR}_{CURRENT_YEAR}.csv"
        pit_file = current_season_dir / f"mlb_pitching_data_{CURRENT_YEAR}_{CURRENT_YEAR}.csv"

        if not bat_file.exists() and not pit_file.exists():
            return

        # Build IDfg lookup from just-inserted records
        idfg_set = {r["idfg"] for r in records}

        bat_by_idfg: dict[int, dict] = {}
        pit_by_idfg: dict[int, dict] = {}

        def _sv(v):
            """Safe value: NaN/None → None, else native type."""
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return v

        if bat_file.exists():
            try:
                bat = pd.read_csv(bat_file, low_memory=False)
                bat["IDfg"] = pd.to_numeric(bat["IDfg"], errors="coerce")
                for _, r in bat[bat["Season"] == CURRENT_YEAR].iterrows():
                    idfg = r.get("IDfg")
                    if pd.isna(idfg):
                        continue
                    idfg = int(idfg)
                    if idfg not in idfg_set:
                        continue
                    bat_by_idfg[idfg] = {
                        "year": CURRENT_YEAR, "season": CURRENT_YEAR, "team": _sv(r.get("Team")),
                        "g": _sv(r.get("G")), "pa": _sv(r.get("PA")),
                        "ab": _sv(r.get("AB")), "h": _sv(r.get("H")),
                        "hr": _sv(r.get("HR")),
                        "doubles": _sv(r.get("2B")), "triples": _sv(r.get("3B")),
                        "r": _sv(r.get("R")), "rbi": _sv(r.get("RBI")),
                        "sb": _sv(r.get("SB")), "cs": _sv(r.get("CS")),
                        "bb": _sv(r.get("BB")), "so": _sv(r.get("SO")),
                        "avg": _sv(r.get("AVG")), "obp": _sv(r.get("OBP")),
                        "slg": _sv(r.get("SLG")), "ops": _sv(r.get("OPS")),
                        "woba": _sv(r.get("wOBA")), "wrc_plus": _sv(r.get("wRC+")),
                        "bb_pct": _sv(r.get("BB%")), "k_pct": _sv(r.get("K%")),
                        "babip": _sv(r.get("BABIP")), "war": _sv(r.get("WAR")),
                        "bat": _sv(r.get("Bat")), "bsr": _sv(r.get("BsR")),
                        "def_value": _sv(r.get("Def")),
                    }
            except Exception as e:
                logger.warning(f"    Failed to load current-season batting CSV: {e}")

        if pit_file.exists():
            try:
                pit = pd.read_csv(pit_file, low_memory=False)
                pit["IDfg"] = pd.to_numeric(pit["IDfg"], errors="coerce")
                for _, r in pit[pit["Season"] == CURRENT_YEAR].iterrows():
                    idfg = r.get("IDfg")
                    if pd.isna(idfg):
                        continue
                    idfg = int(idfg)
                    if idfg not in idfg_set:
                        continue
                    pit_by_idfg[idfg] = {
                        "year": CURRENT_YEAR, "season": CURRENT_YEAR, "team": _sv(r.get("Team")),
                        "g": _sv(r.get("G")), "gs": _sv(r.get("GS")),
                        "ip": _sv(r.get("IP")), "w": _sv(r.get("W")),
                        "l": _sv(r.get("L")), "sv": _sv(r.get("SV")),
                        "era": _sv(r.get("ERA")), "fip": _sv(r.get("FIP")),
                        "k_pct": _sv(r.get("K%")), "bb_pct": _sv(r.get("BB%")),
                        "k_9": _sv(r.get("K/9")), "bb_9": _sv(r.get("BB/9")),
                        "hr_9": _sv(r.get("HR/9")), "babip": _sv(r.get("BABIP")),
                        "whip": _sv(r.get("WHIP")), "gb_pct": _sv(r.get("GB%")),
                        "fb_pct": _sv(r.get("FB%")), "hr_fb": _sv(r.get("HR/FB")),
                        "war": _sv(r.get("WAR")),
                    }
            except Exception as e:
                logger.warning(f"    Failed to load current-season pitching CSV: {e}")

        if not bat_by_idfg and not pit_by_idfg:
            return

        # Update HistoricalPlayer rows: append current-season entries
        updated = 0
        for hp in self.db.query(HistoricalPlayer).filter(
            HistoricalPlayer.idfg.in_(set(bat_by_idfg) | set(pit_by_idfg))
        ):
            changed = False
            if hp.idfg in bat_by_idfg:
                batting = list(hp.batting or [])
                # Remove any existing entry for this season
                batting = [s for s in batting if s.get("season") != CURRENT_YEAR]
                batting.append(bat_by_idfg[hp.idfg])
                hp.batting = batting
                changed = True
            if hp.idfg in pit_by_idfg:
                pitching = list(hp.pitching or [])
                pitching = [s for s in pitching if s.get("season") != CURRENT_YEAR]
                pitching.append(pit_by_idfg[hp.idfg])
                hp.pitching = pitching
                changed = True
            if changed:
                # Update last_year if needed
                if hp.last_year is None or hp.last_year < CURRENT_YEAR:
                    hp.last_year = CURRENT_YEAR
                updated += 1

        if updated:
            self.db.flush()
            logger.info(f"    Injected {CURRENT_YEAR} stats into {updated:,} historical players "
                        f"(bat={len(bat_by_idfg)}, pit={len(pit_by_idfg)})")

    # ── Past trades ───────────────────────────────────────────────────────

    def load_past_trades(self, json_path: Path) -> None:
        """Ingest ``trades.json`` with full augmentation pipeline.

        Must be called *after* ``load_historical`` so the WAR augmentation
        has historical data available.
        """
        with _Timer("Past trades"):
            if not json_path.exists():
                logger.warning(f"  File not found: {json_path}")
                return

            from app.routes.historical import _load_historical
            from app.routes.trades import (
                _augment_with_projections,
                _augment_with_historical_war,
                _attach_future_projections,
                _attach_contract_remaining,
                _augment_with_prospect_values,
                _link_prospect_ids,
                _add_has_data_flags,
                _compute_confidence_and_featured,
            )

            with open(json_path, "r") as f:
                trades = _json.load(f)
            logger.info(f"    {len(trades):,} raw trades read")

            # Historical data needed in-memory for WAR augmentation
            _load_historical()

            _augment_with_projections(trades)
            _augment_with_historical_war(trades)
            _attach_future_projections(trades)
            _attach_contract_remaining(trades)
            _augment_with_prospect_values(trades, db=self.db)
            _link_prospect_ids(trades, db=self.db)
            _add_has_data_flags(trades)
            _compute_confidence_and_featured(trades)

            # ── Build records ─────────────────────────────────────────────
            records = []
            for t in trades:
                all_teams: set = set()
                all_names: list = []
                all_mlb_ids: list = []
                for side in t.get("sides", []):
                    all_teams.add(side["team"])
                    for pl in side.get("players_received", []):
                        all_names.append(pl.get("name", "").lower())
                        if pl.get("mlb_id"):
                            all_mlb_ids.append(str(pl["mlb_id"]))

                records.append({
                    "trade_id": t["trade_id"],
                    "date": t["date"],
                    "year": t["year"],
                    "description": t.get("description"),
                    "has_cash": t.get("has_cash", False),
                    "has_ptbnl": t.get("has_ptbnl", False),
                    "n_teams": t.get("n_teams", 2),
                    "n_players": t.get("n_players", 0),
                    "winner": t.get("winner"),
                    "winner_name": t.get("winner_name"),
                    "loser": t.get("loser"),
                    "loser_name": t.get("loser_name"),
                    "surplus_diff": t.get("surplus_diff", 0),
                    "total_trade_war": t.get("total_trade_war", 0),
                    "max_prospect_fv": t.get("max_prospect_fv"),
                    "evaluation_type": t.get("evaluation_type", "actual"),
                    "evaluation_confidence": t.get("evaluation_confidence", "definitive"),
                    "is_featured": t.get("is_featured", False),
                    "projected_winner": t.get("projected_winner"),
                    "projected_winner_name": t.get("projected_winner_name"),
                    "projected_loser": t.get("projected_loser"),
                    "projected_loser_name": t.get("projected_loser_name"),
                    "projected_surplus_diff": t.get("projected_surplus_diff"),
                    "projected_total_war": t.get("projected_total_war"),
                    "sides_json": t.get("sides", []),
                    "teams_csv": ",".join(sorted(all_teams)),
                    "player_names_lower": ",".join(all_names),
                    "player_mlb_ids_csv": ",".join(all_mlb_ids),
                })

            n = _bulk_insert(self.db, PastTrade, records)
            logger.info(f"    {n:,} trades loaded")

    # ── Trade-value history ───────────────────────────────────────────────

    def load_trade_value_history(self, csv_path: Path) -> None:
        """Ingest trade-value history from the legacy CSV **plus** snapshot files.

        1. Load ``trade_value_history.csv`` for historical/transaction entries.
        2. Scan ``snapshots/`` directory for ``YYYY-MM-DD.csv`` files and
           extract current-year trade-value rows from each, using the
           filename as the date.  This makes the DB loader self-sufficient —
           no separate append step is required.
        """
        from app.config import CURRENT_YEAR

        with _Timer("Trade value history"):
            frames: list[pd.DataFrame] = []

            # ── 1. Legacy CSV (historical + transaction entries) ──────────
            if csv_path.exists():
                legacy = pd.read_csv(csv_path, low_memory=False)
                legacy = legacy.rename(columns={"IDfg": "idfg"})
                # Drop current-year snapshot rows from legacy — snapshots
                # are the authoritative source for those.
                legacy = legacy[
                    ~((legacy["year"] == CURRENT_YEAR)
                      & (legacy["transaction_type"].isna()))
                ]
                frames.append(legacy)
                logger.info(f"    Legacy CSV: {len(legacy):,} rows (excl. {CURRENT_YEAR} snapshots)")
            else:
                logger.warning(f"  Legacy file not found: {csv_path}")

            # ── 2. Snapshot files → current-year entries ──────────────────
            snapshot_dir = csv_path.parent / "snapshots"
            snapshot_count = 0
            if snapshot_dir.is_dir():
                import re
                for snap_file in sorted(snapshot_dir.glob("*.csv")):
                    m = re.match(r"^(\d{4}-\d{2}-\d{2})\.csv$", snap_file.name)
                    if not m:
                        continue
                    snap_date = m.group(1)
                    try:
                        snap = pd.read_csv(snap_file, low_memory=False)
                    except Exception as e:
                        logger.warning(f"    Skipping snapshot {snap_file.name}: {e}")
                        continue

                    cur = snap[
                        (snap["Year"] == CURRENT_YEAR)
                        & snap["mlb_id"].notna()
                        & snap["trade_value"].notna()
                    ].copy()
                    if cur.empty:
                        continue

                    name_col = "Player_Name" if "Player_Name" in cur.columns else "name"
                    entries = pd.DataFrame({
                        "mlb_id": cur["mlb_id"].astype(int),
                        "idfg": pd.to_numeric(cur["IDfg"], errors="coerce"),
                        "name": cur[name_col],
                        "date": snap_date,
                        "year": CURRENT_YEAR,
                        "value": cur["trade_value"].round(0),
                        "value_type": "mlb_surplus",
                        "transaction_type": pd.NA,
                        "label": cur.apply(
                            lambda r: (
                                f"{int(r.get('years_control', 0) or 0)}yr control, "
                                f"{r.get('total_future_war', 0) or 0:.1f} WAR"
                            ), axis=1
                        ),
                        "years_control": cur.get("years_control", 0),
                        "projected_war": cur.get("total_future_war", 0),
                        "projected_salary": cur.get("total_contract", 0),
                        "war_per_year": cur.apply(
                            lambda r: (
                                round((r.get("total_future_war", 0) or 0)
                                      / max(r.get("years_control", 0) or 0, 0.001), 2)
                            ), axis=1
                        ),
                    })
                    frames.append(entries)
                    snapshot_count += 1

                logger.info(f"    Snapshots: {snapshot_count} files ingested")
            else:
                logger.info("    No snapshots/ directory found")

            if not frames:
                logger.warning("    No trade-value history data to load")
                return

            df = pd.concat(frames, ignore_index=True)
            for col in ("mlb_id", "idfg", "year"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["mlb_id"])
            df["mlb_id"] = df["mlb_id"].astype(int)

            model_cols = {c.key for c in TradeValueHistory.__table__.columns if c.key != "id"}
            df = df[[c for c in df.columns if c in model_cols]]
            records = _clean_records(df)
            _coerce_int_cols(records, {"mlb_id", "idfg", "year"})
            n = _bulk_insert(self.db, TradeValueHistory, records)
            logger.info(f"    {n:,} trade-value-history rows loaded")

    def upsert_trade_value_history(self, csv_path: Path, target_date: str) -> None:
        """Insert/update only *target_date* rows in trade_value_history.

        Much faster than a full reload for daily pipeline runs.
        """
        with _Timer(f"TVH upsert ({target_date})"):
            if not csv_path.exists():
                logger.warning(f"  File not found: {csv_path}")
                return
            df = pd.read_csv(csv_path, low_memory=False)
            df = df.rename(columns={"IDfg": "idfg"})
            df = df[df["date"] == target_date]
            if df.empty:
                logger.info(f"    No rows for date {target_date}")
                return

            for col in ("mlb_id", "idfg", "year"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["mlb_id"])
            df["mlb_id"] = df["mlb_id"].astype(int)

            model_cols = {c.key for c in TradeValueHistory.__table__.columns if c.key != "id"}
            df = df[[c for c in df.columns if c in model_cols]]
            records = _clean_records(df)
            _coerce_int_cols(records, {"mlb_id", "idfg", "year"})

            deleted = (
                self.db.query(TradeValueHistory)
                .filter(TradeValueHistory.date == target_date)
                .delete()
            )
            self.db.flush()

            n = _bulk_insert(self.db, TradeValueHistory, records)
            logger.info(
                f"    Upserted {n:,} rows (replaced {deleted:,}) "
                f"for {target_date}"
            )

    # ── Statcast expected stats ───────────────────────────────────────────

    def load_statcast_expected(
        self, batter_csv: Path, pitcher_csv: Path
    ) -> None:
        """Merge batter + pitcher Statcast expected-stats CSVs into one table."""
        with _Timer("Statcast expected stats"):
            merged: dict[tuple[int, int], dict] = {}

            # Batters: xba, xslg, xwoba
            if batter_csv.exists():
                df = pd.read_csv(batter_csv, low_memory=False)
                for _, r in df.iterrows():
                    try:
                        key = (int(r["player_id"]), int(r["year"]))
                    except (ValueError, TypeError):
                        continue
                    merged[key] = {
                        "player_id": key[0], "year": key[1],
                        "xba": r.get("est_ba"), "xslg": r.get("est_slg"),
                        "xwoba": r.get("est_woba"),
                    }
                logger.info(f"    Batter CSV: {len(df):,} rows")
            else:
                logger.warning(f"  Batter CSV not found: {batter_csv}")

            # Pitchers: xera (merge into existing row if present)
            if pitcher_csv.exists():
                df = pd.read_csv(pitcher_csv, low_memory=False)
                for _, r in df.iterrows():
                    try:
                        key = (int(r["player_id"]), int(r["year"]))
                    except (ValueError, TypeError):
                        continue
                    entry = merged.get(key, {"player_id": key[0], "year": key[1]})
                    entry["xera"] = r.get("xera")
                    merged[key] = entry
                logger.info(f"    Pitcher CSV: {len(df):,} rows")
            else:
                logger.warning(f"  Pitcher CSV not found: {pitcher_csv}")

            records = list(merged.values())
            # Clean NaN values
            for rec in records:
                for k, v in rec.items():
                    if isinstance(v, float) and pd.isna(v):
                        rec[k] = None
            n = _bulk_insert(self.db, StatcastExpected, records)
            logger.info(f"    {n:,} statcast expected-stat rows loaded")

    # ── Spotrac transactions ──────────────────────────────────────────────

    def load_spotrac_transactions(self, csv_path: Path) -> None:
        """Ingest ``spotrac_transactions.csv`` — contract events only."""
        with _Timer("Spotrac transactions"):
            if not csv_path.exists():
                logger.warning(f"  File not found: {csv_path}")
                return
            df = pd.read_csv(csv_path, low_memory=False)

            # Keep only contract-relevant types (same filter as the old route code)
            contract_types = {
                "extension", "fa_signing", "signing",
                "elected_fa", "option_exercised", "option_declined",
            }
            df = df[df["transaction_type"].isin(contract_types)].copy()

            # Filter out arbitration settlements misclassified as fa_signing or signing
            arb_mask = (
                df["transaction_type"].isin(["fa_signing", "signing"])
                & df["description"].str.contains("arbitration", case=False, na=False)
            )
            df = df[~arb_mask]

            # Filter out pre-arbitration 1-year minimal salary contracts
            import re
            pre_arb_re = re.compile(r"signed a 1 year \$[\d,]+(?:k|K)?\s+contract", re.IGNORECASE)
            pre_arb_mask = (
                df["transaction_type"].isin(["fa_signing", "signing"])
                & df["description"].str.contains(pre_arb_re, na=False, regex=True)
            )
            # Further verify salary is under a generous minimal threshold (e.g. < $1.0M)
            # Since historical minimums vary, $1M is a safe upper bound for pre-arb before 2025 except 2026. 
            # Or we just rely on exactly this formulation since 1-year low deals without arbitration description are pre-arb.
            
            def is_low_salary(desc: str) -> bool:
                match = re.search(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(k|K)?", str(desc))
                if not match:
                    return False
                sal_str = match.group(1).replace(",", "")
                try:
                    salary = float(sal_str)
                    if match.group(2):
                        salary *= 1000
                    return salary <= 1500000.0  # safe threshold
                except ValueError:
                    return False
            
            low_sal_mask = df["description"].apply(is_low_salary)
            df = df[~(pre_arb_mask & low_sal_mask)]

            # Filter out minor league signings
            minor_league_mask = (
                df["transaction_type"].isin(["fa_signing", "signing"])
                & df["description"].str.contains("minor league", case=False, na=False)
            )
            df = df[~minor_league_mask]

            # Filter out initial free agent/draft signings (no dollar amount)
            initial_signing_mask = (
                df["transaction_type"].isin(["fa_signing", "signing"])
                & ~df["description"].str.contains(r"\$", na=False, regex=True)
                & ~df["description"].str.contains("minor league", case=False, na=False)
            )
            df = df[~initial_signing_mask]

            # Add lowered name for indexed lookups
            df["player_name_lower"] = df["player_name"].str.strip().str.lower()

            model_cols = {c.key for c in SpotracTransaction.__table__.columns if c.key != "id"}
            df = df[[c for c in df.columns if c in model_cols]]
            records = _clean_records(df)
            n = _bulk_insert(self.db, SpotracTransaction, records)
            logger.info(f"    {n:,} spotrac transactions loaded")

    # ── Fielding stats (historical + projected) ─────────────────────────────

    def load_fielding_stats(
        self,
        historical_path: Path,
        projections_path: Path,
        crosswalk_path: Path,
    ) -> None:
        """Load historical fielding data and fielding projections into
        the ``fielding_stats`` table.

        Historical data comes from the FanGraphs + Statcast merged CSV.
        Projections come from the value-determination pipeline output.
        Both use IDfg as primary key; mlb_id is resolved via the crosswalk.
        """
        with _Timer("Fielding stats"):
            # ── Build IDfg → mlbam_id map from crosswalk ──────────────
            id_map: dict[int, int] = {}
            if crosswalk_path.exists():
                cw = pd.read_csv(crosswalk_path, encoding="utf-8-sig")
                cw.columns = cw.columns.str.strip()
                cw["fg_id_num"] = pd.to_numeric(cw["fg_id"], errors="coerce")
                cw = cw.dropna(subset=["fg_id_num", "mlbam_id"])
                id_map = dict(zip(cw["fg_id_num"].astype(int), cw["mlbam_id"].astype(int)))
                logger.info(f"    Crosswalk loaded: {len(id_map):,} IDfg→mlbam mappings")

            all_records: list[dict] = []

            # ── 1. Historical fielding ────────────────────────────────
            if historical_path.exists():
                df = pd.read_csv(historical_path, low_memory=False)
                # Exclude pitchers — only positional fielding
                df = df[df["Pos"] != "P"].copy()

                df["mlb_id"] = df["IDfg"].map(id_map)
                # Also fill from sc_mlbam_id where available
                if "sc_mlbam_id" in df.columns:
                    sc_ids = pd.to_numeric(df["sc_mlbam_id"], errors="coerce")
                    df["mlb_id"] = df["mlb_id"].fillna(sc_ids)

                df["mlb_id"] = df["mlb_id"].apply(
                    lambda x: int(x) if pd.notna(x) else None
                )

                records = []
                for _, row in df.iterrows():
                    rec = {
                        "idfg": int(row["IDfg"]) if pd.notna(row.get("IDfg")) else None,
                        "mlb_id": row["mlb_id"],
                        "name": row.get("Name"),
                        "season": int(row["Season"]),
                        "team": row.get("Team"),
                        "pos": row.get("Pos"),
                        "age": row.get("Age"),
                        "g": int(row["G"]) if pd.notna(row.get("G")) else None,
                        "gs": int(row["GS"]) if pd.notna(row.get("GS")) else None,
                        "inn": row.get("Inn"),
                        "sc_total_runs": row.get("sc_total_runs"),
                        "sc_range_runs": row.get("sc_range_runs"),
                        "sc_arm_runs": row.get("sc_arm_runs"),
                        "sc_dp_runs": row.get("sc_dp_runs"),
                        "sc_framing_runs": row.get("sc_framing_runs"),
                        "sc_throwing_runs": row.get("sc_throwing_runs"),
                        "sc_blocking_runs": row.get("sc_blocking_runs"),
                        "drs": row.get("DRS"),
                        "uzr": row.get("UZR"),
                        "uzr_150": row.get("UZR/150"),
                        "oaa": int(row["OAA"]) if pd.notna(row.get("OAA")) else None,
                        "errors": int(row["E"]) if pd.notna(row.get("E")) else None,
                        "fp": row.get("FP"),
                        "is_projection": 0,
                    }
                    records.append(rec)

                # Vectorised NaN → None cleanup
                for rec in records:
                    for k, v in rec.items():
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            rec[k] = None

                all_records.extend(records)
                logger.info(f"    Historical fielding: {len(records):,} rows (non-pitcher)")
            else:
                logger.warning(f"    Historical fielding file not found: {historical_path}")

            # ── 2. Fielding projections ───────────────────────────────
            if projections_path.exists():
                df = pd.read_csv(projections_path, low_memory=False)
                df["mlb_id"] = df["IDfg"].map(id_map)
                df["mlb_id"] = df["mlb_id"].apply(
                    lambda x: int(x) if pd.notna(x) else None
                )

                records = []
                for _, row in df.iterrows():
                    rec = {
                        "idfg": int(row["IDfg"]) if pd.notna(row.get("IDfg")) else None,
                        "mlb_id": row["mlb_id"],
                        "name": row.get("Name"),
                        "season": int(row["Year"]),
                        "team": row.get("Team"),
                        "pos": row.get("Pos"),
                        "age": row.get("Age"),
                        "g": int(row["G"]) if pd.notna(row.get("G")) else None,
                        "gs": int(row["GS"]) if pd.notna(row.get("GS")) else None,
                        "inn": row.get("Inn"),
                        "sc_total_runs": row.get("sc_total_runs/150") if "sc_total_runs/150" in row.index else row.get("sc_total_runs"),
                        "sc_range_runs": row.get("sc_range_runs/150") if "sc_range_runs/150" in row.index else row.get("sc_range_runs"),
                        "sc_arm_runs": row.get("sc_arm_runs/150") if "sc_arm_runs/150" in row.index else row.get("sc_arm_runs"),
                        "sc_dp_runs": row.get("sc_dp_runs/150") if "sc_dp_runs/150" in row.index else row.get("sc_dp_runs"),
                        "sc_framing_runs": row.get("sc_framing_runs/150") if "sc_framing_runs/150" in row.index else row.get("sc_framing_runs"),
                        "sc_throwing_runs": row.get("sc_throwing_runs/150") if "sc_throwing_runs/150" in row.index else row.get("sc_throwing_runs"),
                        "sc_blocking_runs": row.get("sc_blocking_runs/150") if "sc_blocking_runs/150" in row.index else row.get("sc_blocking_runs"),
                        "drs": None,
                        "uzr": None,
                        "uzr_150": None,
                        "oaa": None,
                        "errors": None,
                        "fp": None,
                        "is_projection": 1,
                    }
                    records.append(rec)

                for rec in records:
                    for k, v in rec.items():
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            rec[k] = None

                all_records.extend(records)
                logger.info(f"    Fielding projections: {len(records):,} rows")
            else:
                logger.warning(f"    Fielding projections file not found: {projections_path}")

            n = _bulk_insert(self.db, FieldingStats, all_records)
            logger.info(f"    {n:,} total fielding rows loaded")

    # ── Legacy shim ───────────────────────────────────────────────────────

    def reset_and_load_data(self, players_csv: str, prospects_csv: str = None):
        """Backward-compat wrapper used by ``init_db``."""
        try:
            logger.info("  Clearing players + prospects ...")
            if self._is_sqlite:
                self.db.execute(text("DELETE FROM players;"))
                self.db.execute(text("DELETE FROM prospects;"))
            else:
                self.db.execute(text("TRUNCATE TABLE players CASCADE;"))
                self.db.execute(text("TRUNCATE TABLE prospects CASCADE;"))
            self.db.commit()

            self.load_players(Path(players_csv), projection_type="ros")
            if prospects_csv:
                self.load_prospects(Path(prospects_csv))
        except Exception as e:
            logger.error(f"Error in reset_and_load_data: {e}")
            self.db.rollback()
            raise


# ══════════════════════════════════════════════════════════════════════════
# Standalone entry-point
# ══════════════════════════════════════════════════════════════════════════

def init_db():
    """Drop -> recreate all tables -> run the full ETL pipeline."""
    banner = "=" * 64
    logger.info(f"\n{banner}\n  DATABASE INITIALISATION — {time.strftime('%Y-%m-%d %H:%M:%S')}\n{banner}")
    t_total = time.perf_counter()

    # ── 1. Schema reset ───────────────────────────────────────────────────
    logger.info("[1/13] Resetting schema ...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("  Tables recreated")

    # ── Verify data files ─────────────────────────────────────────────────
    logger.info("\nData-file manifest:")
    for key, path in DATA_PATHS.items():
        status = "OK" if path.exists() else "MISSING"
        logger.info(f"  {key:22s}  {status:7s}  {path}")

    critical = ("players", "prospects")
    if any(not DATA_PATHS[k].exists() for k in critical):
        logger.error("Critical data files missing — aborting.")
        return

    db = SessionLocal()
    try:
        loader = DataLoader(db)

        # ── 2. Players ────────────────────────────────────────────────────
        logger.info("\n[2/13] Loading players ...")
        loader.load_players(DATA_PATHS["players"], projection_type="ros")

        # ── 2b. Preseason players (current-year only) ─────────────────────
        preseason_dir = DATA_PATHS["players_preseason"]
        preseason_file = _find_latest_preseason(preseason_dir)
        if preseason_file:
            logger.info(f"\n[2b/13] Loading preseason projections ({preseason_file.name}) ...")
            loader.load_players(preseason_file, projection_type="preseason")
        else:
            logger.info("\n[2b/13] No preseason file — skipping")

        # ── 3. Prospects ──────────────────────────────────────────────────
        logger.info("\n[3/13] Loading prospects ...")
        loader.load_prospects(DATA_PATHS["prospects"])

        # ── 4. Historical players ─────────────────────────────────────────
        logger.info("\n[4/13] Loading historical players ...")
        if DATA_PATHS["historical"].exists():
            loader.load_historical(DATA_PATHS["historical"])
        else:
            logger.warning("  Skipped (file not found)")

        # ── 5. Past trades ────────────────────────────────────────────────
        logger.info("\n[5/13] Loading past trades ...")
        if DATA_PATHS["trades"].exists():
            loader.load_past_trades(DATA_PATHS["trades"])
        else:
            logger.warning("  Skipped (file not found)")

        # ── 6. MiLB stats ────────────────────────────────────────────────
        logger.info("\n[6/13] Loading MiLB statistics ...")
        loader.load_milb_stats(DATA_PATHS["milb_hitters"], DATA_PATHS["milb_pitchers"])

        # ── 7. Crosswalk ──────────────────────────────────────────────────
        logger.info("\n[7/13] Loading player-ID crosswalk ...")
        if DATA_PATHS["crosswalk"].exists():
            loader.load_crosswalk(DATA_PATHS["crosswalk"])
        else:
            logger.warning("  Skipped — run scrapers/build_id_crosswalk.py first")

        # ── 8. Prospect IDfg resolution ───────────────────────────────────
        logger.info("\n[8/13] Resolving prospect FanGraphs IDs (crosswalk) ...")
        loader.resolve_prospect_idfg()

        # ── 9. Resolve has_mlb (integer mlbam_id match — <1 s) ────────────
        logger.info("\n[9/13] Resolving prospect has_mlb ...")
        loader.resolve_prospect_has_mlb()

        # ── 10. Trade-value history ───────────────────────────────────────
        logger.info("\n[10/13] Loading trade-value history ...")
        loader.load_trade_value_history(DATA_PATHS["trade_value_history"])

        # ── 11. Statcast expected stats ───────────────────────────────────
        logger.info("\n[11/13] Loading Statcast expected stats ...")
        loader.load_statcast_expected(
            DATA_PATHS["statcast_batter"], DATA_PATHS["statcast_pitcher"]
        )

        # ── 12. Spotrac transactions ──────────────────────────────────────
        logger.info("\n[12/13] Loading Spotrac transactions ...")
        loader.load_spotrac_transactions(DATA_PATHS["spotrac_transactions"])

        # ── 13. Fielding stats (historical + projected) ───────────────────
        logger.info("\n[13/13] Loading fielding statistics ...")
        loader.load_fielding_stats(
            DATA_PATHS["fielding_historical"],
            DATA_PATHS["fielding_projections"],
            DATA_PATHS["crosswalk_for_fielding"],
        )

        elapsed = time.perf_counter() - t_total
        logger.info(
            f"\n{banner}\n"
            f"  COMPLETE — total elapsed: {elapsed:.1f}s\n"
            f"{banner}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    init_db()
