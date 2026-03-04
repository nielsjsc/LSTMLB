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
   prospect ``IDfg`` from the crosswalk, replacing the old Python loop.

Performance
-----------
Full reload (all tables, ~250 K total rows): **< 60 s** on a typical
laptop with SQLite.  Previously 30+ min with row-by-row ORM inserts.
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
from app.config import PROSPECT_YEARS
from app.database import SessionLocal, engine, Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Data-file manifest
# ══════════════════════════════════════════════════════════════════════════
# Single source of truth for every data file the loader touches.
# All paths relative to the project root (LSTMLB/).

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

DATA_PATHS: Dict[str, Path] = {
    # ── Generated / processed ─────────────────────────────────────────
    "players":       PROJECT_ROOT / "data" / "generated" / "value_by_year" / "player_values_complete.csv",
    "prospects":     PROJECT_ROOT / "data" / "generated" / "MiLB" / "prospect_histories.csv",
    "historical":    PROJECT_ROOT / "data" / "generated" / "historical_players" / "historical_players.json",
    "trades":        PROJECT_ROOT / "data" / "generated" / "past_trades" / "trades.json",
    "crosswalk":     PROJECT_ROOT / "data" / "generated" / "player_id_crosswalk.csv",
    # ── Raw / scraped ─────────────────────────────────────────────────
    "milb_hitters":  PROJECT_ROOT / "data" / "MiLB" / "MiLB_Hitters.csv",
    "milb_pitchers": PROJECT_ROOT / "data" / "MiLB" / "MiLB_Pitchers.csv",
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
PLAYER_COL_MAP = {
    "IDfg": "real_id", "mlb_id": "mlb_id",
    "Name": "name", "Team": "team", "Position": "position", "status": "status",
    "Age": "age", "Year": "year",
    "years_control": "years_control", "FA_Year": "fa_year",
    "Probable_FA_Year": "probable_fa_year", "Earliest_FA_Year": "earliest_fa_year",
    "control_through": "control_through",
    "base_value": "base_value", "contract_value": "contract_value",
    "surplus_value": "surplus_value", "trade_value": "trade_value",
    "contract_war": "contract_war",
    # Hitting
    "G_bat": "g_bat", "WAR_bat": "war_bat",
    "BB%_bat": "bb_pct_bat", "K%_bat": "k_pct_bat",
    "AVG": "avg", "OBP": "obp", "SLG": "slg", "OPS": "ops",
    "wOBA": "woba", "wRC+": "wrc_plus", "EV": "ev",
    "Off": "off", "BsR": "bsr", "Def": "def_value",
    "HR": "hr", "2B": "doubles", "3B": "triples",
    "R": "r", "RBI": "rbi", "SB": "sb", "CS": "cs",
    # Pitching
    "G_pit": "g_pit", "GS": "gs", "IP": "ip", "WAR_pit": "war_pit",
    "ERA": "era", "FIP": "fip", "K%_pit": "k_pct_pit", "BB%_pit": "bb_pct_pit",
    # Aggregate values
    "avg_war": "avg_war", "total_contract": "total_contract",
    "avg_contract": "avg_contract",
    "total_future_war": "total_future_war", "total_future_value": "total_future_value",
    "total_value": "total_value", "total_war": "total_war",
    "historical_value": "historical_value", "historical_war": "historical_war",
    "contract_base_value": "contract_base_value",
}

PLAYER_INT_COLS = {
    "real_id", "mlb_id", "age", "year", "years_control", "fa_year",
    "probable_fa_year", "earliest_fa_year", "control_through",
    "g_bat", "hr", "doubles", "triples", "r", "rbi", "sb", "cs",
    "g_pit", "gs",
}

PLAYER_STR_COLS = {"name", "team", "position", "status"}

# Prospect CSV → model
PROSPECT_COL_MAP = {
    "Year": "year", "Name": "name", "Team": "org",
    "Position": "position", "FV": "fv", "Age": "age",
    "Hit": "hit", "Game": "game_power", "Raw": "raw_power", "Spd": "speed",
    "FB": "fastball", "SL": "slider", "CB": "curve",
    "CH": "changeup", "CMD": "command",
}

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

    def load_players(self, csv_path: Path) -> None:
        """Vectorised CSV read → Core bulk insert for the *players* table."""
        with _Timer("Players"):
            df = pd.read_csv(csv_path, low_memory=False)
            initial = len(df)

            # Rename CSV headers → DB column names
            df = df.rename(columns=PLAYER_COL_MAP)

            # Keep only columns that exist in the model
            model_cols = {c.key for c in Player.__table__.columns if c.key != "id"}
            df = df[[c for c in df.columns if c in model_cols]]

            # Drop rows without mlb_id (can't link them to anything)
            df = df.dropna(subset=["mlb_id"])

            # Numeric coercion (vectorised)
            for col in df.columns:
                if col not in PLAYER_STR_COLS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            records = _clean_records(df)
            _coerce_int_cols(records, PLAYER_INT_COLS)

            n = _bulk_insert(self.db, Player, records)
            logger.info(f"    {n:,} players loaded ({initial - n:,} skipped — no mlb_id)")

    # ── Prospects ─────────────────────────────────────────────────────────

    def load_prospects(self, csv_path: Path) -> None:
        """Load prospect data.  JSON dict columns (``values_by_year``,
        ``composites_by_year``) are built via ``apply`` (fast for ~2 K rows).
        """
        with _Timer("Prospects"):
            df = pd.read_csv(csv_path)

            # -- Build year-keyed JSON dicts --------------------------------
            def _build_year_dict(row, suffix: str) -> dict:
                d = {}
                for yr in PROSPECT_YEARS:
                    col = f"{yr}_{suffix}"
                    if col in row.index and pd.notna(row[col]):
                        d[str(yr)] = float(row[col])
                return d

            df["values_by_year"] = df.apply(
                lambda r: _build_year_dict(r, "Value"), axis=1
            )
            df["composites_by_year"] = df.apply(
                lambda r: _build_year_dict(r, "Composite"), axis=1
            )

            # Drop the raw year columns now that they're collapsed
            year_cols = (
                [f"{yr}_Value" for yr in PROSPECT_YEARS]
                + [f"{yr}_Composite" for yr in PROSPECT_YEARS]
            )
            df = df.drop(columns=[c for c in year_cols if c in df.columns])

            # Rename
            df = df.rename(columns=PROSPECT_COL_MAP)

            # -- IDfg (FanGraphs MiLB IDs: "sa"-prefixed strings) ----------
            df["IDfg"] = df["IDfg"].astype(str).str.strip()
            df.loc[df["IDfg"].isin(["nan", "", "None", "NaN", "<NA>"]), "IDfg"] = None

            # -- mlbam_id ---------------------------------------------------
            df["mlbam_id"] = pd.to_numeric(df.get("mlbam_id"), errors="coerce")

            # -- has_mlb default False --------------------------------------
            df["has_mlb"] = df["has_mlb"].fillna(False).astype(bool)

            # Keep only model columns
            model_cols = {c.key for c in Prospect.__table__.columns if c.key != "id"}
            df = df[[c for c in df.columns if c in model_cols]]

            records = _clean_records(df)

            # Fix int types for mlbam_id / year
            for rec in records:
                if rec.get("mlbam_id") is not None:
                    rec["mlbam_id"] = int(rec["mlbam_id"])
                if rec.get("year") is not None:
                    rec["year"] = int(rec["year"])

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

            from app.routes.historical import (
                _load_salary_supplement,
                _load_spotrac_salaries,
                _load_lahman_salaries,
            )

            with open(json_path, "r") as f:
                data = _json.load(f)

            players = data.get("players", {})
            logger.info(f"    {len(players):,} players read from JSON")

            # ── Salary augmentation (three-source merge) ──────────────────
            by_year = _load_salary_supplement()
            spotrac = _load_spotrac_salaries()
            lahman = _load_lahman_salaries()

            for key, sal in spotrac.items():
                if key not in by_year:
                    by_year[key] = sal

            filled = 0
            for pl in players.values():
                name_lower = pl["name"].lower().strip()
                bbref_id = pl.get("bbref", "")
                career_salary_add = 0

                for section in ("batting", "pitching"):
                    for s in pl.get(section, []):
                        if s.get("salary"):
                            continue
                        yr = s.get("year", 0)
                        team = s.get("team", "")
                        sal = by_year.get((name_lower, team, yr))
                        if sal is None and bbref_id:
                            sal = lahman.get((bbref_id, yr))
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
            _augment_with_prospect_values(trades)
            _link_prospect_ids(trades)
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

            self.load_players(Path(players_csv))
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
    logger.info("[1/8] Resetting schema ...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("  Tables recreated")

    # ── Verify data files ─────────────────────────────────────────────────
    logger.info("\nData-file manifest:")
    for key, path in DATA_PATHS.items():
        status = "OK" if path.exists() else "MISSING"
        logger.info(f"  {key:16s}  {status:7s}  {path}")

    critical = ("players", "prospects")
    if any(not DATA_PATHS[k].exists() for k in critical):
        logger.error("Critical data files missing — aborting.")
        return

    db = SessionLocal()
    try:
        loader = DataLoader(db)

        # ── 2. Players ────────────────────────────────────────────────────
        logger.info("\n[2/8] Loading players ...")
        loader.load_players(DATA_PATHS["players"])

        # ── 3. Prospects ──────────────────────────────────────────────────
        logger.info("\n[3/8] Loading prospects ...")
        loader.load_prospects(DATA_PATHS["prospects"])

        # ── 4. Historical players ─────────────────────────────────────────
        logger.info("\n[4/8] Loading historical players ...")
        if DATA_PATHS["historical"].exists():
            loader.load_historical(DATA_PATHS["historical"])
        else:
            logger.warning("  Skipped (file not found)")

        # ── 5. Past trades ────────────────────────────────────────────────
        logger.info("\n[5/8] Loading past trades ...")
        if DATA_PATHS["trades"].exists():
            loader.load_past_trades(DATA_PATHS["trades"])
        else:
            logger.warning("  Skipped (file not found)")

        # ── 6. MiLB stats ────────────────────────────────────────────────
        logger.info("\n[6/8] Loading MiLB statistics ...")
        loader.load_milb_stats(DATA_PATHS["milb_hitters"], DATA_PATHS["milb_pitchers"])

        # ── 7. Crosswalk ──────────────────────────────────────────────────
        logger.info("\n[7/8] Loading player-ID crosswalk ...")
        if DATA_PATHS["crosswalk"].exists():
            loader.load_crosswalk(DATA_PATHS["crosswalk"])
        else:
            logger.warning("  Skipped — run scrapers/build_id_crosswalk.py first")

        # ── 8. Prospect IDfg resolution ───────────────────────────────────
        logger.info("\n[8/8] Resolving prospect FanGraphs IDs ...")
        loader.resolve_prospect_idfg()

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
