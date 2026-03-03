#######
#To reload data to local db and then server, run this script, 
#ensure db is present in /backend, then ctrl+c in backend cmd, and run:
#python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
#######

import sys
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

load_dotenv(backend_dir / '.env')
from app.models.player import Player
from app.models.prospect import Prospect
from app.models.historical import HistoricalPlayer
from app.models.past_trade import PastTrade
from app.models.milb_stats import MiLBHittingStats, MiLBPitchingStats
from app.config import PROSPECT_YEARS
from app.database import SessionLocal, engine, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, db: Session):
        self.db = db

    def transform_player_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform CSV row data into correct types for database model"""
        return {
            # Integer fields
            'real_id': int(row['IDfg']) if pd.notna(row['IDfg']) else None,
            'mlb_id': int(row['mlb_id']) if pd.notna(row.get('mlb_id')) else None,
            'age': int(row['Age']) if pd.notna(row['Age']) else None,
            'year': int(row['Year']) if pd.notna(row['Year']) else None,
            'years_control': int(row.get('years_control')) if pd.notna(row.get('years_control')) else None,
            'fa_year': int(row.get('FA_Year')) if pd.notna(row.get('FA_Year')) else None,
            'probable_fa_year': int(row.get('Probable_FA_Year')) if pd.notna(row.get('Probable_FA_Year')) else None,
            'earliest_fa_year': int(row.get('Earliest_FA_Year')) if pd.notna(row.get('Earliest_FA_Year')) else None,
            'control_through': int(row.get('control_through')) if pd.notna(row.get('control_through')) else None,
            
            # Integer stats
            'g_bat': int(row['G_bat']) if pd.notna(row['G_bat']) else None,
            'g_pit': int(row['G_pit']) if pd.notna(row['G_pit']) else None,
            'gs': int(row['GS']) if pd.notna(row['GS']) else None,
            'hr': int(row.get('HR')) if pd.notna(row.get('HR')) else None,
            'doubles': int(row.get('2B')) if pd.notna(row.get('2B')) else None,
            'triples': int(row.get('3B')) if pd.notna(row.get('3B')) else None,
            'r': int(row.get('R')) if pd.notna(row.get('R')) else None,
            'rbi': int(row.get('RBI')) if pd.notna(row.get('RBI')) else None,
            'sb': int(row.get('SB')) if pd.notna(row.get('SB')) else None,
            'cs': int(row.get('CS')) if pd.notna(row.get('CS')) else None,

            # String fields
            'name': str(row['Player_Name']) if pd.notna(row['Player_Name']) else None,
            'team': str(row['Team']) if pd.notna(row['Team']) else None,
            'position': str(row['Position']) if pd.notna(row['Position']) else None,
            'status': str(row.get('Status')) if pd.notna(row.get('Status')) else None,

            # Float fields - Hitting stats
            'war_bat': float(row.get('WAR_batter')) if pd.notna(row.get('WAR_batter')) else None,
            'bb_pct_bat': float(row.get('BB%_bat')) if pd.notna(row.get('BB%_bat')) else None,
            'k_pct_bat': float(row.get('K%_bat')) if pd.notna(row.get('K%_bat')) else None,
            'avg': float(row.get('AVG')) if pd.notna(row.get('AVG')) else None,
            'obp': float(row.get('OBP')) if pd.notna(row.get('OBP')) else None,
            'slg': float(row.get('SLG')) if pd.notna(row.get('SLG')) else None,
            'ops': float(row.get('OPS')) if pd.notna(row.get('OPS')) else None,
            'woba': float(row.get('wOBA')) if pd.notna(row.get('wOBA')) else None,
            'wrc_plus': float(row.get('wRC+')) if pd.notna(row.get('wRC+')) else None,
            'ev': float(row.get('EV')) if pd.notna(row.get('EV')) else None,
            'off': float(row.get('Off')) if pd.notna(row.get('Off')) else None,
            'bsr': float(row.get('BsR')) if pd.notna(row.get('BsR')) else None,
            'def_value': float(row.get('Def')) if pd.notna(row.get('Def')) else None,

            # Float fields - Pitching stats
            'war_pit': float(row.get('WAR_pitcher')) if pd.notna(row.get('WAR_pitcher')) else None,
            'era': float(row.get('ERA')) if pd.notna(row.get('ERA')) else None,
            'fip': float(row.get('FIP')) if pd.notna(row.get('FIP')) else None,
            'ip': float(row.get('IP')) if pd.notna(row.get('IP')) else None,
            'k_pct_pit': float(row.get('K%_pit')) if pd.notna(row.get('K%_pit')) else None,
            'bb_pct_pit': float(row.get('BB%_pit')) if pd.notna(row.get('BB%_pit')) else None,

            # Float fields - Value metrics
            'base_value': float(row.get('Base_Value')) if pd.notna(row.get('Base_Value')) else None,
            'contract_value': float(row.get('Contract_Value')) if pd.notna(row.get('Contract_Value')) else None,
            'surplus_value': float(row.get('Surplus_Value')) if pd.notna(row.get('Surplus_Value')) else None,
            'trade_value': float(row.get('trade_value')) if pd.notna(row.get('trade_value')) else None,
            'contract_war': float(row.get('contract_war')) if pd.notna(row.get('contract_war')) else None,
            'avg_war': float(row.get('avg_war')) if pd.notna(row.get('avg_war')) else None,
            'total_contract': float(row.get('total_contract')) if pd.notna(row.get('total_contract')) else None,
            'avg_contract': float(row.get('avg_contract')) if pd.notna(row.get('avg_contract')) else None,
            'total_future_war': float(row.get('total_future_war')) if pd.notna(row.get('total_future_war')) else None,
            'total_future_value': float(row.get('total_future_value')) if pd.notna(row.get('total_future_value')) else None,
            'total_value': float(row.get('total_value')) if pd.notna(row.get('total_value')) else None,
            'total_war': float(row.get('total_war')) if pd.notna(row.get('total_war')) else None,
            'historical_value': float(row.get('historical_value')) if pd.notna(row.get('historical_value')) else None,
            'historical_war': float(row.get('historical_war')) if pd.notna(row.get('historical_war')) else None,
            'contract_base_value': float(row.get('contract_base_value')) if pd.notna(row.get('contract_base_value')) else None,
        }
    def transform_prospect_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform CSV row data into correct types for Prospect model"""
        # Handle IDfg as Integer
        id_fg = None
        if pd.notna(row['IDfg']):
            try:
                id_fg = int(row['IDfg'])
            except ValueError:
                id_fg = None
                
        return {
            # Integer fields
            'IDfg': id_fg,
            'year': int(row['Year']) if pd.notna(row['Year']) else None,
            
            # String fields
            'name': str(row['Name']) if pd.notna(row['Name']) else None,
            'org': str(row['Team']) if pd.notna(row['Team']) else None,
            'position': str(row['Position']) if pd.notna(row['Position']) else None,
            'fv': str(row['FV']) if pd.notna(row['FV']) else None,
            
            # Boolean fields
            'has_mlb': bool(row['has_mlb']) if pd.notna(row['has_mlb']) else False,
            
            # Float fields
            'age': float(row['Age']) if pd.notna(row['Age']) else None,
            
            # Year-keyed value & composite maps (dynamic — no schema change per season)
            'values_by_year': {
                str(yr): float(row[f'{yr}_Value'])
                for yr in PROSPECT_YEARS
                if pd.notna(row.get(f'{yr}_Value'))
            },
            'composites_by_year': {
                str(yr): float(row[f'{yr}_Composite'])
                for yr in PROSPECT_YEARS
                if pd.notna(row.get(f'{yr}_Composite'))
            },
            
            # Tool grades (String)
            'hit': str(row.get('Hit')) if pd.notna(row.get('Hit')) else None,
            'game_power': str(row.get('Game')) if pd.notna(row.get('Game')) else None,
            'raw_power': str(row.get('Raw')) if pd.notna(row.get('Raw')) else None,
            'speed': str(row.get('Spd')) if pd.notna(row.get('Spd')) else None,
            'fastball': str(row.get('FB')) if pd.notna(row.get('FB')) else None,
            'slider': str(row.get('SL')) if pd.notna(row.get('SL')) else None,
            'curve': str(row.get('CB')) if pd.notna(row.get('CB')) else None,
            'changeup': str(row.get('CH')) if pd.notna(row.get('CH')) else None,
            'command': str(row.get('CMD')) if pd.notna(row.get('CMD')) else None
        }
        

    def load_player_data(self, players_csv: str) -> None:
        """Bulk-insert player rows, skipping those without mlb_id."""
        try:
            df = pd.read_csv(players_csv)
            logger.info(f"Loading {len(df)} players from {players_csv}")
            
            objects = []
            skipped_count = 0
            
            for _, row in df.iterrows():
                data = self.transform_player_data(row.to_dict())
                if data.get('mlb_id') is None:
                    skipped_count += 1
                    continue
                objects.append(Player(**data))
            
            self.db.bulk_save_objects(objects)
            self.db.commit()
            logger.info(f"Player data loaded: {len(objects)} inserted, {skipped_count} skipped (no mlb_id)")
                
        except Exception as e:
            logger.error(f"Error loading player data: {str(e)}")
            self.db.rollback()
            raise

    def load_prospect_data(self, prospects_csv: str) -> None:
        """Bulk-insert prospect rows."""
        try:
            df = pd.read_csv(prospects_csv)
            logger.info(f"Loading {len(df)} prospects from {prospects_csv}")
            
            objects = [Prospect(**self.transform_prospect_data(row.to_dict()))
                       for _, row in df.iterrows()]
            
            self.db.bulk_save_objects(objects)
            self.db.commit()
            logger.info(f"Prospect data loaded: {len(objects)} inserted")
                
        except Exception as e:
            logger.error(f"Error loading prospect data: {str(e)}")
            self.db.rollback()
            raise

    # ── Historical players ────────────────────────────────────────────────

    def load_historical_data(self, json_path: str) -> None:
        """Ingest historical_players.json (with salary augmentation) into DB.

        Uses the existing loading/augmentation code from routes.historical,
        then writes the fully-augmented data to the HistoricalPlayer table.
        """
        import json as _json
        from app.routes.historical import (
            _load_salary_supplement,
            _load_spotrac_salaries,
            _load_lahman_salaries,
        )

        p = Path(json_path)
        if not p.exists():
            logger.warning(f"Historical players file not found: {p}")
            return

        logger.info(f"Reading historical JSON from {p} ...")
        with open(p, "r") as f:
            data = _json.load(f)

        players = data.get("players", {})
        mlbam_to_idfg = data.get("mlbam_to_idfg", {})
        logger.info(f"  {len(players)} players, {len(mlbam_to_idfg)} MLBAM mappings")

        # ── Salary augmentation (same three-source merge as before) ───────
        by_year_lookup = _load_salary_supplement()
        spotrac_lookup = _load_spotrac_salaries()
        lahman_lookup = _load_lahman_salaries()

        for key, sal in spotrac_lookup.items():
            if key not in by_year_lookup:
                by_year_lookup[key] = sal

        filled = 0
        for _idfg_str, pl in players.items():
            name_lower = pl["name"].lower().strip()
            bbref_id = pl.get("bbref", "")
            career_salary_add = 0

            for season_key in ("batting", "pitching"):
                for s in pl.get(season_key, []):
                    if s.get("salary"):
                        continue
                    team = s.get("team", "")
                    yr = s.get("year", 0)
                    sal = by_year_lookup.get((name_lower, team, yr))
                    if sal is None and bbref_id:
                        sal = lahman_lookup.get((bbref_id, yr))
                    if sal:
                        s["salary"] = sal
                        war_val = s.get("war_value") or 0
                        s["surplus"] = war_val - sal
                        career_salary_add += sal
                        filled += 1

            if career_salary_add > 0:
                old = pl.get("career_salary") or 0
                pl["career_salary"] = old + career_salary_add
                old_surplus = pl.get("career_surplus")
                if old_surplus is not None:
                    pl["career_surplus"] = old_surplus - career_salary_add
                else:
                    war_value = pl.get("career_war_value") or 0
                    pl["career_surplus"] = war_value - pl["career_salary"]

        logger.info(f"  Salary augmentation: filled {filled} season entries")

        # ── Bulk write to DB ──────────────────────────────────────────────
        objects = []
        for idfg_str, pl in players.items():
            idfg = int(idfg_str)
            objects.append(HistoricalPlayer(
                idfg=idfg,
                mlbam=pl.get("mlbam"),
                bbref=pl.get("bbref"),
                name=pl["name"],
                name_lower=pl["name"].lower().strip(),
                birth_year=pl.get("birth_year"),
                death_year=pl.get("death_year"),
                first_year=pl.get("first_year"),
                last_year=pl.get("last_year"),
                teams=pl.get("teams", []),
                career_war=pl.get("career_war", 0),
                career_bat_war=pl.get("career_bat_war", 0),
                career_pit_war=pl.get("career_pit_war", 0),
                career_salary=pl.get("career_salary", 0),
                career_war_value=pl.get("career_war_value", 0),
                career_surplus=pl.get("career_surplus", 0),
                is_pitcher=pl.get("is_pitcher", False),
                batting=pl.get("batting", []),
                pitching=pl.get("pitching", []),
            ))

        self.db.bulk_save_objects(objects)

        # Also store the MLBAM→IDfg crosswalk in a lightweight way:
        # we rely on the mlbam column being indexed, so lookups work.
        self.db.commit()
        logger.info(f"  Loaded {len(objects)} historical players into DB")

    # ── MiLB statistics ─────────────────────────────────────────────────

    def load_milb_stats(self, hitters_csv: str, pitchers_csv: str) -> None:
        """Bulk-insert MiLB hitting and pitching stats from FanGraphs CSVs.

        Each row is one player-season-team-level combination.  The
        ``PlayerId`` column is the FanGraphs ID (same as ``IDfg`` in the
        Prospect table).
        """
        import csv as _csv

        def _safe_float(v):
            if v is None or v == "":
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        def _safe_int(v):
            if v is None or v == "":
                return None
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return None

        def _get_player_id(row):
            """Extract FanGraphs player ID, trying multiple column names."""
            # Try exact name first, then strip whitespace from keys
            for key in ("PlayerId", "playerid", "playerID", "PLAYERID"):
                val = row.get(key)
                if val is not None:
                    return _safe_int(val)
            # Fallback: try stripped/lowered key matching
            for key, val in row.items():
                if key.strip().lower() == "playerid":
                    return _safe_int(val)
            return None

        # ── Hitters ───────────────────────────────────────────────────────
        h_path = Path(hitters_csv)
        if h_path.exists():
            objects = []
            skipped = 0
            with open(h_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    idfg = _get_player_id(row)
                    if idfg is None:
                        skipped += 1
                        continue
                    objects.append(MiLBHittingStats(
                        IDfg=idfg,
                        season=_safe_int(row.get("Season")),
                        name=row.get("Name", ""),
                        team=row.get("Team"),
                        level=row.get("Level"),
                        age=_safe_int(row.get("Age")),
                        pa=_safe_int(row.get("PA")),
                        bb_pct=_safe_float(row.get("BB%")),
                        k_pct=_safe_float(row.get("K%")),
                        bb_k=_safe_float(row.get("BB/K")),
                        avg=_safe_float(row.get("AVG")),
                        obp=_safe_float(row.get("OBP")),
                        slg=_safe_float(row.get("SLG")),
                        ops=_safe_float(row.get("OPS")),
                        iso=_safe_float(row.get("ISO")),
                        spd=_safe_float(row.get("Spd")),
                        babip=_safe_float(row.get("BABIP")),
                        wsb=_safe_float(row.get("wSB")),
                        wrc=_safe_float(row.get("wRC")),
                        wraa=_safe_float(row.get("wRAA")),
                        woba=_safe_float(row.get("wOBA")),
                        wrc_plus=_safe_float(row.get("wRC+")),
                    ))
            self.db.bulk_save_objects(objects)
            self.db.commit()
            logger.info(f"  Loaded {len(objects)} MiLB hitting stat rows (skipped {skipped} with no PlayerId)")
        else:
            logger.warning(f"MiLB hitters CSV not found: {h_path}")

        # ── Pitchers ──────────────────────────────────────────────────────
        p_path = Path(pitchers_csv)
        if p_path.exists():
            objects = []
            skipped = 0
            with open(p_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    idfg = _get_player_id(row)
                    if idfg is None:
                        skipped += 1
                        continue
                    objects.append(MiLBPitchingStats(
                        IDfg=idfg,
                        season=_safe_int(row.get("Season")),
                        name=row.get("Name", ""),
                        team=row.get("Team"),
                        level=row.get("Level"),
                        age=_safe_int(row.get("Age")),
                        ip=_safe_float(row.get("IP")),
                        k_9=_safe_float(row.get("K/9")),
                        bb_9=_safe_float(row.get("BB/9")),
                        k_bb=_safe_float(row.get("K/BB")),
                        hr_9=_safe_float(row.get("HR/9")),
                        k_pct=_safe_float(row.get("K%")),
                        bb_pct=_safe_float(row.get("BB%")),
                        k_bb_pct=_safe_float(row.get("K-BB%")),
                        avg=_safe_float(row.get("AVG")),
                        whip=_safe_float(row.get("WHIP")),
                        babip=_safe_float(row.get("BABIP")),
                        lob_pct=_safe_float(row.get("LOB%")),
                        era=_safe_float(row.get("ERA")),
                        fip=_safe_float(row.get("FIP")),
                        e_f=_safe_float(row.get("E-F")),
                        xfip=_safe_float(row.get("xFIP")),
                    ))
            self.db.bulk_save_objects(objects)
            self.db.commit()
            logger.info(f"  Loaded {len(objects)} MiLB pitching stat rows (skipped {skipped} with no PlayerId)")
        else:
            logger.warning(f"MiLB pitchers CSV not found: {p_path}")

    # ── Past trades ───────────────────────────────────────────────────────

    def load_past_trades_data(self, json_path: str) -> None:
        """Ingest trades.json with full augmentation into DB.

        Temporarily loads historical data + surplus projections + prospect DB
        to perform the same three-pass augmentation as the old startup code,
        then writes the augmented results to the PastTrade table.

        Must be called *after* load_historical_data so the historical table
        is populated (used by the WAR augmentation pass).
        """
        import json as _json
        from app.routes.historical import (
            _load_historical, _players as _hist_players,
            _mlbam_to_idfg as _hist_mlbam,
        )

        p = Path(json_path)
        if not p.exists():
            logger.warning(f"Past trades file not found: {p}")
            return

        logger.info(f"Reading trades JSON from {p} ...")
        with open(p, "r") as f:
            trades = _json.load(f)
        logger.info(f"  {len(trades)} raw trades")

        # We need historical data in memory for the WAR augmentation.
        # Load it (reads from JSON; the DB is already populated but the
        # in-memory format is needed by the augmentation helpers).
        _load_historical()

        # Run the same three-pass augmentation pipeline
        from app.routes.trades import (
            _augment_with_projections,
            _augment_with_historical_war,
            _augment_with_prospect_values,
            _link_prospect_ids,
            _add_has_data_flags,
            _compute_confidence_and_featured,
        )
        _augment_with_projections(trades)
        _augment_with_historical_war(trades)
        _augment_with_prospect_values(trades)
        _link_prospect_ids(trades)
        _add_has_data_flags(trades)
        _compute_confidence_and_featured(trades)

        # ── Bulk write to DB ──────────────────────────────────────────────
        objects = []
        for t in trades:
            # Build denormalised filter columns
            all_teams = set()
            all_names = []
            all_mlb_ids = []
            for side in t.get("sides", []):
                all_teams.add(side["team"])
                for pl in side.get("players_received", []):
                    all_names.append(pl.get("name", "").lower())
                    if pl.get("mlb_id"):
                        all_mlb_ids.append(str(pl["mlb_id"]))

            objects.append(PastTrade(
                trade_id=t["trade_id"],
                date=t["date"],
                year=t["year"],
                description=t.get("description"),
                has_cash=t.get("has_cash", False),
                has_ptbnl=t.get("has_ptbnl", False),
                n_teams=t.get("n_teams", 2),
                n_players=t.get("n_players", 0),
                winner=t.get("winner"),
                winner_name=t.get("winner_name"),
                loser=t.get("loser"),
                loser_name=t.get("loser_name"),
                surplus_diff=t.get("surplus_diff", 0),
                total_trade_war=t.get("total_trade_war", 0),
                max_prospect_fv=t.get("max_prospect_fv"),
                evaluation_type=t.get("evaluation_type", "actual"),
                evaluation_confidence=t.get("evaluation_confidence", "definitive"),
                is_featured=t.get("is_featured", False),
                projected_winner=t.get("projected_winner"),
                projected_winner_name=t.get("projected_winner_name"),
                projected_loser=t.get("projected_loser"),
                projected_loser_name=t.get("projected_loser_name"),
                projected_surplus_diff=t.get("projected_surplus_diff"),
                projected_total_war=t.get("projected_total_war"),
                sides_json=t.get("sides", []),
                teams_csv=",".join(sorted(all_teams)),
                player_names_lower=",".join(all_names),
                player_mlb_ids_csv=",".join(all_mlb_ids),
            ))

        self.db.bulk_save_objects(objects)
        self.db.commit()
        logger.info(f"  Loaded {len(objects)} past trades into DB")

    def reset_and_load_data(self, players_csv: str, prospects_csv: str = None):
        try:
            # Clear existing data - different syntax for SQLite vs PostgreSQL
            logger.info("Clearing existing data...")
            
            # Check if we're using SQLite or PostgreSQL
            db_url = os.getenv("DATABASE_URL", "")
            if db_url.startswith("sqlite"):
                # SQLite syntax
                self.db.execute(text("DELETE FROM players;"))
                self.db.execute(text("DELETE FROM prospects;"))
            else:
                # PostgreSQL syntax
                self.db.execute(text("TRUNCATE TABLE players CASCADE;"))
                self.db.execute(text("TRUNCATE TABLE prospects CASCADE;"))
                
            self.db.commit()
            logger.info("Tables cleared successfully")

            # Load fresh data
            logger.info("Loading fresh data...")
            self.load_player_data(players_csv)
            if prospects_csv:
                self.load_prospect_data(prospects_csv)
            logger.info("Data reload complete")

        except Exception as e:
            logger.error(f"Error during data reset and load: {e}")
            self.db.rollback()
            raise

def init_db():
    """Initialize database with all data: players, prospects, historical, trades."""
    try:
        logger.info("Starting database initialization...")
        
        # Drop and recreate tables so schema changes (e.g. new columns) are picked up.
        # create_all() alone won't add columns to existing tables.
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Get the base path for data files
        base_path = Path(__file__).resolve().parent.parent.parent.parent.parent
        
        # Define paths to data files
        player_data = base_path / "data" / "generated" / "value_by_year" / "player_values_complete.csv"
        prospects_data = base_path / "data" / "generated" / "MiLB" / "prospect_histories.csv"
        historical_data = base_path / "data" / "generated" / "historical_players" / "historical_players.json"
        trades_data = base_path / "data" / "generated" / "past_trades" / "trades.json"
        milb_hitters_data = base_path / "data" / "MiLB" / "MiLB_Hitters.csv"
        milb_pitchers_data = base_path / "data" / "MiLB" / "MiLB_Pitchers.csv"
        
        logger.info(f"Looking for data files in: {base_path}")
        
        if not player_data.exists() or not prospects_data.exists():
            logger.error(f"Data files not found! Checked path: {base_path}")
            return
        
        db = SessionLocal()
        try:
            loader = DataLoader(db)
            loader.reset_and_load_data(str(player_data), str(prospects_data))

            # Historical players (salary-augmented from JSON + CSVs)
            if historical_data.exists():
                loader.load_historical_data(str(historical_data))
            else:
                logger.warning(f"Historical data not found: {historical_data}")

            # Past trades (augmented with projections, historical WAR, prospect values)
            if trades_data.exists():
                loader.load_past_trades_data(str(trades_data))
            else:
                logger.warning(f"Trades data not found: {trades_data}")

            # MiLB performance statistics (batting + pitching)
            if milb_hitters_data.exists() or milb_pitchers_data.exists():
                loader.load_milb_stats(str(milb_hitters_data), str(milb_pitchers_data))
            else:
                logger.warning(f"MiLB stats CSVs not found: {milb_hitters_data}")

            logger.info("Data loading completed successfully!")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

if __name__ == "__main__":
    init_db()