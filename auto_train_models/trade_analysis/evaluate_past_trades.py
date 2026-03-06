"""
Evaluate Past Trades — Actual WAR & Surplus Analysis
=====================================================

Pre-computes trade outcomes using actual historical WAR and salary data.

For each trade (2014-2025):
  • Groups players by side (sending team → receiving team)
  • Tracks each player's actual WAR with the acquiring team
  • Estimates salary paid during that tenure
  • Computes surplus value (WAR value − salary)
  • Determines winner/loser by net surplus difference

Output:  data/generated/past_trades/trades.json

Usage:
    python -m auto_train_models.trade_analysis.evaluate_past_trades
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]  # LSTMLB/
DATA_DIR = ROOT_DIR / "data"

TRADES_FILE = DATA_DIR / "transactions" / "trades.csv"
TRADE_PLAYERS_FILE = DATA_DIR / "transactions" / "trade_players.csv"
MATCHED_FILE = DATA_DIR / "generated" / "trade_analysis" / "results" / "matched_trade_players.csv"
BATTING_FILE = DATA_DIR / "historic_mlb" / "mlb_batting_data_1950_2025.csv"
PITCHING_FILE = DATA_DIR / "historic_mlb" / "mlb_pitching_data_1950_2025.csv"
PROSPECT_FILE = DATA_DIR / "prospect_data" / "prospects_2014_2026_with_top100.csv"
DB_FILE = ROOT_DIR / "web-app" / "backend" / "longball_local.db"
OUTPUT_DIR = DATA_DIR / "generated" / "past_trades"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("evaluate_past_trades")

# ── $/WAR by year (from Config) ─────────────────────────────────────────────
DOLLAR_PER_WAR = {
    2014: 7_600_000, 2015: 8_000_000, 2016: 8_000_000, 2017: 7_900_000,
    2018: 8_000_000, 2019: 8_100_000, 2020: 7_900_000, 2021: 8_100_000,
    2022: 8_200_000, 2023: 8_100_000, 2024: 8_200_000, 2025: 8_500_000,
}

# MLB season approximate boundaries
SEASON_START_DAY = 91   # ~April 1 (day of year)
SEASON_END_DAY = 274    # ~October 1 (day of year)
SEASON_DAYS = SEASON_END_DAY - SEASON_START_DAY  # 183

# ── MLB team name → FanGraphs abbreviation ───────────────────────────────────
TEAM_NAME_TO_FG = {
    "Arizona Diamondbacks": "ARI", "Athletics": "OAK", "Oakland Athletics": "OAK",
    "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA",
    "Los Angeles Angels of Anaheim": "LAA", "Anaheim Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Florida Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}

# Alternative abbreviations in FanGraphs data
FG_ABBREV_ALIASES = {
    "KCR": ["KCR", "KC", "KCA"],
    "TBR": ["TBR", "TB", "TBA"],
    "SDP": ["SDP", "SD"],
    "SFG": ["SFG", "SF"],
    "WSN": ["WSN", "WAS", "WSH"],
    "CHW": ["CHW", "CWS"],
    "LAA": ["LAA", "ANA", "ANH"],
}

# Build reverse lookup: any abbreviation variant → canonical
_ABBREV_CANONICAL = {}
for canon, aliases in FG_ABBREV_ALIASES.items():
    for alias in aliases:
        _ABBREV_CANONICAL[alias] = canon
# All other abbreviations map to themselves


def canonical_team(abbrev: str) -> str:
    """Normalize FanGraphs team abbreviation to canonical form."""
    return _ABBREV_CANONICAL.get(abbrev, abbrev)


# ── Crosswalk: mlbam_id → IDfg ──────────────────────────────────────────────

def build_crosswalk() -> Dict[int, int]:
    """Build mlbam_id → IDfg mapping from multiple sources."""
    xwalk = {}

    # Source 1: Database (most reliable)
    if DB_FILE.exists():
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(f"sqlite:///{DB_FILE}")
            with engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT mlb_id, real_id FROM players "
                         "WHERE mlb_id IS NOT NULL AND real_id IS NOT NULL "
                         "GROUP BY mlb_id")
                ).fetchall()
            for mlb_id, real_id in rows:
                xwalk[int(mlb_id)] = int(real_id)
            logger.info(f"  Crosswalk source 1 (database): {len(rows)} entries")
        except Exception as e:
            logger.warning(f"  Could not read database: {e}")

    # Source 2: matched_trade_players.csv
    if MATCHED_FILE.exists():
        mtp = pd.read_csv(MATCHED_FILE, usecols=["mlbam_id", "IDfg"])
        mtp = mtp.dropna(subset=["IDfg"]).drop_duplicates("mlbam_id")
        added = 0
        for _, row in mtp.iterrows():
            mid = int(row.mlbam_id)
            if mid not in xwalk:
                xwalk[mid] = int(row.IDfg)
                added += 1
        logger.info(f"  Crosswalk source 2 (matched_trade_players): +{added} entries")

    logger.info(f"  Total crosswalk: {len(xwalk)} mlbam_id → IDfg mappings")
    return xwalk


# ── Name-based fallback crosswalk ────────────────────────────────────────────

def build_name_crosswalk(bat_df: pd.DataFrame, pit_df: pd.DataFrame) -> Dict[str, int]:
    """Build name → IDfg mapping for players in WAR data (fallback)."""
    name_map = {}
    for df in [bat_df, pit_df]:
        for _, row in df[["IDfg", "Name"]].drop_duplicates("Name").iterrows():
            key = row.Name.strip().lower()
            if key not in name_map:
                name_map[key] = int(row.IDfg)
    return name_map


# ── Load WAR data ───────────────────────────────────────────────────────────

def load_war_data() -> pd.DataFrame:
    """Load batting + pitching WAR, combine for two-way players."""
    logger.info("Loading WAR data...")

    bat = pd.read_csv(BATTING_FILE, usecols=["IDfg", "Season", "Name", "Team", "WAR", "G", "PA"])
    bat["source"] = "bat"
    bat = bat.rename(columns={"Season": "year"})

    pit = pd.read_csv(PITCHING_FILE, usecols=["IDfg", "Season", "Name", "Team", "WAR", "G", "IP"])
    pit["source"] = "pit"
    pit = pit.rename(columns={"Season": "year"})
    pit["PA"] = 0  # pitchers don't have PA

    # Combine and sum WAR for two-way players in same year+team
    combined = pd.concat([
        bat[["IDfg", "year", "Name", "Team", "WAR", "G"]],
        pit[["IDfg", "year", "Name", "Team", "WAR", "G"]],
    ], ignore_index=True)

    # Group by (IDfg, year, Team) and sum WAR
    war_data = (
        combined
        .groupby(["IDfg", "year", "Team"], as_index=False)
        .agg({"WAR": "sum", "G": "max", "Name": "first"})
    )

    # Normalize team abbreviations
    war_data["Team"] = war_data["Team"].apply(canonical_team)

    logger.info(f"  Loaded {len(war_data)} WAR rows ({bat.shape[0]} batting + {pit.shape[0]} pitching)")
    return war_data


# ── Load salary data ─────────────────────────────────────────────────────────

UNIVERSAL_SALARY_FILE = DATA_DIR / "salary" / "universal_salary.csv"


def load_salary_data() -> Dict[Tuple[str, str, int], float]:
    """Build (name_lower, team_canonical, year) → annual_salary lookup.

    Reads from the single canonical ``universal_salary.csv`` which already
    merges Lahman + Spotrac + Cot's with proper dedup/priority.
    """
    salary_map = {}

    if not UNIVERSAL_SALARY_FILE.exists():
        logger.warning(f"  Universal salary file not found: {UNIVERSAL_SALARY_FILE}")
        return salary_map

    try:
        df = pd.read_csv(UNIVERSAL_SALARY_FILE)
        for _, row in df.iterrows():
            name = str(row.get("player", "")).strip()
            team = str(row.get("team", "")).strip()
            sal = row.get("salary")
            yr = row.get("year")
            if not name or not team or pd.isna(sal) or pd.isna(yr):
                continue
            try:
                sal_f = float(sal)
            except (TypeError, ValueError):
                continue
            if sal_f <= 0:
                continue
            key = (name.lower().strip(), canonical_team(team), int(yr))
            salary_map[key] = sal_f
    except Exception as e:
        logger.warning(f"  Could not load universal salary: {e}")

    logger.info(f"  Loaded {len(salary_map)} salary entries from universal_salary.csv")
    return salary_map


# ── Load prospect data ───────────────────────────────────────────────────────

def load_prospect_data() -> Dict[Tuple[str, int], dict]:
    """Build (name_lower, year) → {fv, rank, level, top_100} lookup."""
    if not PROSPECT_FILE.exists():
        logger.warning("  No prospect file found")
        return {}

    df = pd.read_csv(PROSPECT_FILE)
    prospect_map = {}
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip().lower()
        year = int(row.get("year", 0))
        fv = row.get("grade_overall")
        rank = row.get("rank")
        level = row.get("level")
        top_100_raw = row.get("top_100")
        # top_100 column contains the national rank (int) or NaN
        # bool(NaN) is True in Python, so we must use pd.notna check
        is_top_100 = bool(pd.notna(top_100_raw) and float(top_100_raw) > 0)
        top_100_rank = int(top_100_raw) if pd.notna(top_100_raw) and float(top_100_raw) > 0 else None
        if name and year:
            prospect_map[(name, year)] = {
                "fv": int(fv) if pd.notna(fv) else None,
                "rank": int(rank) if pd.notna(rank) else None,
                "level": str(level) if pd.notna(level) else None,
                "top_100": is_top_100,
                "top_100_rank": top_100_rank,
            }

    logger.info(f"  Loaded {len(prospect_map)} prospect entries")
    return prospect_map


# ── Load trades with mlbam_id → IDfg linkage ────────────────────────────────

def load_trades_and_players(
    xwalk: Dict[int, int],
    name_xwalk: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load trades and trade_players, enrich with IDfg."""
    trades = pd.read_csv(TRADES_FILE)
    players = pd.read_csv(TRADE_PLAYERS_FILE)

    # Add IDfg from crosswalk
    players["IDfg"] = players["mlbam_id"].map(xwalk)

    # Name-based fallback for unmatched
    unmatched_mask = players["IDfg"].isna()
    for idx in players[unmatched_mask].index:
        name = players.loc[idx, "name"].strip().lower()
        if name in name_xwalk:
            players.loc[idx, "IDfg"] = name_xwalk[name]

    matched = players["IDfg"].notna().sum()
    logger.info(f"  Trade players with IDfg: {matched}/{len(players)} ({matched/len(players)*100:.1f}%)")

    # Convert team names to FanGraphs abbreviations
    players["to_team_fg"] = players["to_team_name"].map(TEAM_NAME_TO_FG).apply(
        lambda x: canonical_team(x) if pd.notna(x) else x
    )
    players["from_team_fg"] = players["from_team_name"].map(TEAM_NAME_TO_FG).apply(
        lambda x: canonical_team(x) if pd.notna(x) else x
    )

    return trades, players


# ── Compute per-player WAR with acquiring team ──────────────────────────────

def compute_player_war_tenure(
    idfg: Optional[int],
    to_team: str,
    trade_date: str,
    war_data: pd.DataFrame,
    all_trades_away: pd.DataFrame,  # trades where this player was traded AWAY from to_team
) -> dict:
    """
    For a single traded player, compute their actual WAR with the acquiring team.

    Returns dict with:
      - yearly_war: [{year, war, team}]
      - total_war: float
      - seasons_with_team: int
      - still_on_team: bool
      - departure_year: int or None
    """
    result = {
        "yearly_war": [],
        "total_war": 0.0,
        "seasons_with_team": 0,
        "still_on_team": False,
        "departure_year": None,
    }

    if idfg is None or pd.isna(idfg):
        return result

    idfg = int(idfg)
    trade_dt = datetime.strptime(trade_date[:10], "%Y-%m-%d")
    trade_year = trade_dt.year
    trade_doy = trade_dt.timetuple().tm_yday  # day of year

    # Get this player's WAR data
    player_war = war_data[war_data["IDfg"] == idfg].copy()
    if player_war.empty:
        return result

    # Determine fraction of trade-year season remaining
    if trade_doy < SEASON_START_DAY:
        # Off-season trade (before season starts) → full year
        frac_remaining = 1.0
    elif trade_doy > SEASON_END_DAY:
        # Post-season → next year
        frac_remaining = 0.0
    else:
        frac_remaining = (SEASON_END_DAY - trade_doy) / SEASON_DAYS
        frac_remaining = max(0.0, min(1.0, frac_remaining))

    # Check if there are any trades moving this player away from to_team
    trades_away_years = set()
    trades_away_dates = {}
    if not all_trades_away.empty:
        for _, tr in all_trades_away.iterrows():
            yr = datetime.strptime(str(tr["date"])[:10], "%Y-%m-%d").year
            trades_away_years.add(yr)
            trades_away_dates[yr] = str(tr["date"])[:10]

    # Scan years from trade_year onwards
    max_year = int(player_war["year"].max())
    latest_data_year = 2025  # our data goes through 2025

    for yr in range(trade_year, max_year + 1):
        yr_data = player_war[player_war["year"] == yr]

        if yr_data.empty:
            # No WAR data this year — player might be injured, minors, or retired
            # Don't break yet, they might come back
            continue

        for _, row in yr_data.iterrows():
            team = row["Team"]
            war = float(row["WAR"]) if pd.notna(row["WAR"]) else 0.0

            if team == to_team:
                # Clean: player was on destination team all year
                if yr == trade_year:
                    # Might be a full year or partial (if acquired before season)
                    effective_war = war * frac_remaining if frac_remaining < 0.95 else war
                else:
                    effective_war = war
                result["yearly_war"].append({"year": yr, "war": round(effective_war, 1)})
                result["total_war"] += effective_war
                result["seasons_with_team"] += 1

            elif team == "- - -":
                # Multi-team year — check direction
                if yr == trade_year:
                    # Player was traded mid-season TO this team
                    effective_war = war * frac_remaining
                    result["yearly_war"].append({"year": yr, "war": round(effective_war, 1)})
                    result["total_war"] += effective_war
                    result["seasons_with_team"] += 1
                elif yr in trades_away_years:
                    # Player was traded AWAY from this team mid-season
                    away_date = trades_away_dates[yr]
                    away_dt = datetime.strptime(away_date, "%Y-%m-%d")
                    away_doy = away_dt.timetuple().tm_yday
                    if away_doy < SEASON_START_DAY:
                        frac_with_team = 0.0
                    elif away_doy > SEASON_END_DAY:
                        frac_with_team = 1.0
                    else:
                        frac_with_team = (away_doy - SEASON_START_DAY) / SEASON_DAYS
                        frac_with_team = max(0.0, min(1.0, frac_with_team))
                    effective_war = war * frac_with_team
                    result["yearly_war"].append({"year": yr, "war": round(effective_war, 1)})
                    result["total_war"] += effective_war
                    result["seasons_with_team"] += 1
                    result["departure_year"] = yr
                    break  # traded away
                else:
                    # Multi-team year but not in our trade DB — could be claimed/released
                    # Assign half the WAR as approximation
                    effective_war = war * 0.5
                    result["yearly_war"].append({"year": yr, "war": round(effective_war, 1)})
                    result["total_war"] += effective_war
                    result["seasons_with_team"] += 1
                    result["departure_year"] = yr
                    break

            else:
                # Player is on a different team — they left
                result["departure_year"] = yr
                break
        else:
            # Inner loop didn't break — check if this was the last data year
            if yr == max_year and yr >= latest_data_year:
                result["still_on_team"] = True
            continue
        break  # outer break when inner breaks

    result["total_war"] = round(result["total_war"], 1)
    return result


# ── Estimate salary for a tenure ─────────────────────────────────────────────

def estimate_salary(
    name: str,
    to_team: str,
    trade_year: int,
    seasons_with_team: int,
    salary_map: Dict,
) -> Tuple[float, List[dict]]:
    """Estimate total salary paid for a player's tenure with a team."""
    total_salary = 0.0
    yearly_salary = []
    name_lower = name.strip().lower()

    for yr in range(trade_year, trade_year + max(seasons_with_team, 1)):
        # Try exact match
        key = (name_lower, to_team, yr)
        sal = salary_map.get(key)

        if sal is None:
            # Try with different team abbreviation variants
            for canon, aliases in FG_ABBREV_ALIASES.items():
                if canon == to_team:
                    for alias in aliases:
                        key2 = (name_lower, alias, yr)
                        sal = salary_map.get(key2)
                        if sal is not None:
                            break
                if sal is not None:
                    break

        if sal is None:
            # Estimate minimum salary ($700K-$760K range for 2014-2025)
            sal = 500_000 + (yr - 2014) * 25_000
            sal = max(sal, 500_000)

        total_salary += sal
        yearly_salary.append({"year": yr, "salary": int(sal)})

    return total_salary, yearly_salary


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_all_trades():
    """Main pipeline: evaluate all trades and write results."""
    logger.info("=" * 60)
    logger.info("Evaluating Past Trades")
    logger.info("=" * 60)

    # 1. Build crosswalks
    logger.info("Building ID crosswalk...")
    xwalk = build_crosswalk()

    # 2. Load WAR data
    war_data = load_war_data()
    name_xwalk = build_name_crosswalk(
        pd.read_csv(BATTING_FILE, usecols=["IDfg", "Name"]),
        pd.read_csv(PITCHING_FILE, usecols=["IDfg", "Name"]),
    )

    # 3. Load salary data
    logger.info("Loading salary data...")
    salary_map = load_salary_data()

    # 4. Load prospect data
    logger.info("Loading prospect data...")
    prospect_map = load_prospect_data()

    # 5. Load trades and players
    logger.info("Loading trades...")
    trades_df, players_df = load_trades_and_players(xwalk, name_xwalk)

    # 6. Pre-compute: for each mlbam_id, find future trades away from each team
    logger.info("Pre-computing trade-away events...")
    trades_away_index = {}  # (mlbam_id, to_team_fg) → DataFrame of trades away
    for _, row in players_df.iterrows():
        mlbam_id = int(row["mlbam_id"])
        from_team = row.get("from_team_fg")
        if pd.notna(from_team):
            key = (mlbam_id, from_team)
            if key not in trades_away_index:
                trades_away_index[key] = []
            trades_away_index[key].append(row)

    # Convert to DataFrames
    for key in trades_away_index:
        trades_away_index[key] = pd.DataFrame(trades_away_index[key])

    # 7. Evaluate each trade
    logger.info("Evaluating trades...")
    results = []
    total = len(trades_df)

    for i, (_, trade) in enumerate(trades_df.iterrows()):
        if (i + 1) % 200 == 0:
            logger.info(f"  Processing trade {i+1}/{total}...")

        trade_id = int(trade["trade_id"])
        trade_date = str(trade["date"])
        trade_year = int(trade["year"])
        description = str(trade["description"])
        has_cash = bool(trade["has_cash"])
        has_ptbnl = bool(trade["has_ptbnl"])
        n_teams = int(trade["n_teams"])

        # Get players in this trade
        trade_players = players_df[players_df["trade_id"] == trade_id]

        if trade_players.empty:
            continue

        # Group players into sides:
        # A "side" is a team that RECEIVED players.
        # side = to_team_fg → list of players they received
        sides = {}
        for _, p in trade_players.iterrows():
            to_team = p.get("to_team_fg")
            from_team = p.get("from_team_fg")
            if pd.isna(to_team) or pd.isna(from_team):
                continue

            if to_team not in sides:
                sides[to_team] = {
                    "team": to_team,
                    "team_name": p["to_team_name"],
                    "players_received": [],
                    "total_war": 0.0,
                    "total_salary": 0.0,
                    "total_war_value": 0.0,
                    "total_surplus": 0.0,
                }

            # Compute this player's WAR with acquiring team
            mlbam_id = int(p["mlbam_id"])
            idfg = p.get("IDfg")
            player_name = str(p["name"])

            # Look up trades away from the destination team
            away_key = (mlbam_id, to_team)
            trades_away = trades_away_index.get(away_key, pd.DataFrame())
            # Only future trades
            if not trades_away.empty:
                trades_away = trades_away[trades_away["date"] > trade_date]

            tenure = compute_player_war_tenure(idfg, to_team, trade_date, war_data, trades_away)

            # Salary
            total_sal, yearly_sal = estimate_salary(
                player_name, to_team, trade_year,
                tenure["seasons_with_team"], salary_map,
            )

            # WAR value computation
            war_value = 0.0
            for yw in tenure["yearly_war"]:
                dpw = DOLLAR_PER_WAR.get(yw["year"], 8_500_000)
                war_value += max(0, yw["war"]) * dpw  # only positive WAR counts for value
            war_value = round(war_value)

            surplus = war_value - total_sal

            # Prospect data at time of trade
            prospect_info = prospect_map.get((player_name.strip().lower(), trade_year))
            if prospect_info is None:
                # Try year before (prospect lists come out pre-season)
                prospect_info = prospect_map.get((player_name.strip().lower(), trade_year))

            player_result = {
                "mlb_id": mlbam_id,
                "name": player_name,
                "from_team": from_team,
                "from_team_name": p["from_team_name"],
                "to_team": to_team,
                "to_team_name": p["to_team_name"],
                "war_with_team": tenure["total_war"],
                "seasons_with_team": tenure["seasons_with_team"],
                "yearly_war": tenure["yearly_war"],
                "salary_with_team": int(total_sal),
                "war_value": war_value,
                "surplus": int(surplus),
                "still_on_team": tenure["still_on_team"],
                "departure_year": tenure["departure_year"],
                "prospect_fv": prospect_info["fv"] if prospect_info else None,
                "prospect_rank": prospect_info["rank"] if prospect_info else None,
                "prospect_top_100": prospect_info["top_100"] if prospect_info else None,
                "prospect_level": prospect_info["level"] if prospect_info else None,
            }

            sides[to_team]["players_received"].append(player_result)
            sides[to_team]["total_war"] += tenure["total_war"]
            sides[to_team]["total_salary"] += total_sal
            sides[to_team]["total_war_value"] += war_value
            sides[to_team]["total_surplus"] += surplus

        # Handle multi-team trades by pairing sides
        if len(sides) < 2:
            continue

        # Round side totals
        side_list = list(sides.values())
        for s in side_list:
            s["total_war"] = round(s["total_war"], 1)
            s["total_salary"] = int(s["total_salary"])
            s["total_war_value"] = int(s["total_war_value"])
            s["total_surplus"] = int(s["total_surplus"])

        # Determine winner (side with highest surplus)
        side_list.sort(key=lambda s: s["total_surplus"], reverse=True)
        winner_team = side_list[0]["team"]
        loser_team = side_list[-1]["team"]
        surplus_diff = side_list[0]["total_surplus"] - side_list[-1]["total_surplus"]

        # Figure out the max prospect FV in the trade
        max_prospect_fv = 0
        all_players = []
        for s in side_list:
            for p in s["players_received"]:
                all_players.append(p)
                fv = p.get("prospect_fv") or 0
                if fv > max_prospect_fv:
                    max_prospect_fv = fv

        # Total WAR in the trade
        total_trade_war = sum(s["total_war"] for s in side_list)

        trade_result = {
            "trade_id": trade_id,
            "date": trade_date,
            "year": trade_year,
            "description": description,
            "has_cash": has_cash,
            "has_ptbnl": has_ptbnl,
            "n_teams": n_teams,
            "sides": side_list,
            "winner": winner_team,
            "winner_name": sides[winner_team]["team_name"],
            "loser": loser_team,
            "loser_name": sides[loser_team]["team_name"],
            "surplus_diff": int(surplus_diff),
            "total_trade_war": round(total_trade_war, 1),
            "max_prospect_fv": max_prospect_fv if max_prospect_fv > 0 else None,
            "n_players": len(all_players),
        }

        results.append(trade_result)

    # 8. Sort by date descending
    results.sort(key=lambda t: t["date"], reverse=True)

    logger.info(f"Evaluated {len(results)} trades with 2+ sides")

    # 9. Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "trades.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Wrote {output_file}")

    # 10. Also write a summary CSV for quick inspection
    summary_rows = []
    for t in results:
        row = {
            "trade_id": t["trade_id"],
            "date": t["date"],
            "year": t["year"],
            "description": t["description"][:120],
            "n_teams": t["n_teams"],
            "n_players": t["n_players"],
            "has_cash": t["has_cash"],
            "winner": t["winner"],
            "loser": t["loser"],
            "surplus_diff": t["surplus_diff"],
            "total_war": t["total_trade_war"],
            "max_prospect_fv": t["max_prospect_fv"],
        }
        for i, side in enumerate(t["sides"]):
            row[f"side_{i}_team"] = side["team"]
            row[f"side_{i}_war"] = side["total_war"]
            row[f"side_{i}_salary"] = side["total_salary"]
            row[f"side_{i}_surplus"] = side["total_surplus"]
            row[f"side_{i}_players"] = ", ".join(p["name"] for p in side["players_received"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / "trades_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Wrote {summary_file}")

    # Print some stats
    logger.info("\n=== Summary ===")
    logger.info(f"Total trades evaluated: {len(results)}")
    total_players_matched = sum(
        1 for t in results for s in t["sides"]
        for p in s["players_received"] if p["war_with_team"] != 0
    )
    total_players = sum(t["n_players"] for t in results)
    logger.info(f"Players with WAR data: {total_players_matched}/{total_players}")

    # Top 5 biggest trade wins
    biggest = sorted(results, key=lambda t: t["surplus_diff"], reverse=True)[:5]
    logger.info("\nTop 5 biggest trade wins:")
    for t in biggest:
        logger.info(f"  {t['date']}: {t['winner']} over {t['loser']} "
                     f"(${t['surplus_diff']:,} surplus diff, {t['total_trade_war']} total WAR)")
        logger.info(f"    {t['description'][:100]}...")


if __name__ == "__main__":
    evaluate_all_trades()
