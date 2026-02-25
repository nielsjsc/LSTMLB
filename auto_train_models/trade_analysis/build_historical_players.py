"""
Build historical player database from FanGraphs WAR data + Chadwick register + Lahman salaries.

Produces: data/generated/historical_players/historical_players.json

Usage:
    python build_historical_players.py
"""

import json
import glob
import math
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

BATTING_CSV = ROOT / "data" / "historic_mlb" / "mlb_batting_data_1950_2025.csv"
PITCHING_CSV = ROOT / "data" / "historic_mlb" / "mlb_pitching_data_1950_2025.csv"
CHADWICK_DIR = ROOT / "data" / "register" / "data"
LAHMAN_SALARY_CSV = ROOT / "data" / "salary" / "lahman_salaries.csv"
OUTPUT_DIR = ROOT / "data" / "generated" / "historical_players"

DOLLAR_PER_WAR_KNOWN = {
    2014: 7_600_000, 2015: 7_700_000, 2016: 7_800_000, 2017: 8_000_000,
    2018: 8_000_000, 2019: 8_000_000, 2020: 8_000_000, 2021: 8_000_000,
    2022: 8_000_000, 2023: 8_200_000, 2024: 8_300_000, 2025: 8_500_000,
}
INFLATION_RATE = 0.04
BASE_YEAR = 2025
BASE_DOLLAR_PER_WAR = 8_500_000


def dollar_per_war(year: int) -> float:
    if year in DOLLAR_PER_WAR_KNOWN:
        return DOLLAR_PER_WAR_KNOWN[year]
    return BASE_DOLLAR_PER_WAR / ((1 + INFLATION_RATE) ** (BASE_YEAR - year))


def _jval(val, kind="float", dec=2):
    """Convert to JSON-safe value."""
    if val is None:
        return None
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    if kind == "int":
        return int(f)
    return round(f, dec)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Crosswalk ────────────────────────────────────────────
    print("Loading Chadwick register...")
    cw_files = sorted(glob.glob(str(CHADWICK_DIR / "people-*.csv")))
    cw = pd.concat(
        [pd.read_csv(f, low_memory=False, usecols=[
            "key_fangraphs", "key_mlbam", "key_bbref",
            "birth_year", "death_year"
        ]) for f in cw_files],
        ignore_index=True,
    )
    cw["key_fangraphs"] = pd.to_numeric(cw["key_fangraphs"], errors="coerce")
    cw = cw.dropna(subset=["key_fangraphs"])
    cw["key_fangraphs"] = cw["key_fangraphs"].astype(int)
    cw["key_mlbam"] = pd.to_numeric(cw["key_mlbam"], errors="coerce")
    # Deduplicate by FG ID (keep first)
    cw = cw.drop_duplicates(subset=["key_fangraphs"], keep="first")
    cw_map = cw.set_index("key_fangraphs")
    print(f"  Crosswalk: {len(cw_map):,} entries")

    # ── Salaries ─────────────────────────────────────────────
    print("Loading Lahman salaries...")
    lahman = pd.read_csv(LAHMAN_SALARY_CSV)
    bbref_map = cw.dropna(subset=["key_bbref"])[["key_bbref", "key_fangraphs"]].drop_duplicates("key_bbref")
    b2f = dict(zip(bbref_map["key_bbref"], bbref_map["key_fangraphs"]))
    lahman["IDfg"] = lahman["playerID"].map(b2f)
    lahman = lahman.dropna(subset=["IDfg"])
    lahman["IDfg"] = lahman["IDfg"].astype(int)
    sal_df = lahman.groupby(["IDfg", "yearID"])["salary"].sum().reset_index()
    sal_lookup = {}
    for row in sal_df.itertuples():
        sal_lookup[(row.IDfg, row.yearID)] = int(row.salary)
    print(f"  Salary entries: {len(sal_lookup):,}")

    # ── FanGraphs batting ────────────────────────────────────
    print("Loading batting data...")
    bat_need = {"IDfg", "Season", "Name", "Team", "Age", "G", "PA", "AB", "H", "HR",
                "2B", "3B", "R", "RBI", "BB", "SO", "SB", "CS",
                "AVG", "OBP", "SLG", "OPS", "wOBA", "wRC+", "WAR",
                "BB%", "K%", "Off", "Def", "BsR"}
    bat = pd.read_csv(BATTING_CSV, low_memory=False,
                       usecols=lambda c: c in bat_need)
    # Rename columns with special chars so itertuples can access them
    bat = bat.rename(columns={
        "Season": "year", "WAR": "war",
        "BB%": "bb_pct", "K%": "k_pct", "wRC+": "wrc_plus",
        "2B": "doubles", "3B": "triples",
    })
    print(f"  Batting rows: {len(bat):,}")

    # ── FanGraphs pitching ───────────────────────────────────
    print("Loading pitching data...")
    pit_need = {"IDfg", "Season", "Name", "Team", "Age", "W", "L", "G", "GS", "SV",
                "IP", "ERA", "FIP", "WHIP", "SO", "BB",
                "K/9", "BB/9", "HR/9", "K%", "BB%", "WAR", "SIERA"}
    pit = pd.read_csv(PITCHING_CSV, low_memory=False,
                       usecols=lambda c: c in pit_need)
    # Rename columns with special chars so itertuples can access them
    pit = pit.rename(columns={
        "Season": "year", "WAR": "war",
        "K%": "k_pct", "BB%": "bb_pct",
        "K/9": "k_9", "BB/9": "bb_9", "HR/9": "hr_9",
    })
    print(f"  Pitching rows: {len(pit):,}")

    # ── Pre-compute name index (IDfg → Name) ────────────────
    name_from_bat = bat.drop_duplicates("IDfg").set_index("IDfg")["Name"]
    name_from_pit = pit.drop_duplicates("IDfg").set_index("IDfg")["Name"]
    name_index = name_from_bat.combine_first(name_from_pit).to_dict()

    # ── Compute $/WAR and salary for each row ────────────────
    print("Computing war_value / salary / surplus...")

    # bat
    bat["salary"] = bat.apply(lambda r: sal_lookup.get((int(r["IDfg"]), int(r["year"]))) if pd.notna(r["year"]) else None, axis=1)
    bat["dpw"] = bat["year"].apply(lambda y: dollar_per_war(int(y)) if pd.notna(y) else 0)
    bat["war_value"] = (bat["war"].fillna(0) * bat["dpw"]).round(0).astype(int)
    bat["surplus"] = bat.apply(
        lambda r: int(r["war_value"] - r["salary"]) if pd.notna(r["salary"]) else None, axis=1
    )

    # pit — don't double-count salary for two-way players
    bat_player_years = set(zip(bat["IDfg"].astype(int), bat["year"].dropna().astype(int)))
    pit["has_bat"] = pit.apply(
        lambda r: (int(r["IDfg"]), int(r["year"])) in bat_player_years if pd.notna(r["year"]) else False, axis=1
    )
    pit["salary"] = pit.apply(
        lambda r: sal_lookup.get((int(r["IDfg"]), int(r["year"]))) if pd.notna(r["year"]) and not r["has_bat"] else None,
        axis=1
    )
    pit["dpw"] = pit["year"].apply(lambda y: dollar_per_war(int(y)) if pd.notna(y) else 0)
    pit["war_value"] = (pit["war"].fillna(0) * pit["dpw"]).round(0).astype(int)
    pit["surplus"] = pit.apply(
        lambda r: int(r["war_value"] - r["salary"]) if pd.notna(r["salary"]) else None, axis=1
    )

    # ── Build player dicts ───────────────────────────────────
    print("Building player objects...")
    all_idfg = set(bat["IDfg"].unique()) | set(pit["IDfg"].unique())
    print(f"  Unique players: {len(all_idfg):,}")

    # Group into lists of dicts
    bat_grouped = {idfg: grp for idfg, grp in bat.groupby("IDfg")}
    pit_grouped = {idfg: grp for idfg, grp in pit.groupby("IDfg")}

    players = {}
    mlbam_to_idfg = {}

    for idfg in all_idfg:
        idfg_int = int(idfg)
        name = name_index.get(idfg)
        if not name:
            continue

        # Batting seasons
        bat_seasons = []
        if idfg in bat_grouped:
            bg = bat_grouped[idfg]
            for t in bg.itertuples(index=False):
                yr = _jval(t.year, "int")
                if yr is None:
                    continue
                bat_seasons.append({
                    "year": yr, "team": str(getattr(t, "Team", "") or ""),
                    "age": _jval(getattr(t, "Age", None), "int"),
                    "g": _jval(getattr(t, "G", None), "int"),
                    "pa": _jval(getattr(t, "PA", None), "int"),
                    "hr": _jval(getattr(t, "HR", None), "int"),
                    "doubles": _jval(getattr(t, "doubles", None), "int"),
                    "triples": _jval(getattr(t, "triples", None), "int"),
                    "r": _jval(getattr(t, "R", None), "int"),
                    "rbi": _jval(getattr(t, "RBI", None), "int"),
                    "sb": _jval(getattr(t, "SB", None), "int"),
                    "cs": _jval(getattr(t, "CS", None), "int"),
                    "avg": _jval(getattr(t, "AVG", None), "float", 3),
                    "obp": _jval(getattr(t, "OBP", None), "float", 3),
                    "slg": _jval(getattr(t, "SLG", None), "float", 3),
                    "ops": _jval(getattr(t, "OPS", None), "float", 3),
                    "woba": _jval(getattr(t, "wOBA", None), "float", 3),
                    "wrc_plus": _jval(getattr(t, "wrc_plus", None), "float", 0),
                    "war": _jval(t.war, "float", 1),
                    "bb_pct": _jval(getattr(t, "bb_pct", None), "float", 3),
                    "k_pct": _jval(getattr(t, "k_pct", None), "float", 3),
                    "off": _jval(getattr(t, "Off", None), "float", 1),
                    "def_value": _jval(getattr(t, "Def", None), "float", 1),
                    "bsr": _jval(getattr(t, "BsR", None), "float", 1),
                    "salary": _jval(t.salary, "int"),
                    "war_value": _jval(t.war_value, "int"),
                    "surplus": _jval(t.surplus, "int"),
                })

        # Pitching seasons
        pit_seasons = []
        if idfg in pit_grouped:
            pg = pit_grouped[idfg]
            for t in pg.itertuples(index=False):
                yr = _jval(t.year, "int")
                if yr is None:
                    continue
                pit_seasons.append({
                    "year": yr, "team": str(getattr(t, "Team", "") or ""),
                    "age": _jval(getattr(t, "Age", None), "int"),
                    "w": _jval(getattr(t, "W", None), "int"),
                    "l": _jval(getattr(t, "L", None), "int"),
                    "g": _jval(getattr(t, "G", None), "int"),
                    "gs": _jval(getattr(t, "GS", None), "int"),
                    "sv": _jval(getattr(t, "SV", None), "int"),
                    "ip": _jval(getattr(t, "IP", None), "float", 1),
                    "era": _jval(getattr(t, "ERA", None), "float", 2),
                    "fip": _jval(getattr(t, "FIP", None), "float", 2),
                    "whip": _jval(getattr(t, "WHIP", None), "float", 2),
                    "so": _jval(getattr(t, "SO", None), "int"),
                    "bb": _jval(getattr(t, "BB", None), "int"),
                    "k_9": _jval(getattr(t, "k_9", None), "float", 1),
                    "bb_9": _jval(getattr(t, "bb_9", None), "float", 1),
                    "hr_9": _jval(getattr(t, "hr_9", None), "float", 1),
                    "k_pct": _jval(getattr(t, "k_pct", None), "float", 3),
                    "bb_pct": _jval(getattr(t, "bb_pct", None), "float", 3),
                    "war": _jval(t.war, "float", 1),
                    "siera": _jval(getattr(t, "SIERA", None), "float", 2),
                    "salary": _jval(t.salary, "int"),
                    "war_value": _jval(t.war_value, "int"),
                    "surplus": _jval(t.surplus, "int"),
                })

        bat_seasons.sort(key=lambda x: x["year"])
        pit_seasons.sort(key=lambda x: x["year"])

        all_years = {s["year"] for s in bat_seasons + pit_seasons}
        if not all_years:
            continue

        teams = set()
        for s in bat_seasons + pit_seasons:
            t = s.get("team", "")
            if t and t != "- - -":
                teams.add(t)

        bat_war = sum(s["war"] or 0 for s in bat_seasons)
        pit_war = sum(s["war"] or 0 for s in pit_seasons)
        c_sal = sum(s["salary"] or 0 for s in bat_seasons + pit_seasons)
        c_wval = sum(s["war_value"] or 0 for s in bat_seasons + pit_seasons)

        # Crosswalk
        mlbam = None
        bbref = None
        birth_yr = None
        death_yr = None
        if idfg_int in cw_map.index:
            row = cw_map.loc[idfg_int]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            mlbam = _jval(row.get("key_mlbam"), "int")
            bb = row.get("key_bbref")
            bbref = str(bb) if pd.notna(bb) else None
            birth_yr = _jval(row.get("birth_year"), "int")
            death_yr = _jval(row.get("death_year"), "int")

        players[str(idfg_int)] = {
            "name": name,
            "idfg": idfg_int,
            "mlbam": mlbam,
            "bbref": bbref,
            "birth_year": birth_yr,
            "death_year": death_yr,
            "first_year": min(all_years),
            "last_year": max(all_years),
            "teams": sorted(teams),
            "career_war": round(bat_war + pit_war, 1),
            "career_bat_war": round(bat_war, 1),
            "career_pit_war": round(pit_war, 1),
            "career_salary": c_sal if c_sal > 0 else None,
            "career_war_value": round(c_wval),
            "career_surplus": round(c_wval - c_sal) if c_sal > 0 else None,
            "is_pitcher": pit_war > bat_war and len(pit_seasons) > 0,
            "batting": bat_seasons,
            "pitching": pit_seasons,
        }
        if mlbam:
            mlbam_to_idfg[str(mlbam)] = idfg_int

    print(f"Total players: {len(players):,}")
    print(f"MLBAM index: {len(mlbam_to_idfg):,}")

    # ── Save ─────────────────────────────────────────────────
    output = {
        "players": players,
        "mlbam_to_idfg": mlbam_to_idfg,
        "metadata": {
            "total_players": len(players),
            "year_range": "1950-2025",
            "salary_sources": "Lahman (1985-2016)",
            "dollar_per_war_2025": BASE_DOLLAR_PER_WAR,
            "inflation_rate": INFLATION_RATE,
        },
    }

    out_path = OUTPUT_DIR / "historical_players.json"
    print(f"Writing {out_path}...")
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print(f"Done! {out_path.stat().st_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
