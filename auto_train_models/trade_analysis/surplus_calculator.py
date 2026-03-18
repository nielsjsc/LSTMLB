"""
Surplus Value Calculator (v2)
==============================

For each snapshot year S (cutoff = S − 1):
  1.  Load the four projection CSVs from ``projections/cutoff_{S-1}/``.
  2.  Load the Cot's salary file ``salary/by_year/{S}.csv`` which gives
      each player's *service_time*, *years_of_control*, *salary*,
      and *total_future_salary* — including signed players.
  3.  Match players between projections and Cot's by **name**.
  4.  Compute projected WAR per year for each player (batters: wRAA +
      BsR + Def; pitchers: FIP-WAR), then sum over remaining control years.
  5.  Include the player's projected stats for the snapshot year from the
      prediction CSVs.
  6.  Write one CSV per snapshot year with columns:

          Name, IDfg, Team, Age, Position, player_type,
          <projected stats for the snapshot year>,
          service_time, years_of_control, status,
          total_future_WAR, total_future_WAR_value,
          total_future_salary, surplus

Usage:
    cd auto_train_models
    python -m trade_analysis.surplus_calculator --start 2014 --end 2025
    python -m trade_analysis.surplus_calculator --year 2018 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────
_AUTO_TRAIN = Path(__file__).resolve().parents[1]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from trade_analysis.config import (
    Config, logger,
    PROJECTIONS_DIR, SURPLUS_DIR,
    COTS_BY_YEAR_DIR,
    HISTORIC_BATTING_FILE, HISTORIC_PITCHING_FILE,
    HISTORIC_BATTING_FILE_CLASSIC,
    HISTORIC_PITCHING_FILE_CLASSIC,
    ROOT_DIR, DATA_DIR,
)

# ═══════════════════════════════════════════════════════════════════════════
# ID-based salary matching helpers
# ═══════════════════════════════════════════════════════════════════════════

_idfg_team_cache: Optional[Dict[int, str]] = None


def _build_idfg_team_map() -> Dict[int, str]:
    """Build IDfg → team_abbreviation from the active roster + team_info CSVs.

    The roster CSV has ``fg_id`` (FanGraphs ID) and ``team_id``.
    The team_info CSV maps ``team_id`` → ``abbreviation``.
    Returns {IDfg: 'LAD', ...}.
    """
    global _idfg_team_cache
    if _idfg_team_cache is not None:
        return _idfg_team_cache

    roster_file = DATA_DIR / "active_roster" / "current_rosters.csv"
    team_info_file = DATA_DIR / "active_roster" / "team_info.csv"

    if not roster_file.exists() or not team_info_file.exists():
        logger.warning("Active roster / team_info not found — skipping ID-team crosswalk")
        _idfg_team_cache = {}
        return _idfg_team_cache

    teams = pd.read_csv(team_info_file, usecols=["team_id", "abbreviation"])
    tid_to_abbr = dict(zip(teams["team_id"], teams["abbreviation"]))

    roster = pd.read_csv(roster_file, usecols=["fg_id", "team_id"], low_memory=False)
    roster = roster.dropna(subset=["fg_id"])
    roster["fg_id"] = pd.to_numeric(roster["fg_id"], errors="coerce")
    roster = roster.dropna(subset=["fg_id"])
    roster["fg_id"] = roster["fg_id"].astype(int)
    roster["team_abbr"] = roster["team_id"].map(tid_to_abbr)
    roster = roster.dropna(subset=["team_abbr"])

    _idfg_team_cache = dict(zip(roster["fg_id"], roster["team_abbr"]))
    logger.info(f"Built IDfg→team crosswalk: {len(_idfg_team_cache)} entries")
    return _idfg_team_cache


# Cot's team abbreviations → standardised codes (handle minor discrepancies)
_TEAM_ALIAS: Dict[str, str] = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CWS": "CHW",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU", "KC": "KC", "KCR": "KC",
    "LAA": "LAA", "LAD": "LAD",
    "MIA": "MIA", "FLA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "OAK": "ATH", "ATH": "ATH",
    "PHI": "PHI", "PIT": "PIT",
    "SD": "SD", "SDP": "SD",
    "SF": "SF", "SFG": "SF",
    "SEA": "SEA", "STL": "STL",
    "TB": "TB", "TBR": "TB",
    "TEX": "TEX", "TOR": "TOR",
    "WSH": "WSH", "WSN": "WSH", "WAS": "WSH",
    "FA": "FA",
}


def _normalise_team(t: str) -> str:
    return _TEAM_ALIAS.get(t.upper().strip(), t.upper().strip())


def _enrich_cots_with_idfg(
    cots: pd.DataFrame,
    proj_players: pd.DataFrame,
) -> pd.DataFrame:
    """Add an ``IDfg`` column to the Cot's DataFrame.

    Matching priority:
      1. Unique name_key match (one projection player per name) — most common
      2. Name + team disambiguation (for collisions like Will Smith)
      3. ``NaN`` if no confident match is possible
    """
    # name_key → list of IDfgs from projections
    name_to_idfgs: Dict[str, list] = {}
    for _, row in proj_players.iterrows():
        key = row["_name_key"]
        name_to_idfgs.setdefault(key, []).append(int(row["IDfg"]))

    idfg_team_map = _build_idfg_team_map()
    n_unique = n_team = n_ambiguous = 0

    idfg_col = []
    for _, row in cots.iterrows():
        candidates = name_to_idfgs.get(row["_name_key"], [])

        if len(candidates) == 1:
            idfg_col.append(candidates[0])
            n_unique += 1
        elif len(candidates) > 1:
            # Disambiguate by team
            cots_team = _normalise_team(str(row.get("team", "")))
            matched = None
            for cand in candidates:
                cand_team = _normalise_team(idfg_team_map.get(cand, ""))
                if cand_team and cand_team == cots_team:
                    matched = cand
                    break
            if matched is not None:
                idfg_col.append(matched)
                n_team += 1
            else:
                idfg_col.append(pd.NA)
                n_ambiguous += 1
        else:
            idfg_col.append(pd.NA)

    cots = cots.copy()
    cots["IDfg"] = pd.array(idfg_col, dtype="Int64")

    logger.info(
        f"Cot's ID enrichment: {n_unique} unique-name, {n_team} team-disambiguated, "
        f"{n_ambiguous} unresolved collisions, {cots['IDfg'].isna().sum()} total unmatched"
    )
    return cots

# ═══════════════════════════════════════════════════════════════════════════
# WAR calculation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_batter_war(
    batter_df: pd.DataFrame,
    fielding_df: pd.DataFrame,
    baserunning_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute projected WAR for every batter-year row.

    WAR = (wRAA + BsR + Def + Replacement) / RPW
    """
    from value_determination.config import Config as VDConfig

    WOBA_SCALE    = VDConfig.WAR.WOBA_SCALE
    LG_WOBA       = VDConfig.WAR.LG_WOBA
    RPW           = VDConfig.WAR.RPW
    LG_PA         = VDConfig.WAR.LG_PA
    RPA           = VDConfig.WAR.RPA
    POS_ADJ       = VDConfig.WAR.POSITIONAL_ADJUSTMENTS
    BALLPARK_FACTORS = VDConfig.WAR.BALLPARK_FACTORS

    out = batter_df.copy()

    # Default per-150-game PA when not present
    if "PA" not in out.columns:
        out["PA"] = 650.0

    # --- Park factor --------------------------------------------------------
    if "Team" in out.columns:
        out["_pf"] = out["Team"].map(
            lambda t: BALLPARK_FACTORS.get(str(t).upper().strip(), 100) / 100
        )
    else:
        out["_pf"] = 1.0

    # --- wRAA + park adjustment (matches value_determination) ---------------
    wraa = ((out["wOBA"] - LG_WOBA) / WOBA_SCALE) * out["PA"]
    out["wRAA"] = wraa + (RPA - RPA * out["_pf"]) * out["PA"]

    # --- Baserunning --------------------------------------------------------
    if not baserunning_df.empty:
        if "sc_baserunning_runner_runs_tot_rate" in baserunning_df.columns:
            bsr_lookup = baserunning_df[["IDfg", "Year", "sc_baserunning_runner_runs_tot_rate"]].copy()
            bsr_lookup.rename(columns={"sc_baserunning_runner_runs_tot_rate": "BsR"}, inplace=True)
            bsr_lookup = bsr_lookup.drop_duplicates(subset=["IDfg", "Year"])
            out = out.merge(bsr_lookup, on=["IDfg", "Year"], how="left")
        elif "BsR_rate" in baserunning_df.columns:
            bsr_lookup = baserunning_df[["IDfg", "Year", "BsR_rate"]].copy()
            bsr_lookup.rename(columns={"BsR_rate": "BsR"}, inplace=True)
            bsr_lookup = bsr_lookup.drop_duplicates(subset=["IDfg", "Year"])
            out = out.merge(bsr_lookup, on=["IDfg", "Year"], how="left")
    if "BsR" not in out.columns:
        out["BsR"] = 0.0
    out["BsR"] = out["BsR"].fillna(0.0)

    # --- Fielding -----------------------------------------------------------
    if not fielding_df.empty:
        fld_col = None
        if "sc_total_runs/150" in fielding_df.columns:
            # Combine statcast fielding metrics for a complete run value
            fielding_df["sc_combined_runs"] = fielding_df["sc_total_runs/150"].fillna(0.0)
            if "sc_framing_runs/150" in fielding_df.columns:
                fielding_df["sc_combined_runs"] += fielding_df["sc_framing_runs/150"].fillna(0.0)
            fld_col = "sc_combined_runs"
        elif "UZR/150" in fielding_df.columns:
            fld_col = "UZR/150"
        elif "DRS/150" in fielding_df.columns:
            fld_col = "DRS/150"

    if fld_col and not fielding_df.empty:
        fld_best = (
            fielding_df
            .sort_values(fld_col, ascending=False)
            .drop_duplicates(subset=["IDfg", "Year"], keep="first")
        )
        fld_lookup = fld_best[["IDfg", "Year", fld_col]].rename(columns={fld_col: "Fld"})
        if "Pos" in fld_best.columns:
            fld_lookup = fld_lookup.copy()
            fld_lookup["_Pos"] = fld_best["Pos"].values
        elif "Position_Group" in fld_best.columns:
            fld_lookup = fld_lookup.copy()
            fld_lookup["_Pos"] = fld_best["Position_Group"].values
        else:
            fld_lookup = fld_lookup.copy()
            fld_lookup["_Pos"] = "DH"
        out = out.merge(fld_lookup, on=["IDfg", "Year"], how="left")

    if "Fld" not in out.columns:
        out["Fld"]  = 0.0
        out["_Pos"] = "DH"
    out["Fld"]  = out["Fld"].fillna(0.0)
    out["_Pos"] = out["_Pos"].fillna("DH")

    out["Pos_Adj"] = out["_Pos"].map(POS_ADJ).fillna(0.0) * (150.0 / 162.0)
    out["Def"] = out["Fld"] + out["Pos_Adj"]
    dh_adj = POS_ADJ.get("DH", -17.5) * (150.0 / 162.0)
    out["Def"] = out["Def"].clip(lower=dh_adj)

    # --- Replacement level --------------------------------------------------
    rep = 570 * RPW * out["PA"] / LG_PA

    # --- WAR ----------------------------------------------------------------
    out["WAR"] = (out["wRAA"] + out["BsR"] + out["Def"] + rep) / RPW

    out.drop(columns=["wRAA", "BsR", "Fld", "_Pos", "_pf", "Pos_Adj", "Def"], errors="ignore", inplace=True)
    return out


def _dynamic_pitcher_rpw(era: float, role: str, vd: "type") -> float:
    """Pitcher-specific RPW (FanGraphs methodology) — mirrors value_determination."""
    ip_per_game = vd.DEFAULT_IP_PER_APPEARANCE_RP if role == "RP" else vd.DEFAULT_IP_PER_START
    lg_ra9 = vd.LG_RA9
    ra9 = era if era > 0 else lg_ra9
    game_ra9 = ((18 - ip_per_game) * lg_ra9 + ip_per_game * ra9) / 18
    return (game_ra9 + 2) * 1.5


def _calculate_pitcher_war(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """FIP-based pitcher WAR — identical to value_determination pipeline."""
    from value_determination.config import Config as VDConfig

    W = VDConfig.WAR
    lg_fip = W.LG_FIP
    rep200 = W.REPLACEMENT_LEVEL_RUNS_200IP
    ballpark_factors = W.BALLPARK_FACTORS

    out = pitcher_df.copy()

    if "IP" not in out.columns:
        out["IP"] = np.where(
            out["Role"].str.upper() == "SP",
            W.DEFAULT_SP_IP,
            W.DEFAULT_RP_IP,
        )

    # Park-adjust FIP (same as value_determination/calculate_war.py)
    if "Team" in out.columns:
        pf = out["Team"].map(
            lambda t: ballpark_factors.get(str(t).upper().strip(), 100) / 100
        )
    else:
        pf = 1.0
    park_adj_fip = out["FIP"] / pf

    fip_runs = (lg_fip - park_adj_fip) / 9.0 * out["IP"]
    rep_runs = rep200 * (out["IP"] / 200.0)

    # Dynamic RPW per pitcher (FanGraphs methodology)
    era_col = out["ERA"] if "ERA" in out.columns else pd.Series(W.LG_RA9, index=out.index)
    role_col = out["Role"].str.upper() if "Role" in out.columns else pd.Series("SP", index=out.index)
    pitcher_rpw = pd.Series(
        [_dynamic_pitcher_rpw(e, r, W) for e, r in zip(era_col, role_col)],
        index=out.index,
    )

    out["WAR"] = (fip_runs + rep_runs) / pitcher_rpw
    return out


# ═══════════════════════════════════════════════════════════════════════════
# WAR → dollar conversion
# ═══════════════════════════════════════════════════════════════════════════

def _war_dollar_value(war: float, year: int) -> float:
    """Tiered WAR → dollar conversion with year-based inflation."""
    if pd.isna(war) or war <= 0:
        return 0.0

    tiers = Config.WAR_VALUE_TIERS
    remaining = war
    value = 0.0

    t1 = min(remaining, tiers["tier1"]["max"])
    value += t1 * tiers["tier1"]["value"]
    remaining -= t1
    if remaining <= 0:
        return value * _inflation(year)

    t2 = min(remaining, tiers["tier2"]["max"] - tiers["tier1"]["max"])
    value += t2 * tiers["tier2"]["value"]
    remaining -= t2
    if remaining <= 0:
        return value * _inflation(year)

    value += remaining * tiers["tier3"]["value"]
    return value * _inflation(year)


def _inflation(year: int) -> float:
    return (1 + Config.INFLATION_RATE) ** (year - Config.BASE_YEAR)


# ═══════════════════════════════════════════════════════════════════════════
# Service-time status classification
# ═══════════════════════════════════════════════════════════════════════════

def _classify_status(service_time: float) -> str:
    """Classify a player's contract status from Cot's service_time."""
    if pd.isna(service_time):
        return "Unknown"
    if service_time < 3:
        return "Pre-Arb"
    elif service_time < 4:
        return "Arb-1"
    elif service_time < 5:
        return "Arb-2"
    elif service_time < 6:
        return "Arb-3"
    else:
        return "Signed"  # 6+ service time but still under contract = signed deal


# ═══════════════════════════════════════════════════════════════════════════
# Projected stats columns (from prediction CSVs)
# ═══════════════════════════════════════════════════════════════════════════

# Batter prediction columns to include (excluding Name/IDfg/Year/Age which
# are already in the header).
BATTER_PROJ_COLS = ["BB%", "K%", "AVG", "OBP", "SLG", "wOBA", "HR", "2B", "3B", "RBI", "R", "HBP"]

# Pitcher prediction columns to include.
PITCHER_PROJ_COLS = ["Role", "K%", "BB%", "FIP", "ERA"]


# ═══════════════════════════════════════════════════════════════════════════
# Service-time estimation from historical FanGraphs data
# ═══════════════════════════════════════════════════════════════════════════

_hist_appearances_cache: Optional[pd.DataFrame] = None
_mlbam_crosswalk_cache: Optional[pd.DataFrame] = None


def _build_mlbam_crosswalk() -> pd.DataFrame:
    """Build IDfg → mlbam_id mapping from statcast-enriched historic data."""
    global _mlbam_crosswalk_cache
    if _mlbam_crosswalk_cache is not None:
        return _mlbam_crosswalk_cache

    frames = []
    for path in [HISTORIC_BATTING_FILE, HISTORIC_PITCHING_FILE]:
        if path.exists():
            df = pd.read_csv(path, usecols=["IDfg", "sc_mlbam_id"], low_memory=False)
            df = df.dropna(subset=["sc_mlbam_id"]).drop_duplicates("IDfg")
            frames.append(df)
    if frames:
        xw = pd.concat(frames, ignore_index=True).drop_duplicates("IDfg")
        xw = xw.rename(columns={"sc_mlbam_id": "mlbam_id"})
        xw["mlbam_id"] = xw["mlbam_id"].astype("Int64")
    else:
        xw = pd.DataFrame(columns=["IDfg", "mlbam_id"])
    _mlbam_crosswalk_cache = xw
    return xw


def _load_historical_appearances() -> pd.DataFrame:
    """Build (IDfg, Season) table from historical batting + pitching data."""
    global _hist_appearances_cache
    if _hist_appearances_cache is not None:
        return _hist_appearances_cache

    frames = []
    logger.info("Loading historical appearances for service-time estimation …")
    for path in [HISTORIC_BATTING_FILE_CLASSIC, HISTORIC_PITCHING_FILE_CLASSIC]:
        df = pd.read_csv(path, usecols=["IDfg", "Season", "Name"], low_memory=False)
        frames.append(df[["IDfg", "Season"]].drop_duplicates())
    _hist_appearances_cache = pd.concat(frames, ignore_index=True).drop_duplicates()
    return _hist_appearances_cache


def _estimate_service_time(idfg: int, as_of_season: int) -> float:
    """Approximate MLB service years by counting distinct seasons up to as_of_season."""
    app = _load_historical_appearances()
    n = app.loc[(app["IDfg"] == idfg) & (app["Season"] <= as_of_season), "Season"].nunique()
    return float(n)


def _years_of_control_from_svc(service_time: float) -> int:
    """Standard MLB rule: free agency at 6 years of service."""
    import math
    return max(0, math.ceil(6.0 - service_time))


def _estimate_salary_for_status(status: str, war_value: float, year: int) -> float:
    """Estimate salary for pre-arb / arb players (used when Cot's data is unavailable)."""
    if status in ("FA", "Signed", "Unknown"):
        return 0.0
    if status == "Pre-Arb":
        return Config.HISTORICAL_MIN_SALARY.get(year, 720_000)
    arb_pct = Config.ARB_PERCENT.get(status, 0.40)
    min_sal = Config.MIN_SALARY.get(status, 1_000_000)
    return max(min_sal, war_value * arb_pct)


# ═══════════════════════════════════════════════════════════════════════════
# Core surplus computation for one snapshot year
# ═══════════════════════════════════════════════════════════════════════════

def compute_surplus_for_snapshot(
    snapshot_year: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute per-player surplus value for a single snapshot year.

    For snapshot_year S, cutoff_year = S − 1.
    Players come from Cot's ``by_year/{S}.csv``.
    Projections come from ``projections/cutoff_{S-1}/``.

    Output columns:
        Name, IDfg, Team, Age, Position, player_type,
        proj_<stat> columns for the snapshot year,
        service_time, years_of_control, status,
        total_future_WAR, total_future_WAR_value,
        total_future_salary, surplus
    """
    cutoff_year = snapshot_year - Config.SNAPSHOT_LAG
    out_path = SURPLUS_DIR / f"surplus_{snapshot_year}.csv"

    if out_path.exists() and not force:
        logger.info(f"[{snapshot_year}]  surplus file already exists — skipping")
        return pd.read_csv(out_path)

    SURPLUS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Cot's salary data ────────────────────────────────────────────
    cots_file = COTS_BY_YEAR_DIR / f"{snapshot_year}.csv"
    if not cots_file.exists():
        logger.warning(f"[{snapshot_year}]  Cot's salary file missing: {cots_file}")
        return pd.DataFrame()

    cots = pd.read_csv(cots_file)
    cots = cots.dropna(subset=["player"]).copy()
    # Normalize the Cot's name for matching — strip non-alpha chars (e.g. asterisks)
    cots["_name_key"] = cots["player"].str.replace(r"[^a-zA-Z\s]", "", regex=True).str.strip().str.lower()
    # Pre-sort by salary desc (dedup happens after ID enrichment below)
    cots = cots.sort_values("salary", ascending=False, na_position="last")

    logger.info(f"[{snapshot_year}]  loaded {len(cots)} Cot's rows")

    # ── Load projections ──────────────────────────────────────────────────
    proj_dir = PROJECTIONS_DIR / f"cutoff_{cutoff_year}"
    if not proj_dir.exists():
        logger.warning(f"[{snapshot_year}]  projections dir missing: {proj_dir}")
        return pd.DataFrame()

    batter_file      = proj_dir / "batter_predictions.csv"
    pitcher_file     = proj_dir / "pitcher_predictions.csv"
    fielding_file    = proj_dir / "fielding_predictions.csv"
    baserunning_file = proj_dir / "baserunning_predictions.csv"

    missing = [f for f in [batter_file, pitcher_file, fielding_file, baserunning_file] if not f.exists()]
    if missing:
        logger.warning(f"[{snapshot_year}]  missing prediction files: {[f.name for f in missing]}")
        return pd.DataFrame()

    batter_df      = pd.read_csv(batter_file)
    pitcher_df     = pd.read_csv(pitcher_file)
    fielding_df    = pd.read_csv(fielding_file)
    baserunning_df = pd.read_csv(baserunning_file)

    logger.info(
        f"[{snapshot_year}]  loaded projections — "
        f"{batter_df['IDfg'].nunique()} batters, {pitcher_df['IDfg'].nunique()} pitchers"
    )

    # ── Compute WAR for projections ───────────────────────────────────────
    batter_df  = _calculate_batter_war(batter_df, fielding_df, baserunning_df)
    pitcher_df = _calculate_pitcher_war(pitcher_df)

    # Build a unified WAR lookup: (IDfg, Name, Year) → WAR + projected stats
    batter_keep = ["IDfg", "Name", "Year", "Age", "WAR"] + [c for c in BATTER_PROJ_COLS if c in batter_df.columns]
    batter_slim = batter_df[batter_keep].copy()
    batter_slim["player_type"] = "batter"

    pitcher_keep = ["IDfg", "Name", "Year", "Age", "WAR"] + [c for c in PITCHER_PROJ_COLS if c in pitcher_df.columns]
    pitcher_slim = pitcher_df[pitcher_keep].copy()
    pitcher_slim["player_type"] = "pitcher"

    all_proj = pd.concat([batter_slim, pitcher_slim], ignore_index=True)

    # ── Handle two-way players: sum WAR instead of taking max ─────────────
    # Identify players who appear in both batter & pitcher projections
    dup_mask = all_proj.duplicated(subset=["IDfg", "Year"], keep=False)
    two_way_ids = set(all_proj.loc[dup_mask, "IDfg"].unique())

    if two_way_ids:
        logger.info(f"[{snapshot_year}]  Found {len(two_way_ids)} two-way player(s): "
                    f"{list(two_way_ids)[:5]}")
        # For two-way players: sum WAR across batter+pitcher, keep batter row as base
        two_way_rows = all_proj[all_proj["IDfg"].isin(two_way_ids)].copy()
        non_two_way = all_proj[~all_proj["IDfg"].isin(two_way_ids)].copy()

        summed_rows = []
        for (idfg, yr), grp in two_way_rows.groupby(["IDfg", "Year"]):
            combined_war = grp["WAR"].sum()
            # Use the batter row as the base (for Name, Age, etc.)
            base = grp.sort_values("WAR", ascending=False).iloc[0].copy()
            base["WAR"] = combined_war
            base["player_type"] = "two_way"
            summed_rows.append(base)

        two_way_combined = pd.DataFrame(summed_rows)
        all_proj = pd.concat([non_two_way, two_way_combined], ignore_index=True)
    else:
        # Standard dedup: keep highest WAR per (IDfg, Year)
        all_proj = all_proj.sort_values("WAR", ascending=False).drop_duplicates(
            subset=["IDfg", "Year"], keep="first"
        )

    # Add name key for matching with Cot's — strip non-alpha chars
    all_proj["_name_key"] = all_proj["Name"].str.replace(r"[^a-zA-Z\s]", "", regex=True).str.strip().str.lower()

    # ── Identify unique projected players ─────────────────────────────────
    # For each player: determine type and Age in the snapshot year
    proj_players = (
        all_proj[all_proj["Year"] == snapshot_year]
        .drop_duplicates(subset=["IDfg"])
        [["IDfg", "Name", "_name_key", "Age", "player_type"]]
        .copy()
    )
    # Fallback: players who have projections but not for the exact snapshot year
    proj_players_extra = (
        all_proj[~all_proj["IDfg"].isin(proj_players["IDfg"])]
        .sort_values("Year")
        .drop_duplicates(subset=["IDfg"])
        [["IDfg", "Name", "_name_key", "Age", "player_type"]]
        .copy()
    )
    proj_players = pd.concat([proj_players, proj_players_extra], ignore_index=True)

    logger.info(f"[{snapshot_year}]  {len(proj_players)} unique projected players")

    # ── ID-based Cot's matching (replaces brittle name-only join) ─────────
    cots = _enrich_cots_with_idfg(cots, proj_players)

    # Dedup: by IDfg where we resolved one, by name_key otherwise
    cots_id = cots.dropna(subset=["IDfg"]).drop_duplicates(subset=["IDfg"], keep="first")
    cots_name = cots[cots["IDfg"].isna()].drop_duplicates(subset=["_name_key"], keep="first")
    cots = pd.concat([cots_id, cots_name], ignore_index=True)

    # Merge on IDfg — Cot's rows without an IDfg also failed name resolution
    # (same normalisation), so an ID-only join is both complete and correct.
    merged = proj_players.merge(
        cots.drop(columns=["_name_key"]),
        on="IDfg", how="left", suffixes=("", "_cots"),
    )

    n_in_cots = merged["player"].notna().sum()
    n_missing = merged["player"].isna().sum()
    logger.info(
        f"[{snapshot_year}]  {n_in_cots} matched in Cot's (by IDfg), "
        f"{n_missing} not in Cot's (will estimate from service time)"
    )

    # ── Build surplus rows ─────────────────────────────────────────────────
    records = []
    for _, p in merged.iterrows():
        idfg   = p["IDfg"]
        name   = p["Name"]
        ptype  = p["player_type"]
        age    = p["Age"]

        in_cots = pd.notna(p.get("player"))

        if in_cots:
            # ── Data from Cot's ────────────────────────────────────────
            team  = p.get("team", "")
            pos   = p.get("position", "")
            svc   = p.get("service_time", np.nan)
            yoc   = p.get("years_of_control", 0)
            cots_total_future = p.get("total_future_salary", 0)
        else:
            # ── Fallback: estimate from historical appearances ─────────
            team  = ""
            pos   = ""
            svc   = _estimate_service_time(idfg, snapshot_year - 1)
            yoc   = _years_of_control_from_svc(svc)
            cots_total_future = 0  # computed below from estimated salary

        if pd.isna(yoc) or yoc <= 0:
            continue  # not under team control

        status = _classify_status(svc)

        # Sum projected WAR over control years, storing per-year values
        control_years = list(range(snapshot_year, snapshot_year + int(yoc)))
        player_proj = all_proj[all_proj["IDfg"] == idfg]

        total_war = 0.0
        total_war_value = 0.0
        per_year_war = {}      # WAR_{year} columns for β re-computation
        per_year_salary = {}   # salary_{year} columns for β re-computation
        for yr in control_years:
            war_row = player_proj.loc[player_proj["Year"] == yr, "WAR"]
            war = float(war_row.iloc[0]) if len(war_row) else 0.0
            # Floor negative WAR at 0 — team would cut the player
            war = max(0.0, war)
            total_war += war
            total_war_value += _war_dollar_value(war, yr)
            per_year_war[f"WAR_{yr}"] = round(war, 3)

        # ── Salary ────────────────────────────────────────────────────
        # Always estimate year-by-year salary for pre-arb / arb years.
        # For signed veterans, the estimate returns 0 (handled by status).
        # Then take the MAX of (Cot's total_future_salary, estimated)
        # so that players with extensions use their real contract value,
        # while pre-arb players whose Cot's total only has the current
        # year's salary get a proper multi-year estimate.
        estimated_sal = 0.0
        for i, yr in enumerate(control_years):
            future_svc = svc + i
            future_status = _classify_status(future_svc)
            if future_status in ("FA", "Signed"):
                break
            yr_war_row = player_proj.loc[player_proj["Year"] == yr, "WAR"]
            yr_war = max(0.0, float(yr_war_row.iloc[0])) if len(yr_war_row) else 0.0
            yr_war_val = _war_dollar_value(yr_war, yr)
            yr_sal = _estimate_salary_for_status(future_status, yr_war_val, yr)
            estimated_sal += yr_sal
            per_year_salary[f"salary_{yr}"] = round(yr_sal)

        if in_cots:
            cots_sal = float(cots_total_future) if pd.notna(cots_total_future) else 0.0
            total_future_sal = max(cots_sal, estimated_sal)
            # If Cot's salary wins, scale up per-year salaries proportionally
            if cots_sal > estimated_sal and estimated_sal > 0:
                scale = cots_sal / estimated_sal
                per_year_salary = {k: round(v * scale) for k, v in per_year_salary.items()}
            elif cots_sal > estimated_sal and estimated_sal == 0:
                # Signed veteran: spread Cot's salary evenly across control years
                n_yrs = len(control_years)
                for yr in control_years:
                    per_year_salary[f"salary_{yr}"] = round(cots_sal / n_yrs)
        else:
            total_future_sal = estimated_sal

        surplus = total_war_value - total_future_sal

        # Projected stats for the snapshot year from prediction CSVs
        snap_row = player_proj[player_proj["Year"] == snapshot_year]
        proj_stats = {}
        if not snap_row.empty:
            sr = snap_row.iloc[0]
            stat_cols = BATTER_PROJ_COLS if ptype == "batter" else PITCHER_PROJ_COLS
            for c in stat_cols:
                if c in sr.index and pd.notna(sr[c]):
                    proj_stats[f"proj_{c}"] = sr[c]
            proj_stats["proj_WAR"] = round(float(sr["WAR"]), 2)

        # Look up mlbam_id from crosswalk
        xw = _build_mlbam_crosswalk()
        mlbam_row = xw.loc[xw["IDfg"] == idfg, "mlbam_id"]
        mlbam_id = int(mlbam_row.iloc[0]) if len(mlbam_row) else pd.NA

        record = {
            "Name": name,
            "IDfg": idfg,
            "mlbam_id": mlbam_id,
            "snapshot_year": snapshot_year,
            "Team": team,
            "Age": age,
            "Position": pos,
            "player_type": ptype,
        }
        record.update(proj_stats)

        record.update({
            "service_time": round(svc, 3) if pd.notna(svc) else np.nan,
            "years_of_control": int(yoc),
            "status": status,
            "salary_source": "cots" if in_cots else "estimated",
            "total_future_WAR": round(total_war, 2),
            "total_future_WAR_value": round(total_war_value),
            "total_future_salary": round(total_future_sal),
            "surplus": round(surplus),
        })
        # Per-year WAR and salary columns (used by β optimisation)
        record.update(per_year_war)
        record.update(per_year_salary)

        records.append(record)

    result = pd.DataFrame(records)
    if result.empty:
        logger.warning(f"[{snapshot_year}]  no surplus records produced")
        return result

    result = result.sort_values("surplus", ascending=False).reset_index(drop=True)

    result.to_csv(out_path, index=False)
    logger.info(
        f"[{snapshot_year}]  wrote {len(result)} players to {out_path.name}  "
        f"(median surplus ${result['surplus'].median():,.0f})"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_surpluses(
    start: int | None = None,
    end: int | None = None,
    force: bool = False,
) -> dict[int, pd.DataFrame]:
    """Compute surplus for every snapshot year in [start, end]."""
    start = start or (Config.CUTOFF_START + Config.SNAPSHOT_LAG)
    end   = end   or (Config.CUTOFF_END   + Config.SNAPSHOT_LAG)

    Config.ensure_directories()

    logger.info("=" * 60)
    logger.info("Trade Analysis — Surplus Calculator (v2)")
    logger.info(f"Snapshot years {start} → {end}")
    logger.info("=" * 60)

    results: dict[int, pd.DataFrame] = {}
    for snap_year in range(start, end + 1):
        df = compute_surplus_for_snapshot(snap_year, force=force)
        results[snap_year] = df

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-player surplus values using Cot's salary data."
    )
    p.add_argument("--year", type=int, help="Single snapshot year to process.")
    p.add_argument("--start", type=int, help="First snapshot year.")
    p.add_argument("--end", type=int, help="Last snapshot year.")
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.year:
        compute_surplus_for_snapshot(args.year, force=args.force)
    else:
        compute_all_surpluses(start=args.start, end=args.end, force=args.force)
