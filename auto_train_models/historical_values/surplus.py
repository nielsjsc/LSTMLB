"""
Historical Values — Surplus Calculator
========================================

For each snapshot year S (cutoff = S − 1):
  1. Load the four prediction CSVs from ``projections/cutoff_{S-1}/``.
  2. Build position profiles from historical fielding data.
  3. Compute WAR (batter + pitcher) using the shared WAR engine.
  4. Load the Cot's salary file and adapt to the pipeline timeline format.
  5. Run the shared value-determination pipeline:
     extend_fa_timeline → join_predictions → calculate_contract_value →
     calculate_surplus_value → analyze_contract_options →
     calculate_trade_values → add_trade_ranking_metrics.
  6. Extract per-player summary and save.

This module uses the **exact same** value-pipeline functions as the
current-year value-determination pipeline, ensuring identical logic.
The only difference is the salary source (Cot's vs Spotrac).

Usage:
    cd auto_train_models
    python -m historical_values.surplus --start 2014 --end 2026
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

_AUTO_TRAIN = Path(__file__).resolve().parents[1]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from historical_values.config import (
    Config, logger,
    PROJECTIONS_DIR, SURPLUS_DIR,
    COTS_BY_YEAR_DIR, DATA_DIR,
    HISTORIC_BATTING_FILE, HISTORIC_PITCHING_FILE,
    HISTORIC_BATTING_FILE_CLASSIC, HISTORIC_PITCHING_FILE_CLASSIC,
    HISTORIC_FIELDING_FILE,
    ROSTER_FILE,
)
from historical_values.war import calculate_batter_war, calculate_pitcher_war
from historical_values.cots_adapter import (
    build_salary_timeline,
    _classify_status,
    _years_of_control_from_svc,
)

# ── Shared value-determination pipeline functions ────────────────────────────
from value_determination.contract_processor import extend_fa_timeline
from value_determination.value_calculator import (
    join_predictions_with_timeline,
    calculate_contract_value,
    calculate_surplus_value,
)
from value_determination.trade_value import (
    analyze_contract_options,
    calculate_trade_values,
    add_trade_ranking_metrics,
)
from core.position_profiles import (
    build_position_profiles,
    load_fielding_history,
    load_batting_for_games,
)
from core.name_utils import name_key as _name_key_fn, normalize_team as _normalise_team


# ═══════════════════════════════════════════════════════════════════════════════
# ID crosswalk helpers
# ═══════════════════════════════════════════════════════════════════════════════

_mlbam_xw_cache: Optional[pd.DataFrame] = None
_idfg_team_cache: Optional[Dict[int, str]] = None
_hist_app_cache: Optional[pd.DataFrame] = None


def _build_mlbam_crosswalk() -> pd.DataFrame:
    """Build IDfg → mlbam_id mapping from statcast-enriched historic data."""
    global _mlbam_xw_cache
    if _mlbam_xw_cache is not None:
        return _mlbam_xw_cache

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
    _mlbam_xw_cache = xw
    return xw


def _build_idfg_team_map() -> Dict[int, str]:
    """Build IDfg → team abbreviation from the active roster."""
    global _idfg_team_cache
    if _idfg_team_cache is not None:
        return _idfg_team_cache

    team_info = DATA_DIR / "active_roster" / "team_info.csv"
    if not ROSTER_FILE.exists() or not team_info.exists():
        _idfg_team_cache = {}
        return _idfg_team_cache

    teams = pd.read_csv(team_info, usecols=["team_id", "abbreviation"])
    tid_to_abbr = dict(zip(teams["team_id"], teams["abbreviation"]))

    roster = pd.read_csv(ROSTER_FILE, usecols=["fg_id", "team_id"], low_memory=False)
    roster = roster.dropna(subset=["fg_id"])
    roster["fg_id"] = pd.to_numeric(roster["fg_id"], errors="coerce").dropna().astype(int)
    roster["team_abbr"] = roster["team_id"].map(tid_to_abbr)
    roster = roster.dropna(subset=["team_abbr"])

    _idfg_team_cache = dict(zip(roster["fg_id"], roster["team_abbr"]))
    return _idfg_team_cache


# ═══════════════════════════════════════════════════════════════════════════════
# Cot's ↔ projection matching
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_cots_with_idfg(
    cots: pd.DataFrame,
    proj_players: pd.DataFrame,
) -> pd.DataFrame:
    """Add an ``IDfg`` column to the Cot's DataFrame via name + team matching."""
    name_to_idfgs: Dict[str, list] = {}
    for _, row in proj_players.iterrows():
        key = row["_name_key"]
        name_to_idfgs.setdefault(key, []).append(int(row["IDfg"]))

    idfg_team_map = _build_idfg_team_map()
    idfg_col = []
    for _, row in cots.iterrows():
        candidates = name_to_idfgs.get(row["_name_key"], [])
        if len(candidates) == 1:
            idfg_col.append(candidates[0])
        elif len(candidates) > 1:
            cots_team = _normalise_team(str(row.get("team", "")))
            matched = None
            for cand in candidates:
                if _normalise_team(idfg_team_map.get(cand, "")) == cots_team:
                    matched = cand
                    break
            idfg_col.append(matched if matched else pd.NA)
        else:
            idfg_col.append(pd.NA)

    cots = cots.copy()
    cots["IDfg"] = pd.array(idfg_col, dtype="Int64")
    return cots


# ═══════════════════════════════════════════════════════════════════════════════
# Service-time helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_historical_appearances() -> pd.DataFrame:
    global _hist_app_cache
    if _hist_app_cache is not None:
        return _hist_app_cache

    frames = []
    for path in [HISTORIC_BATTING_FILE_CLASSIC, HISTORIC_PITCHING_FILE_CLASSIC]:
        if path.exists():
            df = pd.read_csv(path, usecols=["IDfg", "Season"], low_memory=False)
            frames.append(df[["IDfg", "Season"]].drop_duplicates())
    _hist_app_cache = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
    return _hist_app_cache


def _estimate_service_time(idfg: int, as_of_season: int) -> float:
    app = _load_historical_appearances()
    if app.empty:
        return 0.0
    return float(app.loc[(app["IDfg"] == idfg) & (app["Season"] <= as_of_season), "Season"].nunique())


# ═══════════════════════════════════════════════════════════════════════════════
# Career game-count helper (for confidence adjustments)
# ═══════════════════════════════════════════════════════════════════════════════

_career_games_cache: Optional[pd.DataFrame] = None


def _build_career_games(snapshot_year: int) -> pd.DataFrame:
    """
    Compute career G_bat, G_pit, GS per player through snapshot_year - 1.

    Replicates what value_calculator.integrate_historical_stats does in the
    current pipeline so that _apply_confidence_adjustments can correctly
    calculate projection_confidence from career game counts.
    """
    frames = []

    # Batting games
    if HISTORIC_BATTING_FILE_CLASSIC.exists():
        bat = pd.read_csv(
            HISTORIC_BATTING_FILE_CLASSIC,
            usecols=["IDfg", "Season", "G"],
            low_memory=False,
        )
        bat = bat[bat["Season"] < snapshot_year]
        bat_g = bat.groupby("IDfg")["G"].sum().reset_index()
        bat_g = bat_g.rename(columns={"G": "G_bat"})
        frames.append(bat_g)

    # Pitching games + GS
    if HISTORIC_PITCHING_FILE_CLASSIC.exists():
        pit_cols = ["IDfg", "Season", "G"]
        pit_raw = pd.read_csv(HISTORIC_PITCHING_FILE_CLASSIC, low_memory=False, nrows=0)
        if "GS" in pit_raw.columns:
            pit_cols.append("GS")
        pit = pd.read_csv(
            HISTORIC_PITCHING_FILE_CLASSIC,
            usecols=pit_cols,
            low_memory=False,
        )
        pit = pit[pit["Season"] < snapshot_year]
        agg = {"G": "sum"}
        if "GS" in pit.columns:
            agg["GS"] = "sum"
        pit_g = pit.groupby("IDfg").agg(agg).reset_index()
        pit_g = pit_g.rename(columns={"G": "G_pit"})
        frames.append(pit_g)

    if not frames:
        return pd.DataFrame(columns=["IDfg", "G_bat", "G_pit", "GS"])

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="IDfg", how="outer")
    return result.fillna(0)


def _add_career_game_counts(
    pipeline_df: pd.DataFrame,
    snapshot_year: int,
) -> pd.DataFrame:
    """
    Insert synthetic historical rows carrying career game counts so that
    _apply_confidence_adjustments (called inside calculate_trade_values)
    can compute correct projection_confidence from each player's MLB
    experience.

    These rows use ``Year = snapshot_year - 1`` which falls under the
    ``Year < current_year`` filter.  They are tagged with
    ``_synthetic = True`` so they can be removed after trade-value
    calculation.
    """
    career = _build_career_games(snapshot_year)
    if career.empty:
        return pipeline_df

    player_ids = set(pipeline_df["IDfg"].unique())
    career = career[career["IDfg"].isin(player_ids)].copy()
    if career.empty:
        return pipeline_df

    career["Year"] = snapshot_year - 1
    career["_synthetic"] = True

    # Ensure all pipeline columns exist in career (as NaN) so concat works
    for col in pipeline_df.columns:
        if col not in career.columns:
            career[col] = np.nan

    # Keep all columns: pipeline cols + game-count cols + _synthetic
    all_cols = list(pipeline_df.columns) + [c for c in ["G_bat", "G_pit", "GS", "_synthetic"]
                                            if c not in pipeline_df.columns]
    # Also add game-count columns to pipeline_df (as NaN) for concat alignment
    for c in ["G_bat", "G_pit", "GS", "_synthetic"]:
        if c not in pipeline_df.columns:
            pipeline_df = pipeline_df.copy()
            pipeline_df[c] = np.nan

    result = pd.concat([pipeline_df, career[all_cols]], ignore_index=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Spotrac luxury_tax override
# ═══════════════════════════════════════════════════════════════════════════════

_spotrac_ltax_cache: Optional[pd.DataFrame] = None


def _load_spotrac_luxury_tax() -> pd.DataFrame:
    """Load Spotrac salary data with luxury_tax values, keyed by (name, year)."""
    global _spotrac_ltax_cache
    if _spotrac_ltax_cache is not None:
        return _spotrac_ltax_cache

    spotrac_path = DATA_DIR / "salary" / "mlb_salary_data.csv"
    if not spotrac_path.exists():
        _spotrac_ltax_cache = pd.DataFrame()
        return _spotrac_ltax_cache

    df = pd.read_csv(spotrac_path)

    def _parse_dollar(s):
        s = str(s).split("(")[0].replace("$", "").replace(",", "").replace("-", "").strip()
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    df["_ltax"] = df["luxury_tax"].apply(_parse_dollar)
    df = df.dropna(subset=["_ltax"])
    df["_name_key"] = df["player_name"].apply(_name_key_fn)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    _spotrac_ltax_cache = df[["_name_key", "year", "_ltax"]].drop_duplicates(
        subset=["_name_key", "year"], keep="first"
    )
    return _spotrac_ltax_cache


def _override_salary_with_luxury_tax(
    salary_timeline: pd.DataFrame,
) -> pd.DataFrame:
    """Override Cot's-derived Payroll with Spotrac luxury_tax where available.

    Cot's data uses nominal total_future_salary which overstates the
    economic burden for deferred-money contracts (e.g. Ohtani $700M nominal
    but $46M/yr luxury tax).  This function cross-references Spotrac data
    to correct those values.
    """
    ltax = _load_spotrac_luxury_tax()
    if ltax.empty:
        return salary_timeline

    st = salary_timeline.copy()
    st["_name_key"] = st["Name"].apply(_name_key_fn)

    merged = st.merge(
        ltax, left_on=["_name_key", "Year"], right_on=["_name_key", "year"],
        how="left", suffixes=("", "_sp"),
    )

    # Override Payroll with luxury_tax where: (a) luxury_tax is available,
    # (b) the player has a non-arb contract (Signed status means Cot's
    #     distributed total_future_salary, which may be wrong for deferrals)
    mask = merged["_ltax"].notna() & merged["Status"].isin(["Signed"])
    n_overridden = mask.sum()
    if n_overridden > 0:
        merged.loc[mask, "Payroll"] = merged.loc[mask, "_ltax"]
        names = merged.loc[mask, "Name"].unique()
        logger.info(
            f"  Overrode {n_overridden} salary rows with Spotrac luxury_tax "
            f"for {len(names)} players"
        )

    merged = merged.drop(columns=["_ltax", "year", "_name_key"], errors="ignore")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Spotrac years-of-control override (all contract options)
# ═══════════════════════════════════════════════════════════════════════════════

_spotrac_control_cache: Optional[pd.DataFrame] = None


def _load_spotrac_control_data() -> pd.DataFrame:
    """Load per-player control boundaries from Spotrac contract data.

    Returns a DataFrame with columns:
        _name_key        — normalised player name key
        _first_ctrl_year — first year in this Spotrac contract
        _last_ctrl_year  — last non-UFA year in Spotrac (includes club options)

    Only includes players who have at least one option clause (Club,
    Player, Opt-Out, Mutual, Vesting) so that normal Pre-Arb/Arb players
    aren't affected by minor Cot's-vs-Spotrac service-time differences.
    """
    global _spotrac_control_cache
    if _spotrac_control_cache is not None:
        return _spotrac_control_cache

    spotrac_path = DATA_DIR / "salary" / "mlb_salary_data.csv"
    if not spotrac_path.exists():
        _spotrac_control_cache = pd.DataFrame(
            columns=["_name_key", "_first_ctrl_year", "_last_ctrl_year"]
        )
        return _spotrac_control_cache

    df = pd.read_csv(spotrac_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "status"])
    df["year"] = df["year"].astype(int)
    df["_name_key"] = df["player_name"].apply(_name_key_fn)

    # Only consider players who have at least one option/exit clause
    option_mask = df["status"].str.contains(
        r"Opt-Out|Club|(?<!\bClub\b.)Player|Mutual|Vesting",
        case=False, na=False, regex=True,
    )
    option_players = set(df.loc[option_mask, "_name_key"].unique())
    df = df[df["_name_key"].isin(option_players)]

    # --- First and last non-UFA year ---
    non_ufa = df[~df["status"].str.contains(r"\bUFA\b", case=False, na=False)]
    first_ctrl = (
        non_ufa.groupby("_name_key")["year"]
        .min()
        .reset_index()
        .rename(columns={"year": "_first_ctrl_year"})
    )
    last_ctrl = (
        non_ufa.groupby("_name_key")["year"]
        .max()
        .reset_index()
        .rename(columns={"year": "_last_ctrl_year"})
    )

    result = first_ctrl.merge(last_ctrl, on="_name_key")
    _spotrac_control_cache = result
    return _spotrac_control_cache


_spotrac_status_cache: Optional[pd.DataFrame] = None


def _load_spotrac_statuses() -> pd.DataFrame:
    """Load per-player-year Spotrac contract statuses.

    Returns a DataFrame with columns: _name_key, year, _spotrac_status
    where _spotrac_status is the normalised status matching the modern
    pipeline's contract_processor vocabulary (e.g. Opt-Out, Team Option,
    Player Option, Signed).
    """
    global _spotrac_status_cache
    if _spotrac_status_cache is not None:
        return _spotrac_status_cache

    spotrac_path = DATA_DIR / "salary" / "mlb_salary_data.csv"
    if not spotrac_path.exists():
        _spotrac_status_cache = pd.DataFrame(
            columns=["_name_key", "year", "_spotrac_status"]
        )
        return _spotrac_status_cache

    df = pd.read_csv(spotrac_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "status"])
    df["year"] = df["year"].astype(int)
    df["_name_key"] = df["player_name"].apply(_name_key_fn)

    def _normalise(s: str) -> str:
        s = str(s).upper().strip()
        if "OPT-OUT" in s or "OPT OUT" in s:
            return "Opt-Out"
        if "CLUB" in s:
            return "Team Option"
        if "MUTUAL" in s:
            return "Mutual Option"
        if "VESTING" in s:
            return "Vesting Option"
        if "PLAYER" in s:
            return "Player Option"
        return "Signed"

    df["_spotrac_status"] = df["status"].apply(_normalise)
    _spotrac_status_cache = df[["_name_key", "year", "_spotrac_status"]].drop_duplicates(
        subset=["_name_key", "year"], keep="first"
    )
    return _spotrac_status_cache


def _override_years_of_control(
    salary_timeline: pd.DataFrame,
    snapshot_year: int,
) -> pd.DataFrame:
    """Align salary timeline with Spotrac contract boundaries.

    Cot's data doesn't know about contract options, so it can both
    over-count (ignoring opt-outs) and under-count (ignoring club options).
    This function adjusts the timeline to match Spotrac's full contract:

    - **Trim**: Cot's exceeds Spotrac → remove excess rows.
    - **Extend**: Cot's falls short of Spotrac → add rows.
    - **Status**: Overlay Spotrac per-year statuses (Opt-Out, Team Option,
      etc.) so ``analyze_contract_options`` can evaluate them downstream,
      exactly as the modern pipeline does.
    """
    ctrl = _load_spotrac_control_data()
    if ctrl.empty:
        return salary_timeline

    ltax = _load_spotrac_luxury_tax()
    statuses = _load_spotrac_statuses()

    st = salary_timeline.copy()
    st["_name_key"] = st["Name"].apply(_name_key_fn)

    # Per-player last non-FA year in the current Cot's timeline
    non_fa = st[st["Normalized_Status"] != "Free Agent"]
    if non_fa.empty:
        st.drop(columns=["_name_key"], inplace=True)
        return st
    cots_last = (
        non_fa.groupby("_name_key")["Year"]
        .max()
        .reset_index()
        .rename(columns={"Year": "_cots_last"})
    )

    # Merge with Spotrac control data
    compare = cots_last.merge(ctrl, on="_name_key", how="inner")
    if compare.empty:
        st.drop(columns=["_name_key"], inplace=True)
        return st

    # Only apply overrides when Spotrac contract overlaps with Cot's
    # timeline.  If Spotrac's first year is after Cot's last control year,
    # it describes a future contract (e.g. Soto's 2025 Mets deal shouldn't
    # affect his 2019 pre-arb snapshot where Cot's had control through ~2024).
    # Also skip contracts that haven't started yet relative to the snapshot
    # (e.g. Witt's 2024 extension shouldn't modify his 2023 snapshot).
    compare = compare[
        (compare["_first_ctrl_year"] <= compare["_cots_last"])
        & (compare["_first_ctrl_year"] <= snapshot_year)
    ].copy()
    if compare.empty:
        st.drop(columns=["_name_key"], inplace=True)
        return st

    # Use the full Spotrac contract end — don't trim at exit years.
    # The downstream analyze_contract_options will evaluate opt-outs
    # using WAR projections vs remaining contract cost, exactly as the
    # modern pipeline does.

    # ── TRIM: Cot's has more years than Spotrac ─────────────────────────
    to_trim = compare[compare["_cots_last"] > compare["_last_ctrl_year"]].copy()
    n_trimmed_rows = 0
    if not to_trim.empty:
        for _, row in to_trim.iterrows():
            nk = row["_name_key"]
            fa_year = int(row["_last_ctrl_year"]) + 1

            player_mask = st["_name_key"] == nk

            # Convert the row at fa_year to FA
            fa_mask = player_mask & (st["Year"] == fa_year)
            if fa_mask.any():
                st.loc[fa_mask, "Status"] = "Free Agent"
                st.loc[fa_mask, "Normalized_Status"] = "Free Agent"
                st.loc[fa_mask, "Payroll"] = np.nan

            # Drop every row after fa_year
            drop_mask = player_mask & (st["Year"] > fa_year)
            n_trimmed_rows += drop_mask.sum()
            st = st[~drop_mask].copy()

        if n_trimmed_rows > 0:
            logger.info(
                f"  Trimmed {n_trimmed_rows} rows for "
                f"{len(to_trim)} players (Spotrac contract end)"
            )

    # ── EXTEND: Cot's has fewer years than Spotrac ──────────────────────
    to_extend = compare[compare["_cots_last"] < compare["_last_ctrl_year"]].copy()
    n_extended_rows = 0
    if not to_extend.empty:
        new_rows_all: list[dict] = []
        for _, row in to_extend.iterrows():
            nk = row["_name_key"]
            cots_last_yr = int(row["_cots_last"])
            target_last = int(row["_last_ctrl_year"])

            player_mask = st["_name_key"] == nk
            player_rows = st[player_mask]
            if player_rows.empty:
                continue

            template = player_rows.iloc[0]
            idfg = template["IDfg"]
            name = template["Name"]
            team = template.get("Team", "")
            pos_group = template.get("position_group", "")

            # Last known Years_of_Service
            last_sorted = player_rows.sort_values("Year")
            last_yr = int(last_sorted.iloc[-1]["Year"])
            last_yos = last_sorted.iloc[-1].get("Years_of_Service", np.nan)

            # Convert the old FA row (at cots_last_yr + 1) to Signed
            old_fa_year = cots_last_yr + 1
            old_fa_mask = player_mask & (st["Year"] == old_fa_year)
            if old_fa_mask.any():
                ltax_match = ltax[
                    (ltax["_name_key"] == nk) & (ltax["year"] == old_fa_year)
                ]
                sal = ltax_match["_ltax"].iloc[0] if not ltax_match.empty else np.nan
                st.loc[old_fa_mask, "Status"] = "Signed"
                st.loc[old_fa_mask, "Normalized_Status"] = "Signed"
                st.loc[old_fa_mask, "Payroll"] = sal
                n_extended_rows += 1

            # Add new Signed rows for years after the old FA year
            for yr in range(old_fa_year + 1, target_last + 1):
                ltax_match = ltax[
                    (ltax["_name_key"] == nk) & (ltax["year"] == yr)
                ]
                sal = ltax_match["_ltax"].iloc[0] if not ltax_match.empty else np.nan
                yos = (
                    round(last_yos + (yr - last_yr), 3)
                    if pd.notna(last_yos)
                    else np.nan
                )
                new_rows_all.append({
                    "IDfg": idfg,
                    "Name": name,
                    "Year": yr,
                    "Status": "Signed",
                    "Normalized_Status": "Signed",
                    "Payroll": sal,
                    "Years_of_Service": yos,
                    "Team": team,
                    "position_group": pos_group,
                    "_name_key": nk,
                })
                n_extended_rows += 1

            # New FA row at target_last + 1
            new_fa_year = target_last + 1
            yos_fa = (
                round(last_yos + (new_fa_year - last_yr), 3)
                if pd.notna(last_yos)
                else np.nan
            )
            new_rows_all.append({
                "IDfg": idfg,
                "Name": name,
                "Year": new_fa_year,
                "Status": "Free Agent",
                "Normalized_Status": "Free Agent",
                "Payroll": np.nan,
                "Years_of_Service": yos_fa,
                "Team": team,
                "position_group": pos_group,
                "_name_key": nk,
            })

        if new_rows_all:
            st = pd.concat([st, pd.DataFrame(new_rows_all)], ignore_index=True)
        if n_extended_rows > 0:
            logger.info(
                f"  Extended {n_extended_rows} club-option rows for "
                f"{len(to_extend)} players (Spotrac control years)"
            )

    # ── OVERLAY Spotrac statuses on all affected players ────────────────
    # So analyze_contract_options can see Opt-Out, Team Option, etc.
    affected_keys = set(compare["_name_key"])
    affected_mask = st["_name_key"].isin(affected_keys)
    if affected_mask.any():
        affected = st[affected_mask].merge(
            statuses, left_on=["_name_key", "Year"],
            right_on=["_name_key", "year"], how="left",
        )
        has_status = affected["_spotrac_status"].notna()
        # Only override non-FA rows (keep FA rows as-is)
        is_not_fa = affected["Status"] != "Free Agent"
        override_mask = has_status & is_not_fa
        if override_mask.any():
            affected.loc[override_mask, "Status"] = (
                affected.loc[override_mask, "_spotrac_status"]
            )
            affected.loc[override_mask, "Normalized_Status"] = (
                affected.loc[override_mask, "_spotrac_status"]
            )
        affected.drop(columns=["_spotrac_status", "year"], inplace=True, errors="ignore")
        st = pd.concat([st[~affected_mask], affected], ignore_index=True)

    st.drop(columns=["_name_key"], inplace=True)
    return st


# ═══════════════════════════════════════════════════════════════════════════════
# Core surplus computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_surplus_for_snapshot(
    snapshot_year: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute per-player surplus value for a single snapshot year.

    Uses the shared value-determination pipeline for all value calculations,
    ensuring identical logic to the current-year pipeline.

    snapshot_year S  →  cutoff_year = S − 1
    """
    cutoff_year = snapshot_year - Config.SNAPSHOT_LAG
    out_path = SURPLUS_DIR / f"surplus_{snapshot_year}.csv"

    if out_path.exists() and not force:
        logger.info(f"[{snapshot_year}]  surplus file already exists — skipping")
        return pd.read_csv(out_path)

    SURPLUS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Cot's salary ────────────────────────────────────────────────
    cots_file = COTS_BY_YEAR_DIR / f"{snapshot_year}.csv"
    if not cots_file.exists():
        logger.warning(f"[{snapshot_year}]  Cot's salary file missing: {cots_file}")
        return pd.DataFrame()

    cots = pd.read_csv(cots_file)
    cots = cots.dropna(subset=["player"]).copy()
    # Filter out aggregate "Running Payroll Total" rows
    cots = cots[~cots["player"].str.contains("Running|Payroll", na=False)]
    cots["_name_key"] = cots["player"].apply(_name_key_fn)
    cots = cots.sort_values("salary", ascending=False, na_position="last")
    logger.info(f"[{snapshot_year}]  loaded {len(cots)} Cot's rows")

    # ── Load projections ─────────────────────────────────────────────────
    proj_dir = PROJECTIONS_DIR / f"cutoff_{cutoff_year}"
    if not proj_dir.exists():
        logger.warning(f"[{snapshot_year}]  projections dir missing: {proj_dir}")
        return pd.DataFrame()

    pred_files = {
        "batter":      proj_dir / "batter_predictions.csv",
        "pitcher":     proj_dir / "pitcher_predictions.csv",
        "fielding":    proj_dir / "fielding_predictions.csv",
        "baserunning": proj_dir / "baserunning_predictions.csv",
    }
    missing = [k for k, v in pred_files.items() if not v.exists()]
    if missing:
        logger.warning(f"[{snapshot_year}]  missing prediction files: {missing}")
        return pd.DataFrame()

    batter_df      = pd.read_csv(pred_files["batter"])
    pitcher_df     = pd.read_csv(pred_files["pitcher"])
    fielding_df    = pd.read_csv(pred_files["fielding"])
    baserunning_df = pd.read_csv(pred_files["baserunning"])

    logger.info(
        f"[{snapshot_year}]  loaded projections — "
        f"{batter_df['IDfg'].nunique()} batters, {pitcher_df['IDfg'].nunique()} pitchers"
    )

    # ── Build position profiles ──────────────────────────────────────────
    hist_fielding = load_fielding_history()
    hist_batting = load_batting_for_games()
    batter_ids = batter_df["IDfg"].unique().tolist()
    position_profiles = build_position_profiles(
        hist_fielding, hist_batting, batter_ids,
        cutoff_year=cutoff_year,
    )
    logger.info(f"[{snapshot_year}]  built position profiles for {len(position_profiles)}/{len(batter_ids)} batters")

    # ── Compute WAR ──────────────────────────────────────────────────────
    batter_df = calculate_batter_war(
        batter_df, fielding_df, baserunning_df,
        position_profiles=position_profiles,
    )
    pitcher_df = calculate_pitcher_war(pitcher_df)

    # ── Unify into single WAR table ──────────────────────────────────────
    batter_slim = batter_df[["IDfg", "Name", "Year", "Age", "WAR"]].copy()
    batter_slim["player_type"] = "batter"
    batter_slim["position_group"] = "batter"

    pitcher_slim = pitcher_df[["IDfg", "Name", "Year", "Age", "WAR"]].copy()
    pitcher_slim["player_type"] = "pitcher"
    # Determine SP vs RP
    if "Role" in pitcher_df.columns:
        pitcher_slim["position_group"] = pitcher_df["Role"].apply(
            lambda r: "SP" if str(r).upper() == "SP" else "RP"
        )
    else:
        pitcher_slim["position_group"] = "SP"

    all_proj = pd.concat([batter_slim, pitcher_slim], ignore_index=True)

    # Handle two-way players
    #
    # Players in both batter and pitcher predictions fall into two camps:
    #   1. Legitimate two-way players (Ohtani) — both batting and pitching
    #      WAR are positive → sum them.
    #   2. NL pitchers who hit (pre-DH) — pitcher WAR is positive but
    #      batting WAR is hugely negative because the model projects
    #      150 games at terrible batting stats → keep only pitcher WAR.
    #
    # Rule: sum WAR only when BOTH contributions are positive.
    # Otherwise keep the higher (positive) side.
    dup_mask = all_proj.duplicated(subset=["IDfg", "Year"], keep=False)
    two_way_ids = set(all_proj.loc[dup_mask, "IDfg"].unique())

    if two_way_ids:
        two_way = all_proj[all_proj["IDfg"].isin(two_way_ids)]
        non_two = all_proj[~all_proj["IDfg"].isin(two_way_ids)]
        summed = []
        for (idfg, yr), grp in two_way.groupby(["IDfg", "Year"]):
            bat_row = grp[grp["player_type"] == "batter"]
            pit_row = grp[grp["player_type"] == "pitcher"]
            bat_war = bat_row["WAR"].iloc[0] if not bat_row.empty else 0
            pit_war = pit_row["WAR"].iloc[0] if not pit_row.empty else 0

            if bat_war > 0 and pit_war > 0:
                # Legitimate two-way (e.g. Ohtani) — sum both
                base = grp.sort_values("WAR", ascending=False).iloc[0].copy()
                base["WAR"] = bat_war + pit_war
                base["player_type"] = "two_way"
            elif pit_war >= bat_war:
                # Pitcher is better — keep pitcher only (NL pitcher who hit)
                base = pit_row.iloc[0].copy() if not pit_row.empty else grp.iloc[0].copy()
            else:
                # Batter is better — keep batter only
                base = bat_row.iloc[0].copy() if not bat_row.empty else grp.iloc[0].copy()
            summed.append(base)
        all_proj = pd.concat([non_two, pd.DataFrame(summed)], ignore_index=True)
    else:
        all_proj = (
            all_proj.sort_values("WAR", ascending=False)
            .drop_duplicates(subset=["IDfg", "Year"], keep="first")
        )

    all_proj["_name_key"] = all_proj["Name"].apply(_name_key_fn)

    # Add prediction_year for join_predictions_with_timeline
    all_proj["prediction_year"] = all_proj["Year"]

    # ── Identify unique projected players ────────────────────────────────
    proj_players = (
        all_proj[all_proj["Year"] == snapshot_year]
        .drop_duplicates(subset=["IDfg"])
        [["IDfg", "Name", "_name_key", "Age", "player_type", "position_group"]]
        .copy()
    )
    extra = (
        all_proj[~all_proj["IDfg"].isin(proj_players["IDfg"])]
        .sort_values("Year")
        .drop_duplicates(subset=["IDfg"])
        [["IDfg", "Name", "_name_key", "Age", "player_type", "position_group"]]
        .copy()
    )
    proj_players = pd.concat([proj_players, extra], ignore_index=True)

    # ── Match Cot's to projections ───────────────────────────────────────
    cots = _enrich_cots_with_idfg(cots, proj_players)
    cots_id   = cots.dropna(subset=["IDfg"]).drop_duplicates(subset=["IDfg"], keep="first")
    cots_name = cots[cots["IDfg"].isna()].drop_duplicates(subset=["_name_key"], keep="first")
    cots = pd.concat([cots_id, cots_name], ignore_index=True)

    # Merge player_type into Cot's for position group mapping
    cots = cots.merge(
        proj_players[["IDfg", "Name", "player_type"]],
        on="IDfg", how="left", suffixes=("", "_proj"),
    )
    if "Name_proj" in cots.columns:
        cots["Name"] = cots["Name_proj"].fillna(cots.get("player", ""))
        cots.drop(columns=["Name_proj"], inplace=True)
    if "player_type" not in cots.columns or cots["player_type"].isna().all():
        cots["player_type"] = "batter"
    cots["player_type"] = cots["player_type"].fillna("batter")

    matched_idfgs = set(cots.dropna(subset=["IDfg"])["IDfg"].astype(int))
    n_matched = len(matched_idfgs)
    logger.info(f"[{snapshot_year}]  {n_matched} matched in Cot's")

    # ── Build estimated players (not in Cot's) ───────────────────────────
    all_idfgs = set(proj_players["IDfg"].astype(int))
    unmatched_idfgs = all_idfgs - matched_idfgs
    estimated_rows = []
    for idfg in unmatched_idfgs:
        pp = proj_players[proj_players["IDfg"] == idfg]
        if pp.empty:
            continue
        pp = pp.iloc[0]
        svc = _estimate_service_time(idfg, snapshot_year - 1)
        yoc = _years_of_control_from_svc(svc)
        if yoc <= 0:
            continue
        estimated_rows.append({
            "IDfg": int(idfg),
            "Name": pp["Name"],
            "player_type": pp["player_type"],
            "service_time": svc,
            "years_of_control": yoc,
        })
    estimated_df = pd.DataFrame(estimated_rows) if estimated_rows else pd.DataFrame(
        columns=["IDfg", "Name", "player_type", "service_time", "years_of_control"]
    )
    logger.info(f"[{snapshot_year}]  {len(estimated_df)} estimated (no Cot's match)")

    # ── Build salary timeline via adapter ────────────────────────────────
    salary_timeline = build_salary_timeline(
        cots.dropna(subset=["IDfg"]),
        estimated_df,
        snapshot_year,
    )

    if salary_timeline.empty:
        logger.warning(f"[{snapshot_year}]  empty salary timeline — no surplus output")
        return pd.DataFrame()

    # ── Override deferred salaries with Spotrac luxury_tax ───────────────
    salary_timeline = _override_salary_with_luxury_tax(salary_timeline)

    # ── Override years of control using Spotrac opt-outs ─────────────────
    salary_timeline = _override_years_of_control(salary_timeline, snapshot_year)

    # ── Add mlbam_id for prospect matching ───────────────────────────────
    xw = _build_mlbam_crosswalk()
    salary_timeline = salary_timeline.merge(
        xw[["IDfg", "mlbam_id"]], on="IDfg", how="left",
    )

    # ══════════════════════════════════════════════════════════════════════
    # SHARED VALUE PIPELINE (identical to value_determination/main.py)
    # ══════════════════════════════════════════════════════════════════════

    # Step 1: Extend FA timeline through 2040
    extended = extend_fa_timeline(salary_timeline)
    logger.info(f"[{snapshot_year}]  extended timeline: {len(extended)} rows")

    # Step 2: Join predictions with timeline → adds WAR and Base_Value
    with_war = join_predictions_with_timeline(extended, all_proj)

    # Step 3: Calculate contract value (arb model for NaN Payroll rows)
    with_contract = calculate_contract_value(with_war)

    # Step 4: Calculate surplus value (Base_Value - contract_value)
    with_surplus = calculate_surplus_value(with_contract)

    # Step 4b: Add career game counts for confidence adjustments
    #   The trade_value pipeline uses G_bat / G_pit / GS to compute
    #   projection_confidence. Without these, every player gets minimum
    #   confidence (0.10), which massively distorts blended trade values.
    with_surplus = _add_career_game_counts(with_surplus, snapshot_year)

    # Step 5: Analyze contract options (opt-outs, team options)
    with_options = analyze_contract_options(with_surplus, current_year=snapshot_year)

    # Step 6: Calculate trade values (surplus over control years)
    with_trade = calculate_trade_values(with_options, current_year=snapshot_year)

    # Step 7: Add trade ranking metrics
    result = add_trade_ranking_metrics(with_trade, current_year=snapshot_year)

    # Remove synthetic game-count rows added in Step 4b
    if "_synthetic" in result.columns:
        result = result[result["_synthetic"] != True].drop(columns=["_synthetic"])
        # Also drop G_bat/G_pit/GS columns (not needed in summary)
        result = result.drop(columns=[c for c in ["G_bat", "G_pit", "GS"] if c in result.columns],
                             errors="ignore")

    # ══════════════════════════════════════════════════════════════════════
    # EXTRACT PER-PLAYER SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    summary = _extract_player_summary(result, all_proj, snapshot_year)
    if summary.empty:
        logger.warning(f"[{snapshot_year}]  no surplus records produced")
        return summary

    summary = summary.sort_values("trade_value", ascending=False, na_position="last")
    summary = summary.reset_index(drop=True)
    summary.to_csv(out_path, index=False)
    logger.info(
        f"[{snapshot_year}]  wrote {len(summary)} players to {out_path.name}  "
        f"(median trade_value ${summary['trade_value'].median():,.0f})"
    )
    return summary


def _extract_player_summary(
    pipeline_result: pd.DataFrame,
    all_proj: pd.DataFrame,
    snapshot_year: int,
) -> pd.DataFrame:
    """
    Extract one-row-per-player summary from the pipeline output.

    Columns mirror the old surplus format for timeline.py compatibility,
    plus the pre-computed trade_value from the shared pipeline.
    """
    if pipeline_result.empty:
        return pd.DataFrame()

    xw = _build_mlbam_crosswalk()
    records = []

    for idfg in pipeline_result["IDfg"].unique():
        pdata = pipeline_result[pipeline_result["IDfg"] == idfg].sort_values("Year")
        if pdata.empty:
            continue

        first = pdata.iloc[0]
        name = first.get("Name", "")
        team = first.get("Team", "")
        pos_group = first.get("position_group", "")

        # Get mlbam_id
        mlbam_row = xw.loc[xw["IDfg"] == idfg, "mlbam_id"]
        mlbam_id = int(mlbam_row.iloc[0]) if len(mlbam_row) else pd.NA

        # Snapshot year data
        snap = pdata[pdata["Year"] == snapshot_year]
        age = float(snap["Age"].iloc[0]) if not snap.empty and "Age" in snap.columns and pd.notna(snap["Age"].iloc[0]) else np.nan

        # Get player_type from all_proj
        pp = all_proj[all_proj["IDfg"] == idfg]
        player_type = pp["player_type"].iloc[0] if not pp.empty else ""

        # Trade metrics (from add_trade_ranking_metrics — same for all rows)
        trade_value = first.get("trade_value", np.nan)
        years_control = first.get("years_control", 0)
        total_future_war = first.get("total_future_war", 0)
        total_contract = first.get("total_contract", 0)
        contract_war = first.get("contract_war", 0)
        avg_war = first.get("avg_war", 0)
        total_surplus = first.get("total_surplus", 0)

        # Service time / status
        svc = first.get("Years_of_Service", np.nan)
        status = first.get("Normalized_Status", "")
        salary_source = "cots" if pd.notna(first.get("Payroll")) else "estimated"

        # WAR per year
        war_per_year = (
            total_future_war / years_control
            if pd.notna(years_control) and years_control > 0
            else 0.0
        )

        # Per-year WAR and salary columns (for timeline.py compatibility)
        per_year = {}
        for _, row in pdata.iterrows():
            yr = int(row["Year"])
            war = row.get("WAR", np.nan)
            sal = row.get("contract_value", np.nan)
            if pd.notna(war):
                per_year[f"WAR_{yr}"] = round(war, 3)
            if pd.notna(sal):
                per_year[f"salary_{yr}"] = round(sal)

        record = {
            "Name": name,
            "IDfg": int(idfg),
            "mlbam_id": mlbam_id,
            "snapshot_year": snapshot_year,
            "Team": team,
            "Age": age,
            "player_type": player_type,
            "position_group": pos_group,
            "service_time": round(svc, 3) if pd.notna(svc) else np.nan,
            "years_of_control": int(years_control) if pd.notna(years_control) else 0,
            "status": status,
            "salary_source": salary_source,
            "trade_value": round(trade_value) if pd.notna(trade_value) else np.nan,
            "total_future_WAR": round(total_future_war, 2) if pd.notna(total_future_war) else 0,
            "total_future_WAR_value": round(first.get("total_future_value", 0)) if pd.notna(first.get("total_future_value")) else 0,
            "total_future_salary": round(total_contract) if pd.notna(total_contract) else 0,
            "surplus": round(total_surplus) if pd.notna(total_surplus) else 0,
            "contract_war": round(contract_war, 1) if pd.notna(contract_war) else 0,
            "avg_war": round(avg_war, 2) if pd.notna(avg_war) else 0,
            "war_per_year": round(war_per_year, 2),
        }
        record.update(per_year)
        records.append(record)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════════════════

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
    logger.info("Historical Values — Surplus Calculator")
    logger.info(f"Snapshot years {start} → {end}")
    logger.info("=" * 60)

    results: dict[int, pd.DataFrame] = {}
    for snap_year in range(start, end + 1):
        df = compute_surplus_for_snapshot(snap_year, force=force)
        results[snap_year] = df

    total = sum(len(df) for df in results.values())
    logger.info(f"Surplus computation complete — {total} total player rows across {len(results)} years")
    return results
