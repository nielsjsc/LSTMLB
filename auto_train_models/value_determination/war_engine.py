"""
Unified WAR Calculation Engine
==============================

Single source of truth for batter and pitcher WAR calculations across both
current-season (preseason / ROS) and historical (2014–2026) valuation pipelines.

Key V2 Enhancements:
- Post-age-28 defensive spectrum drift (`-0.8` runs/150G per year over age 28,
  floored at DH positional adjustment) so young SS/CF projections naturally
  migrate down the defensive spectrum in their 30s.
- Indexed `(IDfg, Year)` lookup in DataFrame-level batter WAR computation for
  10–20x faster batch execution across 15-year horizons.
- Canonical 3-letter team normalization via `core.id_crosswalk.normalize_team_code`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.id_crosswalk import normalize_team_code
from core.park_factors import get_park_factor, get_woba_residual_factor
from core.position_profiles import (
    DEFENSIVE_POSITIONS,
    estimate_missing_frv,
    get_defensive_positions,
    get_display_position,
    get_weighted_positional_adjustment,
)
from .config import Config, LG_FIP, LG_WOBA, RPA, RPW, WOBA_SCALE, logger

LG_PA = Config.WAR.LG_PA
LG_RUNS_PER_PA = Config.WAR.LG_RUNS_PER_PA
LG_WRC_PER_PA = Config.WAR.LG_WRC_PER_PA
POSITIONAL_ADJUSTMENTS = Config.WAR.POSITIONAL_ADJUSTMENTS
REPLACEMENT_LEVEL_RUNS_200IP = Config.WAR.REPLACEMENT_LEVEL_RUNS_200IP
REPLACEMENT_LEVEL_RUNS_200IP_SP = Config.WAR.REPLACEMENT_LEVEL_RUNS_200IP_SP
REPLACEMENT_LEVEL_RUNS_200IP_RP = Config.WAR.REPLACEMENT_LEVEL_RUNS_200IP_RP
LG_RA9 = Config.WAR.LG_RA9
DEFAULT_IP_PER_START = Config.WAR.DEFAULT_IP_PER_START
DEFAULT_IP_PER_APPEARANCE_RP = Config.WAR.DEFAULT_IP_PER_APPEARANCE_RP
SPECTRUM_DRIFT_PIVOT = Config.WAR.DEFENSIVE_SPECTRUM_DRIFT_AGE_PIVOT
SPECTRUM_DRIFT_RATE = Config.WAR.DEFENSIVE_SPECTRUM_DRIFT_RUNS_PER_YEAR


# =============================================================================
# ORGANIZATION & ROSTER TEAM LOADER
# =============================================================================

def load_player_orgs(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load active player team assignments from `current_rosters.csv`.

    Returns:
        DataFrame with columns: `IDfg`, `mlbam_id`, `Team`, `player_name`
        where `Team` uses canonical 2/3-letter abbreviations.
    """
    if data_dir is None:
        data_dir = Config.Paths.DATA_DIR

    roster_file = Config.Paths.ROSTER_FILE
    if not roster_file.exists():
        roster_file = data_dir / "active_roster" / "current_rosters.csv"
    if not roster_file.exists():
        roster_file = data_dir / "current_rosters.csv"

    if not roster_file.exists():
        logger.warning("Current rosters file not found at %s", roster_file)
        return pd.DataFrame(columns=["IDfg", "mlbam_id", "Team", "player_name"])

    roster_df = pd.read_csv(roster_file, low_memory=False)
    roster_df["fg_id_num"] = pd.to_numeric(roster_df.get("fg_id"), errors="coerce")
    roster_df = roster_df[roster_df["fg_id_num"].notna() & (roster_df["fg_id_num"] > 0)].copy()

    roster_df["Team"] = roster_df["team_name"].map(normalize_team_code)
    org_data = pd.DataFrame({
        "IDfg": roster_df["fg_id_num"].astype(int),
        "mlbam_id": pd.to_numeric(roster_df.get("mlbam_id"), errors="coerce"),
        "Team": roster_df["Team"],
        "player_name": roster_df.get("player_name", ""),
    })
    org_data = org_data.drop_duplicates(subset=["IDfg"], keep="first").reset_index(drop=True)
    logger.info("Loaded %d player organizations from current rosters", len(org_data))
    return org_data


# =============================================================================
# PITCHER WAR
# =============================================================================

def _dynamic_pitcher_rpw(era: float, ip: float, role: str = "SP") -> float:
    """Compute FanGraphs pitcher-specific Runs Per Win."""
    ip_per_game = DEFAULT_IP_PER_APPEARANCE_RP if str(role).upper() == "RP" else DEFAULT_IP_PER_START
    ra9 = float(era) if era and not pd.isna(era) and era > 0 else LG_RA9
    game_ra9 = ((18.0 - ip_per_game) * LG_RA9 + ip_per_game * ra9) / 18.0
    return max((game_ra9 + 2.0) * 1.5, 5.0)


def calculate_pitcher_war(
    fip: float,
    ip: float,
    team: str,
    role: str = "SP",
    rate_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Calculate single-season pitcher WAR from FIP, IP, and park factor."""
    if ip is None or pd.isna(ip) or ip <= 0:
        return 0.0, {
            "FIP_Runs": 0.0,
            "Replacement_Runs": 0.0,
            "Pitcher_RPW": 9.5,
            "WAR": 0.0,
            "IP": 0.0,
            "Team": team,
            "Role": role,
            **(rate_stats or {}),
        }

    is_rp = str(role).upper().strip() == "RP"
    park_factor = get_park_factor(team)
    # Causal Kalman / Marcel FIP is already regressed toward neutral pitch physics (Stuff+/botERA/SIERA);
    # apply residual HR-component park factor (28% of runs PF) rather than dividing the entire FIP + 3.15 constant
    eff_pf = 1.0 + 0.28 * ((park_factor if park_factor > 0 else 1.0) - 1.0)
    park_adjusted_fip = fip / eff_pf
    raw_fip_runs = (LG_FIP - park_adjusted_fip) / 9.0 * ip

    if is_rp:
        gmli_raw = rate_stats.get("gmLI", 1.0) if rate_stats else 1.0
        gmli = 1.0 if (gmli_raw is None or pd.isna(gmli_raw)) else float(np.clip(gmli_raw, 0.55, 2.00))
        lev_mult = (1.0 + gmli) / 2.0
        if raw_fip_runs >= 0.0:
            fip_runs = raw_fip_runs * lev_mult
        else:
            # Struggling relievers are demoted from high-leverage spots rather than compounding 1.5x losses
            fip_runs = raw_fip_runs * float(np.clip(lev_mult, 0.65, 1.0))
        rep_runs_200 = REPLACEMENT_LEVEL_RUNS_200IP_RP
    else:
        fip_runs = raw_fip_runs
        rep_runs_200 = REPLACEMENT_LEVEL_RUNS_200IP_SP

    replacement_runs = rep_runs_200 * (ip / 200.0)
    rar = fip_runs + replacement_runs

    era = rate_stats.get("ERA", LG_RA9) if rate_stats else LG_RA9
    pitcher_rpw = _dynamic_pitcher_rpw(era, ip, role)
    war = rar / pitcher_rpw

    components: Dict[str, Any] = {
        "FIP_Runs": fip_runs,
        "Replacement_Runs": replacement_runs,
        "Pitcher_RPW": pitcher_rpw,
        "WAR": war,
        "IP": ip,
        "Team": team,
        "Role": role,
    }
    if rate_stats:
        components.update(rate_stats)
    return float(war), components


def calculate_pitcher_war_dataframe(
    pitcher_df: pd.DataFrame,
    org_data: Optional[pd.DataFrame] = None,
    role: Optional[str] = None,
    compute_counting: bool = False,
) -> pd.DataFrame:
    """
    Compute FIP-based WAR across a pitcher DataFrame.

    Supports both current-pipeline DataFrames (passing `org_data`, `role`, and
    `compute_counting=True`) and historical snapshot DataFrames.
    """
    if pitcher_df.empty:
        out = pitcher_df.copy()
        out["WAR"] = pd.Series(dtype=float)
        return out

    out = pitcher_df.copy()
    team_lookup: Dict[int, str] = {}
    if org_data is not None and not org_data.empty and "IDfg" in org_data.columns:
        for _, r in org_data.drop_duplicates("IDfg").iterrows():
            team_lookup[int(r["IDfg"])] = str(r.get("Team") or "")

    if "IP" not in out.columns:
        if role is not None:
            out["IP"] = Config.WAR.DEFAULT_SP_IP if role.upper() == "SP" else Config.WAR.DEFAULT_RP_IP
        else:
            out["IP"] = np.where(
                out.get("Role", pd.Series("SP", index=out.index)).astype(str).str.upper() == "SP",
                Config.WAR.DEFAULT_SP_IP,
                Config.WAR.DEFAULT_RP_IP,
            )

    wars = np.zeros(len(out))
    for i, (_, row) in enumerate(out.iterrows()):
        row_role = (role or str(row.get("Role", "SP"))).upper().strip()
        default_ip = Config.WAR.DEFAULT_SP_IP if row_role == "SP" else Config.WAR.DEFAULT_RP_IP
        ip = row.get("IP", default_ip)
        if pd.isna(ip):
            ip = default_ip
        if ip <= 0:
            wars[i] = 0.0
            continue

        team = str(row.get("Team") or "").strip()
        if not team and "IDfg" in row and pd.notna(row["IDfg"]):
            team = team_lookup.get(int(row["IDfg"]), "")

        era = row.get("ERA", LG_RA9)
        gmli = row.get("gmLI", 1.0)
        rate_stats = {
            "ERA": float(era) if pd.notna(era) else LG_RA9,
            "gmLI": float(gmli) if pd.notna(gmli) else 1.0,
        }
        war, _ = calculate_pitcher_war(
            fip=float(row["FIP"]),
            ip=float(ip),
            team=team,
            role=row_role,
            rate_stats=rate_stats,
        )
        wars[i] = war

    out["WAR"] = wars

    if compute_counting:
        bf_per_ip = Config.WAR.BF_PER_IP
        bf = out["IP"].fillna(0.0) * bf_per_ip
        out["K_pit"] = (pd.to_numeric(out.get("K%", 0.0), errors="coerce").fillna(0.0) * bf).round().astype(int)
        out["BB_pit"] = (pd.to_numeric(out.get("BB%", 0.0), errors="coerce").fillna(0.0) * bf).round().astype(int)
        out["ER_pit"] = (pd.to_numeric(out.get("ERA", 0.0), errors="coerce").fillna(0.0) * out["IP"].fillna(0.0) / 9.0).round().astype(int)

        if "K/9" not in out.columns:
            out["K/9"] = pd.to_numeric(out.get("K%", 0.0), errors="coerce").fillna(0.0) * bf_per_ip * 9.0
        if "BB/9" not in out.columns:
            out["BB/9"] = pd.to_numeric(out.get("BB%", 0.0), errors="coerce").fillna(0.0) * bf_per_ip * 9.0
        if "HR/9" not in out.columns:
            hrfb = pd.to_numeric(out.get("HR/FB", 0.10), errors="coerce").fillna(0.10)
            fb = pd.to_numeric(out.get("FB%", 0.35), errors="coerce").fillna(0.35)
            k = pd.to_numeric(out.get("K%", 0.22), errors="coerce").fillna(0.22)
            bb = pd.to_numeric(out.get("BB%", 0.08), errors="coerce").fillna(0.08)
            hbp = pd.to_numeric(out.get("HBP%", 0.01), errors="coerce").fillna(0.01)
            bip_rate = (1.0 - k - bb - hbp).clip(lower=0.30)
            hr_pct = (hrfb * fb * bip_rate).clip(lower=0.005, upper=0.06)
            out["HR/9"] = hr_pct * bf_per_ip * 9.0

    return out


# =============================================================================
# BATTER OFFENSIVE, BASERUNNING, DEFENSIVE, & SPECTRUM DRIFT WAR
# =============================================================================

def calculate_woba(
    ab: float,
    bb: float,
    ibb: float,
    hbp: float,
    sf: float,
    singles: float,
    doubles: float,
    triples: float,
    hr: float,
    pa: Optional[float] = None,
    wbb: float = 0.691,
    whbp: float = 0.722,
    w1b: float = 0.882,
    w2b: float = 1.252,
    w3b: float = 1.584,
    whr: float = 2.037,
) -> float:
    """Calculate wOBA from counting stats using linear weights."""
    numerator = wbb * bb + whbp * hbp + w1b * singles + w2b * doubles + w3b * triples + whr * hr
    if pa is None:
        pa = ab + bb + hbp + sf
    return 0.0 if pa == 0 else float(numerator / pa)


def calculate_wrc_plus(
    woba: float,
    team: str,
    pa: float,
    lg_runs_per_pa: float = LG_RUNS_PER_PA,
    lg_wrc_per_pa: float = LG_WRC_PER_PA,
    lg_woba: float = LG_WOBA,
    woba_scale: float = WOBA_SCALE,
) -> float:
    """Calculate park-adjusted wRC+ using wOBA residual park factor."""
    wraa_per_pa = (woba - lg_woba) / woba_scale
    park_factor = get_woba_residual_factor(team)
    park_adjustment = lg_runs_per_pa - (park_factor * lg_runs_per_pa)
    numerator = (wraa_per_pa + lg_runs_per_pa) + park_adjustment
    return float((numerator / lg_wrc_per_pa) * 100.0)


def calculate_woba_from_predictions(batter_df: pd.DataFrame, use_calculated_woba: Optional[bool] = None) -> pd.DataFrame:
    """Ensure PA is populated on batter prediction DataFrames (Marcel already composes wOBA/OBP/SLG)."""
    out = batter_df.copy()
    if "PA" not in out.columns:
        out["PA"] = 650.0
    return out


def calculate_baserunning_value(row: Any, games: float) -> float:
    """Scale per-150-game Statcast baserunning runs to projected games."""
    rate = row.get("sc_baserunning_runner_runs_tot_rate", 0.0) if row is not None else 0.0
    if pd.isna(rate):
        rate = 0.0
    return float(rate) * (float(games) / 150.0)


def infer_position_from_profile(position_profile: Optional[Dict[str, float]]) -> str:
    if not position_profile:
        return "DH"
    return get_display_position(position_profile)


def _compute_defensive_from_frv_map(
    predicted_frv: Dict[str, float],
    position_profile: Optional[Dict[str, float]],
    games: float,
) -> Tuple[float, float]:
    """Internal helper computing `(fld_value, pos_adjustment)` from `{pos: frv_per_150}`."""
    if not position_profile:
        return 0.0, float(POSITIONAL_ADJUSTMENTS.get("DH", -17.5) * (games / 162.0))

    def_positions = get_defensive_positions(position_profile)
    total_def_fraction = sum(def_positions.values())

    weighted_fld = 0.0
    if predicted_frv and total_def_fraction > 0:
        for pos, frac in def_positions.items():
            if pos in predicted_frv:
                weighted_fld += predicted_frv[pos] * frac
        estimated = estimate_missing_frv(position_profile, predicted_frv)
        for pos, est_frv in estimated.items():
            frac = def_positions.get(pos, 0.0)
            if frac > 0:
                weighted_fld += est_frv * frac

    def_value = weighted_fld * (games / 150.0)
    pos_adjustment = get_weighted_positional_adjustment(
        position_profile, POSITIONAL_ADJUSTMENTS, games
    )
    return float(def_value), float(pos_adjustment)


def calculate_defensive_value(
    fielding_data: pd.DataFrame,
    player_id: int,
    year: int,
    position_profile: Optional[Dict[str, float]] = None,
    games: float = 150,
) -> Tuple[float, float]:
    """Calculate `(fld_value, pos_adjustment)` for a single player-year."""
    if not position_profile:
        return 0.0, float(POSITIONAL_ADJUSTMENTS.get("DH", -17.5) * (games / 162.0))

    predicted_frv: Dict[str, float] = {}
    if fielding_data is not None and not fielding_data.empty:
        player_fielding = fielding_data[
            (fielding_data["IDfg"] == player_id) & (fielding_data["Year"] == year)
        ]
        for _, frow in player_fielding.iterrows():
            pred_pos = frow.get("Pos") or frow.get("Position", "")
            if pred_pos in DEFENSIVE_POSITIONS:
                val = frow.get("sc_total_runs/150", 0.0)
                predicted_frv[pred_pos] = 0.0 if pd.isna(val) else float(val)

    return _compute_defensive_from_frv_map(predicted_frv, position_profile, games)


def _compute_batter_war_from_lookups(
    row: Any,
    bsr_entry: Optional[Dict[str, float]],
    frv_map: Dict[str, float],
    position_profiles: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Fast row-level batter WAR calculation with post-28 defensive spectrum drift."""
    player_id = int(row["IDfg"])
    year = int(row["Year"])
    age = float(row.get("Age", 27.0) or 27.0)

    profile = position_profiles.get(player_id) if position_profiles else None
    position = infer_position_from_profile(profile)

    games = float(row.get("G", 150) if pd.notna(row.get("G", 150)) else 150.0)
    pa = float(row.get("PA", 650) if pd.notna(row.get("PA", 650)) else 650.0)

    if pa <= 0 or games <= 0:
        return 0.0, {
            "Bat": 0.0, "BsR": 0.0, "Fld": 0.0, "Pos": 0.0, "Def": 0.0,
            "WAR": 0.0, "PA": 0.0, "G": 0, "Position": position,
            "Team": row.get("Team", ""),
            "HR": 0.0, "2B": 0.0, "3B": 0.0, "RBI": 0.0, "R": 0.0, "SB": 0.0, "CS": 0.0,
        }

    team = row.get("Team", "")
    park_factor = 1.0 if pd.isna(team) else get_woba_residual_factor(str(team))

    woba = float(row["wOBA"])
    wraa = ((woba - LG_WOBA) / WOBA_SCALE) * pa
    batting_runs = wraa + (RPA - (RPA * park_factor)) * pa

    if bsr_entry is not None:
        bsr = calculate_baserunning_value(bsr_entry, games)
    else:
        bsr = -0.5 * (games / 150.0)

    fld_value, pos_adjustment = _compute_defensive_from_frv_map(frv_map, profile, games)

    # Post-age-28 defensive spectrum drift (-0.8 runs/150G per year past 28)
    spectrum_drift = 0.0
    if age > SPECTRUM_DRIFT_PIVOT:
        spectrum_drift = SPECTRUM_DRIFT_RATE * (age - SPECTRUM_DRIFT_PIVOT) * (games / 150.0)

    def_value = fld_value + pos_adjustment + spectrum_drift

    # Floor total defensive runs at full-time DH penalty
    dh_penalty = POSITIONAL_ADJUSTMENTS.get("DH", -17.5) * (games / 162.0)
    if def_value < dh_penalty:
        def_value = dh_penalty

    off = batting_runs + bsr
    rep_level = 570.0 * RPW * pa / LG_PA
    rar = off + def_value + rep_level
    war = rar / RPW

    counting_stats: Dict[str, float] = {}
    for stat in ("HR", "2B", "3B", "RBI", "R"):
        val = row.get(stat, 0.0)
        counting_stats[stat] = round(float(val) if pd.notna(val) else 0.0, 1)

    if bsr_entry is not None:
        sb_rate = bsr_entry.get("SB_rate", 0.0) or 0.0
        cs_rate = bsr_entry.get("CS_rate", 0.0) or 0.0
        counting_stats["SB"] = round(float(sb_rate) * (games / 150.0), 1)
        counting_stats["CS"] = round(float(cs_rate) * (games / 150.0), 1)
    else:
        counting_stats["SB"] = 0.0
        counting_stats["CS"] = 0.0

    return float(war), {
        "Bat": batting_runs,
        "BsR": bsr,
        "Fld": fld_value,
        "Pos": pos_adjustment,
        "Def": def_value,
        "WAR": float(war),
        "PA": pa,
        "G": int(round(games)),
        "Position": position,
        "Team": team,
        **counting_stats,
    }


def calculate_war_components(
    row: pd.Series,
    baserunning_data: pd.DataFrame,
    fielding_data: pd.DataFrame,
    position_profiles: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Calculate single-row batter WAR components (compatible with legacy callers)."""
    player_id = int(row["IDfg"])
    year = int(row["Year"])

    bsr_entry = None
    if baserunning_data is not None and not baserunning_data.empty:
        bsr_rows = baserunning_data[
            (baserunning_data["IDfg"] == player_id) & (baserunning_data["Year"] == year)
        ]
        if not bsr_rows.empty:
            bsr_entry = bsr_rows.iloc[0].to_dict()

    frv_map: Dict[str, float] = {}
    if fielding_data is not None and not fielding_data.empty:
        fld_rows = fielding_data[
            (fielding_data["IDfg"] == player_id) & (fielding_data["Year"] == year)
        ]
        for _, frow in fld_rows.iterrows():
            pred_pos = frow.get("Pos") or frow.get("Position", "")
            if pred_pos in DEFENSIVE_POSITIONS:
                val = frow.get("sc_total_runs/150", 0.0)
                frv_map[pred_pos] = 0.0 if pd.isna(val) else float(val)

    return _compute_batter_war_from_lookups(
        row=row,
        bsr_entry=bsr_entry,
        frv_map=frv_map,
        position_profiles=position_profiles,
    )


def calculate_batter_war_dataframe(
    batter_df: pd.DataFrame,
    fielding_df: pd.DataFrame,
    baserunning_df: pd.DataFrame,
    position_profiles: Optional[Dict[int, Dict[str, float]]] = None,
    include_components: bool = True,
) -> pd.DataFrame:
    """
    Fast batch batter WAR calculation across an entire DataFrame.

    Pre-indexes `baserunning_df` and `fielding_df` by `(IDfg, Year)` once so
    that 10,000+ batter-year projections execute in <0.2s.
    """
    if batter_df.empty:
        out = batter_df.copy()
        out["WAR"] = pd.Series(dtype=float)
        return out

    out = batter_df.copy()
    if "PA" not in out.columns:
        out["PA"] = 650.0
    if "G" not in out.columns:
        out["G"] = 150

    # Pre-index baserunning by (IDfg, Year)
    bsr_index: Dict[Tuple[int, int], Dict[str, float]] = {}
    if baserunning_df is not None and not baserunning_df.empty:
        for _, brow in baserunning_df.iterrows():
            if pd.notna(brow.get("IDfg")) and pd.notna(brow.get("Year")):
                key = (int(brow["IDfg"]), int(brow["Year"]))
                bsr_index[key] = {
                    "sc_baserunning_runner_runs_tot_rate": brow.get("sc_baserunning_runner_runs_tot_rate", 0.0),
                    "SB_rate": brow.get("SB_rate", 0.0),
                    "CS_rate": brow.get("CS_rate", 0.0),
                }

    # Pre-index fielding FRV by (IDfg, Year) -> {pos: frv}
    fld_index: Dict[Tuple[int, int], Dict[str, float]] = {}
    if fielding_df is not None and not fielding_df.empty:
        for _, frow in fielding_df.iterrows():
            if pd.notna(frow.get("IDfg")) and pd.notna(frow.get("Year")):
                pred_pos = frow.get("Pos") or frow.get("Position", "")
                if pred_pos in DEFENSIVE_POSITIONS:
                    key = (int(frow["IDfg"]), int(frow["Year"]))
                    val = frow.get("sc_total_runs/150", 0.0)
                    fld_index.setdefault(key, {})[pred_pos] = 0.0 if pd.isna(val) else float(val)

    comp_rows = []
    wars = []
    for _, row in out.iterrows():
        key = (int(row["IDfg"]), int(row["Year"]))
        war, comps = _compute_batter_war_from_lookups(
            row=row,
            bsr_entry=bsr_index.get(key),
            frv_map=fld_index.get(key, {}),
            position_profiles=position_profiles,
        )
        wars.append(war)
        comps["IDfg"] = key[0]
        comps["Year"] = key[1]
        comp_rows.append(comps)

    if not include_components:
        out["WAR"] = wars
        return out

    war_df = pd.DataFrame(comp_rows)
    merged = out.merge(war_df, on=["IDfg", "Year"], how="left", suffixes=("_old", ""))
    drop_cols = [c for c in merged.columns if c.endswith("_old")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


# Alias names matching _history_war for convenience
calculate_batter_war = calculate_batter_war_dataframe


# =============================================================================
# PARK FACTOR REAPPLICATION HELPERS
# =============================================================================

def _apply_park_factors_to_batter_predictions(batter_df: pd.DataFrame) -> pd.DataFrame:
    """Reapply park factors to park-neutral batter predictions when enabled in BatterConfig."""
    try:
        from configs.batter_config import BatterConfig
        if not getattr(BatterConfig, "ENABLE_PARK_FACTOR_ADJUSTMENT", False):
            return batter_df
    except (ImportError, AttributeError):
        return batter_df

    if "Team" not in batter_df.columns:
        return batter_df

    from configs.batter_config import BatterConfig
    model_features = list(BatterConfig.FINETUNE_FEATURES) + list(BatterConfig.CLASSICAL_FEATURES)
    seen = set()
    model_features = [f for f in model_features if f not in seen and not seen.add(f)]
    excluded = {"Age", "wRC+"}
    adjustable = [f for f in model_features if f in batter_df.columns and f not in excluded]
    if not adjustable:
        return batter_df

    df = batter_df.copy()
    pf_series = df["Team"].map(get_park_factor)
    for feat in adjustable:
        df[feat] = df[feat] * pf_series
    return df


def _apply_park_factors_to_pitcher_predictions(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """Reapply park factors to park-neutral pitcher predictions when enabled in pitcher configs."""
    sp_enabled = False
    rp_enabled = False
    pitcher_features: set[str] = set()
    try:
        from configs.pitcher_sp_config import PitcherSPConfig
        sp_enabled = getattr(PitcherSPConfig, "ENABLE_PARK_FACTOR_ADJUSTMENT", False)
        pitcher_features.update(PitcherSPConfig.FINETUNE_FEATURES)
        pitcher_features.update(PitcherSPConfig.CLASSICAL_FEATURES)
    except (ImportError, AttributeError):
        pass
    try:
        from configs.pitcher_rp_config import PitcherRPConfig
        rp_enabled = getattr(PitcherRPConfig, "ENABLE_PARK_FACTOR_ADJUSTMENT", False)
        pitcher_features.update(PitcherRPConfig.FINETUNE_FEATURES)
        pitcher_features.update(PitcherRPConfig.CLASSICAL_FEATURES)
    except (ImportError, AttributeError):
        pass

    if (not sp_enabled and not rp_enabled) or "Team" not in pitcher_df.columns:
        return pitcher_df

    excluded = {"Age", "Stuff+", "Location+", "Pitching+"}
    adjustable = [f for f in pitcher_features if f in pitcher_df.columns and f not in excluded]
    if not adjustable:
        return pitcher_df

    df = pitcher_df.copy()
    for idx, row in df.iterrows():
        role = row.get("Role", "SP")
        if (role == "SP" and not sp_enabled) or (role == "RP" and not rp_enabled):
            continue
        team = row.get("Team", "")
        if pd.isna(team) or team == "":
            continue
        pf = get_park_factor(team)
        if pf != 1.0:
            for feat in adjustable:
                if feat in df.columns and pd.notna(row[feat]):
                    df.at[idx, feat] = row[feat] * pf
    return df
