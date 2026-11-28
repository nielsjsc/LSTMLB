"""
Playing Time Estimator (Pitchers & Batters)
===========================================

Estimates realistic, individualized playing time for both pitchers (IP, GS, G)
and batters (PA, G) in the valuation pipeline:

1. **Pitchers**:
   - Rate-based IP ceiling (trailing weighted IP/GS or IP/G) with sample-size regression
   - Starts escalation cap (`max(career_max_GS, 12) + 8` in Year 1, `+4 GS/yr` ramp)
     and a `105 IP` minimum SP ceiling so unproven arms aren't handed 180+ IP
   - Cross-season major surgery recovery (TJS / InternalBrace / labrum) + chronic
     injury discount with workhorse durability credit

2. **Batters**:
   - Trailing 3-season weighted MLB PA (2020 prorated to 162-game pace), regressed
     toward `500 PA` for unproven batters and capped at `650 PA` (`560 PA` for catchers)
   - Empirical age-availability curve past age 30 (`-1.8%/yr`) + current-IL proration
   - Proportional scaling of full-season (per-650-PA) counting stats (`HR`, `2B`, `3B`, `RBI`, `R`)
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_HIST_PITCHING_SC = _ROOT / "data" / "historic_mlb" / "mlb_pitching_data_1950_2025_with_statcast.csv"
_HIST_BATTING_SC = _ROOT / "data" / "historic_mlb" / "mlb_batting_data_1950_2025_with_statcast.csv"
_HIST_PITCHING = _HIST_PITCHING_SC if _HIST_PITCHING_SC.exists() else (_ROOT / "data" / "historic_mlb" / "mlb_pitching_data_1950_2025.csv")
_HIST_BATTING = _HIST_BATTING_SC if _HIST_BATTING_SC.exists() else (_ROOT / "data" / "historic_mlb" / "mlb_batting_data_1950_2025.csv")
_INJURY_CSV = _ROOT / "data" / "injury" / "fangraphs_injury_data.csv"

# ── Season length ────────────────────────────────────────────────────────────
FULL_SEASON_DAYS = 183
GAMES_162_SEASON = 162
SEASON_2020_GAMES = 60

# ── Pitcher IP ceiling parameters ────────────────────────────────────────────
MIN_SP_CEILING = 70.0
MAX_SP_CEILING = 220.0
MIN_RP_CEILING = 25.0
DEFAULT_RP_CEILING = 62.0
MAX_RP_CEILING = 90.0
MIN_IP_THRESHOLD = 10.0
MIN_GAMES_FOR_RATE = 3
TRAILING_YEARS = 3
TRAILING_WEIGHTS = [0.65, 0.25, 0.10]

FULL_SEASON_STARTS_SP = 32
FULL_SEASON_APPEARANCES_RP = 62
STARTS_ESCALATION = 8        # Year-1 max GS = max(career_max_GS, 12) + 8
STARTS_ANNUAL_RAMP = 4       # Additional GS ceiling unlocked per future projection year

DURABILITY_THRESHOLD_SP = 150
DURABILITY_THRESHOLD_RP = 50
DURABILITY_CREDIT_PER_SEASON = 0.30
MIN_DURABILITY_FACTOR = 0.15

LG_IP_PER_GS = 5.45
LG_IP_PER_G_RP = 1.05
RP_IP_PER_G_BASE = 1.45
RP_IP_PER_G_DECLINE = 0.015

# ── Batter PA parameters ─────────────────────────────────────────────────────
FULL_SEASON_PA = 650.0
CATCHER_MAX_PA = 560.0
MIN_BENCH_BATTER_PA = 75.0
MIN_PA_THRESHOLD = 5.0
BATTER_AGE_PIVOT = 30.0
BATTER_AGE_DECLINE_PER_YEAR = 0.018

# ── Injury risk parameters ───────────────────────────────────────────────────
BASE_INJURY_RATE_SP = 0.45
BASE_INJURY_RATE_RP = 0.30
RECURRENCE_ADD_1YR = 0.10
RECURRENCE_ADD_2YR = 0.15
INJURY_AGE_SLOPE = 0.008
INJURY_AGE_PIVOT = 28
MEAN_DAYS_LOST_IF_INJURED_SP = 24.0
MEAN_DAYS_LOST_IF_INJURED_RP = 18.0

MAJOR_SURGERY_RECOVERY: dict[str, int] = {
    "tommy john": 400,
    "ucl surgery": 400,
    "ucl reconstruction": 400,
    "ucl revision": 400,
    "ucl repair": 400,
    "internal brace": 400,
    "labrum surgery": 330,
    "labrum repair": 330,
    "labrum tear": 330,
    "rotator cuff": 330,
}

INJURY_DAYS_LOOKUP: dict[str, int] = {
    "tommy john": 400,
    "ucl surgery": 400,
    "ucl reconstruction": 400,
    "ucl revision": 400,
    "ucl repair": 400,
    "internal brace": 400,
    "labrum surgery": 330,
    "labrum repair": 330,
    "labrum tear": 330,
    "rotator cuff": 330,
    "shoulder surgery": 210,
    "elbow surgery": 210,
    "knee surgery": 120,
    "hip surgery": 120,
    "back surgery": 150,
    "strained lat": 65,
    "strained shoulder": 65,
    "strained forearm": 45,
    "strained oblique": 42,
    "strained hamstring": 28,
    "strained groin": 25,
    "strained lower back": 35,
    "strained calf": 35,
    "strained quad": 30,
    "shoulder impingement": 48,
    "shoulder inflammation": 45,
    "elbow inflammation": 42,
    "forearm inflammation": 36,
    "knee inflammation": 30,
    "elbow discomfort": 45,
    "shoulder discomfort": 35,
    "shoulder fatigue": 17,
    "sprained ankle": 35,
    "sprained elbow": 50,
    "sprained knee": 45,
    "concussion": 25,
    "fractured": 60,
    "broken": 60,
    "non-displaced": 45,
}
DEFAULT_INJURY_DAYS = 40


# ═══════════════════════════════════════════════════════════════════════════════
#  Caches & Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_hist_pit_cache: Optional[pd.DataFrame] = None
_hist_pit_by_fg: Optional[dict[int, pd.DataFrame]] = None
_hist_bat_cache: Optional[pd.DataFrame] = None
_hist_bat_by_fg: Optional[dict[int, pd.DataFrame]] = None
_injury_all_cache: Optional[pd.DataFrame] = None
_pit_inj_by_fg: Optional[dict[int, pd.DataFrame]] = None
_bat_inj_by_fg: Optional[dict[int, pd.DataFrame]] = None


def _to_baseball_ip(ip: float) -> float:
    """Convert raw IP float to baseball notation (.1 = 1/3, .2 = 2/3)."""
    whole = int(ip)
    frac = ip - whole
    thirds = round(frac * 3)
    if thirds >= 3:
        return float(whole + 1)
    return whole + thirds * 0.1


def _estimate_days_from_injury(injury_text: str) -> int:
    if not isinstance(injury_text, str):
        return DEFAULT_INJURY_DAYS
    text = injury_text.lower().strip()
    for pattern, days in INJURY_DAYS_LOOKUP.items():
        if pattern in text:
            return days
    return DEFAULT_INJURY_DAYS


def _parse_date_flexible(d) -> Optional[date]:
    if pd.isna(d) or not isinstance(d, str):
        return None
    try:
        ts = pd.to_datetime(d, errors="coerce")
        return None if pd.isna(ts) else ts.date()
    except Exception:
        return None


def _load_historical_ip() -> pd.DataFrame:
    global _hist_pit_cache, _hist_pit_by_fg
    if _hist_pit_cache is not None:
        return _hist_pit_cache
    if not _HIST_PITCHING.exists():
        logger.warning("Historical pitching file not found: %s", _HIST_PITCHING)
        _hist_pit_cache = pd.DataFrame()
        _hist_pit_by_fg = {}
        return _hist_pit_cache

    df = pd.read_csv(
        _HIST_PITCHING,
        low_memory=False,
        usecols=["IDfg", "Season", "Age", "IP", "GS", "G"],
    )
    df = df.rename(columns={"IDfg": "fg_id", "Season": "season"})
    df = df.dropna(subset=["fg_id", "IP"])
    df["fg_id"] = pd.to_numeric(df["fg_id"], errors="coerce")
    df = df.dropna(subset=["fg_id"])
    df["fg_id"] = df["fg_id"].astype(int)
    df["gs_rate"] = df["GS"] / df["G"].clip(lower=1)
    df = df.sort_values(["fg_id", "season"], ascending=[True, False])
    _hist_pit_cache = df
    _hist_pit_by_fg = {int(k): grp for k, grp in df.groupby("fg_id", sort=False)}
    return _hist_pit_cache


def _get_pitcher_hist(fg_id: int) -> Optional[pd.DataFrame]:
    if _hist_pit_by_fg is None:
        _load_historical_ip()
    return _hist_pit_by_fg.get(int(fg_id)) if _hist_pit_by_fg else None


def _load_historical_pa() -> pd.DataFrame:
    global _hist_bat_cache, _hist_bat_by_fg
    if _hist_bat_cache is not None:
        return _hist_bat_cache
    if not _HIST_BATTING.exists():
        logger.warning("Historical batting file not found: %s", _HIST_BATTING)
        _hist_bat_cache = pd.DataFrame()
        _hist_bat_by_fg = {}
        return _hist_bat_cache

    df = pd.read_csv(
        _HIST_BATTING,
        low_memory=False,
        usecols=["IDfg", "Season", "Age", "PA", "G"],
    )
    df = df.rename(columns={"IDfg": "fg_id", "Season": "season"})
    df["fg_id"] = pd.to_numeric(df["fg_id"], errors="coerce")
    df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
    df["G"] = pd.to_numeric(df["G"], errors="coerce")
    df = df.dropna(subset=["fg_id", "PA"])
    df["fg_id"] = df["fg_id"].astype(int)
    df = df.sort_values(["fg_id", "season"], ascending=[True, False])
    _hist_bat_cache = df
    _hist_bat_by_fg = {int(k): grp for k, grp in df.groupby("fg_id", sort=False)}
    return _hist_bat_cache


_season_pace_cache: Optional[dict[int, float]] = None


def _get_season_pace_mult(season: int) -> float:
    """Return `162 / season_games` (>1.0) for shortened or partial seasons (e.g. 2020, mid-season 2026)."""
    global _season_pace_cache
    if _season_pace_cache is None:
        _load_historical_pa()
        _season_pace_cache = {2020: 162.0 / 60.0}
        if _hist_bat_cache is not None and not _hist_bat_cache.empty:
            for yr, grp in _hist_bat_cache[_hist_bat_cache["season"] >= 2018].groupby("season"):
                q99_g = float(grp["G"].quantile(0.99)) if len(grp) >= 50 else 162.0
                if 20.0 <= q99_g < 148.0:
                    _season_pace_cache[int(yr)] = min(3.0, 162.0 / max(q99_g, 20.0))
    return _season_pace_cache.get(int(season), 1.0)


def _get_batter_hist(fg_id: int) -> Optional[pd.DataFrame]:
    if _hist_bat_by_fg is None:
        _load_historical_pa()
    return _hist_bat_by_fg.get(int(fg_id)) if _hist_bat_by_fg else None


def _load_all_injury_data() -> pd.DataFrame:
    global _injury_all_cache, _pit_inj_by_fg, _bat_inj_by_fg
    if _injury_all_cache is not None:
        return _injury_all_cache
    if not _INJURY_CSV.exists():
        _injury_all_cache = pd.DataFrame()
        _pit_inj_by_fg = {}
        _bat_inj_by_fg = {}
        return _injury_all_cache

    df = pd.read_csv(_INJURY_CSV, low_memory=False)
    df["fg_id"] = pd.to_numeric(df.get("fg_id"), errors="coerce")
    df["retro"] = pd.to_datetime(df.get("il_retro_date"), format="mixed", errors="coerce")
    df["ret"] = pd.to_datetime(df.get("return_date"), format="mixed", errors="coerce")
    df["days_out"] = (df["ret"] - df["retro"]).dt.days
    _injury_all_cache = df

    valid = df.dropna(subset=["fg_id"]).copy()
    valid["fg_id_int"] = valid["fg_id"].astype(int)
    if "position" in valid.columns:
        pit_sub = valid[valid["position"].isin(["SP", "RP", "SP/RP", "RP/SP"])]
        bat_sub = valid[~valid["position"].isin(["SP", "RP", "SP/RP", "RP/SP"])]
    else:
        pit_sub = pd.DataFrame()
        bat_sub = pd.DataFrame()
    _pit_inj_by_fg = {int(k): grp for k, grp in pit_sub.groupby("fg_id_int", sort=False)} if not pit_sub.empty else {}
    _bat_inj_by_fg = {int(k): grp for k, grp in bat_sub.groupby("fg_id_int", sort=False)} if not bat_sub.empty else {}
    return _injury_all_cache


def _load_pitcher_injury_data() -> pd.DataFrame:
    df = _load_all_injury_data()
    if df.empty or "position" not in df.columns:
        return pd.DataFrame()
    return df[df["position"].isin(["SP", "RP", "SP/RP", "RP/SP"])].copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Pitcher Ceiling, Injury Discount, & Playing Time
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_ip_ceiling(
    fg_id: int,
    age: float,
    role: str,
    projection_year: int,
    anchor_year: int,
    proj_fip: float = 4.14,
    proj_gmli: float = 1.0,
    fv: Optional[float] = None,
    include_anchor_season: bool = False,
) -> float:
    """
    Estimate a pitcher's healthy-season IP ceiling conditioned on historical
    usage, projected run-prevention skill (`proj_fip`), leverage (`proj_gmli`),
    and prospect pedigree (`fv`).
    """
    p_hist = _get_pitcher_hist(fg_id)
    yr_offset = max(0, int(projection_year) - int(anchor_year))
    max_hist_yr = int(anchor_year) if (include_anchor_season and int(projection_year) > int(anchor_year)) else int(anchor_year) - 1
    has_pedigree = fv is not None and float(fv) >= 45.0

    if p_hist is None or p_hist.empty:
        if role == "SP":
            base_starts = 18 if has_pedigree else 12
            proj_starts = min(FULL_SEASON_STARTS_SP, base_starts + STARTS_ANNUAL_RAMP * yr_offset)
            raw_ceil = LG_IP_PER_GS * proj_starts
            if not has_pedigree and proj_fip > 4.35:
                raw_ceil *= float(np.clip(1.0 - (proj_fip - 4.35) * 0.60, 0.55, 1.0))
            return float(np.clip(raw_ceil, MIN_SP_CEILING, MAX_SP_CEILING))
        if not has_pedigree and proj_fip > 4.20 and proj_gmli < 0.90:
            return 35.0
        return DEFAULT_RP_CEILING

    pitcher_all = p_hist[p_hist["season"] <= max_hist_yr]
    career_max_gs = 0
    career_ip = 0.0
    if not pitcher_all.empty:
        prorated = pitcher_all.copy()
        pace_s = prorated["season"].map(_get_season_pace_mult)
        prorated["GS"] = (prorated["GS"] * pace_s).round()
        career_max_gs = int(min(34, prorated["GS"].max() or 0))
        career_ip = float(pitcher_all["IP"].sum())

    pitcher = pitcher_all[pitcher_all["IP"] >= MIN_IP_THRESHOLD]

    if role == "SP":
        # Unpedigreed fringe starters (FIP > 4.40, < 18 career max GS) don't automatically ramp to 32 GS
        if not has_pedigree and career_max_gs < 18 and proj_fip > 4.40:
            ramp = 1 if proj_fip <= 4.60 else 0
            esc = 4 if proj_fip <= 4.60 else 2
            projected_starts = min(22, max(career_max_gs, 10) + esc + ramp * yr_offset)
        else:
            projected_starts = min(
                FULL_SEASON_STARTS_SP,
                max(career_max_gs, 12) + STARTS_ESCALATION + STARTS_ANNUAL_RAMP * yr_offset,
            )

        if pitcher.empty:
            ceiling = LG_IP_PER_GS * projected_starts
        else:
            recent = pitcher.head(TRAILING_YEARS)
            rates, weights = [], []
            w_idx = 0
            for _, row in recent.iterrows():
                gs = row["GS"]
                g = max(row["G"], 1)
                if gs < MIN_GAMES_FOR_RATE or gs / g < 0.5:
                    continue
                rates.append(row["IP"] / gs)
                weights.append(TRAILING_WEIGHTS[w_idx] if w_idx < len(TRAILING_WEIGHTS) else 0.10)
                w_idx += 1

            if not rates:
                ip_per_gs = LG_IP_PER_GS
            else:
                total_w = sum(weights)
                ip_per_gs = sum(r * (w / total_w) for r, w in zip(rates, weights))
                reg = 0.75 if len(rates) == 1 else (0.40 if len(rates) == 2 else 0.05)
                ip_per_gs = ip_per_gs * (1.0 - reg) + LG_IP_PER_GS * reg

            ceiling = ip_per_gs * projected_starts

        # Skill modulation for sub-replacement / #6-#7 depth starters
        if not has_pedigree and career_max_gs < 22 and proj_fip > 4.35:
            skill_mult = float(np.clip(1.0 - (proj_fip - 4.35) * 0.65, 0.50, 1.0))
            ceiling *= skill_mult
        return float(np.clip(ceiling, MIN_SP_CEILING, MAX_SP_CEILING))

    # Relief pitcher: modulate appearances by trailing MLB workload, leverage, & projected FIP
    trailing_rp_ip = 0.0
    if not pitcher_all.empty:
        rec_rp = pitcher_all.head(2).copy()
        rec_rp["IP"] = rec_rp["IP"] * rec_rp["season"].map(_get_season_pace_mult)
        trailing_rp_ip = float(rec_rp["IP"].mean())

    if pitcher.empty:
        base_apps = 56 if (has_pedigree or proj_fip <= 3.90 or proj_gmli >= 1.15) else 34
        return float(np.clip(LG_IP_PER_G_RP * base_apps, MIN_RP_CEILING, MAX_RP_CEILING))

    recent = pitcher.head(TRAILING_YEARS)
    rates, weights = [], []
    w_idx = 0
    for _, row in recent.iterrows():
        g, gs = row["G"], row["GS"]
        if g < MIN_GAMES_FOR_RATE or (g > 0 and gs / g >= 0.5):
            continue
        rates.append(row["IP"] / g)
        weights.append(TRAILING_WEIGHTS[w_idx] if w_idx < len(TRAILING_WEIGHTS) else 0.10)
        w_idx += 1

    if not rates:
        ip_per_g = LG_IP_PER_G_RP
    else:
        total_w = sum(weights)
        ip_per_g = sum(r * (w / total_w) for r, w in zip(rates, weights))
        reg = 0.50 if len(rates) == 1 else (0.25 if len(rates) == 2 else 0.05)
        ip_per_g = ip_per_g * (1.0 - reg) + LG_IP_PER_G_RP * reg

    if age > 28:
        ip_per_g *= max(0.70, 1.0 - RP_IP_PER_G_DECLINE * (age - 28))

    # High-leverage / proven relievers earn ~60-64 appearances; low-leverage / fringe
    # shuttle arms (FIP > 4.15, gmLI < 0.90, low career IP) earn ~30-45 appearances
    if proj_gmli >= 1.20 or proj_fip <= 3.85 or trailing_rp_ip >= 50.0 or has_pedigree:
        target_apps = FULL_SEASON_APPEARANCES_RP
    else:
        skill_factor = float(np.clip(1.0 - (proj_fip - 3.95) * 0.70, 0.45, 0.95))
        usage_factor = float(np.clip(trailing_rp_ip / 50.0, 0.45, 1.0))
        target_apps = FULL_SEASON_APPEARANCES_RP * (0.55 * skill_factor + 0.45 * usage_factor)

    ceiling = ip_per_g * target_apps
    return float(np.clip(ceiling, MIN_RP_CEILING, MAX_RP_CEILING))


def _is_major_surgery(injury_text: str) -> Optional[int]:
    if not isinstance(injury_text, str):
        return None
    text = injury_text.lower().strip()
    for pattern, days in MAJOR_SURGERY_RECOVERY.items():
        if pattern in text:
            return days
    return None


def _is_end_of_season_date(d: Optional[date], year: int) -> bool:
    if d is None:
        return False
    return d.year == year and d.month >= 9 and d.day >= 25


def _major_surgery_return_date(fg_id: float, projection_year: int, player_inj_df: Optional[pd.DataFrame]) -> Optional[date]:
    if player_inj_df is None or player_inj_df.empty:
        return None
    records = player_inj_df[
        player_inj_df["season"].between(projection_year - 2, projection_year)
    ].sort_values("season", ascending=True)

    for _, row in records.iterrows():
        recovery_days = _is_major_surgery(str(row.get("injury_surgery", "")))
        if recovery_days is None:
            continue

        injury_date_parsed = _parse_date_flexible(row.get("injury_date"))
        il_retro_parsed = _parse_date_flexible(row.get("il_retro_date"))
        if injury_date_parsed is not None and il_retro_parsed is not None:
            surgery_date = max(injury_date_parsed, il_retro_parsed)
        elif il_retro_parsed is not None:
            surgery_date = il_retro_parsed
        elif injury_date_parsed is not None:
            surgery_date = injury_date_parsed
        else:
            continue

        expected_return = surgery_date + pd.Timedelta(days=recovery_days)
        try:
            if hasattr(expected_return, "date"):
                expected_return = expected_return.date()
            if pd.isna(expected_return):
                continue
        except (TypeError, ValueError):
            continue

        actual_return = _parse_date_flexible(row.get("return_date"))
        season_of_injury = int(row["season"])
        if actual_return is not None and not _is_end_of_season_date(actual_return, season_of_injury):
            continue

        season_start = date(projection_year, 3, 27)
        if expected_return > season_start:
            return expected_return

    return None


def _current_il_days_remaining(
    fg_id: float,
    projection_year: int,
    today: Optional[date] = None,
    inj_df: Optional[pd.DataFrame] = None,
    player_inj_df: Optional[pd.DataFrame] = None,
) -> float:
    if today is None:
        today = date.today()
    if player_inj_df is None:
        if inj_df is not None and not inj_df.empty:
            player_inj_df = inj_df[inj_df["fg_id"] == fg_id]
        else:
            _load_all_injury_data()
            player_inj_df = _pit_inj_by_fg.get(int(fg_id)) if _pit_inj_by_fg else None

    surgery_return = _major_surgery_return_date(fg_id, projection_year, player_inj_df)
    if surgery_return is not None:
        season_start = date(projection_year, 3, 27)
        season_end = date(projection_year, 9, 30)
        if surgery_return > season_end:
            return float(FULL_SEASON_DAYS)
        ref = max(today, season_start)
        return float(max(0.0, (surgery_return - ref).days))

    if player_inj_df is None or player_inj_df.empty:
        return 0.0

    current = player_inj_df[player_inj_df["season"] == projection_year]
    if current.empty:
        return 0.0

    latest = current.iloc[-1]
    ret_date = _parse_date_flexible(latest.get("return_date"))
    if ret_date is not None and ret_date <= today:
        return 0.0

    update = str(latest.get("latest_update", "")).lower()
    if f"out for {projection_year}" in update:
        season_end = date(projection_year, 9, 30)
        return float(max(0.0, (season_end - today).days))

    if ret_date is not None and ret_date > today:
        return float((ret_date - today).days)

    etr = _parse_date_flexible(latest.get("eligible_to_return"))
    if etr is not None and etr > today:
        return float((etr - today).days)

    est_days = _estimate_days_from_injury(str(latest.get("injury_surgery", "")))
    il_start = _parse_date_flexible(latest.get("il_retro_date"))
    if il_start is not None:
        return float(max(0, est_days - (today - il_start).days))
    return float(est_days)


def _history_injury_discount(
    fg_id: float,
    age: float,
    role: str,
    projection_year: int,
    anchor_year: int,
) -> float:
    _load_all_injury_data()
    p_inj = _pit_inj_by_fg.get(int(fg_id)) if _pit_inj_by_fg else None
    base_rate = BASE_INJURY_RATE_SP if role == "SP" else BASE_INJURY_RATE_RP
    if age > INJURY_AGE_PIVOT:
        base_rate += INJURY_AGE_SLOPE * (age - INJURY_AGE_PIVOT)
    base_rate = min(base_rate, 0.75)

    if p_inj is not None and not p_inj.empty:
        prior = p_inj[p_inj["season"].isin([anchor_year - 1, anchor_year - 2])]
        seasons_injured = prior["season"].nunique()
        if seasons_injured >= 2:
            base_rate += RECURRENCE_ADD_2YR
        elif seasons_injured == 1:
            base_rate += RECURRENCE_ADD_1YR
        base_rate = min(base_rate, 0.75)

    mean_days = MEAN_DAYS_LOST_IF_INJURED_SP if role == "SP" else MEAN_DAYS_LOST_IF_INJURED_RP
    expected_days_lost = base_rate * mean_days

    p_hist = _get_pitcher_hist(int(fg_id))
    if p_hist is not None and not p_hist.empty:
        threshold = DURABILITY_THRESHOLD_SP if role == "SP" else DURABILITY_THRESHOLD_RP
        recent_hist = p_hist[
            p_hist["season"].between(anchor_year - 3, anchor_year - 1)
            & (p_hist["season"] != 2020)
        ]
        durable_count = int((recent_hist["IP"] >= threshold).sum())
        durability_factor = max(MIN_DURABILITY_FACTOR, 1.0 - durable_count * DURABILITY_CREDIT_PER_SEASON)
        expected_days_lost *= durability_factor

        most_recent = p_hist[
            (p_hist["season"] < anchor_year)
            & (p_hist["season"] != 2020)
        ]
        if not most_recent.empty:
            ceiling = _estimate_ip_ceiling(int(fg_id), age, role, anchor_year, anchor_year)
            if ceiling > 0:
                achievement = min(1.0, most_recent.iloc[0]["IP"] / ceiling)
                recent_health_mult = max(
                    0.30,
                    min(2.0, 2.0 - max(0.0, achievement - 0.30) / 0.70 * 1.70),
                )
                # Decay recent-season health penalty across distant future years
                yr_offset = max(0, projection_year - anchor_year)
                decay = 0.65 ** yr_offset
                blended_mult = 1.0 + (recent_health_mult - 1.0) * decay
                expected_days_lost *= blended_mult

    return float(expected_days_lost)


def _estimate_gs_rate(fg_id: int, role: str, anchor_year: int) -> float:
    p_hist = _get_pitcher_hist(fg_id)
    if p_hist is None or p_hist.empty:
        return 1.0 if role == "SP" else 0.0

    recent = p_hist[
        (p_hist["season"] < anchor_year)
        & (p_hist["IP"] >= MIN_IP_THRESHOLD)
    ].head(2)

    if recent.empty:
        return 1.0 if role == "SP" else 0.0

    gs_rate = (
        recent.iloc[0]["gs_rate"] * 0.65 + recent.iloc[1]["gs_rate"] * 0.35
        if len(recent) >= 2
        else recent.iloc[0]["gs_rate"]
    )
    archetype = 1.0 if role == "SP" else 0.0
    return float(np.clip(gs_rate * 0.80 + archetype * 0.20, 0.0, 1.0))


def _ip_to_gs_g(ip: float, gs_rate: float, age: float) -> tuple[int, int]:
    if ip <= 0:
        return 0, 0
    if gs_rate >= 0.5:
        gs = min(round(ip / LG_IP_PER_GS * gs_rate), FULL_SEASON_STARTS_SP)
        relief_g = round(ip / LG_IP_PER_GS * (1.0 - gs_rate)) if gs_rate < 1.0 else 0
        g = gs + relief_g
    else:
        ip_per_g = max(0.7, RP_IP_PER_G_BASE - RP_IP_PER_G_DECLINE * max(0.0, age - 22))
        g = round(ip / ip_per_g)
        gs = round(g * gs_rate)
    return max(0, int(gs)), max(1, int(g))


_prospect_by_anchor_cache: dict[int, dict] = {}


def _get_prospect_lookup_for_pt(anchor_year: int) -> dict:
    if anchor_year not in _prospect_by_anchor_cache:
        try:
            from core.id_crosswalk import load_recent_prospect_info
            _prospect_by_anchor_cache[anchor_year] = load_recent_prospect_info(int(anchor_year), recency_years=3)
        except Exception:
            _prospect_by_anchor_cache[anchor_year] = {}
    return _prospect_by_anchor_cache[anchor_year]


def estimate_playing_time(
    pitcher_df: pd.DataFrame,
    projection_year: int,
    anchor_year: Optional[int] = None,
    include_anchor_season: bool = False,
) -> pd.DataFrame:
    """
    Estimate IP, GS, and G for every pitcher row in *pitcher_df* conditioned on
    usage history, injury status, projected FIP/gmLI, and prospect pedigree.
    """
    if pitcher_df.empty:
        return pitcher_df.copy()

    _load_historical_ip()
    _load_all_injury_data()

    if anchor_year is None:
        from .config import CURRENT_YEAR
        anchor_year = min(int(projection_year), CURRENT_YEAR)

    prospect_lookup = _get_prospect_lookup_for_pt(int(anchor_year))

    id_col = "fg_id" if "fg_id" in pitcher_df.columns else "IDfg"
    age_col = "age" if "age" in pitcher_df.columns else "Age"
    role_col = "player_role" if "player_role" in pitcher_df.columns else "Role"

    today = date.today()
    n = len(pitcher_df)
    ips = np.zeros(n)
    gss = np.zeros(n, dtype=int)
    gs_arr = np.zeros(n, dtype=int)

    for i, (_, row) in enumerate(pitcher_df.iterrows()):
        fg_id = int(row[id_col])
        age = float(row[age_col])
        role = str(row[role_col]).upper()
        role = "SP" if role == "SP" else "RP"

        proj_fip = float(row["FIP"]) if ("FIP" in row and pd.notna(row["FIP"])) else 4.14
        proj_gmli = float(row["gmLI"]) if ("gmLI" in row and pd.notna(row["gmLI"])) else 1.0
        p_info = prospect_lookup.get(fg_id)
        fv_val = float(p_info["fv"]) if (p_info and p_info.get("fv") is not None) else None

        ceiling = _estimate_ip_ceiling(
            fg_id,
            age,
            role,
            int(projection_year),
            int(anchor_year),
            proj_fip=proj_fip,
            proj_gmli=proj_gmli,
            fv=fv_val,
            include_anchor_season=include_anchor_season,
        )

        il_days = (
            _current_il_days_remaining(
                fg_id, int(projection_year), today,
                player_inj_df=_pit_inj_by_fg.get(fg_id) if _pit_inj_by_fg else None,
            )
            if int(projection_year) >= int(anchor_year)
            else 0.0
        )
        if il_days >= FULL_SEASON_DAYS:
            ips[i] = 0.0
            gss[i] = 0
            gs_arr[i] = 0
            continue

        hist_days_lost = _history_injury_discount(
            fg_id, age, role, int(projection_year), int(anchor_year)
        )
        il_fraction = max(0.0, 1.0 - il_days / FULL_SEASON_DAYS) if il_days > 0 else 1.0
        hist_fraction = max(0.20, 1.0 - hist_days_lost / FULL_SEASON_DAYS)
        ip_final = ceiling * min(il_fraction, hist_fraction)

        gs_rate = _estimate_gs_rate(fg_id, role, int(anchor_year))
        gs, g = _ip_to_gs_g(ip_final, gs_rate, age)

        ips[i] = _to_baseball_ip(ip_final)
        gss[i] = gs
        gs_arr[i] = g

    out = pitcher_df.copy()
    out["IP"] = ips
    out["GS"] = gss
    out["G"] = gs_arr
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Batter Playing Time (PA & G) — Skill- & Usage-Earned
# ═══════════════════════════════════════════════════════════════════════════════

_POS_PT_BONUS = {
    "C": 4.0, "SS": 5.5, "CF": 4.5, "2B": 2.5, "3B": 2.0,
    "RF": -2.0, "LF": -3.0, "1B": -7.0, "DH": -10.0,
}


def _skill_earned_batter_pa(
    proj_woba: float,
    primary_pos: str,
    age: float,
    yr_offset: int,
    fv: Optional[float],
) -> float:
    """
    Map projected offensive skill (`proj_woba`), positional scarcity, and prospect
    pedigree (`fv`) to earned healthy-season PA.
    """
    wrc_approx = 100.0 + ((proj_woba - 0.313) / 1.232) / 0.117 * 100.0
    pos_bonus = _POS_PT_BONUS.get(primary_pos, 0.0)
    eff_wrc = wrc_approx + pos_bonus

    # Piecewise-linear skill curve:
    #   eff_wrc <= 78 -> 105 PA (up-and-down bench / waiver depth)
    #   eff_wrc == 88 -> 235 PA (backup / utility / weak platoon)
    #   eff_wrc == 97 -> 430 PA (strong platoon / low-end regular)
    #   eff_wrc == 106 -> 565 PA (everyday starter)
    #   eff_wrc >= 118 -> 625 PA (lineup anchor)
    skill_pa = float(np.interp(
        eff_wrc,
        [76.0, 86.0, 96.0, 105.0, 118.0],
        [100.0, 215.0, 420.0, 560.0, 625.0],
    ))

    if fv is not None and fv >= 45.0 and age <= 26.0:
        pedigree_floor = 290.0 + 11.0 * (fv - 45.0) + (38.0 * yr_offset if fv >= 50.0 else 18.0 * yr_offset)
        skill_pa = max(skill_pa, min(615.0, pedigree_floor))

    return skill_pa


def _estimate_single_batter_pa(
    fg_id: int,
    age: float,
    projection_year: int,
    anchor_year: int,
    primary_pos: str = "",
    proj_woba: float = 0.313,
    fv: Optional[float] = None,
    include_anchor_season: bool = False,
) -> float:
    """
    Estimate healthy-season PA for a batter by combining revealed-preference
    trailing MLB usage with projected skill-earned playing time (`proj_woba` + `fv`).
    """
    b_hist = _get_batter_hist(fg_id)
    yr_offset = max(0, int(projection_year) - int(anchor_year))
    max_hist_yr = int(anchor_year) if (include_anchor_season and int(projection_year) > int(anchor_year)) else int(anchor_year) - 1

    skill_pa = _skill_earned_batter_pa(proj_woba, primary_pos, age, yr_offset, fv)
    has_pedigree = fv is not None and float(fv) >= 45.0

    career_pa = 0.0
    if b_hist is None or b_hist.empty:
        base_pa = skill_pa
    else:
        player = b_hist[
            (b_hist["season"] <= max_hist_yr)
            & (b_hist["PA"] >= MIN_PA_THRESHOLD)
        ]
        career_pa = float(player["PA"].sum()) if not player.empty else 0.0

        if player.empty:
            base_pa = skill_pa
        else:
            recent = player.head(TRAILING_YEARS)
            pas, weights = [], []
            for idx_w, (_, r) in enumerate(recent.iterrows()):
                pa_val = float(r["PA"]) * _get_season_pace_mult(int(r["season"]))
                pa_val = min(FULL_SEASON_PA, pa_val)
                pas.append(pa_val)
                weights.append(TRAILING_WEIGHTS[idx_w] if idx_w < len(TRAILING_WEIGHTS) else 0.10)

            total_w = sum(weights)
            trailing_pa = sum(p * (w / total_w) for p, w in zip(pas, weights))
            n_seasons = len(pas)

            if n_seasons == 1:
                if has_pedigree and age <= 25.0 and trailing_pa < skill_pa:
                    base_pa = 0.35 * trailing_pa + 0.65 * skill_pa
                else:
                    base_pa = 0.50 * trailing_pa + 0.50 * skill_pa
                # Unproven <250 career PA rookies ramp into full-time MLB workload over Year 1..2
                if career_pa < 250.0 and (fv is None or float(fv) < 55.0):
                    rookie_cap = 390.0 + 0.45 * career_pa + 45.0 * yr_offset
                    base_pa = min(base_pa, rookie_cap)
            elif n_seasons == 2:
                base_pa = 0.60 * trailing_pa + 0.40 * skill_pa
            else:
                base_pa = 0.72 * trailing_pa + 0.28 * skill_pa

    # Sub-regular non-prospects gradually lose MLB playing time to roster competition
    wrc_approx = 100.0 + ((proj_woba - 0.313) / 1.232) / 0.117 * 100.0 + _POS_PT_BONUS.get(primary_pos, 0.0)
    if not has_pedigree and wrc_approx < 90.0 and yr_offset >= 1:
        base_pa *= (0.88 ** yr_offset)

    if primary_pos == "C":
        # Catchers rarely exceed 420-450 PA unless proven multi-year workhorses (e.g. Raleigh/Perez)
        max_pa = float(np.interp(career_pa + 180.0 * yr_offset, [150.0, 500.0, 1400.0], [400.0, 470.0, CATCHER_MAX_PA]))
    else:
        max_pa = FULL_SEASON_PA
    base_pa = float(np.clip(base_pa, MIN_BENCH_BATTER_PA, max_pa))

    # Empirical age-availability curve past age 30
    if age > BATTER_AGE_PIVOT:
        age_mult = max(0.60, 1.0 - BATTER_AGE_DECLINE_PER_YEAR * (age - BATTER_AGE_PIVOT))
        base_pa *= age_mult

    return float(np.clip(base_pa, 60.0, max_pa))


def estimate_batter_playing_time(
    batter_df: pd.DataFrame,
    projection_year: Optional[int] = None,
    anchor_year: Optional[int] = None,
    position_profiles: Optional[dict] = None,
    include_anchor_season: bool = False,
) -> pd.DataFrame:
    """
    Estimate individualized PA and G for batter prediction rows, scaling
    per-650-PA counting stats (`HR`, `2B`, `3B`, `RBI`, `R`, `HBP`, `SF`)
    proportionally.
    """
    if batter_df.empty:
        return batter_df.copy()

    _load_historical_pa()
    _load_all_injury_data()

    from .config import CURRENT_YEAR
    if anchor_year is None:
        min_yr = int(batter_df["Year"].min()) if "Year" in batter_df.columns else CURRENT_YEAR
        anchor_year = min(min_yr, CURRENT_YEAR)

    prospect_lookup = _get_prospect_lookup_for_pt(int(anchor_year))

    id_col = "fg_id" if "fg_id" in batter_df.columns else "IDfg"
    age_col = "age" if "age" in batter_df.columns else "Age"
    yr_col = "year" if "year" in batter_df.columns else "Year"

    out = batter_df.copy()
    today = date.today()

    new_pa = np.zeros(len(out))
    new_g = np.zeros(len(out), dtype=int)
    scale_ratios = np.ones(len(out))

    for i, (_, row) in enumerate(out.iterrows()):
        fg_id = int(row[id_col])
        age = float(row[age_col])
        p_year = int(projection_year if projection_year is not None else row[yr_col])

        primary_pos = str(row.get("Position", "") or "")
        if not primary_pos and position_profiles and fg_id in position_profiles:
            prof = position_profiles[fg_id]
            if prof:
                primary_pos = max(prof, key=prof.get)

        proj_woba = float(row["wOBA"]) if ("wOBA" in row and pd.notna(row["wOBA"])) else 0.313
        p_info = prospect_lookup.get(fg_id)
        fv_val = float(p_info["fv"]) if (p_info and p_info.get("fv") is not None) else None

        pa_est = _estimate_single_batter_pa(
            fg_id=fg_id,
            age=age,
            projection_year=p_year,
            anchor_year=int(anchor_year),
            primary_pos=primary_pos,
            proj_woba=proj_woba,
            fv=fv_val,
            include_anchor_season=include_anchor_season,
        )

        # Check current-IL limitation if batter is out for part/all of p_year
        if p_year == CURRENT_YEAR and _bat_inj_by_fg:
            b_inj = _bat_inj_by_fg.get(fg_id)
            if b_inj is not None:
                il_days = _current_il_days_remaining(fg_id, p_year, today, player_inj_df=b_inj)
                if il_days >= FULL_SEASON_DAYS:
                    pa_est = 0.0
                elif il_days > 15:
                    pa_est *= max(0.0, 1.0 - il_days / FULL_SEASON_DAYS)

        old_pa = float(row.get("PA", FULL_SEASON_PA) or FULL_SEASON_PA)
        ratio = (pa_est / old_pa) if old_pa > 0 else 1.0

        new_pa[i] = round(pa_est)
        new_g[i] = 0 if pa_est <= 0 else max(1, min(162, round(pa_est * (150.0 / FULL_SEASON_PA))))
        scale_ratios[i] = ratio

    out["PA"] = new_pa
    out["G"] = new_g

    for stat in ("HR", "2B", "3B", "RBI", "R", "HBP", "SF"):
        if stat in out.columns:
            out[stat] = pd.to_numeric(out[stat], errors="coerce").fillna(0.0) * scale_ratios

    return out
