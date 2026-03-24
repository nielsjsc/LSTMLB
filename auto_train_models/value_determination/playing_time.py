"""
Pitcher Playing Time Estimator
==============================

Estimates IP, GS, and G for pitcher projections in the value determination
pipeline.  Replaces the fixed DEFAULT_SP_IP / DEFAULT_RP_IP approach with
individualised estimates based on:

    1. **IP ceiling** — trailing weighted IP from prior seasons, age-adjusted
    2. **Injury discount** — current-IL limitation and injury-history risk
    3. **GS / G split** — projected GS rate from historical GS/G pattern

Design principle: estimate the **maximum innings a pitcher could throw
given their health**, not how many a particular team would use them.
A back-end starter with a 4.4 ERA should still be projected for full
innings so his WAR (and therefore trade value) isn't artificially deflated.

Called from ``value_determination/main.py`` Step 2, before WAR calculation.
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_HIST_PITCHING = _ROOT / "data" / "historic_mlb" / "mlb_pitching_data_1950_2025.csv"
_INJURY_CSV = _ROOT / "data" / "injury" / "fangraphs_injury_data.csv"

# ── Season length ────────────────────────────────────────────────────────────
FULL_SEASON_DAYS = 183  # ~Apr 1 – Sep 30
GAMES_162_SEASON = 162
SEASON_2020_GAMES = 60

# ── IP Ceiling defaults ─────────────────────────────────────────────────────
# Rate-based: ceiling = IP/GS × full_starts (SP), IP/G × full_apps (RP)
# This avoids injury-shortened seasons dragging the ceiling down.
DEFAULT_SP_CEILING = 180.0  # fallback when no historical data
DEFAULT_RP_CEILING = 65.0   # fallback when no historical data
MIN_IP_THRESHOLD = 10.0     # ignore seasons below this
MIN_GAMES_FOR_RATE = 3      # need at least this many GS (SP) or G (RP) for rate
TRAILING_YEARS = 3          # how many prior seasons to consider
TRAILING_WEIGHTS = [0.65, 0.25, 0.10]  # most-recent first

# Full-season opportunity counts (healthy workhorse)
FULL_SEASON_STARTS_SP = 32   # 162 / 5-man rotation
FULL_SEASON_APPEARANCES_RP = 62
STARTS_ESCALATION = 8        # max projected GS = career_max_GS + this

# Durability credit: reduce injury discount for proven workhorses
DURABILITY_THRESHOLD_SP = 150  # IP threshold for a "durable" SP season
DURABILITY_THRESHOLD_RP = 50   # IP threshold for a "durable" RP season
DURABILITY_CREDIT_PER_SEASON = 0.30  # 30% discount reduction per durable season
MIN_DURABILITY_FACTOR = 0.15   # floor: even 3+ durable seasons still get 15%

# ── IP/GS and IP/G league averages (2021-2025) ──────────────────────────────
# Used for regression when sample is small
LG_IP_PER_GS = 5.45
LG_IP_PER_G_RP = 1.05
# RP IP/G declines with age — fit from analysis
RP_IP_PER_G_BASE = 1.45
RP_IP_PER_G_DECLINE = 0.015  # per year above age 22

# ── Injury risk parameters (from recurrence analysis) ───────────────────────
# Base injury rates by role (P(hitting IL at least once) per season)
BASE_INJURY_RATE_SP = 0.45
BASE_INJURY_RATE_RP = 0.30

# Recurrence multiplier: if injured prior year, injury risk goes up
# Analysis showed: 37% base → 53% for 1yr prior → 55% for 2yr prior
# (additive, not geometric)
RECURRENCE_ADD_1YR = 0.10   # +10% absolute for 1 prior injured year
RECURRENCE_ADD_2YR = 0.15   # +15% absolute for 2 prior injured years

# Age adjustment: injury rate increases ~0.8% per year above 28
INJURY_AGE_SLOPE = 0.008
INJURY_AGE_PIVOT = 28

# Mean days lost when injured (conditional on being injured)
MEAN_DAYS_LOST_IF_INJURED_SP = 45.0
MEAN_DAYS_LOST_IF_INJURED_RP = 35.0

# Major surgeries with long recovery — keyed by substring match.
# Recovery days are from SURGERY DATE, not IL placement.  Used to compute
# an expected return date that overrides the often-bogus eligible_to_return
# / return_date fields in the FG data (e.g. TJS return_date = last day of
# the regular season, not when the pitcher actually comes back).
# NOTE: Only truly major reconstructive/repair surgeries belong here.
# Arthroscopic cleanups (bone chips, bone spurs) do NOT — they're handled
# by the standard IL-days path which uses eligible_to_return dates.
MAJOR_SURGERY_RECOVERY: dict[str, int] = {
    "tommy john": 400,           # ~13.3 months (modern TJS game-readiness)
    "ucl surgery": 400,
    "ucl reconstruction": 400,
    "ucl revision": 400,         # UCL revision repair (at least as long as TJS)
    "ucl repair": 400,
    "internal brace": 400,       # UCL repair with InternalBrace (same recovery as TJS)
    "labrum surgery": 330,       # ~11 months
    "labrum repair": 330,
    "labrum tear": 330,
    "rotator cuff": 330,         # ~11 months
}

# Injury type → typical days lost (for current-IL estimation)
# Derived from FanGraphs injury data 2020-2025
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

# Fallback if injury text doesn't match any known type
DEFAULT_INJURY_DAYS = 40


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_days_from_injury(injury_text: str) -> int:
    """Estimate recovery days from injury description text."""
    if not isinstance(injury_text, str):
        return DEFAULT_INJURY_DAYS
    text = injury_text.lower().strip()
    for pattern, days in INJURY_DAYS_LOOKUP.items():
        if pattern in text:
            return days
    return DEFAULT_INJURY_DAYS


def _parse_date_flexible(d) -> Optional[date]:
    """Parse date from various formats in injury data."""
    if pd.isna(d) or not isinstance(d, str):
        return None
    try:
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Load historical IP data
# ═══════════════════════════════════════════════════════════════════════════════

_hist_cache: Optional[pd.DataFrame] = None
_injury_cache: Optional[pd.DataFrame] = None


def _load_historical_ip() -> pd.DataFrame:
    """Load historical pitching data with IP, GS, G by pitcher-season."""
    global _hist_cache
    if _hist_cache is not None:
        return _hist_cache

    if not _HIST_PITCHING.exists():
        logger.warning(f"Historical pitching file not found: {_HIST_PITCHING}")
        _hist_cache = pd.DataFrame()
        return _hist_cache

    df = pd.read_csv(
        _HIST_PITCHING,
        low_memory=False,
        usecols=["IDfg", "Season", "Age", "IP", "GS", "G"],
    )
    df = df.rename(columns={"IDfg": "fg_id", "Season": "season"})
    df = df.dropna(subset=["fg_id", "IP"])
    df["fg_id"] = df["fg_id"].astype(int)
    df["gs_rate"] = df["GS"] / df["G"].clip(lower=1)
    _hist_cache = df
    return _hist_cache


def _load_injury_data() -> pd.DataFrame:
    """Load and parse FanGraphs injury data for pitchers."""
    global _injury_cache
    if _injury_cache is not None:
        return _injury_cache

    if not _INJURY_CSV.exists():
        logger.warning(f"Injury data not found: {_INJURY_CSV}")
        _injury_cache = pd.DataFrame()
        return _injury_cache

    df = pd.read_csv(_INJURY_CSV)
    # Keep pitcher injuries only
    df = df[df["position"].isin(["SP", "RP", "SP/RP", "RP/SP"])].copy()
    df["retro"] = pd.to_datetime(df["il_retro_date"], format="mixed", errors="coerce")
    df["ret"] = pd.to_datetime(df["return_date"], format="mixed", errors="coerce")
    df["days_out"] = (df["ret"] - df["retro"]).dt.days
    _injury_cache = df
    return _injury_cache


# ═══════════════════════════════════════════════════════════════════════════════
#  2. IP ceiling (what they'd throw if healthy all year)
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_ip_ceiling(
    fg_id: int,
    age: float,
    role: str,
    projection_year: int,
) -> float:
    """
    Estimate a pitcher's full-season healthy IP ceiling using rate-based method.

    For SPs: trailing weighted IP/GS × projected starts.
        - Only uses seasons where the pitcher primarily started (GS/G ≥ 0.5)
          to avoid inflated rates from mixed-role seasons.
        - Projected starts are capped at career_max_GS + STARTS_ESCALATION
          to prevent unproven arms from being projected for 32 starts.
    For RPs: trailing weighted IP/G × FULL_SEASON_APPEARANCES_RP (62).

    IP/GS is stable regardless of how many games a pitcher appeared in
    (r=0.491 y2y, flat by age at ~5.5).
    """
    hist = _load_historical_ip()
    if hist.empty:
        return DEFAULT_SP_CEILING if role == "SP" else DEFAULT_RP_CEILING

    # Get this pitcher's recent seasons (before the projection year)
    pitcher = hist[
        (hist["fg_id"] == fg_id) & (hist["season"] < projection_year)
    ].sort_values("season", ascending=False)

    # Filter to meaningful seasons
    pitcher = pitcher[pitcher["IP"] >= MIN_IP_THRESHOLD]

    if pitcher.empty:
        return DEFAULT_SP_CEILING if role == "SP" else DEFAULT_RP_CEILING

    # Take up to TRAILING_YEARS most recent seasons
    recent = pitcher.head(TRAILING_YEARS)

    if role == "SP":
        # --- Rate-based ceiling: IP/GS × full-season starts ---
        # Only use seasons where the pitcher primarily started (GS/G >= 0.5).
        # Mixed-role seasons inflate IP/GS because relief IP is in the
        # numerator but only starts are in the denominator.
        rates = []
        weights = []
        w_idx = 0
        for _, row in recent.iterrows():
            gs = row["GS"]
            g = max(row["G"], 1)
            if gs < MIN_GAMES_FOR_RATE or gs / g < 0.5:
                continue
            rate = row["IP"] / gs
            rates.append(rate)
            weights.append(
                TRAILING_WEIGHTS[w_idx] if w_idx < len(TRAILING_WEIGHTS) else 0.10
            )
            w_idx += 1

        if not rates:
            # No valid primarily-starter seasons — use league average rate
            ip_per_gs = LG_IP_PER_GS
        else:
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            ip_per_gs = sum(r * w for r, w in zip(rates, weights))
            # Regress 5% toward league average (small-sample stabilisation)
            ip_per_gs = ip_per_gs * 0.95 + LG_IP_PER_GS * 0.05

        # Cap projected starts based on proven workload.
        # A pitcher who has never made more than 16 MLB starts shouldn't
        # be projected for 32.  Allow career_max_GS + STARTS_ESCALATION.
        max_career_gs = int(pitcher["GS"].max()) if not pitcher.empty else 0
        projected_starts = min(FULL_SEASON_STARTS_SP,
                               max_career_gs + STARTS_ESCALATION)

        ceiling = ip_per_gs * projected_starts
        ceiling = float(np.clip(ceiling, 80.0, 220.0))


    else:  # RP
        # Only use seasons where the pitcher primarily relieved (GS/G < 0.5).
        rates = []
        weights = []
        w_idx = 0
        for _, row in recent.iterrows():
            g = row["G"]
            gs = row["GS"]
            if g < MIN_GAMES_FOR_RATE or (g > 0 and gs / g >= 0.5):
                continue
            rate = row["IP"] / g
            rates.append(rate)
            weights.append(
                TRAILING_WEIGHTS[w_idx] if w_idx < len(TRAILING_WEIGHTS) else 0.10
            )
            w_idx += 1

        if not rates:
            ip_per_g = LG_IP_PER_G_RP
        else:
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            ip_per_g = sum(r * w for r, w in zip(rates, weights))
            # Regress 5% toward league average
            ip_per_g = ip_per_g * 0.95 + LG_IP_PER_G_RP * 0.05

        # RP IP/G declines with age (from analysis: 1.45@22 → 1.0@38)
        if age > 28:
            age_adj = max(0.70, 1.0 - RP_IP_PER_G_DECLINE * (age - 28))
            ip_per_g *= age_adj

        ceiling = ip_per_g * FULL_SEASON_APPEARANCES_RP
        ceiling = float(np.clip(ceiling, 25.0, 90.0))

    return ceiling


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Injury discount
# ═══════════════════════════════════════════════════════════════════════════════

def _is_major_surgery(injury_text: str) -> Optional[int]:
    """If injury_text describes a major surgery, return recovery days; else None."""
    if not isinstance(injury_text, str):
        return None
    text = injury_text.lower().strip()
    for pattern, days in MAJOR_SURGERY_RECOVERY.items():
        if pattern in text:
            return days
    return None


def _is_end_of_season_date(d: Optional[date], year: int) -> bool:
    """True if the date is just the last week of the regular season — a
    common administrative placeholder, not a real 'return' date."""
    if d is None:
        return False
    return d.year == year and d.month >= 9 and d.day >= 25


def _major_surgery_return_date(
    fg_id: float,
    projection_year: int,
) -> Optional[date]:
    """Check ALL injury records (not just current year) for a recent major
    surgery that would still be limiting the pitcher in *projection_year*.

    Returns the computed return date if one exists, else None.
    """
    inj_df = _load_injury_data()
    if inj_df.empty:
        return None

    # Look at injuries from (projection_year - 2) through projection_year
    # Sort ASCENDING so the original surgery record (with actual dates) is
    # found before any projected follow-up records for the same surgery.
    records = inj_df[
        (inj_df["fg_id"] == fg_id)
        & (inj_df["season"].between(projection_year - 2, projection_year))
    ].sort_values("season", ascending=True)

    for _, row in records.iterrows():
        injury_text = str(row.get("injury_surgery", ""))
        recovery_days = _is_major_surgery(injury_text)
        if recovery_days is None:
            continue

        # Find the actual surgery date.
        # Use the LATER of injury_date and il_retro_date, because the
        # surgery always happens after or on the IL placement date:
        #   - injury_date can be diagnosis (before surgery) or surgery date
        #   - il_retro_date can be backdated IL placement (before surgery)
        # Taking the max ensures we use the actual procedure date.
        il_retro_str = row.get("il_retro_date")
        injury_date_str = row.get("injury_date")

        injury_date_parsed = _parse_date_flexible(injury_date_str)
        il_retro_parsed = _parse_date_flexible(il_retro_str)

        if injury_date_parsed is not None and il_retro_parsed is not None:
            surgery_date = max(injury_date_parsed, il_retro_parsed)
        elif il_retro_parsed is not None:
            surgery_date = il_retro_parsed
        elif injury_date_parsed is not None:
            surgery_date = injury_date_parsed
        else:
            continue

        expected_return = surgery_date + pd.Timedelta(days=recovery_days)
        # Ensure we have a plain date (not NaT)
        try:
            if hasattr(expected_return, 'date'):
                expected_return = expected_return.date()
            if pd.isna(expected_return):
                continue
        except (TypeError, ValueError):
            continue

        # Did they *actually* return and pitch meaningfully?
        # If return_date is end-of-season or NaN, treat as NOT returned.
        actual_return = _parse_date_flexible(row.get("return_date"))
        season_of_injury = int(row["season"])

        if actual_return is not None and not _is_end_of_season_date(actual_return, season_of_injury):
            # They really came back before end of season → no longer injured
            continue

        # Still recovering — does the expected return fall in projection_year?
        season_start = date(projection_year, 3, 27)  # spring training / opening day
        season_end = date(projection_year, 9, 30)
        if expected_return > season_start:
            return expected_return

    return None


def _current_il_days_remaining(
    fg_id: float,
    projection_year: int,
    today: Optional[date] = None,
) -> float:
    """
    If the pitcher is currently on the IL, estimate days remaining.

    Returns 0 if not currently injured.

    Priority order:
    1. Major surgery (TJS, labrum, shoulder, elbow) — compute return from
       surgery date + recovery time.  This overrides all FG metadata.
    2. "Out for {year} season" in latest_update → full season.
    3. Future return_date or eligible_to_return → use directly.
    4. Estimate from injury type + time already elapsed.
    """
    if today is None:
        today = date.today()

    # ── 1. Major surgery return (cross-season) ──────────────────────────
    surgery_return = _major_surgery_return_date(fg_id, projection_year)
    if surgery_return is not None:
        season_start = date(projection_year, 3, 27)
        season_end = date(projection_year, 9, 30)
        if surgery_return > season_end:
            # Not back this season at all
            return FULL_SEASON_DAYS
        # Days from season start (or today, whichever is later) to return
        ref = max(today, season_start)
        return max(0.0, (surgery_return - ref).days)

    # ── 2–4. Non-surgery IL check (current year only) ──────────────────
    inj_df = _load_injury_data()
    if inj_df.empty:
        return 0.0

    current = inj_df[
        (inj_df["fg_id"] == fg_id) & (inj_df["season"] == projection_year)
    ]
    if current.empty:
        return 0.0

    latest = current.iloc[-1]

    # Already returned?
    ret_date = _parse_date_flexible(latest.get("return_date"))
    if ret_date is not None and ret_date <= today:
        return 0.0

    # "Out for {year} season"
    update = str(latest.get("latest_update", "")).lower()
    if f"out for {projection_year}" in update:
        season_end = date(projection_year, 9, 30)
        return max(0.0, (season_end - today).days)

    # Future return date
    if ret_date is not None and ret_date > today:
        return (ret_date - today).days

    # Eligible to return
    etr = _parse_date_flexible(latest.get("eligible_to_return"))
    if etr is not None and etr > today:
        return (etr - today).days

    # Estimate from injury type
    injury_text = str(latest.get("injury_surgery", ""))
    est_days = _estimate_days_from_injury(injury_text)
    il_start = _parse_date_flexible(latest.get("il_retro_date"))
    if il_start is not None:
        already_out = (today - il_start).days
        return max(0, est_days - already_out)

    return est_days


def _history_injury_discount(
    fg_id: float,
    age: float,
    role: str,
    projection_year: int,
) -> float:
    """
    Estimate expected days lost to injury based on historical patterns.

    Returns expected days lost (0 = perfectly healthy projection).

    Combines:
    - Base injury rate for role + age
    - Recurrence upward adjustment if injured in prior seasons
    """
    inj_df = _load_injury_data()

    # Base rate by role
    base_rate = BASE_INJURY_RATE_SP if role == "SP" else BASE_INJURY_RATE_RP

    # Age adjustment
    if age > INJURY_AGE_PIVOT:
        base_rate += INJURY_AGE_SLOPE * (age - INJURY_AGE_PIVOT)
    base_rate = min(base_rate, 0.75)  # cap at 75%

    # Recurrence: check prior 2 seasons
    if not inj_df.empty:
        prior = inj_df[
            (inj_df["fg_id"] == fg_id)
            & (inj_df["season"].isin([projection_year - 1, projection_year - 2]))
        ]
        seasons_injured = prior["season"].nunique()
        if seasons_injured >= 2:
            base_rate += RECURRENCE_ADD_2YR
        elif seasons_injured == 1:
            base_rate += RECURRENCE_ADD_1YR
        base_rate = min(base_rate, 0.75)

    # Expected days lost
    mean_days = (
        MEAN_DAYS_LOST_IF_INJURED_SP if role == "SP"
        else MEAN_DAYS_LOST_IF_INJURED_RP
    )
    expected_days_lost = base_rate * mean_days

    # Durability credit: pitchers with recent full healthy seasons get a
    # reduced injury discount.  A workhorse with 200+ IP for 3 straight
    # years shouldn't be penalised as much as an unknown.
    hist = _load_historical_ip()
    if not hist.empty:
        threshold = DURABILITY_THRESHOLD_SP if role == "SP" else DURABILITY_THRESHOLD_RP
        recent_hist = hist[
            (hist["fg_id"] == fg_id)
            & (hist["season"].between(projection_year - 3, projection_year - 1))
            & (hist["season"] != 2020)  # exclude shortened season
        ]
        durable_count = int((recent_hist["IP"] >= threshold).sum())
        durability_factor = max(
            MIN_DURABILITY_FACTOR,
            1.0 - durable_count * DURABILITY_CREDIT_PER_SEASON,
        )
        expected_days_lost *= durability_factor

    # Recent-health adjustment: compare most recent year's actual IP
    # to the pitcher's ceiling.  Full-health seasons → lower discount;
    # injury-shortened seasons → higher discount.
    # Linear scale: achievement 1.0 → mult 0.30, achievement 0.3 → mult 2.0
    if not hist.empty:
        most_recent = hist[
            (hist["fg_id"] == fg_id)
            & (hist["season"] < projection_year)
            & (hist["season"] != 2020)
        ].sort_values("season", ascending=False)
        if not most_recent.empty:
            ceiling = _estimate_ip_ceiling(fg_id, age, role, projection_year)
            if ceiling > 0:
                achievement = min(1.0, most_recent.iloc[0]["IP"] / ceiling)
                recent_health_mult = max(
                    0.30,
                    min(2.0, 2.0 - max(0, achievement - 0.30) / 0.70 * 1.70),
                )
                expected_days_lost *= recent_health_mult

    return expected_days_lost


# ═══════════════════════════════════════════════════════════════════════════════
#  4. GS / G split
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_gs_rate(
    fg_id: int,
    role: str,
    projection_year: int,
) -> float:
    """
    Project GS rate (0.0–1.0) from historical pattern.

    Uses trailing 2-year weighted GS/G average, regressed toward role
    archetype (1.0 for SP, 0.0 for RP).
    """
    hist = _load_historical_ip()
    if hist.empty:
        return 1.0 if role == "SP" else 0.0

    recent = hist[
        (hist["fg_id"] == fg_id)
        & (hist["season"] < projection_year)
        & (hist["IP"] >= MIN_IP_THRESHOLD)
    ].sort_values("season", ascending=False).head(2)

    if recent.empty:
        return 1.0 if role == "SP" else 0.0

    # Weighted average GS rate
    if len(recent) >= 2:
        gs_rate = recent.iloc[0]["gs_rate"] * 0.65 + recent.iloc[1]["gs_rate"] * 0.35
    else:
        gs_rate = recent.iloc[0]["gs_rate"]

    # Regress toward archetype (20% regression)
    archetype = 1.0 if role == "SP" else 0.0
    gs_rate = gs_rate * 0.80 + archetype * 0.20

    return float(np.clip(gs_rate, 0.0, 1.0))


def _ip_to_gs_g(
    ip: float,
    gs_rate: float,
    age: float,
) -> tuple[int, int]:
    """Convert projected IP into GS and G counts."""
    if ip <= 0:
        return 0, 0

    if gs_rate >= 0.5:
        # Primarily starter: starts determine games
        gs = round(ip / LG_IP_PER_GS * gs_rate)
        relief_g = round(ip / LG_IP_PER_GS * (1.0 - gs_rate)) if gs_rate < 1.0 else 0
        g = gs + relief_g
    else:
        # Primarily reliever
        ip_per_g = max(0.7, RP_IP_PER_G_BASE - RP_IP_PER_G_DECLINE * max(0, age - 22))
        g = round(ip / ip_per_g)
        gs = round(g * gs_rate)

    return max(0, int(gs)), max(1, int(g))


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_playing_time(pitcher_df: pd.DataFrame, projection_year: int) -> pd.DataFrame:
    """
    Estimate IP, GS, G for every pitcher in the DataFrame.

    Parameters
    ----------
    pitcher_df : pd.DataFrame
        Must have columns: ``IDfg``, ``Age``, ``Role``.
    projection_year : int
        The year being projected.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``IP``, ``GS``, ``G`` columns added/replaced.
    """
    _load_historical_ip()  # prime caches
    _load_injury_data()

    today = date.today()
    n = len(pitcher_df)
    ips = np.zeros(n)
    gss = np.zeros(n, dtype=int)
    gs_arr = np.zeros(n, dtype=int)

    for i, (idx, row) in enumerate(pitcher_df.iterrows()):
        fg_id = int(row["IDfg"])
        age = float(row["Age"])
        role = row["Role"]

        # 1. Healthy-season IP ceiling
        ceiling = _estimate_ip_ceiling(fg_id, age, role, projection_year)

        # 2. Injury discount
        # 2a. Currently on IL?
        il_days = _current_il_days_remaining(fg_id, projection_year, today)

        if il_days >= FULL_SEASON_DAYS:
            # Out for the season
            ips[i] = 0.0
            gss[i] = 0
            gs_arr[i] = 0
            continue

        # 2b. Historical injury risk discount
        hist_days_lost = _history_injury_discount(fg_id, age, role, projection_year)

        # Apply the LARGER of the two discounts (IL or history).
        # History discount already captures chronic-injury risk, which
        # subsumes short IL stints.  For long IL stints (e.g. TJS)
        # the IL discount dominates.
        il_fraction = max(0.0, 1.0 - il_days / FULL_SEASON_DAYS) if il_days > 0 else 1.0
        hist_fraction = max(0.2, 1.0 - hist_days_lost / FULL_SEASON_DAYS)
        ip_final = ceiling * min(il_fraction, hist_fraction)

        # 3. GS / G split
        gs_rate = _estimate_gs_rate(fg_id, role, projection_year)
        gs, g = _ip_to_gs_g(ip_final, gs_rate, age)

        ips[i] = round(ip_final, 1)
        gss[i] = gs
        gs_arr[i] = g

    pitcher_df = pitcher_df.copy()
    pitcher_df["IP"] = ips
    pitcher_df["GS"] = gss
    pitcher_df["G"] = gs_arr

    # Log summary
    sp_mask = pitcher_df["Role"] == "SP"
    rp_mask = pitcher_df["Role"] == "RP"
    sp_ip = pitcher_df.loc[sp_mask, "IP"]
    rp_ip = pitcher_df.loc[rp_mask, "IP"]
    logger.info(
        f"Playing time estimated: "
        f"SP avg IP={sp_ip.mean():.1f} (median={sp_ip.median():.1f}, n={sp_mask.sum()}), "
        f"RP avg IP={rp_ip.mean():.1f} (median={rp_ip.median():.1f}, n={rp_mask.sum()})"
    )

    # Count currently-injured
    zero_ip = (pitcher_df["IP"] == 0).sum()
    if zero_ip > 0:
        logger.info(f"  {zero_ip} pitchers projected for 0 IP (season-ending injury)")

    return pitcher_df
