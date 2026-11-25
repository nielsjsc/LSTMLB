"""
Historical Values — Cot's Salary Adapter (V2)
=============================================

Converts Cot's per-player salary snapshots (`data/salary/by_year/YYYY.csv`)
into the standardized per-year contract timeline format expected by the shared
V2 valuation modules (`extend_fa_timeline`, `join_predictions_with_timeline`,
`calculate_salary_cost`, `monte_carlo_valuation`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.id_crosswalk import normalize_team_code
from value_determination.config import Config, logger

_H = Config.History


def _classify_status(service_time: float) -> str:
    if pd.isna(service_time):
        return "Unknown"
    if service_time < 3:
        return "Pre-Arb"
    if service_time < 4:
        return "Arb-1"
    if service_time < 5:
        return "Arb-2"
    if service_time < 6:
        return "Arb-3"
    return "Free Agent"


def _years_of_control_from_svc(service_time: float) -> int:
    return max(0, math.ceil(_H.SERVICE_TIME_FA - service_time))


def _map_player_role(cots_position: str, player_role: str) -> str:
    """Map Cot's position string + projection role to canonical `'Batter' | 'SP' | 'RP'`."""
    pos = str(cots_position or "").lower().strip()
    if "rhp-s" in pos or "lhp-s" in pos:
        return "SP"
    if "rhp-r" in pos or "lhp-r" in pos:
        return "RP"
    if "rhp" in pos or "lhp" in pos:
        return "RP" if str(player_role).upper() == "RP" else "SP"
    if str(player_role).upper() in ("SP", "RP"):
        return str(player_role).upper()
    return "Batter"


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if not pd.isna(f) else default
    except (TypeError, ValueError):
        return default


def build_salary_timeline(
    cots_with_idfg: pd.DataFrame,
    estimated_players: pd.DataFrame,
    snapshot_year: int,
) -> pd.DataFrame:
    """
    Build a per-year contract timeline from Cot's data + MLB-debuted pre-arb players.
    """
    rows: list[dict] = []

    # ── 1. Cot's matched players ─────────────────────────────────────────────
    for _, p in cots_with_idfg.iterrows():
        idfg = p.get("IDfg")
        if pd.isna(idfg):
            continue

        name = p.get("Name", p.get("player", ""))
        team = normalize_team_code(p.get("team", ""))
        if team == "FA":
            # Cot's rows belong to 30 MLB clubs; fallback to 'MLB' if team slug missing
            raw_t = str(p.get("team", "")).strip()
            team = raw_t.upper()[:3] if raw_t else "MLB"

        svc = _safe_float(p.get("service_time"), default=np.nan)
        yoc = int(_safe_float(p.get("years_of_control"), default=0))
        total_sal = _safe_float(p.get("total_future_salary"))
        curr_sal = _safe_float(p.get("salary"))
        proj_role = str(p.get("player_role", p.get("position_group", "Batter")))
        role = _map_player_role(p.get("position", ""), proj_role)

        if yoc <= 0:
            continue

        has_deal = (
            (total_sal > curr_sal * 1.2 and yoc > 1 and total_sal > 1_000_000)
            or (pd.notna(svc) and svc >= _H.SERVICE_TIME_FA and total_sal > 0)
        )

        year_salaries: dict[int, float] = {}
        if has_deal and total_sal > 0:
            if curr_sal > 0:
                year_salaries[0] = curr_sal
                remaining = max(0.0, total_sal - curr_sal)
                for i in range(1, yoc):
                    year_salaries[i] = remaining / (yoc - 1) if yoc > 1 else 0.0
            else:
                for i in range(yoc):
                    year_salaries[i] = total_sal / yoc
        elif curr_sal > 0:
            year_salaries[0] = curr_sal

        for yr_offset in range(yoc):
            yr = snapshot_year + yr_offset
            future_svc = (svc + yr_offset) if pd.notna(svc) else float(yr_offset)

            if has_deal:
                status = "Signed"
            else:
                status = _classify_status(future_svc)
                if status in ("Free Agent", "Unknown"):
                    break

            if yr_offset in year_salaries:
                payroll = year_salaries[yr_offset]
            elif status == "Pre-Arb":
                payroll = float(_H.HISTORICAL_MIN_SALARY.get(yr, 720_000))
            else:
                payroll = np.nan

            rows.append({
                "IDfg": int(idfg),
                "Name": name,
                "Year": yr,
                "Status": status,
                "Normalized_Status": status,
                "Payroll": payroll,
                "Years_of_Service": round(future_svc, 3) if pd.notna(future_svc) else np.nan,
                "Team": team,
                "player_role": role,
                "position_group": role,
            })

        fa_year = snapshot_year + yoc
        fa_svc = (svc + yoc) if pd.notna(svc) else float(yoc)
        rows.append({
            "IDfg": int(idfg),
            "Name": name,
            "Year": fa_year,
            "Status": "Free Agent",
            "Normalized_Status": "Free Agent",
            "Payroll": np.nan,
            "Years_of_Service": round(fa_svc, 3),
            "Team": team,
            "player_role": role,
            "position_group": role,
        })

    # ── 2. Estimated controlled MLB players (not listed in Cot's) ────────────
    for _, p in estimated_players.iterrows():
        idfg = p["IDfg"]
        name = p.get("Name", "")
        svc = _safe_float(p.get("service_time"), default=np.nan)
        yoc = int(_safe_float(p.get("years_of_control"), default=0))
        role = str(p.get("player_role", "Batter"))
        if role not in ("Batter", "SP", "RP"):
            role = "SP" if str(p.get("player_type", "")).lower() == "pitcher" else "Batter"
        team = normalize_team_code(p.get("Team", ""))
        if team == "FA":
            team = "MLB"  # Controlled pre-arb/arb MLB player whose exact historical club was multi-team/unknown

        if yoc <= 0:
            continue

        for yr_offset in range(yoc):
            yr = snapshot_year + yr_offset
            future_svc = (svc + yr_offset) if pd.notna(svc) else float(yr_offset)
            status = _classify_status(future_svc)
            if status in ("Free Agent", "Unknown"):
                break

            payroll = float(_H.HISTORICAL_MIN_SALARY.get(yr, 720_000)) if status == "Pre-Arb" else np.nan
            rows.append({
                "IDfg": int(idfg),
                "Name": name,
                "Year": yr,
                "Status": status,
                "Normalized_Status": status,
                "Payroll": payroll,
                "Years_of_Service": round(future_svc, 3) if pd.notna(future_svc) else np.nan,
                "Team": team,
                "player_role": role,
                "position_group": role,
            })

        fa_year = snapshot_year + yoc
        fa_svc = (svc + yoc) if pd.notna(svc) else float(yoc)
        rows.append({
            "IDfg": int(idfg),
            "Name": name,
            "Year": fa_year,
            "Status": "Free Agent",
            "Normalized_Status": "Free Agent",
            "Payroll": np.nan,
            "Years_of_Service": round(fa_svc, 3),
            "Team": team,
            "player_role": role,
            "position_group": role,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "IDfg", "Name", "Year", "Status", "Normalized_Status",
            "Payroll", "Years_of_Service", "Team", "player_role", "position_group",
        ])

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["IDfg", "Year"], keep="first")
    result = result.sort_values(["IDfg", "Year"]).reset_index(drop=True)
    logger.info("[%d] Cot's salary timeline: %d players, %d rows", snapshot_year, result["IDfg"].nunique(), len(result))
    return result
