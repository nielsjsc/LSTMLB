"""
Historical Values — Cot's Salary Adapter
==========================================

Converts Cot's per-player salary data into the per-year timeline format
expected by the shared value-determination pipeline functions
(``extend_fa_timeline``, ``join_predictions_with_timeline``,
``calculate_contract_value``, ``calculate_surplus_value``, etc.).

Cot's columns:  player, team, position, service_time, years_of_control,
                 total_future_salary, salary

Pipeline needs:  IDfg, Name, Year, Status, Normalized_Status, Payroll,
                 Years_of_Service, Team, position_group

Usage:
    from historical_values.cots_adapter import build_salary_timeline
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from value_determination.config import Config, logger

# Alias for brevity — history-specific settings
_H = Config.History


# ═══════════════════════════════════════════════════════════════════════════════
# Service-time classification (mirrors surplus.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_status(service_time: float) -> str:
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
        return "Free Agent"


def _years_of_control_from_svc(service_time: float) -> int:
    return max(0, math.ceil(_H.SERVICE_TIME_FA - service_time))


# ═══════════════════════════════════════════════════════════════════════════════
# Position mapping
# ═══════════════════════════════════════════════════════════════════════════════

def _map_position_group(cots_position: str, player_type: str) -> str:
    """Map Cot's position string + player_type to pipeline position_group."""
    pos = str(cots_position).lower().strip()
    if "rhp-s" in pos or "lhp-s" in pos:
        return "SP"
    if "rhp-r" in pos or "lhp-r" in pos:
        return "RP"
    if "rhp" in pos or "lhp" in pos:
        # Ambiguous — use player_type
        return "SP" if player_type == "pitcher" else "RP"
    # Non-pitcher
    if player_type in ("pitcher",):
        return "SP"
    return "batter"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if not pd.isna(f) else default
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# Core adapter
# ═══════════════════════════════════════════════════════════════════════════════

def build_salary_timeline(
    cots_with_idfg: pd.DataFrame,
    estimated_players: pd.DataFrame,
    snapshot_year: int,
) -> pd.DataFrame:
    """
    Build a per-year salary timeline from Cot's data + estimated players.

    For players **in Cot's**: uses service_time, years_of_control,
    total_future_salary, and salary to build rows with appropriate
    Normalized_Status and Payroll.

    For players **not in Cot's** (estimated): uses service_time inferred
    from historical appearances to set status and lets the arb model
    in ``calculate_contract_value`` estimate salary.

    Args:
        cots_with_idfg: Cot's data enriched with IDfg and player_type
        estimated_players: DataFrame of players not in Cot's, with columns
            IDfg, Name, player_type, service_time, years_of_control
        snapshot_year: The calendar year for this snapshot

    Returns:
        DataFrame with columns: IDfg, Name, Year, Status,
        Normalized_Status, Payroll, Years_of_Service, Team, position_group
    """
    rows: list[dict] = []

    # ── Cot's players ────────────────────────────────────────────────────
    for _, p in cots_with_idfg.iterrows():
        idfg = p.get("IDfg")
        if pd.isna(idfg):
            continue

        name = p.get("Name", p.get("player", ""))
        team = p.get("team", "")
        svc = _safe_float(p.get("service_time"), default=np.nan)
        yoc = int(_safe_float(p.get("years_of_control"), default=0))
        total_sal = _safe_float(p.get("total_future_salary"))
        curr_sal = _safe_float(p.get("salary"))
        ptype = p.get("player_type", "batter")
        pos_group = _map_position_group(p.get("position", ""), ptype)

        if yoc <= 0:
            continue

        # Detect multi-year deal:
        #   - total_future > current_salary × 1.2 with years remaining, OR
        #   - player is past FA eligibility but still under control
        has_deal = (
            (total_sal > curr_sal * 1.2 and yoc > 1 and total_sal > 1_000_000)
            or (pd.notna(svc) and svc >= _H.SERVICE_TIME_FA and total_sal > 0)
        )

        # Distribute salary across years
        year_salaries: dict[int, float] = {}
        if has_deal and total_sal > 0:
            if curr_sal > 0:
                year_salaries[0] = curr_sal
                remaining = total_sal - curr_sal
                for i in range(1, yoc):
                    year_salaries[i] = remaining / (yoc - 1) if yoc > 1 else 0
            else:
                for i in range(yoc):
                    year_salaries[i] = total_sal / yoc
        elif curr_sal > 0:
            year_salaries[0] = curr_sal

        # Generate per-year rows
        for yr_offset in range(yoc):
            yr = snapshot_year + yr_offset
            future_svc = (svc + yr_offset) if pd.notna(svc) else yr_offset

            if has_deal:
                status = "Signed"
            else:
                status = _classify_status(future_svc)
                if status in ("Free Agent", "Unknown"):
                    break

            # Payroll
            if yr_offset in year_salaries:
                payroll = year_salaries[yr_offset]
            elif status == "Pre-Arb":
                payroll = _H.HISTORICAL_MIN_SALARY.get(yr, 720_000)
            else:
                payroll = np.nan  # arb → pipeline estimates from Base_Value

            rows.append({
                "IDfg": int(idfg),
                "Name": name,
                "Year": yr,
                "Status": status,
                "Normalized_Status": status,
                "Payroll": payroll,
                "Years_of_Service": round(future_svc, 3) if pd.notna(future_svc) else np.nan,
                "Team": team,
                "position_group": pos_group,
            })

        # Free Agent row at end of control
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
            "position_group": pos_group,
        })

    # ── Estimated players (not in Cot's) ──────────────────────────────────
    for _, p in estimated_players.iterrows():
        idfg = p["IDfg"]
        name = p.get("Name", "")
        svc = _safe_float(p.get("service_time"), default=np.nan)
        yoc = int(_safe_float(p.get("years_of_control"), default=0))
        ptype = p.get("player_type", "batter")
        pos_group = "SP" if ptype == "pitcher" else "batter"

        if yoc <= 0:
            continue

        for yr_offset in range(yoc):
            yr = snapshot_year + yr_offset
            future_svc = (svc + yr_offset) if pd.notna(svc) else yr_offset
            status = _classify_status(future_svc)
            if status in ("Free Agent", "Unknown"):
                break

            if status == "Pre-Arb":
                payroll = _H.HISTORICAL_MIN_SALARY.get(yr, 720_000)
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
                "Team": "",
                "position_group": pos_group,
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
            "Team": "",
            "position_group": pos_group,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "IDfg", "Name", "Year", "Status", "Normalized_Status",
            "Payroll", "Years_of_Service", "Team", "position_group",
        ])

    result = pd.DataFrame(rows)
    # Deduplicate: if a player appears in both Cot's and estimated, keep Cot's
    result = result.drop_duplicates(subset=["IDfg", "Year"], keep="first")
    result = result.sort_values(["IDfg", "Year"]).reset_index(drop=True)

    logger.info(
        f"[{snapshot_year}]  salary timeline: "
        f"{result['IDfg'].nunique()} players, {len(result)} rows"
    )
    return result
