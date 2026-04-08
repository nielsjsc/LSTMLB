"""
Historical Values — WAR Calculation
=====================================

Converts prediction DataFrames into WAR values using FanGraphs methodology.

Delegates all actual WAR math to ``value_determination.calculate_war`` and
dollar-conversion to ``value_determination.value_calculator`` — the
canonical single-source-of-truth implementations.  This module provides
thin DataFrame-level wrappers that call the row-level functions from
value_determination.

Usage:
    from historical_values.war import calculate_batter_war, calculate_pitcher_war
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from value_determination.config import Config, logger

# ── Import canonical WAR functions from value_determination ──────────────
from value_determination.calculate_war import (
    calculate_war_components as _vd_war_components,
    calculate_pitcher_war as _vd_pitcher_war,
    calculate_baserunning_value as _vd_bsr,
    calculate_defensive_value as _vd_def_value,
    infer_position_from_profile as _vd_infer_pos,
)
from value_determination.value_calculator import (
    calculate_war_value as _vd_war_to_dollars,
)


# ═══════════════════════════════════════════════════════════════════════════════
# BATTER WAR
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_batter_war(
    batter_df: pd.DataFrame,
    fielding_df: pd.DataFrame,
    baserunning_df: pd.DataFrame,
    position_profiles: dict | None = None,
) -> pd.DataFrame:
    """
    Compute projected WAR for every batter-year row.

    Iterates rows and delegates each to
    ``value_determination.calculate_war.calculate_war_components()``.

    Args:
        position_profiles: Optional dict mapping IDfg → {pos: fraction}.
            When provided, enables correct positional adjustments instead
            of defaulting every player to DH.
    """
    out = batter_df.copy()

    if "PA" not in out.columns:
        out["PA"] = 650.0

    wars = []
    for _, row in out.iterrows():
        war, _ = _vd_war_components(row, baserunning_df, fielding_df,
                                    position_profiles=position_profiles)
        wars.append(war)
    out["WAR"] = wars
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PITCHER WAR
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_pitcher_war(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """
    FIP-based pitcher WAR — delegates each row to
    ``value_determination.calculate_war.calculate_pitcher_war()``.
    """
    C = Config
    out = pitcher_df.copy()

    if "IP" not in out.columns:
        out["IP"] = np.where(
            out["Role"].str.upper() == "SP",
            C.DEFAULT_SP_IP,
            C.DEFAULT_RP_IP,
        )

    wars = []
    for _, row in out.iterrows():
        fip  = row["FIP"]
        ip   = row["IP"]
        team = str(row.get("Team", "")).upper().strip()
        role = str(row.get("Role", "SP")).upper().strip()
        era  = row.get("ERA", C.LG_RA9)

        rate_stats = {"ERA": era if not pd.isna(era) else C.LG_RA9}
        war, _ = _vd_pitcher_war(fip, ip, team, role, rate_stats)
        wars.append(war)

    out["WAR"] = wars
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CONVEX DOLLAR CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def war_to_dollars(war: float, year: int) -> float:
    """Convert WAR to dollar value using the convex power-law model.

    Delegates to ``value_determination.value_calculator.calculate_war_value()``.
    """
    return _vd_war_to_dollars(war, year)
