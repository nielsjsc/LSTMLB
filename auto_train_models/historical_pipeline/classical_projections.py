#!/usr/bin/env python3
"""
Classical Projections Generator
-------------------------------
Generates simple statistical projections for fielding (DRS/150, UZR/150) and
baserunning (BsR) for cutoff years where Statcast data is unavailable or
insufficient for the LSTM models.

The LSTM fielding/baserunning models require Statcast features (sc_total_runs/150,
sc_baserunning_runner_runs_tot_rate, etc.) which are only available from 2016+.
Since those models use seq_length=3, they realistically need cutoff_year >= 2018.

For earlier cutoff years, this module produces projection CSVs using traditional
metrics that the surplus calculator already supports as fallbacks:
  - Fielding:    UZR/150 (non-catchers) and DRS/150 (catchers)
  - Baserunning: BsR_rate (FanGraphs BsR per PA, scaled)

The aging model is a simple piecewise-linear decline calibrated to published
baseball aging research.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("classical_projections")

ROOT_DIR = Path(__file__).resolve().parents[2]


# ── Aging curves ───────────────────────────────────────────────────────────────
# Defense: peak ~26, gradual decline 27-31, accelerating decline 32+
_DEF_PEAK_AGE = 26
_DEF_DECLINE_PER_YEAR_EARLY = 0.6   # runs/year, ages 27-31
_DEF_DECLINE_PER_YEAR_LATE  = 1.2   # runs/year, ages 32+

# Baserunning: peak ~25, decline accelerates with age
_BSR_PEAK_AGE = 25
_BSR_DECLINE_PER_YEAR_EARLY = 0.4   # runs/year, ages 26-30
_BSR_DECLINE_PER_YEAR_LATE  = 0.8   # runs/year, ages 31+

# SB/CS rates: speed declines similarly
_SPD_PEAK_AGE = 25
_SPD_DECLINE_PCT_EARLY = 0.03       # 3% decline/year, ages 26-30
_SPD_DECLINE_PCT_LATE  = 0.06       # 6% decline/year, ages 31+

FUTURE_YEARS = 15
HISTORY_WINDOW = 3        # years of history for weighted average
MIN_INNINGS_FIELDING = 50
MIN_PA_BASERUNNING = 100


def _age_adjustment_defense(current_age: float, years_forward: int) -> float:
    """Cumulative defensive aging adjustment (negative = decline)."""
    adj = 0.0
    for y in range(1, years_forward + 1):
        age = current_age + y
        if age > 31:
            adj -= _DEF_DECLINE_PER_YEAR_LATE
        elif age > _DEF_PEAK_AGE:
            adj -= _DEF_DECLINE_PER_YEAR_EARLY
    return adj


def _age_adjustment_bsr(current_age: float, years_forward: int) -> float:
    """Cumulative baserunning aging adjustment (negative = decline)."""
    adj = 0.0
    for y in range(1, years_forward + 1):
        age = current_age + y
        if age > 30:
            adj -= _BSR_DECLINE_PER_YEAR_LATE
        elif age > _BSR_PEAK_AGE:
            adj -= _BSR_DECLINE_PER_YEAR_EARLY
    return adj


def _age_factor_speed(current_age: float, years_forward: int) -> float:
    """Cumulative multiplicative speed factor (< 1 = slower)."""
    factor = 1.0
    for y in range(1, years_forward + 1):
        age = current_age + y
        if age > 30:
            factor *= (1.0 - _SPD_DECLINE_PCT_LATE)
        elif age > _SPD_PEAK_AGE:
            factor *= (1.0 - _SPD_DECLINE_PCT_EARLY)
    return factor


def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    """Recency-weighted average. weights should increase with recency."""
    mask = values.notna() & weights.notna()
    v, w = values[mask], weights[mask]
    if len(v) == 0:
        return np.nan
    return float(np.average(v, weights=w))


# ── Fielding Projections ──────────────────────────────────────────────────────

def generate_classical_fielding(cutoff_year: int, output_file: str) -> pd.DataFrame:
    """
    Generate fielding projections using DRS/150 (catchers) and UZR/150 (non-catchers).
    
    For each player active near the cutoff year, computes a recency-weighted
    average of their defensive rate stat and projects forward with an aging curve.
    """
    data_file = ROOT_DIR / "data" / "historic_mlb" / "mlb_fielding_data_2000_2025_with_statcast.csv"
    logger.info(f"Generating classical fielding projections for cutoff {cutoff_year}")
    
    df = pd.read_csv(data_file)
    df = df[df["Season"] <= cutoff_year].copy()
    
    # Compute DRS/150 where DRS and Inn are available
    mask = (df["Inn"] > 0) & df["DRS"].notna()
    df.loc[mask, "DRS/150"] = df.loc[mask, "DRS"] * 1350.0 / df.loc[mask, "Inn"]
    
    # Determine position group for each position
    infield_pos = {"1B", "2B", "3B", "SS"}
    outfield_pos = {"LF", "CF", "RF"}
    
    def classify_pos(pos):
        if pos == "C":
            return "catcher"
        elif pos in infield_pos:
            return "infield"
        elif pos in outfield_pos:
            return "outfield"
        return None
    
    df["Position_Group"] = df["Pos"].apply(classify_pos)
    df = df[df["Position_Group"].notna()].copy()
    
    # Filter to sufficient innings
    df = df[df["Inn"] >= MIN_INNINGS_FIELDING]
    
    # For non-catchers use UZR/150, for catchers use DRS/150
    df["fld_rate"] = np.where(
        df["Position_Group"] == "catcher",
        df["DRS/150"],
        df["UZR/150"]
    )
    df = df[df["fld_rate"].notna()]
    
    # Find players active in the cutoff window
    window_start = cutoff_year - HISTORY_WINDOW + 1
    active_df = df[df["Season"] >= window_start]
    
    # For each player-position, compute weighted average
    all_projections = []
    
    for (idfg, pos), group in active_df.groupby(["IDfg", "Pos"]):
        group = group.sort_values("Season")
        
        # Recency weights: most recent season gets highest weight
        weights = group["Season"] - group["Season"].min() + 1.0
        base_rate = _weighted_avg(group["fld_rate"], weights)
        if np.isnan(base_rate):
            continue
        
        # Player metadata from most recent season
        latest = group.iloc[-1]
        name = latest["Name"]
        current_age = latest["Age"]
        pos_group = latest["Position_Group"]
        
        # Project forward
        for yr_offset in range(1, FUTURE_YEARS + 1):
            proj_year = cutoff_year + yr_offset
            proj_age = current_age + yr_offset
            
            # Stop projecting past age 42
            if proj_age > 42:
                break
            
            aging = _age_adjustment_defense(current_age, yr_offset)
            proj_rate = base_rate + aging
            
            row = {
                "Name": name,
                "Age": proj_age,
                "Year": proj_year,
                "IDfg": idfg,
                "Pos": pos,
                "Position_Group": pos_group,
            }
            
            # Output the appropriate column based on position
            if pos_group == "catcher":
                row["UZR/150"] = np.nan
                row["DRS/150"] = proj_rate
            else:
                row["UZR/150"] = proj_rate
                row["DRS/150"] = np.nan
            
            all_projections.append(row)
    
    if not all_projections:
        logger.warning("No classical fielding projections generated")
        result = pd.DataFrame(columns=["Name", "Age", "Year", "IDfg", "Pos", "Position_Group", "UZR/150", "DRS/150"])
    else:
        result = pd.DataFrame(all_projections)
    
    result.to_csv(output_file, index=False)
    n_players = result["IDfg"].nunique() if len(result) > 0 else 0
    logger.info(f"Saved {len(result)} classical fielding projections ({n_players} players) to {output_file}")
    return result


# ── Baserunning Projections ───────────────────────────────────────────────────

def generate_classical_baserunning(cutoff_year: int, output_file: str) -> pd.DataFrame:
    """
    Generate baserunning projections using FanGraphs BsR, SB, and CS rates.
    
    Computes recency-weighted rates and projects forward with aging curves.
    """
    data_file = ROOT_DIR / "data" / "historic_mlb" / "mlb_batting_data_1950_2025_with_statcast.csv"
    logger.info(f"Generating classical baserunning projections for cutoff {cutoff_year}")
    
    df = pd.read_csv(data_file, usecols=["IDfg", "Name", "Season", "Age", "PA", "BsR", "SB", "CS"])
    df = df[df["Season"] <= cutoff_year].copy()
    df = df[df["PA"] >= MIN_PA_BASERUNNING]
    df = df[df["BsR"].notna()]
    
    # Compute per-PA rates (multiply by 600 to match the existing prediction scale)
    df["BsR_rate"] = df["BsR"] / df["PA"] * 600.0
    df["SB_rate"]  = df["SB"]  / df["PA"] * 600.0
    df["CS_rate"]  = df["CS"]  / df["PA"] * 600.0
    
    # Find players active in window
    window_start = cutoff_year - HISTORY_WINDOW + 1
    active_df = df[df["Season"] >= window_start]
    
    all_projections = []
    
    for idfg, group in active_df.groupby("IDfg"):
        group = group.sort_values("Season")
        
        weights = group["Season"] - group["Season"].min() + 1.0
        
        base_bsr = _weighted_avg(group["BsR_rate"], weights)
        base_sb  = _weighted_avg(group["SB_rate"], weights)
        base_cs  = _weighted_avg(group["CS_rate"], weights)
        
        if np.isnan(base_bsr):
            continue
        
        latest = group.iloc[-1]
        name = latest["Name"]
        current_age = latest["Age"]
        
        for yr_offset in range(1, FUTURE_YEARS + 1):
            proj_year = cutoff_year + yr_offset
            proj_age  = current_age + yr_offset
            
            if proj_age > 42:
                break
            
            bsr_aging = _age_adjustment_bsr(current_age, yr_offset)
            speed_factor = _age_factor_speed(current_age, yr_offset)
            
            all_projections.append({
                "Name": name,
                "IDfg": idfg,
                "Year": proj_year,
                "Age": proj_age,
                "BsR_rate": base_bsr + bsr_aging,
                "SB_rate": base_sb * speed_factor if not np.isnan(base_sb) else 0.0,
                "CS_rate": base_cs * speed_factor if not np.isnan(base_cs) else 0.0,
            })
    
    if not all_projections:
        logger.warning("No classical baserunning projections generated")
        result = pd.DataFrame(columns=["Name", "IDfg", "Year", "Age", "BsR_rate", "SB_rate", "CS_rate"])
    else:
        result = pd.DataFrame(all_projections)
    
    result.to_csv(output_file, index=False)
    n_players = result["IDfg"].nunique() if len(result) > 0 else 0
    logger.info(f"Saved {len(result)} classical baserunning projections ({n_players} players) to {output_file}")
    return result
