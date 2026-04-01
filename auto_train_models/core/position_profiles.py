"""
Position Profiles Module
========================

Builds multi-position profiles for each player from historical MLB fielding data.
Uses the most recent season of actual fielding data as the source of truth.

A position profile is a dictionary mapping positions to their playing-time fractions,
e.g. {'3B': 0.75, '2B': 0.18, 'DH': 0.07}

DH games are inferred from: batting games - sum(fielding games started).

This module is used UPSTREAM of predictions to determine what positions a player
should be projected at, and DOWNSTREAM in WAR calculation to weight positional
adjustments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Valid defensive positions (excludes P and DH)
DEFENSIVE_POSITIONS = {'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF'}

# Position group mapping for defense model routing
POSITION_TO_GROUP = {
    'C': 'catcher',
    '1B': 'infield', '2B': 'infield', '3B': 'infield', 'SS': 'infield',
    'LF': 'outfield', 'CF': 'outfield', 'RF': 'outfield',
}

# ── Position-to-position FRV transfer map ───────────────────────────────────
#
# When a player moves to a new position and has no fielding history there,
# we estimate their FRV at the new position from their FRV at a known position.
#
# Transfer factor: expected_new_FRV ≈ old_FRV × factor
#
# Factors are based on the defensive spectrum (C > SS > CF > 2B ~ 3B > RF ~ LF > 1B)
# and skill transferability within / across position groups.
#
# Within-group transfers have higher factors (skills overlap more).
# Cross-group transfers are lower (different skill sets: range vs. arm vs. footwork).

FRV_TRANSFER_MAP: Dict[Tuple[str, str], float] = {
    # ── Infield → Infield ──
    ('SS', '3B'):  0.70,   # SS range/arm translates well to 3B
    ('SS', '2B'):  0.75,   # SS range is top-tier; 2B is similar but easier pivot
    ('SS', '1B'):  0.30,   # Very different skill set at 1B
    ('3B', 'SS'):  0.55,   # 3B→SS is harder (less range usually)
    ('3B', '2B'):  0.60,   # Moderate overlap
    ('3B', '1B'):  0.35,   # 3B arm helps but 1B is scooping/stretch
    ('2B', 'SS'):  0.55,   # 2B→SS is a step up in difficulty
    ('2B', '3B'):  0.60,   # Moderate overlap
    ('2B', '1B'):  0.30,   # Different skill set
    ('1B', '3B'):  0.25,   # 1B rarely has the range/arm for 3B
    ('1B', '2B'):  0.20,   # Very unlikely to translate
    ('1B', 'SS'):  0.15,   # Almost no transferability

    # ── Outfield → Outfield ──
    ('CF', 'RF'):  0.80,   # CF range/speed translates well to corners
    ('CF', 'LF'):  0.80,   # Same — usually an upgrade
    ('RF', 'CF'):  0.50,   # RF→CF requires more range
    ('RF', 'LF'):  0.85,   # Very similar, arm matters less in LF
    ('LF', 'CF'):  0.45,   # LF→CF is a big step up
    ('LF', 'RF'):  0.75,   # Similar, RF values arm a bit more

    # ── Infield → Outfield (cross-group) ──
    ('SS', 'CF'):  0.30,   ('SS', 'RF'):  0.25,   ('SS', 'LF'):  0.25,
    ('2B', 'CF'):  0.25,   ('2B', 'RF'):  0.20,   ('2B', 'LF'):  0.20,
    ('3B', 'CF'):  0.20,   ('3B', 'RF'):  0.20,   ('3B', 'LF'):  0.20,
    ('1B', 'LF'):  0.15,   ('1B', 'CF'):  0.10,   ('1B', 'RF'):  0.15,

    # ── Outfield → Infield (cross-group) ──
    ('CF', 'SS'):  0.20,   ('CF', '2B'):  0.25,   ('CF', '3B'):  0.20,   ('CF', '1B'):  0.15,
    ('RF', 'SS'):  0.15,   ('RF', '2B'):  0.20,   ('RF', '3B'):  0.20,   ('RF', '1B'):  0.15,
    ('LF', 'SS'):  0.10,   ('LF', '2B'):  0.15,   ('LF', '3B'):  0.15,   ('LF', '1B'):  0.15,

    # ── Catcher → others ──
    ('C', 'SS'):   0.20,   ('C', '2B'):   0.20,   ('C', '3B'):   0.25,   ('C', '1B'):   0.30,
    ('C', 'CF'):   0.15,   ('C', 'RF'):   0.15,   ('C', 'LF'):   0.15,

    # ── Others → Catcher (very rare, very low transferability) ──
    ('1B', 'C'):   0.10,   ('2B', 'C'):   0.15,   ('3B', 'C'):   0.15,   ('SS', 'C'):   0.15,
    ('LF', 'C'):   0.10,   ('CF', 'C'):   0.10,   ('RF', 'C'):   0.10,
}


def get_frv_transfer_factor(from_pos: str, to_pos: str) -> float:
    """
    Get the FRV transfer factor for a position change.
    
    Returns the multiplicative factor to estimate FRV at to_pos
    given a known FRV at from_pos.
    
    Falls back to 0.20 for any unmapped pair.
    """
    if from_pos == to_pos:
        return 1.0
    return FRV_TRANSFER_MAP.get((from_pos, to_pos), 0.20)


def estimate_missing_frv(
    position_profile: Dict[str, float],
    existing_predictions: Dict[str, float],
) -> Dict[str, float]:
    """
    Estimate FRV for positions in the profile that have no LSTM prediction.
    
    For each defensive position in the profile that lacks a prediction,
    finds the best source position (one with a prediction and the highest
    transfer factor) and applies the transfer.
    
    Args:
        position_profile: {pos: fraction} — the player's current profile
        existing_predictions: {pos: frv} — FRV values from LSTM predictions
        
    Returns:
        Dict of {pos: estimated_frv} for the MISSING positions only
    """
    defensive = get_defensive_positions(position_profile)
    missing = {p for p in defensive if p not in existing_predictions}
    
    if not missing or not existing_predictions:
        return {}
    
    estimated = {}
    for new_pos in missing:
        best_factor = 0.0
        best_frv = 0.0
        for src_pos, src_frv in existing_predictions.items():
            factor = get_frv_transfer_factor(src_pos, new_pos)
            if factor > best_factor:
                best_factor = factor
                best_frv = src_frv * factor
        estimated[new_pos] = best_frv
    
    return estimated


def build_position_profiles(
    fielding_df: pd.DataFrame,
    batting_df: pd.DataFrame,
    player_ids: list[int],
    cutoff_year: int = 2025,
    min_games: int = 10,
) -> Dict[int, Dict[str, float]]:
    """
    Build position profiles for all players from historical fielding data.
    
    For each player, finds their most recent season with sufficient playing time
    and calculates the fraction of games started at each position (including DH).
    
    Args:
        fielding_df: Historical fielding data (mlb_fielding_data_2000_2025_with_statcast.csv)
        batting_df: Historical batting data (for total games to infer DH)
        player_ids: List of IDfg values to build profiles for
        cutoff_year: Most recent year to consider
        min_games: Minimum total games in a season to use it for the profile
        
    Returns:
        Dict mapping IDfg -> {position: fraction} e.g. {27815: {'3B': 0.75, '2B': 0.18, 'DH': 0.07}}
    """
    # Filter to relevant players and years
    fld = fielding_df[
        (fielding_df['IDfg'].isin(player_ids)) &
        (fielding_df['Season'] <= cutoff_year) &
        (fielding_df['Pos'].isin(DEFENSIVE_POSITIONS))
    ].copy()
    
    bat = batting_df[
        (batting_df['IDfg'].isin(player_ids)) &
        (batting_df['Season'] <= cutoff_year)
    ][['IDfg', 'Season', 'G']].copy()
    
    profiles = {}
    
    for pid in player_ids:
        profile = _build_single_profile(fld, bat, pid, cutoff_year, min_games)
        if profile:
            profiles[pid] = profile
    
    return profiles


def _build_single_profile(
    fld: pd.DataFrame,
    bat: pd.DataFrame,
    player_id: int,
    cutoff_year: int,
    min_games: int,
) -> Optional[Dict[str, float]]:
    """
    Build position profile for a single player.
    
    Looks at their most recent season with >= min_games total batting games.
    Falls back to earlier seasons if the most recent one is too small (injury year).
    """
    player_bat = bat[bat['IDfg'] == player_id].sort_values('Season', ascending=False)
    player_fld = fld[fld['IDfg'] == player_id]
    
    if player_bat.empty:
        return None
    
    # Try the most recent season first, then fall back to earlier ones
    for _, bat_row in player_bat.iterrows():
        season = int(bat_row['Season'])
        total_games = int(bat_row['G'])
        
        if total_games < min_games:
            continue
        
        season_fld = player_fld[player_fld['Season'] == season]
        
        # Build profile from this season's fielding GS
        profile = {}
        total_field_gs = 0
        
        for _, fld_row in season_fld.iterrows():
            pos = fld_row['Pos']
            gs = int(fld_row.get('GS', fld_row.get('G', 0)))
            if gs > 0:
                profile[pos] = gs
                total_field_gs += gs
        
        # Infer DH games
        dh_games = max(0, total_games - total_field_gs)
        if dh_games > 0:
            profile['DH'] = dh_games
        
        if not profile:
            continue
        
        # Convert to fractions
        total = sum(profile.values())
        profile = {pos: games / total for pos, games in profile.items()}
        
        return profile
    
    # No usable season found — check if they have any fielding data at all
    # to at least get a position, even from a short stint
    if not player_fld.empty:
        latest = player_fld.sort_values(['Season', 'Inn'], ascending=[False, False])
        pos = latest.iloc[0]['Pos']
        return {pos: 1.0}
    
    return None


def get_primary_position(profile: Dict[str, float]) -> str:
    """Get the position with the highest fraction (excluding DH if possible)."""
    if not profile:
        return 'DH'
    
    # Prefer a defensive position over DH
    defensive = {p: f for p, f in profile.items() if p != 'DH'}
    if defensive:
        return max(defensive, key=defensive.get)
    
    return 'DH'


def get_display_position(profile: Dict[str, float], dh_threshold: float = 0.80) -> str:
    """
    Get a display-friendly position string.
    
    If a player plays 80%+ DH, they're labeled DH.
    Otherwise, primary defensive position.
    """
    if not profile:
        return 'DH'
    
    dh_frac = profile.get('DH', 0.0)
    if dh_frac >= dh_threshold:
        return 'DH'
    
    return get_primary_position(profile)


def get_weighted_positional_adjustment(
    profile: Dict[str, float],
    positional_adjustments: Dict[str, float],
    games: int = 150,
) -> float:
    """
    Calculate a weighted positional adjustment based on multi-position profile.
    
    Instead of assigning one position's adjustment, weights by actual playing time.
    E.g., a player who is 75% 3B / 25% DH gets:
      0.75 * 2.5 + 0.25 * (-17.5) = -2.5 runs per 162 games
    
    Args:
        profile: Position fractions {pos: fraction}
        positional_adjustments: {pos: runs_per_162}
        games: Number of games to scale to
        
    Returns:
        Weighted positional adjustment in runs, scaled to games played
    """
    if not profile:
        return positional_adjustments.get('DH', -17.5) * (games / 162.0)
    
    weighted_adj = 0.0
    for pos, fraction in profile.items():
        adj = positional_adjustments.get(pos, 0.0)
        weighted_adj += fraction * adj
    
    return weighted_adj * (games / 162.0)


def get_defensive_positions(profile: Dict[str, float]) -> Dict[str, float]:
    """Get only the defensive (non-DH) positions from a profile."""
    return {p: f for p, f in profile.items() if p in DEFENSIVE_POSITIONS}


def get_position_groups(profile: Dict[str, float]) -> Dict[str, float]:
    """
    Get the position groups this player needs fielding predictions for,
    with the fraction of defensive time in each group.
    
    E.g., {'3B': 0.75, '2B': 0.18, 'DH': 0.07} ->
          {'infield': 0.93}  (all infield time combined)
    """
    groups = {}
    for pos, frac in profile.items():
        group = POSITION_TO_GROUP.get(pos)
        if group:
            groups[group] = groups.get(group, 0.0) + frac
    return groups


def load_fielding_history(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the historical fielding data CSV."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / 'data' / 'historic_mlb'
    
    path = data_dir / 'mlb_fielding_data_2000_2025_with_statcast.csv'
    return pd.read_csv(path)


def load_batting_for_games(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load batting data (just IDfg, Season, G columns for DH inference)."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / 'data' / 'historic_mlb'
    
    path = data_dir / 'mlb_batting_data_1950_2025.csv'
    df = pd.read_csv(path, usecols=['IDfg', 'Season', 'G'], low_memory=False)
    return df
