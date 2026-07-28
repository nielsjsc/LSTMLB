#!/usr/bin/env python3
"""
WAR Calculation Module
======================

This module provides all WAR (Wins Above Replacement) calculation functions for both
batters and pitchers. It combines projected stats with baserunning and fielding data
to calculate comprehensive WAR values using FanGraphs methodology.

This is the SINGLE SOURCE OF TRUTH for WAR calculations in the value_determination module.
Do not duplicate these functions elsewhere.

Functions:
    - calculate_pitcher_war(): FIP-based WAR for pitchers
    - calculate_war_components(): Full WAR breakdown for batters
    - calculate_woba(): Calculate wOBA from counting stats (2025 weights)
    - calculate_woba_from_predictions(): Calculate wOBA from batter prediction DataFrame
    - calculate_wrc_plus(): wRC+ calculation with park factors
    - load_player_orgs(): Load team assignments from roster data

Usage:
    from value_determination.calculate_war import (
        calculate_pitcher_war, calculate_war_components, 
        calculate_woba_from_predictions, load_player_orgs
    )
    
    # Pitcher WAR
    war, components = calculate_pitcher_war(fip=3.50, ip=180, team='NYY', role='SP')
    
    # Batter WAR with calculated wOBA from predictions
    batter_data = calculate_woba_from_predictions(batter_predictions_df)
    war, components = calculate_war_components(player_row, baserunning_df, fielding_df, position_profiles)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Import from central config - SINGLE SOURCE OF TRUTH for constants
from .config import (
    Config, logger,
    # Backward compatibility exports
    # NOTE: BALLPARK_FACTORS is intentionally NOT imported for lookups anymore.
    # It used to be a second, independently-maintained park-factor table that
    # drifted from core.park_factors.PARK_FACTORS_5YR (different values and no
    # team-code aliasing for SFG/SDP/KCR/TBR/WSN), which caused the
    # neutralize -> reapply round trip to over/under-correct. All park-factor
    # lookups in this module now go through core.park_factors.get_park_factor()
    # so there is exactly one source of truth.
    WOBA_SCALE, RPA, LG_WOBA, RPW, LG_FIP
)

# Single source of truth for park factors (see note above)
from core.park_factors import get_park_factor, get_woba_residual_factor

from core.position_profiles import (
    get_primary_position, get_display_position,
    get_weighted_positional_adjustment, get_defensive_positions,
    POSITION_TO_GROUP, DEFENSIVE_POSITIONS,
)

# Additional constants from config
LG_PA = Config.WAR.LG_PA
LG_RUNS_PER_PA = Config.WAR.LG_RUNS_PER_PA
LG_WRC_PER_PA = Config.WAR.LG_WRC_PER_PA
POSITIONAL_ADJUSTMENTS = Config.WAR.POSITIONAL_ADJUSTMENTS
REPLACEMENT_LEVEL_RUNS_200IP = Config.WAR.REPLACEMENT_LEVEL_RUNS_200IP
TEAM_ABBREVIATIONS = Config.WAR.TEAM_ABBREVIATIONS
LG_RA9 = Config.WAR.LG_RA9
DEFAULT_IP_PER_START = Config.WAR.DEFAULT_IP_PER_START
DEFAULT_IP_PER_APPEARANCE_RP = Config.WAR.DEFAULT_IP_PER_APPEARANCE_RP


def _dynamic_pitcher_rpw(era: float, ip: float, role: str = 'SP') -> float:
    """
    Compute pitcher-specific Runs Per Win (FanGraphs methodology).

    Unlike batters, a pitcher directly affects the run environment while on
    the mound.  An ace suppresses scoring, making each run saved worth MORE
    wins.  This dynamic RPW accounts for that by blending the league run
    environment (when the pitcher is NOT on the mound) with the pitcher's
    own run rate (when they ARE pitching).

    Formula (FanGraphs):
        RPW = ((18 - ip_per_game) * lgRA9 + ip_per_game * RA9) / 18 + 2) * 1.5

    Where:
        - ip_per_game: average IP per appearance (SP ≈ 5.75, RP ≈ 1.0)
        - lgRA9: league average runs per 9 innings (~4.50)
        - RA9: pitcher's own run rate (approximated by ERA)
        - 18 = total half-innings in a regulation game
        - +2 accounts for the non-linear Pythagorean win% effect
        - ×1.5 converts from runs-per-game to runs-per-win

    Examples (SP at 180 IP):
        ERA=2.70 → RPW≈8.88  (ace: each run worth more)
        ERA=4.50 → RPW≈9.75  (average: close to league RPW)
        ERA=5.50 → RPW≈10.07 (bad: runs worth less because game is already lost)

    Args:
        era: Pitcher's projected ERA (used as proxy for RA9)
        ip: Total innings pitched (used only for fallback; ip_per_game from defaults)
        role: 'SP' or 'RP'

    Returns:
        Pitcher-specific runs per win value
    """
    if role == 'RP':
        ip_per_game = DEFAULT_IP_PER_APPEARANCE_RP
    else:
        ip_per_game = DEFAULT_IP_PER_START

    # Use ERA as proxy for RA9 (unearned runs add ~0.15;
    # close enough for projection purposes)
    ra9 = era if era > 0 else LG_RA9

    # FanGraphs dynamic RPW formula
    game_ra9 = ((18 - ip_per_game) * LG_RA9 + ip_per_game * ra9) / 18
    rpw = (game_ra9 + 2) * 1.5

    # Safety floor: RPW should never be absurdly low
    return max(rpw, 5.0)


def load_player_orgs(data_dir: Path = None) -> pd.DataFrame:
    """
    Load player organizations from current rosters file.
    
    Returns DataFrame with IDfg (fg_id), mlbam_id, and their current team.
    This function provides the link between player IDs and team assignments
    needed for park factor adjustments.
    
    Args:
        data_dir: Path to data directory. If None, uses Config.Paths.DATA_DIR
        
    Returns:
        DataFrame with columns: IDfg, mlbam_id, Team, player_name
        
    Note:
        TODO: Transition to using mlbam_id as primary identifier
    """
    if data_dir is None:
        data_dir = Config.Paths.DATA_DIR
    
    # Use roster file from config
    roster_file = Config.Paths.ROSTER_FILE
    
    # Fallback paths if config path doesn't exist
    if not roster_file.exists():
        roster_file = data_dir / "active_roster" / "current_rosters.csv"
    if not roster_file.exists():
        roster_file = data_dir / "current_rosters.csv"
    
    if not roster_file.exists():
        logger.warning(f"Current rosters file not found at: {roster_file}")
        logger.warning("WAR calculations will use park factor of 1.0 for all players")
        return pd.DataFrame(columns=['IDfg', 'mlbam_id', 'Team', 'player_name'])
    
    # Load roster data
    roster_df = pd.read_csv(roster_file)
    
    # Filter out players with no fg_id (fg_id == -1.0 means no mapping found)
    roster_df = roster_df[roster_df['fg_id'].notna() & (roster_df['fg_id'] != -1.0)]
    
    # Use team abbreviation mapping from config
    roster_df['Team'] = roster_df['team_name'].map(TEAM_ABBREVIATIONS)
    
    # Select columns - include mlbam_id for future ID migration
    # TODO: Make mlbam_id the primary identifier
    org_data = roster_df[['fg_id', 'mlbam_id', 'Team', 'player_name']].copy()
    org_data = org_data.rename(columns={'fg_id': 'IDfg'})
    
    # Convert IDfg to int to match prediction data format
    org_data['IDfg'] = pd.to_numeric(org_data['IDfg'], errors='coerce').dropna().astype(int)
    
    logger.info(f"Loaded {len(org_data)} player organizations from current rosters")
    
    return org_data

def calculate_woba(ab: float, bb: float, ibb: float, hbp: float, sf: float,
                  singles: float, doubles: float, triples: float, hr: float,
                  pa: float = None,
                  wbb: float = 0.691, whbp: float = 0.722, w1b: float = 0.882,
                  w2b: float = 1.252, w3b: float = 1.584, whr: float = 2.037) -> float:
    """
    Calculate wOBA from counting stats using modified formula.
    
    Formula: wOBA = (wBB*BB + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR) / PA
    
    Note: This uses ALL walks (including IBB), not just unintentional walks.
    
    Args:
        ab: At bats
        bb: Total walks (includes IBB)
        ibb: Intentional walks (not used in this formula)
        hbp: Hit by pitch
        sf: Sacrifice flies
        singles: Singles (1B)
        doubles: Doubles (2B)
        triples: Triples (3B)
        hr: Home runs
        pa: Plate appearances (if None, calculated from AB+BB+HBP+SF)
        wbb, whbp, w1b, w2b, w3b, whr: 2025 wOBA weights
    
    Returns:
        wOBA value
    """
    # Numerator: weighted sum of positive offensive events (using ALL BB)
    numerator = (wbb * bb + whbp * hbp + w1b * singles + 
                 w2b * doubles + w3b * triples + whr * hr)
    
    # Denominator: plate appearances
    if pa is None:
        pa = ab + bb + hbp + sf
    
    # Avoid division by zero
    if pa == 0:
        return 0.0
    
    return numerator / pa


def calculate_wrc_plus(woba: float, team: str, pa: float, 
                      lg_runs_per_pa: float = LG_RUNS_PER_PA,
                      lg_wrc_per_pa: float = LG_WRC_PER_PA,
                      lg_woba: float = LG_WOBA,
                      woba_scale: float = WOBA_SCALE) -> float:
    """
    Calculate wRC+ using the proper formula with park factors.
    Exactly matches the calculation from the batter notebook.
    """
    # Calculate wRAA per PA
    wraa_per_pa = (woba - lg_woba) / woba_scale
    
    # Get park factor (defaults to 1.0 / neutral if no team/FA)
    # NOTE: uses the wOBA-scale RESIDUAL factor, not the runs-scale
    # PARK_FACTORS_5YR. `woba` here has already had most of its park effect
    # stripped upstream (xwOBA substitution in Marcel), so only the
    # residual — the part xwOBA doesn't already explain — should be
    # reapplied here. Using the runs-scale factor overstates the effect
    # for extreme parks (Coors, T-Mobile) since it was calibrated to a
    # different quantity (runs scored) than what's actually left to
    # correct for on a park-neutral wOBA.
    park_factor = get_woba_residual_factor(team)
    
    # Calculate Park Adjustment
    park_adjustment = lg_runs_per_pa - (park_factor * lg_runs_per_pa)
    
    # Calculate the numerator for wRC+
    numerator = (wraa_per_pa + lg_runs_per_pa) + park_adjustment
    
    # Calculate wRC+
    wrc_plus = (numerator / lg_wrc_per_pa) * 100
    
    return wrc_plus

def calculate_woba_from_predictions(batter_df: pd.DataFrame, use_calculated_woba: bool = None) -> pd.DataFrame:
    """
    Reconcile rate stats and counting stats for batter predictions.

    Two mutually exclusive modes controlled by ``BatterConfig``:

    **Mode A — CALCULATE_COMPONENTS_FROM_WOBA = True** (recommended)
        The model's rate stats (wOBA, OBP, SLG, AVG, BB%, K%) are kept as-is.
        Counting stats (HR, 2B, 3B, RBI, R, HBP) are *derived* from each
        player's career counting profile scaled by (*predicted_wOBA / career_wOBA*).
        PA is set to 650.  This is the inverse of "wOBA from components" — it
        uses the model's well-calibrated wOBA as the source of truth and
        produces player-specific counting stats (Raleigh's HR-heavy mix,
        Witt's doubles+triples).

    **Mode B — CALCULATE_WOBA_FROM_COMPONENTS = True** (legacy)
        Counting stats are taken from the model.  wOBA/OBP/SLG are optionally
        recalculated from those counting stats.

    The two modes never run together — Mode A takes priority when enabled.

    Args:
        batter_df: DataFrame with batter predictions (per-150-game rates).
        use_calculated_woba: Legacy override for Mode B.  Ignored in Mode A.

    Returns:
        DataFrame with reconciled rate + counting stats and PA set.
    """
    from .config import Config

    # ── Load config ──────────────────────────────────────────────────────
    components_from_woba = False
    use_calculated_obp = False
    use_calculated_slg = False
    pa_full = 1500.0
    n_recent = 3

    try:
        try:
            from ..configs.batter_config import BatterConfig
        except (ImportError, ValueError):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from configs.batter_config import BatterConfig

        components_from_woba = getattr(BatterConfig, 'CALCULATE_COMPONENTS_FROM_WOBA', False)
        pa_full = getattr(BatterConfig, 'COMPONENTS_FROM_WOBA_PA_WEIGHT', 1500.0)
        n_recent = getattr(BatterConfig, 'COMPONENTS_FROM_WOBA_RECENT_SEASONS', 3)

        if use_calculated_woba is None:
            use_calculated_woba = BatterConfig.CALCULATE_WOBA_FROM_COMPONENTS
        use_calculated_obp = getattr(BatterConfig, 'CALCULATE_OBP_FROM_COMPONENTS', False)
        use_calculated_slg = getattr(BatterConfig, 'CALCULATE_SLG_FROM_COMPONENTS', False)

    except (ImportError, AttributeError) as e:
        if use_calculated_woba is None:
            use_calculated_woba = True
        logger.warning(f"Could not load BatterConfig ({e}), using defaults")

    # ==================================================================
    # MODE A — Derive counting stats from wOBA × career profile
    # ==================================================================
    if components_from_woba:
        logger.info("CALCULATE_COMPONENTS_FROM_WOBA = True — deriving counting stats from predicted wOBA")
        return _derive_components_from_woba(batter_df, pa_full, n_recent)

    # ==================================================================
    # MODE B — Legacy: optionally recalculate wOBA/OBP/SLG from counting stats
    # ==================================================================
    if not (use_calculated_woba or use_calculated_obp or use_calculated_slg):
        logger.info("All component calculations disabled — using LSTM's direct predictions.")
        return batter_df

    df = batter_df.copy()

    # Build intermediate counting stats from predictions
    df['PA'] = 650.0
    games_estimate = 150.0
    df['BB'] = df['BB%'] * df['PA']
    df['K'] = df['K%'] * df['PA']
    df['HR_count'] = df['HR'] * (games_estimate / 150)
    df['2B_count'] = df['2B'] * (games_estimate / 150)
    df['3B_count'] = df['3B'] * (games_estimate / 150)

    if 'HBP' in df.columns:
        df['HBP_count'] = df['HBP'] * (games_estimate / 150)
    else:
        df['HBP_count'] = df['PA'] * 0.01

    if 'SF' in df.columns:
        df['SF_count'] = df['SF'] * (games_estimate / 150)
    else:
        df['SF_count'] = df['PA'] * 0.007

    df['AB'] = df['PA'] - df['BB'] - df['HBP_count'] - df['SF_count']
    df['H'] = df['AVG'] * df['AB']
    df['1B'] = df['H'] - df['2B_count'] - df['3B_count'] - df['HR_count']
    df['IBB'] = df['BB'] * 0.10

    if use_calculated_woba:
        weights = Config.WAR.WOBA_WEIGHTS
        df['wOBA_calculated'] = df.apply(
            lambda row: calculate_woba(
                ab=row['AB'], bb=row['BB'], ibb=row['IBB'], hbp=row['HBP_count'], sf=row['SF_count'],
                singles=row['1B'], doubles=row['2B_count'], triples=row['3B_count'], hr=row['HR_count'],
                pa=row['PA'],
                wbb=weights['wBB'], whbp=weights['wHBP'], w1b=weights['w1B'],
                w2b=weights['w2B'], w3b=weights['w3B'], whr=weights['wHR']
            ),
            axis=1
        )
        logger.info(f"Average wOBA — LSTM: {df['wOBA'].mean():.3f}, Calculated: {df['wOBA_calculated'].mean():.3f}")
        df['wOBA'] = df['wOBA_calculated']
        df = df.drop(columns=['wOBA_calculated'])

    if use_calculated_obp:
        obp_num = df['H'] + df['BB'] + df['HBP_count']
        obp_den = df['AB'] + df['BB'] + df['HBP_count'] + df['SF_count']
        df['OBP'] = (obp_num / obp_den).clip(0, 1)
        logger.info("Recalculated OBP from components")

    if use_calculated_slg:
        slg_num = df['1B'] + 2 * df['2B_count'] + 3 * df['3B_count'] + 4 * df['HR_count']
        df['SLG'] = (slg_num / df['AB']).clip(0, 4)
        logger.info("Recalculated SLG from components")

    return df


def _derive_components_from_woba(
    batter_df: pd.DataFrame,
    pa_full: float = 1500.0,
    n_recent: int = 3,
) -> pd.DataFrame:
    """
    Derive counting stats from the model's predicted wOBA.

    Rate stats (wOBA, OBP, SLG, AVG, BB%, K%) are KEPT as the model predicted.
    Counting stats (HR, 2B, 3B, RBI, R, HBP) are replaced by each player's
    career per-150 profile scaled by (predicted_wOBA / career_wOBA).

    For young players (career PA < ``pa_full``), a blend of career-derived and
    model-predicted values is used.  Players without a historical profile are
    left unchanged.
    """
    from .counting_recalibration import build_career_profiles, _load_historical_batting

    COUNTING_STATS = ['HR', '2B', '3B', 'RBI', 'R', 'HBP']

    df = batter_df.copy()
    df['PA'] = 650.0

    # Build career profiles from historical data
    hist_df = _load_historical_batting()
    if hist_df.empty:
        logger.warning("No historical data — counting stats unchanged")
        return df

    career_profiles = build_career_profiles(hist_df, n_recent=n_recent, min_pa=50)
    if not career_profiles:
        logger.warning("No career profiles built — counting stats unchanged")
        return df

    available_stats = [s for s in COUNTING_STATS if s in df.columns]
    n_recalibrated = 0

    # Ensure counting stat columns are float so we can write blended values
    for s in available_stats:
        df[s] = df[s].astype(float)

    for idx, row in df.iterrows():
        player_id = int(row['IDfg'])
        profile = career_profiles.get(player_id)
        if profile is None:
            continue

        pred_woba = row['wOBA']
        base_woba = profile['base_woba']
        if base_woba < 0.15 or pd.isna(pred_woba):
            continue

        ratio = max(0.50, min(1.50, pred_woba / base_woba))
        blend = min(profile['career_pa'] / pa_full, 1.0)

        for stat in available_stats:
            if stat not in profile['base_counts']:
                continue
            derived = profile['base_counts'][stat] * ratio
            model_pred = row[stat]
            df.at[idx, stat] = max(0.0, blend * derived + (1.0 - blend) * model_pred)

        n_recalibrated += 1

    logger.info(
        f"Derived counting stats from wOBA for {n_recalibrated}/{len(df)} player-year rows "
        f"({len(available_stats)} stats: {available_stats})"
    )

    # =====================================================================
    # RECONSTRUCT OBP AND wOBA FOR INTERNAL CONSISTENCY
    # =====================================================================
    # The model predicts AVG, OBP, SLG, wOBA independently — they may not
    # be self-consistent.  After deriving counting stats above, recompute
    # OBP from AVG + BB% + HBP, and wOBA from the counting stats using
    # standard linear weights.  This guarantees everything displayed matches.
    from .config import Config
    weights = Config.WAR.WOBA_WEIGHTS

    pa = df['PA'].fillna(650.0)
    bb = df['BB%'] * pa
    hbp_count = df['HBP'] if 'HBP' in df.columns else pa * 0.01
    sf_count = pa * 0.007  # sacrifice fly estimate
    ab = pa - bb - hbp_count - sf_count

    # Hits from AVG × AB
    h = df['AVG'] * ab
    hr = df['HR'] if 'HR' in df.columns else 0.0
    doubles = df['2B'] if '2B' in df.columns else 0.0
    triples = df['3B'] if '3B' in df.columns else 0.0
    singles = (h - doubles - triples - hr).clip(lower=0)

    # Reconstruct OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    obp_num = h + bb + hbp_count
    obp_den = ab + bb + hbp_count + sf_count
    df['OBP'] = (obp_num / obp_den.replace(0, 1)).clip(0, 1)

    # Reconstruct wOBA from counting stats
    woba_num = (weights['wBB'] * bb + weights['wHBP'] * hbp_count +
                weights['w1B'] * singles + weights['w2B'] * doubles +
                weights['w3B'] * triples + weights['wHR'] * hr)
    df['wOBA'] = (woba_num / pa.replace(0, 1)).clip(0, 1)

    # Also reconstruct SLG for consistency
    slg_num = singles + 2 * doubles + 3 * triples + 4 * hr
    df['SLG'] = (slg_num / ab.replace(0, 1)).clip(0, 4)

    logger.info(
        f"Reconstructed OBP (mean={df['OBP'].mean():.3f}), "
        f"wOBA (mean={df['wOBA'].mean():.3f}), "
        f"SLG (mean={df['SLG'].mean():.3f}) from components for internal consistency"
    )

    return df

def calculate_baserunning_value(row: pd.Series, games: int) -> float:
    """
    Calculate baserunning value (BsR) from baserunning predictions.
    Uses the combined Statcast total baserunning run value (per 150 games)
    and scales by actual games played.
    """
    # sc_baserunning_runner_runs_tot_rate is already the combined metric (XB + SBX)
    # Rate is per 150 games, so: (rate / 150) * actual_games
    bsr = row.get('sc_baserunning_runner_runs_tot_rate', 0) * (games / 150.0)
    
    return bsr

def calculate_pitcher_war(fip: float,
                         ip: float,
                         team: str,
                         role: str = 'SP',
                         rate_stats: Optional[Dict] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate pitcher WAR from FIP and allocated innings.
    
    Uses FanGraphs-style dynamic RPW: an ace's runs are worth more wins
    because they compress the run environment while on the mound.
    
    Formula:
        FIP_runs  = (lgFIP - park_adj_FIP) / 9 × IP
        Repl_runs = replacement_per_200IP × (IP / 200)
        RPW       = dynamic (pitcher-specific, based on ERA)
        WAR       = (FIP_runs + Repl_runs) / RPW
    
    Args:
        fip: Projected FIP
        ip: Allocated innings pitched
        team: Team abbreviation for park factor
        role: 'SP' or 'RP'
        rate_stats: Dict with rate stats (K%, BB%, ERA, etc.)
        
    Returns:
        Tuple of (war, components_dict) with full breakdown
    """
    # Park factor adjustment for pitchers (inverse of batters)
    park_factor = get_park_factor(team)
    
    # Adjust FIP for park (pitcher in a hitter's park has inflated FIP)
    park_adjusted_fip = fip / park_factor if park_factor != 0 else fip
    
    # FIP runs saved (positive = better than league)
    fip_runs = (LG_FIP - park_adjusted_fip) / 9.0 * ip
    
    # Replacement level runs
    replacement_runs = REPLACEMENT_LEVEL_RUNS_200IP * (ip / 200.0)
    
    # Total runs above replacement
    rar = fip_runs + replacement_runs
    
    # Dynamic RPW — aces get more credit per run saved
    era = rate_stats.get('ERA', LG_RA9) if rate_stats else LG_RA9
    pitcher_rpw = _dynamic_pitcher_rpw(era, ip, role)
    
    # WAR
    war = rar / pitcher_rpw
    
    # Build components dict
    components = {
        'FIP_Runs': fip_runs,
        'Replacement_Runs': replacement_runs,
        'Pitcher_RPW': pitcher_rpw,
        'WAR': war,
        'IP': ip,
        'Team': team,
        'Role': role
    }
    
    # Add rate stats if provided
    if rate_stats:
        for key, value in rate_stats.items():
            components[key] = value
    
    return war, components

def infer_position_from_profile(position_profile: Optional[Dict[str, float]]) -> str:
    """
    Get display position from a position profile.
    Returns primary defensive position, or 'DH' if player is 80%+ DH.
    """
    if not position_profile:
        return 'DH'
    return get_display_position(position_profile)

def calculate_defensive_value(
    fielding_data: pd.DataFrame,
    player_id: int,
    year: int,
    position_profile: Optional[Dict[str, float]] = None,
    games: int = 150,
) -> tuple[float, float]:
    """
    Calculate defensive value and positional adjustment using position profiles.
    
    Fielding run value is weighted across all positions the player has predictions for,
    proportional to their position profile fractions.
    
    Positional adjustment is weighted by the full profile (including DH fraction).
    
    Args:
        fielding_data: Fielding predictions DataFrame
        player_id: Player IDfg
        year: Projection year
        position_profile: {pos: fraction} from build_position_profiles
        games: Projected games (default 150, reduced for ROS projections)
        
    Returns:
        tuple: (defensive_value, positional_adjustment)
    """
    if not position_profile:
        return 0.0, POSITIONAL_ADJUSTMENTS.get('DH', -17.5) * (games / 162.0)
    
    # Get all fielding predictions for this player-year
    player_fielding = fielding_data[
        (fielding_data['IDfg'] == player_id) & (fielding_data['Year'] == year)
    ]
    
    # Calculate weighted fielding run value across predicted positions
    # Only defensive positions have fielding predictions — DH has none
    def_positions = get_defensive_positions(position_profile)
    total_def_fraction = sum(def_positions.values())  # fraction of time playing defense
    
    weighted_fld = 0.0
    if not player_fielding.empty and total_def_fraction > 0:
        # Collect FRV predictions by position
        predicted_frv = {}
        for _, row in player_fielding.iterrows():
            pred_pos = row.get('Pos') or row.get('Position', '')
            if pred_pos in DEFENSIVE_POSITIONS:
                predicted_frv[pred_pos] = row.get('sc_total_runs/150', 0)
        
        # Use direct predictions for positions in the profile
        for pos, frac in def_positions.items():
            if pos in predicted_frv:
                weighted_fld += predicted_frv[pos] * frac
        
        # Estimate FRV for profile positions with no prediction (position switchers)
        # Uses the position-to-position FRV transfer map
        from core.position_profiles import estimate_missing_frv
        estimated = estimate_missing_frv(position_profile, predicted_frv)
        for pos, est_frv in estimated.items():
            frac = def_positions.get(pos, 0.0)
            if frac > 0:
                weighted_fld += est_frv * frac
    
    # Scale fielding value to games
    def_value = weighted_fld * (games / 150.0)
    
    # Weighted positional adjustment across ALL positions (including DH)
    pos_adjustment = get_weighted_positional_adjustment(
        position_profile, POSITIONAL_ADJUSTMENTS, games
    )
    
    return def_value, pos_adjustment

def calculate_war_components(
    row: pd.Series,
    baserunning_data: pd.DataFrame,
    fielding_data: pd.DataFrame,
    position_profiles: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate comprehensive WAR components combining all three prediction types.
    
    Uses position profiles (from historical fielding data) to:
    - Determine player's position(s)
    - Weight positional adjustments across all positions played
    - Weight fielding run values across predicted positions
    """
    player_id = row['IDfg']
    year = row['Year']
    
    # Get position profile for this player
    profile = position_profiles.get(player_id) if position_profiles else None
    position = infer_position_from_profile(profile)
    
    # Use PA from the row (already reduced to remaining season if applicable)
    games = row.get('G', 150)
    pa = row.get('PA', 650)
    
    # Get team for park factor
    # NOTE: wOBA-scale RESIDUAL factor, not runs-scale — see note in
    # calculate_wrc_plus above. row['wOBA'] has already had most of its
    # park effect removed upstream via xwOBA substitution, so this should
    # only reapply the leftover (residual) piece, matched to the same
    # factor used in _apply_park_factors_to_batter_predictions and
    # calculate_wrc_plus so the three don't compound on each other.
    team = row.get('Team', '')
    if pd.isnull(team):
        park_factor = 1.0
    else:
        park_factor = get_woba_residual_factor(team)
    
    # Batting value calculation (wRAA + park adjustment)
    woba = row['wOBA']
    wraa = ((woba - LG_WOBA) / WOBA_SCALE) * pa
    batting_runs = wraa + (RPA - (RPA * park_factor)) * pa
    
    # Get baserunning value
    bsr_row = baserunning_data[(baserunning_data['IDfg'] == player_id) & 
                               (baserunning_data['Year'] == year)]
    
    if not bsr_row.empty:
        bsr = calculate_baserunning_value(bsr_row.iloc[0], games)
    else:
        bsr = -0.5
    
    # Get defensive value and positional adjustment (weighted by position profile)
    fld_value, pos_adjustment = calculate_defensive_value(
        fielding_data, player_id, year, position_profile=profile,
        games=games,
    )
    
    # Total defensive value = fielding + positional adjustment
    def_value = fld_value + pos_adjustment
    
    # Cap negative defensive value at DH level (worst case = just be a DH)
    dh_penalty = POSITIONAL_ADJUSTMENTS.get('DH', -17.5) * (games / 162.0)
    if def_value < dh_penalty:
        def_value = dh_penalty
    
    # Offensive value
    off = batting_runs + bsr
    
    # Replacement level
    rep_level = 570 * RPW * pa / LG_PA
    
    # RAR (Runs Above Replacement)
    rar = off + def_value + rep_level
    
    # WAR
    war = rar / RPW
    
    # Counting stats — already reduced for ROS projections, just round
    counting_stats = {}
    for stat in ['HR', '2B', '3B', 'RBI', 'R']:
        if stat in row:
            counting_stats[stat] = round(float(row[stat]), 1)
        else:
            counting_stats[stat] = 0.0
    
    # Baserunning counting stats (SB_rate/CS_rate are per 150 → scale by games)
    if not bsr_row.empty:
        counting_stats['SB'] = round(bsr_row.iloc[0].get('SB_rate', 0) * (games / 150.0), 1)
        counting_stats['CS'] = round(bsr_row.iloc[0].get('CS_rate', 0) * (games / 150.0), 1)
    else:
        counting_stats['SB'] = 0.0
        counting_stats['CS'] = 0.0
    
    return war, {
        'Bat': batting_runs,
        'BsR': bsr,
        'Fld': fld_value,
        'Pos': pos_adjustment,
        'Def': def_value,
        'WAR': war,
        'PA': pa,
        'G': games,
        'Position': position,
        'Team': team,
        **counting_stats
    }


# =============================================================================
# PARK FACTOR REAPPLICATION (post-prediction, pre-WAR)
# =============================================================================

def _apply_park_factors_to_batter_predictions(batter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reapply park factors to park-neutral batter predictions.
    
    When ENABLE_PARK_FACTOR_ADJUSTMENT is True in BatterConfig, the LSTM
    received park-neutralized inputs and therefore outputs park-neutral
    predictions.  Before computing wOBA/wRC+/WAR, we multiply the predicted
    stats back by the player's current team park factor so that the final
    numbers reflect their actual home environment.
    
    Stats excluded from adjustment: Age, wRC+ (already park-adjusted by formula).
    
    Args:
        batter_df: DataFrame with park-neutral batter predictions and 'Team' column
        
    Returns:
        DataFrame with park-adjusted predictions (or unchanged if toggle is off)
    """
    # Check if park factor adjustment is enabled
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from configs.batter_config import BatterConfig
        if not getattr(BatterConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False):
            logger.info("Park factor reapplication DISABLED (ENABLE_PARK_FACTOR_ADJUSTMENT=False)")
            return batter_df
    except (ImportError, AttributeError):
        logger.info("Park factor reapplication DISABLED (BatterConfig not available)")
        return batter_df
    
    if 'Team' not in batter_df.columns:
        logger.warning("No 'Team' column in batter predictions — skipping park factor reapplication")
        return batter_df
    
    # Get the list of features to adjust (exclude Age, wRC+)
    # Use the batter config's feature list to know which columns are model outputs
    from configs.batter_config import BatterConfig
    model_features = list(BatterConfig.FINETUNE_FEATURES) + list(BatterConfig.CLASSICAL_FEATURES)
    # Deduplicate while preserving order
    seen = set()
    model_features = [f for f in model_features if f not in seen and not seen.add(f)]
    
    EXCLUDED = {'Age', 'wRC+'}
    adjustable = [f for f in model_features if f in batter_df.columns and f not in EXCLUDED]
    
    if not adjustable:
        logger.warning("No adjustable features found for park factor reapplication")
        return batter_df
    
    df = batter_df.copy()
    
    # Build park factor series aligned to the DataFrame
    # (get_park_factor handles NaN/None itself and resolves team-code
    # aliases like SFG/SDP/KCR/TBR/WSN, so no separate notna() check needed)
    pf_series = df['Team'].map(get_park_factor)
    
    # Multiply each adjustable feature by the park factor
    n_adjusted = 0
    for feat in adjustable:
        df[feat] = df[feat] * pf_series
        n_adjusted += 1
    
    logger.info(
        f"Park factor reapplication applied to {n_adjusted} batter features "
        f"(excluded: {EXCLUDED & set(model_features)})"
    )
    
    return df


def _apply_park_factors_to_pitcher_predictions(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reapply park factors to park-neutral pitcher predictions.
    
    When ENABLE_PARK_FACTOR_ADJUSTMENT is True in the pitcher config, the LSTM
    received park-neutralized inputs and therefore outputs park-neutral
    predictions.  Before computing pitcher WAR, we multiply the predicted
    stats back by the player's current team park factor so that the final
    numbers reflect their actual home environment.
    
    For pitchers, higher park factor means the park inflates runs — so a
    pitcher's ERA/FIP in Coors should be higher than in Petco.  We multiply
    by PF (same direction as batters) because PF > 1 means run-inflating park.
    
    Args:
        pitcher_df: DataFrame with park-neutral pitcher predictions and 'Team' column
        
    Returns:
        DataFrame with park-adjusted predictions (or unchanged if toggle is off)
    """
    # Check if park factor adjustment is enabled for either SP or RP
    sp_enabled = False
    rp_enabled = False
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from configs.pitcher_sp_config import PitcherSPConfig
        sp_enabled = getattr(PitcherSPConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)
    except (ImportError, AttributeError):
        pass
    try:
        from configs.pitcher_rp_config import PitcherRPConfig
        rp_enabled = getattr(PitcherRPConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)
    except (ImportError, AttributeError):
        pass
    
    if not sp_enabled and not rp_enabled:
        logger.info("Park factor reapplication DISABLED for pitchers")
        return pitcher_df
    
    if 'Team' not in pitcher_df.columns:
        logger.warning("No 'Team' column in pitcher predictions — skipping park factor reapplication")
        return pitcher_df
    
    # Pitcher features that should be park-adjusted
    # ERA and FIP are directly affected by park (run environment)
    # K%, BB% are not park-affected (they're outcomes of pitcher skill)
    # But we apply uniformly to be consistent with the neutralization step
    EXCLUDED = {'Age', 'Stuff+', 'Location+', 'Pitching+'}
    
    # Collect all possible pitcher features from both configs
    pitcher_features = set()
    try:
        from configs.pitcher_sp_config import PitcherSPConfig
        pitcher_features.update(PitcherSPConfig.FINETUNE_FEATURES)
        pitcher_features.update(PitcherSPConfig.CLASSICAL_FEATURES)
    except (ImportError, AttributeError):
        pass
    try:
        from configs.pitcher_rp_config import PitcherRPConfig
        pitcher_features.update(PitcherRPConfig.FINETUNE_FEATURES)
        pitcher_features.update(PitcherRPConfig.CLASSICAL_FEATURES)
    except (ImportError, AttributeError):
        pass
    
    adjustable = [f for f in pitcher_features if f in pitcher_df.columns and f not in EXCLUDED]
    
    if not adjustable:
        logger.warning("No adjustable features found for pitcher park factor reapplication")
        return pitcher_df
    
    df = pitcher_df.copy()
    
    # Apply park factors only to rows whose role has the toggle enabled
    for idx, row in df.iterrows():
        role = row.get('Role', 'SP')
        if (role == 'SP' and not sp_enabled) or (role == 'RP' and not rp_enabled):
            continue
        
        team = row.get('Team', '')
        if pd.isna(team) or team == '':
            continue
        
        pf = get_park_factor(team)
        if pf != 1.0:
            for feat in adjustable:
                if feat in df.columns and pd.notna(row[feat]):
                    df.at[idx, feat] = row[feat] * pf
    
    logger.info(
        f"Park factor reapplication applied to {len(adjustable)} pitcher features "
        f"(excluded: {EXCLUDED & pitcher_features})"
    )
    
    return df


def process_predictions(data_dir: Path, output_dir: Path, target_year: Optional[int] = None) -> None:
    """
    Main processing function that combines all predictions and calculates comprehensive WAR.
    """
    logger.info("Starting WAR calculation post-processing...")
    
    # Load all prediction files from pipeline subdirectory
    batter_file = data_dir / "pipeline" / "batter_predictions.csv"
    baserunning_file = data_dir / "pipeline" / "baserunning_predictions.csv"
    fielding_file = data_dir / "pipeline" / "fielding_predictions.csv"
    
    if not all(f.exists() for f in [batter_file, baserunning_file, fielding_file]):
        missing = [f for f in [batter_file, baserunning_file, fielding_file] if not f.exists()]
        raise FileNotFoundError(f"Missing prediction files: {missing}")
    
    logger.info("Loading prediction files...")
    batter_df = pd.read_csv(batter_file)
    baserunning_df = pd.read_csv(baserunning_file)
    fielding_df = pd.read_csv(fielding_file)
    
    logger.info(f"Loaded {len(batter_df)} batter predictions")
    logger.info(f"Loaded {len(baserunning_df)} baserunning predictions") 
    logger.info(f"Loaded {len(fielding_df)} fielding predictions")
    
    # Filter by year if specified
    if target_year:
        batter_df = batter_df[batter_df['Year'] == target_year]
        baserunning_df = baserunning_df[baserunning_df['Year'] == target_year]
        fielding_df = fielding_df[fielding_df['Year'] == target_year]
        logger.info(f"Filtered to {target_year}: {len(batter_df)} batters")
    
    # Load organization data for park factors
    org_data = load_player_orgs(data_dir)
    
    # Merge organization data with batter predictions
    batter_df = batter_df.merge(org_data, on='IDfg', how='left')
    
    # =========================================================================
    # PARK FACTOR REAPPLICATION
    # =========================================================================
    # If park factor neutralization was enabled during predictions, the model's
    # outputs are park-neutral. We now reapply park factors (multiply) so that
    # wOBA/wRC+/WAR reflect the player's actual home environment.
    #
    # This step converts: park-neutral predictions → park-adjusted predictions
    # BEFORE calculating wRC+.
    batter_df = _apply_park_factors_to_batter_predictions(batter_df)
    
    # Counting stat derivation + rate stat reconstruction now happen in-loop
    # inside core/batter_prediction.py. PA is set during prediction.
    if 'PA' not in batter_df.columns:
        batter_df['PA'] = 650
    
    # Calculate wRC+ with proper park factors
    logger.info("Calculating wRC+ with park factors...")
    batter_df['wRC+_new'] = batter_df.apply(
        lambda row: calculate_wrc_plus(row['wOBA'], row.get('Team', ''), row.get('PA', 630)),
        axis=1
    )
    
    # Calculate comprehensive WAR components
    logger.info("Calculating comprehensive WAR components...")
    war_components_list = []
    
    for idx, row in batter_df.iterrows():
        try:
            war, components = calculate_war_components(row, baserunning_df, fielding_df)
            components['IDfg'] = row['IDfg']
            components['Year'] = row['Year']
            war_components_list.append(components)
        except Exception as e:
            logger.error(f"Error calculating WAR for {row['Name']} ({row['IDfg']}): {e}")
            continue
    
    # Convert to DataFrame and merge back
    war_df = pd.DataFrame(war_components_list)
    batter_df = batter_df.merge(war_df, on=['IDfg', 'Year'], how='left', suffixes=('_old', ''))
    
    # Clean up columns - remove old WAR components and keep new ones
    columns_to_remove = [col for col in batter_df.columns if col.endswith('_old')]
    batter_df = batter_df.drop(columns=columns_to_remove)
    
    # Update wRC+ 
    if 'wRC+_new' in batter_df.columns:
        batter_df['wRC+'] = batter_df['wRC+_new']
        batter_df = batter_df.drop(columns=['wRC+_new'])
    
    # Reorder columns for better readability
    column_order = ['Name', 'IDfg', 'Year', 'Age', 'Team', 'Position', 'BB%', 'K%', 'AVG', 'OBP', 'SLG', 
                   'wOBA', 'wRC+',
                   'Bat', 'BsR', 'Fld', 'Pos', 'Def', 'WAR', 'PA', 'G', 
                   'HR', '2B','3B', 'RBI', 'R', 'SB', 'CS']
    
    # Keep only columns that exist in the DataFrame
    final_columns = [col for col in column_order if col in batter_df.columns]
    batter_df = batter_df[final_columns]
    
    # Sort by Year and WAR
    batter_df = batter_df.sort_values(['Year', 'WAR'], ascending=[True, False])
    
    # Save updated predictions to pipeline subdirectory
    output_file = output_dir / "pipeline" / "batter_predictions_with_war.csv"
    batter_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved comprehensive predictions to {output_file}")
    logger.info(f"Processed {len(batter_df)} player seasons")
    
    # Display top performers for latest year
    latest_year = batter_df['Year'].max()
    top_performers = (batter_df[batter_df['Year'] == latest_year]
                     .nlargest(10, 'WAR')[['Name', 'Age', 'Position', 'wOBA', 'wRC+', 'Bat', 'BsR', 'Fld', 'Pos', 'Def', 'WAR']])
    
    print(f"\nTop 10 Predicted WAR for {latest_year}:")
    print(top_performers.to_string(index=False))

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Calculate comprehensive WAR from prediction files')
    parser.add_argument('--year', type=int, help='Filter to specific year (optional)')
    parser.add_argument('--data-dir', type=Path, default=Path('../data/generated'),
                       help='Directory containing prediction CSV files')
    parser.add_argument('--output-dir', type=Path, default=Path('../data/generated'),
                       help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        process_predictions(args.data_dir, args.output_dir, args.year)
        logger.info("WAR calculation completed successfully!")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()