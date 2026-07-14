#!/usr/bin/env python
"""
Value Determination — Current-Season Production Pipeline
=========================================================

Orchestrates the 10-step value determination pipeline: load predictions
→ calculate WAR → merge salary/contracts → compute surplus → calculate
trade values → export.

Moved from ``value_determination/main.py`` to ``value_determination/pipelines/current.py``
to separate pipeline orchestration from the shared engine modules.

Pipeline Steps:
    1. Load prediction and salary data
    2. Calculate pitcher WAR from FIP
    3. Merge prediction data (SP, RP, batters)
    4. Clean and integrate salary data
    5. Normalize contract statuses
    6. Generate contract timeline
    7. Calculate WAR values and surplus
    8. Apply prospect adjustments
    9. Calculate trade values
    10. Export results

Usage:
    python -m value_determination.pipelines.current
    python run_value_determination.py          (convenience wrapper)

Input:
    - data/generated/pipeline/: Prediction files
    - data/salary/: Salary data
    - data/prospect_data/: Prospect rankings

Output:
    - data/generated/value_by_year/player_values_complete.csv

Configuration:
    All settings are in config.py - edit there, not here.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from centralized config (SINGLE SOURCE OF TRUTH)
from value_determination.config import (
    Config, logger, CURRENT_YEAR,
    # Backward compatibility
    OUTPUT_DIR, HITTER_COLUMNS, PITCHER_COLUMNS
)
from value_determination.data_loader import (
    load_prediction_files, merge_prediction_data, load_historical_data
)
from value_determination.salary_processor import (
    clean_salary_data, merge_salary_with_ids, complete_years_of_service
)
from value_determination.contract_processor import (
    normalize_contract_status, check_none_statuses,
    generate_contract_timeline, validate_fa_years, extend_fa_timeline
)
from value_determination.value_calculator import (
    join_predictions_with_timeline, calculate_contract_value,
    calculate_surplus_value, integrate_historical_stats,
    integrate_player_statistics, post_process_export_data
)
from value_determination.trade_value import (
    analyze_contract_options, calculate_trade_values,
    add_trade_ranking_metrics, update_prospect_mlb_status
)
from value_determination.exporter import export_value_data
from value_determination.calculate_war import (
    calculate_war_components, calculate_pitcher_war,
    load_player_orgs, calculate_wrc_plus
)
from value_determination.playing_time import estimate_playing_time
from value_determination.milb_regression import apply_milb_regression
from core.position_profiles import (
    build_position_profiles, load_fielding_history, load_batting_for_games,
    get_display_position, get_defensive_positions
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _export_fielding_projections(
    fielding_data: pd.DataFrame,
    position_profiles: dict,
    org_data: pd.DataFrame,
    batter_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export per-position fielding projections with innings estimates.

    Produces ``fielding_projections_complete.csv`` consumed by the web DB
    loader.  Each row represents one player × year × position with:
    - Statcast run-value components (per 150)
    - Estimated games / innings at that position (from profile fractions)
    - Team assignment
    """
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    if fielding_data.empty:
        logger.warning("No fielding predictions to export")
        return

    rows = []
    # Build a team lookup from org_data
    team_map = {}
    if org_data is not None and not org_data.empty:
        for _, r in org_data.drop_duplicates("IDfg").iterrows():
            team_map[r["IDfg"]] = r.get("Team", "")

    # Build a name lookup from batter_data
    name_map = {}
    if batter_data is not None and not batter_data.empty:
        for idfg, grp in batter_data.groupby("IDfg"):
            name_map[idfg] = grp.iloc[0].get("Name", "")

    total_games = 150  # standard full season
    innings_per_game = 8.5  # approximate defensive innings per 9-inning game

    for _, pred in fielding_data.iterrows():
        idfg = pred["IDfg"]
        year = pred["Year"]
        pos = pred.get("Pos", "")
        profile = position_profiles.get(idfg, {})
        pos_frac = profile.get(pos, 0.0) if profile else 0.0

        est_games = round(total_games * pos_frac)
        est_gs = est_games  # assume all are starts
        est_inn = round(est_games * innings_per_game, 1)

        rows.append({
            "IDfg": idfg,
            "Name": pred.get("Name") or name_map.get(idfg, ""),
            "Year": year,
            "Team": team_map.get(idfg, ""),
            "Pos": pos,
            "Age": pred.get("Age"),
            "G": est_games,
            "GS": est_gs,
            "Inn": est_inn,
            "sc_total_runs": pred.get("sc_total_runs/150", pred.get("sc_total_runs")),
            "sc_range_runs": pred.get("sc_range_runs/150", pred.get("sc_range_runs")),
            "sc_arm_runs": pred.get("sc_arm_runs/150", pred.get("sc_arm_runs")),
            "sc_dp_runs": pred.get("sc_dp_runs/150", pred.get("sc_dp_runs")),
            "sc_framing_runs": pred.get("sc_framing_runs/150", pred.get("sc_framing_runs")),
            "sc_throwing_runs": pred.get("sc_throwing_runs/150", pred.get("sc_throwing_runs")),
            "sc_blocking_runs": pred.get("sc_blocking_runs/150", pred.get("sc_blocking_runs")),
        })

    out_df = pd.DataFrame(rows)
    # Replace NaN with empty
    out_path = output_dir / "fielding_projections_complete.csv"
    out_df.to_csv(out_path, index=False, na_rep="")
    logger.info(f"Exported {len(out_df)} fielding projection rows to {out_path}")

def calculate_pitcher_war_for_dataframe(pitcher_df: pd.DataFrame, 
                                        org_data: pd.DataFrame, 
                                        role: str) -> pd.DataFrame:
    """
    Calculate WAR for all pitchers in a DataFrame.
    
    Expects ``IP`` (and optionally ``GS``, ``G``) to already be set by
    the playing-time estimator.  Falls back to config defaults if ``IP``
    is missing.
    
    Args:
        pitcher_df: DataFrame with pitcher predictions (must have 'FIP', 'IDfg' columns)
        org_data: DataFrame with player team assignments (from load_player_orgs)
        role: 'SP' or 'RP'
        
    Returns:
        DataFrame with 'WAR' column added
    """
    default_ip = Config.WAR.DEFAULT_SP_IP if role == 'SP' else Config.WAR.DEFAULT_RP_IP
    
    for idx, row in pitcher_df.iterrows():
        # Get team for park factor
        # TODO: Transition to mlbam_id matching
        team = ''
        if row['IDfg'] in org_data['IDfg'].values:
            team = org_data[org_data['IDfg'] == row['IDfg']]['Team'].iloc[0]
        
        # Build rate stats dict
        rate_stats = {
            'ERA': row.get('ERA', 0),
            'K%': row.get('K%', 0),
            'BB%': row.get('BB%', 0)
        }
        
        # Use per-pitcher IP from playing time estimator, fall back to default
        ip = row.get('IP', default_ip)
        if pd.isna(ip):
            ip = default_ip
        
        # 0 IP means intentionally projected as out (season-ending injury)
        if ip <= 0:
            pitcher_df.at[idx, 'WAR'] = 0.0
            continue
        
        # Calculate WAR using centralized function
        war, components = calculate_pitcher_war(
            fip=row['FIP'],
            ip=ip,
            team=team,
            role=role,
            rate_stats=rate_stats
        )
        
        pitcher_df.at[idx, 'WAR'] = war

    # Compute pitcher counting stats from rate stats and IP
    bf_per_ip = Config.WAR.BF_PER_IP
    bf = pitcher_df['IP'] * bf_per_ip
    pitcher_df['K_pit']  = (pitcher_df.get('K%',  pd.Series(0.0, index=pitcher_df.index)) * bf).round().astype(int)
    pitcher_df['BB_pit'] = (pitcher_df.get('BB%', pd.Series(0.0, index=pitcher_df.index)) * bf).round().astype(int)
    pitcher_df['ER_pit'] = (pitcher_df.get('ERA', pd.Series(0.0, index=pitcher_df.index)) * pitcher_df['IP'] / 9).round().astype(int)

    # Compute per-9 rates from per-TBF rates (only if not already supplied by prediction)
    if 'K/9' not in pitcher_df.columns:
        pitcher_df['K/9']  = pitcher_df.get('K%',  pd.Series(0.0, index=pitcher_df.index)) * bf_per_ip * 9
    if 'BB/9' not in pitcher_df.columns:
        pitcher_df['BB/9'] = pitcher_df.get('BB%', pd.Series(0.0, index=pitcher_df.index)) * bf_per_ip * 9
    if 'HR/9' not in pitcher_df.columns:
        # Derive HR% from HR/FB × FB% × BIP_rate when HR% is not a direct model feature
        _hrfb = pitcher_df.get('HR/FB', pd.Series(0.10, index=pitcher_df.index))
        _fb = pitcher_df.get('FB%', pd.Series(0.35, index=pitcher_df.index))
        _k = pitcher_df.get('K%', pd.Series(0.22, index=pitcher_df.index))
        _bb = pitcher_df.get('BB%', pd.Series(0.08, index=pitcher_df.index))
        _hbp = pitcher_df.get('HBP%', pd.Series(0.01, index=pitcher_df.index))
        _bip_rate = (1.0 - _k - _bb - _hbp).clip(lower=0.3)
        _hr_pct = (_hrfb * _fb * _bip_rate).clip(lower=0.005, upper=0.06)
        pitcher_df['HR/9'] = _hr_pct * bf_per_ip * 9

    return pitcher_df


def validate_input_data(sp_data, rp_data, batter_data, salary_data) -> bool:
    """
    Validate that all required input data is present and has expected columns.
    
    Args:
        sp_data: Starting pitcher predictions
        rp_data: Relief pitcher predictions  
        batter_data: Batter predictions
        salary_data: Salary/contract data
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails and Config.Pipeline.FAIL_ON_MISSING_DATA is True
    """
    errors = []
    
    # Check pitcher data has FIP
    for name, df in [('SP', sp_data), ('RP', rp_data)]:
        required = Config.Columns.REQUIRED['pitcher_predictions']
        missing = [col for col in required if col not in df.columns]
        if missing:
            errors.append(f"{name} data missing columns: {missing}")
    
    # Check batter data
    required = Config.Columns.REQUIRED['batter_predictions']
    missing = [col for col in required if col not in batter_data.columns]
    if missing:
        errors.append(f"Batter data missing columns: {missing}")
    
    # Check salary data
    required = Config.Columns.REQUIRED['salary']
    # Handle case-insensitive column names
    salary_cols_lower = [c.lower() for c in salary_data.columns]
    for col in required:
        if col.lower() not in salary_cols_lower:
            errors.append(f"Salary data missing column: {col}")
    
    if errors:
        for error in errors:
            logger.error(error)
        if Config.Pipeline.FAIL_ON_MISSING_DATA:
            raise ValueError(f"Input validation failed: {errors}")
        return False
    
    return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(pipeline_dir=None, output_filename=None):
    """
    Main pipeline execution.
    
    Runs all steps of the value determination pipeline in order.
    Each step is clearly logged and errors are handled gracefully.
    
    Args:
        pipeline_dir: Override directory for prediction CSVs (default: PIPELINE_DIR)
        output_filename: Override output CSV filename (default: player_values_complete.csv)
    """
    
    logger.info("=" * 60)
    logger.info("MLB Trade Simulator - Value Determination Pipeline")
    logger.info("=" * 60)
    
    # Ensure output directories exist
    Config.Paths.ensure_directories()
    
    try:
        # ============================================================
        # Step 1: Load Data
        # ============================================================
        logger.info("\n[Step 1/10] Loading prediction and salary data...")
        sp_data, rp_data, batter_data, baserunning_data, fielding_data, salary_data = load_prediction_files(
            pipeline_dir=pipeline_dir
        )
        
        # Validate input data
        validate_input_data(sp_data, rp_data, batter_data, salary_data)
        
        # Log unique status values for debugging
        status_col = 'status' if 'status' in salary_data.columns else 'Status'
        if status_col in salary_data.columns:
            logger.debug(f"Unique status values: {salary_data[status_col].unique()}")
        
        # ============================================================
        # Step 2: Calculate Pitcher WAR
        # ============================================================
        logger.info("\n[Step 2/10] Calculating pitcher WAR from FIP...")
        
        # Load team data for park factors
        org_data = load_player_orgs()
        
        # Merge team info for park factors
        sp_data = sp_data.merge(org_data[['IDfg', 'Team']], on='IDfg', how='left')
        rp_data = rp_data.merge(org_data[['IDfg', 'Team']], on='IDfg', how='left')
        
        # Apply park factors to park-neutral pitcher predictions (reverse neutralization)
        # Only when ENABLE_PARK_FACTOR_ADJUSTMENT is enabled in the respective config.
        from value_determination.calculate_war import _apply_park_factors_to_pitcher_predictions
        
        # Combine SP + RP for unified park factor application (respects per-role toggle)
        combined_pitcher = pd.concat([sp_data, rp_data], ignore_index=True)
        combined_pitcher = _apply_park_factors_to_pitcher_predictions(combined_pitcher)
        sp_data = combined_pitcher[combined_pitcher['Role'] == 'SP'].copy()
        rp_data = combined_pitcher[combined_pitcher['Role'] == 'RP'].copy()
        
        # NOTE: Pitcher post-processing (output regression, FIP/SIERA reconstruction,
        # HR% decomposition, ERA derivation) has been removed. All reconstruction
        # and rate normalization is now handled inside the autoregressive loop in
        # core/pitcher_prediction.py. Output regression and aging constraints are
        # no longer applied — input regression + in-loop constraints are sufficient.
        
        # Estimate pitcher playing time (IP, GS, G) before WAR calculation
        # Run per-year so each projection year gets age-appropriate injury risk
        # and current-IL status is only checked for the current season.
        logger.info("\n[Step 2a/10] Estimating pitcher playing time (IP, GS, G)...")
        combined_for_pt = pd.concat([sp_data, rp_data], ignore_index=True)
        year_chunks = []
        for proj_year in sorted(combined_for_pt['Year'].unique()):
            chunk = combined_for_pt[combined_for_pt['Year'] == proj_year].copy()
            chunk = estimate_playing_time(chunk, int(proj_year))
            year_chunks.append(chunk)
        combined_for_pt = pd.concat(year_chunks, ignore_index=True)
        sp_data = combined_for_pt[combined_for_pt['Role'] == 'SP'].copy()
        rp_data = combined_for_pt[combined_for_pt['Role'] == 'RP'].copy()
        
        # Calculate WAR for SP and RP using per-pitcher IP
        sp_data = calculate_pitcher_war_for_dataframe(sp_data, org_data, role='SP')
        rp_data = calculate_pitcher_war_for_dataframe(rp_data, org_data, role='RP')
        
        logger.info(f"SP WAR: n={len(sp_data)}, avg={sp_data['WAR'].mean():.2f}")
        logger.info(f"RP WAR: n={len(rp_data)}, avg={rp_data['WAR'].mean():.2f}")
        
        # ============================================================
        # Step 2.25: MiLB Regression for Low-Sample Batters
        # ============================================================
        logger.info("\n[Step 2.25/10] Applying MiLB regression to batter predictions...")
        batter_data = apply_milb_regression(batter_data, CURRENT_YEAR)
        
        # ── Filter pitchers who no longer bat (universal DH era) ────────
        # Pitchers with old NL batting history still get batter predictions.
        # Remove any pitcher who didn't record a PA in the prior season.
        pitcher_idfgs = set(sp_data['IDfg'].unique()) | set(rp_data['IDfg'].unique())
        pitchers_in_batters = pitcher_idfgs & set(batter_data['IDfg'].unique())
        if pitchers_in_batters:
            hist_bat = pd.read_csv(
                Config.Paths.HISTORIC_MLB_DIR / 'mlb_batting_data_1950_2025.csv',
                usecols=['IDfg', 'Season', 'PA'], low_memory=False,
            )
            hist_bat['PA'] = pd.to_numeric(hist_bat['PA'], errors='coerce').fillna(0)
            recent_batters = set(
                hist_bat.loc[
                    (hist_bat['Season'] >= CURRENT_YEAR - 1) & (hist_bat['PA'] >= 1),
                    'IDfg',
                ].unique()
            )
            pitchers_to_remove = pitchers_in_batters - recent_batters
            if pitchers_to_remove:
                n_before = len(batter_data)
                batter_data = batter_data[~batter_data['IDfg'].isin(pitchers_to_remove)]
                logger.info(
                    f"Removed {n_before - len(batter_data)} pitcher rows from "
                    f"batter predictions ({len(pitchers_to_remove)} pitchers "
                    f"with no batting PA since {CURRENT_YEAR - 1})"
                )
        
        # ============================================================
        # Step 2.5: Calculate Batter WAR Components
        # ============================================================
        logger.info("\n[Step 2.5/10] Calculating comprehensive WAR components for batters...")
        
        # Merge org data with batter predictions for park factors
        batter_data = batter_data.merge(org_data, on='IDfg', how='left', suffixes=('', '_org'))
        
        # Reapply park factors to park-neutral batter predictions (reverse neutralization)
        # Only when ENABLE_PARK_FACTOR_ADJUSTMENT is enabled in BatterConfig.
        from value_determination.calculate_war import _apply_park_factors_to_batter_predictions
        batter_data = _apply_park_factors_to_batter_predictions(batter_data)
        
        # Counting stat derivation + rate stat reconstruction now happen
        # in-loop inside core/batter_prediction.py (analogous to pitcher
        # FIP/ERA reconstruction).  PA is set during prediction.
        if 'PA' not in batter_data.columns:
            batter_data['PA'] = 650
        
        # Calculate wRC+ with park factors
        logger.info("Calculating wRC+ with park factors...")
        batter_data['wRC+'] = batter_data.apply(
            lambda row: calculate_wrc_plus(row['wOBA'], row.get('Team', ''), row.get('PA', 630)),
            axis=1
        )
        
        # Calculate WAR components for each batter
        # Build position profiles from historical fielding data
        logger.info("Building position profiles from historical fielding data...")
        hist_fielding = load_fielding_history()
        hist_batting = load_batting_for_games()
        batter_ids = batter_data['IDfg'].unique().tolist()
        position_profiles = build_position_profiles(
            hist_fielding, hist_batting, batter_ids,
            cutoff_year=CURRENT_YEAR
        )
        logger.info(f"Built position profiles for {len(position_profiles)}/{len(batter_ids)} batters")
        
        # Log position distribution
        from collections import Counter
        pos_counts = Counter(
            get_display_position(position_profiles.get(pid))
            for pid in batter_ids
        )
        logger.info(f"Position distribution: {dict(pos_counts.most_common())}")
        
        logger.info("Calculating WAR components (Off, BsR, Def, Position)...")
        war_components_list = []
        for idx, row in batter_data.iterrows():
            try:
                war, components = calculate_war_components(
                    row, baserunning_data, fielding_data,
                    position_profiles=position_profiles
                )
                components['IDfg'] = row['IDfg']
                components['Year'] = row['Year']
                war_components_list.append(components)
            except Exception as e:
                logger.error(f"Error calculating WAR for {row.get('Name', 'Unknown')} ({row['IDfg']}): {e}")
                # Use position profile for fallback instead of hardcoded 'OF'
                fallback_pos = get_display_position(position_profiles.get(row['IDfg']))
                war_components_list.append({
                    'IDfg': row['IDfg'],
                    'Year': row['Year'],
                    'WAR': 0.0,
                    'Bat': 0.0,
                    'BsR': 0.0,
                    'Fld': 0.0,
                    'Pos': 0.0,
                    'Def': 0.0,
                    'Position': fallback_pos,
                    'PA': 630,
                    'G': 150
                })
        
        # Merge WAR components back into batter data
        war_df = pd.DataFrame(war_components_list)
        batter_data = batter_data.merge(war_df, on=['IDfg', 'Year'], how='left', suffixes=('_old', ''))
        
        # Clean up duplicate columns
        columns_to_remove = [col for col in batter_data.columns if col.endswith('_old')]
        batter_data = batter_data.drop(columns=columns_to_remove)

        # Compute batter counting stats from rate stats and PA
        if 'BB%' in batter_data.columns and 'PA' in batter_data.columns:
            batter_data['BB_bat'] = (batter_data['BB%'] * batter_data['PA']).round().astype(int)
        if 'K%' in batter_data.columns and 'PA' in batter_data.columns:
            batter_data['K_bat'] = (batter_data['K%'] * batter_data['PA']).round().astype(int)

        logger.info(f"Calculated WAR components for {len(batter_data)} batters")
        logger.info(f"Batter WAR: avg={batter_data['WAR'].mean():.2f}")
        
        # ============================================================
        # Step 3: Merge Prediction Data
        # ============================================================
        logger.info("\n[Step 3/10] Merging prediction data...")
        player_predictions = merge_prediction_data(sp_data, rp_data, batter_data)
        
        # ============================================================
        # Step 4: Clean Salary Data
        # ============================================================
        logger.info("\n[Step 4/10] Cleaning salary data...")
        salary_data_clean = clean_salary_data(salary_data)
        
        logger.debug(f"Cleaned salary data: {len(salary_data_clean)} records")
        logger.debug(f"Null values: {salary_data_clean.isnull().sum().to_dict()}")
        
        # ============================================================
        # Step 5: Merge Salary with Player IDs
        # ============================================================
        logger.info("\n[Step 5/10] Integrating player IDs with salary data...")
        # TODO: Transition from FG ID matching to MLB ID matching
        salary_data_with_id = merge_salary_with_ids(
            salary_data_clean, sp_data, rp_data, batter_data
        )
        
        # Log unmatched players
        unmatched = salary_data_with_id[salary_data_with_id['IDfg'].isna()]
        if not unmatched.empty and Config.Pipeline.LOG_UNMATCHED_PLAYERS:
            logger.warning(f"{len(unmatched)} players could not be matched to predictions")

        # Attach canonical team from current_rosters.csv.
        # Salary data is unreliable for team assignment: traded players appear under
        # multiple teams in the Spotrac scrape (e.g. Devers under both Red Sox and
        # Giants). The roster file is the single source of truth for current team.
        # Free agents (not in roster) will have Team=NaN → displayed as 'FA' in output.
        salary_data_with_id = salary_data_with_id.drop(columns=['Team'], errors='ignore')
        salary_data_with_id = salary_data_with_id.merge(
            org_data[['IDfg', 'Team']].drop_duplicates('IDfg'),
            on='IDfg',
            how='left'
        )
        roster_matched = salary_data_with_id['Team'].notna().sum()
        logger.info(
            f"Team assignment from roster: {roster_matched} players matched. "
            f"Unmatched (free agents / not on 40-man): "
            f"{salary_data_with_id['IDfg'].notna().sum() - roster_matched}"
        )

        # ── Fill missing Years_of_Service from historical MLB data ────────
        # Players on current rosters but without Spotrac salary data have
        # no Years_of_Service.  Estimate it from how many MLB seasons
        # appear in the FanGraphs historical leaderboards so that the
        # contract-timeline generator can place them correctly on the
        # Pre-Arb → Arb → FA progression.
        # IMPORTANT: Only fill for players with NO Spotrac YoS at all.
        # Players who have real Spotrac YoS on some rows must keep those
        # values — the timeline generator picks the best value itself.
        try:
            batting_history, pitching_history = load_historical_data()
        except Exception as e:
            logger.warning(f"Could not load historical data for YoS estimation: {e}")
            batting_history, pitching_history = None, None

        # Find players where ALL rows have NaN YoS (no Spotrac data at all)
        salary_data_with_id = complete_years_of_service(
            salary_data_with_id,
            batting_history=batting_history,
            pitching_history=pitching_history,
            current_year=CURRENT_YEAR,
        )

        # ============================================================
        # Step 6: Normalize Contract Status & Generate Timeline
        # ============================================================
        logger.info("\n[Step 6/10] Processing contracts and generating timeline...")
        contract_data = normalize_contract_status(salary_data_with_id)
        
        contract_timeline = generate_contract_timeline(contract_data)
        logger.info(f"Generated {len(contract_timeline)} timeline records")
        
        missing_fa = validate_fa_years(contract_timeline)
        extended_timeline = extend_fa_timeline(contract_timeline)
        
        # Check for any remaining None statuses after timeline generation
        problem_cases = check_none_statuses(extended_timeline)
        
        # ============================================================
        # Step 7: Calculate Values
        # ============================================================
        logger.info("\n[Step 7/10] Calculating WAR and contract values...")
        timeline_with_war = join_predictions_with_timeline(
            extended_timeline, player_predictions
        )
        timeline_with_values = calculate_contract_value(timeline_with_war)
        timeline_with_values = calculate_surplus_value(timeline_with_values)
        
        # ============================================================
        # Step 8: Integrate Historical Data
        # ============================================================
        logger.info("\n[Step 8/10] Integrating historical data...")
        try:
            # batting_history / pitching_history already loaded before Step 6
            if batting_history is None and pitching_history is None:
                batting_history, pitching_history = load_historical_data()
            timeline_with_history = integrate_historical_stats(
                timeline_with_values, batting_history, pitching_history
            )
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
            timeline_with_history = timeline_with_values
        
        export_data = integrate_player_statistics(
            timeline_with_history, batter_data, sp_data, rp_data
        )
        export_data = post_process_export_data(export_data)
        logger.info(f"Processed {len(export_data)} total records")
        
        # ============================================================
        # Step 9: Calculate Trade Values
        # ============================================================
        logger.info("\n[Step 9/10] Calculating trade values with prospect adjustments...")
        export_data = analyze_contract_options(export_data)
        export_data = calculate_trade_values(export_data)
        export_data = add_trade_ranking_metrics(export_data)
        
        # Update prospect MLB status
        try:
            update_prospect_mlb_status(export_data)
        except Exception as e:
            logger.warning(f"Could not update prospect status: {e}")
        
        # ============================================================
        # Step 10: Export Data
        # ============================================================
        logger.info("\n[Step 10/10] Exporting value data...")
        export_value_data(export_data, OUTPUT_DIR, filename=output_filename)

        # ── Export fielding projections for the web database ──────────
        logger.info("Exporting fielding projections (per-position, with innings)...")
        _export_fielding_projections(
            fielding_data, position_profiles, org_data,
            batter_data, OUTPUT_DIR.parent / "pipeline",
        )

        # ── Append to trade-value history (daily snapshot) ────────────
        # Only for the default output (player_values_complete.csv), not
        # preseason or other alternate outputs.
        if output_filename is None:
            try:
                from value_determination.pipelines.snapshots import save_daily_trade_value_snapshot
                logger.info("Saving daily trade-value snapshot...")
                save_daily_trade_value_snapshot()
            except Exception as e:
                logger.warning(f"Could not save trade-value snapshot: {e}")

        # Log final summary
        if 'Year' in export_data.columns and 'Status' in export_data.columns:
            year_2025 = export_data[export_data['Year'] == 2025]
            logger.info(f"2025 records: {len(year_2025)}")
            logger.info(f"Unique players: {year_2025['IDfg'].nunique()}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Output saved to: {OUTPUT_DIR / (output_filename or 'player_values_complete.csv')}")
        logger.info("=" * 60)
        
        return export_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
