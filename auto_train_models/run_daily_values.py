#!/usr/bin/env python
"""
Daily ROS Value Determination
==============================

Lightweight entry point that skips model inference.  Loads pre-season
predictions, blends in current-season actuals via Bayesian shrinkage,
prorates current-year WAR (actual + projected × remaining), then runs
the full value determination pipeline (salary → contracts → surplus →
trade values → export).

Designed to run in < 2 minutes (no LSTM inference, just arithmetic).

Pipeline:
    1.   Load pre-season predictions
    1.5  Load current-season actuals & blend ROS projections
    2.   Calculate pitcher WAR (park factors, playing time, FIP-WAR)
    2.25 MiLB regression for low-sample batters
    2.5  Calculate batter WAR (wRC+, BsR, Def, positional adjustment)
    2.6  Prorate current-year WAR (actual + projected × remaining)
    3–10 Standard pipeline (merge, salary, contracts, surplus, trade values, export)

Usage:
    cd auto_train_models
    python run_daily_values.py
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure auto_train_models is on the path
sys.path.insert(0, str(Path(__file__).parent))

# ── Value determination imports ──────────────────────────────────────────
from value_determination.config import (
    Config, logger, CURRENT_YEAR, OUTPUT_DIR,
)
from value_determination.data_loader import (
    load_prediction_files, merge_prediction_data, load_historical_data,
)
from value_determination.salary_processor import (
    clean_salary_data, merge_salary_with_ids,
)
from value_determination.contract_processor import (
    normalize_contract_status, check_none_statuses,
    generate_contract_timeline, validate_fa_years, extend_fa_timeline,
)
from value_determination.value_calculator import (
    join_predictions_with_timeline, calculate_contract_value,
    calculate_surplus_value, integrate_historical_stats,
    integrate_player_statistics, post_process_export_data,
)
from value_determination.trade_value import (
    analyze_contract_options, calculate_trade_values,
    add_trade_ranking_metrics, update_prospect_mlb_status,
)
from value_determination.exporter import export_value_data
from value_determination.calculate_war import (
    calculate_war_components, load_player_orgs, calculate_wrc_plus,
)
from value_determination.playing_time import estimate_playing_time
from value_determination.milb_regression import apply_milb_regression
from core.position_profiles import (
    build_position_profiles, load_fielding_history, load_batting_for_games,
    get_display_position,
)

# Reuse helpers already defined in main.py
from value_determination.main import (
    calculate_pitcher_war_for_dataframe,
    validate_input_data,
    _export_fielding_projections,
)

# ── Daily ROS blending ──────────────────────────────────────────────────
from value_determination.pipelines.ros import (
    load_current_season_actuals,
    blend_batter_projections,
    blend_pitcher_projections,
    blend_fielding_projections,
    blend_baserunning_projections,
    reduce_to_remaining_season,
    prorate_current_year_war,
    fetch_team_games_played,
)
from value_determination.pipelines.snapshots import save_daily_trade_value_snapshot


def main():
    logger.info("=" * 60)
    logger.info("Daily ROS Value Determination  (skip model inference)")
    logger.info("=" * 60)

    Config.Paths.ensure_directories()

    try:
        # ============================================================
        # Step 1: Load pre-season predictions
        # ============================================================
        logger.info("\n[Step 1/10] Loading pre-season prediction and salary data...")
        (sp_data, rp_data, batter_data,
         baserunning_data, fielding_data, salary_data) = load_prediction_files()
        validate_input_data(sp_data, rp_data, batter_data, salary_data)

        # ============================================================
        # Step 1.5 (NEW): Load actuals & blend ROS projections
        # ============================================================
        logger.info("\n[Step 1.5] Loading current-season actuals and blending ROS projections...")
        actual_batting, actual_pitching, actual_year = load_current_season_actuals()

        # Fetch team games played for team-based remaining fraction
        team_games_map = fetch_team_games_played(season=CURRENT_YEAR)
        org_data = load_player_orgs()
        org_data['IDfg'] = pd.to_numeric(org_data['IDfg'], errors='coerce')
        org_valid = org_data.dropna(subset=['IDfg'])
        player_team_map = dict(zip(
            org_valid['IDfg'].astype(int),
            org_valid['Team'],
        ))

        batter_data, batter_proration = blend_batter_projections(
            batter_data, actual_batting, current_year=CURRENT_YEAR,
            team_games_map=team_games_map, player_team_map=player_team_map,
        )
        sp_data, rp_data, pitcher_proration = blend_pitcher_projections(
            sp_data, rp_data, actual_pitching, current_year=CURRENT_YEAR,
            team_games_map=team_games_map, player_team_map=player_team_map,
        )
        war_proration = {**batter_proration, **pitcher_proration}

        # Blend fielding and baserunning projections with actuals
        fielding_data = blend_fielding_projections(
            fielding_data, actual_batting, current_year=CURRENT_YEAR,
        )
        baserunning_data = blend_baserunning_projections(
            baserunning_data, actual_batting, current_year=CURRENT_YEAR,
        )

        # ============================================================
        # Step 2: Calculate Pitcher WAR
        # ============================================================
        logger.info("\n[Step 2/10] Calculating pitcher WAR from FIP...")

        sp_data = sp_data.merge(
            org_data[['IDfg', 'Team']], on='IDfg', how='left',
        )
        rp_data = rp_data.merge(
            org_data[['IDfg', 'Team']], on='IDfg', how='left',
        )

        # Apply park factors to park-neutral predictions
        from value_determination.calculate_war import (
            _apply_park_factors_to_pitcher_predictions,
        )
        combined_pitcher = pd.concat([sp_data, rp_data], ignore_index=True)
        combined_pitcher = _apply_park_factors_to_pitcher_predictions(combined_pitcher)
        sp_data = combined_pitcher[combined_pitcher['Role'] == 'SP'].copy()
        rp_data = combined_pitcher[combined_pitcher['Role'] == 'RP'].copy()

        # Step 2a: Estimate pitcher playing time (IP, GS, G)
        logger.info("\n[Step 2a/10] Estimating pitcher playing time...")
        combined_for_pt = pd.concat([sp_data, rp_data], ignore_index=True)
        year_chunks = []
        for proj_year in sorted(combined_for_pt['Year'].unique()):
            chunk = combined_for_pt[combined_for_pt['Year'] == proj_year].copy()
            chunk = estimate_playing_time(chunk, int(proj_year))
            year_chunks.append(chunk)
        combined_for_pt = pd.concat(year_chunks, ignore_index=True)
        sp_data = combined_for_pt[combined_for_pt['Role'] == 'SP'].copy()
        rp_data = combined_for_pt[combined_for_pt['Role'] == 'RP'].copy()

        # Reduce pitcher playing time to remaining season
        sp_data = reduce_to_remaining_season(
            sp_data, pitcher_proration, player_type='pitcher',
        )
        rp_data = reduce_to_remaining_season(
            rp_data, pitcher_proration, player_type='pitcher',
        )

        # Calculate WAR for SP and RP
        sp_data = calculate_pitcher_war_for_dataframe(sp_data, org_data, role='SP')
        rp_data = calculate_pitcher_war_for_dataframe(rp_data, org_data, role='RP')

        logger.info(f"SP WAR: n={len(sp_data)}, avg={sp_data['WAR'].mean():.2f}")
        logger.info(f"RP WAR: n={len(rp_data)}, avg={rp_data['WAR'].mean():.2f}")

        # ============================================================
        # Step 2.25: MiLB Regression for Low-Sample Batters
        # ============================================================
        logger.info("\n[Step 2.25/10] Applying MiLB regression to batter predictions...")
        batter_data = apply_milb_regression(batter_data, CURRENT_YEAR)

        # Filter pitchers who no longer bat (universal DH era)
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

        batter_data = batter_data.merge(
            org_data, on='IDfg', how='left', suffixes=('', '_org'),
        )

        from value_determination.calculate_war import (
            _apply_park_factors_to_batter_predictions,
        )
        batter_data = _apply_park_factors_to_batter_predictions(batter_data)

        if 'PA' not in batter_data.columns:
            batter_data['PA'] = 650
        if 'G' not in batter_data.columns:
            batter_data['G'] = 150

        # Reduce batter playing time to remaining season
        batter_data = reduce_to_remaining_season(
            batter_data, batter_proration, player_type='batter',
        )

        logger.info("Calculating wRC+ with park factors...")
        batter_data['wRC+'] = batter_data.apply(
            lambda row: calculate_wrc_plus(
                row['wOBA'], row.get('Team', ''), row.get('PA', 630),
            ),
            axis=1,
        )

        logger.info("Building position profiles from historical fielding data...")
        hist_fielding = load_fielding_history()
        hist_batting = load_batting_for_games()
        batter_ids = batter_data['IDfg'].unique().tolist()
        position_profiles = build_position_profiles(
            hist_fielding, hist_batting, batter_ids,
            cutoff_year=CURRENT_YEAR,
        )
        logger.info(
            f"Built position profiles for "
            f"{len(position_profiles)}/{len(batter_ids)} batters"
        )

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
                    position_profiles=position_profiles,
                )
                components['IDfg'] = row['IDfg']
                components['Year'] = row['Year']
                war_components_list.append(components)
            except Exception as e:
                logger.error(
                    f"Error calculating WAR for "
                    f"{row.get('Name', 'Unknown')} ({row['IDfg']}): {e}"
                )
                fallback_pos = get_display_position(
                    position_profiles.get(row['IDfg'])
                )
                war_components_list.append({
                    'IDfg': row['IDfg'],
                    'Year': row['Year'],
                    'WAR': 0.0, 'Bat': 0.0, 'BsR': 0.0,
                    'Fld': 0.0, 'Pos': 0.0, 'Def': 0.0,
                    'Position': fallback_pos,
                    'PA': 630, 'G': 150,
                })

        war_df = pd.DataFrame(war_components_list)
        batter_data = batter_data.merge(
            war_df, on=['IDfg', 'Year'], how='left', suffixes=('_old', ''),
        )
        columns_to_remove = [
            col for col in batter_data.columns if col.endswith('_old')
        ]
        batter_data = batter_data.drop(columns=columns_to_remove)

        if 'BB%' in batter_data.columns and 'PA' in batter_data.columns:
            batter_data['BB_bat'] = (
                batter_data['BB%'] * batter_data['PA']
            ).round().astype(int)
        if 'K%' in batter_data.columns and 'PA' in batter_data.columns:
            batter_data['K_bat'] = (
                batter_data['K%'] * batter_data['PA']
            ).round().astype(int)

        logger.info(
            f"Calculated WAR for {len(batter_data)} batters, "
            f"avg={batter_data['WAR'].mean():.2f}"
        )

        # ============================================================
        # Step 2.6 (NEW): Prorate current-year WAR
        # ============================================================
        logger.info(
            "\n[Step 2.6] Prorating current-year WAR "
            "(actual + projected × remaining)..."
        )
        batter_data = prorate_current_year_war(batter_data, war_proration)
        sp_data = prorate_current_year_war(sp_data, war_proration)
        rp_data = prorate_current_year_war(rp_data, war_proration)

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

        # ============================================================
        # Step 5: Merge Salary with Player IDs
        # ============================================================
        logger.info("\n[Step 5/10] Integrating player IDs with salary data...")
        salary_data_with_id = merge_salary_with_ids(
            salary_data_clean, sp_data, rp_data, batter_data,
        )

        # Attach canonical team from roster
        salary_data_with_id = salary_data_with_id.drop(
            columns=['Team'], errors='ignore',
        )
        salary_data_with_id = salary_data_with_id.merge(
            org_data[['IDfg', 'Team']].drop_duplicates('IDfg'),
            on='IDfg', how='left',
        )

        # Fill missing Years_of_Service from historic data
        try:
            batting_history, pitching_history = load_historical_data()
        except Exception as e:
            logger.warning(
                f"Could not load historical data for YoS estimation: {e}"
            )
            batting_history, pitching_history = None, None

        players_with_any_yos = set(
            salary_data_with_id
            .loc[salary_data_with_id['Years_of_Service'].notna(), 'IDfg']
            .dropna().unique()
        )
        all_player_ids = set(
            salary_data_with_id['IDfg'].dropna().unique()
        )
        players_missing_all_yos = all_player_ids - players_with_any_yos

        yos_filled = 0
        for pid in players_missing_all_yos:
            bat_seasons = 0
            pit_seasons = 0
            if batting_history is not None:
                bat_seasons = int(
                    batting_history[
                        batting_history['IDfg'] == pid
                    ]['Season'].nunique()
                )
            if pitching_history is not None:
                pit_seasons = int(
                    pitching_history[
                        pitching_history['IDfg'] == pid
                    ]['Season'].nunique()
                )
            estimated_yos = max(bat_seasons, pit_seasons)
            if estimated_yos > 0:
                salary_data_with_id.loc[
                    salary_data_with_id['IDfg'] == pid,
                    'Years_of_Service',
                ] = float(estimated_yos)
                yos_filled += 1
        if yos_filled:
            logger.info(
                f"Filled Years_of_Service from historical data "
                f"for {yos_filled} players"
            )

        # ============================================================
        # Step 6: Contracts & Timeline
        # ============================================================
        logger.info("\n[Step 6/10] Processing contracts and generating timeline...")
        contract_data = normalize_contract_status(salary_data_with_id)
        contract_timeline = generate_contract_timeline(contract_data)
        logger.info(f"Generated {len(contract_timeline)} timeline records")

        missing_fa = validate_fa_years(contract_timeline)
        extended_timeline = extend_fa_timeline(contract_timeline)
        problem_cases = check_none_statuses(extended_timeline)

        # ============================================================
        # Step 7: Calculate Values
        # ============================================================
        logger.info("\n[Step 7/10] Calculating WAR and contract values...")
        timeline_with_war = join_predictions_with_timeline(
            extended_timeline, player_predictions,
        )
        timeline_with_values = calculate_contract_value(timeline_with_war)
        timeline_with_values = calculate_surplus_value(timeline_with_values)

        # ============================================================
        # Step 8: Integrate Historical Data
        # ============================================================
        logger.info("\n[Step 8/10] Integrating historical data...")
        try:
            if batting_history is None and pitching_history is None:
                batting_history, pitching_history = load_historical_data()
            timeline_with_history = integrate_historical_stats(
                timeline_with_values, batting_history, pitching_history,
            )
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
            timeline_with_history = timeline_with_values

        export_data = integrate_player_statistics(
            timeline_with_history, batter_data, sp_data, rp_data,
        )
        export_data = post_process_export_data(export_data)
        logger.info(f"Processed {len(export_data)} total records")

        # ============================================================
        # Step 9: Calculate Trade Values
        # ============================================================
        logger.info(
            "\n[Step 9/10] Calculating trade values "
            "with prospect adjustments..."
        )
        export_data = analyze_contract_options(export_data)
        export_data = calculate_trade_values(export_data)
        export_data = add_trade_ranking_metrics(export_data)

        try:
            update_prospect_mlb_status(export_data)
        except Exception as e:
            logger.warning(f"Could not update prospect status: {e}")

        # ============================================================
        # Step 10: Export
        # ============================================================
        logger.info("\n[Step 10/10] Exporting value data...")
        export_value_data(export_data, OUTPUT_DIR)

        # Export fielding projections for web database
        logger.info("Exporting fielding projections (per-position)...")
        _export_fielding_projections(
            fielding_data, position_profiles, org_data,
            batter_data, OUTPUT_DIR.parent / "pipeline",
        )

        # ============================================================
        # Step 10.5: Daily Trade-Value Snapshot
        # ============================================================
        logger.info("\n[Step 10.5] Saving daily trade-value snapshot...")
        save_daily_trade_value_snapshot()

        logger.info("\n" + "=" * 60)
        logger.info("Daily ROS pipeline completed successfully!")
        logger.info(f"Output: {OUTPUT_DIR / 'player_values_complete.csv'}")
        logger.info("=" * 60)

        return export_data

    except Exception as e:
        logger.error(f"Daily ROS pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
