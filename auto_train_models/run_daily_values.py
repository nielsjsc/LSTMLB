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
    clean_salary_data, merge_salary_with_ids, complete_years_of_service,
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
    derive_missing_batter_baselines,
    derive_missing_pitcher_baselines,
    derive_missing_fielding_baseline,
    _team_remaining_fraction,
)
from value_determination.pipelines.snapshots import save_daily_trade_value_snapshot


def _backfill_proration_for_derived_rows(
    new_rows, proration_dict, team_games_map, player_team_map,
):
    """Add war_proration entries for mid-season call-ups / signings.

    derive_missing_batter_baselines() / derive_missing_pitcher_baselines()
    add CURRENT_YEAR rows for players who had no preseason projection —
    but that happens *after* blend_batter_projections()/blend_pitcher_
    projections() already built the war_proration dict, so these players
    are never seen by that loop and get no entry.

    Without an entry, reduce_to_remaining_season() skips them entirely
    (it does `if info is None: continue`), so their counting stats
    (including G) are left at the full-season value baked into the
    derived baseline instead of being scaled down to the games remaining.

    This backfills a team-based remaining_fraction for exactly the IDs
    that are missing, so they get reduced like everyone else.
    """
    if new_rows.empty:
        return

    use_team_remaining = bool(team_games_map and player_team_map)

    for idfg in new_rows['IDfg'].dropna().astype(int):
        if idfg in proration_dict:
            continue  # already has an entry — don't clobber it

        if use_team_remaining:
            team = player_team_map.get(idfg)
            remaining_frac = _team_remaining_fraction(team, team_games_map)
        else:
            remaining_frac = 1.0

        proration_dict[idfg] = {
            'actual_war': 0.0,
            'remaining_fraction': remaining_frac,
        }


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

        # --- PATCH: load actual fielding for call-up Pos resolution ---
        try:
            # Use already-imported Config and Path (don't re-import to avoid UnboundLocalError)
            current_season_dir = Config.Paths.DATA_DIR / 'current_season'
            fld_path = current_season_dir / f"mlb_fielding_data_{CURRENT_YEAR}_{CURRENT_YEAR}.csv"
            if fld_path.exists():
                actual_fielding = pd.read_csv(fld_path, low_memory=False)
            else:
                hist_dir = Config.Paths.HISTORIC_MLB_DIR
                hf = pd.read_csv(hist_dir / 'mlb_fielding_data_1950_2025_with_statcast.csv', low_memory=False, usecols=lambda c: c in ('IDfg','Season','Pos','Inn','InnOuts'))
                actual_fielding = hf[hf['Season']==CURRENT_YEAR].copy() if 'Season' in hf.columns else pd.DataFrame()
        except Exception as _e:
            actual_fielding = pd.DataFrame()
            logger.warning(f"Could not load actual_fielding for patch: {_e}")


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
        # Step 1.6 (NEW): Derive current-year baselines for players
        # with no preseason projection — mid-season call-ups / signings.
        #
        # These players have no CURRENT_YEAR row (Round 1 predates their
        # debut) but DO have a (CURRENT_YEAR + 1) row (Round 2 already
        # projected them forward using their real debut-season stats).
        # We recover the current-year talent estimate by reversing the
        # one year of aging Marcel applied to build that next-year row,
        # using the same aging curves — so the result is methodologically
        # identical to every other player's preseason row.
        # ============================================================
        logger.info("\n[Step 1.6] Deriving baselines for players with no preseason row...")
 
        new_batter_rows = derive_missing_batter_baselines(batter_data, CURRENT_YEAR, actual_batting=actual_batting)
        if not new_batter_rows.empty:
            batter_data = pd.concat([batter_data, new_batter_rows], ignore_index=True)
            _backfill_proration_for_derived_rows(
                new_batter_rows, batter_proration, team_games_map, player_team_map,
            )

        # blend_fielding_projections() (Step 1.5) only blends actuals into
        # fielding_data rows that already exist for CURRENT_YEAR — it can't
        # create a row for a player who has none. A mid-season call-up
        # (Round 1 predates their debut) has ZERO fielding_data rows for
        # any year, so calculate_defensive_value() finds nothing for them
        # at every year it's asked about, weighted_fld is permanently 0.0,
        # and Def collapses onto the positional adjustment alone — the
        # same flat number on every projection row. Derive a real
        # current-year baseline for them now that batter_data (with
        # call-ups concatenated above) tells us the full current-year
        # batter ID set.
        current_year_batter_ids = batter_data.loc[
            batter_data['Year'] == CURRENT_YEAR, 'IDfg'
        ].dropna().astype(int)
        new_fielding_rows = derive_missing_fielding_baseline(
            fielding_data, current_year_batter_ids, actual_batting,
            current_year=CURRENT_YEAR,
            actual_fielding=actual_fielding,
        )
        if not new_fielding_rows.empty:
            fielding_data = pd.concat(
                [fielding_data, new_fielding_rows], ignore_index=True,
            )
 
        new_sp_rows = derive_missing_pitcher_baselines(sp_data, CURRENT_YEAR, 'SP', actual_pitching=actual_pitching)
        if not new_sp_rows.empty:
            sp_data = pd.concat([sp_data, new_sp_rows], ignore_index=True)
            _backfill_proration_for_derived_rows(
                new_sp_rows, pitcher_proration, team_games_map, player_team_map,
            )
 
        new_rp_rows = derive_missing_pitcher_baselines(rp_data, CURRENT_YEAR, 'RP', actual_pitching=actual_pitching)
        if not new_rp_rows.empty:
            rp_data = pd.concat([rp_data, new_rp_rows], ignore_index=True)
            _backfill_proration_for_derived_rows(
                new_rp_rows, pitcher_proration, team_games_map, player_team_map,
            )
 
        # Rebuild the combined dict now that call-up/signing entries have
        # been backfilled into batter_proration / pitcher_proration above —
        # war_proration was first assembled in Step 1.5, before any of
        # these IDs existed.
        war_proration = {**batter_proration, **pitcher_proration}
 
        n_derived = (
            len(new_batter_rows) + len(new_sp_rows) + len(new_rp_rows)
            + len(new_fielding_rows)
        )
        if n_derived:
            logger.info(f"  Derived {n_derived} current-year baseline rows total "
                        f"({len(new_batter_rows)} batters, {len(new_sp_rows)} SP, "
                        f"{len(new_rp_rows)} RP, {len(new_fielding_rows)} fielding)")
        else:
            logger.info("  No players needed a derived baseline")
 
        # Residual gap: a player missing from BOTH CURRENT_YEAR and
        # CURRENT_YEAR+1 (e.g. debuted with too few PA/IP to clear Round 2's
        # own inclusion threshold yet) still won't get a row this run. This
        # is expected to be rare and self-resolving — they'll be picked up
        # automatically once they clear that threshold on a later day.
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

        logger.info("Building position profiles from historical and actual fielding data...")
        hist_fielding = load_fielding_history()
        if 'actual_fielding' in locals() and not actual_fielding.empty:
            hist_fielding = pd.concat([hist_fielding, actual_fielding], ignore_index=True)

        hist_batting = load_batting_for_games()
        if 'actual_batting' in locals() and not actual_batting.empty:
            hist_batting = pd.concat([hist_batting, actual_batting], ignore_index=True)

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

        salary_data_with_id = complete_years_of_service(
            salary_data_with_id,
            batting_history=batting_history,
            pitching_history=pitching_history,
            current_year=CURRENT_YEAR,
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

        # ============================================================
        # Step 6.5 (NEW): Prorate current-year salary
        # ============================================================
        # Must happen AFTER calculate_contract_value() so Contract_Value
        # column exists, but BEFORE calculate_surplus_value() uses it
        logger.info("\n[Step 6.5] Prorating current-year salary to remaining season...")
        from value_determination.pipelines.ros import prorate_current_year_salary
        timeline_with_values = prorate_current_year_salary(
            timeline_with_values, war_proration, current_year=CURRENT_YEAR
        )

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

