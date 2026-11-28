"""
Unified MLB Trade Valuation Pipeline (Preseason & Daily ROS)
============================================================

Orchestrates the end-to-end V2 valuation pipeline in a single clean function
`run_valuation_pipeline(mode="preseason" | "ros")`, eliminating ~500 lines of
duplicated pipeline glue between the old `pipelines/current.py` and `run_daily_values.py`.

Steps:
    1. Load Marcel prediction CSVs & Spotrac salary data
    2. (ROS mode) Blend in-season actuals & derive baselines for new call-ups
    3. Estimate pitcher (`IP`, `GS`, `G`) and batter (`PA`, `G`) playing time
    4. Calculate pitcher FIP-WAR & batter WAR (with post-28 defensive spectrum drift)
    5. Clean Spotrac salaries (CBA cash vs post-FA deferral escrow) & match IDs
    6. Build normalized `Pre-Arb -> Arb -> FA` contract timeline
    7. Compute baseline `production_value`, `salary_cost`, and `annual_surplus`
    8. Integrate historical MLB stats & assemble two-way timelines
    9. Run vectorized 1,000-path Monte Carlo simulator (`trade_value`, `P10/P50/P90`)
   10. Refresh `prospect_values.csv`, export canonical `snake_case` CSVs, & snapshot
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from core.position_profiles import (
    build_position_profiles,
    get_display_position,
    load_batting_for_games,
    load_fielding_history,
)
from ..config import CURRENT_YEAR, OUTPUT_DIR, Config, logger
from ..contract_timeline import (
    check_none_statuses,
    extend_fa_timeline,
    generate_contract_timeline,
    normalize_contract_status,
    validate_fa_years,
)
from ..data_loader import load_historical_data, load_prediction_files, merge_prediction_data
from ..dollar_valuation import (
    calculate_annual_surplus,
    calculate_salary_cost,
    join_predictions_with_timeline,
)
from ..exporter import export_value_data
from ..monte_carlo_valuation import (
    add_control_horizon_metrics,
    analyze_contract_options,
    calculate_trade_values,
)
from ..playing_time import estimate_batter_playing_time, estimate_playing_time
from ..prospect_valuation import generate_prospect_values
from ..salary_cleaner import clean_salary_data, complete_years_of_service, merge_salary_with_ids
from ..timeline_assembler import assemble_player_timeline, integrate_historical_stats
from ..war_engine import (
    _apply_park_factors_to_batter_predictions,
    _apply_park_factors_to_pitcher_predictions,
    calculate_batter_war_dataframe,
    calculate_pitcher_war_dataframe,
    calculate_wrc_plus,
    load_player_orgs,
)


def validate_input_data(
    sp_data: pd.DataFrame,
    rp_data: pd.DataFrame,
    batter_data: pd.DataFrame,
    salary_data: pd.DataFrame,
) -> bool:
    """Verify required columns exist on upstream prediction and salary DataFrames."""
    errors = []
    for name, df in (("SP", sp_data), ("RP", rp_data)):
        missing = [c for c in Config.Columns.REQUIRED["pitcher_predictions"] if c not in df.columns]
        if missing:
            errors.append(f"{name} data missing columns: {missing}")

    missing_bat = [c for c in Config.Columns.REQUIRED["batter_predictions"] if c not in batter_data.columns]
    if missing_bat:
        errors.append(f"Batter data missing columns: {missing_bat}")

    sal_lower = {c.lower() for c in salary_data.columns}
    for col in Config.Columns.REQUIRED["salary"]:
        if col.lower() not in sal_lower:
            errors.append(f"Salary data missing column: {col}")

    if errors:
        for err in errors:
            logger.error(err)
        if Config.Pipeline.FAIL_ON_MISSING_DATA:
            raise ValueError(f"Input validation failed: {errors}")
        return False
    return True


def _export_fielding_projections(
    fielding_data: pd.DataFrame,
    position_profiles: dict,
    org_data: pd.DataFrame,
    batter_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export per-position fielding projections (`fielding_projections_complete.csv`)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if fielding_data.empty:
        return

    team_map = {}
    if org_data is not None and not org_data.empty:
        for _, r in org_data.drop_duplicates("IDfg").iterrows():
            team_map[r["IDfg"]] = r.get("Team", "")

    name_map = {}
    if batter_data is not None and not batter_data.empty:
        for idfg, grp in batter_data.groupby("IDfg"):
            name_map[idfg] = grp.iloc[0].get("Name", "")

    total_games = 150
    innings_per_game = 8.5
    rows = []
    for _, pred in fielding_data.iterrows():
        idfg = pred["IDfg"]
        pos = pred.get("Pos", "")
        profile = position_profiles.get(idfg, {})
        pos_frac = profile.get(pos, 0.0) if profile else 0.0
        est_games = round(total_games * pos_frac)
        rows.append({
            "IDfg": idfg,
            "Name": pred.get("Name") or name_map.get(idfg, ""),
            "Year": pred["Year"],
            "Team": team_map.get(idfg, ""),
            "Pos": pos,
            "Age": pred.get("Age"),
            "G": est_games,
            "GS": est_games,
            "Inn": round(est_games * innings_per_game, 1),
            "sc_total_runs": pred.get("sc_total_runs/150", pred.get("sc_total_runs")),
            "sc_range_runs": pred.get("sc_range_runs/150", pred.get("sc_range_runs")),
            "sc_arm_runs": pred.get("sc_arm_runs/150", pred.get("sc_arm_runs")),
            "sc_dp_runs": pred.get("sc_dp_runs/150", pred.get("sc_dp_runs")),
            "sc_framing_runs": pred.get("sc_framing_runs/150", pred.get("sc_framing_runs")),
            "sc_throwing_runs": pred.get("sc_throwing_runs/150", pred.get("sc_throwing_runs")),
            "sc_blocking_runs": pred.get("sc_blocking_runs/150", pred.get("sc_blocking_runs")),
        })

    out_path = output_dir / "fielding_projections_complete.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, na_rep="")
    logger.info("Exported %d fielding projection rows to %s", len(rows), out_path)


def _backfill_proration_for_derived_rows(
    new_rows: pd.DataFrame,
    proration_dict: dict,
    team_games_map: dict,
    player_team_map: dict,
    remaining_fn,
) -> None:
    if new_rows.empty:
        return
    use_team = bool(team_games_map and player_team_map)
    for idfg in new_rows["IDfg"].dropna().astype(int):
        if idfg in proration_dict:
            continue
        rem_frac = remaining_fn(player_team_map.get(idfg), team_games_map) if use_team else 1.0
        proration_dict[idfg] = {"actual_war": 0.0, "remaining_fraction": rem_frac}


def run_valuation_pipeline(
    mode: Literal["preseason", "ros"] = "preseason",
    pipeline_dir: Optional[Path | str] = None,
    output_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Execute the unified V2 MLB valuation pipeline in either `preseason` or `ros` mode.
    """
    is_ros = (mode == "ros")
    logger.info("=" * 64)
    logger.info("MLB Trade Valuation Engine V2 — mode=%s", mode.upper())
    logger.info("=" * 64)

    Config.Paths.ensure_directories()

    # ── Step 1: Load predictions & salary data ───────────────────────────────
    sp_data, rp_data, batter_data, baserunning_data, fielding_data, salary_data = load_prediction_files(
        pipeline_dir=Path(pipeline_dir) if pipeline_dir else None
    )
    validate_input_data(sp_data, rp_data, batter_data, salary_data)
    org_data = load_player_orgs()

    # ── Step 1.5 (ROS only): Blend current-season actuals & call-up baselines ─
    war_proration: dict = {}
    batter_proration: dict = {}
    pitcher_proration: dict = {}
    actual_batting = pd.DataFrame()
    actual_fielding = pd.DataFrame()

    if is_ros:
        from .ros import (
            _team_remaining_fraction,
            blend_baserunning_projections,
            blend_batter_projections,
            blend_fielding_projections,
            blend_pitcher_projections,
            derive_missing_batter_baselines,
            derive_missing_fielding_baseline,
            derive_missing_pitcher_baselines,
            fetch_team_games_played,
            load_current_season_actuals,
            prorate_current_year_salary,
            prorate_current_year_war,
            reduce_to_remaining_season,
        )

        actual_batting, actual_pitching, _ = load_current_season_actuals()
        fld_path = Config.Paths.DATA_DIR / "current_season" / f"mlb_fielding_data_{CURRENT_YEAR}_{CURRENT_YEAR}.csv"
        if fld_path.exists():
            actual_fielding = pd.read_csv(fld_path, low_memory=False)

        team_games_map = fetch_team_games_played(season=CURRENT_YEAR)
        player_team_map = dict(zip(org_data["IDfg"].astype(int), org_data["Team"]))

        batter_data, batter_proration = blend_batter_projections(
            batter_data, actual_batting, current_year=CURRENT_YEAR,
            team_games_map=team_games_map, player_team_map=player_team_map,
        )
        sp_data, rp_data, pitcher_proration = blend_pitcher_projections(
            sp_data, rp_data, actual_pitching, current_year=CURRENT_YEAR,
            team_games_map=team_games_map, player_team_map=player_team_map,
        )
        fielding_data = blend_fielding_projections(fielding_data, actual_batting, current_year=CURRENT_YEAR)
        baserunning_data = blend_baserunning_projections(baserunning_data, actual_batting, current_year=CURRENT_YEAR)

        new_bat = derive_missing_batter_baselines(batter_data, CURRENT_YEAR, actual_batting=actual_batting)
        if not new_bat.empty:
            new_bat["player_role"] = "Batter"
            new_bat["position_group"] = "Batter"
            batter_data = pd.concat([batter_data, new_bat], ignore_index=True)
            _backfill_proration_for_derived_rows(
                new_bat, batter_proration, team_games_map, player_team_map, _team_remaining_fraction
            )

        cur_bat_ids = batter_data.loc[batter_data["Year"] == CURRENT_YEAR, "IDfg"].dropna().astype(int)
        new_fld = derive_missing_fielding_baseline(
            fielding_data, cur_bat_ids, actual_batting, current_year=CURRENT_YEAR, actual_fielding=actual_fielding
        )
        if not new_fld.empty:
            fielding_data = pd.concat([fielding_data, new_fld], ignore_index=True)

        for role_label, df_target in (("SP", "sp"), ("RP", "rp")):
            src_df = sp_data if df_target == "sp" else rp_data
            new_pit = derive_missing_pitcher_baselines(src_df, CURRENT_YEAR, role_label, actual_pitching=actual_pitching)
            if not new_pit.empty:
                new_pit["player_role"] = role_label
                new_pit["position_group"] = role_label
                new_pit["Position"] = role_label
                if df_target == "sp":
                    sp_data = pd.concat([sp_data, new_pit], ignore_index=True)
                else:
                    rp_data = pd.concat([rp_data, new_pit], ignore_index=True)
                _backfill_proration_for_derived_rows(
                    new_pit, pitcher_proration, team_games_map, player_team_map, _team_remaining_fraction
                )

        war_proration = {**batter_proration, **pitcher_proration}

    # ── Step 2: Pitcher playing time & FIP-WAR ───────────────────────────────
    sp_data = sp_data.merge(org_data[["IDfg", "Team"]], on="IDfg", how="left")
    rp_data = rp_data.merge(org_data[["IDfg", "Team"]], on="IDfg", how="left")

    combined_pitcher = pd.concat([sp_data, rp_data], ignore_index=True)
    combined_pitcher = _apply_park_factors_to_pitcher_predictions(combined_pitcher)

    year_chunks = []
    for proj_year in sorted(combined_pitcher["Year"].unique()):
        chunk = combined_pitcher[combined_pitcher["Year"] == proj_year].copy()
        chunk = estimate_playing_time(
            chunk, int(proj_year), anchor_year=CURRENT_YEAR, include_anchor_season=is_ros
        )
        year_chunks.append(chunk)
    combined_pitcher = pd.concat(year_chunks, ignore_index=True)

    sp_data = combined_pitcher[combined_pitcher["Role"] == "SP"].copy()
    rp_data = combined_pitcher[combined_pitcher["Role"] == "RP"].copy()

    if is_ros:
        from .ros import reduce_to_remaining_season
        sp_data = reduce_to_remaining_season(sp_data, pitcher_proration, player_type="pitcher")
        rp_data = reduce_to_remaining_season(rp_data, pitcher_proration, player_type="pitcher")

    sp_data = calculate_pitcher_war_dataframe(sp_data, org_data=org_data, role="SP", compute_counting=True)
    rp_data = calculate_pitcher_war_dataframe(rp_data, org_data=org_data, role="RP", compute_counting=True)

    # Filter pitchers with legacy NL batting history who no longer bat
    pitcher_idfgs = set(sp_data["IDfg"].unique()) | set(rp_data["IDfg"].unique())
    pitchers_in_batters = pitcher_idfgs & set(batter_data["IDfg"].unique())
    if pitchers_in_batters and Config.Paths.HISTORIC_BATTING_FILE_CLASSIC.exists():
        hist_bat = pd.read_csv(
            Config.Paths.HISTORIC_BATTING_FILE_CLASSIC,
            usecols=["IDfg", "Season", "PA"],
            low_memory=False,
        )
        hist_bat["PA"] = pd.to_numeric(hist_bat["PA"], errors="coerce").fillna(0)
        recent_batters = set(
            hist_bat.loc[
                (hist_bat["Season"] >= CURRENT_YEAR - 1) & (hist_bat["PA"] >= 10),
                "IDfg",
            ].unique()
        )
        # Also check current-season actual batting for genuine two-way players (e.g. Ohtani)
        if not actual_batting.empty and "PA" in actual_batting.columns:
            cur_tw = actual_batting.loc[
                pd.to_numeric(actual_batting["PA"], errors="coerce").fillna(0) >= 10, "IDfg"
            ].dropna().astype(int)
            recent_batters |= set(cur_tw)
        pitchers_to_remove = pitchers_in_batters - recent_batters
        if pitchers_to_remove:
            batter_data = batter_data[~batter_data["IDfg"].isin(pitchers_to_remove)].copy()

        # Symmetrically remove primary batters who only pitched mop-up blowout innings (<15 IP, 0 GS)
        remaining_two_way = (set(sp_data["IDfg"].unique()) | set(rp_data["IDfg"].unique())) & set(batter_data["IDfg"].unique())
        if remaining_two_way and Config.Paths.HISTORIC_PITCHING_FILE.exists():
            hist_pit_tw = pd.read_csv(
                Config.Paths.HISTORIC_PITCHING_FILE,
                usecols=["IDfg", "Season", "IP", "GS"],
                low_memory=False,
            )
            hist_pit_tw = hist_pit_tw[hist_pit_tw["Season"] <= CURRENT_YEAR]
            mopup_batters = set()
            for tw_id in remaining_two_way:
                sub_p = hist_pit_tw[hist_pit_tw["IDfg"] == tw_id]
                max_ip = float(pd.to_numeric(sub_p["IP"], errors="coerce").max() or 0.0) if not sub_p.empty else 0.0
                max_gs = float(pd.to_numeric(sub_p["GS"], errors="coerce").max() or 0.0) if not sub_p.empty else 0.0
                if max_ip < 15.0 and max_gs < 1.0:
                    mopup_batters.add(int(tw_id))
            if mopup_batters:
                sp_data = sp_data[~sp_data["IDfg"].isin(mopup_batters)].copy()
                rp_data = rp_data[~rp_data["IDfg"].isin(mopup_batters)].copy()
                logger.info("Removed %d mop-up position players from pitcher projections", len(mopup_batters))

    # ── Step 3: Batter playing time, spectrum drift, & WAR ───────────────────
    batter_data = batter_data.merge(org_data, on="IDfg", how="left", suffixes=("", "_org"))
    batter_data = _apply_park_factors_to_batter_predictions(batter_data)

    hist_fielding = load_fielding_history()
    if is_ros and not actual_fielding.empty:
        hist_fielding = pd.concat([hist_fielding, actual_fielding], ignore_index=True)

    hist_batting = load_batting_for_games()
    if is_ros and not actual_batting.empty:
        hist_batting = pd.concat([hist_batting, actual_batting], ignore_index=True)

    batter_ids = batter_data["IDfg"].unique().tolist()
    position_profiles = build_position_profiles(
        hist_fielding, hist_batting, batter_ids, cutoff_year=CURRENT_YEAR
    )
    pos_counts = Counter(get_display_position(position_profiles.get(pid)) for pid in batter_ids)
    logger.info("Position distribution: %s", dict(pos_counts.most_common()))

    # Estimate individualized batter PA & G (skill- & usage-earned)
    batter_data = estimate_batter_playing_time(
        batter_data,
        anchor_year=CURRENT_YEAR,
        position_profiles=position_profiles,
        include_anchor_season=is_ros,
    )

    if is_ros:
        from .ros import prorate_current_year_war, reduce_to_remaining_season
        batter_data = reduce_to_remaining_season(batter_data, batter_proration, player_type="batter")

    batter_data["wRC+"] = [
        calculate_wrc_plus(w, t if pd.notna(t) else "", p)
        for w, t, p in zip(batter_data["wOBA"], batter_data.get("Team", ""), batter_data["PA"])
    ]

    batter_data = calculate_batter_war_dataframe(
        batter_data,
        fielding_df=fielding_data,
        baserunning_df=baserunning_data,
        position_profiles=position_profiles,
        include_components=True,
    )

    if "BB%" in batter_data.columns and "PA" in batter_data.columns:
        batter_data["BB_bat"] = (
            pd.to_numeric(batter_data["BB%"], errors="coerce").fillna(0.0)
            * pd.to_numeric(batter_data["PA"], errors="coerce").fillna(0.0)
        ).round().astype(int)
    if "K%" in batter_data.columns and "PA" in batter_data.columns:
        batter_data["K_bat"] = (
            pd.to_numeric(batter_data["K%"], errors="coerce").fillna(0.0)
            * pd.to_numeric(batter_data["PA"], errors="coerce").fillna(0.0)
        ).round().astype(int)

    if is_ros:
        from .ros import prorate_current_year_war
        batter_data = prorate_current_year_war(batter_data, war_proration)
        sp_data = prorate_current_year_war(sp_data, war_proration)
        rp_data = prorate_current_year_war(rp_data, war_proration)

    # ── Step 4: Merge predictions & clean Spotrac contracts ──────────────────
    player_predictions = merge_prediction_data(sp_data, rp_data, batter_data)
    salary_clean = clean_salary_data(salary_data)
    salary_with_id = merge_salary_with_ids(
        salary_clean, sp_data, rp_data, batter_data, org_data=org_data
    )
    if not is_ros:
        predicted_ids = set(player_predictions["IDfg"].dropna().astype(int))
        salary_with_id = salary_with_id[salary_with_id["IDfg"].isin(predicted_ids)].copy()

    salary_with_id = salary_with_id.drop(columns=["Team"], errors="ignore")
    salary_with_id = salary_with_id.merge(
        org_data[["IDfg", "Team"]].drop_duplicates("IDfg"),
        on="IDfg",
        how="left",
    )

    try:
        batting_history, pitching_history = load_historical_data()
    except Exception as exc:
        logger.warning("Could not load historical MLB data: %s", exc)
        batting_history, pitching_history = None, None

    salary_with_id = complete_years_of_service(
        salary_with_id,
        batting_history=batting_history,
        pitching_history=pitching_history,
        current_year=CURRENT_YEAR,
    )

    # ── Step 5: Build contract timeline & baseline annual values ─────────────
    contract_data = normalize_contract_status(salary_with_id)
    contract_timeline = generate_contract_timeline(contract_data, current_year=CURRENT_YEAR)
    validate_fa_years(contract_timeline)
    extended_timeline = extend_fa_timeline(contract_timeline)
    check_none_statuses(extended_timeline)

    timeline_with_war = join_predictions_with_timeline(extended_timeline, player_predictions)
    timeline_with_values = calculate_salary_cost(timeline_with_war)

    if is_ros:
        from .ros import prorate_current_year_salary
        timeline_with_values = prorate_current_year_salary(
            timeline_with_values, war_proration, current_year=CURRENT_YEAR
        )

    timeline_with_values = calculate_annual_surplus(timeline_with_values)

    # ── Step 6: Integrate historical stats & assemble player timeline ────────
    if batting_history is not None and pitching_history is not None:
        timeline_with_history = integrate_historical_stats(
            timeline_with_values,
            batting_history,
            pitching_history,
            current_year=CURRENT_YEAR,
        )
    else:
        timeline_with_history = timeline_with_values

    assembled = assemble_player_timeline(
        timeline_with_history,
        batter_data,
        sp_data,
        rp_data,
        current_year=CURRENT_YEAR,
    )

    # ── Step 7: Vectorized Monte Carlo simulation & horizon metrics ──────────
    assembled = analyze_contract_options(assembled, current_year=CURRENT_YEAR)
    assembled = calculate_trade_values(
        assembled,
        current_year=CURRENT_YEAR,
        apply_ros_fraction=is_ros,
    )
    assembled = add_control_horizon_metrics(
        assembled,
        current_year=CURRENT_YEAR,
        apply_ros_fraction=is_ros,
    )

    # ── Step 8: Refresh unified prospect valuations & export CSVs ────────────
    try:
        generate_prospect_values()
    except Exception as exc:
        logger.warning("Could not refresh prospect_values.csv: %s", exc)

    exported_df = export_value_data(assembled, OUTPUT_DIR, filename=output_filename)

    _export_fielding_projections(
        fielding_data,
        position_profiles,
        org_data,
        batter_data,
        OUTPUT_DIR,
    )

    if output_filename is None:
        try:
            from .snapshots import save_daily_trade_value_snapshot
            save_daily_trade_value_snapshot()
        except Exception as exc:
            logger.warning("Could not save daily trade-value snapshot: %s", exc)

    logger.info("Pipeline completed -> %s", OUTPUT_DIR / (output_filename or "player_values_complete.csv"))
    return exported_df


def main(pipeline_dir=None, output_filename=None) -> pd.DataFrame:
    return run_valuation_pipeline(
        mode="preseason",
        pipeline_dir=pipeline_dir,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    main()
