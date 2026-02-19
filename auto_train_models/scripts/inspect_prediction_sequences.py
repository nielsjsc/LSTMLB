#!/usr/bin/env python3
"""
Inspect Prediction Sequences
==============================

Prints the EXACT sequences fed into the LSTM model during prediction, mirroring
the live production pipeline.  All steps are gated by the same config toggles
used in core/prediction.py — so what you see here is what the model sees.

Steps shown (batter, regression DISABLED):
    1. Raw historical data
    2. xStat substitution          [if USE_XWOBA/XBA/XSLG_FOR_PREDICTIONS=True]
    3. Park factor neutralization  [if ENABLE_PARK_FACTOR_ADJUSTMENT=True]
       (applied to all adjustable features, including x-stat positions)
    4. Scaled sequence (model input)

Steps shown (batter, regression ENABLED):
    1. Raw historical data
    2. xStat substitution          [if USE_XWOBA/XBA/XSLG_FOR_PREDICTIONS=True]
    3. Reliability regression
    4. Park factor neutralization  [if ENABLE_PARK_FACTOR_ADJUSTMENT=True]
       (applied to all adjustable features, including x-stat positions)
    5. Scaled sequence (model input)

Steps shown (pitcher):
    1. Raw historical data
    2. Reliability regression      [always on for pitchers]
    3. Park factor neutralization  [if ENABLE_PARK_FACTOR_ADJUSTMENT=True]
    4. Scaled sequence (model input)

Usage:
    python scripts/inspect_prediction_sequences.py --players "Aaron Judge" "Juan Soto"
    python scripts/inspect_prediction_sequences.py --player-ids 17350 20123
    python scripts/inspect_prediction_sequences.py --type pitcher --players "Gerrit Cole"
    python scripts/inspect_prediction_sequences.py --type batter --top 5
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from typing import List, Optional, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_processing import DataConfig, calculate_rate_stats
from core.prediction import load_model_from_checkpoint, generate_batter_names
from core.reliability import (
    regress_player_sequence,
    compute_regressed_career_mean,
    get_era_for_features,
    compute_league_priors_from_df,
    _get_stabilization_point,
    _estimate_volume,
    regress_single_value,
    SKIP_FEATURES,
)
from core.park_factors import get_park_factor, EXCLUDED_STATS


def format_value(val, feat):
    """Format a value for display based on the feature type."""
    if pd.isna(val):
        return "  NaN  "
    if feat in ('Age',):
        return f"{val:6.1f}"
    if feat in ('BB%', 'K%'):
        return f"{val:6.3f}"
    if feat in ('AVG', 'OBP', 'SLG', 'wOBA', 'xwOBA'):
        return f"{val:6.3f}"
    if feat in ('EV', 'FBv'):
        return f"{val:6.1f}"
    if feat in ('wRC+', 'Stuff+', 'Location+', 'Pitching+'):
        return f"{val:6.1f}"
    if feat in ('ERA', 'FIP', 'xFIP', 'SIERA', 'xERA'):
        return f"{val:6.2f}"
    if feat in ('SwStr%', 'CSW%', 'GB%', 'FB%', 'Contact%'):
        return f"{val:6.3f}"
    return f"{val:7.2f}"


def print_table(title: str, data: np.ndarray, features: List[str],
                seasons: List = None, extra_cols: Dict[str, List] = None):
    """Print a nicely formatted table of sequence values."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Header
    header = f"{'Seq':>4}"
    if seasons:
        header += f" {'Season':>6}"
    if extra_cols:
        for col_name in extra_cols:
            header += f" {col_name:>6}"
    for feat in features:
        header += f" {feat:>8}"
    print(header)
    print("-" * len(header))

    # Rows
    for i in range(data.shape[0]):
        row = f"{i:4d}"
        if seasons:
            row += f" {str(seasons[i]):>6}"
        if extra_cols:
            for col_name, col_vals in extra_cols.items():
                row += f" {str(col_vals[i]):>6}"
        for j, feat in enumerate(features):
            row += f" {format_value(data[i, j], feat):>8}"
        print(row)


def _print_config_banner(cfg_items: Dict[str, object]):
    """Print a box summarising the active config flags for this run."""
    print(f"\n{'=' * 80}")
    print(f"  ACTIVE CONFIG SETTINGS")
    print(f"{'=' * 80}")
    for key, val in cfg_items.items():
        flag = "✓ ON " if val is True else ("✗ OFF" if val is False else f"     {val}")
        print(f"  {flag}  {key}")
    print(f"{'=' * 80}")


def inspect_batter(player_id: int, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                   input_features: List[str], scaler, seq_length: int,
                   cutoff_year: int, league_priors: Optional[Dict] = None,
                   batter_config=None):
    """
    Inspect the full prediction pipeline for a single batter.

    Pipeline path is chosen to EXACTLY match core/prediction.py:
      - ENABLE_RELIABILITY_REGRESSION_PREDICTION=False  →  _prepare_player_sequence path
        Order: park-neutralize FIRST, then x-stats override
      - ENABLE_RELIABILITY_REGRESSION_PREDICTION=True   →  _predict_with_regression path
        Order: x-stats FIRST (tracked), then regression, then park-neutralize
               (x-stat-replaced features are skipped during neutralization)
    """
    # ---- Read config toggles ----
    cfg = batter_config
    use_regression   = getattr(cfg, 'ENABLE_RELIABILITY_REGRESSION_PREDICTION', False)
    use_pf           = getattr(cfg, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)
    use_xwoba        = getattr(cfg, 'USE_XWOBA_FOR_PREDICTIONS', False)
    use_xwoba_blend  = getattr(cfg, 'USE_XWOBA_BLEND_FOR_PREDICTIONS', False)
    use_xba          = getattr(cfg, 'USE_XBA_FOR_PREDICTIONS', False)
    use_xslg         = getattr(cfg, 'USE_XSLG_FOR_PREDICTIONS', False)

    player_data = raw_df[
        (raw_df['IDfg'] == player_id) &
        (raw_df['Season'] <= cutoff_year)
    ].copy()
    player_data = player_data[player_data['PA'] >= 50].sort_values('Season')

    if len(player_data) == 0:
        print(f"No data found for player ID {player_id}")
        return

    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        player_name = f"ID:{player_id}"

    print(f"\n{'#' * 80}")
    print(f"  BATTER: {player_name} (IDfg={player_id})")
    print(f"  Seasons available: {len(player_data)} ({player_data['Season'].min()}-{player_data['Season'].max()})")
    print(f"  Sequence length: {seq_length}")
    print(f"  Cutoff year: {cutoff_year}")
    print(f"  Pipeline path: {'REGRESSION (_predict_with_regression)' if use_regression else 'LEGACY (_prepare_player_sequence)'}")
    print(f"{'#' * 80}")

    _print_config_banner({
        'ENABLE_RELIABILITY_REGRESSION_PREDICTION': use_regression,
        'ENABLE_PARK_FACTOR_ADJUSTMENT':            use_pf,
        'USE_XWOBA_FOR_PREDICTIONS':               use_xwoba,
        'USE_XWOBA_BLEND_FOR_PREDICTIONS':         use_xwoba_blend,
        'USE_XBA_FOR_PREDICTIONS':                 use_xba,
        'USE_XSLG_FOR_PREDICTIONS':                use_xslg,
    })

    num_seasons = len(player_data)

    # ==================== STEP 1: Raw data ====================
    raw_n = min(num_seasons, seq_length)
    raw_seq = player_data[input_features].iloc[-raw_n:].values
    raw_seasons = player_data['Season'].iloc[-raw_n:].tolist()
    raw_teams = (player_data['Team'].iloc[-raw_n:].tolist()
                 if 'Team' in player_data.columns else ['?'] * raw_n)
    raw_pa = (player_data['PA'].iloc[-raw_n:].tolist()
              if 'PA' in player_data.columns else ['?'] * raw_n)

    print_table(
        "STEP 1: Raw Historical Data (before any transformations)",
        raw_seq, input_features,
        seasons=raw_seasons,
        extra_cols={'Team': raw_teams,
                    'PA': [f"{p:.0f}" if isinstance(p, (int, float)) else p for p in raw_pa]}
    )

    # ==========================================================================
    # BRANCH: non-regression path  (_prepare_player_sequence)
    # Order: build sequence → x-stats FIRST → park neutralize everything
    # ==========================================================================
    if not use_regression:
        # Build initial recent_data (with naive padding of last row if needed)
        if num_seasons < seq_length:
            recent_data = player_data[input_features].copy()
            while len(recent_data) < seq_length:
                recent_data = pd.concat([recent_data, recent_data.iloc[-1:]], ignore_index=True)
            seq_seasons = (['pad'] * (seq_length - num_seasons) +
                           player_data['Season'].tolist())
            if 'Team' in player_data.columns:
                seq_teams = ([player_data['Team'].iloc[-1]] * (seq_length - num_seasons) +
                             player_data['Team'].tolist())
            else:
                seq_teams = ['?'] * seq_length
        else:
            recent_data = player_data[input_features].iloc[-seq_length:].copy().reset_index(drop=True)
            seq_seasons = player_data['Season'].iloc[-seq_length:].tolist()
            seq_teams = (player_data['Team'].iloc[-seq_length:].tolist()
                         if 'Team' in player_data.columns else ['?'] * seq_length)

        # ---- STEP 2: x-stat substitution (FIRST in this path) ----
        step_num = 2
        xstat_replaced = set()
        xstat_mode = {}  # feature -> 'full' or 'blend'
        if use_xwoba and 'wOBA' in input_features and 'xwOBA' in player_data.columns:
            if num_seasons < seq_length:
                xwoba_vals = player_data['xwOBA'].copy()
                while len(xwoba_vals) < seq_length:
                    xwoba_vals = pd.concat([xwoba_vals.iloc[:1], xwoba_vals], ignore_index=True)
                recent_data['wOBA'] = xwoba_vals.values
            else:
                recent_data['wOBA'] = player_data['xwOBA'].iloc[-seq_length:].values
            xstat_replaced.add('wOBA')
            xstat_mode['wOBA'] = 'full xwOBA'
        elif use_xwoba_blend and 'wOBA' in input_features and 'xwOBA' in player_data.columns:
            raw_woba = recent_data['wOBA'].values.copy()
            if num_seasons < seq_length:
                xwoba_vals = player_data['xwOBA'].copy()
                while len(xwoba_vals) < seq_length:
                    xwoba_vals = pd.concat([xwoba_vals.iloc[:1], xwoba_vals], ignore_index=True)
                recent_data['wOBA'] = (raw_woba + xwoba_vals.values) / 2
            else:
                xwoba_vals = player_data['xwOBA'].iloc[-seq_length:].values
                recent_data['wOBA'] = (raw_woba + xwoba_vals) / 2
            xstat_replaced.add('wOBA')
            xstat_mode['wOBA'] = 'blend (wOBA+xwOBA)/2'
        if use_xba and 'AVG' in input_features and 'xBA' in player_data.columns:
            if num_seasons < seq_length:
                xba_vals = player_data['xBA'].copy()
                while len(xba_vals) < seq_length:
                    xba_vals = pd.concat([xba_vals.iloc[:1], xba_vals], ignore_index=True)
                recent_data['AVG'] = xba_vals.values
            else:
                recent_data['AVG'] = player_data['xBA'].iloc[-seq_length:].values
            xstat_replaced.add('AVG')
            xstat_mode['AVG'] = 'full xBA'
        if use_xslg and 'SLG' in input_features and 'xSLG' in player_data.columns:
            if num_seasons < seq_length:
                xslg_vals = player_data['xSLG'].copy()
                while len(xslg_vals) < seq_length:
                    xslg_vals = pd.concat([xslg_vals.iloc[:1], xslg_vals], ignore_index=True)
                recent_data['SLG'] = xslg_vals.values
            else:
                recent_data['SLG'] = player_data['xSLG'].iloc[-seq_length:].values
            xstat_replaced.add('SLG')
            xstat_mode['SLG'] = 'full xSLG'

        if xstat_replaced:
            print(f"\n  STEP {step_num}: x-stat substitution (applied BEFORE park neutralization)")
            for feat in sorted(xstat_replaced):
                print(f"    {feat}: {xstat_mode.get(feat, '')}")
        else:
            print(f"\n  STEP {step_num}: x-stat substitution — SKIPPED (all USE_X*_FOR_PREDICTIONS=False or data unavailable)")
        step_num += 1

        # ---- STEP 3: Park factor neutralization (AFTER x-stats, applied to everything) ----
        if use_pf and 'Team' in player_data.columns:
            adjustable_feats = [f for f in input_features if f not in EXCLUDED_STATS]
            print(f"\n  STEP {step_num}: Park factor neutralization (all adjustable features including x-stat positions)")
            for row_idx, team in enumerate(seq_teams):
                pf = get_park_factor(team)
                label = f"Row {row_idx} ({'pad' if seq_seasons[row_idx] == 'pad' else int(seq_seasons[row_idx])}) ({team})"
                if pf != 1.0:
                    print(f"    {label}: ÷ {pf:.4f}")
                    for feat in adjustable_feats:
                        if feat in recent_data.columns:
                            recent_data.iloc[row_idx, recent_data.columns.get_loc(feat)] /= pf
                else:
                    print(f"    {label}: no adjustment (PF=1.00)")

            print_table(
                f"STEP {step_num}: After Park Factor Neutralization",
                recent_data.values.astype(np.float64), input_features,
                seasons=seq_seasons
            )
            step_num += 1
        else:
            if not use_pf:
                print(f"\n  STEP {step_num}: Park factor neutralization — SKIPPED (ENABLE_PARK_FACTOR_ADJUSTMENT=False)")
            else:
                print(f"\n  STEP {step_num}: Park factor neutralization — SKIPPED (no Team column)")
            step_num += 1

        sequence = recent_data.values.astype(np.float64)
        final_seq_seasons = seq_seasons

        print_table(
            f"STEP {step_num - 1}: Final sequence before scaling",
            sequence, input_features,
            seasons=seq_seasons
        )

    # ==========================================================================
    # BRANCH: regression path  (_predict_with_regression)
    # Order: x-stats FIRST → regression → park neutralize all adjustable features
    # ==========================================================================
    else:
        # ---- STEP 2: x-stat substitution into player_data (BEFORE regression) ----
        xstat_data = player_data.copy()
        xstat_replaced = set()
        substitutions = {}

        def _apply_xstat(src_col, dst_col, label):
            for idx, row in xstat_data.iterrows():
                if not pd.isna(row.get(src_col, np.nan)):
                    orig = row[dst_col]
                    xstat_data.at[idx, dst_col] = row[src_col]
                    if idx in xstat_data.iloc[-seq_length:].index:
                        substitutions.setdefault(idx, []).append(
                            f"{label} {orig:.3f}\u2192{row[src_col]:.3f}"
                        )

        if use_xwoba and 'wOBA' in input_features and 'xwOBA' in player_data.columns:
            _apply_xstat('xwOBA', 'wOBA', 'wOBA')
            xstat_replaced.add('wOBA')
        elif use_xwoba_blend and 'wOBA' in input_features and 'xwOBA' in player_data.columns:
            for idx, row in xstat_data.iterrows():
                if not pd.isna(row.get('xwOBA', np.nan)):
                    orig = row['wOBA']
                    blended = (orig + row['xwOBA']) / 2
                    xstat_data.at[idx, 'wOBA'] = blended
                    if idx in xstat_data.iloc[-seq_length:].index:
                        substitutions.setdefault(idx, []).append(
                            f"wOBA {orig:.3f}\u2192{blended:.3f} (blend w/ xwOBA={row['xwOBA']:.3f})"
                        )
            xstat_replaced.add('wOBA')
        if use_xba and 'AVG' in input_features and 'xBA' in player_data.columns:
            _apply_xstat('xBA', 'AVG', 'AVG')
            xstat_replaced.add('AVG')
        if use_xslg and 'SLG' in input_features and 'xSLG' in player_data.columns:
            _apply_xstat('xSLG', 'SLG', 'SLG')
            xstat_replaced.add('SLG')

        if xstat_replaced:
            print(f"\n  STEP 2: x-stat substitution INTO player_data BEFORE regression")
            print(f"    Features replaced: {sorted(xstat_replaced)}")
            print(f"    (regression will operate on the more-predictive expected metrics)")
            for idx, subs in substitutions.items():
                season = xstat_data.loc[idx, 'Season']
                print(f"    Season {int(season)}: {', '.join(subs)}")
        else:
            print(f"\n  STEP 2: x-stat substitution — SKIPPED")

        # ---- STEP 3: Reliability regression ----
        era = get_era_for_features(input_features)
        player_data_regressed = regress_player_sequence(
            xstat_data, input_features,
            model_type='batter', era=era,
            league_priors=league_priors
        )
        regressed_seasons = player_data_regressed['Season'].iloc[-seq_length:].tolist()

        print(f"\n  STEP 3: Reliability regression (era={era}, model=batter):")
        print(f"  (Values shown: xStat-substituted → after regression)")
        for _, row in player_data_regressed.iloc[-seq_length:].iterrows():
            season = row['Season']
            volume = row.get('PA', 0)
            orig_row = xstat_data[xstat_data['Season'] == season]
            if orig_row.empty:
                continue
            orig_row = orig_row.iloc[0]
            changed = []
            for feat in input_features:
                if feat in SKIP_FEATURES:
                    continue
                n0 = _get_stabilization_point(feat, era, 'batter')
                if n0 is None:
                    continue
                ov, rv = orig_row[feat], row[feat]
                if pd.notna(ov) and pd.notna(rv) and abs(rv - ov) > 1e-6:
                    w = volume / (volume + n0) if (volume + n0) > 0 else 0
                    changed.append(f"{feat}: {ov:.3f}\u2192{rv:.3f} (w={w:.2f}, n0={n0})")
            if changed:
                print(f"    Season {int(season)} (PA={volume:.0f}):")
                for c in changed:
                    print(f"      {c}")

        # Career mean for padding
        career_mean = compute_regressed_career_mean(
            xstat_data, input_features,
            model_type='batter', era=era,
            league_priors=league_priors
        )

        recent_data = player_data_regressed[input_features].iloc[-seq_length:].copy().reset_index(drop=True)
        n_regressed = len(recent_data)
        n_pad = 0
        if n_regressed < seq_length:
            padding_vector = np.array([career_mean.get(f, 0.0) for f in input_features], dtype=np.float64)
            n_pad = seq_length - n_regressed
            padding_df = pd.DataFrame([padding_vector] * n_pad, columns=input_features)
            recent_data = pd.concat([padding_df, recent_data], ignore_index=True)
            print(f"\n  Padding: {n_pad} rows filled with regressed career mean")
            for feat in input_features:
                print(f"    {feat}: {career_mean.get(feat, 0.0):.4f}")

        seq_seasons = (['pad'] * n_pad + [int(s) for s in regressed_seasons])
        if 'Team' in player_data_regressed.columns:
            num_actual = min(len(player_data_regressed), seq_length)
            seq_teams = (player_data_regressed['Team'].iloc[-num_actual:].tolist())
            last_team = player_data['Team'].iloc[-1] if 'Team' in player_data.columns else '?'
            seq_teams = [last_team] * n_pad + seq_teams
        else:
            seq_teams = ['?'] * seq_length

        print_table(
            "STEP 3: After Regression (x-stats substituted before regression)",
            recent_data.values.astype(np.float64), input_features,
            seasons=seq_seasons
        )

        sequence = recent_data.values.astype(np.float64)
        final_seq_seasons = seq_seasons

        # ---- STEP 4: Park factor neutralization (all adjustable features including x-stat positions) ----
        if use_pf and 'Team' in player_data.columns:
            adjustable_indices = [
                (i, f) for i, f in enumerate(input_features) if f not in EXCLUDED_STATS
            ]
            print(f"\n  STEP 4: Park factor neutralization (applied to all adjustable features)")

            for row_idx, team in enumerate(seq_teams):
                pf = get_park_factor(team)
                label = f"Row {row_idx} ({seq_seasons[row_idx]}) ({team})"
                if pf != 1.0:
                    print(f"    {label}: ÷ {pf:.4f}")
                    for col_idx, _ in adjustable_indices:
                        sequence[row_idx, col_idx] /= pf
                else:
                    print(f"    {label}: no adjustment (PF=1.00)")

            print_table(
                "STEP 4: After Park Factor Neutralization (final pre-scale)",
                sequence, input_features,
                seasons=seq_seasons
            )
        else:
            if not use_pf:
                print(f"\n  STEP 4: Park factor neutralization — SKIPPED (ENABLE_PARK_FACTOR_ADJUSTMENT=False)")

    # ==================== Scaled sequence ====================
    if scaler is not None:
        if not recent_data.isna().any().any():
            sequence_scaled = scaler.transform(sequence)
            print_table(
                "FINAL: Scaled Sequence (what the model actually sees)",
                sequence_scaled, input_features,
                seasons=final_seq_seasons
            )
        else:
            print("  WARNING: NaN values present — cannot scale sequence")
    else:
        print("  (No scaler loaded — skipping scaled table)")

    # ==================== Comparison: raw vs final ====================
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: Raw vs Final (pre-scale) — most recent season")
    print(f"{'=' * 80}")
    last_raw_season = raw_seasons[-1] if raw_seasons else '?'
    print(f"  Season: {last_raw_season}")
    print(f"  {'Feature':>12} {'Raw':>10} {'Final':>10} {'Diff':>10} {'%Change':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for j, feat in enumerate(input_features):
        raw_val = raw_seq[-1, j] if len(raw_seq) > 0 else np.nan
        final_val = sequence[-1, j]
        diff = final_val - raw_val if not np.isnan(raw_val) else np.nan
        pct = (diff / raw_val * 100) if raw_val != 0 and not np.isnan(raw_val) else np.nan
        changed_marker = ' *' if not np.isnan(diff) and abs(diff) > 1e-6 else ''
        print(f"  {feat:>12} {format_value(raw_val, feat):>10} {format_value(final_val, feat):>10} "
              f"{f'{diff:+.4f}':>10} {f'{pct:+.2f}%' if not np.isnan(pct) else '':>10}{changed_marker}")
    print("  (* = value changed from raw)")


def inspect_pitcher(player_id: int, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                    input_features: List[str], scaler, seq_length: int,
                    cutoff_year: int, role: str, league_priors: Optional[Dict] = None,
                    pitcher_config=None):
    """
    Inspect the full prediction pipeline for a single pitcher.

    Mirrors predict_future_stats_pitcher in core/prediction.py exactly,
    respecting ENABLE_PARK_FACTOR_ADJUSTMENT from the pitcher config.
    """
    from core.reliability import (
        regress_player_sequence,
        compute_regressed_career_mean,
        get_era_for_features,
    )

    # ---- Read config toggles ----
    use_pf = getattr(pitcher_config, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)

    player_data = raw_df[
        (raw_df['IDfg'] == player_id) &
        (raw_df['Season'] <= cutoff_year)
    ].copy().sort_values('Season')

    if len(player_data) == 0:
        print(f"No data found for player ID {player_id}")
        return

    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        player_name = f"ID:{player_id}"

    # Determine role from GS rate
    if 'GS' in player_data.columns and 'G' in player_data.columns:
        latest = player_data.iloc[-1]
        gs_rate = latest['GS'] / latest['G'] if latest['G'] > 0 else 0
        detected_role = 'SP' if gs_rate >= 0.8 else 'RP'
    else:
        detected_role = role

    print(f"\n{'#' * 80}")
    print(f"  PITCHER: {player_name} (IDfg={player_id}, Role={detected_role})")
    print(f"  Seasons available: {len(player_data)} ({player_data['Season'].min()}-{player_data['Season'].max()})")
    print(f"  Sequence length: {seq_length}")
    print(f"  Cutoff year: {cutoff_year}")
    print(f"{'#' * 80}")

    _print_config_banner({
        'ENABLE_RELIABILITY_REGRESSION_PREDICTION': True,  # always on for pitchers
        'ENABLE_PARK_FACTOR_ADJUSTMENT': use_pf,
    })

    # Thresholds
    sequence_ip_threshold = 20 if detected_role == 'SP' else 10

    # ==================== STEP 1: Raw data ====================
    raw_display = player_data.tail(seq_length + 2)
    raw_features = raw_display[input_features].values
    raw_seasons = raw_display['Season'].tolist()
    raw_teams = (raw_display['Team'].tolist()
                 if 'Team' in raw_display.columns else ['?'] * len(raw_seasons))
    raw_ip = (raw_display['IP'].tolist()
              if 'IP' in raw_display.columns else ['?'] * len(raw_seasons))

    print_table(
        "STEP 1: Raw Historical Data",
        raw_features, input_features,
        seasons=raw_seasons,
        extra_cols={'Team': raw_teams,
                    'IP': [f"{ip:.0f}" if isinstance(ip, (int, float)) else ip for ip in raw_ip]}
    )

    # ==================== STEP 2: Reliability regression (always on) ====================
    era = get_era_for_features(input_features)
    player_data_regressed = regress_player_sequence(
        player_data, input_features,
        model_type='pitcher', era=era,
        league_priors=league_priors
    )

    career_mean = compute_regressed_career_mean(
        player_data, input_features,
        model_type='pitcher', era=era,
        league_priors=league_priors
    )

    # Build sequence (same logic as predict_future_stats_pitcher)
    recent_seasons = player_data_regressed.tail(seq_length + 2)
    sequence_data = []
    sequence_seasons = []
    sequence_teams = []

    for idx, season in recent_seasons.iterrows():
        if season['IP'] >= sequence_ip_threshold:
            base_features = season[input_features].values
            sequence_data.append(base_features)
            sequence_seasons.append(season['Season'])
            sequence_teams.append(season.get('Team', '?'))

    if len(sequence_data) > seq_length:
        sequence_data = sequence_data[-seq_length:]
        sequence_seasons = sequence_seasons[-seq_length:]
        sequence_teams = sequence_teams[-seq_length:]

    n_pad = 0
    if len(sequence_data) < seq_length:
        padding_vector = np.array([career_mean.get(f, 0.0) for f in input_features], dtype=np.float32)
        n_pad = seq_length - len(sequence_data)
        sequence_data = [padding_vector] * n_pad + sequence_data
        print(f"\n  Padding: {n_pad} rows filled with regressed career mean")
        for feat in input_features:
            print(f"    {feat}: {career_mean.get(feat, 0.0):.4f}")

    current_sequence = np.array(sequence_data[-seq_length:], dtype=np.float32)

    # Show regression details
    print(f"\n  STEP 2: Reliability regression (era={era}, model=pitcher):")
    for _, row in player_data_regressed.tail(seq_length + 2).iterrows():
        if row['IP'] < sequence_ip_threshold:
            continue
        season = row['Season']
        volume = _estimate_volume(row, 'pitcher')
        orig_row = player_data[player_data['Season'] == season]
        if orig_row.empty:
            continue
        orig_row = orig_row.iloc[0]

        changed_feats = []
        for feat in input_features:
            if feat in SKIP_FEATURES:
                continue
            n0 = _get_stabilization_point(feat, era, 'pitcher')
            if n0 is None:
                continue
            orig_val = orig_row[feat]
            reg_val = row[feat]
            if pd.notna(orig_val) and pd.notna(reg_val) and abs(reg_val - orig_val) > 1e-6:
                weight = volume / (volume + n0) if (volume + n0) > 0 else 0
                changed_feats.append(
                    f"{feat}: {orig_val:.3f}\u2192{reg_val:.3f} (weight={weight:.2f}, n0={n0})"
                )

        if changed_feats:
            print(f"    Season {int(season)} (IP={row['IP']:.0f}, est BF={volume:.0f}):")
            for cf in changed_feats:
                print(f"      {cf}")

    display_seasons = ['pad'] * n_pad + [int(s) for s in sequence_seasons[-seq_length:]]
    display_teams   = ['pad'] * n_pad + sequence_teams[-seq_length:]

    print_table(
        "STEP 2: After Regression (pre-park-adjustment)",
        current_sequence, input_features,
        seasons=display_seasons,
        extra_cols={'Team': display_teams}
    )

    # ==================== STEP 3: Park factor neutralization ====================
    if use_pf and 'Team' in player_data.columns:
        adjustable_indices = [i for i, f in enumerate(input_features) if f not in EXCLUDED_STATS]
        last_team = player_data['Team'].iloc[-1]
        all_teams_for_pf = [last_team] * n_pad + sequence_teams[-seq_length:]

        print(f"\n  STEP 3: Park factor neutralization")
        for row_idx, team in enumerate(all_teams_for_pf):
            pf = get_park_factor(team)
            label = f"Row {row_idx} ({display_seasons[row_idx]}) ({team})"
            if pf != 1.0:
                print(f"    {label}: ÷ {pf:.4f}")
                for col_idx in adjustable_indices:
                    current_sequence[row_idx, col_idx] = current_sequence[row_idx, col_idx] / pf
            else:
                print(f"    {label}: no adjustment (PF=1.00)")

        print_table(
            "STEP 3: After Park Factor Neutralization (final pre-scale)",
            current_sequence, input_features,
            seasons=display_seasons,
            extra_cols={'Team': display_teams}
        )
    else:
        if not use_pf:
            print(f"\n  STEP 3: Park factor neutralization — SKIPPED (ENABLE_PARK_FACTOR_ADJUSTMENT=False)")
        else:
            print(f"\n  STEP 3: Park factor neutralization — SKIPPED (no Team column)")

    # ==================== STEP 4 (or 3): Scaled ====================
    if scaler is not None:
        sequence_scaled = scaler.transform(current_sequence)
        print_table(
            "FINAL: Scaled Sequence (what the model actually sees)",
            sequence_scaled, input_features,
            seasons=display_seasons
        )
    else:
        print("  (No scaler loaded — skipping scaled table)")

    # ==================== Summary ===================================
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: Raw vs Final per feature — most recent qualified season")
    print(f"{'=' * 80}")
    # Find the last row in raw_display that passed the IP threshold
    raw_for_compare = player_data.tail(seq_length)
    if len(raw_for_compare) == 0:
        return
    last_raw = raw_for_compare.iloc[-1]
    last_final = current_sequence[-1]
    print(f"  Season: {int(last_raw['Season'])} | Team: {last_raw.get('Team','?')} | IP: {last_raw.get('IP','?')}")
    print(f"  {'Feature':>12} {'Raw':>10} {'Final':>10} {'Diff':>10} {'%Change':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for j, feat in enumerate(input_features):
        raw_val = last_raw[feat] if feat in last_raw.index else np.nan
        final_val = last_final[j]
        diff = final_val - raw_val if not np.isnan(raw_val) else np.nan
        pct = (diff / raw_val * 100) if raw_val != 0 and not np.isnan(raw_val) else np.nan
        changed_marker = ' *' if not np.isnan(diff) and abs(diff) > 1e-6 else ''
        print(f"  {feat:>12} {format_value(raw_val, feat):>10} {format_value(final_val, feat):>10} "
              f"{f'{diff:+.4f}':>10} {f'{pct:+.2f}%' if not np.isnan(pct) else '':>10}{changed_marker}")
    print("  (* = value changed from raw)")


def main():
    parser = argparse.ArgumentParser(description='Inspect prediction input sequences')
    parser.add_argument('--players', nargs='+', help='Player names to inspect')
    parser.add_argument('--player-ids', nargs='+', type=int, help='Player FanGraphs IDs to inspect')
    parser.add_argument('--type', choices=['batter', 'pitcher'], default='batter', help='Model type')
    parser.add_argument('--role', choices=['SP', 'RP'], default=None, help='Pitcher role (auto-detected if not specified)')
    parser.add_argument('--top', type=int, default=None, help='Inspect top N players by PA/IP in cutoff year')
    parser.add_argument('--cutoff-year', type=int, default=2025, help='Last year of actual data')
    args = parser.parse_args()

    print("Loading data and models...")

    if args.type == 'batter':
        from configs.batter_config import BatterConfig

        # Determine which mode to use (finetune if checkpoint exists)
        finetune_ckpt = Path(BatterConfig.CHECKPOINT_DIR) / BatterConfig.FINETUNE_CHECKPOINT_FILE
        pretrain_ckpt = Path(BatterConfig.CHECKPOINT_DIR) / BatterConfig.PRETRAIN_CHECKPOINT_FILE

        if finetune_ckpt.exists():
            checkpoint_path = str(finetune_ckpt)
            data_config = BatterConfig.get_data_config('finetune')
            scaler_path = BatterConfig.FINETUNE_SCALER_FILE
            input_features = BatterConfig.FINETUNE_FEATURES
            print(f"  Using finetuned model: {checkpoint_path}")
        elif pretrain_ckpt.exists():
            checkpoint_path = str(pretrain_ckpt)
            data_config = BatterConfig.get_data_config('pretrain')
            scaler_path = BatterConfig.PRETRAIN_SCALER_FILE
            input_features = BatterConfig.CLASSICAL_FEATURES
            print(f"  Using pretrained model: {checkpoint_path}")
        else:
            print(f"ERROR: No batter checkpoint found at {finetune_ckpt} or {pretrain_ckpt}")
            sys.exit(1)

        seq_length = data_config.seq_length

        # Load data
        data_file = BatterConfig.FINETUNE_DATA_FILE if finetune_ckpt.exists() else BatterConfig.DATA_FILE
        print(f"  Loading data from: {data_file}")
        raw_df = pd.read_csv(data_file, low_memory=False)
        raw_df = calculate_rate_stats(raw_df)
        player_names = generate_batter_names(raw_df)

        # Load scaler  
        scaler = None
        for sp in [scaler_path, f"data/{os.path.basename(scaler_path)}"]:
            if os.path.exists(sp):
                scaler = joblib.load(sp)
                print(f"  Loaded scaler from: {sp}")
                break
        if scaler is None:
            print(f"  WARNING: Could not load scaler from {scaler_path}")

        # Compute league priors
        league_priors = None
        if BatterConfig.ENABLE_RELIABILITY_REGRESSION_PREDICTION:
            league_priors = compute_league_priors_from_df(
                raw_df, input_features, model_type='batter',
                season=args.cutoff_year, window=3
            )
            print(f"  Computed league priors for regression ({len(league_priors)} features)")

        # Resolve player list
        player_ids = []
        if args.player_ids:
            player_ids = args.player_ids
        elif args.players:
            for name in args.players:
                matches = player_names[player_names['Name'].str.contains(name, case=False, na=False)]
                if len(matches) == 0:
                    print(f"  WARNING: No player found matching '{name}'")
                else:
                    for _, m in matches.iterrows():
                        player_ids.append(m['IDfg'])
                        print(f"  Found: {m['Name']} (IDfg={m['IDfg']})")
        elif args.top:
            qualified = raw_df[
                (raw_df['Season'] == args.cutoff_year) &
                (raw_df['PA'] >= BatterConfig.MIN_PA_CURRENT)
            ].nlargest(args.top, 'PA')
            player_ids = qualified['IDfg'].tolist()
            print(f"  Top {args.top} batters by PA in {args.cutoff_year}")

        if not player_ids:
            print("ERROR: No players specified. Use --players, --player-ids, or --top")
            sys.exit(1)

        print(f"\n  Features: {input_features}")
        print(f"  Seq length: {seq_length}")

        for pid in player_ids:
            inspect_batter(
                pid, raw_df, player_names,
                input_features, scaler, seq_length,
                args.cutoff_year, league_priors,
                batter_config=BatterConfig,
            )

    elif args.type == 'pitcher':
        from configs.pitcher_sp_config import PitcherSPConfig
        from configs.pitcher_rp_config import PitcherRPConfig

        # Load data
        data_file = PitcherSPConfig.DATA_FILE
        print(f"  Loading data from: {data_file}")
        raw_df = pd.read_csv(data_file, low_memory=False)
        raw_df = calculate_rate_stats(raw_df)
        player_names = generate_batter_names(raw_df)

        # Resolve player list first to determine roles
        player_ids = []
        if args.player_ids:
            player_ids = args.player_ids
        elif args.players:
            for name in args.players:
                matches = player_names[player_names['Name'].str.contains(name, case=False, na=False)]
                if len(matches) == 0:
                    print(f"  WARNING: No player found matching '{name}'")
                else:
                    for _, m in matches.iterrows():
                        player_ids.append(m['IDfg'])
                        print(f"  Found: {m['Name']} (IDfg={m['IDfg']})")
        elif args.top:
            qualified = raw_df[
                (raw_df['Season'] == args.cutoff_year) &
                (raw_df['IP'] >= 15)
            ].nlargest(args.top, 'IP')
            player_ids = qualified['IDfg'].tolist()

        if not player_ids:
            print("ERROR: No players specified.")
            sys.exit(1)

        for pid in player_ids:
            # Detect role
            p_data = raw_df[raw_df['IDfg'] == pid].sort_values('Season')
            if len(p_data) == 0:
                continue
            latest = p_data.iloc[-1]
            if args.role:
                role = args.role
            elif 'GS' in latest.index and 'G' in latest.index and latest['G'] > 0:
                role = 'SP' if (latest['GS'] / latest['G']) >= 0.8 else 'RP'
            else:
                role = 'SP'

            if role == 'SP':
                config = PitcherSPConfig
            else:
                config = PitcherRPConfig

            seq_length = config.SEQ_LENGTH

            # Load scaler and determine canonical feature list from it.
            # The checkpoint/scaler may have been trained with a different feature set
            # than what the current config's FINETUNE_FEATURES says (e.g. the saved
            # finetune scaler uses CLASSICAL+PITCHFX=13 while the config now lists
            # CLASSICAL+STATCAST=8).  Always trust the scaler's feature count.
            finetune_ckpt = Path(config.CHECKPOINT_DIR) / config.FINETUNE_CHECKPOINT_FILE
            pretrain_ckpt = Path(config.CHECKPOINT_DIR) / config.PRETRAIN_CHECKPOINT_FILE

            def _try_load_scaler(path: str):
                for sp in [path, f"data/{os.path.basename(path)}"]:
                    if os.path.exists(sp):
                        return joblib.load(sp), sp
                return None, None

            scaler = None
            scaler_src = None
            if finetune_ckpt.exists():
                scaler, scaler_src = _try_load_scaler(config.FINETUNE_SCALER_FILE)
            if scaler is None and pretrain_ckpt.exists():
                scaler, scaler_src = _try_load_scaler(config.PRETRAIN_SCALER_FILE)
            if scaler is None:
                scaler, scaler_src = _try_load_scaler(config.SCALER_FILE)

            # Build a map of feature_count → feature_list so we can match the scaler
            candidate_feature_lists = []
            if hasattr(config, 'FINETUNE_FEATURES'):
                candidate_feature_lists.append(('finetune', config.FINETUNE_FEATURES))
            if hasattr(config, 'PITCHFX_FEATURES'):
                candidate_feature_lists.append((
                    'classical+pitchfx',
                    config.CLASSICAL_FEATURES + config.PITCHFX_FEATURES
                ))
            candidate_feature_lists.append(('classical', config.CLASSICAL_FEATURES))
            if hasattr(config, 'INPUT_FEATURES'):
                candidate_feature_lists.append(('legacy', config.INPUT_FEATURES))

            # Pick the feature list that matches what the scaler was trained on
            input_features = None
            if scaler is not None:
                expected_n = scaler.n_features_in_
                print(f"  Loaded scaler from: {scaler_src} (expects {expected_n} features)")
                for feat_label, feat_list in candidate_feature_lists:
                    if len(feat_list) == expected_n:
                        input_features = feat_list
                        print(f"  Matched to '{feat_label}' features ({len(feat_list)} features)")
                        break
                if input_features is None:
                    print(f"  WARNING: No config feature list matches scaler's {expected_n} features.")
                    print(f"  Falling back to FINETUNE_FEATURES ({len(config.FINETUNE_FEATURES)} features) — scaler will be skipped.")
                    input_features = config.FINETUNE_FEATURES
                    scaler = None
            else:
                print("  WARNING: No scaler found — sequence will not be scaled.")
                input_features = config.FINETUNE_FEATURES if finetune_ckpt.exists() else config.CLASSICAL_FEATURES

            league_priors = compute_league_priors_from_df(
                raw_df, input_features, model_type='pitcher',
                season=args.cutoff_year, window=3
            )

            inspect_pitcher(
                pid, raw_df, player_names,
                input_features, scaler, seq_length,
                args.cutoff_year, role, league_priors,
                pitcher_config=config,
            )


if __name__ == '__main__':
    main()
