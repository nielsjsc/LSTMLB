"""
Export Regressed Sequences
==========================

Outputs the exact input sequences that the LSTM model sees for each pitcher,
after reliability regression (career/league prior blending + Bayesian shrinkage).

For each pitcher, the CSV contains:
  - Raw (observed) values for each historical season
  - Regressed values for each historical season
  - Career mean padding row (if the sequence is shorter than SEQ_LENGTH)
  - League priors used in the blending
  - Metadata: career TBF, career_weight, effective stabilization points

Usage:
    cd auto_train_models
    python scripts/export_regressed_sequences.py
    python scripts/export_regressed_sequences.py --player "Bryan Woo"
    python scripts/export_regressed_sequences.py --player-id 30279
    python scripts/export_regressed_sequences.py --cutoff-year 2024
"""

import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure auto_train_models is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.pitcher_sp_config import PitcherSPConfig
from core.data_processing import calculate_rate_stats
from core.reliability import (
    regress_player_sequence,
    compute_regressed_career_mean,
    compute_league_priors_from_df,
    get_era_for_features,
    PITCHER_STABILIZATION_POINTS,
    _compute_career_weight,
    _effective_stabilization_point,
    _get_career_stabilization,
    _estimate_volume,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_data_path(config_data_file: str) -> Path:
    config_path = Path(config_data_file)
    if config_path.exists():
        return config_path
    return Path(__file__).parent.parent / config_data_file


def build_rows_for_pitcher(
    player_id,
    player_name: str,
    raw_df: pd.DataFrame,
    input_features: list,
    league_priors: dict,
    config,
    cutoff_year: int,
    seq_length: int,
) -> list[dict]:
    """
    Build raw + regressed rows for one pitcher, mirroring the exact logic
    in predict_future_stats_pitcher.
    """
    era = get_era_for_features(input_features)
    recency_halflife = getattr(config, 'PRIOR_RECENCY_HALFLIFE', 0)
    league_weight_overrides = getattr(config, 'PRIOR_LEAGUE_WEIGHT_OVERRIDES', None)

    player_data = raw_df[
        (raw_df['IDfg'] == player_id) & (raw_df['Season'] <= cutoff_year)
    ].sort_values('Season').copy()

    if player_data.empty:
        return []

    # Determine role from GS rate in most recent season
    last_row = player_data.iloc[-1]
    gs_rate = last_row['GS'] / last_row['G'] if last_row['G'] > 0 else 0.0
    role = 'SP' if gs_rate >= 0.8 else 'RP'
    sequence_ip_threshold = 20 if role == 'SP' else 10

    # --- Regress ---
    player_data_regressed = regress_player_sequence(
        player_data, input_features, model_type='pitcher', era=era,
        league_priors=league_priors, recency_halflife=recency_halflife,
        league_weight_overrides=league_weight_overrides,
        seq_length=seq_length, sequence_ip_threshold=sequence_ip_threshold,
    )

    career_mean = compute_regressed_career_mean(
        player_data, input_features, model_type='pitcher', era=era,
        league_priors=league_priors, recency_halflife=recency_halflife,
        league_weight_overrides=league_weight_overrides,
        seq_length=seq_length, sequence_ip_threshold=sequence_ip_threshold,
    )

    # --- Metadata: use sequence volume for career_weight (matches regression) ---
    candidates = player_data.tail(seq_length + 2)
    seq_rows_for_meta = candidates[candidates['IP'] >= sequence_ip_threshold].tail(seq_length)
    sequence_tbf = float(seq_rows_for_meta['TBF'].sum()) if 'TBF' in seq_rows_for_meta.columns else 0.0
    total_career_tbf = float(player_data['TBF'].sum()) if 'TBF' in player_data.columns else 0.0
    career_stab = _get_career_stabilization('pitcher')
    career_weight = _compute_career_weight(sequence_tbf, 'pitcher')

    # --- Build sequence (mirrors predict_future_stats_pitcher) ---
    recent_seasons_raw = player_data.tail(seq_length + 2)
    recent_seasons_reg = player_data_regressed.loc[recent_seasons_raw.index]

    sequence_indices = []
    for idx, season in recent_seasons_reg.iterrows():
        if season['IP'] >= sequence_ip_threshold:
            sequence_indices.append(idx)
    if len(sequence_indices) > seq_length:
        sequence_indices = sequence_indices[-seq_length:]

    rows = []

    # --- League priors row ---
    lp_row = {
        'Name': player_name,
        'IDfg': player_id,
        'Role': role,
        'row_type': 'league_prior',
        'Season': '',
        'IP': '',
        'TBF': '',
        'sequence_TBF': sequence_tbf,
        'career_TBF': total_career_tbf,
        'career_weight': career_weight,
        'seq_position': '',
    }
    for feat in input_features:
        lp_row[f'{feat}_raw'] = ''
        lp_row[f'{feat}_regressed'] = ''
        lp_row[f'{feat}_league'] = league_priors.get(feat, '')
    rows.append(lp_row)

    # --- Per-season rows (raw + regressed side by side) ---
    for seq_pos, idx in enumerate(sequence_indices, start=1):
        raw_season = player_data.loc[idx]
        reg_season = player_data_regressed.loc[idx]
        season_tbf = float(raw_season.get('TBF', 0))

        r = {
            'Name': player_name,
            'IDfg': player_id,
            'Role': role,
            'row_type': 'season',
            'Season': int(raw_season['Season']),
            'IP': float(raw_season['IP']),
            'TBF': season_tbf,
            'sequence_TBF': sequence_tbf,
            'career_TBF': total_career_tbf,
            'career_weight': career_weight,
            'seq_position': seq_pos,
        }
        for feat in input_features:
            raw_val = raw_season.get(feat, np.nan)
            reg_val = reg_season.get(feat, np.nan)
            r[f'{feat}_raw'] = raw_val
            r[f'{feat}_regressed'] = reg_val
            r[f'{feat}_league'] = league_priors.get(feat, '')

            # Effective n0 for this stat
            base_n0 = PITCHER_STABILIZATION_POINTS.get(feat)
            if base_n0 is not None:
                eff_n0 = _effective_stabilization_point(
                    base_n0, sequence_tbf, career_stab
                )
                r[f'{feat}_eff_n0'] = round(eff_n0, 1)
        rows.append(r)

    # --- Padding rows (career mean) if sequence is short ---
    n_pad = seq_length - len(sequence_indices)
    for p in range(n_pad):
        pad_row = {
            'Name': player_name,
            'IDfg': player_id,
            'Role': role,
            'row_type': 'padding',
            'Season': '',
            'IP': '',
            'TBF': '',
            'sequence_TBF': sequence_tbf,
            'career_TBF': total_career_tbf,
            'career_weight': career_weight,
            'seq_position': len(sequence_indices) + p + 1,
        }
        for feat in input_features:
            pad_row[f'{feat}_raw'] = ''
            pad_row[f'{feat}_regressed'] = career_mean.get(feat, '')
            pad_row[f'{feat}_league'] = league_priors.get(feat, '')
        rows.append(pad_row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Export regressed pitcher sequences')
    parser.add_argument('--player', type=str, default=None,
                        help='Filter to a specific player by name (partial match)')
    parser.add_argument('--player-id', type=int, default=None,
                        help='Filter to a specific player by IDfg')
    parser.add_argument('--cutoff-year', type=int, default=None,
                        help='Last year of actual data (default: latest in data)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: data/generated/regressed_sequences.csv)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    config = PitcherSPConfig
    input_features = config.INPUT_FEATURES
    seq_length = config.SEQ_LENGTH

    # Load data
    data_path = resolve_data_path(config.DATA_FILE)
    logger.info(f"Loading data from {data_path}")
    raw_df = pd.read_csv(data_path)
    raw_df = calculate_rate_stats(raw_df)

    cutoff_year = args.cutoff_year or int(raw_df['Season'].max())
    logger.info(f"Cutoff year: {cutoff_year}")
    logger.info(f"Input features: {input_features}")
    logger.info(f"Sequence length: {seq_length}")

    # Player names
    names_path = Path(__file__).parent.parent.parent / 'data' / 'pitcher_names.csv'
    if names_path.exists():
        player_names_df = pd.read_csv(names_path)
    else:
        player_names_df = raw_df[['Name', 'IDfg']].drop_duplicates().sort_values('Name')

    # Compute league priors
    era = get_era_for_features(input_features)
    league_priors = compute_league_priors_from_df(
        raw_df, input_features, model_type='pitcher', season=cutoff_year, window=3
    )
    logger.info(f"League priors ({cutoff_year}, 3-yr window):")
    for feat, val in league_priors.items():
        logger.info(f"  {feat}: {val:.4f}")

    # Determine which pitchers to process
    unified = getattr(config, 'UNIFIED_PITCHER_MODEL', False)
    pitchers_current = raw_df[raw_df['Season'] == cutoff_year]

    if args.player_id:
        pitcher_ids = [args.player_id]
    elif args.player:
        match_mask = player_names_df['Name'].str.contains(args.player, case=False, na=False)
        matched = player_names_df[match_mask]
        if matched.empty:
            logger.error(f"No players matched '{args.player}'")
            return
        pitcher_ids = matched['IDfg'].tolist()
        logger.info(f"Matched {len(pitcher_ids)} player(s): {matched['Name'].tolist()}")
    else:
        # All qualified pitchers from the cutoff year
        qualified = pitchers_current[
            (pitchers_current['IP'] >= 15) & (pitchers_current['G'] >= 5)
        ]
        pitcher_ids = sorted(qualified['IDfg'].unique().tolist())
        logger.info(f"Processing {len(pitcher_ids)} qualified pitchers from {cutoff_year}")

    # Build rows
    all_rows = []
    for pid in pitcher_ids:
        name_match = player_names_df[player_names_df['IDfg'] == pid]
        name = name_match['Name'].iloc[0] if not name_match.empty else str(pid)

        rows = build_rows_for_pitcher(
            player_id=pid,
            player_name=name,
            raw_df=raw_df,
            input_features=input_features,
            league_priors=league_priors,
            config=config,
            cutoff_year=cutoff_year,
            seq_length=seq_length,
        )
        all_rows.extend(rows)

    if not all_rows:
        logger.warning("No rows generated")
        return

    result_df = pd.DataFrame(all_rows)

    # Output
    out_path = args.output or str(
        Path(__file__).parent.parent.parent / 'data' / 'generated' / 'regressed_sequences.csv'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result_df.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(result_df)} rows ({len(pitcher_ids)} pitchers) to {out_path}")

    # Print summary for single-player mode
    if len(pitcher_ids) <= 3:
        for pid in pitcher_ids:
            player_rows = result_df[result_df['IDfg'] == pid]
            name = player_rows['Name'].iloc[0]
            print(f"\n{'='*80}")
            print(f"  {name} (IDfg={pid})")
            print(f"{'='*80}")

            lp = player_rows[player_rows['row_type'] == 'league_prior'].iloc[0]
            print(f"  Role: {lp['Role']}  |  Seq TBF: {lp['sequence_TBF']:.0f}  |  Career TBF: {lp['career_TBF']:.0f}  |  Career Weight: {lp['career_weight']:.3f}")

            season_rows = player_rows[player_rows['row_type'] == 'season']
            pad_rows = player_rows[player_rows['row_type'] == 'padding']

            print(f"\n  Sequence ({len(season_rows)} season(s) + {len(pad_rows)} padding):")
            print(f"  {'':>4} {'Season':>6} {'IP':>6} {'TBF':>6}  ", end='')
            for feat in input_features:
                if feat == 'Age':
                    continue
                print(f" {feat:>8} {'→reg':>8}", end='')
            print()

            for _, row in season_rows.iterrows():
                print(f"  S{int(row['seq_position']):>2}  {int(row['Season']):>6} {row['IP']:>6.0f} {row['TBF']:>6.0f}  ", end='')
                for feat in input_features:
                    if feat == 'Age':
                        continue
                    raw_v = row.get(f'{feat}_raw', '')
                    reg_v = row.get(f'{feat}_regressed', '')
                    if isinstance(raw_v, (int, float)) and not np.isnan(raw_v):
                        print(f" {raw_v:>8.4f} {reg_v:>8.4f}", end='')
                    else:
                        print(f" {'':>8} {'':>8}", end='')
                print()

            for _, row in pad_rows.iterrows():
                print(f"  P{int(row['seq_position']):>2}  {'pad':>6} {'':>6} {'':>6}  ", end='')
                for feat in input_features:
                    if feat == 'Age':
                        continue
                    reg_v = row.get(f'{feat}_regressed', '')
                    if isinstance(reg_v, (int, float)) and not np.isnan(reg_v):
                        print(f" {'':>8} {reg_v:>8.4f}", end='')
                    else:
                        print(f" {'':>8} {'':>8}", end='')
                print()

            # Show league priors for context
            print(f"\n  League priors:")
            print(f"  {'':>4} {'':>6} {'':>6} {'':>6}  ", end='')
            for feat in input_features:
                if feat == 'Age':
                    continue
                lp_val = league_priors.get(feat, '')
                if isinstance(lp_val, (int, float)):
                    print(f" {'':>8} {lp_val:>8.4f}", end='')
                else:
                    print(f" {'':>8} {'':>8}", end='')
            print()


if __name__ == '__main__':
    main()
