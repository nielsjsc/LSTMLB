"""
Pitcher-Specific Prediction Functions
======================================

Extracted from core/prediction.py to keep the main prediction module focused
on shared infrastructure.  Contains:

- FIP reconstruction from component rates (K%, BB%, HBP%) with HR% derived from HR/FB × FB%
- HR% derivation helper (_derive_hr_pct) for FIP and HR/9 output
- ERA-FIP and ERA-SIERA career gap computation and adjustment
- SIERA reconstruction from K%, BB%, GB%
- Pitcher aging constraints (physical bounds + improvement caps)
- Output regression (Bayesian shrinkage of model outputs)
- predict_future_stats_pitcher() — single-player pitcher projection
- predict_all_pitchers() — bulk pitcher projection orchestration

All functions are re-exported from core/prediction.py for backward
compatibility, so existing imports continue to work.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List, Set
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# FIP RECONSTRUCTION FROM COMPONENTS
# =============================================================================
# Instead of relying on the model's direct FIP prediction (which diverges from
# ERA because MinMax scaling compresses them differently), we reconstruct FIP
# mathematically from the model's predicted component rates: K%, BB%, HBP%,
# plus HR% derived on-the-fly from HR/FB × FB% × BIP_rate.
#
# Formula: FIP = (TBF/IP) × (13×HR% + 3×(BB%+HBP%) − 2×K%) + cFIP
#
# BF/IP is derived per-pitcher from predicted BABIP and component rates:
#   bip_rate = 1 - K% - BB% - HBP% - HR%
#   out_rate = K% + bip_rate × (1 - BABIP)
#   BF/IP   = 3 / out_rate
#
# This replaces the old constant BF/IP approximation (4.25 for everyone),
# which introduced ~0.3-0.5 FIP error for pitchers at the extremes.
# A high-K, low-walk pitcher might have BF/IP ≈ 3.8, while a contact-heavy
# pitcher with walks could be at 4.5+.
#
# The FIP constant (cFIP) is the league-level offset that anchors FIP to the
# league ERA. Historically ~3.10, updated annually by FanGraphs.

FIP_CONSTANT = 3.15          # cFIP — anchors FIP to league ERA scale (modern era avg)
BF_PER_IP_RATIO = 4.25       # Fallback when BABIP is not available


def derive_bf_per_ip(k_pct: float, bb_pct: float, hbp_pct: float,
                     hr_pct: float, babip: float) -> float:
    """
    Derive batters faced per inning pitched from per-TBF rates and BABIP.

    Every plate appearance either results in an out or a baserunner.
    Outs come from strikeouts and non-hit balls in play (BIP × (1 - BABIP)).

    out_rate = K% + (1 - K% - BB% - HBP% - HR%) × (1 - BABIP)
    BF/IP   = 3 / out_rate

    Args:
        k_pct:   Strikeout rate (K/TBF)
        bb_pct:  Walk rate (BB/TBF)
        hbp_pct: Hit-by-pitch rate (HBP/TBF)
        hr_pct:  Home run rate (HR/TBF)
        babip:   Batting average on balls in play

    Returns:
        Batters faced per inning pitched (typically 3.7-5.0)
    """
    bip_rate = 1.0 - k_pct - bb_pct - hbp_pct - hr_pct
    bip_rate = max(bip_rate, 0.20)  # Floor: at least 20% BIP
    out_rate = k_pct + bip_rate * (1.0 - babip)
    out_rate = np.clip(out_rate, 0.45, 0.85)  # Sane bounds: ~3.5-6.7 BF/IP
    return 3.0 / out_rate


def reconstruct_fip_from_components(k_pct: float, bb_pct: float,
                                     hbp_pct: float, hr_pct: float,
                                     bf_per_ip: float = BF_PER_IP_RATIO,
                                     fip_constant: float = FIP_CONSTANT) -> float:
    """
    Reconstruct FIP from per-TBF component rates.
    
    FIP = (TBF/IP) × (13×HR% + 3×(BB%+HBP%) − 2×K%) + cFIP
    
    Args:
        k_pct: Strikeout rate (K/TBF), e.g. 0.25 for 25%
        bb_pct: Walk rate (BB/TBF), e.g. 0.08 for 8%
        hbp_pct: Hit-by-pitch rate (HBP/TBF), e.g. 0.01 for 1%
        hr_pct: Home run rate (HR/TBF), e.g. 0.03 for 3%
        bf_per_ip: Average batters faced per inning (~4.3)
        fip_constant: League-level FIP constant (~3.10)
    
    Returns:
        Reconstructed FIP value
    """
    component_sum = 13.0 * hr_pct + 3.0 * (bb_pct + hbp_pct) - 2.0 * k_pct
    return bf_per_ip * component_sum + fip_constant


def _derive_hr_pct(prediction: np.ndarray, input_features: list) -> float:
    """
    Derive HR% (HR per TBF) from HR/FB, FB%, and component rates.

    HR% = HR/FB × FB% × BIP_rate, where BIP_rate = 1 - K% - BB% - HBP%.

    Falls back to a league-average HR% (~0.028) if HR/FB or FB% are not
    available in the feature set.

    Args:
        prediction: Array of predicted values (in real/unscaled space)
        input_features: List of feature names corresponding to prediction indices

    Returns:
        Derived HR% as a float (e.g. 0.03 for 3%)
    """
    if 'HR/FB' not in input_features or 'FB%' not in input_features:
        return 0.028  # League-average fallback

    hrfb = prediction[input_features.index('HR/FB')]
    fb_pct = prediction[input_features.index('FB%')]

    k_pct = prediction[input_features.index('K%')] if 'K%' in input_features else 0.22
    bb_pct = prediction[input_features.index('BB%')] if 'BB%' in input_features else 0.08
    hbp_pct = prediction[input_features.index('HBP%')] if 'HBP%' in input_features else 0.01

    bip_rate = max(1.0 - k_pct - bb_pct - hbp_pct, 0.3)  # Floor at 30% BIP
    hr_pct = hrfb * fb_pct * bip_rate
    return np.clip(hr_pct, 0.005, 0.06)


def _apply_fip_reconstruction(prediction: np.ndarray,
                               input_features: list,
                               player_name: str = '') -> np.ndarray:
    """
    Overwrite the model's direct FIP prediction with a value reconstructed
    from the predicted K%, BB%, HBP% components and HR% derived from
    HR/FB × FB% × BIP_rate.
    
    This is applied in the autoregressive prediction loop so that:
    1. The pred_dict contains the component-based FIP
    2. The next iteration's input sequence sees the reconstructed FIP
    
    If any required component feature is missing from input_features, the
    model's direct FIP prediction is kept unchanged (graceful degradation).
    
    Args:
        prediction: Array of predicted values (in real/unscaled space)
        input_features: List of feature names corresponding to prediction indices
        player_name: For logging only
    
    Returns:
        prediction array with FIP overwritten (mutated in-place and returned)
    """
    required = {'K%', 'BB%', 'HBP%', 'FIP'}
    if not required.issubset(set(input_features)):
        return prediction  # Not all components available — keep model FIP
    
    k_idx = input_features.index('K%')
    bb_idx = input_features.index('BB%')
    hbp_idx = input_features.index('HBP%')
    fip_idx = input_features.index('FIP')

    # Derive HR% from HR/FB and FB% (no longer a direct model feature)
    hr_pct = _derive_hr_pct(prediction, input_features)
    
    # Derive per-pitcher BF/IP from BABIP if available, else fall back to constant
    if 'BABIP' in input_features:
        babip_idx = input_features.index('BABIP')
        bf_per_ip = derive_bf_per_ip(
            k_pct=prediction[k_idx],
            bb_pct=prediction[bb_idx],
            hbp_pct=prediction[hbp_idx],
            hr_pct=hr_pct,
            babip=prediction[babip_idx],
        )
    else:
        bf_per_ip = BF_PER_IP_RATIO
    
    model_fip = prediction[fip_idx]
    recon_fip = reconstruct_fip_from_components(
        k_pct=prediction[k_idx],
        bb_pct=prediction[bb_idx],
        hbp_pct=prediction[hbp_idx],
        hr_pct=hr_pct,
        bf_per_ip=bf_per_ip,
    )
    
    # Sanity bounds: FIP should be in [0, 10] for any reasonable pitcher
    recon_fip = np.clip(recon_fip, 0.0, 10.0)
    
    if abs(model_fip - recon_fip) > 1.5:
        logger.debug(
            f"FIP reconstruction for {player_name}: model={model_fip:.3f} → "
            f"reconstructed={recon_fip:.3f} (BF/IP={bf_per_ip:.2f}, "
            f"HR%={hr_pct:.4f}, Δ={recon_fip - model_fip:+.3f})"
        )
    
    prediction[fip_idx] = recon_fip
    return prediction


# =============================================================================
# HR% DECOMPOSITION FROM HR/FB + FB% COMPONENTS
# =============================================================================
# HR% is an opaque blend of two very different signals:
#   - FB% (fly ball tendency): YoY r=0.753 — mechanical skill, very persistent
#   - HR/FB (HRs per fly ball): YoY r=0.170 — mostly luck, regresses hard
#
# Since HR% ≈ HR/FB × FB% × BIP_rate, a pitcher with low HR% could have:
#   (A) Low FB% → fewer HR opportunities (skill, will persist)
#   (B) Low HR/FB → got lucky on fly balls (noise, will regress)
#
# This decomposition overwrites the model's direct HR% prediction with:
#   HR% = predicted_HR/FB × predicted_FB% × (1 - K% - BB% - HBP%)
#
# The LSTM independently predicts HR/FB (which gets regressed in reliability
# regression with n0=1200 TBF) and FB% (very stable, n0=70 TBF), so the
# reconstructed HR% inherits the correct regression behavior.
#
# Example (Chris Bubic 2025):
#   Observed: HR/FB=5.7% (career 12.8%), GB%=47%, HR/9=0.46
#   After reliability regression (n0=1200): HR/FB→~10.9%
#   Decomposed: HR% = 10.9% × 33.3% × 66.6% → HR/9 ≈ 0.89
#   Projection systems expect: 0.85-1.08 HR/9 ✓


def _apply_hr_decomposition(prediction: np.ndarray,
                            input_features: list,
                            player_name: str = '') -> np.ndarray:
    """
    Overwrite HR% with a value reconstructed from predicted HR/FB and FB%.

    HR% = HR/FB × FB% × BIP_rate, where BIP_rate = 1 - K% - BB% - HBP%.

    This decomposes HR prevention into fly-ball tendency (stable skill) and
    HR-on-contact rate (noisy, heavily regressed), producing a more robust
    HR% estimate than the model's direct prediction.

    Applied BEFORE FIP reconstruction so that FIP inherits the decomposed HR%.

    Args:
        prediction: Array of predicted values (in real/unscaled space)
        input_features: List of feature names corresponding to prediction indices
        player_name: For logging only

    Returns:
        prediction array with HR% overwritten (mutated in-place and returned)
    """
    required = {'HR/FB', 'FB%', 'K%', 'BB%', 'HR%'}
    if not required.issubset(set(input_features)):
        return prediction  # Not all components available — skip

    hrfb_idx = input_features.index('HR/FB')
    fb_idx = input_features.index('FB%')
    k_idx = input_features.index('K%')
    bb_idx = input_features.index('BB%')
    hr_idx = input_features.index('HR%')

    # HBP% is optional (small effect)
    hbp_pct = 0.0
    if 'HBP%' in input_features:
        hbp_idx = input_features.index('HBP%')
        hbp_pct = prediction[hbp_idx]

    hrfb = prediction[hrfb_idx]
    fb_pct = prediction[fb_idx]
    k_pct = prediction[k_idx]
    bb_pct = prediction[bb_idx]

    bip_rate = max(1.0 - k_pct - bb_pct - hbp_pct, 0.3)  # Floor at 30% BIP
    decomposed_hr_pct = hrfb * fb_pct * bip_rate

    # Sanity bounds (from PITCHER_PHYSICAL_BOUNDS)
    decomposed_hr_pct = np.clip(decomposed_hr_pct, 0.005, 0.06)

    model_hr_pct = prediction[hr_idx]
    if abs(model_hr_pct - decomposed_hr_pct) > 0.01:
        logger.debug(
            f"HR% decomposition for {player_name}: model={model_hr_pct:.4f} → "
            f"decomposed={decomposed_hr_pct:.4f} (HR/FB={hrfb:.3f}, "
            f"FB%={fb_pct:.3f}, BIP_rate={bip_rate:.3f})"
        )

    prediction[hr_idx] = decomposed_hr_pct
    return prediction


# =============================================================================
# ERA-FIP DIFFERENTIAL (SAMPLE-SIZE REGRESSED)
# =============================================================================
# Some pitchers genuinely over- or under-perform their FIP over large sample
# sizes (e.g., Justin Verlander consistently beating his FIP). The ERA-FIP gap
# is mostly noise at small samples but becomes predictive with enough career IP.
#
# Statistical findings (data-derived from 1950-2024):
#   - Single-season Y-o-Y ERA-FIP correlation: r = 0.052 (essentially noise)
#   - Career ERA-FIP stabilization: n0 ≈ 6472 TBF (~1505 IP)
#   - At career IP >= 2000: r = 0.170, regression slope = 0.515
#   - Only 3.1% of 100-300 IP careers have a significant ERA-FIP gap
#   - But 56% of 2000-3000 IP careers do
#
# We use James-Stein shrinkage:
#   regressed_gap = (career_gap × career_TBF) / (career_TBF + n0)
# where n0 = 6472 is the stabilization point for ERA-FIP.

ERA_FIP_STABILIZATION_TBF = 2000  # From data: ~1505 IP to 50% signal


def compute_career_era_fip_gap(player_data: pd.DataFrame) -> dict:
    """
    Compute the IP-weighted career ERA-FIP gap and regress it toward zero
    based on career sample size.
    
    Args:
        player_data: DataFrame of a single pitcher's career (one row per season).
                     Must have 'ERA', 'FIP', 'IP', and 'TBF' columns.
    
    Returns:
        Dict with:
            'raw_gap': IP-weighted career (ERA - FIP) before regression
            'regressed_gap': Bayesian-regressed gap (toward 0)
            'career_tbf': Total career batters faced
            'career_ip': Total career IP
            'signal_fraction': career_tbf / (career_tbf + n0) — how much to trust
    """
    result = {
        'raw_gap': 0.0,
        'regressed_gap': 0.0,
        'career_tbf': 0.0,
        'career_ip': 0.0,
        'signal_fraction': 0.0,
    }
    
    # Need ERA, FIP, and IP columns
    required = {'ERA', 'FIP', 'IP'}
    if not required.issubset(set(player_data.columns)):
        return result
    
    valid = player_data.dropna(subset=['ERA', 'FIP', 'IP'])
    valid = valid[valid['IP'] > 0]
    
    if len(valid) == 0:
        return result
    
    # IP-weighted career ERA-FIP gap
    total_ip = valid['IP'].sum()
    if total_ip <= 0:
        return result
    
    weighted_era = (valid['ERA'] * valid['IP']).sum() / total_ip
    weighted_fip = (valid['FIP'] * valid['IP']).sum() / total_ip
    raw_gap = weighted_era - weighted_fip
    
    # Career TBF (use actual if available, else estimate from IP)
    if 'TBF' in valid.columns and valid['TBF'].notna().any():
        career_tbf = valid['TBF'].sum()
    else:
        career_tbf = total_ip * BF_PER_IP_RATIO
    
    # James-Stein shrinkage toward 0
    signal_fraction = career_tbf / (career_tbf + ERA_FIP_STABILIZATION_TBF)
    regressed_gap = raw_gap * signal_fraction
    
    result['raw_gap'] = raw_gap
    result['regressed_gap'] = regressed_gap
    result['career_tbf'] = career_tbf
    result['career_ip'] = total_ip
    result['signal_fraction'] = signal_fraction
    
    return result


def _apply_era_fip_adjustment(prediction: np.ndarray,
                               input_features: list,
                               era_fip_info: dict,
                               player_name: str = '') -> np.ndarray:
    """
    Adjust the model's ERA prediction using the regressed career ERA-FIP gap.
    
    Final ERA = reconstructed_FIP + regressed_career_ERA_FIP_gap
    
    This replaces the model's direct ERA prediction with one anchored to the
    component-based FIP, adjusted for the pitcher's genuine ERA-FIP tendency
    (if they have enough career IP for it to be statistically meaningful).
    
    Args:
        prediction: Array of predicted values (in real/unscaled space)
        input_features: Feature name list
        era_fip_info: Output of compute_career_era_fip_gap()
        player_name: For logging
    
    Returns:
        prediction array with ERA overwritten
    """
    if 'ERA' not in input_features or 'FIP' not in input_features:
        return prediction
    
    era_idx = input_features.index('ERA')
    fip_idx = input_features.index('FIP')
    
    reconstructed_fip = prediction[fip_idx]
    regressed_gap = era_fip_info.get('regressed_gap', 0.0)
    signal_frac = era_fip_info.get('signal_fraction', 0.0)
    
    # Only apply if there's meaningful signal
    model_era = prediction[era_idx]
    adjusted_era = reconstructed_fip + regressed_gap
    
    # Sanity bounds
    adjusted_era = np.clip(adjusted_era, 0.5, 10.0)
    
    if signal_frac > 0.1 and abs(regressed_gap) > 0.05:
        logger.debug(
            f"ERA-FIP adjustment for {player_name}: model_ERA={model_era:.3f}, "
            f"FIP={reconstructed_fip:.3f}, gap={regressed_gap:+.3f} "
            f"(signal={signal_frac:.1%}) → ERA={adjusted_era:.3f}"
        )
    
    prediction[era_idx] = adjusted_era
    return prediction


# =============================================================================
# CAREER ERA-SIERA DIFFERENTIAL (Alternative to ERA-FIP)
# =============================================================================
# While FIP is mechanically closer to ERA (shared HR component, same-season
# r=0.764), SIERA is a better predictor of *future* ERA (next-year r=0.403 vs
# 0.372 for FIP). This means the ERA-SIERA gap captures genuinely persistent
# pitcher-specific skill that goes beyond what K%, BB%, GB% explain — e.g.,
# BABIP management beyond GB%, sequencing ability, LOB% skill.
#
# Empirical properties of ERA-SIERA gap (7,714 pitcher-seasons, 2002-2025):
#   ICC = 0.124 (12.4% pitcher identity, vs 8.8% for ERA-FIP)
#   YoY r = 0.152 (vs 0.109 for ERA-FIP)
#   Stabilization: n0 = 3103 TBF ≈ 722 IP (vs 4552 TBF for ERA-FIP)
#   std = 0.957 (wider than ERA-FIP's 0.765)
#
# For same-season ERA derivation, FIP-based is more accurate (r=0.775 vs 0.618).
# ERA-SIERA is offered as a toggleable alternative for users who want the
# higher-ICC, faster-stabilizing gap for projection-oriented use cases.

ERA_SIERA_STABILIZATION_TBF = 3103  # From ICC analysis: ~722 IP to 50% signal


def compute_career_era_siera_gap(player_data: pd.DataFrame) -> dict:
    """
    Compute the IP-weighted career ERA-SIERA gap and regress it toward zero
    based on career sample size.

    SIERA is reconstructed from each season's K%, BB%, GB% using the OLS
    quadratic model (same coefficients as reconstruct_siera_from_components).
    This avoids needing a 'SIERA' column in the historical data — it's
    always computable from the three component rates.

    Args:
        player_data: DataFrame of a single pitcher's career (one row per season).
                     Must have 'ERA', 'K%', 'BB%', 'GB%', 'IP', and 'TBF' columns.

    Returns:
        Dict with:
            'raw_gap': IP-weighted career (ERA - SIERA) before regression
            'regressed_gap': Bayesian-regressed gap (toward 0)
            'career_tbf': Total career batters faced
            'career_ip': Total career IP
            'signal_fraction': career_tbf / (career_tbf + n0)
    """
    result = {
        'raw_gap': 0.0,
        'regressed_gap': 0.0,
        'career_tbf': 0.0,
        'career_ip': 0.0,
        'signal_fraction': 0.0,
    }

    required = {'ERA', 'K%', 'BB%', 'GB%', 'IP'}
    if not required.issubset(set(player_data.columns)):
        return result

    valid = player_data.dropna(subset=['ERA', 'K%', 'BB%', 'GB%', 'IP'])
    valid = valid[valid['IP'] > 0]

    if len(valid) == 0:
        return result

    total_ip = valid['IP'].sum()
    if total_ip <= 0:
        return result

    # Reconstruct SIERA for each season from components
    siera_values = np.array([
        reconstruct_siera_from_components(row['K%'], row['BB%'], row['GB%'])
        for _, row in valid.iterrows()
    ])

    # IP-weighted career means
    weighted_era = (valid['ERA'].values * valid['IP'].values).sum() / total_ip
    weighted_siera = (siera_values * valid['IP'].values).sum() / total_ip
    raw_gap = weighted_era - weighted_siera

    # Career TBF
    if 'TBF' in valid.columns and valid['TBF'].notna().any():
        career_tbf = valid['TBF'].sum()
    else:
        career_tbf = total_ip * BF_PER_IP_RATIO

    # James-Stein shrinkage toward 0
    signal_fraction = career_tbf / (career_tbf + ERA_SIERA_STABILIZATION_TBF)
    regressed_gap = raw_gap * signal_fraction

    result['raw_gap'] = raw_gap
    result['regressed_gap'] = regressed_gap
    result['career_tbf'] = career_tbf
    result['career_ip'] = total_ip
    result['signal_fraction'] = signal_fraction

    return result


def _apply_era_siera_adjustment(prediction: np.ndarray,
                                 input_features: list,
                                 era_siera_info: dict,
                                 player_name: str = '') -> np.ndarray:
    """
    Adjust the model's ERA prediction using the regressed career ERA-SIERA gap.

    Final ERA = reconstructed_SIERA + regressed_career_ERA_SIERA_gap

    This is an alternative to ERA-FIP adjustment. Uses SIERA as the anchor
    instead of FIP. SIERA captures GB% → BABIP relationships that FIP misses,
    so the ERA-SIERA gap represents purer pitcher-specific effects (sequencing,
    LOB%, BABIP management beyond GB%).

    Args:
        prediction: Array of predicted values (in real/unscaled space)
        input_features: Feature name list
        era_siera_info: Output of compute_career_era_siera_gap()
        player_name: For logging

    Returns:
        prediction array with ERA overwritten
    """
    if 'ERA' not in input_features or 'SIERA' not in input_features:
        return prediction

    era_idx = input_features.index('ERA')
    siera_idx = input_features.index('SIERA')

    reconstructed_siera = prediction[siera_idx]
    regressed_gap = era_siera_info.get('regressed_gap', 0.0)
    signal_frac = era_siera_info.get('signal_fraction', 0.0)

    model_era = prediction[era_idx]
    adjusted_era = reconstructed_siera + regressed_gap

    # Sanity bounds
    adjusted_era = np.clip(adjusted_era, 0.5, 10.0)

    if signal_frac > 0.1 and abs(regressed_gap) > 0.05:
        logger.debug(
            f"ERA-SIERA adjustment for {player_name}: model_ERA={model_era:.3f}, "
            f"SIERA={reconstructed_siera:.3f}, gap={regressed_gap:+.3f} "
            f"(signal={signal_frac:.1%}) → ERA={adjusted_era:.3f}"
        )

    prediction[era_idx] = adjusted_era
    return prediction


# =============================================================================
# SIERA RECONSTRUCTION FROM COMPONENTS (OLS Approximation)
# =============================================================================
# Reconstruct SIERA from K%, BB%, GB% using an OLS-derived quadratic model.
#
# NOTE: This is NOT the published SIERA formula. The published formula uses
# SO/PA, BB/PA, and a "net ground ball rate" (GB-FB-PU)/PA with a conditional
# sign on the squared term:
#
#   SIERA = 6.145 - 16.986(SO/PA) + 11.434(BB/PA) - 1.858((GB-FB-PU)/PA)
#         + 7.653(SO/PA)^2 +/- 6.664((GB-FB-PU)/PA)^2
#         + 10.130(SO/PA)((GB-FB-PU)/PA) - 5.195(BB/PA)((GB-FB-PU)/PA)
#
# We deliberately use an OLS proxy instead because it empirically outperforms
# the published formula at reproducing FanGraphs SIERA values:
#
#   Our OLS proxy (K%, BB%, GB%):   r=0.896, RMSE=0.346  (N=5133, 2010-2025)
#   Published SIERA (exact coeffs): r=0.875, RMSE=0.438
#   Refit OLS on published terms:   r=0.886, RMSE=0.351
#
# The published coefficients appear dated or fit to a different sample. The
# (GB-FB-PU)/PA term also introduces collinearity with K% and BB% (since
# BIP_rate = 1 - K% - BB% - HBP%), degrading OLS fit quality. Using GB%
# directly avoids this problem.
#
# OLS fit statistics (7,714 pitcher-seasons, 2002-2025, ≥50 IP):
#   Overall: R²=0.822, r=0.907, RMSE=0.321
#   Modern era (2020+): r=0.948, RMSE=0.332

# Regression coefficients (OLS on 2002-2025 data)
_SIERA_INTERCEPT = 6.8905
_SIERA_COEFS = {
    'K%':     -16.9845,
    'BB%':     +4.4756,
    'GB%':     -0.5257,
    'K%^2':    +3.9039,
    'BB%^2':  +11.9800,
    'GB%^2':   -6.4369,
    'K%*BB%':  -6.3522,
    'K%*GB%': +12.0693,
    'BB%*GB%':+13.9631,
}


def reconstruct_siera_from_components(k_pct: float, bb_pct: float,
                                       gb_pct: float) -> float:
    """
    Reconstruct SIERA from per-TBF strikeout/walk rates and ground ball rate.
    
    Uses a full quadratic model (linear + squared + interaction terms) derived
    from 7,714 pitcher-seasons (2002-2025). The formula captures the non-linear
    relationships between these components and run prevention.
    
    Args:
        k_pct:  Strikeout rate (K/TBF), e.g. 0.25 for 25%
        bb_pct: Walk rate (BB/TBF), e.g. 0.08 for 8%
        gb_pct: Ground ball rate (GB/BIP), e.g. 0.45 for 45%
    
    Returns:
        Reconstructed SIERA value
    """
    return (
        _SIERA_INTERCEPT
        + _SIERA_COEFS['K%']     * k_pct
        + _SIERA_COEFS['BB%']    * bb_pct
        + _SIERA_COEFS['GB%']    * gb_pct
        + _SIERA_COEFS['K%^2']   * k_pct ** 2
        + _SIERA_COEFS['BB%^2']  * bb_pct ** 2
        + _SIERA_COEFS['GB%^2']  * gb_pct ** 2
        + _SIERA_COEFS['K%*BB%'] * k_pct * bb_pct
        + _SIERA_COEFS['K%*GB%'] * k_pct * gb_pct
        + _SIERA_COEFS['BB%*GB%']* bb_pct * gb_pct
    )


def _apply_siera_reconstruction(prediction: np.ndarray,
                                 input_features: list,
                                 player_name: str = '') -> np.ndarray:
    """
    Overwrite the model's direct SIERA prediction with a value reconstructed
    from the predicted K%, BB%, and GB% components.
    
    Analogous to _apply_fip_reconstruction but for SIERA. Only applied when
    all required features (K%, BB%, GB%, SIERA) are present in the model.
    
    Args:
        prediction: Array of predicted values (in real/unscaled space)
        input_features: List of feature names corresponding to prediction indices
        player_name: For logging only
    
    Returns:
        prediction array with SIERA overwritten (mutated in-place and returned)
    """
    required = {'K%', 'BB%', 'GB%', 'SIERA'}
    if not required.issubset(set(input_features)):
        return prediction  # Not all components available — keep model SIERA
    
    k_idx = input_features.index('K%')
    bb_idx = input_features.index('BB%')
    gb_idx = input_features.index('GB%')
    siera_idx = input_features.index('SIERA')
    
    model_siera = prediction[siera_idx]
    recon_siera = reconstruct_siera_from_components(
        k_pct=prediction[k_idx],
        bb_pct=prediction[bb_idx],
        gb_pct=prediction[gb_idx],
    )
    
    # Sanity bounds: SIERA should be in [1.0, 8.0] for any reasonable pitcher
    recon_siera = np.clip(recon_siera, 1.0, 8.0)
    
    if abs(model_siera - recon_siera) > 1.0:
        logger.debug(
            f"SIERA reconstruction for {player_name}: model={model_siera:.3f} → "
            f"reconstructed={recon_siera:.3f} (Δ={recon_siera - model_siera:+.3f})"
        )
    
    prediction[siera_idx] = recon_siera
    return prediction


# =============================================================================
# PITCHER AGING CONSTRAINTS
# =============================================================================
# Prevents the autoregressive prediction loop from extrapolating recent
# improvement trends indefinitely. Without these constraints, a young pitcher
# trending upward (e.g., Hunter Greene) will show K% rising forever and BB%
# going negative by age 40.
#
# Based on empirical aging curves from 1950-2024 historical data (N≥30 per age):
#   K%:  Peaks ~24-25, declines ~0.003/yr in 20s, ~0.005-0.01/yr in 30s+
#   BB%: Improves until ~25-26, then slowly worsens ~0.001-0.003/yr
#   HBP%: Fairly random, minimal aging signal
#   GB%: Stable-to-declining with age (slight decrease in ground balls)
#
# The constraints work in two layers:
#   1. PHYSICAL BOUNDS: Hard min/max for each rate stat (BB% can't go negative)
#   2. IMPROVEMENT CAPS: Maximum year-over-year improvement per age band
#      (still allows the model to predict decline at any age)

# Physical bounds: absolute min/max for pitcher rate stats
PITCHER_PHYSICAL_BOUNDS = {
    'K%':   (0.05, 0.50),   # 5%-50% (league range ~15-40%)
    'BB%':  (0.02, 0.20),   # 2%-20% (hard floor prevents negative walks)
    'HBP%': (0.002, 0.04),  # 0.2%-4% (small positive floor)
    'BABIP':(0.220, 0.380),  # .220-.380 (league avg ~.295, tight range for pitchers)
    'GB%':  (0.15, 0.70),   # 15%-70% (extreme fly-ball to extreme ground-ball)
    'FB%':  (0.10, 0.65),   # 10%-65% (extreme GB pitcher to extreme FB pitcher)
    'LD%':  (0.10, 0.35),   # 10%-35% (league avg ~20%, very stable)
    'HR/FB':(0.02, 0.25),   # 2%-25% (data range 0-32%, but >25% is unsustainable)
    'FIP':  (1.5, 8.0),     # Reasonable FIP range
    'ERA':  (1.5, 10.0),    # Reasonable ERA range
    'SIERA':(1.5, 8.0),     # Reasonable SIERA range
}

# Aging constraint parameters: max improvement per year by age band
# Format: {stat: (peak_age, direction, [(age_lo, age_hi, max_improvement), ...])}
# direction: 'higher_is_better' (K%) or 'lower_is_better' (BB%, HR%, FIP, ERA, SIERA)
# max_improvement is in absolute terms (positive number = how much the stat can
# improve in the favorable direction per year)
PITCHER_AGING_PARAMS = {
    'K%': {
        'peak_age': 25,
        'direction': 'higher_is_better',
        'bands': [
            (25, 28, 0.005),   # ~0.5% per year improvement allowed
            (28, 32, 0.002),   # Very limited improvement
            (32, 36, 0.001),   # Almost no improvement
            (36, 50, 0.000),   # No improvement allowed
        ],
    },
    'BB%': {
        'peak_age': 26,
        'direction': 'lower_is_better',
        'bands': [
            (26, 30, 0.003),   # Can still improve command
            (30, 34, 0.001),   # Very limited improvement
            (34, 50, 0.000),   # No improvement allowed
        ],
    },
    'HBP%': {
        'peak_age': 30,
        'direction': 'lower_is_better',
        'bands': [
            (30, 50, 0.002),   # HBP is mostly random, lenient cap
        ],
    },
    'GB%': {
        'peak_age': 26,
        'direction': 'higher_is_better',
        'bands': [
            (26, 32, 0.010),   # GB% relatively stable, lenient cap
            (32, 50, 0.005),
        ],
    },
    'FB%': {
        'peak_age': 26,
        'direction': 'lower_is_better',  # Lower FB% means fewer HR opportunities
        'bands': [
            (26, 32, 0.010),   # Mirror GB% — batted ball profile is stable
            (32, 50, 0.005),
        ],
    },
    'LD%': {
        'peak_age': 26,
        'direction': 'neutral',  # LD% is largely out of pitcher control
        'bands': [
            (26, 32, 0.010),   # Very stable / noisy — lenient cap
            (32, 50, 0.005),
        ],
    },
    'HR/FB': {
        'peak_age': 24,
        'direction': 'lower_is_better',
        'bands': [
            (24, 30, 0.010),   # Very lenient — HR/FB is noisy, let regression handle it
            (30, 50, 0.005),
        ],
    },
}


def _get_max_improvement(stat: str, age: float) -> Optional[float]:
    """
    Get the maximum allowed year-over-year improvement for a stat at a given age.
    
    Returns None if the stat has no aging constraint or the age is below peak.
    """
    params = PITCHER_AGING_PARAMS.get(stat)
    if params is None:
        return None
    
    if age < params['peak_age']:
        return None  # Before peak age — no constraint on improvement
    
    for lo, hi, max_imp in params['bands']:
        if lo <= age < hi:
            return max_imp
    
    # Beyond the last band — use the last band's value
    if params['bands']:
        return params['bands'][-1][2]
    return None


# =============================================================================
# OUTPUT REGRESSION (POST-PREDICTION)
# =============================================================================
# The input sequence is regressed, but the model's OUTPUT predictions can still
# be extreme for short-career pitchers.  The skip connection anchors to the last
# regressed input, and the learned delta is small, so the output is close to the
# regressed input — but K%, BB%, HBP% still carry more signal than
# warranted by a 48 IP career.
#
# Output regression applies Bayesian shrinkage to the model's predicted rate
# stats BEFORE FIP/ERA reconstruction.  This ensures the component rates that
# feed into FIP and ERA are appropriately conservative.
#
# Formula per stat:
#   regressed_pred = (pred × career_tbf + league_avg × n0) / (career_tbf + n0)
#
# where n0 is the stat-specific stabilization point (same as input regression).

# Features to skip for output regression (volume/exposure measures, not rates)
_OUTPUT_REGRESSION_SKIP = {
    'Age', 'IP', 'Inn', 'G', 'PA', 'TBF', 'FBv',
    # Derived stats — regressing them independently double-counts because
    # their components (K%, BB%, HBP%, GB%) are already regressed.
    # After output regression the caller reconstructs these from the
    # regressed components instead.
    'FIP', 'ERA', 'SIERA',
}

def _apply_output_regression(prediction: np.ndarray,
                              input_features: list,
                              career_tbf: float,
                              league_priors: Dict[str, float],
                              era: str = 'statcast',
                              player_name: str = '') -> np.ndarray:
    """
    Apply Bayesian shrinkage to the model's output prediction based on career
    sample size.  Rate stats are regressed toward league average; the strength
    of regression depends on career TBF relative to each stat's stabilization
    point.

    Args:
        prediction: Model output array (one value per input feature, unscaled).
        input_features: Feature names matching prediction indices.
        career_tbf: Total career batters faced.
        league_priors: Dict mapping feature name → league average value.
        era: Era string for stabilization point lookup.
        player_name: For logging.

    Returns:
        Regression-adjusted prediction array (modified in place).
    """
    from core.reliability import _get_stabilization_point

    pred = prediction.copy()

    for i, feat in enumerate(input_features):
        if feat in _OUTPUT_REGRESSION_SKIP:
            continue

        n0 = _get_stabilization_point(feat, era=era, model_type='pitcher')
        if n0 is None:
            continue

        lg_avg = league_priors.get(feat)
        if lg_avg is None:
            continue

        # Bayesian shrinkage: weight prediction by career TBF, prior by n0
        weight = career_tbf / (career_tbf + n0)
        regressed = weight * pred[i] + (1.0 - weight) * lg_avg

        if abs(pred[i] - regressed) > 1e-6:
            logger.debug(
                f"  Output regression {player_name} {feat}: "
                f"{pred[i]:.4f} → {regressed:.4f} (weight={weight:.2f}, n0={n0}, tbf={career_tbf:.0f})"
            )
        pred[i] = regressed

    return pred


def _apply_pitcher_aging_constraints(prediction: np.ndarray,
                                      previous_values: np.ndarray,
                                      input_features: list,
                                      age: float,
                                      player_name: str = '') -> np.ndarray:
    """
    Apply aging constraints to a pitcher prediction.
    
    Two layers of constraints:
    1. Improvement caps: prevent year-over-year improvement beyond what's
       empirically plausible for the pitcher's age
    2. Physical bounds: hard min/max for each rate stat
    
    The model can still predict DECLINE at any age — these constraints only
    limit improvement (i.e., they're one-directional).
    
    Args:
        prediction: Array of predicted values (in real/unscaled space)
        previous_values: Previous year's values (from last prediction or actuals)
        input_features: List of feature names corresponding to prediction indices
        age: The pitcher's age for this prediction year
        player_name: For logging only
    
    Returns:
        prediction array with aging constraints applied (mutated in-place)
    """
    for stat, params in PITCHER_AGING_PARAMS.items():
        if stat not in input_features:
            continue
        
        idx = input_features.index(stat)
        current_val = prediction[idx]
        prev_val = previous_values[idx]
        
        max_imp = _get_max_improvement(stat, age)
        if max_imp is None:
            continue  # Below peak age or no constraint
        
        direction = params['direction']
        
        if direction == 'higher_is_better':
            # K%: higher = better, improvement = current > previous
            improvement = current_val - prev_val
            if improvement > max_imp:
                capped_val = prev_val + max_imp
                logger.debug(
                    f"Aging cap ({player_name}, age {age:.0f}): "
                    f"{stat} {current_val:.4f} → {capped_val:.4f} "
                    f"(max improvement {max_imp:.4f}/yr)"
                )
                prediction[idx] = capped_val
        elif direction == 'neutral':
            # LD%: neither direction is improvement — cap total change
            change = abs(current_val - prev_val)
            if change > max_imp:
                sign = 1.0 if current_val > prev_val else -1.0
                capped_val = prev_val + sign * max_imp
                logger.debug(
                    f"Aging cap ({player_name}, age {age:.0f}): "
                    f"{stat} {current_val:.4f} → {capped_val:.4f} "
                    f"(max change {max_imp:.4f}/yr)"
                )
                prediction[idx] = capped_val
        else:
            # BB%: lower = better, improvement = current < previous
            improvement = prev_val - current_val
            if improvement > max_imp:
                capped_val = prev_val - max_imp
                logger.debug(
                    f"Aging cap ({player_name}, age {age:.0f}): "
                    f"{stat} {current_val:.4f} → {capped_val:.4f} "
                    f"(max improvement {max_imp:.4f}/yr)"
                )
                prediction[idx] = capped_val
    
    # Physical bounds (always applied regardless of age)
    for stat, (lo, hi) in PITCHER_PHYSICAL_BOUNDS.items():
        if stat not in input_features:
            continue
        idx = input_features.index(stat)
        original = prediction[idx]
        prediction[idx] = np.clip(prediction[idx], lo, hi)
        if prediction[idx] != original:
            logger.debug(
                f"Physical bound ({player_name}): {stat} "
                f"{original:.4f} → {prediction[idx]:.4f} [{lo}, {hi}]"
            )
    
    return prediction


# =============================================================================
# STATCAST QUALITY ADJUSTMENT
# =============================================================================

def _apply_statcast_adjustment(
    player_data_regressed: pd.DataFrame,
    player_data_raw: pd.DataFrame,
    input_features: List[str],
    config,
    ip_threshold: float = 20,
) -> pd.DataFrame:
    """
    Nudge regressed classical stats based on Stuff+/Location+/Pitching+.

    For each season that has Statcast metrics, apply a small multiplicative
    adjustment to correlated classical features:

        z = Σ (coefficient × (metric - 100) / 100)
        adjusted = regressed × (1 + clamp(z, -cap, +cap))

    This injects pitch-quality information without requiring Statcast features
    in the model architecture.  Applied only to seasons with valid Statcast
    data (2020+), leaving earlier seasons unchanged.

    Args:
        player_data_regressed: Regressed player DataFrame (modified in place)
        player_data_raw: Original player DataFrame (source of Statcast columns)
        input_features: Classical features the model uses
        config: Config object with STATCAST_ADJUSTMENT_MAP and _CAP
        ip_threshold: Minimum IP for a season to receive adjustment

    Returns:
        Modified regressed DataFrame
    """
    adj_map = getattr(config, 'STATCAST_ADJUSTMENT_MAP', None)
    cap = getattr(config, 'STATCAST_ADJUSTMENT_CAP', 0.10)
    if not adj_map:
        return player_data_regressed

    result = player_data_regressed
    adjusted_count = 0

    for idx in result.index:
        if idx not in player_data_raw.index:
            continue
        raw_row = player_data_raw.loc[idx]

        # Skip low-IP seasons
        if raw_row.get('IP', 0) < ip_threshold:
            continue

        # Accumulate per-classical-stat adjustment from all Statcast metrics
        z_accum: Dict[str, float] = {}
        has_any_statcast = False

        for sc_metric, mapping in adj_map.items():
            sc_val = raw_row.get(sc_metric)
            if sc_val is None or (isinstance(sc_val, float) and np.isnan(sc_val)):
                continue
            has_any_statcast = True
            deviation = (sc_val - 100.0) / 100.0
            for classical_feat, coeff in mapping.items():
                if classical_feat in input_features:
                    z_accum[classical_feat] = z_accum.get(classical_feat, 0.0) + coeff * deviation

        if not has_any_statcast:
            continue

        # Apply clamped adjustment to each affected classical stat
        for feat, z in z_accum.items():
            z_clamped = max(-cap, min(cap, z))
            if abs(z_clamped) < 1e-6:
                continue
            old_val = result.at[idx, feat]
            if not np.isnan(old_val):
                result.at[idx, feat] = old_val * (1.0 + z_clamped)
                adjusted_count += 1

    if adjusted_count > 0:
        logger.debug(f"Statcast adjustment: {adjusted_count} stat-season values nudged")

    return result


# =============================================================================
# PITCHER PREDICTION
# =============================================================================

def predict_future_stats_pitcher(player_id: str, input_features: List[str], model, 
                                scaler, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                role: str, future_years: int = 16, seq_length: int = 4,
                                target_year: int = None,
                                league_priors: Optional[Dict[str, float]] = None,
                                config=None) -> List[Dict]:
    """
    Predict future stats for a pitcher with reliability regression.
    
    Key design:
    1. Reliability regression: each season's rate stats are regressed toward the
       player's TBF-weighted career mean (or league average for rookies) based on
       sample size and stat-specific stabilization rates. This means a 75 IP season
       keeps its K% mostly intact but its ERA gets pulled toward career norm.
    2. Regressed career mean padding: when the sequence is shorter than seq_length,
       padding uses the player's regressed career average — a principled true-talent
       estimate rather than duplicating the earliest season.
    3. Lower IP threshold for inclusion: since regression handles reliability, we
       can include more seasons (≥20 IP for SP, ≥10 IP for RP) without worrying
       about small-sample noise distorting the sequence.
    
    Args:
        model: Trained ImprovedLSTM model
        scaler: Fitted scaler
        target_year: The year projections should start from (e.g., 2026).
        league_priors: Pre-computed league average priors per feature (from caller).
                      Used as fallback for rookies with <2 career seasons.
    """
    from core.reliability import (
        regress_player_sequence, 
        compute_regressed_career_mean, 
        get_era_for_features,
    )
    from .prediction import _is_park_factor_enabled
    
    # Get initial player data
    player_data = raw_df[raw_df['IDfg'] == player_id].sort_values('Season')
    if len(player_data) < 1:
        return []
    
    # Check for required features — skip players with too many NaN values
    required_features = [f for f in input_features if f != 'Age']
    last_valid_season = player_data[player_data['IP'] >= 15].tail(1)
    if last_valid_season.empty:
        last_valid_season = player_data.tail(1)
    
    nan_count = last_valid_season[required_features].isna().sum().sum()
    if nan_count > len(required_features) * 0.3:
        logger.debug(f"Skipping player {player_id} - too many NaN features ({nan_count}/{len(required_features)})")
        return []
        
    # Get player info
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        return []
        
    last_season = player_data['Season'].max()
    last_age = player_data[player_data['Season'] == last_season]['Age'].iloc[0]
    
    # Determine the projection start year
    if target_year is not None and last_season < target_year:
        years_missed = target_year - 1 - last_season
        last_age = last_age + years_missed
        last_season = target_year - 1
    
    # Store player context for post-processing
    player_context = {
        'career_high_ip': player_data['IP'].max(),
        'recent_ip': player_data.tail(3)['IP'].mean(),
        'last_fbv': None,
        'last_age': last_age,
        'role': role,
        'recent_surgery': False,
        'recent_performance': {},
    }
    
    # Detect recent surgery/injury
    recent_surgery_detected = False
    if len(player_data) >= 2:
        recent_ip = player_data.tail(2)['IP'].values
        prior_avg = player_data.iloc[:-2]['IP'].mean() if len(player_data) > 2 else player_data.iloc[0]['IP']
        if len(recent_ip) >= 2:
            if (recent_ip[-2] < 50 and prior_avg > 100) or (recent_ip[-1] >= 80 and prior_avg > 130):
                recent_surgery_detected = True
        elif len(recent_ip) > 0 and recent_ip[-1] < 50 and prior_avg > 100:
            recent_surgery_detected = True
    
    player_context['recent_surgery'] = recent_surgery_detected
    
    if recent_surgery_detected:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            most_recent_valid = valid_seasons.iloc[-1]
            player_context['recent_performance'] = {
                'recent_era': most_recent_valid.get('ERA', 4.5),
                'recent_fip': most_recent_valid.get('FIP', 4.5),
                'recent_k_pct': most_recent_valid.get('K%', 0.22),
                'recent_bb_pct': most_recent_valid.get('BB%', 0.09),
            }
    
    if 'FBv' in player_data.columns:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            player_context['last_fbv'] = valid_seasons.iloc[-1]['FBv']
    if 'Stuff+' in player_data.columns:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            player_context['last_stuff'] = valid_seasons.iloc[-1]['Stuff+']
    
    # =========================================================================
    # QUALIFICATION & IP THRESHOLDS
    # =========================================================================
    sequence_ip_threshold = 20 if role == 'SP' else 10
    qualification_ip_threshold = 45 if role == 'SP' else 15
    
    # Check if player qualifies for predictions
    recent_ip = player_data.tail(2)['IP'].max()
    if recent_ip < qualification_ip_threshold:
        logger.debug(f"Skipping {player_name} - insufficient recent IP ({recent_ip:.1f} < {qualification_ip_threshold})")
        return []
    
    # =========================================================================
    # RELIABILITY REGRESSION
    # =========================================================================
    era = get_era_for_features(input_features)
    recency_halflife = getattr(config, 'PRIOR_RECENCY_HALFLIFE', 0) if config else 0
    league_weight_overrides = getattr(config, 'PRIOR_LEAGUE_WEIGHT_OVERRIDES', None) if config else None
    player_data_regressed = regress_player_sequence(
        player_data, input_features, model_type='pitcher', era=era,
        league_priors=league_priors, recency_halflife=recency_halflife,
        league_weight_overrides=league_weight_overrides,
        seq_length=seq_length, sequence_ip_threshold=sequence_ip_threshold,
    )
    
    # Compute regressed career mean for padding
    career_mean = compute_regressed_career_mean(
        player_data, input_features, model_type='pitcher', era=era,
        league_priors=league_priors, recency_halflife=recency_halflife,
        league_weight_overrides=league_weight_overrides,
        seq_length=seq_length, sequence_ip_threshold=sequence_ip_threshold,
    )
    
    # =========================================================================
    # STATCAST QUALITY ADJUSTMENT
    # =========================================================================
    if getattr(config, 'ENABLE_STATCAST_ADJUSTMENT', False) and config is not None:
        player_data_regressed = _apply_statcast_adjustment(
            player_data_regressed, player_data, input_features, config,
            ip_threshold=sequence_ip_threshold,
        )
    
    # Build sequence from regressed data
    recent_seasons = player_data_regressed.tail(seq_length + 2)  # extra buffer for skipping
    sequence_data = []
    mask = []
    
    for idx, season in recent_seasons.iterrows():
        if season['IP'] >= sequence_ip_threshold:
            base_features = season[input_features].values
            sequence_data.append(base_features)
            mask.append(1)
    
    # Keep only the most recent seq_length valid seasons
    if len(sequence_data) > seq_length:
        sequence_data = sequence_data[-seq_length:]
        mask = mask[-seq_length:]
    
    # Check if we have any valid seasons at all
    if len(sequence_data) == 0:
        logger.debug(f"Skipping pitcher {player_name} - no seasons with IP >= {sequence_ip_threshold}")
        return []
    
    # Pad with regressed career mean if not enough seasons
    if len(sequence_data) < seq_length:
        padding_vector = np.array(
            [career_mean.get(f, 0.0) for f in input_features], dtype=np.float32
        )
        n_pad = seq_length - len(sequence_data)
        sequence_data = sequence_data + [padding_vector] * n_pad
        mask = mask + [0] * n_pad
    
    # Ensure numeric array
    current_sequence = np.array(sequence_data[-seq_length:], dtype=np.float32)
    mask = np.array(mask[-seq_length:], dtype=np.int64)
    
    # Park factor neutralization
    park_factor_enabled = _is_park_factor_enabled(role)
    
    if park_factor_enabled and 'Team' in player_data.columns:
        from core.park_factors import get_park_factor, EXCLUDED_STATS
        adjustable_indices = [
            i for i, f in enumerate(input_features) if f not in EXCLUDED_STATS
        ]
        valid_seasons_with_team = recent_seasons[recent_seasons['IP'] >= sequence_ip_threshold].tail(seq_length)
        season_teams = valid_seasons_with_team['Team'].tolist() if 'Team' in valid_seasons_with_team.columns else []
        last_team = player_data['Team'].iloc[-1] if len(player_data) > 0 else None
        n_actual = len(season_teams)
        n_pad = seq_length - n_actual
        all_teams = season_teams + [last_team] * n_pad
        
        for row_idx, team in enumerate(all_teams):
            pf = get_park_factor(team)
            if pf != 1.0:
                for col_idx in adjustable_indices:
                    current_sequence[row_idx, col_idx] = current_sequence[row_idx, col_idx] / pf
    
    # Final NaN/Inf safety check
    if np.isnan(current_sequence).any() or np.isinf(current_sequence).any():
        current_sequence = np.nan_to_num(current_sequence, nan=0.0, posinf=0.0, neginf=0.0)
        logger.warning(f"NaN/Inf cleaned from sequence for {player_name}")
    
    device = next(model.parameters()).device
    predictions_list = []
    
    # Get number of base features (model was trained on base features only)
    n_features = len(input_features)
    
    # In-loop reconstruction config flags
    enable_fip_recon = getattr(config, 'ENABLE_FIP_RECONSTRUCTION', True) if config else True
    enable_siera_recon = getattr(config, 'ENABLE_SIERA_RECONSTRUCTION', False) if config else False
    enable_era_fip = getattr(config, 'ENABLE_ERA_FIP_ADJUSTMENT', True) if config else True
    enable_era_siera = getattr(config, 'ENABLE_ERA_SIERA_ADJUSTMENT', False) if config else False
    enable_output_reg = getattr(config, 'ENABLE_OUTPUT_REGRESSION', False) if config else False
    enable_aging = getattr(config, 'ENABLE_PITCHER_AGING_CONSTRAINTS', False) if config else False

    # Career TBF for output regression (incremented each projected year)
    if enable_output_reg:
        if 'TBF' in player_data.columns and player_data['TBF'].notna().any():
            career_tbf = float(player_data['TBF'].sum())
        elif 'IP' in player_data.columns:
            career_tbf = float(player_data['IP'].sum()) * BF_PER_IP_RATIO
        else:
            career_tbf = 0.0
        tbf_per_season = 700.0 if role == 'SP' else 250.0

    # Track previous prediction for aging constraints
    previous_prediction = None

    # Compute career ERA-FIP / ERA-SIERA gap (needed for ERA derivation in the loop)
    era_fip_info = compute_career_era_fip_gap(player_data)
    era_siera_info = compute_career_era_siera_gap(player_data)
    
    # Generate predictions
    for year in range(1, future_years + 1):
        # Scale the sequence
        sequence_scaled = scaler.transform(current_sequence)
        
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
        mask_tensor = torch.LongTensor(mask).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(sequence_tensor, mask_tensor.sum(1))
            prediction = prediction.cpu().numpy()
        
        # Inverse transform to get actual values
        prediction_constrained = scaler.inverse_transform(prediction)[0]
        
        # Physical bounds to prevent autoregressive divergence
        for stat, (lo, hi) in PITCHER_PHYSICAL_BOUNDS.items():
            if stat in input_features:
                idx = input_features.index(stat)
                prediction_constrained[idx] = np.clip(prediction_constrained[idx], lo, hi)
        
        # Batted-ball distribution: GB% + FB% + LD% = 1
        # All three are model features — normalize to sum to exactly 1
        # so they stay internally consistent across autoregressive steps.
        _BB_STATS = ['GB%', 'FB%', 'LD%']
        _bb_indices = [input_features.index(s) for s in _BB_STATS if s in input_features]
        if len(_bb_indices) == 3:
            _bb_sum = sum(prediction_constrained[i] for i in _bb_indices)
            if _bb_sum > 0:
                _scale = 1.0 / _bb_sum
                for i in _bb_indices:
                    prediction_constrained[i] *= _scale
        
        # FIP reconstruction from components (HR% derived from HR/FB × FB%)
        if enable_fip_recon:
            prediction_constrained = _apply_fip_reconstruction(
                prediction_constrained, input_features, player_name
            )
        
        # SIERA reconstruction from components (if enabled)
        if enable_siera_recon:
            prediction_constrained = _apply_siera_reconstruction(
                prediction_constrained, input_features, player_name
            )
        
        # ERA derivation: ERA = anchor + regressed career gap.
        if enable_era_fip and not enable_era_siera:
            prediction_constrained = _apply_era_fip_adjustment(
                prediction_constrained, input_features, era_fip_info, player_name
            )
        if enable_era_siera:
            prediction_constrained = _apply_era_siera_adjustment(
                prediction_constrained, input_features, era_siera_info, player_name
            )
        
        # Output regression: shrink component rates toward league average
        # based on career sample size. Applied AFTER reconstruction so that
        # component rates (K%, BB%, HBP%) are regressed, then FIP/ERA
        # are re-derived from the regressed components.
        if enable_output_reg and league_priors:
            prediction_constrained = _apply_output_regression(
                prediction_constrained, input_features,
                career_tbf, league_priors,
                era=era, player_name=player_name
            )
            career_tbf += tbf_per_season

            # Re-reconstruct FIP/ERA/SIERA from regressed components
            if enable_fip_recon:
                prediction_constrained = _apply_fip_reconstruction(
                    prediction_constrained, input_features, player_name
                )
            if enable_siera_recon:
                prediction_constrained = _apply_siera_reconstruction(
                    prediction_constrained, input_features, player_name
                )
            if enable_era_fip and not enable_era_siera:
                prediction_constrained = _apply_era_fip_adjustment(
                    prediction_constrained, input_features, era_fip_info, player_name
                )
            if enable_era_siera:
                prediction_constrained = _apply_era_siera_adjustment(
                    prediction_constrained, input_features, era_siera_info, player_name
                )

        # Aging constraints: cap unrealistic year-over-year improvement.
        # Applied IN-LOOP so each capped prediction feeds back into the
        # next step, preventing compounding trend extrapolation.
        current_age = last_age + year
        if enable_aging and previous_prediction is not None:
            prediction_constrained = _apply_pitcher_aging_constraints(
                prediction_constrained, previous_prediction,
                input_features, current_age, player_name
            )
        # For year 1, compare against the last actual season in the sequence
        elif enable_aging and previous_prediction is None:
            prev_vals = current_sequence[int(mask.sum()) - 1]
            prediction_constrained = _apply_pitcher_aging_constraints(
                prediction_constrained, prev_vals,
                input_features, current_age, player_name
            )

        # Re-reconstruct derived stats after aging constraints modify components.
        # Aging can change K%, BB%, HR/FB etc., which invalidates the
        # previously reconstructed FIP/SIERA/ERA values.
        if enable_aging:
            if enable_fip_recon:
                prediction_constrained = _apply_fip_reconstruction(
                    prediction_constrained, input_features, player_name
                )
            if enable_siera_recon:
                prediction_constrained = _apply_siera_reconstruction(
                    prediction_constrained, input_features, player_name
                )
            if enable_era_fip and not enable_era_siera:
                prediction_constrained = _apply_era_fip_adjustment(
                    prediction_constrained, input_features, era_fip_info, player_name
                )
            if enable_era_siera:
                prediction_constrained = _apply_era_siera_adjustment(
                    prediction_constrained, input_features, era_siera_info, player_name
                )

        previous_prediction = prediction_constrained.copy()

        pred_dict = {
            'Name': player_name,
            'Year': last_season + year,
            'Age': last_age + year,
            'Role': role,
            'IDfg': player_id
        }
        
        # Add predicted stats (except Age, which is metadata)
        for i, feature in enumerate(input_features):
            if feature == 'Age':
                continue
            pred_dict[feature] = prediction_constrained[i]
        
        # Derive per-9 rates from per-TBF rates using pitcher-specific BF/IP
        _k  = prediction_constrained[input_features.index('K%')]  if 'K%'  in input_features else 0.0
        _bb = prediction_constrained[input_features.index('BB%')] if 'BB%' in input_features else 0.0
        _hr = _derive_hr_pct(prediction_constrained, input_features)
        _hbp = prediction_constrained[input_features.index('HBP%')] if 'HBP%' in input_features else 0.0
        _babip = prediction_constrained[input_features.index('BABIP')] if 'BABIP' in input_features else 0.295
        _bf_per_ip = derive_bf_per_ip(_k, _bb, _hbp, _hr, _babip)
        pred_dict['K/9']  = _k  * _bf_per_ip * 9
        pred_dict['BB/9'] = _bb * _bf_per_ip * 9
        pred_dict['HR/9'] = _hr * _bf_per_ip * 9
        
        predictions_list.append(pred_dict)
        
        # Update sequence for next prediction
        next_sequence = prediction_constrained.copy()
        age_index = input_features.index('Age')
        next_sequence[age_index] = last_age + year + 1
        
        # Safety check: ensure no NaN/Inf in prediction before adding to sequence
        if np.isnan(next_sequence).any() or np.isinf(next_sequence).any():
            logger.warning(f"NaN/Inf detected in prediction for {player_name} year {last_season + year} - replacing with last valid values")
            last_valid = current_sequence[-1].copy()
            nan_mask = np.isnan(next_sequence) | np.isinf(next_sequence)
            next_sequence[nan_mask] = last_valid[nan_mask]
            next_sequence[age_index] = last_age + year + 1
        
        # Update sequence for next prediction (right-padded layout).
        n_valid = int(mask.sum())
        if n_valid < seq_length:
            current_sequence[n_valid] = next_sequence
            mask[n_valid] = 1
        else:
            current_sequence = np.vstack([current_sequence[1:], next_sequence])
    
    return predictions_list


def predict_all_pitchers(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame, 
    sp_model, 
    rp_model,
    sp_scaler, 
    rp_scaler, 
    sp_input_features: List[str],
    rp_input_features: List[str], 
    seq_length: int, 
    future_years: int = 16, 
    cutoff_year: int = 2024,
    sp_config = None,
    rp_config = None,
    roster_ids: Optional[Set[int]] = None
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all qualified pitchers.
    
    Identifies starting and relief pitchers from the cutoff year (and previous year 
    for returning/injured players), then generates multi-year projections for each.
    
    Args:
        raw_df: Historical pitcher data with Season, IDfg, IP, G, GS columns
        player_names: DataFrame mapping IDfg to Name
        sp_model: Trained LSTM model for starting pitchers
        rp_model: Trained LSTM model for relief pitchers
        sp_scaler: Fitted scaler for SP features
        rp_scaler: Fitted scaler for RP features
        sp_input_features: List of input features for SP model
        rp_input_features: List of input features for RP model
        seq_length: Number of historical seasons used as input sequence
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        roster_ids: Optional set of IDfg values for players on active rosters.
                   Roster pitchers who don't meet normal IP/G thresholds will be
                   recovered from historical data.
        
    Returns:
        DataFrame with predictions for all pitchers, or None if no predictions generated
    """
    from .prediction import _is_park_factor_enabled
    
    # Detect unified pitcher model mode
    unified = getattr(sp_config, 'UNIFIED_PITCHER_MODEL', False) if sp_config else False
    mode_label = "UNIFIED" if unified else "separate SP/RP"
    logger.info(f"Starting predictions for pitchers from cutoff year {cutoff_year} ({mode_label} mode)")
    
    # Pre-compute league average priors for reliability regression.
    from core.reliability import compute_league_priors_from_df, get_era_for_features
    sp_era = get_era_for_features(sp_input_features)
    rp_era = get_era_for_features(rp_input_features)
    sp_league_priors = compute_league_priors_from_df(
        raw_df, sp_input_features, model_type='pitcher',
        season=cutoff_year, window=3
    )
    rp_league_priors = compute_league_priors_from_df(
        raw_df, rp_input_features, model_type='pitcher',
        season=cutoff_year, window=3
    )
    logger.info(f"Computed league priors for SP ({len(sp_league_priors)} features) and RP ({len(rp_league_priors)} features)")
    
    # Log park factor adjustment status
    sp_pf = _is_park_factor_enabled('SP')
    rp_pf = _is_park_factor_enabled('RP')
    logger.info(f"Park factor adjustment: SP={'ENABLED' if sp_pf else 'DISABLED'}, RP={'ENABLED' if rp_pf else 'DISABLED'}")
    
    # Get current year and previous year pitchers
    pitchers_current = raw_df[raw_df['Season'] == cutoff_year].copy()
    pitchers_prev = raw_df[raw_df['Season'] == cutoff_year - 1].copy()
    
    # Calculate GS rates
    pitchers_current['GS_rate'] = pitchers_current['GS'] / pitchers_current['G']
    pitchers_prev['GS_rate'] = pitchers_prev['GS'] / pitchers_prev['G']
    
    # Get minimum IP thresholds from config or use defaults
    sp_min_ip = sp_config.MIN_IP_CURRENT if sp_config and hasattr(sp_config, 'MIN_IP_CURRENT') else 25
    # In unified mode rp_config is the SP config — fall back to 15 IP for RP qualification
    if unified:
        rp_min_ip = 15
    else:
        rp_min_ip = rp_config.MIN_IP_CURRENT if rp_config and hasattr(rp_config, 'MIN_IP_CURRENT') else 15
    
    # First determine current year roles by GS rate only
    qualified_current_sp = pitchers_current[
        (pitchers_current['IP'] >= sp_min_ip) & 
        (pitchers_current['G'] >= 6)
    ]
    qualified_current_rp = pitchers_current[
        (pitchers_current['IP'] >= rp_min_ip) & 
        (pitchers_current['G'] >= 15)
    ]
    
    # Use current year role if they appear at all
    sp_ids_current = set(qualified_current_sp[qualified_current_sp['GS_rate'] >= 0.8]['IDfg'])
    rp_ids_current = set(qualified_current_rp[qualified_current_rp['GS_rate'] < 0.8]['IDfg'])
    
    # Only look at previous year for players missing from current year
    missing_current = set(pitchers_prev['IDfg']) - set(pitchers_current['IDfg'])
    sp_ids_prev = set(pitchers_prev[
        (pitchers_prev['IDfg'].isin(missing_current)) &
        (pitchers_prev['IP'] >= sp_min_ip) & 
        (pitchers_prev['G'] >= 6) & 
        (pitchers_prev['GS_rate'] >= 0.8)
    ]['IDfg'])
    
    rp_ids_prev = set(pitchers_prev[
        (pitchers_prev['IDfg'].isin(missing_current)) &
        (pitchers_prev['IP'] >= rp_min_ip) & 
        (pitchers_prev['G'] >= 15) & 
        (pitchers_prev['GS_rate'] < 0.8)
    ]['IDfg'])
    
    # Combine IDs
    sp_ids = sp_ids_current.union(sp_ids_prev)
    rp_ids = rp_ids_current.union(rp_ids_prev)
    
    logger.info(f"Found {len(sp_ids_current)} qualified {cutoff_year} SPs and {len(sp_ids_prev)} returning/recovering SPs")
    logger.info(f"Found {len(rp_ids_current)} qualified {cutoff_year} RPs and {len(rp_ids_prev)} returning/recovering RPs")
    
    # =========================================================================
    # ROSTER RECOVERY: recover roster pitchers who didn't meet normal thresholds
    # =========================================================================
    if roster_ids is not None:
        all_pitcher_ids = sp_ids | rp_ids
        missing_roster = roster_ids - all_pitcher_ids
        if missing_roster:
            recovered_sp = set()
            recovered_rp = set()
            for pid in missing_roster:
                pitcher_hist = raw_df[
                    (raw_df['IDfg'] == pid) &
                    (raw_df['Season'] <= cutoff_year) &
                    (raw_df['IP'] >= 10)
                ]
                if len(pitcher_hist) == 0:
                    continue
                recent = pitcher_hist.sort_values('Season').iloc[-1]
                gs_rate = recent['GS'] / recent['G'] if recent['G'] > 0 else 0
                if gs_rate >= 0.8:
                    sp_ids.add(pid)
                    recovered_sp.add(pid)
                else:
                    rp_ids.add(pid)
                    recovered_rp.add(pid)
            total_recovered = len(recovered_sp) + len(recovered_rp)
            logger.info(
                f"Roster recovery: {total_recovered} pitchers recovered "
                f"({len(recovered_sp)} SP, {len(recovered_rp)} RP) from historical data "
                f"({len(missing_roster)} roster pitchers were missing)"
            )
    
    # Target year is cutoff_year + 1
    target_year = cutoff_year + 1
    
    all_predictions = []
    
    if unified:
        # Unified mode: run ALL pitchers through the single (SP) model.
        # Role is still assigned from GS rate for downstream WAR calculations.
        all_ids_with_role = [(pid, 'SP') for pid in sp_ids] + [(pid, 'RP') for pid in rp_ids]
        logger.info(f"Generating unified predictions for {len(all_ids_with_role)} pitchers ({len(sp_ids)} SP, {len(rp_ids)} RP)...")
        for player_id, role in tqdm(all_ids_with_role, desc="All Pitchers (unified)"):
            player_historical_data = raw_df[
                (raw_df['IDfg'] == player_id) & 
                (raw_df['Season'] <= cutoff_year)
            ].copy()
            
            predictions = predict_future_stats_pitcher(
                player_id=player_id,
                input_features=sp_input_features,
                model=sp_model,
                scaler=sp_scaler,
                raw_df=player_historical_data,
                player_names=player_names,
                role=role,
                seq_length=seq_length,
                future_years=future_years,
                target_year=target_year,
                league_priors=sp_league_priors,
                config=sp_config
            )
            if predictions:
                all_predictions.extend(predictions)
    else:
        # Separate mode: SP and RP use their own model/scaler/features.
        # Predict SPs
        logger.info("Generating SP predictions...")
        for player_id in tqdm(sp_ids, desc="Starting Pitchers"):
            player_historical_data = raw_df[
                (raw_df['IDfg'] == player_id) & 
                (raw_df['Season'] <= cutoff_year)
            ].copy()
            
            predictions = predict_future_stats_pitcher(
                player_id=player_id,
                input_features=sp_input_features,
                model=sp_model,
                scaler=sp_scaler,
                raw_df=player_historical_data,
                player_names=player_names,
                role='SP',
                seq_length=seq_length,
                future_years=future_years,
                target_year=target_year,
                league_priors=sp_league_priors,
                config=sp_config
            )
            if predictions:
                all_predictions.extend(predictions)
                
        # Predict RPs
        logger.info("Generating RP predictions...")
        for player_id in tqdm(rp_ids, desc="Relief Pitchers"):
            player_historical_data = raw_df[
                (raw_df['IDfg'] == player_id) & 
                (raw_df['Season'] <= cutoff_year)
            ].copy()
            
            predictions = predict_future_stats_pitcher(
                player_id=player_id,
                input_features=rp_input_features,
                model=rp_model,
                scaler=rp_scaler,
                raw_df=player_historical_data,
                player_names=player_names,
                role='RP',
                seq_length=seq_length,
                future_years=future_years,
                target_year=target_year,
                league_priors=rp_league_priors,
                config=rp_config
            )
            if predictions:
                all_predictions.extend(predictions)
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df = predictions_df.sort_values(['Year', 'Role', 'Name'], ascending=[True, True, True])
        return predictions_df
    else:
        logger.warning("No predictions were generated")
        return None
