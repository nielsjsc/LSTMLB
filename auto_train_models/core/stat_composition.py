"""
Stat Composition — derive all display stats from 8 base batter components.

Base components (projected via Marcel + multivariate adjustments):
    K%, BB%, HBP%, ISO, BABIP, HR/FB, GB%, LD%

Composition tree
================
Intermediate:
    FB%      = 1 − GB% − LD%
    BIP      = PA × (1 − K% − BB% − HBP%)           balls in play
    AB       = PA − BB − HBP − SF                    at-bats
    HR       = HR/FB × FB% × BIP                     home runs
    non_HR_H = BABIP × (BIP − HR)                    non-HR hits
    H        = non_HR_H + HR                           total hits

Rate stats:
    AVG  = H / AB
    SLG  = AVG + ISO                                  (identity: ISO ≡ SLG − AVG)
    OBP  = (H + BB + HBP) / PA                        (≈ AB + BB + HBP + SF)

Counting stats (per 150 games, PA ≈ 650):
    2B + 2×3B  = ISO × AB − 3 × HR
    XBH_no_HR  = (ISO × AB − 3 × HR) / (1 + triple_share)
    3B         = triple_share × XBH_no_HR
    2B         = (1 − triple_share) × XBH_no_HR
    1B         = H − HR − 2B − 3B

Weighted stats:
    wOBA = (wBB×BB + wHBP×HBP + w1B×1B + w2B×2B + w3B×3B + wHR×HR) / PA
    wRC+ = 100 × ((wRAA/PA + lgR/PA + park_adj) / lgWRC/PA)

All functions accept scalars or numpy arrays for vectorised use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants  (2025 FanGraphs; must stay in sync w/ value_determination/config)
# ---------------------------------------------------------------------------
WOBA_WEIGHTS = {
    'wBB':  0.691,
    'wHBP': 0.722,
    'w1B':  0.882,
    'w2B':  1.252,
    'w3B':  1.584,
    'wHR':  2.037,
}

WOBA_SCALE       = 1.232
LG_WOBA          = 0.313
LG_RUNS_PER_PA   = 0.118
LG_WRC_PER_PA    = 0.117

DEFAULT_PA       = 650.0   # full-season PA assumption
DEFAULT_SF_RATE  = 0.007   # sac-fly share of PA (≈ 4.5 SF / 650 PA)
DEFAULT_3B_SHARE = 0.088   # league-avg 3B / (2B + 3B)

# Physical bounds for composed stats
COMPOSED_BOUNDS = {
    'AVG':  (0.100, 0.400),
    'SLG':  (0.150, 0.900),
    'OBP':  (0.150, 0.550),
    'wOBA': (0.150, 0.550),
    'wRC+': (0, 250),
}


# ---------------------------------------------------------------------------
# Core composition  (scalar / numpy-compatible)
# ---------------------------------------------------------------------------

def compose_rates(
    k_pct: float,
    bb_pct: float,
    hbp_pct: float,
    iso: float,
    babip: float,
    hr_fb: float,
    gb_pct: float,
    ld_pct: float,
    *,
    sf_rate: float = DEFAULT_SF_RATE,
) -> Dict[str, float]:
    """Derive AVG, SLG, OBP from the 8 base components.

    Parameters are all *rates* (0-1 scale, not percentage).
    Returns dict of {'AVG', 'SLG', 'OBP', 'FB%', 'HR_per_PA'}.
    """
    fb_pct = np.clip(1.0 - gb_pct - ld_pct, 0.0, 1.0)
    bip_rate = np.clip(1.0 - k_pct - bb_pct - hbp_pct, 0.01, 1.0)
    ab_rate = np.clip(1.0 - bb_pct - hbp_pct - sf_rate, 0.01, 1.0)

    # HR rate per PA
    hr_per_pa = hr_fb * fb_pct * bip_rate
    hr_per_pa = np.clip(hr_per_pa, 0.0, 0.12)  # cap ~78 HR / 650 PA

    # Hits via BABIP identity:  H = BABIP × (BIP − HR) + HR
    # where BIP − HR = PA × bip_rate − PA × hr_per_pa  (but we work per-PA)
    non_hr_hit_rate = babip * (bip_rate - hr_per_pa)
    hit_rate = np.clip(non_hr_hit_rate + hr_per_pa, 0.0, 1.0)  # H / PA

    # AVG = H / AB  →  (hit_rate × PA) / (ab_rate × PA)  →  hit_rate / ab_rate
    avg = np.clip(hit_rate / ab_rate, 0.100, 0.400)

    # SLG = AVG + ISO  (identity)
    slg = np.clip(avg + iso, 0.150, 0.900)

    # OBP = (H + BB + HBP) / PA  (≈ denominator = AB + BB + HBP + SF ≈ PA)
    obp = np.clip(hit_rate + bb_pct + hbp_pct, 0.150, 0.550)

    return {
        'AVG': avg,
        'SLG': slg,
        'OBP': obp,
        'FB%': fb_pct,
        'HR_per_PA': hr_per_pa,
    }


def compose_counting(
    k_pct: float,
    bb_pct: float,
    hbp_pct: float,
    iso: float,
    babip: float,
    hr_fb: float,
    gb_pct: float,
    ld_pct: float,
    *,
    pa: float = DEFAULT_PA,
    sf_rate: float = DEFAULT_SF_RATE,
    triple_share: float = DEFAULT_3B_SHARE,
) -> Dict[str, float]:
    """Derive per-season counting stats from base components.

    Returns dict with HR, 2B, 3B, 1B, H, BB, HBP, AB, SF, plus all rates.
    """
    rates = compose_rates(
        k_pct, bb_pct, hbp_pct, iso, babip, hr_fb, gb_pct, ld_pct,
        sf_rate=sf_rate,
    )
    avg = rates['AVG']
    slg = rates['SLG']
    hr_per_pa = rates['HR_per_PA']

    bb  = bb_pct * pa
    hbp = hbp_pct * pa
    sf  = sf_rate * pa
    ab  = pa - bb - hbp - sf
    ab  = np.maximum(ab, 1.0)

    hr = hr_per_pa * pa
    h  = avg * ab

    # Split extra-base hits via ISO identity:
    #   ISO × AB = 2B + 2×3B + 3×HR   →   2B + 2×3B = ISO × AB − 3×HR
    doubles_equiv = np.maximum(iso * ab - 3.0 * hr, 0.0)  # 2B + 2×3B

    #   Let xbh = 2B + 3B.  Then 2B + 2×3B = xbh + 3B = xbh(1 + ts)
    xbh_no_hr = doubles_equiv / np.maximum(1.0 + triple_share, 1.001)
    triples   = triple_share * xbh_no_hr
    doubles   = xbh_no_hr - triples
    singles   = np.maximum(h - hr - doubles - triples, 0.0)

    # Guard against negative doubles (when ISO is very low relative to HR)
    doubles  = np.maximum(doubles, 0.0)
    triples  = np.maximum(triples, 0.0)

    return {
        **rates,
        'PA':  pa,
        'AB':  ab,
        'H':   h,
        'HR':  hr,
        '2B':  doubles,
        '3B':  triples,
        '1B':  singles,
        'BB':  bb,
        'HBP': hbp,
        'SF':  sf,
        'K':   k_pct * pa,
    }


def compose_woba(counts: Dict[str, float]) -> float:
    """Compute wOBA from counting-stat dict (output of compose_counting)."""
    w = WOBA_WEIGHTS
    pa = counts['PA']
    if pa <= 0:
        return 0.0
    num = (w['wBB']  * counts['BB']  +
           w['wHBP'] * counts['HBP'] +
           w['w1B']  * counts['1B']  +
           w['w2B']  * counts['2B']  +
           w['w3B']  * counts['3B']  +
           w['wHR']  * counts['HR'])
    return float(np.clip(num / pa, 0.0, 0.600))


def compose_wrc_plus(
    woba: float,
    park_factor: float = 100.0,
    *,
    lg_woba: float = LG_WOBA,
    woba_scale: float = WOBA_SCALE,
    lg_r_pa: float = LG_RUNS_PER_PA,
    lg_wrc_pa: float = LG_WRC_PER_PA,
) -> float:
    """Compute wRC+ from wOBA and park factor (100 = neutral)."""
    wraa_per_pa = (woba - lg_woba) / woba_scale
    pf = park_factor / 100.0
    park_adj = lg_r_pa - (pf * lg_r_pa)
    wrc_plus = 100.0 * ((wraa_per_pa + lg_r_pa + park_adj) / lg_wrc_pa)
    return float(np.clip(wrc_plus, 0, 300))


# ---------------------------------------------------------------------------
# All-in-one convenience
# ---------------------------------------------------------------------------

def compose_all(
    k_pct: float,
    bb_pct: float,
    hbp_pct: float,
    iso: float,
    babip: float,
    hr_fb: float,
    gb_pct: float,
    ld_pct: float,
    *,
    pa: float = DEFAULT_PA,
    sf_rate: float = DEFAULT_SF_RATE,
    triple_share: float = DEFAULT_3B_SHARE,
    park_factor: float = 100.0,
) -> Dict[str, float]:
    """Derive every stat from the 8 base components in one call.

    Returns a dict with:
        Rate:     AVG, SLG, OBP, wOBA, wRC+, FB%
        Counting: PA, AB, H, HR, 2B, 3B, 1B, BB, HBP, SF, K
    """
    counts = compose_counting(
        k_pct, bb_pct, hbp_pct, iso, babip, hr_fb, gb_pct, ld_pct,
        pa=pa, sf_rate=sf_rate, triple_share=triple_share,
    )
    woba = compose_woba(counts)
    wrc_plus = compose_wrc_plus(woba, park_factor)

    counts['wOBA']  = woba
    counts['wRC+']  = wrc_plus
    return counts


# ---------------------------------------------------------------------------
# DataFrame-level composition
# ---------------------------------------------------------------------------

def compose_from_df(
    df: pd.DataFrame,
    *,
    pa: float = DEFAULT_PA,
    sf_rate: float = DEFAULT_SF_RATE,
    triple_share_col: Optional[str] = None,
    park_factor_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compose all derived stats for a DataFrame of projections.

    Expects columns: K%, BB%, HBP%, ISO, BABIP, HR/FB, GB%, LD%.
    Adds columns: AVG, SLG, OBP, wOBA, wRC+, FB%, HR, 2B, 3B, 1B, H, AB, K.
    Existing columns with those names are overwritten.

    Parameters
    ----------
    triple_share_col : str, optional
        Column name holding per-player triple share.  Falls back to league
        average (DEFAULT_3B_SHARE) when absent or NaN.
    park_factor_col : str, optional
        Column name with park factor (100 = neutral).  Default 100 for all.
    """
    required = ['K%', 'BB%', 'HBP%', 'ISO', 'BABIP', 'HR/FB', 'GB%', 'LD%']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    k_pct   = out['K%'].values.astype(float)
    bb_pct  = out['BB%'].values.astype(float)
    hbp_pct = out['HBP%'].values.astype(float)
    iso     = out['ISO'].values.astype(float)
    babip   = out['BABIP'].values.astype(float)
    hr_fb   = out['HR/FB'].values.astype(float)
    gb_pct  = out['GB%'].values.astype(float)
    ld_pct  = out['LD%'].values.astype(float)

    triple_share = DEFAULT_3B_SHARE
    if triple_share_col and triple_share_col in out.columns:
        triple_share = out[triple_share_col].fillna(DEFAULT_3B_SHARE).values.astype(float)

    park_factor = 100.0
    if park_factor_col and park_factor_col in out.columns:
        park_factor = out[park_factor_col].fillna(100.0).values.astype(float)

    # --- Intermediate ---
    fb_pct   = np.clip(1.0 - gb_pct - ld_pct, 0.0, 1.0)
    bip_rate = np.clip(1.0 - k_pct - bb_pct - hbp_pct, 0.01, 1.0)
    ab_rate  = np.clip(1.0 - bb_pct - hbp_pct - sf_rate, 0.01, 1.0)

    hr_per_pa = np.clip(hr_fb * fb_pct * bip_rate, 0.0, 0.12)

    non_hr_hit_rate = babip * (bip_rate - hr_per_pa)
    hit_rate = np.clip(non_hr_hit_rate + hr_per_pa, 0.0, 1.0)

    ab = pa * ab_rate
    ab = np.maximum(ab, 1.0)

    avg = np.clip(hit_rate / ab_rate, 0.100, 0.400)
    slg = np.clip(avg + iso, 0.150, 0.900)
    obp = np.clip(hit_rate + bb_pct + hbp_pct, 0.150, 0.550)

    # --- Counting ---
    hr      = hr_per_pa * pa
    h       = avg * ab
    bb      = bb_pct * pa
    hbp     = hbp_pct * pa
    k       = k_pct * pa
    sf      = sf_rate * pa

    doubles_equiv = np.maximum(iso * ab - 3.0 * hr, 0.0)
    xbh_no_hr     = doubles_equiv / np.maximum(1.0 + triple_share, 1.001)
    triples        = triple_share * xbh_no_hr
    doubles        = np.maximum(xbh_no_hr - triples, 0.0)
    triples        = np.maximum(triples, 0.0)
    singles        = np.maximum(h - hr - doubles - triples, 0.0)

    # --- wOBA ---
    w = WOBA_WEIGHTS
    woba_num = (w['wBB'] * bb + w['wHBP'] * hbp + w['w1B'] * singles +
                w['w2B'] * doubles + w['w3B'] * triples + w['wHR'] * hr)
    woba = np.clip(woba_num / pa, 0.0, 0.600)

    # --- wRC+ ---
    wraa_per_pa = (woba - LG_WOBA) / WOBA_SCALE
    pf = np.where(np.isscalar(park_factor), park_factor / 100.0,
                  np.asarray(park_factor, dtype=float) / 100.0)
    park_adj = LG_RUNS_PER_PA - pf * LG_RUNS_PER_PA
    wrc_plus = np.clip(
        100.0 * ((wraa_per_pa + LG_RUNS_PER_PA + park_adj) / LG_WRC_PER_PA),
        0, 300,
    )

    # --- Write columns ---
    out['AVG']  = avg
    out['SLG']  = slg
    out['OBP']  = obp
    out['wOBA'] = woba
    out['wRC+'] = wrc_plus
    out['FB%']  = fb_pct
    out['HR']   = hr
    out['2B']   = doubles
    out['3B']   = triples
    out['1B']   = singles
    out['H']    = h
    out['AB']   = ab
    out['K']    = k
    out['BB_count'] = bb
    out['HBP_count'] = hbp

    n = len(out)
    logger.info(
        f"Composed stats for {n} rows — "
        f"AVG={avg.mean():.3f}, OBP={obp.mean():.3f}, "
        f"SLG={slg.mean():.3f}, wOBA={woba.mean():.3f}, "
        f"HR={hr.mean():.1f}, wRC+={wrc_plus.mean():.1f}"
    )
    return out
