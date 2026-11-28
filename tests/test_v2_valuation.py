"""
Unit & Economic Invariant Tests for the V2 Monte Carlo Trade Valuation System
=============================================================================

Verifies:
1. Contract Option Monotonicity (Player Option <= Signed; in-the-money Team Option > FA;
   Non-tendered Arb/Pre-Arb downside protection vs guaranteed salary).
2. Quantile Ordering (`trade_value_p10 <= trade_value_p50 <= trade_value_p90`) across
   both synthetic cases and live generated MLB & Prospect datasets.
3. Package Consolidation Sub-Additivity (`L_gamma` norm strictly penalizes multi-piece
   positive packages while subtracting salary dumps linearly).
4. Hard MLB Debut Boundary (zero `mlbam_id` overlap between 2026 MLB players and
   2026 pre-debut `has_mlb=False` prospects; zero `'FA'` players with non-null `trade_value`).
5. Spotrac Multi-Span Dollar Parsing (`"$11,885,591$114,409" -> 12_000_000.0`).
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
AUTO_TRAIN_DIR = ROOT_DIR / "auto_train_models"
if str(AUTO_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_TRAIN_DIR))

from value_determination.config import Config
from value_determination.monte_carlo_valuation import (
    analyze_contract_options,
    calculate_trade_values,
)
from value_determination.salary_cleaner import _parse_dollar_col


def _make_timeline(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = analyze_contract_options(df, current_year=2026)
    out = calculate_trade_values(out, current_year=2026, apply_ros_fraction=False)
    return out[out["Year"] == 2026].set_index("IDfg")


def test_player_option_never_increases_club_value():
    """Giving a player a Player Option / Opt-Out cannot increase club trade_value vs Signed."""
    rows = [
        # Player 1: 2-year guaranteed Signed deal ($22M/yr, 3.5 WAR/yr)
        {"IDfg": 1, "Name": "SignedStar", "Team": "NYY", "Year": 2026, "Age": 27, "player_role": "Batter",
         "Normalized_Status": "Signed", "WAR": 3.5, "PA": 600, "Payroll": 22_000_000,
         "salary_cost": 22_000_000, "production_value": 38_000_000, "annual_surplus": 16_000_000},
        {"IDfg": 1, "Name": "SignedStar", "Team": "NYY", "Year": 2027, "Age": 28, "player_role": "Batter",
         "Normalized_Status": "Signed", "WAR": 3.5, "PA": 600, "Payroll": 22_000_000,
         "salary_cost": 22_000_000, "production_value": 39_500_000, "annual_surplus": 17_500_000},
        # Player 2: Identical, except 2027 is a Player Option
        {"IDfg": 2, "Name": "OptOutStar", "Team": "NYY", "Year": 2026, "Age": 27, "player_role": "Batter",
         "Normalized_Status": "Signed", "WAR": 3.5, "PA": 600, "Payroll": 22_000_000,
         "salary_cost": 22_000_000, "production_value": 38_000_000, "annual_surplus": 16_000_000},
        {"IDfg": 2, "Name": "OptOutStar", "Team": "NYY", "Year": 2027, "Age": 28, "player_role": "Batter",
         "Normalized_Status": "Player Option", "WAR": 3.5, "PA": 600, "Payroll": 22_000_000,
         "salary_cost": 22_000_000, "production_value": 39_500_000, "annual_surplus": 17_500_000},
    ]
    res = _make_timeline(rows)
    assert res.loc[2, "trade_value"] < res.loc[1, "trade_value"]


def test_team_option_adds_positive_optionality():
    """Adding an affordable Team Option strictly raises club trade_value vs immediate FA."""
    rows = [
        # Player 1: 1-year Signed then Free Agent
        {"IDfg": 10, "Name": "OneYr", "Team": "LAD", "Year": 2026, "Age": 28, "player_role": "SP",
         "Normalized_Status": "Signed", "WAR": 3.0, "IP": 175, "Payroll": 18_000_000,
         "salary_cost": 18_000_000, "production_value": 31_000_000, "annual_surplus": 13_000_000},
        {"IDfg": 10, "Name": "OneYr", "Team": "LAD", "Year": 2027, "Age": 29, "player_role": "SP",
         "Normalized_Status": "Free Agent", "WAR": 2.8, "IP": 170, "Payroll": np.nan,
         "salary_cost": np.nan, "production_value": 29_000_000, "annual_surplus": np.nan},
        # Player 2: Identical, plus $18M Team Option in 2027
        {"IDfg": 11, "Name": "WithClubOpt", "Team": "LAD", "Year": 2026, "Age": 28, "player_role": "SP",
         "Normalized_Status": "Signed", "WAR": 3.0, "IP": 175, "Payroll": 18_000_000,
         "salary_cost": 18_000_000, "production_value": 31_000_000, "annual_surplus": 13_000_000},
        {"IDfg": 11, "Name": "WithClubOpt", "Team": "LAD", "Year": 2027, "Age": 29, "player_role": "SP",
         "Normalized_Status": "Team Option", "WAR": 2.8, "IP": 170, "Payroll": 18_000_000,
         "salary_cost": 18_000_000, "production_value": 29_000_000, "annual_surplus": 11_000_000},
    ]
    res = _make_timeline(rows)
    assert res.loc[11, "trade_value"] > res.loc[10, "trade_value"]


def test_nontender_protects_arb_downside_vs_guaranteed_deal():
    """Arb non-tender optionality strictly bounds P10 downside vs an equal-salary guaranteed deal."""
    rows = []
    for pid, status in ((20, "Signed"), (21, "Arb-3")):
        for yr in (2026, 2027, 2028):
            rows.append({
                "IDfg": pid, "Name": f"P{pid}", "Team": "CHC", "Year": yr, "Age": 29 + (yr - 2026),
                "player_role": "Batter", "Normalized_Status": status if yr > 2026 else "Signed",
                "WAR": 0.6, "PA": 450, "Payroll": 12_000_000 if status == "Signed" or yr == 2026 else np.nan,
                "salary_cost": 12_000_000, "production_value": 5_500_000, "annual_surplus": -6_500_000,
            })
    res = _make_timeline(rows)
    assert res.loc[21, "trade_value"] > res.loc[20, "trade_value"]
    assert res.loc[21, "trade_value_p10"] > res.loc[20, "trade_value_p10"]


def test_package_consolidation_subadditivity():
    """Multi-asset positive packages consolidate strictly below their raw linear sum."""
    c1, raw1, pen1 = Config.Market.consolidate_package([40_000_000.0])
    assert math.isclose(c1, 40_000_000.0, rel_tol=1e-6)
    assert math.isclose(pen1, 0.0, abs_tol=1e-6)

    c2, raw2, pen2 = Config.Market.consolidate_package([20_000_000.0, 15_000_000.0, 10_000_000.0])
    assert math.isclose(raw2, 45_000_000.0, rel_tol=1e-6)
    assert c2 < raw2
    assert pen2 < 0.0

    # Negative salary dump subtracts linearly
    c3, raw3, _ = Config.Market.consolidate_package([30_000_000.0, -12_000_000.0])
    assert math.isclose(c3, 18_000_000.0, rel_tol=1e-6)
    assert math.isclose(raw3, 18_000_000.0, rel_tol=1e-6)


def test_spotrac_multispan_dollar_parser():
    """Spotrac cells with multiple concatenated $ spans must be summed, never digit-concatenated."""
    s = pd.Series(["$11,885,591$114,409", "$15,000,000$2,000,000", "$25,000,000", "-", None])
    parsed = _parse_dollar_col(s)
    assert math.isclose(parsed.iloc[0], 12_000_000.0, rel_tol=1e-6)
    assert math.isclose(parsed.iloc[1], 17_000_000.0, rel_tol=1e-6)
    assert math.isclose(parsed.iloc[2], 25_000_000.0, rel_tol=1e-6)
    assert pd.isna(parsed.iloc[3])
    assert pd.isna(parsed.iloc[4])


def test_generated_datasets_invariants():
    """Verify quantile ordering, zero FA trade_value, and zero 2026 MLB/Prospect overlap."""
    pv_path = Config.Paths.OUTPUT_DIR / "player_values_complete.csv"
    pr_path = Config.Paths.PROSPECT_VALUES_FILE
    assert pv_path.exists(), f"Missing {pv_path}"
    assert pr_path.exists(), f"Missing {pr_path}"

    pv = pd.read_csv(pv_path, low_memory=False)
    cur_pv = pv[pv["year"] == 2026].copy()
    assert not cur_pv.empty

    # 1. Zero Free Agents with non-null trade_value
    fa_rows = cur_pv[cur_pv["team"] == "FA"]
    assert fa_rows["trade_value"].isna().all(), "Free Agents must have NaN trade_value"

    # 2. Quantile ordering on all valued MLB players
    valued_mlb = cur_pv.dropna(subset=["trade_value", "trade_value_p10", "trade_value_p50", "trade_value_p90"])
    assert len(valued_mlb) > 500
    assert (valued_mlb["trade_value_p10"] <= valued_mlb["trade_value_p50"] + 1.0).all()
    assert (valued_mlb["trade_value_p50"] <= valued_mlb["trade_value_p90"] + 1.0).all()

    # 3. Top MLB trade_value is economically anchored (well below V1's $738M)
    max_tv = valued_mlb["trade_value"].max()
    assert 130_000_000 <= max_tv <= 280_000_000, f"Unexpected top MLB trade_value: ${max_tv/1e6:.1f}M"

    # 4. Quantile ordering & zero overlap on 2026 pre-debut prospects
    pr = pd.read_csv(pr_path, low_memory=False)
    cur_pr = pr[pr["year"] == 2026].copy()
    assert not cur_pr.empty
    assert (cur_pr["trade_value_p10"] <= cur_pr["trade_value_p50"] + 1.0).all()
    assert (cur_pr["trade_value_p50"] <= cur_pr["trade_value_p90"] + 1.0).all()

    pre_debut = cur_pr[cur_pr["has_mlb"] == False]  # noqa: E712
    mlb_ids = set(cur_pv["mlbam_id"].dropna().astype(int))
    pre_debut_ids = set(pre_debut["mlbam_id"].dropna().astype(int))
    overlap = mlb_ids & pre_debut_ids
    assert overlap == set(), f"Found MLB/Prospect overlap IDs: {overlap}"


def test_martingale_no_vega_bias_on_low_or_negative_war():
    """Martingale + WORP must never manufacture millions of trade value out of 0/negative WAR."""
    rows = []
    for yr in range(2026, 2031):
        # Player 100: -0.3 WAR/yr Pre-Arb 27yo unpedigreed depth
        rows.append({
            "IDfg": 999100, "Name": "NegWarDepth", "Team": "SEA", "Year": yr, "Age": 27 + (yr - 2026),
            "player_role": "Batter", "Normalized_Status": "Pre-Arb" if yr < 2029 else "Arb-1",
            "WAR": -0.3, "PA": 180, "Payroll": np.nan,
            "salary_cost": 780_000, "production_value": 0.0, "annual_surplus": -780_000,
        })
        # Player 101: +0.15 WAR/yr mop-up Pre-Arb 26yo unpedigreed RP (below 0.35 WORP hurdle)
        rows.append({
            "IDfg": 999101, "Name": "MopUpRP", "Team": "SEA", "Year": yr, "Age": 26 + (yr - 2026),
            "player_role": "RP", "Normalized_Status": "Pre-Arb" if yr < 2029 else "Arb-1",
            "WAR": 0.15, "IP": 30, "Payroll": np.nan,
            "salary_cost": 780_000, "production_value": 330_000, "annual_surplus": -450_000,
        })
    res = _make_timeline(rows)
    assert math.isclose(res.loc[999100, "trade_value"], 0.0, abs_tol=1.0), (
        f"Negative-WAR Pre-Arb player had non-zero trade_value: {res.loc[999100, 'trade_value']}"
    )
    assert 0.0 <= res.loc[999101, "trade_value"] <= 1_500_000.0, (
        f"0.15-WAR mop-up RP had inflated trade_value: {res.loc[999101, 'trade_value']}"
    )


def test_v3_low_end_two_way_and_mariners_invariants():
    """Verify V3 low-end discipline, zero fake mop-up two-way players, and Mariners ordering."""
    from core.id_crosswalk import load_recent_prospect_info

    pv_path = Config.Paths.OUTPUT_DIR / "player_values_complete.csv"
    pv = pd.read_csv(pv_path, low_memory=False)
    cur = pv[pv["year"] == 2026].copy()

    # 1. Zero mop-up position-player reliever hybrids (RP/C, RP/1B, RP/RF, etc.)
    pos_series = cur["position"].fillna("").astype(str)
    fake_tw = cur[
        pos_series.str.contains("RP/", regex=False)
        | pos_series.str.contains("/RP", regex=False)
    ]
    assert fake_tw.empty, (
        f"Found fake mop-up two-way RP hybrids: {fake_tw[['player_name', 'team', 'position']].to_dict('records')}"
    )

    # 2. Low-end discipline across all 30 teams (excluding pedigreed recent FV>=45/Top-100 rookies)
    recent_pros = load_recent_prospect_info(as_of_year=2026, recency_years=Config.Prospects.RECENCY_YEARS)
    pedigreed_fg = {
        fg for fg, info in recent_pros.items()
        if float(info.get("fv") or 0.0) >= 45.0 or pd.notna(info.get("top_100"))
    }
    valued = cur.dropna(subset=["trade_value"]).copy()
    unped_low = valued[
        (~valued["fg_id"].isin(pedigreed_fg))
        & (valued["avg_war"] <= 0.45)
        & (valued["years_control"] >= 1.0)
    ]
    over_3_5m = unped_low[unped_low["trade_value"] > 3_500_000]
    assert over_3_5m.empty, (
        f"Unpedigreed <=0.45 avg_war players valued >$3.5M:\n"
        f"{over_3_5m[['player_name', 'team', 'position', 'avg_war', 'control_war', 'trade_value']].to_string()}"
    )

    neg_prearb = valued[
        (~valued["fg_id"].isin(pedigreed_fg))
        & (valued["control_war"] <= 0.0)
        & (valued["status"] == "Pre-Arb")
    ]
    over_min_sal = neg_prearb[neg_prearb["trade_value"] > 780_000]
    assert over_min_sal.empty, (
        f"Non-positive control_war Pre-Arb players valued >$0.78M (1x MLB min):\n"
        f"{over_min_sal[['player_name', 'team', 'control_war', 'trade_value']].to_string()}"
    )

    # 3. Seattle Mariners ordering invariants
    sea = valued[valued["team"] == "SEA"].set_index("player_name")
    munoz_tv = float(sea.loc[sea.index.str.contains("Muñoz|Munoz", case=False), "trade_value"].iloc[0])
    wilcox_rows = sea.loc[sea.index.str.contains("Cole Wilcox", case=False), "trade_value"]
    wilcox_tv = float(wilcox_rows.iloc[0]) if not wilcox_rows.empty else 0.0
    assert munoz_tv > wilcox_tv + 6_000_000, f"Expected Muñoz (${munoz_tv/1e6:.1f}M) >> Wilcox (${wilcox_tv/1e6:.1f}M)"

    bliss_rows = sea.loc[sea.index.str.contains("Ryan Bliss", case=False), "trade_value"]
    bliss_tv = float(bliss_rows.iloc[0]) if not bliss_rows.empty else 0.0
    kirby_tv = float(sea.loc[sea.index.str.contains("George Kirby", case=False), "trade_value"].iloc[0])
    gilbert_tv = float(sea.loc[sea.index.str.contains("Logan Gilbert", case=False), "trade_value"].iloc[0])
    assert kirby_tv > bliss_tv + 5_000_000, f"Expected Kirby (${kirby_tv/1e6:.1f}M) >> Bliss (${bliss_tv/1e6:.1f}M)"
    assert gilbert_tv > bliss_tv, f"Expected Gilbert (${gilbert_tv/1e6:.1f}M) > Bliss (${bliss_tv/1e6:.1f}M)"

    woo_tv = float(sea.loc[sea.index.str.contains("Bryan Woo", case=False), "trade_value"].iloc[0])
    evans_rows = sea.loc[sea.index.str.contains("Logan Evans", case=False), "trade_value"]
    evans_tv = float(evans_rows.iloc[0]) if not evans_rows.empty else 0.0
    assert woo_tv > evans_tv + 15_000_000, f"Expected Woo (${woo_tv/1e6:.1f}M) >> Evans (${evans_tv/1e6:.1f}M)"

    pereda_rows = cur[cur["player_name"].str.contains("Jhonny Pereda", case=False, na=False)]
    if not pereda_rows.empty:
        p_tv = float(pereda_rows["trade_value"].fillna(0.0).iloc[0])
        p_pos = str(pereda_rows["position"].iloc[0])
        assert "RP" not in p_pos, f"Jhonny Pereda still classified as pitcher: {p_pos}"
        assert p_tv <= 1_500_000, f"Jhonny Pereda trade_value still inflated: ${p_tv/1e6:.2f}M"


class TestV2ValuationSuite(unittest.TestCase):
    def test_01_player_option_never_increases_club_value(self):
        test_player_option_never_increases_club_value()

    def test_02_team_option_adds_positive_optionality(self):
        test_team_option_adds_positive_optionality()

    def test_03_nontender_protects_arb_downside_vs_guaranteed_deal(self):
        test_nontender_protects_arb_downside_vs_guaranteed_deal()

    def test_04_package_consolidation_subadditivity(self):
        test_package_consolidation_subadditivity()

    def test_05_spotrac_multispan_dollar_parser(self):
        test_spotrac_multispan_dollar_parser()

    def test_06_generated_datasets_invariants(self):
        test_generated_datasets_invariants()

    def test_07_martingale_no_vega_bias_on_low_or_negative_war(self):
        test_martingale_no_vega_bias_on_low_or_negative_war()

    def test_08_v3_low_end_two_way_and_mariners_invariants(self):
        test_v3_low_end_two_way_and_mariners_invariants()


if __name__ == "__main__":
    unittest.main(verbosity=2)
