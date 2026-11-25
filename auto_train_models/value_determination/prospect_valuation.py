"""
Unified Pre-Debut Prospect Valuation Engine
===========================================

Single source of truth for valuing pre-debut prospects (`0 MLB games` as of
the valuation year) across all years (`2014–2026`), replacing the four
conflicting FV lookup tables and noisy MiLB box-score projection solvers.

Formula:
    trade_value = V_FV(grade) * M_rank(top_100, org_rank) * M_role(position)
                  * (1 + r)^(-delta_eta) * M_perf(milb_age_adjusted)

Uncertainty Quantiles (`P10`, `P50`, `P90`):
    Modeled via a mean-preserving zero-inflated lognormal distribution
    parameterized by bust probability `p_bust(FV, eta, role)` and log-scale
    dispersion `sigma_ln(eta, role)`, capturing both minor-league attrition
    (`P10 = $0` for distant/low-FV prospects) and right-skewed star upside (`P90`).

Outputs:
    `data/generated/MiLB/prospect_values.csv`
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.id_crosswalk import get_crosswalk, normalize_team_code
from .config import CURRENT_YEAR, Config, logger

_PITCHER_POSITIONS = {"P", "RHP", "LHP", "SP", "RP", "SRHP", "SLHP"}
_RELIEVER_POSITIONS = {"RP", "RHP-R", "LHP-R"}
_TWO_WAY_TOKENS = ("/", "TWP")

# Benchmark age by MiLB level for age-relative performance modifier
_LEVEL_BENCHMARK_AGE = {
    "AAA": 24.5,
    "AA": 22.5,
    "A+": 21.2,
    "A": 20.0,
    "A-": 19.5,
    "A (SHORT)": 19.5,
    "ROK": 18.5,
    "CPX": 18.5,
    "DSL": 17.5,
}


def extract_mlbam_from_url(url: Any) -> Optional[int]:
    """Extract integer `mlbam_id` from a prospect URL ending in `-<digits>`."""
    if url is None or pd.isna(url):
        return None
    m = re.search(r"-(\d{5,7})$", str(url).strip())
    return int(m.group(1)) if m else None


def parse_fv_numeric(fv_raw: Any) -> Optional[float]:
    """Parse FanGraphs/Pipeline FV grade (supporting `55+` -> `57.5`)."""
    if fv_raw is None or pd.isna(fv_raw):
        return None
    s = str(fv_raw).strip()
    if not s:
        return None
    plus = 2.5 if s.endswith("+") else 0.0
    try:
        val = float(s.replace("+", "")) + plus
    except ValueError:
        return None
    return val if val > 0 else None


def classify_prospect_role(
    position: Any,
    fv_num: float = 45.0,
    grade_control: Optional[float] = None,
) -> str:
    """Classify a prospect's role into `'Batter'`, `'SP'`, or `'RP'`."""
    pos = str(position or "").upper().strip()
    if any(tok in pos for tok in _TWO_WAY_TOKENS):
        return "Batter"
    if pos in _RELIEVER_POSITIONS:
        return "RP"
    if pos in _PITCHER_POSITIONS:
        if grade_control is not None and not pd.isna(grade_control) and float(grade_control) < 40 and fv_num <= 45:
            return "RP"
        return "SP"
    return "Batter"


def _interpolate_fv_base_value(fv_num: float) -> float:
    """Log-linear interpolation across `Config.Prospects.FV_BASE_VALUES`."""
    Config.Market.load_calibration()
    table = Config.Prospects.FV_BASE_VALUES
    tiers = sorted(table.keys())
    clamped = float(np.clip(fv_num, tiers[0], tiers[-1]))

    for i in range(len(tiers) - 1):
        lo, hi = tiers[i], tiers[i + 1]
        if lo <= clamped <= hi:
            if hi == lo:
                return float(table[lo]) * Config.Prospects.PROSPECT_SCALE
            frac = (clamped - lo) / (hi - lo)
            v_lo, v_hi = math.log(table[lo]), math.log(table[hi])
            return float(math.exp(v_lo + frac * (v_hi - v_lo))) * Config.Prospects.PROSPECT_SCALE

    return float(table[tiers[-1]]) * Config.Prospects.PROSPECT_SCALE


def _rank_multiplier(top_100: Optional[float], org_rank: Optional[float] = None) -> float:
    """Smooth Top-100 bonus `[1.00, 1.35]`, with gentle taper for deep org ranks."""
    if top_100 is not None and not pd.isna(top_100) and 1 <= float(top_100) <= 100:
        r = float(top_100)
        max_b = Config.Prospects.TOP100_MAX_BONUS
        min_b = Config.Prospects.TOP100_MIN_BONUS
        return float(min_b + (max_b - min_b) * (((100.0 - r) / 99.0) ** 1.20))

    if org_rank is not None and not pd.isna(org_rank):
        orank = float(org_rank)
        if orank > 15:
            # Gradual taper down to 0.85x for #30 org depth prospects with same nominal FV
            return float(max(0.85, 1.0 - 0.01 * (orank - 15.0)))
    return 1.0


def _zero_inflated_lognormal_quantiles(
    expected_value: float,
    fv_num: float,
    delta_eta: float,
    role: str,
) -> tuple[float, float, float]:
    """
    Compute analytical `(P10, P50, P90)` of a zero-inflated lognormal prospect
    outcome distribution whose exact expectation equals *expected_value*.
    """
    if expected_value <= 0:
        return 0.0, 0.0, 0.0

    # Bust probability decreases with higher FV and closer MLB proximity, higher for pitchers
    fv_term = np.clip((65.0 - fv_num) / 30.0, -0.10, 1.0)
    base_bust = 0.12 + 0.42 * fv_term + 0.04 * min(delta_eta, 4.0)
    if role in ("SP", "RP"):
        base_bust += 0.08
    p_bust = float(np.clip(base_bust, 0.05, 0.72))

    sigma_ln = 0.62 + 0.07 * min(delta_eta, 4.0) + (0.10 if role in ("SP", "RP") else 0.0)
    # Mean of positive branch must be expected_value / (1 - p_bust)
    cond_mean = expected_value / (1.0 - p_bust)
    mu_ln = math.log(cond_mean) - 0.5 * (sigma_ln ** 2)

    def _quantile(q: float) -> float:
        if q <= p_bust:
            return 0.0
        q_cond = (q - p_bust) / (1.0 - p_bust)
        q_cond = min(max(q_cond, 1e-6), 1.0 - 1e-6)
        return float(math.exp(mu_ln + sigma_ln * float(norm.ppf(q_cond))))

    return _quantile(0.10), _quantile(0.50), _quantile(0.90)


def compute_single_prospect_valuation(
    fv: Any,
    top_100: Optional[float] = None,
    org_rank: Optional[float] = None,
    position: str = "OF",
    delta_eta_years: float = 1.0,
    valuation_year: int = CURRENT_YEAR,
    perf_modifier: float = 1.0,
    grade_control: Optional[float] = None,
) -> dict[str, Any]:
    """
    Compute `trade_value`, `trade_value_p10`, `trade_value_p50`, and `trade_value_p90`
    for a single prospect-year observation.
    """
    fv_num = parse_fv_numeric(fv)
    if fv_num is None:
        return {
            "trade_value": 0.0,
            "trade_value_p10": 0.0,
            "trade_value_p50": 0.0,
            "trade_value_p90": 0.0,
            "player_role": classify_prospect_role(position),
        }

    Config.Market.load_calibration()
    role = classify_prospect_role(position, fv_num=fv_num, grade_control=grade_control)

    base_val = _interpolate_fv_base_value(fv_num)
    m_rank = _rank_multiplier(top_100, org_rank=org_rank)

    if any(tok in str(position or "").upper() for tok in _TWO_WAY_TOKENS):
        m_role = 0.95
    else:
        m_role = Config.Prospects.ROLE_MULTIPLIER.get(role, 1.0)

    r_disc = Config.Market.DISCOUNT_RATE
    delta_t = float(np.clip(delta_eta_years, 0.0, 5.0))
    m_eta = (1.0 + r_disc) ** (-delta_t)

    lo_p, hi_p = Config.Prospects.PERF_MODIFIER_BOUNDS
    m_perf = float(np.clip(perf_modifier, lo_p, hi_p))

    inflation = (1.0 + Config.Market.INFLATION_RATE) ** (int(valuation_year) - Config.Prospects.BASE_YEAR)
    expected_val = base_val * m_rank * m_role * m_eta * m_perf * inflation

    p10, p50, p90 = _zero_inflated_lognormal_quantiles(
        expected_value=expected_val,
        fv_num=fv_num,
        delta_eta=delta_t,
        role=role,
    )
    return {
        "trade_value": round(expected_val),
        "trade_value_p10": round(p10),
        "trade_value_p50": round(p50),
        "trade_value_p90": round(p90),
        "player_role": role,
    }


def _load_first_mlb_season_by_fg() -> dict[int, int]:
    """Return `{fg_id_int: first_mlb_season}` across historical + current-season MLB files."""
    first_season: dict[int, int] = {}
    candidates = [
        Config.Paths.HISTORIC_BATTING_FILE_CLASSIC,
        Config.Paths.HISTORIC_PITCHING_FILE_CLASSIC,
        Config.Paths.DATA_DIR / "current_season" / f"mlb_batting_data_{CURRENT_YEAR}_{CURRENT_YEAR}.csv",
        Config.Paths.DATA_DIR / "current_season" / f"mlb_pitching_data_{CURRENT_YEAR}_{CURRENT_YEAR}.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["IDfg", "Season", "G"], low_memory=False)
            df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")
            df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
            df["G"] = pd.to_numeric(df["G"], errors="coerce").fillna(0)
            df = df[(df["IDfg"].notna()) & (df["Season"].notna()) & (df["G"] >= 1)]
            for fg_id, min_yr in df.groupby("IDfg")["Season"].min().items():
                fg = int(fg_id)
                yr = int(min_yr)
                if fg not in first_season or yr < first_season[fg]:
                    first_season[fg] = yr
        except Exception as exc:
            logger.warning("Could not scan MLB debut years from %s: %s", path.name, exc)
    return first_season


def _load_raw_crosswalk_strings() -> dict[int, str]:
    """
    Load `mlbam_id -> fg_id_str` preserving `'sa######'` MiLB FanGraphs IDs
    for pre-debut minor leaguers alongside integer FanGraphs IDs.
    """
    mapping: dict[int, str] = {}
    # Seed with integer crosswalk first
    mlbam_to_int, _ = get_crosswalk()
    for m_id, fg_int in mlbam_to_int.items():
        mapping[int(m_id)] = str(int(fg_int))

    cw_path = Config.Paths.CROSSWALK_FILE
    if cw_path.exists():
        cw = pd.read_csv(cw_path, encoding="utf-8-sig", low_memory=False)
        cw.columns = cw.columns.str.strip()
        for _, r in cw.iterrows():
            m_val = pd.to_numeric(r.get("mlbam_id"), errors="coerce")
            fg_raw = str(r.get("fg_id", "")).strip()
            if pd.isna(m_val) or not fg_raw or fg_raw.lower() in ("nan", "none"):
                continue
            m_int = int(m_val)
            if fg_raw.endswith(".0"):
                fg_raw = fg_raw[:-2]
            if m_int not in mapping:
                mapping[m_int] = fg_raw
    return mapping


def _load_milb_performance_lookup() -> tuple[dict[tuple[str, int], tuple[float, str]], dict[tuple[str, int], str]]:
    """
    Build MiLB age-adjusted performance modifiers `[0.85, 1.15]` and highest
    MiLB level per `(fg_id_str, season)` from `MiLB_Hitters.csv` and `MiLB_Pitchers.csv`.
    """
    perf_map: dict[tuple[str, int], tuple[float, str]] = {}
    level_rank = {"DSL": 1, "ROK": 2, "CPX": 2, "A-": 3, "A (SHORT)": 3, "A": 4, "A+": 5, "AA": 6, "AAA": 7}

    hit_path = Config.Paths.DATA_DIR / "MiLB" / "MiLB_Hitters.csv"
    if hit_path.exists():
        try:
            hdf = pd.read_csv(hit_path, usecols=["PlayerID", "Season", "Level", "Age", "PA", "wRC+"], low_memory=False)
            hdf["PA"] = pd.to_numeric(hdf["PA"], errors="coerce").fillna(0.0)
            hdf["wRC+"] = pd.to_numeric(hdf["wRC+"], errors="coerce")
            hdf["Age"] = pd.to_numeric(hdf["Age"], errors="coerce")
            hdf["Season"] = pd.to_numeric(hdf["Season"], errors="coerce")
            hdf = hdf.dropna(subset=["PlayerID", "Season", "wRC+"])
            hdf = hdf[hdf["PA"] >= 50].copy()
            hdf["pid"] = hdf["PlayerID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

            for (pid, season), grp in hdf.groupby(["pid", "Season"]):
                tot_pa = grp["PA"].sum()
                if tot_pa < 60:
                    continue
                wrc = float((grp["wRC+"] * grp["PA"]).sum() / tot_pa)
                top_row = grp.loc[grp["PA"].idxmax()]
                lvl = str(top_row.get("Level") or "A").upper().strip()
                age = float(top_row["Age"]) if pd.notna(top_row["Age"]) else _LEVEL_BENCHMARK_AGE.get(lvl, 21.0)
                bench_age = _LEVEL_BENCHMARK_AGE.get(lvl, 21.0)
                # Being younger than level benchmark adds bonus; older subtracts
                age_bonus = (bench_age - age) * 0.025
                perf_raw = 1.0 + ((wrc - 105.0) / 400.0) + age_bonus
                sample_wt = min(1.0, tot_pa / 300.0)
                mod = float(np.clip(1.0 + sample_wt * (perf_raw - 1.0), 0.85, 1.15))
                perf_map[(str(pid), int(season))] = (mod, lvl)
        except Exception as exc:
            logger.warning("Could not build MiLB hitter performance modifiers: %s", exc)

    pit_path = Config.Paths.DATA_DIR / "MiLB" / "MiLB_Pitchers.csv"
    if pit_path.exists():
        try:
            pdf = pd.read_csv(pit_path, usecols=["PlayerID", "Season", "Level", "Age", "IP", "xFIP"], low_memory=False)
            pdf["IP"] = pd.to_numeric(pdf["IP"], errors="coerce").fillna(0.0)
            pdf["xFIP"] = pd.to_numeric(pdf["xFIP"], errors="coerce")
            pdf["Age"] = pd.to_numeric(pdf["Age"], errors="coerce")
            pdf["Season"] = pd.to_numeric(pdf["Season"], errors="coerce")
            pdf = pdf.dropna(subset=["PlayerID", "Season", "xFIP"])
            pdf = pdf[pdf["IP"] >= 20].copy()
            pdf["pid"] = pdf["PlayerID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

            for (pid, season), grp in pdf.groupby(["pid", "Season"]):
                if (str(pid), int(season)) in perf_map:
                    continue
                tot_ip = grp["IP"].sum()
                if tot_ip < 25:
                    continue
                xfip = float((grp["xFIP"] * grp["IP"]).sum() / tot_ip)
                top_row = grp.loc[grp["IP"].idxmax()]
                lvl = str(top_row.get("Level") or "A").upper().strip()
                age = float(top_row["Age"]) if pd.notna(top_row["Age"]) else _LEVEL_BENCHMARK_AGE.get(lvl, 21.5)
                bench_age = _LEVEL_BENCHMARK_AGE.get(lvl, 21.5)
                age_bonus = (bench_age - age) * 0.02
                perf_raw = 1.0 + ((4.20 - xfip) * 0.06) + age_bonus
                sample_wt = min(1.0, tot_ip / 90.0)
                mod = float(np.clip(1.0 + sample_wt * (perf_raw - 1.0), 0.85, 1.15))
                perf_map[(str(pid), int(season))] = (mod, lvl)
        except Exception as exc:
            logger.warning("Could not build MiLB pitcher performance modifiers: %s", exc)

    highest_level: dict[tuple[str, int], str] = {k: v[1] for k, v in perf_map.items()}
    return perf_map, highest_level


def generate_prospect_values(
    prospect_csv: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate `prospect_values.csv` across all prospect rankings (`2014–2026`).

    Computes true season-specific `age` from `dob` (fixing archive scrape-time
    ages), resolves `fg_id` & `has_mlb`, and outputs expected `trade_value` +
    `trade_value_p10 / p50 / p90`.
    """
    src_path = prospect_csv or Config.Paths.PROSPECT_FILE
    out_path = output_csv or Config.Paths.PROSPECT_VALUES_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_path, low_memory=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # Compute accurate season-specific age from dob (MLB Pipeline archive pages
    # display current age at scrape time rather than age during the ranking year!)
    dob_ts = pd.to_datetime(df.get("dob"), format="mixed", errors="coerce")
    ref_ts = pd.to_datetime(df["year"].astype(str) + "-07-01")
    dob_age = ((ref_ts - dob_ts).dt.days / 365.25).round(1)
    raw_age = pd.to_numeric(df.get("age"), errors="coerce")
    df["age"] = dob_age.where(dob_age.notna() & (dob_age.between(15.0, 35.0)), raw_age)

    df["mlbam_id"] = df["prospect_url"].map(extract_mlbam_from_url)
    mlbam_to_fg_str = _load_raw_crosswalk_strings()
    first_mlb_by_fg = _load_first_mlb_season_by_fg()
    perf_map, milb_level_map = _load_milb_performance_lookup()

    df["org"] = df["team_slug"].map(normalize_team_code)
    df["org_rank"] = pd.to_numeric(df.get("rank"), errors="coerce")
    df["top_100"] = pd.to_numeric(df.get("top_100"), errors="coerce")
    df["eta_num"] = pd.to_numeric(df.get("eta"), errors="coerce")

    rows = []
    for _, row in df.iterrows():
        yr = int(row["year"])
        mlbam = row["mlbam_id"]
        mlbam_int = int(mlbam) if pd.notna(mlbam) else None
        fg_str = mlbam_to_fg_str.get(mlbam_int) if mlbam_int is not None else None

        # Check MLB debut status as of `yr` (and overall through CURRENT_YEAR for 2026)
        has_mlb = False
        if fg_str is not None and fg_str.isdigit():
            first_yr = first_mlb_by_fg.get(int(fg_str))
            if first_yr is not None and first_yr <= yr:
                has_mlb = True

        # Determine MiLB performance modifier & historical level from prior season (yr - 1)
        perf_mod = 1.0
        hist_lvl = None
        if fg_str is not None:
            for check_yr in (yr - 1, yr):
                if (fg_str, check_yr) in perf_map:
                    perf_mod, hist_lvl = perf_map[(fg_str, check_yr)]
                    break

        raw_lvl = str(row.get("level") or "").upper().strip()
        # For historical archive years (< 2025), raw_lvl often shows current 'MLB'
        effective_lvl = hist_lvl if (yr < 2025 and hist_lvl) else (raw_lvl or hist_lvl or "")

        # Estimate delta_eta (years to MLB arrival)
        eta_num = row.get("eta_num")
        age_val = float(row["age"]) if pd.notna(row["age"]) else 21.0
        if pd.notna(eta_num) and float(eta_num) >= yr:
            delta_eta = float(np.clip(float(eta_num) - yr, 0.0, 5.0))
        elif effective_lvl in Config.Prospects.LEVEL_ETA_YEARS and not (yr < 2025 and effective_lvl == "MLB"):
            delta_eta = Config.Prospects.LEVEL_ETA_YEARS[effective_lvl]
        else:
            # Smooth age-based fallback when historical archive level/ETA is absent
            if age_val <= 18.5:
                delta_eta = 3.5
            elif age_val <= 20.5:
                delta_eta = 2.2
            elif age_val <= 22.0:
                delta_eta = 1.2
            else:
                delta_eta = 0.5

        val_dict = compute_single_prospect_valuation(
            fv=row.get("grade_overall"),
            top_100=row.get("top_100"),
            org_rank=row.get("org_rank"),
            position=str(row.get("position") or "OF"),
            delta_eta_years=delta_eta,
            valuation_year=yr,
            perf_modifier=perf_mod,
            grade_control=row.get("grade_control"),
        )

        fv_num = parse_fv_numeric(row.get("grade_overall"))
        fv_display = str(int(round(fv_num / 5.0) * 5)) if fv_num is not None else None

        def _tool_str(col_name: str) -> Optional[str]:
            v = row.get(col_name)
            return str(int(float(v))) if pd.notna(v) else None

        rows.append({
            "mlbam_id": mlbam_int,
            "fg_id": fg_str,
            "player_name": row.get("name", ""),
            "org": row["org"],
            "year": yr,
            "age": row["age"],
            "position": row.get("position"),
            "player_role": val_dict["player_role"],
            "level": effective_lvl if effective_lvl else None,
            "eta": int(eta_num) if pd.notna(eta_num) else int(round(yr + delta_eta)),
            "org_rank": int(row["org_rank"]) if pd.notna(row["org_rank"]) else None,
            "top_100": int(row["top_100"]) if pd.notna(row["top_100"]) else None,
            "fv": fv_display,
            "fv_numeric": fv_num,
            "trade_value": val_dict["trade_value"],
            "trade_value_p10": val_dict["trade_value_p10"],
            "trade_value_p50": val_dict["trade_value_p50"],
            "trade_value_p90": val_dict["trade_value_p90"],
            "perf_modifier": round(perf_mod, 3),
            "has_mlb": has_mlb,
            "hit": _tool_str("grade_hit"),
            "game_power": _tool_str("grade_power"),
            "raw_power": _tool_str("grade_power"),
            "speed": _tool_str("grade_run"),
            "arm": _tool_str("grade_arm"),
            "field": _tool_str("grade_field"),
            "fastball": _tool_str("grade_fastball"),
            "slider": _tool_str("grade_slider"),
            "curve": _tool_str("grade_curveball"),
            "changeup": _tool_str("grade_changeup"),
            "command": _tool_str("grade_control"),
        })

    out_df = pd.DataFrame(rows)
    for int_col in ("mlbam_id", "year", "eta", "org_rank", "top_100"):
        out_df[int_col] = pd.to_numeric(out_df[int_col], errors="coerce").astype("Int64")

    out_df = out_df.sort_values(
        ["year", "trade_value"],
        ascending=[True, False],
    ).reset_index(drop=True)
    out_df.to_csv(out_path, index=False, na_rep="", encoding="utf-8")

    cur_df = out_df[out_df["year"] == CURRENT_YEAR]
    n_debuted_cur = int(cur_df["has_mlb"].sum()) if not cur_df.empty else 0
    logger.info(
        "Generated %d prospect valuations (%d–%d) -> %s | %d: %d prospects (%d has_mlb=True)",
        len(out_df),
        int(out_df["year"].min()),
        int(out_df["year"].max()),
        out_path,
        CURRENT_YEAR,
        len(cur_df),
        n_debuted_cur,
    )
    return out_df


if __name__ == "__main__":
    generate_prospect_values()
