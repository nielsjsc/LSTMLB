"""
Spotrac Salary Cleaner & Player ID Matcher
==========================================

Cleans raw Spotrac contract data (`mlb_salary_data.csv`), computes accurate
CBA annual cash/escrow obligations, deduplicates using `spotrac_id` (never
confusing Spotrac's internal ID with FanGraphs `IDfg`), and matches contracts
to MLB player IDs with automatic team-based homonym disambiguation.

CBA Annual Obligation Rules:
1. **Contracts WITH post-FA deferred cash rows** (e.g., Ohtani, Betts, Snell,
   Tucker, where `cash_annual > 0` rows appear at/after the UFA year with
   `payroll_annual = NaN` and `luxury_tax = NaN`):
   - Drop post-FA deferral payout rows so they never extend team control.
   - Use `luxury_tax` on active playing years (= base salary + mandatory MLB
     CBA Art. XVI annual present-value deferral escrow funding).
2. **All other contracts** (~98% of MLB):
   - Use `cash_annual.fillna(payroll_annual).fillna(luxury_tax)` so frontloaded
     signing bonuses already paid (e.g., Vlad Jr.) are not re-charged, backloaded
     escalators (e.g., PCA) reflect actual future cash owed, and Club/Player
     options (`cash_annual = NaN`) use their exact option exercise salary (`payroll_annual`).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from core.id_crosswalk import normalize_team_code
from core.name_utils import (
    _SUFFIX_RE as _SUFFIXES,
    normalize_name,
    strip_middle_initials,
)
from .config import CURRENT_YEAR, Config, logger

_FA_KEYWORDS = ("UFA", "RFA", "FA", "FREE AGENT")
_DOLLAR_TOKEN_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def _normalize_name_no_suffix(name: str) -> str:
    normalized = normalize_name(name)
    if pd.isna(normalized):
        return normalized
    return _SUFFIXES.sub("", normalized).strip()


def _parse_dollar_col(series: pd.Series) -> pd.Series:
    """Parse Spotrac dollar strings, summing multi-span cells like ``'$15,000,000$2,000,000'``."""
    def _parse_one(val) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val) if 0 <= float(val) <= 120_000_000 else np.nan
        txt = str(val).split("(")[0].strip()
        if not txt or txt in ("-", "nan", "None", "NaN"):
            return np.nan
        tokens = _DOLLAR_TOKEN_RE.findall(txt)
        if tokens:
            total = sum(float(tok.replace(",", "")) for tok in tokens)
            return total if 0 <= total <= 120_000_000 else np.nan
        cleaned = txt.replace("$", "").replace(",", "").strip()
        try:
            num = float(cleaned)
            return num if 0 <= num <= 120_000_000 else np.nan
        except ValueError:
            return np.nan

    return series.map(_parse_one)


def _has_fa_status(status_val) -> bool:
    if pd.isna(status_val):
        return False
    text = str(status_val).upper().strip()
    return any(kw == text or f" {kw}" in f" {text} " for kw in _FA_KEYWORDS)


def clean_salary_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Spotrac salary data and compute CBA-accurate annual salary obligations.

    Returns:
        DataFrame with columns:
        `['spotrac_id', 'Player Name', 'Year', 'Team', 'Payroll', 'Status', 'Years_of_Service']`
    """
    logger.info("Cleaning Spotrac salary data (%d raw rows)...", len(df))
    cleaned = df.copy()

    name_col = "player_name" if "player_name" in cleaned.columns else "Player Name"
    payroll_col = "payroll_annual" if "payroll_annual" in cleaned.columns else "Payroll"
    ltax_col = "luxury_tax" if "luxury_tax" in cleaned.columns else None
    cash_col = "cash_annual" if "cash_annual" in cleaned.columns else None
    status_col = "status" if "status" in cleaned.columns else "Status"
    year_col = "year" if "year" in cleaned.columns else "Year"
    team_col = "team" if "team" in cleaned.columns else "Team"
    raw_id_col = "player_id" if "player_id" in cleaned.columns else None

    # Drop summary / non-player rows
    cleaned = cleaned[
        ~cleaned[name_col].astype(str).str.contains(
            "OPT-OUT|UFA|PLAYER OPT|CLUB OPT|RUNNING PAYROLL",
            na=False,
            case=False,
        )
    ].copy()

    cleaned["Year"] = pd.to_numeric(cleaned[year_col], errors="coerce")
    cleaned = cleaned.dropna(subset=["Year"]).copy()
    cleaned["Year"] = cleaned["Year"].astype(int)

    # Scoped strictly to salary cleaning — never named IDfg!
    if raw_id_col and raw_id_col in cleaned.columns:
        cleaned["spotrac_id"] = pd.to_numeric(cleaned[raw_id_col], errors="coerce")
        fallback_mask = cleaned["spotrac_id"].isna()
        if fallback_mask.any():
            # Deterministic negative IDs for any rows lacking numeric player_id
            cleaned.loc[fallback_mask, "spotrac_id"] = -(
                pd.factorize(cleaned.loc[fallback_mask, name_col])[0] + 1
            )
        cleaned["spotrac_id"] = cleaned["spotrac_id"].astype(int)
    else:
        cleaned["spotrac_id"] = -(pd.factorize(cleaned[name_col])[0] + 1)

    payroll_str = cleaned[payroll_col].astype(str)
    is_fa_in_payroll = payroll_str.str.contains(
        r"\b(?:UFA|RFA|FA)\b", case=False, na=False, regex=True
    )

    cleaned["_payroll_annual"] = _parse_dollar_col(cleaned[payroll_col])
    cleaned["_luxury_tax"] = (
        _parse_dollar_col(cleaned[ltax_col]) if ltax_col else pd.Series(np.nan, index=cleaned.index)
    )
    cleaned["_cash_annual"] = (
        _parse_dollar_col(cleaned[cash_col]) if cash_col else pd.Series(np.nan, index=cleaned.index)
    )

    cleaned["Status"] = cleaned[status_col] if status_col in cleaned.columns else None
    cleaned.loc[is_fa_in_payroll, "Status"] = "UFA"

    yos_col = "years_of_service" if "years_of_service" in cleaned.columns else "Years of Service"
    if yos_col in cleaned.columns:
        cleaned["Years_of_Service"] = pd.to_numeric(
            cleaned[yos_col].astype(str).str.replace("-", "", regex=False).str.strip(),
            errors="coerce",
        )
    else:
        cleaned["Years_of_Service"] = np.nan

    cleaned["Player Name"] = cleaned[name_col]
    cleaned["Team"] = cleaned[team_col].map(normalize_team_code)

    # ── Identify post-FA deferred cash payout rows per spotrac_id ────────────
    # A post-FA deferral row has cash_annual > 0 while both payroll_annual and
    # luxury_tax are NaN (and Status is not an FA marker), occurring at or after
    # the player's UFA year (or after their last active payroll/ltax year).
    is_fa_row = cleaned["Status"].map(_has_fa_status)
    fa_min_by_player = cleaned.loc[is_fa_row].groupby("spotrac_id")["Year"].min()
    active_max_by_player = (
        cleaned.loc[cleaned["_payroll_annual"].notna() | cleaned["_luxury_tax"].notna()]
        .groupby("spotrac_id")["Year"]
        .max()
    )

    cleaned["_first_fa_yr"] = cleaned["spotrac_id"].map(fa_min_by_player)
    cleaned["_last_active_yr"] = cleaned["spotrac_id"].map(active_max_by_player)

    is_deferral_payout_row = (
        ~is_fa_row
        & cleaned["_cash_annual"].notna()
        & (cleaned["_cash_annual"] > 0)
        & cleaned["_payroll_annual"].isna()
        & cleaned["_luxury_tax"].isna()
        & (
            (cleaned["_first_fa_yr"].notna() & (cleaned["Year"] >= cleaned["_first_fa_yr"]))
            | (cleaned["_last_active_yr"].notna() & (cleaned["Year"] > cleaned["_last_active_yr"]))
        )
    )

    deferred_spotrac_ids = set(cleaned.loc[is_deferral_payout_row, "spotrac_id"].unique())
    n_deferral_rows = int(is_deferral_payout_row.sum())
    if n_deferral_rows:
        cleaned = cleaned[~is_deferral_payout_row].copy()
        logger.info(
            "Dropped %d post-FA deferral payout rows across %d deferred-money contracts "
            "(using CBA luxury_tax escrow obligation for those contracts)",
            n_deferral_rows,
            len(deferred_spotrac_ids),
        )

    # ── Compute CBA-accurate annual Payroll obligation ───────────────────────
    is_deferred_contract = cleaned["spotrac_id"].isin(deferred_spotrac_ids)
    is_option_row = cleaned["Status"].astype(str).str.contains(
        "CLUB|PLAYER|MUTUAL|VESTING", case=False, na=False
    )

    # Standard contracts + explicit option exercise rows: actual remaining cash -> payroll -> ltax
    standard_obligation = (
        cleaned["_cash_annual"]
        .fillna(cleaned["_payroll_annual"])
        .fillna(cleaned["_luxury_tax"])
    )
    # Post-FA deferred contracts (active guaranteed years): CBA escrow + salary = luxury_tax
    deferred_obligation = (
        cleaned["_luxury_tax"]
        .fillna(cleaned["_payroll_annual"])
        .fillna(cleaned["_cash_annual"])
    )

    cleaned["Payroll"] = np.where(
        is_deferred_contract & ~is_option_row,
        deferred_obligation,
        standard_obligation,
    )
    # Ensure pure FA marker rows never carry salary
    cleaned.loc[cleaned["Status"].map(_has_fa_status), "Payroll"] = np.nan

    # ── Deduplicate cross-team scrape duplicates & merge same-year components ─
    pre_dedup = len(cleaned)
    cleaned["_payroll_key"] = cleaned["Payroll"].fillna(-9999.0)
    cleaned["_fa_key"] = cleaned["Status"].map(_has_fa_status)
    cleaned = cleaned.drop_duplicates(
        subset=["spotrac_id", "Year", "_payroll_key", "_fa_key"],
        keep="first",
    )

    def _agg_components(g: pd.DataFrame) -> pd.Series:
        non_fa = g[~g["Status"].map(_has_fa_status)]
        primary = non_fa if len(non_fa) > 0 else g
        return pd.Series({
            "Player Name": g["Player Name"].dropna().iloc[0] if g["Player Name"].notna().any() else np.nan,
            "Team": g["Team"].dropna().iloc[0] if g["Team"].notna().any() else "FA",
            "Payroll": primary["Payroll"].sum() if primary["Payroll"].notna().any() else np.nan,
            "Status": primary["Status"].dropna().iloc[0] if primary["Status"].notna().any() else np.nan,
            "Years_of_Service": g["Years_of_Service"].dropna().iloc[0] if g["Years_of_Service"].notna().any() else np.nan,
            "_has_fa_marker": bool(g["Status"].map(_has_fa_status).any()),
        })

    cleaned = (
        cleaned.groupby(["spotrac_id", "Year"])
        .apply(_agg_components, include_groups=False)
        .reset_index()
    )

    # Re-emit dual-row same-year UFA markers shifted +1 year (e.g., PCA 2032 Active + 2032 UFA -> 2033 UFA)
    dual_fa = cleaned[cleaned["_has_fa_marker"] & ~cleaned["Status"].map(_has_fa_status)]
    if not dual_fa.empty:
        # Only emit Year + 1 UFA if that player doesn't already have a row at Year + 1
        existing_pairs = set(zip(cleaned["spotrac_id"], cleaned["Year"]))
        extra_fa_rows = []
        for _, frow in dual_fa.iterrows():
            next_yr = int(frow["Year"]) + 1
            if (int(frow["spotrac_id"]), next_yr) not in existing_pairs:
                extra_fa_rows.append({
                    "spotrac_id": int(frow["spotrac_id"]),
                    "Year": next_yr,
                    "Player Name": frow["Player Name"],
                    "Team": frow["Team"],
                    "Payroll": np.nan,
                    "Status": "UFA",
                    "Years_of_Service": np.nan,
                })
        if extra_fa_rows:
            cleaned = pd.concat([cleaned, pd.DataFrame(extra_fa_rows)], ignore_index=True)

    cleaned = cleaned.drop(columns=["_has_fa_marker"], errors="ignore")

    # Project Years_of_Service forward within each spotrac_id
    for sp_id, grp in cleaned.groupby("spotrac_id"):
        grp_sorted = grp.sort_values("Year")
        valid_yos = grp_sorted[grp_sorted["Years_of_Service"].notna()]
        if not valid_yos.empty:
            base_yr = int(valid_yos.iloc[0]["Year"])
            base_yos = float(valid_yos.iloc[0]["Years_of_Service"])
            missing_idx = grp_sorted.index[grp_sorted["Years_of_Service"].isna()]
            if len(missing_idx):
                cleaned.loc[missing_idx, "Years_of_Service"] = (
                    base_yos + (cleaned.loc[missing_idx, "Year"] - base_yr)
                ).clip(lower=0.0)

    logger.info(
        "Cleaned Spotrac salary data: %d -> %d deduplicated player-year rows",
        pre_dedup,
        len(cleaned),
    )
    output_cols = ["spotrac_id", "Player Name", "Year", "Team", "Payroll", "Status", "Years_of_Service"]
    return cleaned[output_cols].copy()


def _resolve_spotrac_to_fg_id(
    salary_df: pd.DataFrame,
    player_ref: pd.DataFrame,
    org_data: Optional[pd.DataFrame] = None,
) -> dict[int, tuple[int, str, str]]:
    """
    Map `spotrac_id -> (IDfg, Name, player_role)` using normalized names,
    suffix-stripping, roster alias fallback, and automatic team disambiguation
    for same-name players (e.g. Max Muncy, Luis Garcia, Will Smith, Logan Allen).
    """
    # Build candidate pool from predictions: IDfg -> {name, role, team, name_variants}
    roster_team_by_fg: dict[int, str] = {}
    roster_variants_by_fg: dict[int, set[str]] = {}

    if org_data is not None and not org_data.empty:
        for _, r in org_data.iterrows():
            if pd.notna(r.get("IDfg")):
                fg = int(r["IDfg"])
                roster_team_by_fg[fg] = normalize_team_code(r.get("Team"))

    if Config.Paths.ROSTER_FILE.exists():
        try:
            roster_raw = pd.read_csv(Config.Paths.ROSTER_FILE, low_memory=False)
            for _, r in roster_raw.iterrows():
                fg_val = pd.to_numeric(r.get("fg_id"), errors="coerce")
                if pd.isna(fg_val) or fg_val <= 0:
                    continue
                fg = int(fg_val)
                if fg not in roster_team_by_fg and pd.notna(r.get("team_name")):
                    roster_team_by_fg[fg] = normalize_team_code(r.get("team_name"))
                variants = roster_variants_by_fg.setdefault(fg, set())
                for col in ("mapped_name", "player_name"):
                    val = r.get(col)
                    if pd.notna(val) and str(val).strip():
                        n1 = normalize_name(str(val))
                        n2 = strip_middle_initials(n1)
                        n3 = _normalize_name_no_suffix(str(val))
                        n4 = strip_middle_initials(n3)
                        variants.update(x for x in (n1, n2, n3, n4) if x)
        except Exception as exc:
            logger.debug("Could not read roster variants: %s", exc)

    pred_players = player_ref[["IDfg", "Name", "player_role"]].drop_duplicates("IDfg")
    name_to_candidates: dict[str, list[tuple[int, str, str, str]]] = {}

    for _, prow in pred_players.iterrows():
        fg = int(prow["IDfg"])
        pname = str(prow["Name"])
        prole = str(prow["player_role"])
        pteam = roster_team_by_fg.get(fg, "FA")

        n1 = normalize_name(pname)
        n2 = strip_middle_initials(n1)
        n3 = _normalize_name_no_suffix(pname)
        n4 = strip_middle_initials(n3)
        all_keys = {x for x in (n1, n2, n3, n4) if x} | roster_variants_by_fg.get(fg, set())

        for k in all_keys:
            name_to_candidates.setdefault(k, []).append((fg, pname, prole, pteam))

    # Also allow roster-only players (not in predictions) as fallback
    roster_only_lookup: dict[str, list[tuple[int, str, str]]] = {}
    for fg, variants in roster_variants_by_fg.items():
        if fg in set(pred_players["IDfg"].astype(int)):
            continue
        rteam = roster_team_by_fg.get(fg, "FA")
        for k in variants:
            roster_only_lookup.setdefault(k, []).append((fg, k.title(), rteam))

    mapping: dict[int, tuple[int, str, str]] = {}
    spotrac_players = (
        salary_df.sort_values("Year")
        .groupby("spotrac_id", as_index=False)
        .first()[["spotrac_id", "Player Name", "Team"]]
    )

    for _, srow in spotrac_players.iterrows():
        sp_id = int(srow["spotrac_id"])
        raw_name = str(srow["Player Name"] or "")
        sp_team = normalize_team_code(srow.get("Team"))

        k1 = strip_middle_initials(normalize_name(raw_name))
        k2 = normalize_name(raw_name)
        k3 = strip_middle_initials(_normalize_name_no_suffix(raw_name))
        k4 = _normalize_name_no_suffix(raw_name)

        cands: list[tuple[int, str, str, str]] = []
        for k in (k1, k2, k3, k4):
            if k and k in name_to_candidates:
                # Deduplicate by IDfg while preserving order
                seen_fg = set()
                cands = [
                    c for c in name_to_candidates[k]
                    if c[0] not in seen_fg and not seen_fg.add(c[0])
                ]
                break

        if len(cands) == 1:
            fg, pname, prole, _ = cands[0]
            mapping[sp_id] = (fg, pname, prole)
        elif len(cands) > 1:
            # Disambiguate homonyms by matching Spotrac team against active roster team
            team_matches = [c for c in cands if sp_team != "FA" and c[3] == sp_team]
            if len(team_matches) == 1:
                fg, pname, prole, _ = team_matches[0]
                mapping[sp_id] = (fg, pname, prole)
            else:
                active_matches = [c for c in cands if c[3] != "FA"]
                chosen = active_matches[0] if len(active_matches) == 1 else cands[0]
                mapping[sp_id] = (chosen[0], chosen[1], chosen[2])
        else:
            # Roster-only fallback
            for k in (k1, k2, k3, k4):
                if k and k in roster_only_lookup:
                    rcands = roster_only_lookup[k]
                    tmatch = [c for c in rcands if sp_team != "FA" and c[2] == sp_team]
                    chosen = tmatch[0] if tmatch else rcands[0]
                    mapping[sp_id] = (chosen[0], raw_name, "Batter")
                    break

    return mapping


def merge_salary_with_ids(
    salary_df: pd.DataFrame,
    sp_df: pd.DataFrame,
    rp_df: pd.DataFrame,
    batter_df: pd.DataFrame,
    org_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Resolve Spotrac contracts to `IDfg` and outer-merge with prediction player-years.

    Unlike V1's name-based outer join, resolving `spotrac_id -> IDfg` up front
    prevents homonym Cartesian collisions (e.g., Max Muncy LAD vs Max Muncy ATH).
    """
    def _prep_ref(df: pd.DataFrame, default_role: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["Name", "IDfg", "player_role", "Year"])
        role_col = "player_role" if "player_role" in df.columns else "position_group"
        out = df[["Name", "IDfg", "Year"]].copy()
        if role_col in df.columns:
            out["player_role"] = df[role_col].replace({"POS": "Batter", "batter": "Batter", "hitter": "Batter"})
        else:
            out["player_role"] = default_role
        return out

    player_ref = pd.concat(
        [
            _prep_ref(sp_df, "SP"),
            _prep_ref(rp_df, "RP"),
            _prep_ref(batter_df, "Batter"),
        ],
        ignore_index=True,
    )
    # Collapse two-way players to one reference row per (IDfg, Year) for contract timeline building
    player_ref = player_ref.drop_duplicates(subset=["IDfg", "Year"], keep="first")

    sp_to_fg = _resolve_spotrac_to_fg_id(salary_df, player_ref, org_data=org_data)

    sal = salary_df.copy()
    sal["IDfg"] = sal["spotrac_id"].map(lambda sid: sp_to_fg[sid][0] if sid in sp_to_fg else np.nan)
    sal["_matched_name"] = sal["spotrac_id"].map(lambda sid: sp_to_fg[sid][1] if sid in sp_to_fg else np.nan)
    sal["_matched_role"] = sal["spotrac_id"].map(lambda sid: sp_to_fg[sid][2] if sid in sp_to_fg else np.nan)

    matched_sal = sal[sal["IDfg"].notna()].copy()
    matched_sal["IDfg"] = matched_sal["IDfg"].astype(int)
    # If two Spotrac entries mapped to the same (IDfg, Year), keep the one with valid Payroll
    matched_sal = (
        matched_sal.sort_values(["Payroll"], ascending=False, na_position="last")
        .drop_duplicates(subset=["IDfg", "Year"], keep="first")
    )

    unmatched_count = sal["spotrac_id"].nunique() - matched_sal["spotrac_id"].nunique()
    if unmatched_count > 0 and Config.Pipeline.LOG_UNMATCHED_PLAYERS:
        logger.info(
            "Spotrac ID resolution: matched %d/%d unique players (%d retired/minors/unmatched skipped)",
            matched_sal["spotrac_id"].nunique(),
            sal["spotrac_id"].nunique(),
            unmatched_count,
        )

    merge_cols = ["IDfg", "Year", "Payroll", "Status", "Years_of_Service", "_matched_name", "_matched_role"]
    merged = player_ref.merge(matched_sal[merge_cols], on=["IDfg", "Year"], how="outer")

    merged["Name"] = merged["Name"].fillna(merged["_matched_name"])
    merged["player_role"] = merged["player_role"].fillna(merged["_matched_role"]).fillna("Batter")
    merged["position_group"] = merged["player_role"]  # Keep alias for downstream callers during transition
    merged = merged.drop(columns=["_matched_name", "_matched_role"], errors="ignore")

    merged = merged[merged["IDfg"].notna()].copy()
    merged["IDfg"] = merged["IDfg"].astype(int)
    logger.info(
        "Salary merge complete: %d player-year rows across %d unique MLB players",
        len(merged),
        merged["IDfg"].nunique(),
    )
    return merged


def complete_years_of_service(
    df: pd.DataFrame,
    batting_history: Optional[pd.DataFrame] = None,
    pitching_history: Optional[pd.DataFrame] = None,
    current_year: int = CURRENT_YEAR,
) -> pd.DataFrame:
    """
    Complete missing `Years_of_Service` values after salary/prediction merging.

    1. Preserve valid Spotrac values and project them linearly across missing years.
    2. Otherwise estimate service seasons from MLB history strictly prior to `current_year`.
    3. Mark fresh debutants (`0` prior seasons) with `Partial_Current_Year = True`.
    """
    result = df.copy()
    if "Years_of_Service" not in result.columns:
        result["Years_of_Service"] = np.nan
    result["Years_of_Service"] = pd.to_numeric(result["Years_of_Service"], errors="coerce")

    if "Partial_Current_Year" not in result.columns:
        result["Partial_Current_Year"] = False
    result["Partial_Current_Year"] = result["Partial_Current_Year"].fillna(False).astype(bool)

    if "IDfg" not in result.columns or "Year" not in result.columns:
        return result

    historical_yos: dict[int, int] = {}
    for history in (batting_history, pitching_history):
        if history is None or not {"IDfg", "Season"}.issubset(history.columns):
            continue
        h_ids = pd.to_numeric(history["IDfg"], errors="coerce")
        h_seasons = pd.to_numeric(history["Season"], errors="coerce")
        h_frame = pd.DataFrame({"IDfg": h_ids, "Season": h_seasons}).dropna()
        h_frame = h_frame[h_frame["Season"] < current_year]
        for pid, seasons in h_frame.groupby("IDfg")["Season"].nunique().items():
            key = int(pid)
            historical_yos[key] = max(historical_yos.get(key, 0), int(seasons))

    for player_id, group_indices in result.groupby("IDfg", dropna=True).groups.items():
        group = result.loc[group_indices].sort_values("Year")
        valid = group.dropna(subset=["Years_of_Service", "Year"])

        if not valid.empty:
            anchor = valid.iloc[0]
            anchor_year = float(anchor["Year"])
            anchor_yos = float(anchor["Years_of_Service"])
            missing_indices = group.index[group["Years_of_Service"].isna()]
            if len(missing_indices):
                result.loc[missing_indices, "Years_of_Service"] = (
                    pd.to_numeric(result.loc[missing_indices, "Year"], errors="coerce")
                    - anchor_year
                    + anchor_yos
                ).clip(lower=0.0)
            continue

        estimated_yos = historical_yos.get(int(player_id), 0)
        years = pd.to_numeric(group["Year"], errors="coerce")
        if estimated_yos == 0:
            result.loc[
                group.index[result.loc[group.index, "Year"].eq(current_year)],
                "Partial_Current_Year",
            ] = True
            future_years = (years - current_year - 1).clip(lower=0)
        else:
            future_years = (years - current_year).clip(lower=0)
        result.loc[group.index, "Years_of_Service"] = (estimated_yos + future_years).clip(lower=0.0)

    return result
