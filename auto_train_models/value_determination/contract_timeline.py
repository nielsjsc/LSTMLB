"""
Contract Timeline Builder
=========================

Normalizes raw contract statuses (`Pre-Arb`, `Arb-1..4`, `Signed`, `Team Option`,
`Player Option`, `Mutual Option`, `Opt-Out`, `Free Agent`) and constructs a
complete player-year control timeline through 2040.

Key V2 Fixes:
- Backfills `CURRENT_YEAR` (and any leading pre-extension years) with service-time
  status (`Pre-Arb` / `Arb-*`) when a pre-debut or rookie extension in Spotrac
  starts in `CURRENT_YEAR + 1`, ensuring Year-1 MLB minimum salary is properly
  imputed rather than left blank.
- Uses `Config.logger` directly (removing legacy `constants.py` dependency).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import CURRENT_YEAR, logger


def normalize_contract_status(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw contract statuses into canonical valuation status labels."""
    result_df = df.copy().sort_values(["IDfg", "Year"])

    contract_indicators = (
        "ACTIVE", "INJURED", "RESERVE", "RETAINED", "DEFERRED",
        "BURIED", "BUYOUT", "BONUS", "VOIDED", "SIGNING BONUS",
    )

    def _status_text(value) -> str:
        return "" if pd.isna(value) else str(value).upper().strip()

    def _is_roster_status(value) -> bool:
        status = _status_text(value)
        return status in ("", "-") or any(ind in status for ind in contract_indicators)

    def _is_guaranteed_contract_year(row, group: pd.DataFrame) -> bool:
        if pd.isna(row.get("Payroll")) or not _is_roster_status(row.get("Status")):
            return False
        paid_rows = group[group["Payroll"].notna()]
        if paid_rows.empty:
            return False

        arb_mask = group["Status"].map(_status_text).str.contains(r"\bARB\b", regex=True, na=False)
        if arb_mask.any():
            last_arb_year = pd.to_numeric(group.loc[arb_mask, "Year"], errors="coerce").max()
            return pd.notna(last_arb_year) and pd.to_numeric(row.get("Year"), errors="coerce") > last_arb_year

        paid_years = pd.to_numeric(paid_rows["Year"], errors="coerce").dropna().unique()
        return len(paid_years) > 1

    def _prep_group(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["Next_Status"] = g["Status"].shift(-1)
        g["Guaranteed_Contract_Year"] = g.apply(lambda r: _is_guaranteed_contract_year(r, g), axis=1)
        return g

    def _status_from_service_time(row) -> str | None:
        yos = row.get("Years_of_Service")
        if pd.isna(yos):
            return None
        try:
            yos_float = float(yos)
        except (ValueError, TypeError):
            return None
        if yos_float < 3:
            return "Pre-Arb"
        if yos_float < 4:
            return "Arb-1"
        if yos_float < 5:
            return "Arb-2"
        if yos_float < 6:
            return "Arb-3"
        return "Signed"

    def _normalize_single_status(row) -> str | None:
        if pd.isna(row["Status"]):
            if pd.notna(row["Payroll"]):
                if row.get("Guaranteed_Contract_Year", False):
                    return "Signed"
                return _status_from_service_time(row) or "Signed"
            return None

        status = str(row["Status"]).upper().strip()
        if status == "-":
            if pd.notna(row["Payroll"]):
                if row.get("Guaranteed_Contract_Year", False):
                    return "Signed"
                return _status_from_service_time(row) or "Signed"
            return None

        if status.startswith("$"):
            return "Signed"
        if "OPT-OUT" in status or "OPT OUT" in status:
            return "Opt-Out"
        if "CLUB" in status:
            return "Team Option"
        if "PLAYER" in status:
            return "Player Option"
        if "MUTUAL" in status:
            return "Mutual Option"
        if "VESTING" in status:
            return "Vesting Option"

        if status == "ESTIMATE":
            next_status = str(row["Next_Status"]).upper().strip() if pd.notna(row["Next_Status"]) else ""
            if any(x in next_status for x in ("UFA", "FA")):
                return "Arb-3"
            if "ARB 1" in next_status:
                return "Pre-Arb"
            if "ARB 2" in next_status:
                return "Arb-1"
            if "ARB 3" in next_status:
                return "Arb-2"
            if "ARB 4" in next_status:
                return "Arb-3"
            return "Pre-Arb"

        if any(tok in status for tok in ("AVOIDED", "BYPASSED", "FILED", "SETTLED")):
            next_status = str(row["Next_Status"]).upper().strip() if pd.notna(row["Next_Status"]) else ""
            if "ARB 2" in next_status:
                return "Arb-1"
            if "ARB 3" in next_status:
                return "Arb-2"
            if "ARB 4" in next_status or "UFA" in next_status:
                return "Arb-3"
            if pd.isna(row["Next_Status"]):
                return "Signed"
            return "Arb-1"

        if "PRE" in status and "ARB" in status:
            return "Pre-Arb"

        if "ARB" in status:
            if "S2" in status:
                return "Arb-1 (Super 2)"
            if "ARB 4" in status:
                return "Arb-4"
            if "ARB 3" in status:
                return "Arb-3"
            if "ARB 2" in status:
                return "Arb-2"
            return "Arb-1"

        if any(x in status for x in ("UFA", "RFA", "FA", "FREE AGENT")):
            return "Free Agent"

        if any(ind in status for ind in contract_indicators):
            if row.get("Guaranteed_Contract_Year", False):
                return "Signed"
            service_status = _status_from_service_time(row)
            if service_status is not None:
                return service_status
            if pd.notna(row["Payroll"]):
                return "Signed"
            return None

        if pd.notna(row["Payroll"]):
            return "Signed"
        return "Unknown"

    groups = []
    for player_id, group in result_df.groupby("IDfg"):
        processed = _prep_group(group)
        if "IDfg" not in processed.columns:
            processed["IDfg"] = player_id
        groups.append(processed)
    result_df = pd.concat(groups, ignore_index=True)
    result_df["Normalized_Status"] = result_df.apply(_normalize_single_status, axis=1)
    return result_df.drop(columns=["Next_Status", "Guaranteed_Contract_Year"], errors="ignore")


def check_none_statuses(contract_data: pd.DataFrame) -> list[dict]:
    """Identify any players still missing `Normalized_Status` on their latest row."""
    problems = []
    for player_id, player_data in contract_data.groupby("IDfg"):
        player_data = player_data.sort_values("Year")
        if pd.isna(player_data.iloc[-1]["Normalized_Status"]):
            problems.append({
                "IDfg": player_id,
                "Name": player_data.iloc[0].get("Name", "?"),
                "Latest Year": int(player_data["Year"].max()),
            })
    if problems:
        logger.warning("Found %d players with unresolved latest contract status", len(problems))
    return problems


def generate_contract_timeline(df: pd.DataFrame, current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """
    Build each player's complete year-by-year contract progression (`Pre-Arb` -> `Arb` -> `Free Agent`).
    """
    result_df = df.copy()

    def _project_yos(anchor_yos, anchor_year: int, target_year: int, partial_anchor: bool = False) -> float:
        if pd.isna(anchor_yos):
            return np.nan
        offset = int(target_year) - int(anchor_year)
        if partial_anchor and offset > 0:
            offset -= 1
        return max(0.0, float(anchor_yos) + offset)

    def _tier_from_yos(yos_val: float) -> str:
        if pd.isna(yos_val) or yos_val < 3.0:
            return "Pre-Arb"
        if yos_val < 4.0:
            return "Arb-1"
        if yos_val < 5.0:
            return "Arb-2"
        if yos_val < 6.0:
            return "Arb-3"
        return "Free Agent"

    def _process_player(group: pd.DataFrame) -> pd.DataFrame:
        player_rows = group.copy().sort_values("Year")
        is_super2 = any("Super 2" in str(s) for s in player_rows["Normalized_Status"])

        # Enforce explicit Spotrac Free Agent boundary
        explicit_fa_years = player_rows.loc[player_rows["Normalized_Status"] == "Free Agent", "Year"]
        first_explicit_fa_year = int(explicit_fa_years.min()) if len(explicit_fa_years) else None
        if first_explicit_fa_year is not None:
            player_rows.loc[player_rows["Year"] >= first_explicit_fa_year, "Normalized_Status"] = "Free Agent"

        valid_status_mask = player_rows["Normalized_Status"].notna()
        non_fa_mask = valid_status_mask & (player_rows["Normalized_Status"] != "Free Agent")

        has_active_team = (
            "Team" in player_rows.columns
            and player_rows["Team"].notna().any()
            and not player_rows["Team"].dropna().astype(str).isin(["", "FA"]).all()
        )

        # Case 1: No Spotrac contract rows at all
        if not valid_status_mask.any():
            if has_active_team:
                first_year = int(player_rows["Year"].min())
                yos_vals = player_rows.loc[
                    (player_rows["Year"] <= first_year) & player_rows["Years_of_Service"].notna(),
                    "Years_of_Service",
                ]
                partial_current = bool(
                    player_rows.loc[player_rows["Year"] == first_year, "Partial_Current_Year"].any()
                ) if "Partial_Current_Year" in player_rows.columns else False
                starting_yos = float(yos_vals.iloc[-1]) if len(yos_vals) else 0.0
                end_yos = starting_yos + (0.0 if partial_current else 1.0)

                if end_yos >= 6.0:
                    player_rows["Normalized_Status"] = "Free Agent"
                else:
                    for idx, r in player_rows.iterrows():
                        yr = int(r["Year"])
                        proj_yos = _project_yos(starting_yos, first_year, yr, partial_anchor=partial_current)
                        player_rows.loc[idx, "Normalized_Status"] = _tier_from_yos(proj_yos)

                    fa_start_year = first_year + math.ceil(max(0.0, 6.0 - starting_yos)) + (1 if partial_current else 0)
                    if fa_start_year <= 2040 and not (player_rows["Year"] == fa_start_year).any():
                        new_row = {
                            col: player_rows.iloc[0][col]
                            for col in player_rows.columns
                            if col not in ("Year", "Normalized_Status", "Status", "Payroll")
                        }
                        new_row["Year"] = fa_start_year
                        if "Years_of_Service" in new_row:
                            new_row["Years_of_Service"] = _project_yos(
                                starting_yos, first_year, fa_start_year, partial_anchor=partial_current
                            )
                        if "Partial_Current_Year" in new_row:
                            new_row["Partial_Current_Year"] = False
                        new_row["Normalized_Status"] = "Free Agent"
                        new_row["Status"] = np.nan
                        new_row["Payroll"] = np.nan
                        player_rows = pd.concat([player_rows, pd.DataFrame([new_row])], ignore_index=True)
            else:
                player_rows["Normalized_Status"] = "Free Agent"
            return player_rows

        # Case 2: Backfill any blank years BEFORE the first Spotrac contract row
        # (e.g., pre-debut / rookie extension starting in CURRENT_YEAR + 1)
        first_valid_year = int(player_rows.loc[valid_status_mask, "Year"].min())
        leading_blank_mask = (
            player_rows["Normalized_Status"].isna()
            & (player_rows["Year"] >= current_year)
            & (player_rows["Year"] < first_valid_year)
        )
        if leading_blank_mask.any():
            for idx in player_rows.index[leading_blank_mask]:
                if has_active_team:
                    yos_val = player_rows.loc[idx, "Years_of_Service"] if "Years_of_Service" in player_rows.columns else 0.0
                    tier = _tier_from_yos(0.0 if pd.isna(yos_val) else float(yos_val))
                    player_rows.loc[idx, "Normalized_Status"] = "Pre-Arb" if tier == "Free Agent" else tier
                else:
                    player_rows.loc[idx, "Normalized_Status"] = "Free Agent"

        # Case 3: Project forward from the last valid non-FA status
        valid_status_mask = player_rows["Normalized_Status"].notna()
        non_fa_mask = valid_status_mask & (player_rows["Normalized_Status"] != "Free Agent")
        last_valid_idx = player_rows[non_fa_mask].index[-1] if non_fa_mask.any() else player_rows[valid_status_mask].index[-1]

        latest_status = player_rows.loc[last_valid_idx, "Normalized_Status"]
        latest_year = int(player_rows.loc[last_valid_idx, "Year"])

        anchor_yos = player_rows.loc[last_valid_idx, "Years_of_Service"]
        if pd.isna(anchor_yos):
            prior_yos = player_rows.loc[
                (player_rows["Year"] <= latest_year) & player_rows["Years_of_Service"].notna(),
                "Years_of_Service",
            ]
            anchor_yos = prior_yos.iloc[-1] if len(prior_yos) else np.nan
        anchor_partial = bool(
            player_rows.loc[player_rows.index == last_valid_idx, "Partial_Current_Year"].any()
        ) if "Partial_Current_Year" in player_rows.columns else False

        if latest_status == "Free Agent":
            player_rows.loc[player_rows["Normalized_Status"].isna(), "Normalized_Status"] = "Free Agent"
            return player_rows

        new_statuses: list[tuple[str, int]] = []
        if latest_status.startswith("Arb-4"):
            new_statuses = [("Free Agent", 1)]
        elif latest_status.startswith("Arb-3"):
            new_statuses = [("Arb-4", 1), ("Free Agent", 2)] if is_super2 else [("Free Agent", 1)]
        elif latest_status.startswith("Arb-2"):
            new_statuses = (
                [("Arb-3", 1), ("Arb-4", 2), ("Free Agent", 3)]
                if is_super2
                else [("Arb-3", 1), ("Free Agent", 2)]
            )
        elif latest_status.startswith("Arb-1"):
            new_statuses = (
                [("Arb-2", 1), ("Arb-3", 2), ("Arb-4", 3), ("Free Agent", 4)]
                if is_super2
                else [("Arb-2", 1), ("Arb-3", 2), ("Free Agent", 3)]
            )
        elif latest_status == "Pre-Arb":
            if pd.notna(anchor_yos):
                end_of_season_yos = float(anchor_yos) + (0.0 if anchor_partial else 1.0)
                remaining_pre_arb = math.ceil(max(0.0, 3.0 - end_of_season_yos))
            else:
                pre_arb_years = len(player_rows[player_rows["Normalized_Status"] == "Pre-Arb"])
                remaining_pre_arb = max(0, 3 - pre_arb_years)

            year_offset = 0
            for _ in range(remaining_pre_arb):
                year_offset += 1
                new_statuses.append(("Pre-Arb", year_offset))

            arb1_status = "Arb-1 (Super 2)" if is_super2 else "Arb-1"
            if is_super2:
                new_statuses.extend([
                    (arb1_status, year_offset + 1),
                    ("Arb-2", year_offset + 2),
                    ("Arb-3", year_offset + 3),
                    ("Arb-4", year_offset + 4),
                    ("Free Agent", year_offset + 5),
                ])
            else:
                new_statuses.extend([
                    (arb1_status, year_offset + 1),
                    ("Arb-2", year_offset + 2),
                    ("Arb-3", year_offset + 3),
                    ("Free Agent", year_offset + 4),
                ])
        else:
            payroll_years = player_rows[player_rows["Payroll"].notna()]["Year"]
            last_contract_year = int(payroll_years.max()) if len(payroll_years) else latest_year
            year_offset = int(last_contract_year - latest_year + 1)
            new_statuses = [("Free Agent", year_offset)]

        for status, year_offset in new_statuses:
            target_year = int(latest_year + year_offset)
            if first_explicit_fa_year is not None and target_year >= first_explicit_fa_year:
                continue
            year_mask = player_rows["Year"] == target_year
            if year_mask.any():
                blank_mask = year_mask & player_rows["Normalized_Status"].isna()
                player_rows.loc[blank_mask, "Normalized_Status"] = status
            elif target_year <= 2040:
                new_row = {
                    col: player_rows.iloc[0][col]
                    for col in player_rows.columns
                    if col not in ("Year", "Normalized_Status", "Status", "Payroll")
                }
                new_row["Year"] = target_year
                if "Years_of_Service" in new_row:
                    new_row["Years_of_Service"] = _project_yos(
                        anchor_yos, latest_year, target_year, partial_anchor=anchor_partial
                    )
                if "Partial_Current_Year" in new_row:
                    new_row["Partial_Current_Year"] = False
                new_row["Normalized_Status"] = status
                new_row["Status"] = np.nan
                new_row["Payroll"] = np.nan
                player_rows = pd.concat([player_rows, pd.DataFrame([new_row])], ignore_index=True)

        # Fill any remaining interior/trailing blanks
        player_rows = player_rows.sort_values("Year")
        fa_years = player_rows[player_rows["Normalized_Status"] == "Free Agent"]["Year"]
        if len(fa_years) > 0:
            first_fa_year = int(fa_years.min())
            none_after_fa = (player_rows["Year"] >= first_fa_year) & player_rows["Normalized_Status"].isna()
            player_rows.loc[none_after_fa, "Normalized_Status"] = "Free Agent"

        still_blank = player_rows["Normalized_Status"].isna()
        if still_blank.any():
            player_rows.loc[still_blank, "Normalized_Status"] = (
                "Pre-Arb" if has_active_team else "Free Agent"
            )

        return player_rows

    groups = []
    for player_id, group in result_df.groupby("IDfg"):
        processed = _process_player(group)
        if "IDfg" not in processed.columns:
            processed["IDfg"] = player_id
        groups.append(processed)
    result_df = pd.concat(groups, ignore_index=True)
    return result_df.sort_values(["IDfg", "Year"]).reset_index(drop=True)


def validate_fa_years(contract_timeline: pd.DataFrame) -> list[dict]:
    """Verify every player has an explicit Free Agent transition year."""
    missing_fa = []
    for idfg, group in contract_timeline.groupby("IDfg"):
        if not (group["Normalized_Status"] == "Free Agent").any():
            missing_fa.append({
                "IDfg": idfg,
                "Player": group["Name"].iloc[0] if "Name" in group.columns else "?",
                "Last Status": group.sort_values("Year")["Normalized_Status"].iloc[-1],
                "Last Year": int(group["Year"].max()),
            })
    if missing_fa:
        logger.warning("Found %d players missing Free Agent status", len(missing_fa))
    return missing_fa


def extend_fa_timeline(timeline_df: pd.DataFrame) -> pd.DataFrame:
    """Extend Free Agent rows through 2040 so 15-year projections merge cleanly."""
    timeline_df = timeline_df.copy()
    fa_years = (
        timeline_df[timeline_df["Normalized_Status"] == "Free Agent"]
        .groupby("IDfg")["Year"]
        .min()
        .reset_index()
    )
    if fa_years.empty:
        return timeline_df

    role_col = "player_role" if "player_role" in timeline_df.columns else "position_group"
    future_rows = []
    for _, row in fa_years.iterrows():
        idfg = row["IDfg"]
        p_rows = timeline_df[timeline_df["IDfg"] == idfg]
        player_info = p_rows.iloc[0]
        max_year = int(p_rows["Year"].max())
        role_val = player_info.get(role_col, "Batter")

        for year in range(max_year + 1, 2041):
            rec = {
                "Name": player_info.get("Name", ""),
                "IDfg": idfg,
                "player_role": role_val,
                "position_group": role_val,
                "Year": year,
                "Team": player_info.get("Team", np.nan),
                "Payroll": np.nan,
                "Status": np.nan,
                "Normalized_Status": "Free Agent",
            }
            if "mlbam_id" in timeline_df.columns:
                rec["mlbam_id"] = player_info.get("mlbam_id", np.nan)
            future_rows.append(rec)

    if future_rows:
        extended = pd.concat([timeline_df, pd.DataFrame(future_rows)], ignore_index=True)
    else:
        extended = timeline_df

    return (
        extended.sort_values(["IDfg", "Year"])
        .drop_duplicates(subset=["IDfg", "Year"], keep="first")
        .reset_index(drop=True)
    )
