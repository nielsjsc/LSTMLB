"""
Historical Values — Timeline Generator
========================================

Combines three data sources into a unified trade-value history CSV:

  1. **Prospect rankings**  — FV grade + top-100 rank → dollar value
  2. **MLB surplus files**  — projected WAR at each snapshot year
  3. **Spotrac transactions** — every transaction (trades, signings,
     extensions, arbitration, releases, DFA, claims, drafts, …)

Each row carries a concrete ``date`` (YYYY-MM-DD).  Multiple entries per
player per year are expected (e.g. a pre-season projection *and* a
mid-season trade).  Spotrac ``fa_signing`` entries whose description
contains "avoiding arbitration" are reclassified as ``arbitration``.

Output:  data/generated/value_by_year/trade_value_history.csv

Usage:
    cd auto_train_models
    python -m historical_values.timeline
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_AUTO_TRAIN = Path(__file__).resolve().parents[2]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from value_determination.config import Config, logger, CURRENT_YEAR

# Path shortcuts
SURPLUS_DIR     = Config.Paths.SURPLUS_DIR
OUTPUT_FILE     = Config.Paths.TRADE_VALUE_HISTORY_FILE
PROSPECT_FILE   = Config.Paths.PROSPECT_FILE
CROSSWALK_FILE  = Config.Paths.CROSSWALK_FILE
PLAYER_VALUES_FILE = Config.Paths.PLAYER_VALUES_FILE
SPOTRAC_TRANSACTIONS_FILE = Config.Paths.SPOTRAC_TRANSACTIONS_FILE

from value_determination.pipelines._history_war import war_to_dollars

# Import canonical prospect valuation from value_determination
from value_determination.trade_value import (
    _prospect_dollar_value as _vd_prospect_dollar_value,
)

# Canonical name normalization
from core.name_utils import name_key as _normalise_name

# ═══════════════════════════════════════════════════════════════════════════════
# ID cross-walk helpers
# ═══════════════════════════════════════════════════════════════════════════════

_xw_cache: pd.DataFrame | None = None


def _load_crosswalk() -> pd.DataFrame:
    """Load the mlb_id ↔ IDfg crosswalk CSV (one row per player)."""
    global _xw_cache
    if _xw_cache is not None:
        return _xw_cache

    if not CROSSWALK_FILE.exists():
        logger.warning(f"Crosswalk file missing: {CROSSWALK_FILE}")
        _xw_cache = pd.DataFrame(columns=["mlb_id", "IDfg", "name"])
        return _xw_cache

    xw = pd.read_csv(CROSSWALK_FILE, low_memory=False)

    # Normalise column names — accept common variants
    col_map = {}
    for c in xw.columns:
        cl = c.lower().strip()
        if cl in ("mlb_id", "mlbam_id", "key_mlbam"):
            col_map[c] = "mlb_id"
        elif cl in ("idfg", "key_fangraphs", "fg_id"):
            col_map[c] = "IDfg"
        elif cl in ("name", "player_name", "name_common"):
            col_map[c] = "name"
    xw = xw.rename(columns=col_map)

    for needed in ("mlb_id", "IDfg"):
        if needed not in xw.columns:
            logger.warning(f"Crosswalk missing column '{needed}'")
            _xw_cache = pd.DataFrame(columns=["mlb_id", "IDfg", "name"])
            return _xw_cache

    xw = xw.dropna(subset=["mlb_id", "IDfg"])
    # Filter out non-numeric IDs (e.g. 'sa657920' prospect IDs)
    xw["mlb_id"] = pd.to_numeric(xw["mlb_id"], errors="coerce")
    xw["IDfg"]   = pd.to_numeric(xw["IDfg"], errors="coerce")
    xw = xw.dropna(subset=["mlb_id", "IDfg"])
    xw["mlb_id"] = xw["mlb_id"].astype(int)
    xw["IDfg"]   = xw["IDfg"].astype(int)
    xw = xw.drop_duplicates(subset=["mlb_id"])

    _xw_cache = xw
    return xw


def _build_idfg_to_mlb(player_values: pd.DataFrame) -> dict[int, int]:
    """IDfg → mlb_id mapping from player_values + crosswalk."""
    mapping: dict[int, int] = {}

    xw = _load_crosswalk()
    if not xw.empty:
        for _, row in xw.iterrows():
            mapping[int(row["IDfg"])] = int(row["mlb_id"])

    if player_values is not None and not player_values.empty:
        subset = player_values.dropna(subset=["mlb_id", "IDfg"]).drop_duplicates("IDfg")
        for _, row in subset.iterrows():
            mapping[int(row["IDfg"])] = int(row["mlb_id"])

    return mapping


def _build_mlb_to_name(player_values: pd.DataFrame) -> dict[int, str]:
    """mlb_id → canonical player name (from player_values + crosswalk)."""
    lookup: dict[int, str] = {}

    # Seed from crosswalk
    xw = _load_crosswalk()
    if not xw.empty and "name" in xw.columns:
        for _, row in xw.iterrows():
            lookup[int(row["mlb_id"])] = str(row["name"])

    # Overwrite with player_values (canonical names)
    if player_values is not None and not player_values.empty:
        name_col = "Player_Name" if "Player_Name" in player_values.columns else "name"
        subset = player_values.dropna(subset=["mlb_id"]).drop_duplicates("mlb_id")
        for _, row in subset.iterrows():
            lookup[int(row["mlb_id"])] = row[name_col]
    return lookup


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Prospect timeline
# ═══════════════════════════════════════════════════════════════════════════════

def _prospect_dollar_value(fv, rank) -> float:
    """Convert FV grade + top-100 rank into a dollar value.

    Delegates to ``value_determination.trade_value._prospect_dollar_value()``.
    """
    result = _vd_prospect_dollar_value(fv, rank)
    return result if result is not None else 0.0


def build_prospect_timeline(
    player_values: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build prospect-value entries from the raw prospect data.

    Returns DataFrame with columns:
        mlb_id, IDfg, name, year, value, value_type, label
    """
    if not PROSPECT_FILE.exists():
        logger.warning(f"Prospect file not found: {PROSPECT_FILE}")
        return pd.DataFrame()

    prospects = pd.read_csv(PROSPECT_FILE)
    logger.info(f"Loaded {len(prospects)} prospect entries")

    # Extract mlb_id from prospect URL  (...-123456)
    prospects["mlb_id"] = (
        prospects["prospect_url"]
        .str.extract(r"(\d{5,7})$")
        .astype("float")
    )

    idfg_to_mlb = _build_idfg_to_mlb(player_values)
    mlb_to_name = _build_mlb_to_name(player_values)
    # Reverse lookup: mlb_id → IDfg
    mlb_to_idfg: dict[int, int] = {v: k for k, v in idfg_to_mlb.items()}

    rows: list[dict] = []
    for _, p in prospects[prospects["mlb_id"].notna()].iterrows():
        mid = int(p["mlb_id"])
        idfg = mlb_to_idfg.get(mid)
        name = mlb_to_name.get(mid, p.get("name", ""))

        year = int(p["year"])
        fv   = p.get("grade_overall")
        rank = p.get("top_100")

        value = _prospect_dollar_value(fv, rank)
        if value <= 0:
            continue

        fv_str = str(fv).replace(".0", "") if pd.notna(fv) else "?"
        label_parts = [f"FV {fv_str}"]
        if pd.notna(rank):
            label_parts.append(f"#{int(rank)}")
        label = ", ".join(label_parts)

        rows.append({
            "mlb_id": mid,
            "IDfg": idfg if idfg is not None else pd.NA,
            "name": name,
            # Prospect snapshots are anchored to the prior season's end so the
            # FV step-up lines up with the value level that season reaches.
            "date": f"{year - 1}-10-01",
            "year": year,
            "value": round(value),
            "value_type": "prospect",
            "transaction_type": pd.NA,
            "label": label,
            "years_control": pd.NA,
            "projected_war": pd.NA,
            "projected_salary": pd.NA,
            "war_per_year": pd.NA,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("value", ascending=False).drop_duplicates(["mlb_id", "year"])
    logger.info(
        f"Built {len(result)} prospect entries for "
        f"{result['mlb_id'].nunique() if not result.empty else 0} players"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MLB surplus timeline
# ═══════════════════════════════════════════════════════════════════════════════

def _load_surplus_files() -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    surplus_by_year: dict[int, pd.DataFrame] = {}
    surplus_eoy_by_year: dict[int, pd.DataFrame] = {}
    
    if not SURPLUS_DIR.exists():
        return surplus_by_year, surplus_eoy_by_year

    for path in sorted(SURPLUS_DIR.glob("surplus_*.csv")):
        try:
            parts = path.stem.split("_")
            yr = int(parts[1])
            is_eoy = len(parts) > 2 and parts[2] == "eoy"
            
            if is_eoy:
                surplus_eoy_by_year[yr] = pd.read_csv(path, low_memory=False)
                logger.info(f"  Loaded {path.name}: {len(surplus_eoy_by_year[yr])} players")
            else:
                surplus_by_year[yr] = pd.read_csv(path, low_memory=False)
                logger.info(f"  Loaded {path.name}: {len(surplus_by_year[yr])} players")
        except Exception as e:
            logger.warning(f"  Failed to load {path.name}: {e}")
            
    return surplus_by_year, surplus_eoy_by_year

    for path in sorted(SURPLUS_DIR.glob("surplus_*.csv")):
        try:
            parts = path.stem.split("_")
            yr = int(parts[1])
            is_eoy = len(parts) > 2 and parts[2] == "eoy"
            
            if is_eoy:
                surplus_eoy_by_year[yr] = pd.read_csv(path, low_memory=False)
                logger.info(f"  Loaded {path.name}: {len(surplus_eoy_by_year[yr])} players")
            else:
                surplus_by_year[yr] = pd.read_csv(path, low_memory=False)
                logger.info(f"  Loaded {path.name}: {len(surplus_by_year[yr])} players")
        except Exception as e:
            logger.warning(f"  Failed to load {path.name}: {e}")
            
    return surplus_by_year, surplus_eoy_by_year

    for path in sorted(SURPLUS_DIR.glob("surplus_*.csv")):
        try:
            yr = int(path.stem.split("_")[1])
            surplus_by_year[yr] = pd.read_csv(path, low_memory=False)
            logger.info(f"  Loaded {path.name}: {len(surplus_by_year[yr])} players")
        except Exception as e:
            logger.warning(f"  Failed to load {path.name}: {e}")
    return surplus_by_year


def build_mlb_timeline(
    player_values: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build MLB trade-value entries using the pre-computed trade_value from
    each snapshot year's surplus file.

    Supports both the new surplus format (with pre-computed ``trade_value``,
    ``years_of_control``, ``total_future_WAR``, ``total_future_salary``)
    and the legacy format (``WAR_YYYY`` / ``salary_YYYY`` columns that
    require re-computation).

    For the current year, appends entries from the value_determination
    pipeline (``player_values_complete.csv``) so the chart includes the
    most recent LSTM-based projections.

    Returns DataFrame: mlb_id, IDfg, name, year, value, value_type, label
    """
    surplus_by_year, surplus_eoy_by_year = _load_surplus_files()

    idfg_to_mlb = _build_idfg_to_mlb(player_values)
    mlb_to_name = _build_mlb_to_name(player_values)

    rows: list[dict] = []

    def _build_projection_overrides(sdf: pd.DataFrame) -> dict[int, tuple[float, float]]:
        """Build mlb_id → (projected_war, war_per_year) lookup from a surplus snapshot."""
        overrides: dict[int, tuple[float, float]] = {}
        has_trade_value = "trade_value" in sdf.columns

        for _, pr in sdf.iterrows():
            mlb_id = None
            mlbam = pr.get("mlbam_id")
            if pd.notna(mlbam):
                try:
                    mlb_id = int(mlbam)
                except (TypeError, ValueError):
                    mlb_id = None
            if mlb_id is None and pd.notna(pr.get("IDfg")):
                try:
                    mlb_id = idfg_to_mlb.get(int(pr["IDfg"]))
                except (TypeError, ValueError):
                    mlb_id = None
            if mlb_id is None:
                continue

            if has_trade_value:
                total_war = pr.get("total_future_WAR", 0) or 0
                yrs_ctrl = pr.get("years_of_control", 0) or 0
                war_per_yr = pr.get("war_per_year", np.nan)
                if pd.isna(war_per_yr):
                    war_per_yr = (total_war / yrs_ctrl) if yrs_ctrl > 0 else 0.0
            else:
                war_cols = [c for c in sdf.columns if re.match(r"^WAR_\d{4}$", c)]
                total_war = sum(float(pr.get(c, 0) or 0) for c in war_cols if pd.notna(pr.get(c)))
                yrs_ctrl = sum(1 for c in war_cols if pd.notna(pr.get(c)))
                war_per_yr = (total_war / yrs_ctrl) if yrs_ctrl > 0 else 0.0

            if pd.notna(total_war):
                overrides[mlb_id] = (float(total_war), float(war_per_yr))

        return overrides

    def _build_projection_overrides_from_player_values(target_year: int) -> dict[int, tuple[float, float]]:
        """Build mlb_id → (projected_war, war_per_year) from player_values for a given year."""
        overrides: dict[int, tuple[float, float]] = {}
        if player_values is None or player_values.empty or "Year" not in player_values.columns:
            return overrides

        year_rows = player_values[
            (player_values["Year"] == target_year)
            & player_values["mlb_id"].notna()
        ]
        if year_rows.empty:
            return overrides

        for _, row in year_rows.iterrows():
            try:
                mlb_id = int(row["mlb_id"])
            except (TypeError, ValueError):
                continue

            fut_war = row.get("total_future_war", np.nan)
            if pd.isna(fut_war):
                fut_war = row.get("total_future_WAR", np.nan)
            yrs = row.get("years_control", np.nan)
            wpy = row.get("war_per_year", np.nan)
            if pd.isna(wpy) and pd.notna(fut_war) and pd.notna(yrs) and float(yrs) > 0:
                wpy = float(fut_war) / float(yrs)

            if pd.notna(fut_war):
                overrides[mlb_id] = (float(fut_war), float(wpy) if pd.notna(wpy) else np.nan)

        return overrides

    def _process_surplus_df(
        sdf: pd.DataFrame,
        snap_year: int,
        date_str: str,
        year_value: int,
        projection_overrides: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        has_trade_value = "trade_value" in sdf.columns
        for _, pr in sdf.iterrows():
            mlb_id = None
            mlbam = pr.get("mlbam_id")
            if pd.notna(mlbam):
                try:
                    mlb_id = int(mlbam)
                except (TypeError, ValueError):
                    mlb_id = None
            if mlb_id is None and pd.notna(pr.get("IDfg")):
                try:
                    mlb_id = idfg_to_mlb.get(int(pr["IDfg"]))
                except (TypeError, ValueError):
                    mlb_id = None
            if mlb_id is None:
                continue

            name = mlb_to_name.get(mlb_id, pr.get("Name", ""))

            if has_trade_value:
                # New format: read pre-computed values directly
                trade_val = pr.get("trade_value", 0)
                if pd.isna(trade_val):
                    continue
                yrs_ctrl = pr.get("years_of_control", 0) or 0
                total_war = pr.get("total_future_WAR", 0) or 0
                total_salary = pr.get("total_future_salary", 0) or 0
            else:
                # Legacy format: recompute from WAR_YYYY / salary_YYYY columns
                war_cols = sorted([c for c in sdf.columns if c.startswith("WAR_")])
                sal_cols = sorted([c for c in sdf.columns if c.startswith("salary_") and c != "salary_source"])
                yrs_ctrl = pr.get("years_of_control", 0) or 0

                total_value  = 0.0
                total_salary = 0.0
                total_war    = 0.0

                for wc in war_cols:
                    proj_year = int(wc.split("_")[1])
                    war = pr.get(wc)
                    if pd.isna(war) or war <= 0:
                        continue
                    total_value += war_to_dollars(war, proj_year)
                    total_war += war

                for sc in sal_cols:
                    sal = pr.get(sc)
                    if pd.notna(sal):
                        total_salary += sal

                trade_val = total_value - total_salary

            if total_war <= 0 and trade_val == 0:
                continue

            war_per_yr = total_war / yrs_ctrl if yrs_ctrl > 0 else 0.0

            if projection_overrides and mlb_id in projection_overrides:
                total_war, war_per_yr = projection_overrides[mlb_id]

            label = (
                f"{int(yrs_ctrl)}yr control, {total_war:.1f} WAR"
                if yrs_ctrl > 0
                else f"{total_war:.1f} WAR projected"
            )

            rows.append({
                "mlb_id": mlb_id,
                "IDfg": int(pr["IDfg"]) if pd.notna(pr.get("IDfg")) else pd.NA,
                "name": name,
                "date": date_str,
                "year": year_value,
                "value": round(trade_val),
                "value_type": "mlb_surplus",
                "transaction_type": pd.NA,
                "label": label,
                "years_control": round(yrs_ctrl, 1),
                "projected_war": round(total_war, 1),
                "projected_salary": round(total_salary),
                "war_per_year": round(war_per_yr, 2),
            })

    for snap_year in sorted(surplus_by_year):
        _process_surplus_df(
            surplus_by_year[snap_year],
            snap_year,
            f"{snap_year}-04-01",
            snap_year,
        )

    # EOY rows use end-of-season value/control from explicit EOY files.
    # Note: surplus_YYYY_eoy.csv represents the end of the (YYYY - 1) season
    # because it uses cutoff_year = (YYYY - 1).
    for snap_year in sorted(surplus_eoy_by_year):
        eoy_year = snap_year - 1
        if eoy_year < Config.History.CUTOFF_START:
            continue
        _process_surplus_df(
            surplus_eoy_by_year[snap_year],
            eoy_year,
            f"{eoy_year}-10-01",
            eoy_year,
        )

    # ── Current year from value_determination ─────────────────────────────
    if player_values is not None and not player_values.empty:
        pv = player_values[
            (player_values["Year"] == CURRENT_YEAR)
            & player_values["mlb_id"].notna()
            & player_values["trade_value"].notna()
        ].copy()

        name_col = "Player_Name" if "Player_Name" in pv.columns else "name"
        for _, row in pv.iterrows():
            mlb_id = int(row["mlb_id"])
            idfg   = int(row["IDfg"])
            name   = row[name_col]
            tv     = row["trade_value"]
            yrs    = row.get("years_control", 0)
            if pd.isna(yrs):
                yrs = 0
            fut_war = row.get("total_future_war", 0)
            if pd.isna(fut_war):
                fut_war = 0
            cwar   = row.get("contract_war", 0)
            if pd.isna(cwar):
                cwar = 0
            # total_salary: prefer total_contract (always available from pipeline)
            total_sal = row.get("total_contract", 0)
            if pd.isna(total_sal) or total_sal == 0:
                # Fallback: sum individual salary_ columns if they exist
                sal_cols_pv = [c for c in pv.columns if c.startswith("salary_") and c != "salary_source"]
                total_sal = sum(row.get(sc, 0) or 0 for sc in sal_cols_pv if pd.notna(row.get(sc)))
            war_per_yr = fut_war / yrs if yrs > 0 else 0.0

            label = (
                f"{int(yrs)}yr control, {fut_war:.1f} WAR"
                if yrs > 0
                else f"{fut_war:.1f} WAR projected"
            )

            rows.append({
                "mlb_id": mlb_id,
                "IDfg": idfg,
                "name": name,
                "date": f"{CURRENT_YEAR}-03-01",
                "year": CURRENT_YEAR,
                "value": round(tv),
                "value_type": "mlb_surplus",
                "transaction_type": pd.NA,
                "label": label,
                "years_control": round(yrs, 1),
                "projected_war": round(fut_war, 1),
                "projected_salary": round(total_sal),
                "war_per_year": round(war_per_yr, 2),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = (
            result.sort_values("value", ascending=False)
            .drop_duplicates(["mlb_id", "date"], keep="first")
        )
    logger.info(
        f"Built {len(result)} MLB timeline entries for "
        f"{result['mlb_id'].nunique() if not result.empty else 0} players"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Spotrac transaction entries
# ═══════════════════════════════════════════════════════════════════════════════

# _normalise_name is imported from core.name_utils as name_key (see top of file)

# Transaction types where the player has no team control → value = 0
_ZERO_VALUE_TYPES = {"elected_fa", "released", "designated", "non_tendered", "opt_out", "option_declined"}


def _build_name_to_mlbid(player_values: pd.DataFrame) -> dict[str, int]:
    """player_name → mlb_id lookup from player_values + crosswalk."""
    lookup: dict[str, int] = {}

    # Seed from crosswalk (broader coverage)
    xw = _load_crosswalk()
    if not xw.empty and "name" in xw.columns:
        for _, row in xw.iterrows():
            key = _normalise_name(str(row["name"]))
            if key:
                lookup[key] = int(row["mlb_id"])

    # Overwrite with player_values (canonical names + IDs)
    if player_values is not None and not player_values.empty:
        name_col = "Player_Name" if "Player_Name" in player_values.columns else "name"
        subset = player_values.dropna(subset=["mlb_id"]).drop_duplicates("mlb_id")
        for _, row in subset.iterrows():
            key = _normalise_name(row[name_col])
            lookup[key] = int(row["mlb_id"])
    return lookup


def _reclassify_arbitration(txn: pd.DataFrame) -> pd.DataFrame:
    """Reclassify ``fa_signing`` rows that are actually arbitration deals.

    Spotrac labels arb-avoiding deals as ``fa_signing`` but the description
    contains 'avoiding arbitration'.  Similarly, 'settling in arbitration'
    indicates an arb settlement.  We reclassify them so the trade-value
    history can distinguish true FA signings from arb settlements.
    """
    arb_mask = (
        txn["transaction_type"].isin(["fa_signing", "signing"])
        & txn["description"].str.contains(
            "avoiding arbitration|settling in arbitration",
            case=False, na=False, regex=True,
        )
    )
    txn.loc[arb_mask, "transaction_type"] = "arbitration"
    n = int(arb_mask.sum())
    if n:
        logger.info(f"  Reclassified {n} fa_signing/signing → arbitration")
    return txn


# Pre-arb contract detection: 1-year contracts at or near league minimum
_PRE_ARB_SALARY_RE = re.compile(
    r"Signed a 1 year \$[\d,]+(?:k|K)?\s+contract with",
    re.IGNORECASE,
)


def _reclassify_pre_arb(txn: pd.DataFrame, name_to_mlbid: dict[str, int], cots_lookup: dict[tuple[int, int], dict]) -> pd.DataFrame:
    """Reclassify ``fa_signing`` rows that are actually pre-arb contracts.

    Pre-arb contracts are 1-year deals at or near the league minimum salary.
    We detect them by: (a) the contract is 1 year, (b) the salary is near
    the historical minimum for that year, and (c) the event is not already
    classified as arbitration, extension, etc.
    
    We ALSO use Cot's service time: if service time < 3.0, they are pre-arb.
    If service time < 6.0, they are arbitration (unless they have an extension, but this affects fa_signing).
    """
    min_sal = Config.History.HISTORICAL_MIN_SALARY
    mask = txn["transaction_type"].isin(["fa_signing", "signing"])
    count_salary = 0
    count_cots = 0

    for idx in txn.index[mask]:
        desc = str(txn.at[idx, "description"])
        ev_date = txn.at[idx, "date"]
        player_name = str(txn.at[idx, "player_name"])

        # Determine year
        try:
            ts = pd.Timestamp(ev_date)
            yr = ts.year
            if ts.month >= 10:
                yr += 1
        except Exception:
            continue
            
        mlb_id = name_to_mlbid.get(_normalise_name(player_name))
        cots_svc = None
        if mlb_id:
            cots_data = cots_lookup.get((mlb_id, yr))
            if cots_data and 'service_time' in cots_data:
                cots_svc = cots_data['service_time']

        if cots_svc is not None:
            if cots_svc < 3.0:
                txn.at[idx, "transaction_type"] = "pre_arb"
                count_cots += 1
                continue
            elif cots_svc < 6.0:
                txn.at[idx, "transaction_type"] = "arbitration"
                count_cots += 1
                continue

        # Must match "Signed a 1 year $XXX,XXX contract"
        if not _PRE_ARB_SALARY_RE.search(desc):
            continue

        # Extract salary from description (handle "k" suffix)
        sal_match = re.search(r"\$([\d,]+(?:\.\d+)?)(k|K)?\s", desc)
        if not sal_match:
            continue
        sal_str = sal_match.group(1).replace(",", "")
        try:
            salary = float(sal_str)
        except ValueError:
            continue
        if sal_match.group(2):     # "k" suffix → thousands
            salary *= 1_000

        # Pre-arb threshold: within 50% above the league minimum
        threshold = min_sal.get(yr, 800_000) * 1.5
        if salary <= threshold:
            txn.at[idx, "transaction_type"] = "pre_arb"
            count_salary += 1

    if count_cots or count_salary:
        logger.info(f"  Reclassified fa_signing → {count_cots} by Cot's service time, {count_salary} by salary")
    return txn


def _reclassify_minor_league(txn: pd.DataFrame) -> pd.DataFrame:
    """Reclassify ``fa_signing`` rows that are minor-league signings.

    Spotrac descriptions containing 'minor league contract' are not real
    FA signings — they are minor-league depth signings or MiLB free-agent
    pickups.  Reclassify so they can be filtered from the trade-value chart.
    """
    mask = (
        txn["transaction_type"].isin(["fa_signing", "signing"])
        & txn["description"].str.contains(
            "minor league", case=False, na=False,
        )
    )
    txn.loc[mask, "transaction_type"] = "minor_league_signing"
    n = int(mask.sum())
    if n:
        logger.info(f"  Reclassified {n} fa_signing/signing → minor_league_signing")
    return txn


def _reclassify_initial_signing(txn: pd.DataFrame) -> pd.DataFrame:
    """Reclassify ``fa_signing`` rows that are draft or IFA signings.

    Spotrac records draft bonus signings and international free-agent
    signings as plain "Signed a contract with [team]" (no years or salary).
    Because the old ``fa_signing`` regex matched ``contract with``, these
    were incorrectly classified.

    Detection:  fa_signing + no dollar amount in description + no explicit
    year/salary contract terms → almost certainly a draft or IFA signing,
    not a true MLB free-agent deal.
    """
    mask = (
        txn["transaction_type"].isin(["fa_signing", "signing"])
        & ~txn["description"].str.contains(r"\$", na=False, regex=True)
        & ~txn["description"].str.contains("minor league", case=False, na=False)
    )
    txn.loc[mask, "transaction_type"] = "initial_signing"
    n = int(mask.sum())
    if n:
        logger.info(f"  Reclassified {n} fa_signing/signing → initial_signing")
    return txn


def _consolidate_qualifying_offers(txn: pd.DataFrame) -> pd.DataFrame:
    """Consolidate QO extended + QO declined into a single free-agency event.

    When a player receives a Qualifying Offer:
      1. Team extends QO (transaction_type='other', desc contains 'extended...Qualifying Offer')
      2. Player declines QO (transaction_type='other', desc contains 'Declined...Qualifying Offer')

    We keep only the QO-extension event and reclassify it as ``elected_fa``
    (free agency entry point, value=0).  The declined event is dropped.
    """
    qo_extended_mask = txn["description"].str.contains(
        r"extended.*Qualifying Offer", case=False, na=False, regex=True,
    )
    qo_declined_mask = txn["description"].str.contains(
        r"Declined.*Qualifying Offer", case=False, na=False, regex=True,
    )

    # Reclassify extensions as elected_fa
    txn.loc[qo_extended_mask, "transaction_type"] = "elected_fa"
    n_ext = int(qo_extended_mask.sum())

    # Drop declined rows (redundant info)
    n_dec = int(qo_declined_mask.sum())
    txn = txn[~qo_declined_mask].copy()

    if n_ext or n_dec:
        logger.info(
            f"  QO consolidation: {n_ext} extended → elected_fa, "
            f"{n_dec} declined dropped"
        )
    return txn


def _build_value_lookup(
    value_timeline: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    if value_timeline is None or value_timeline.empty:
        return {}
    # Keep both prospect entries (always) and mlb_surplus entries (always)
    # Don't filter out prospects — they're needed for drafted transaction lookups
    valid = value_timeline[
        (value_timeline["value_type"] == "prospect") |
        (value_timeline["value_type"] == "mlb_surplus")
    ].copy()
    valid["_dt"] = pd.to_datetime(valid["date"], errors="coerce")
    valid = valid.dropna(subset=["_dt"]).sort_values("_dt")
    return {mlb_id: grp for mlb_id, grp in valid.groupby("mlb_id")}

def _build_cots_lookup() -> dict[tuple[int, int], dict]:
    """Build a lookup of Cot's data by (mlb_id, year) for exact-year fallback."""
    lookup: dict[tuple[int, int], dict] = {}
    cots_dir = Config.Paths.DATA_DIR / 'salary' / 'by_year'
    if not cots_dir.exists(): return lookup
    xw = _load_crosswalk()
    if xw.empty: return lookup
    name_to_mlbid = {}
    for _, row in xw.iterrows():
        name = str(row.get('name', '')).lower().strip()
        mlb_id = row.get('mlb_id')
        if name and pd.notna(mlb_id): name_to_mlbid[name] = int(mlb_id)
    
    for cots_file in sorted(cots_dir.glob('*.csv')):
        try:
            year = int(cots_file.stem)
            cots = pd.read_csv(cots_file, low_memory=False)
            if 'player' not in cots.columns or 'years_of_control' not in cots.columns: continue
            for _, row in cots.iterrows():
                player_name = str(row.get('player', '')).lower().strip()
                yoc = row.get('years_of_control')
                svc = row.get('service_time')
                if not player_name or pd.isna(yoc): continue
                mlb_id = name_to_mlbid.get(player_name)
                if mlb_id:
                    lookup[(int(mlb_id), year)] = {
                        'years_control': int(yoc),
                        'service_time': float(svc) if pd.notna(svc) else 0.0
                    }
        except: pass
    return lookup

def _lookup_value(
    mlb_id: int,
    ev_date: pd.Timestamp,
    value_map: dict[int, pd.DataFrame],
    cots_lookup: dict[tuple[int, int], dict] | None = None,
) -> dict | None:
    if mlb_id not in value_map: return None
    df = value_map[mlb_id]
    prior = df[(df["_dt"] <= ev_date) & (df["value_type"] != "prospect")]
    if prior.empty:
        prior_p = df[df["_dt"] <= ev_date]
        prior = prior_p if not prior_p.empty else df
    row = prior.iloc[-1] if not prior.empty else df.iloc[0]
    yr = ev_date.year
    entry = {
        "value": row["value"],
        "years_control": row.get("years_control"),
        "projected_war": row.get("projected_war"),
        "projected_salary": row.get("projected_salary"),
        "war_per_year": row.get("war_per_year"),
        "_dt": row["_dt"]
    }
    if cots_lookup and (mlb_id, yr) in cots_lookup:
        cots_entry = cots_lookup[(mlb_id, yr)]
        if pd.isna(entry.get('years_control')) or entry.get('years_control') == 0:
            entry['years_control'] = cots_entry['years_control']
    return entry


def _lookup_prospect_value(
    mlb_id: int,
    ev_date: pd.Timestamp,
    value_map: dict[int, pd.DataFrame],
) -> tuple[dict | None, bool]:
    """For drafted/initial-signing transactions, prefer prospect values.
    
    Returns (entry_dict, is_prospect_flag).
    is_prospect_flag is True if the value came from a prospect row.
    """
    if mlb_id not in value_map: return None, False
    df = value_map[mlb_id]
    
    # First, look for prospect values
    prior_prospect = df[(df["_dt"] <= ev_date) & (df["value_type"] == "prospect")]
    if not prior_prospect.empty:
        row = prior_prospect.iloc[-1]
        return {
            "value": row["value"],
            "years_control": row.get("years_control"),
            "projected_war": row.get("projected_war"),
            "projected_salary": row.get("projected_salary"),
            "war_per_year": row.get("war_per_year"),
            "_dt": row["_dt"]
        }, True  # is_prospect = True
    
    # Fall back to mlb_surplus if no prospect
    prior_mlb = df[(df["_dt"] <= ev_date) & (df["value_type"] != "prospect")]
    if prior_mlb.empty:
        prior_mlb = df[df["_dt"] <= ev_date]
    
    if prior_mlb.empty:
        return None, False
    
    row = prior_mlb.iloc[-1]
    return {
        "value": row["value"],
        "years_control": row.get("years_control"),
        "projected_war": row.get("projected_war"),
        "projected_salary": row.get("projected_salary"),
        "war_per_year": row.get("war_per_year"),
        "_dt": row["_dt"]
    }, False  # is_prospect = False


def build_transaction_entries(
    player_values: pd.DataFrame | None,
    value_timeline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a timeline entry for significant Spotrac transactions.

    Pre-arb and arbitration contracts are excluded (they don't change a
    player's value — still under team control for the same duration).
    Qualifying Offer pairs are consolidated into a single ``elected_fa``
    event with value=0.

    Each kept transaction carries:
      - the actual date (``date`` column, YYYY-MM-DD)
      - the player's trade-value from the nearest annual snapshot
      - the Spotrac ``transaction_type``
      - the Spotrac description as the ``label``
      - hover-card metadata: years_control, projected_war,
        projected_salary, war_per_year

    For zero-value events (released, DFA, elected FA / QO) the value is 0,
    and metadata fields are null (free agents have no control/projections).
    """
    if not SPOTRAC_TRANSACTIONS_FILE.exists():
        logger.warning(f"Spotrac file not found: {SPOTRAC_TRANSACTIONS_FILE}")
        return pd.DataFrame()

    txn = pd.read_csv(SPOTRAC_TRANSACTIONS_FILE, parse_dates=["date"])
    logger.info(f"Loaded {len(txn)} Spotrac transactions")

    # ── Player-ID lookups (needed early for Cot's matching) ───────────────
    name_to_mlbid = _build_name_to_mlbid(player_values)
    idfg_to_mlb   = _build_idfg_to_mlb(player_values)
    mlb_to_name   = _build_mlb_to_name(player_values)
    mlb_to_idfg: dict[int, int] = {v: k for k, v in idfg_to_mlb.items()}

    # ── Load Cot's data as fallback for exact-year lookups ────────────────
    cots_lookup = _build_cots_lookup()

    # ── Reclassify arb and pre-arb signings ───────────────────────────────
    txn = _reclassify_arbitration(txn)
    txn = _reclassify_pre_arb(txn, name_to_mlbid, cots_lookup)

    # ── Reclassify minor-league and draft/IFA signings ────────────────────
    txn = _reclassify_minor_league(txn)
    txn = _reclassify_initial_signing(txn)

    # ── Consolidate Qualifying Offer pairs ────────────────────────────────
    txn = _consolidate_qualifying_offers(txn)

    # ── Filter out non-value-changing contract types ────────────────────
    # These are team-controlled contracts that don't change the player's
    # years of control or value — just salary negotiations or initial signings.
    # Option exercises are handled by analyze_contract_options downstream.
    _SKIP_TYPES = {
        "pre_arb", "arbitration", "option_exercised",
        "minor_league_signing", "initial_signing",
    }
    pre_filter = len(txn)
    txn = txn[~txn["transaction_type"].isin(_SKIP_TYPES)].copy()

    # Also filter "other" reclassified option exercises (before they're reclassified)
    _option_exercised_mask = (
        (txn["transaction_type"] == "other")
        & txn["description"].str.contains(
            r"exercised.*option|option\s+for",
            case=False, na=False, regex=True,
        )
    )
    txn = txn[~_option_exercised_mask].copy()
    pre_filter_after = pre_filter - len(txn)
    if pre_filter_after:
        logger.info(f"  Filtered out {pre_filter_after} pre-arb/arbitration/option_exercised entries")

    # ── Filter "other" transactions: keep only contract-affecting events ──
    # Many "other" entries are minor-league roster moves (optioned, recalled,
    # promoted, demoted, contract purchased) that don't affect trade value.
    # Keep only: non-tendered, club/player option exercised, opt-outs.
    _KEEP_OTHER_PATTERNS = re.compile(
        r"non.?tendered|option\s+for|player\s+option|termination\s+option|opted?\s+out",
        re.IGNORECASE,
    )
    if "other" in txn["transaction_type"].values:
        other_mask = txn["transaction_type"] == "other"
        other_relevant = txn.loc[other_mask, "description"].str.contains(
            _KEEP_OTHER_PATTERNS, na=False,
        )
        drop_mask = other_mask & ~other_relevant
        n_other_dropped = drop_mask.sum()
        txn = txn[~drop_mask].copy()
        if n_other_dropped:
            logger.info(f"  Filtered out {n_other_dropped} non-relevant 'other' transactions")

    # ── Value lookup from existing surplus + prospect timeline ────────────
    value_map = _build_value_lookup(value_timeline)

    rows: list[dict] = []

    for _, player_txns in txn.groupby("spotrac_id"):
        player_name = player_txns.iloc[0]["player_name"]
        mlb_id = name_to_mlbid.get(_normalise_name(player_name))
        if mlb_id is None:
            continue

        idfg = mlb_to_idfg.get(mlb_id)
        name = mlb_to_name.get(mlb_id, player_name)

        for _, ev in player_txns.sort_values("date").iterrows():
            ev_type = ev["transaction_type"]
            ev_date = ev["date"]
            desc = str(ev.get("description", "") or "")

            # Reclassify remaining "other" transactions to proper types
            if ev_type == "other":
                desc_lower = desc.lower()
                if "non" in desc_lower and "tender" in desc_lower:
                    ev_type = "non_tendered"
                elif "termination" in desc_lower or "opted out" in desc_lower or "opt out" in desc_lower:
                    ev_type = "opt_out"

            if pd.isna(ev_date):
                continue
            if isinstance(ev_date, str):
                try:
                    ev_date = pd.Timestamp(ev_date)
                except Exception:
                    continue

            ev_year = ev_date.year
            date_str = ev_date.strftime("%Y-%m-%d")

            # After October, the offseason effectively belongs to the next
            # season — use next year's projections so that new signings
            # reflect the updated contract/control situation.
            lookup_year = ev_year + 1 if ev_date.month >= 10 else ev_year

            # Skip events before our data range
            if ev_year < Config.History.CUTOFF_START:
                continue

            # ── Determine value + metadata ────────────────────────────────
            if ev_type in _ZERO_VALUE_TYPES:
                value = 0
                vtype = "free_agent"
                yrs_ctrl = pd.NA
                proj_war = pd.NA
                proj_sal = pd.NA
                war_yr   = pd.NA
            elif ev_type in ("extension", "fa_signing", "signing"):
                # For extensions and major FA deals, look forward to the NEXT surplus snapshot
                # so the value jump occurs ON the transaction date.
                entry = None
                if mlb_id in value_map:
                    df = value_map[mlb_id]
                    post = df[(df["_dt"] > ev_date) & (df["value_type"] != "prospect")]
                    if not post.empty:
                        r = post.iloc[0]
                        yr = ev_date.year
                        entry = {
                            "value": r["value"],
                            "years_control": r.get("years_control"),
                            "projected_war": r.get("projected_war"),
                            "projected_salary": r.get("projected_salary"),
                            "war_per_year": r.get("war_per_year"),
                            "_dt": r["_dt"]
                        }
                if entry is None:
                    entry = _lookup_value(mlb_id, ev_date, value_map, cots_lookup)
                
                if entry is None:
                    continue
                value    = entry["value"]
                vtype    = "mlb_surplus"
                yrs_ctrl = entry.get("years_control")
                proj_war = entry.get("projected_war")
                proj_sal = entry.get("projected_salary")
                war_yr   = entry.get("war_per_year")
            elif ev_type in ("drafted", "initial_signing"):
                # For drafted/signed prospects, prefer prospect values
                result = _lookup_prospect_value(mlb_id, ev_date, value_map)
                if result is None or result[0] is None:
                    continue
                entry, is_prospect = result
                value    = entry["value"]
                vtype    = "prospect" if is_prospect else "mlb_surplus"
                yrs_ctrl = entry.get("years_control")
                proj_war = entry.get("projected_war")
                proj_sal = entry.get("projected_salary")
                war_yr   = entry.get("war_per_year")
            else:
                entry = _lookup_value(mlb_id, ev_date, value_map, cots_lookup)
                if entry is None:
                    # No snapshot data at all — skip this transaction
                    continue
                value    = entry["value"]
                vtype    = "mlb_surplus"
                yrs_ctrl = entry.get("years_control")
                proj_war = entry.get("projected_war")
                proj_sal = entry.get("projected_salary")
                war_yr   = entry.get("war_per_year")

            # ── Skip fa_signing that are actually arb (player still
            #    under team control for >1 year with a 1-year deal) ────────
            if (
                ev_type == "fa_signing"
                and yrs_ctrl is not None
                and not pd.isna(yrs_ctrl)
                and yrs_ctrl > 1
                and "1 year" in desc.lower()
            ):
                continue

            # ── Label from Spotrac description ────────────────────────────
            label = ev.get("description", "")
            if pd.isna(label) or not str(label).strip():
                label = ev_type.replace("_", " ").title()

            rows.append({
                "mlb_id": mlb_id,
                "IDfg": idfg if idfg is not None else pd.NA,
                "name": name,
                "date": date_str,
                "year": ev_year,
                "value": round(value),
                "value_type": vtype,
                "transaction_type": "fa_signing" if ev_type == "signing" else ev_type,
                "label": str(label),
                "years_control": yrs_ctrl,
                "projected_war": proj_war,
                "projected_salary": proj_sal,
                "war_per_year": war_yr,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.drop_duplicates(["mlb_id", "date", "transaction_type"])
    n_by_type = (
        result["transaction_type"].value_counts().to_dict()
        if not result.empty else {}
    )
    logger.info(
        f"Built {len(result)} transaction entries for "
        f"{result['mlb_id'].nunique() if not result.empty else 0} players"
    )
    for tt, cnt in sorted(n_by_type.items(), key=lambda x: -x[1]):
        logger.info(f"    {tt}: {cnt}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Monthly interpolation for annual snapshots
# ═══════════════════════════════════════════════════════════════════════════════

def _interpolate_monthly(combined: pd.DataFrame) -> pd.DataFrame:
    """Insert monthly interpolated points between consecutive anchor points.

    Every entry (snapshot or transaction) is an anchor.  Transactions act
    as breakpoints — interpolation restarts from the transaction value.

    Interpolation rules
    ───────────────────
    • **In-season** (April – September): linearly interpolate between
      adjacent anchors, proportional to *in-season days* elapsed.
    • **Offseason** (October – March): hold flat at the last in-season
      interpolated value.  Value changes are frozen until the next April.
    • Offseason *transactions* still cause jumps — they are anchor points
      whose value is authoritative at that date.
    • When the destination anchor is a **transaction** (extension, signing,
      etc.), we do NOT interpolate toward it — the pre-transaction months
      hold flat at the starting anchor's value because the transaction was
      not predictable.  The transaction creates a discontinuous jump at its
      date.
    """
    if combined.empty:
        return combined

    combined = combined.copy()
    combined["_dt"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values(["mlb_id", "_dt"]).reset_index(drop=True)

    # ── helper: count in-season (Apr–Sep) days in [start, end) ────────────
    def _season_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
        if start >= end:
            return 0
        total = 0
        for y in range(start.year, end.year + 1):
            s0 = pd.Timestamp(y, 4, 1)
            s1 = pd.Timestamp(y, 10, 1)  # Oct 1 = first day after season
            lo = max(start, s0)
            hi = min(end, s1)
            if lo < hi:
                total += (hi - lo).days
        return total

    def _is_transaction(row) -> bool:
        """Check if a row is a transaction (has a non-null transaction_type)."""
        tt = row.get("transaction_type")
        return pd.notna(tt) and tt != ""

    interpolated_rows: list[dict] = []

    for mlb_id, grp in combined.groupby("mlb_id"):
        grp = grp.sort_values("_dt").reset_index(drop=True)
        if len(grp) < 2:
            continue

        for idx in range(len(grp) - 1):
            row_a = grp.iloc[idx]
            row_b = grp.iloc[idx + 1]
            dt_a = row_a["_dt"]
            dt_b = row_b["_dt"]
            if pd.isna(dt_a) or pd.isna(dt_b):
                continue

            total_days = (dt_b - dt_a).days
            if total_days <= 32:       # less than ~1 month apart → skip
                continue

            val_a = row_a["value"]
            val_b = row_b["value"]

            # If the destination is a transaction, don't interpolate toward it
            # unless it is elected free agency, which should taper to zero.
            b_is_txn = _is_transaction(row_b)
            b_is_free_agent_exit = str(row_b.get("transaction_type", "")) == "elected_fa"

            total_season = _season_days(dt_a, dt_b)

            # Walk month-by-month (1st of each month, exclusive of endpoints)
            current = dt_a + pd.DateOffset(months=1)
            current = pd.Timestamp(current.year, current.month, 1)

            while current < dt_b:
                if b_is_txn and not b_is_free_agent_exit:
                    # Hold flat — we can't predict the transaction.
                    frac = 0.0
                elif b_is_free_agent_exit:
                    # Free agency should decay smoothly from the last controlled
                    # anchor to zero at the FA event.
                    frac = min(((current - dt_a).days / total_days), 1.0)
                elif total_season > 0:
                    elapsed = _season_days(dt_a, current)
                    frac = min(elapsed / total_season, 1.0)
                else:
                    # Both anchors fall in the offseason with no in-season
                    # time between them → hold flat at val_a.
                    frac = 0.0

                interp_val = val_a + frac * (val_b - val_a)
                # Only the trade value is interpolated. Metadata fields (years_control,
                # projected_war, projected_salary, war_per_year, label) stay constant
                # from the source anchor (row_a) throughout the interpolation period.
                # However, if both anchors have projecting WAR, we can interpolate it smoothly.
                # If only one has it, we maintain the source block (row_a) strictly 
                # so future projections don't magically bleed backwards.
                
                a_war_ok = pd.notna(row_a.get("projected_war"))
                b_war_ok = pd.notna(row_b.get("projected_war"))
                if a_war_ok and b_war_ok:
                    interp_proj_war = float(row_a["projected_war"]) + frac * (float(row_b["projected_war"]) - float(row_a["projected_war"]))
                elif a_war_ok:
                    interp_proj_war = float(row_a["projected_war"])
                else:
                    interp_proj_war = pd.NA

                a_wpy_ok = pd.notna(row_a.get("war_per_year"))
                b_wpy_ok = pd.notna(row_b.get("war_per_year"))
                if a_wpy_ok and b_wpy_ok:
                    interp_war_per_year = float(row_a["war_per_year"]) + frac * (float(row_b["war_per_year"]) - float(row_a["war_per_year"]))
                elif a_wpy_ok:
                    interp_war_per_year = float(row_a["war_per_year"])
                else:
                    interp_war_per_year = pd.NA

                interpolated_rows.append({
                    "mlb_id": mlb_id,
                    "IDfg": row_a.get("IDfg") if pd.notna(row_a.get("IDfg")) else row_b.get("IDfg"),
                    "name": row_a["name"],
                    "date": current.strftime("%Y-%m-%d"),
                    "year": current.year,
                    "value": round(interp_val),
                    "value_type": "mlb_surplus",
                    "transaction_type": pd.NA,
                    "label": row_a.get("label", ""),
                    "years_control": row_a.get("years_control"),
                    "projected_war": round(interp_proj_war, 1) if pd.notna(interp_proj_war) else pd.NA,
                    "projected_salary": row_a.get("projected_salary"),
                    "war_per_year": round(interp_war_per_year, 2) if pd.notna(interp_war_per_year) else pd.NA,
                })

                current += pd.DateOffset(months=1)
                current = pd.Timestamp(current.year, current.month, 1)

    combined = combined.drop(columns=["_dt"])

    if interpolated_rows:
        interp_df = pd.DataFrame(interpolated_rows)
        logger.info(f"  Interpolated {len(interp_df)} monthly points for "
                    f"{interp_df['mlb_id'].nunique()} players")
        result = pd.concat([combined, interp_df], ignore_index=True)
    else:
        result = combined

    return result.sort_values(["mlb_id", "date"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_timeline() -> pd.DataFrame:
    """
    Generate (or regenerate) the complete trade-value history CSV.

    Combines prospect, MLB surplus, and Spotrac transaction data.  Each row
    carries a ``date`` (YYYY-MM-DD) and an optional ``transaction_type``
    from Spotrac.  Multiple entries per player per year are allowed (e.g.
    a pre-season projection and a mid-season trade).

    Dedup within each source:
        - Prospect entries:  (mlb_id, year)  → keep highest value
        - MLB surplus:       (mlb_id, year)  → keep highest value
        - Transactions:      (mlb_id, date, transaction_type)

    Across sources no dedup is applied — prospect (Jan), surplus (Mar),
    and transaction entries coexist for the same player/year.
    """
    logger.info("=" * 60)
    logger.info("Historical Values — Timeline Generator")
    logger.info("=" * 60)

    # Load player_values for crosswalks / current-year overlay
    pv: pd.DataFrame | None = None
    if PLAYER_VALUES_FILE.exists():
        pv = pd.read_csv(PLAYER_VALUES_FILE, low_memory=False)
        logger.info(f"Loaded player_values_complete.csv: {len(pv)} rows")
    else:
        logger.warning(
            f"player_values_complete.csv not found at {PLAYER_VALUES_FILE}  — "
            "current-year overlay disabled"
        )

    # ── Build the three sub-timelines ─────────────────────────────────────
    prospect_tl = build_prospect_timeline(pv)
    mlb_tl      = build_mlb_timeline(pv)

    # Combine prospect + MLB.
    pre_combined = pd.concat([prospect_tl, mlb_tl], ignore_index=True)

    # If prospect and surplus land on the same date, keep the prospect row so
    # the FV label remains attached to the value anchor.
    _TYPE_PRI = {"prospect": 0, "mlb_surplus": 1}
    pre_combined["_p"] = pre_combined["value_type"].map(_TYPE_PRI).fillna(2)
    pre_combined = (
        pre_combined.sort_values("_p")
        .drop_duplicates(["mlb_id", "date"], keep="first")
        .drop(columns=["_p"])
    )

    # Build transaction entries from Spotrac (uses snapshot values for lookup)
    txn_entries = build_transaction_entries(pv, pre_combined)

    # Final combine — no cross-source dedup (dates are different)
    combined = pd.concat([pre_combined, txn_entries], ignore_index=True)
    combined = combined.sort_values(["mlb_id", "date"]).reset_index(drop=True)

    # ── Suppress April 1 dots when a nearby transaction already exists ────
    # If a player has a transaction in the off-season / spring of the same
    # year (Jan 1 – Mar 31) or within 14 days after April 1, suppress the
    # April 1 surplus dot to avoid redundant/crowded points.
    if not combined.empty:
        combined["_dt"] = pd.to_datetime(combined["date"], errors="coerce")
        drop_idx: set[int] = set()
        for mid, grp in combined.groupby("mlb_id"):
            for yr in grp["year"].unique():
                yr_data = grp[grp["year"] == yr]
                # Find April 1 surplus dots (no transaction_type)
                april1 = yr_data[
                    (yr_data["date"] == f"{yr}-04-01")
                    & yr_data["transaction_type"].isna()
                ]
                if april1.empty:
                    continue
                # Check for transactions within 90 days before or 14 days
                # after April 1 of the same year
                txns = yr_data[yr_data["transaction_type"].notna()]
                if txns.empty:
                    continue
                april1_dt = pd.Timestamp(f"{yr}-04-01")
                for _, t in txns.iterrows():
                    t_dt = t["_dt"]
                    if pd.isna(t_dt):
                        continue
                    delta = (april1_dt - t_dt).days
                    if -14 <= delta <= 90:
                        drop_idx.update(april1.index)
                        break
        if drop_idx:
            combined = combined.drop(index=drop_idx).reset_index(drop=True)
            logger.info(f"  Suppressed {len(drop_idx)} April-1 dots near transactions")
        combined = combined.drop(columns=["_dt"])

    # ── Monthly interpolation of annual snapshots ─────────────────────────
    # Hide QO/elected FA labels before interpolation so they don't bleed into intermediate months
    if not combined.empty:
        qo_mask = combined["transaction_type"] == "elected_fa"
        combined.loc[qo_mask, "label"] = pd.NA

    # Annual snapshots jump from one year to the next.  Insert monthly
    # interpolated points between consecutive annual snapshots so the
    # chart shows smooth transitions instead of abrupt jumps.
    combined = _interpolate_monthly(combined)

    # Now that interpolation is done to 0, remove the elected_fa marker so it doesn't render a dot
    if not combined.empty:
        qo_mask_post = combined["transaction_type"] == "elected_fa"
        combined.loc[qo_mask_post, "transaction_type"] = pd.NA

    # ── Enhance labels with FV for prospect years ─────────────────────────
    # If a player was a prospect in a given year, prepend their FV grade
    # (e.g. "FV 50") to ALL rows for that year (transactions, interpolations)
    if not combined.empty:
        prospect_lookup = {}
        prospect_rows = combined[combined["value_type"] == "prospect"]
        for _, r in prospect_rows.iterrows():
            if pd.notna(r["mlb_id"]) and pd.notna(r["year"]) and pd.notna(r["label"]):
                prospect_lookup[(int(r["mlb_id"]), int(r["year"]))] = str(r["label"])
                
        def _enhance_with_fv(row):
            mid = row.get("mlb_id")
            yr = row.get("year")
            if pd.isna(mid) or pd.isna(yr):
                return row["label"]
            
            fv_label = prospect_lookup.get((int(mid), int(yr)))
            if not fv_label:
                return row["label"]
                
            # If it's already the prospect row, no change needed
            if row.get("value_type") == "prospect":
                return row["label"]
                
            curr_label = row.get("label")
            if pd.isna(curr_label) or not str(curr_label).strip():
                return fv_label
                
            # If the current label already contains the FV label, don't duplicate
            if fv_label in str(curr_label):
                return curr_label
                
            return f"[{fv_label}] {curr_label}"

        combined["label"] = combined.apply(_enhance_with_fv, axis=1)

    # ── Write output ─────────────────────────────────────────────────────
    # Ensure column order
    col_order = [
        "mlb_id", "IDfg", "name", "date", "year",
        "value", "value_type", "transaction_type", "label",
        "years_control", "projected_war", "projected_salary", "war_per_year",
    ]
    for c in col_order:
        if c not in combined.columns:
            combined[c] = pd.NA
    combined = combined[col_order]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    n_prospect = int((combined["value_type"] == "prospect").sum())
    n_mlb      = int((combined["value_type"] == "mlb_surplus").sum())
    n_fa       = int((combined["value_type"] == "free_agent").sum())
    n_txn      = int(combined["transaction_type"].notna().sum())
    n_players  = combined["mlb_id"].nunique() if not combined.empty else 0

    logger.info(f"Wrote {len(combined)} entries for {n_players} players → {OUTPUT_FILE}")
    logger.info(
        f"  Prospect: {n_prospect}  |  MLB surplus: {n_mlb}  |  "
        f"FA/zero: {n_fa}  |  Transactions: {n_txn}"
    )
    return combined
