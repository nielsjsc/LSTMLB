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

_AUTO_TRAIN = Path(__file__).resolve().parents[1]
if str(_AUTO_TRAIN) not in sys.path:
    sys.path.insert(0, str(_AUTO_TRAIN))

from historical_values.config import (
    Config, logger,
    SURPLUS_DIR, OUTPUT_FILE,
    PROSPECT_FILE, CROSSWALK_FILE,
    PLAYER_VALUES_FILE, SPOTRAC_TRANSACTIONS_FILE,
)
from historical_values.war import war_to_dollars

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
            "date": f"{year}-01-15",
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

def _load_surplus_files() -> dict[int, pd.DataFrame]:
    """Load all ``surplus_YYYY.csv`` from the surplus directory."""
    surplus_by_year: dict[int, pd.DataFrame] = {}
    if not SURPLUS_DIR.exists():
        return surplus_by_year

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
    surplus_by_year = _load_surplus_files()

    idfg_to_mlb = _build_idfg_to_mlb(player_values)
    mlb_to_name = _build_mlb_to_name(player_values)

    rows: list[dict] = []

    # ── Historical years ─────────────────────────────────────────────────
    for snap_year, sdf in surplus_by_year.items():
        has_trade_value = "trade_value" in sdf.columns

        for _, pr in sdf.iterrows():
            idfg = int(pr["IDfg"])
            mlbam = int(pr["mlbam_id"]) if pd.notna(pr.get("mlbam_id")) else None

            mlb_id = mlbam
            if mlb_id is None:
                mlb_id = idfg_to_mlb.get(idfg)
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

            label = (
                f"{int(yrs_ctrl)}yr control, {total_war:.1f} WAR"
                if yrs_ctrl > 0
                else f"{total_war:.1f} WAR projected"
            )

            rows.append({
                "mlb_id": mlb_id,
                "IDfg": idfg,
                "name": name,
                "date": f"{snap_year}-04-01",
                "year": snap_year,
                "value": round(trade_val),
                "value_type": "mlb_surplus",
                "transaction_type": pd.NA,
                "label": label,
                "years_control": round(yrs_ctrl, 1),
                "projected_war": round(total_war, 1),
                "projected_salary": round(total_salary),
                "war_per_year": round(war_per_yr, 2),
            })

    # ── Current year from value_determination ─────────────────────────────
    if player_values is not None and not player_values.empty:
        pv = player_values[
            (player_values["Year"] == Config.CURRENT_YEAR)
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
                "date": f"{Config.CURRENT_YEAR}-03-01",
                "year": Config.CURRENT_YEAR,
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
            .drop_duplicates(["mlb_id", "year"], keep="first")
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
_ZERO_VALUE_TYPES = {"elected_fa", "released", "designated"}


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
        (txn["transaction_type"] == "fa_signing")
        & txn["description"].str.contains(
            "avoiding arbitration|settling in arbitration",
            case=False, na=False, regex=True,
        )
    )
    txn.loc[arb_mask, "transaction_type"] = "arbitration"
    n = int(arb_mask.sum())
    if n:
        logger.info(f"  Reclassified {n} fa_signing → arbitration")
    return txn


# Pre-arb contract detection: 1-year contracts at or near league minimum
_PRE_ARB_SALARY_RE = re.compile(
    r"Signed a 1 year \$[\d,]+ contract with",
    re.IGNORECASE,
)


def _reclassify_pre_arb(txn: pd.DataFrame) -> pd.DataFrame:
    """Reclassify ``fa_signing`` rows that are actually pre-arb contracts.

    Pre-arb contracts are 1-year deals at or near the league minimum salary.
    We detect them by: (a) the contract is 1 year, (b) the salary is near
    the historical minimum for that year, and (c) the event is not already
    classified as arbitration, extension, etc.
    """
    min_sal = Config.HISTORICAL_MIN_SALARY
    mask = txn["transaction_type"] == "fa_signing"
    count = 0

    for idx in txn.index[mask]:
        desc = str(txn.at[idx, "description"])
        ev_date = txn.at[idx, "date"]

        # Must match "Signed a 1 year $XXX,XXX contract"
        if not _PRE_ARB_SALARY_RE.search(desc):
            continue

        # Extract salary from description
        sal_match = re.search(r"\$([\d,]+(?:\.\d+)?)\s", desc)
        if not sal_match:
            continue
        sal_str = sal_match.group(1).replace(",", "")
        try:
            salary = float(sal_str)
        except ValueError:
            continue

        # Determine year
        try:
            yr = pd.Timestamp(ev_date).year
        except Exception:
            continue

        # Pre-arb threshold: within 50% above the league minimum
        threshold = min_sal.get(yr, 800_000) * 1.5
        if salary <= threshold:
            txn.at[idx, "transaction_type"] = "pre_arb"
            count += 1

    if count:
        logger.info(f"  Reclassified {count} fa_signing → pre_arb")
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
) -> dict[tuple[int, int], dict]:
    """(mlb_id, year) → {value, years_control, projected_war, projected_salary, war_per_year}.

    When both surplus and prospect exist for the same key, surplus wins.
    """
    lookup: dict[tuple[int, int], dict] = {}
    if value_timeline is None or value_timeline.empty:
        return lookup
    for _, r in value_timeline.iterrows():
        key = (int(r["mlb_id"]), int(r["year"]))
        entry = {
            "value": r["value"],
            "years_control": r.get("years_control"),
            "projected_war": r.get("projected_war"),
            "projected_salary": r.get("projected_salary"),
            "war_per_year": r.get("war_per_year"),
        }
        # mlb_surplus overwrites prospect
        if key not in lookup or r["value_type"] == "mlb_surplus":
            lookup[key] = entry
    return lookup


def _lookup_value(
    mlb_id: int,
    ev_year: int,
    value_map: dict[tuple[int, int], dict],
) -> dict | None:
    """Find the best value entry for *mlb_id* near *ev_year*."""
    if (mlb_id, ev_year) in value_map:
        return value_map[(mlb_id, ev_year)]
    # Off-season signing may precede the next snapshot
    if (mlb_id, ev_year + 1) in value_map:
        return value_map[(mlb_id, ev_year + 1)]
    if (mlb_id, ev_year - 1) in value_map:
        return value_map[(mlb_id, ev_year - 1)]
    return None


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

    # ── Reclassify arb and pre-arb signings ───────────────────────────────
    txn = _reclassify_arbitration(txn)
    txn = _reclassify_pre_arb(txn)

    # ── Consolidate Qualifying Offer pairs ────────────────────────────────
    txn = _consolidate_qualifying_offers(txn)

    # ── Filter out pre-arb and arb (they don't affect trade value) ────────
    # These are team-controlled contracts that don't change the player's
    # years of control or value — just salary negotiations.
    _SKIP_TYPES = {"pre_arb", "arbitration"}
    pre_filter = len(txn)
    txn = txn[~txn["transaction_type"].isin(_SKIP_TYPES)].copy()
    n_skipped = pre_filter - len(txn)
    if n_skipped:
        logger.info(f"  Filtered out {n_skipped} pre-arb/arbitration entries")

    # ── Player-ID lookups ─────────────────────────────────────────────────
    name_to_mlbid = _build_name_to_mlbid(player_values)
    idfg_to_mlb   = _build_idfg_to_mlb(player_values)
    mlb_to_name   = _build_mlb_to_name(player_values)
    mlb_to_idfg: dict[int, int] = {v: k for k, v in idfg_to_mlb.items()}

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
            if pd.isna(ev_date):
                continue
            if isinstance(ev_date, str):
                try:
                    ev_date = pd.Timestamp(ev_date)
                except Exception:
                    continue

            ev_year = ev_date.year
            date_str = ev_date.strftime("%Y-%m-%d")

            # Skip events before our data range
            if ev_year < Config.CUTOFF_START:
                continue

            # ── Determine value + metadata ────────────────────────────────
            if ev_type in _ZERO_VALUE_TYPES:
                value = 0
                vtype = "free_agent"
                yrs_ctrl = pd.NA
                proj_war = pd.NA
                proj_sal = pd.NA
                war_yr   = pd.NA
            else:
                entry = _lookup_value(mlb_id, ev_year, value_map)
                if entry is None:
                    # No snapshot data at all — skip this transaction
                    continue
                value    = entry["value"]
                vtype    = "mlb_surplus"
                yrs_ctrl = entry.get("years_control")
                proj_war = entry.get("projected_war")
                proj_sal = entry.get("projected_salary")
                war_yr   = entry.get("war_per_year")

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
                "transaction_type": ev_type,
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

    # Combine prospect + MLB (both are annual snapshots with different dates)
    pre_combined = pd.concat([prospect_tl, mlb_tl], ignore_index=True)

    # Within the snapshots: if a player has both prospect and surplus for the
    # same year, keep both (they have different dates: Jan vs Mar).
    # Dedup only exact duplicates on (mlb_id, date).
    _TYPE_PRI = {"mlb_surplus": 0, "prospect": 1}
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
