"""Statistical analysis of pitcher injury recurrence patterns."""
import pandas as pd
import numpy as np

df = pd.read_csv("data/injury/fangraphs_injury_data.csv")
pit = pd.read_csv("data/historic_mlb/mlb_pitching_data_1950_2025.csv", low_memory=False)

# ── Build per-pitcher-season injury summary ──────────────────────────────
inj = df[df["position"].isin(["SP", "RP", "SP/RP", "RP/SP"])].copy()
inj["retro"] = pd.to_datetime(inj["il_retro_date"], errors="coerce")
inj["ret"] = pd.to_datetime(inj["return_date"], errors="coerce")
inj["days_out"] = (inj["ret"] - inj["retro"]).dt.days

by_ps = (
    inj.groupby(["season", "fg_id"])
    .agg(total_days=("days_out", "sum"), n_stints=("days_out", "count"))
    .reset_index()
)
by_ps["total_days"] = by_ps["total_days"].fillna(0)

# ── Join with pitching stats ─────────────────────────────────────────────
pit_recent = pit[(pit["Season"] >= 2021) & (pit["Season"] <= 2025)][
    ["IDfg", "Season", "IP", "GS", "G", "Age"]
].copy()
pit_recent = pit_recent.rename(columns={"IDfg": "fg_id", "Season": "season"})

merged = pit_recent.merge(by_ps, on=["season", "fg_id"], how="left")
merged["total_days"] = merged["total_days"].fillna(0)
merged["n_stints"] = merged["n_stints"].fillna(0)
merged["was_injured"] = merged["total_days"] > 0
merged["gs_rate"] = merged["GS"] / merged["G"].clip(lower=1)
merged["is_sp"] = merged["gs_rate"] >= 0.8

# ── Build consecutive-year pairs ─────────────────────────────────────────
pairs = merged.merge(merged, on="fg_id", suffixes=("_y1", "_y2"))
pairs = pairs[pairs["season_y2"] == pairs["season_y1"] + 1]

print("=" * 60)
print("INJURY RECURRENCE ANALYSIS (2021-2025)")
print("=" * 60)
print(f"Pitcher season-pairs: {len(pairs)}")
print()

# ── Q1: Does prior injury predict next-year IP & injury? ────────────────
print("=== PRIOR YEAR HEALTHY vs INJURED ===")
for label, group in pairs.groupby("was_injured_y1"):
    tag = "INJURED" if label else "HEALTHY"
    n = len(group)
    ip_mean = group["IP_y2"].mean()
    ip_med = group["IP_y2"].median()
    inj_rate = group["was_injured_y2"].mean()
    days_mean = group["total_days_y2"].mean()
    print(f"  Prior year {tag} (n={n}):")
    print(f"    Next-year IP:          mean={ip_mean:.1f}  median={ip_med:.1f}")
    print(f"    Next-year injury rate: {inj_rate:.1%}")
    print(f"    Next-year days lost:   {days_mean:.1f}")
    print()

# ── Q2: By severity tier ────────────────────────────────────────────────
def sev(d):
    if d == 0:
        return "0_healthy"
    if d <= 30:
        return "1_minor (1-30d)"
    if d <= 90:
        return "2_moderate (31-90d)"
    return "3_severe (>90d)"

pairs["sev_y1"] = pairs["total_days_y1"].apply(sev)

print("=== BY PRIOR YEAR INJURY SEVERITY ===")
for sev_label in sorted(pairs["sev_y1"].unique()):
    group = pairs[pairs["sev_y1"] == sev_label]
    n = len(group)
    ip_mean = group["IP_y2"].mean()
    inj_rate = group["was_injured_y2"].mean()
    days_mean = group["total_days_y2"].mean()
    print(f"  {sev_label} (n={n}): IP={ip_mean:.1f}, inj_rate={inj_rate:.1%}, days_lost={days_mean:.1f}")
print()

# ── Q3: SP vs RP difference ─────────────────────────────────────────────
print("=== SP vs RP INJURY PATTERNS ===")
for role, role_grp in pairs.groupby("is_sp_y1"):
    tag = "SP" if role else "RP"
    n = len(role_grp)
    base_inj = role_grp["was_injured_y1"].mean()
    next_inj = role_grp["was_injured_y2"].mean()
    ip_healthy = role_grp[~role_grp["was_injured_y1"]]["IP_y2"].mean()
    ip_injured = role_grp[role_grp["was_injured_y1"]]["IP_y2"].mean()
    print(f"  {tag} (n={n}):")
    print(f"    Base injury rate: {base_inj:.1%} → Next year: {next_inj:.1%}")
    print(f"    Next IP if prev healthy: {ip_healthy:.1f} vs injured: {ip_injured:.1f}")
    print()

# ── Q4: 2+ consecutive injured years ────────────────────────────────────
print("=== MULTI-YEAR INJURY HISTORY ===")
# Look at Y3 outcomes based on Y1+Y2 injury status
triples = merged.merge(merged, on="fg_id", suffixes=("_y1", "_y2"))
triples = triples[triples["season_y2"] == triples["season_y1"] + 1]
triples = triples.merge(
    merged.rename(columns={c: c + "_y3" for c in merged.columns if c != "fg_id"}),
    on="fg_id",
)
triples = triples[triples["season_y3"] == triples["season_y2"] + 1]

both_healthy = triples[~triples["was_injured_y1"] & ~triples["was_injured_y2"]]
one_injured = triples[triples["was_injured_y1"] ^ triples["was_injured_y2"]]
both_injured = triples[triples["was_injured_y1"] & triples["was_injured_y2"]]

for label, grp in [
    ("0 of 2 years injured", both_healthy),
    ("1 of 2 years injured", one_injured),
    ("2 of 2 years injured", both_injured),
]:
    if len(grp) == 0:
        continue
    n = len(grp)
    ip3 = grp["IP_y3"].mean()
    inj3 = grp["was_injured_y3"].mean()
    days3 = grp["total_days_y3"].mean()
    print(f"  {label} (n={n}): Y3 IP={ip3:.1f}, inj_rate={inj3:.1%}, days_lost={days3:.1f}")
print()

# ── Q5: Specific injury type recurrence ─────────────────────────────────
print("=== INJURY TYPE → DAYS LOST ===")
inj_types = inj.dropna(subset=["days_out"]).copy()
inj_types = inj_types[inj_types["days_out"] > 0]
type_stats = (
    inj_types.groupby("injury_surgery")["days_out"]
    .agg(["mean", "median", "count"])
    .sort_values("count", ascending=False)
    .head(20)
)
for itype, row in type_stats.iterrows():
    print(f"  {itype:35s}  mean={row['mean']:5.0f}d  median={row['median']:5.0f}d  n={row['count']:4.0f}")

# ── Q6: Age effect on injury rate ────────────────────────────────────────
print()
print("=== AGE vs INJURY RATE (pitchers) ===")
age_inj = merged.groupby("Age").agg(
    n=("was_injured", "count"),
    inj_rate=("was_injured", "mean"),
    mean_days=("total_days", "mean"),
).reset_index()
age_inj = age_inj[(age_inj["Age"] >= 21) & (age_inj["Age"] <= 40) & (age_inj["n"] >= 20)]
for _, row in age_inj.iterrows():
    bar = "#" * int(row["inj_rate"] * 50)
    print(f"  Age {int(row['Age']):2d}: {row['inj_rate']:.1%} ({row['mean_days']:.0f}d avg) n={int(row['n']):4d}  {bar}")
