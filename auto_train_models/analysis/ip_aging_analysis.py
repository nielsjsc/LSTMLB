"""IP aging curve analysis using delta method."""
import pandas as pd
import numpy as np

pit = pd.read_csv("data/historic_mlb/mlb_pitching_data_1950_2025.csv", low_memory=False)

def analyze_role(df, role_name, gs_min, gs_max, ip_min):
    sub = df[(df["Season"] >= 2014) & (df["Season"] != 2020)].copy()
    sub["gs_rate"] = sub["GS"] / sub["G"].clip(lower=1)
    sub = sub[(sub["gs_rate"] >= gs_min) & (sub["gs_rate"] < gs_max) & (sub["IP"] >= ip_min)]

    pairs = sub.merge(sub, on="IDfg", suffixes=("_y1", "_y2"))
    pairs = pairs[pairs["Season_y2"] == pairs["Season_y1"] + 1]
    pairs = pairs[~((pairs["Season_y1"] == 2019) | (pairs["Season_y2"] == 2020))]
    pairs["ip_delta"] = pairs["IP_y2"] - pairs["IP_y1"]
    pairs["ip_pct_change"] = pairs["ip_delta"] / pairs["IP_y1"]
    pairs["age_mid"] = (pairs["Age_y1"] + pairs["Age_y2"]) / 2
    pairs["age_bucket"] = pd.cut(
        pairs["age_mid"],
        bins=[20, 25, 28, 31, 34, 37, 42],
        labels=["21-25", "26-28", "29-31", "32-34", "35-37", "38+"],
    )

    print(f"=== {role_name} IP AGING (delta method, 2014-2025 ex-2020) ===")
    agg = pairs.groupby("age_bucket", observed=True).agg(
        n=("ip_delta", "count"),
        mean_delta=("ip_delta", "mean"),
        median_delta=("ip_delta", "median"),
        mean_pct=("ip_pct_change", "mean"),
        mean_ip_y1=("IP_y1", "mean"),
        mean_ip_y2=("IP_y2", "mean"),
    )
    for bucket, row in agg.iterrows():
        d = row["mean_delta"]
        p = row["mean_pct"]
        print(
            f"  {bucket}: IP Δ = {d:+.1f} ({p:+.1%}), "
            f"n={int(row['n'])}, "
            f"avg IP: {row['mean_ip_y1']:.0f} → {row['mean_ip_y2']:.0f}"
        )
    print()
    return pairs


sp_pairs = analyze_role(pit, "SP", 0.8, 1.01, 40)
rp_pairs = analyze_role(pit, "RP", 0.0, 0.5, 20)

# ── SP IP/GS aging (innings per start) ──────────────────────────────────
print("=== SP INNINGS PER START BY AGE ===")
sps = pit[(pit["Season"] >= 2014) & (pit["Season"] != 2020)].copy()
sps["gs_rate"] = sps["GS"] / sps["G"].clip(lower=1)
sps = sps[(sps["gs_rate"] >= 0.8) & (sps["IP"] >= 40)]
sps["ip_per_gs"] = sps["IP"] / sps["GS"]
agg = sps.groupby("Age")["ip_per_gs"].agg(["mean", "median", "count"])
agg = agg[(agg.index >= 21) & (agg.index <= 40) & (agg["count"] >= 10)]
for age, row in agg.iterrows():
    bar = "#" * int(row["mean"] * 8)
    print(f"  Age {age:2.0f}: {row['mean']:.2f} IP/GS (n={int(row['count']):3d})  {bar}")

# ── RP innings per appearance by age ────────────────────────────────────
print()
print("=== RP INNINGS PER APPEARANCE BY AGE ===")
rps = pit[(pit["Season"] >= 2014) & (pit["Season"] != 2020)].copy()
rps["gs_rate"] = rps["GS"] / rps["G"].clip(lower=1)
rps = rps[(rps["gs_rate"] < 0.5) & (rps["IP"] >= 20)]
rps["ip_per_g"] = rps["IP"] / rps["G"]
agg = rps.groupby("Age")["ip_per_g"].agg(["mean", "median", "count"])
agg = agg[(agg.index >= 21) & (agg.index <= 40) & (agg["count"] >= 10)]
for age, row in agg.iterrows():
    bar = "#" * int(row["mean"] * 40)
    print(f"  Age {age:2.0f}: {row['mean']:.2f} IP/G (n={int(row['count']):3d})  {bar}")

# ── Year-to-year correlation by component ───────────────────────────────
print()
print("=== YEAR-TO-YEAR CORRELATIONS (SP, IP>=40) ===")
for col in ["IP", "GS", "G"]:
    r = sp_pairs[f"{col}_y1"].corr(sp_pairs[f"{col}_y2"])
    print(f"  {col}: r = {r:.3f}")

# IP/GS ratio correlation
sp_pairs["ipgs_y1"] = sp_pairs["IP_y1"] / sp_pairs["GS_y1"]
sp_pairs["ipgs_y2"] = sp_pairs["IP_y2"] / sp_pairs["GS_y2"]
r = sp_pairs["ipgs_y1"].corr(sp_pairs["ipgs_y2"])
print(f"  IP/GS: r = {r:.3f}")

print()
print("=== YEAR-TO-YEAR CORRELATIONS (RP, IP>=20) ===")
for col in ["IP", "G"]:
    r = rp_pairs[f"{col}_y1"].corr(rp_pairs[f"{col}_y2"])
    print(f"  {col}: r = {r:.3f}")
rp_pairs["ipg_y1"] = rp_pairs["IP_y1"] / rp_pairs["G_y1"]
rp_pairs["ipg_y2"] = rp_pairs["IP_y2"] / rp_pairs["G_y2"]
r = rp_pairs["ipg_y1"].corr(rp_pairs["ipg_y2"])
print(f"  IP/G: r = {r:.3f}")
