"""Audit: Find 40-man roster players missing projections."""
import pandas as pd

roster = pd.read_csv("data/active_roster/current_rosters.csv")
roster_with_fg = roster.dropna(subset=["fg_id"]).copy()
roster_with_fg["fg_str"] = roster_with_fg["fg_id"].astype(int).astype(str)

bat = pd.read_csv("data/generated/pipeline/batter_predictions.csv")
pit = pd.read_csv("data/generated/pipeline/pitcher_predictions.csv")
bsr = pd.read_csv("data/generated/pipeline/baserunning_predictions.csv")
fld = pd.read_csv("data/generated/pipeline/fielding_predictions.csv")

all_predicted = set(bat["IDfg"].astype(str).unique()) | set(pit["IDfg"].astype(str).unique())
bat_predicted = set(bat["IDfg"].astype(str).unique())
pit_predicted = set(pit["IDfg"].astype(str).unique())
bsr_predicted = set(bsr["IDfg"].astype(str).unique())
fld_predicted = set(fld["IDfg"].astype(str).unique())

missing_mask = ~roster_with_fg["fg_str"].isin(all_predicted)
missing_rows = roster_with_fg[missing_mask].copy()

# Check MLB experience
mlb_bat = pd.read_csv("data/historic_mlb/mlb_batting_data_1950_2025.csv", usecols=["IDfg", "Season", "PA"])
mlb_pit = pd.read_csv("data/historic_mlb/mlb_pitching_data_1950_2025.csv", usecols=["IDfg", "Season", "IP"])

bat_ids = set(mlb_bat["IDfg"].astype(str).unique())
pit_ids = set(mlb_pit["IDfg"].astype(str).unique())
mlb_ids = bat_ids | pit_ids

missing_with_mlb = missing_rows[missing_rows["fg_str"].isin(mlb_ids)]
missing_no_mlb = missing_rows[~missing_rows["fg_str"].isin(mlb_ids)]

print(f"40-man roster with fg_id: {len(roster_with_fg)}")
print(f"Unique predicted players (bat+pit): {len(all_predicted)}")
print(f"  Batter predictions: {len(bat_predicted)}")
print(f"  Pitcher predictions: {len(pit_predicted)}")
print(f"  Baserunning predictions: {len(bsr_predicted)}")
print(f"  Fielding predictions: {len(fld_predicted)}")
print(f"\nRoster MISSING projections: {len(missing_rows)}")
print(f"  With MLB experience: {len(missing_with_mlb)}")
print(f"  No MLB experience (prospects): {len(missing_no_mlb)}")

# For those with MLB experience, check last year and volume
results = []
for _, row in missing_with_mlb.iterrows():
    fid = row["fg_str"]
    bat_data = mlb_bat[mlb_bat["IDfg"].astype(str) == fid]
    pit_data = mlb_pit[mlb_pit["IDfg"].astype(str) == fid]
    last_bat = int(bat_data["Season"].max()) if len(bat_data) > 0 else None
    last_pit = int(pit_data["Season"].max()) if len(pit_data) > 0 else None
    last_yr = max(y for y in [last_bat, last_pit] if y is not None)
    total_pa = int(bat_data["PA"].sum()) if len(bat_data) > 0 else 0
    total_ip = float(pit_data["IP"].sum()) if len(pit_data) > 0 else 0
    # PA/IP in 2025
    pa_2025 = int(bat_data[bat_data["Season"] == 2025]["PA"].sum())
    ip_2025 = float(pit_data[pit_data["Season"] == 2025]["IP"].sum())
    pa_2024 = int(bat_data[bat_data["Season"] == 2024]["PA"].sum())
    ip_2024 = float(pit_data[pit_data["Season"] == 2024]["IP"].sum())
    results.append({
        "name": row["player_name"], "pos": row["position_name"],
        "last_yr": last_yr, "last_bat": last_bat, "last_pit": last_pit,
        "total_pa": total_pa, "total_ip": total_ip,
        "pa_2024": pa_2024, "ip_2024": ip_2024,
        "pa_2025": pa_2025, "ip_2025": ip_2025,
    })

df_r = pd.DataFrame(results).sort_values("last_yr", ascending=False)

print(f"\n{'='*100}")
print(f"Missing players WITH MLB experience ({len(df_r)} total)")
print(f"{'='*100}")
for _, r in df_r.iterrows():
    print(
        f"  {r['name']:30s} {r['pos']:15s} last={r['last_yr']}  "
        f"PA24={r['pa_2024']:>4}  IP24={r['ip_2024']:>6.1f}  "
        f"PA25={r['pa_2025']:>4}  IP25={r['ip_2025']:>6.1f}  "
        f"career_PA={r['total_pa']:>5}  career_IP={r['total_ip']:>7.1f}"
    )

print(f"\n--- Count by last MLB year ---")
print(df_r["last_yr"].value_counts().sort_index(ascending=False).to_string())

# Also check: players IN batter predictions but NOT in fielding/baserunning
print(f"\n{'='*100}")
print(f"Batter predictions but missing from fielding: {len(bat_predicted - fld_predicted)}")
print(f"Batter predictions but missing from baserunning: {len(bat_predicted - bsr_predicted)}")
print(f"Fielding but missing from batter: {len(fld_predicted - bat_predicted)}")
print(f"Baserunning but missing from batter: {len(bsr_predicted - bat_predicted)}")

# Check example: Vaughn Grissom
print(f"\n{'='*100}")
print("EXAMPLE: Vaughn Grissom")
grissom_bat = mlb_bat[mlb_bat["IDfg"].astype(str) == "25430"]
grissom_pit = mlb_pit[mlb_pit["IDfg"].astype(str) == "25430"]
print(f"Batting history: {grissom_bat[['Season','PA']].to_string(index=False)}")
print(f"In batter predictions: {'25430' in bat_predicted}")
print(f"In pitcher predictions: {'25430' in pit_predicted}")
