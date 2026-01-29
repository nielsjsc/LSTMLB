"""
Diagnostic script to identify players on rosters but missing from projections.
"""

import pandas as pd
from pathlib import Path

# Load data
data_dir = Path(r'C:\Users\nlsc\Desktop\LSTMLB\data')

print("Loading data...")
roster = pd.read_csv(data_dir / 'active_roster' / 'current_rosters.csv')
batter_preds = pd.read_csv(data_dir / 'generated' / 'pipeline' / 'batter_predictions.csv')
pitcher_preds = pd.read_csv(data_dir / 'generated' / 'pipeline' / 'pitcher_predictions.csv')

# Filter to 2026 projections
batter_2026 = batter_preds[batter_preds['Year'] == 2026]
pitcher_2026 = pitcher_preds[pitcher_preds['Year'] == 2026]

print(f"\nRoster players: {len(roster)}")
print(f"2026 batter projections: {len(batter_2026)}")
print(f"2026 pitcher projections: {len(pitcher_2026)}")

# Get roster with valid FG IDs
roster_with_id = roster[roster['fg_id'].notna() & (roster['fg_id'] != -1)]
roster_no_id = roster[roster['fg_id'].isna() | (roster['fg_id'] == -1)]

print(f"\nRoster players WITH FG ID: {len(roster_with_id)}")
print(f"Roster players WITHOUT FG ID: {len(roster_no_id)}")

# Get all projected player IDs
all_proj_ids = set(batter_2026['IDfg'].unique()) | set(pitcher_2026['IDfg'].unique())
roster_ids = set(roster_with_id['fg_id'].astype(int))

print(f"\nUnique projected player IDs (2026): {len(all_proj_ids)}")
print(f"Unique roster player IDs: {len(roster_ids)}")

# Find mismatches
on_roster_not_projected = roster_ids - all_proj_ids
projected_not_on_roster = all_proj_ids - roster_ids

print(f"\n{'='*80}")
print(f"ROSTER PLAYERS WITHOUT 2026 PROJECTIONS: {len(on_roster_not_projected)}")
print(f"{'='*80}")

if on_roster_not_projected:
    missing = roster_with_id[roster_with_id['fg_id'].isin(on_roster_not_projected)].copy()
    missing = missing.sort_values(['team_name', 'position_type'])
    
    # Show by team
    team_counts = missing['team_name'].value_counts().sort_index()
    print("\nBy Team:")
    for team, count in team_counts.items():
        print(f"  {team}: {count} players")
    
    print("\n\nDetailed List (first 50):")
    print(missing[['player_name', 'team_name', 'position_name', 'position_type', 'fg_id']].head(50).to_string(index=False))
    
    # Export full list
    output_file = Path(r'C:\Users\nlsc\Desktop\LSTMLB\data\generated\playing_time\missing_from_projections.csv')
    missing.to_csv(output_file, index=False)
    print(f"\n\nFull list saved to: {output_file}")

print(f"\n{'='*80}")
print(f"PROJECTED PLAYERS NOT ON CURRENT ROSTER: {len(projected_not_on_roster)}")
print(f"{'='*80}")
print("(These are likely retired/free agent/traded players)")

# Check a few examples
if projected_not_on_roster:
    print("\nExamples (first 20 IDs):")
    for fg_id in list(projected_not_on_roster)[:20]:
        # Try to find in batter predictions
        batter_match = batter_2026[batter_2026['IDfg'] == fg_id]
        if not batter_match.empty:
            print(f"  IDfg {fg_id}: {batter_match.iloc[0]['Name']} (Batter)")
        else:
            pitcher_match = pitcher_2026[pitcher_2026['IDfg'] == fg_id]
            if not pitcher_match.empty:
                print(f"  IDfg {fg_id}: {pitcher_match.iloc[0]['Name']} (Pitcher)")

print(f"\n{'='*80}")
print("ROSTER PLAYERS WITHOUT FG ID (cannot be matched)")
print(f"{'='*80}")

if len(roster_no_id) > 0:
    print(f"\nTotal: {len(roster_no_id)}")
    no_id_counts = roster_no_id['team_name'].value_counts().sort_index()
    print("\nBy Team:")
    for team, count in no_id_counts.items():
        print(f"  {team}: {count} players")
    
    print("\n\nDetailed List (first 30):")
    print(roster_no_id[['player_name', 'team_name', 'position_name', 'position_type']].head(30).to_string(index=False))
    
    # Export
    output_file = Path(r'C:\Users\nlsc\Desktop\LSTMLB\data\generated\playing_time\missing_fg_id.csv')
    roster_no_id.to_csv(output_file, index=False)
    print(f"\n\nFull list saved to: {output_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total roster spots: {len(roster)}")
print(f"  - With FG ID and 2026 projection: {len(roster_ids & all_proj_ids)}")
print(f"  - With FG ID but NO 2026 projection: {len(on_roster_not_projected)}")
print(f"  - Without FG ID: {len(roster_no_id)}")
print(f"\nPlayers allocated playing time: Should be ~{len(roster_ids & all_proj_ids)}")
