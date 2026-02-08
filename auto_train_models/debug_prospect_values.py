from value_determination.config import Config
from value_determination.generate_prospect_histories import calculate_prospect_value

print("FV_BASE_VALUES:")
for k, v in sorted(Config.Prospects.FV_BASE_VALUES.items()):
    print(f"  {k}: ${v:,}")

print("\nTest calculations:")
val_top100 = calculate_prospect_value(55, 43, 2025)
print(f"55 FV, rank 43: ${val_top100:,.0f}")

val_no_rank = calculate_prospect_value(55, None, 2025)
print(f"55 FV, no rank: ${val_no_rank:,.0f}")

base = Config.Prospects.FV_BASE_VALUES[55]
adj43 = Config.Prospects.calculate_rank_adjustment(43)
print(f"\nBase 55 FV value: ${base:,}")
print(f"Rank 43 adjustment: {adj43:.4f}")
print(f"Expected (55M * {adj43:.4f}): ${base * adj43:,.0f}")
