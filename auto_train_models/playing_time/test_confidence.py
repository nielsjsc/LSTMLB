#!/usr/bin/env python3
"""
Test script for confidence calculations.

This validates the confidence calculator and shows example outputs
for various player types (established stars, prospects, veterans, etc.)
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from playing_time.confidence import ConfidenceCalculator, get_confidence_tier

def main():
    print("=" * 80)
    print("CONFIDENCE CALCULATOR TEST")
    print("=" * 80)
    
    # Initialize calculator
    calc = ConfidenceCalculator()
    
    # Test players representing different archetypes
    # IDfgs verified from predictions file
    test_players = [
        # (IDfg, Name, Age, Type, Description)
        (15640, "Aaron Judge", 34, "batter", "Established superstar, high sample"),
        (25764, "Bobby Witt Jr.", 26, "batter", "Young star, former prospect"),
        (28163, "Junior Caminero", 22, "batter", "Top prospect, limited MLB sample"),
        (29712, "Colson Montgomery", 24, "batter", "Prospect, very limited sample"),
        (10155, "Mike Trout", 35, "batter", "Veteran star"),
        (19755, "Shohei Ohtani", 31, "batter", "Superstar DH"),
        (11579, "Bryce Harper", 33, "batter", "Established star"),
    ]
    
    print("\n" + "-" * 80)
    print(f"{'Player':<25} {'Type':<8} {'Age':<4} {'Sample':<8} {'Stat':<6} {'Pros':<6} {'Cons':<6} {'Total':<6} {'Tier':<12}")
    print("-" * 80)
    
    for idfg, name, age, model_type, description in test_players:
        conf = calc.calculate_confidence(
            player_id=idfg,
            player_name=name,
            player_age=age,
            model_type=model_type,
            projection_year=2026
        )
        
        tier = get_confidence_tier(conf.combined)
        
        print(f"{name:<25} {model_type:<8} {age:<4} {conf.sample_size:<8.0f} "
              f"{conf.statistical:<6.2f} {conf.prospect:<6.2f} {conf.consistency:<6.2f} "
              f"{conf.combined:<6.2f} {tier:<12}")
    
    print("-" * 80)
    
    # Show detailed breakdown for one player
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN: Bobby Witt Jr.")
    print("=" * 80)
    
    conf = calc.calculate_confidence(
        player_id=25764,
        player_name="Bobby Witt Jr.",
        player_age=26,
        model_type="batter",
        projection_year=2026
    )
    
    print(f"\nStatistical Confidence: {conf.statistical:.3f}")
    print(f"  - Sample size (PA over 5 years): {conf.sample_size:.0f}")
    print(f"  - Expected wOBA RMSE: {conf.expected_rmse_woba:.4f}")
    
    print(f"\nProspect Confidence: {conf.prospect:.3f}")
    if conf.prospect_rank:
        print(f"  - Best prospect rank: #{conf.prospect_rank}")
        print(f"  - Years since ranking: {conf.years_since_prospect}")
    else:
        print(f"  - No prospect ranking found")
    
    print(f"\nConsistency Confidence: {conf.consistency:.3f}")
    print(f"  - Based on PA variance over last 3 years")
    
    print(f"\nCombined Confidence: {conf.combined:.3f}")
    print(f"  - Tier: {get_confidence_tier(conf.combined)}")
    
    # Test batch calculation
    print("\n" + "=" * 80)
    print("BATCH CALCULATION TEST")
    print("=" * 80)
    
    batch_df = pd.DataFrame([
        {'IDfg': 15640, 'Name': 'Aaron Judge', 'Age': 34},
        {'IDfg': 25764, 'Name': 'Bobby Witt Jr.', 'Age': 26},
        {'IDfg': 28163, 'Name': 'Junior Caminero', 'Age': 22},
        {'IDfg': 29712, 'Name': 'Colson Montgomery', 'Age': 24},
    ])
    
    result_df = calc.calculate_batch_confidence(batch_df, 'batter', 2026)
    print("\nBatch results:")
    print(result_df[['Name', 'confidence', 'sample_size', 'prospect_rank']].to_string(index=False))


if __name__ == '__main__':
    main()
