# Value Determination Module
==========================

This module calculates player trade values based on WAR projections and contract status.
It provides the core valuation logic for the MLB Trade Simulator.

## Quick Start

```bash
# Run from project root
python -m auto_train_models.value_determination.main

# Or via pipeline
cd auto_train_models
python scripts/pipeline.py  # Select option 4
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE FLOW                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [1] LOAD DATA                                                       │
│      ├── Predictions (SP, RP, Batters)                               │
│      ├── Salary/Contract data                                        │
│      └── Prospect rankings                                           │
│              ↓                                                       │
│  [2] CALCULATE PITCHER WAR                                           │
│      └── FIP-based WAR with park factors                             │
│              ↓                                                       │
│  [3-6] PROCESS CONTRACTS                                             │
│      ├── Normalize status (Pre-Arb, ARB1-3, FA)                      │
│      ├── Generate year-by-year timeline                              │
│      └── Handle options (player, team, opt-out)                      │
│              ↓                                                       │
│  [7] CALCULATE VALUES                                                │
│      ├── WAR → Dollar value (convex power-law)                       │
│      ├── Contract value (arb percentages)                            │
│      └── Surplus = Base Value - Contract                             │
│              ↓                                                       │
│  [8-9] TRADE VALUE                                                   │
│      ├── Sum surplus through FA year                                 │
│      ├── Apply prospect adjustments (FV + experience weight)         │
│      └── Calculate ranking metrics                                   │
│              ↓                                                       │
│  [10] EXPORT                                                         │
│      └── player_values_complete.csv                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
value_determination/
├── config.py           # ★ CENTRAL CONFIG - edit settings here
├── main.py             # Pipeline entry point
├── calculate_war.py    # WAR calculations (batters + pitchers)
├── data_loader.py      # Load prediction/salary files
├── salary_processor.py # Clean and merge salary data
├── contract_processor.py # Contract timeline generation
├── value_calculator.py # WAR-to-dollars, surplus calc
├── trade_value.py      # Trade value + prospect adjustments
├── exporter.py         # Output to CSV
├── constants.py        # Legacy constants (use config.py instead)
└── README.md           # This file
```

## Configuration

All settings are centralized in `config.py`. Key configuration classes:

### Config.Paths
- `DATA_DIR`: Root data directory
- `PROSPECT_FILE`: Path to prospect rankings
- `ROSTER_FILE`: Path to team rosters

### Config.WAR
- `BALLPARK_FACTORS`: Park factor by team
- `LG_FIP`: League average FIP
- `DEFAULT_SP_IP`: Assumed IP for SP projections
- `POSITIONAL_ADJUSTMENTS`: Position value adjustments

### Config.ConvexModel
- `ALPHA_DEFAULT`: Base $/WAR coefficient ($8.59M)
- `BETA_DEFAULT`: Convexity exponent (1.323)
- `CALIBRATION_FILE`: Path to trade-analysis calibration JSON
- `load_calibration()`: Load (alpha, beta) from file or use defaults
- `calculate_value()`: Convert WAR to dollars using convex formula

### Config.Contracts
- `HISTORICAL_WAR_VALUE`: $/WAR by year
- `WAR_VALUE_DEFAULT`: Default $/WAR for future years
- `INFLATION_RATE`: Annual inflation (4%)
- `ARB_PERCENT`: Arbitration salary percentages

### Config.Prospects
- `FV_BASE_VALUES`: Dollar value per FV grade
- `EXPERIENCE_THRESHOLD_GAMES`: When prospect value diminishes
- `calculate_prospect_weight()`: Experience-based weighting

## Key Calculations

### WAR Calculation (Pitchers)
```
FIP Runs = (LG_FIP - park_adj_FIP) / 9 * IP
Replacement Runs = 20.0 * (IP / 200)
WAR = (FIP Runs + Replacement Runs) / 9.8
```

### WAR Value: Convex Power-Law Model (with 4% annual inflation from 2025)

The pipeline uses an empirically calibrated convex model to convert WAR to dollars:

```
value = alpha * max(WAR, 0)^beta * (1 + 0.04)^(year - 2025)
```

Parameters (calibrated over 744 MLB trades, 2014-2024):
- **alpha = $8,592,188** (~$8.59M base coefficient)
- **beta = 1.323** (convex exponent, >1 means superlinear)

The convex shape means each additional WAR is worth MORE than the last:

| WAR | Dollar Value (2025) | Effective $/WAR |
|-----|-------------------|----------------|
| 1   | $8.6M             | $8.6M/WAR      |
| 2   | $21.5M            | $10.8M/WAR     |
| 3   | $36.8M            | $12.3M/WAR     |
| 4   | $53.8M            | $13.5M/WAR     |
| 5   | $72.3M            | $14.5M/WAR     |
| 8   | $134.6M           | $16.8M/WAR     |

This reflects real market dynamics:
- **Scarcity premium**: elite players are rare and irreplaceable
- **Certainty premium**: high-WAR players have lower variance
- **Optionality**: surplus WAR can be traded for prospects/assets
- **Roster-slot cost**: one 6-WAR player > two 3-WAR players

Parameters are auto-loaded from `convex_calibration.json` (generated by the
trade analysis pipeline). If unavailable, hardcoded defaults above are used.

> **Legacy model** (deprecated): Tier-based linear pricing ($8M for 0-2 WAR,
> $9M for 2-4 WAR, $10M for 4+ WAR) is kept as `_calculate_war_value_tiered()`
> in value_calculator.py for reference.

### Arbitration Percentages:
- Pre-Arb: Minimum salary ($720K)
- Arb-1: 15% of market value (min $1M)
- Arb-2: 25% of market value (min $2.5M)
- Arb-3: 40% of market value (min $4M)
- Arb-4: 60% of market value (min $5M)

### Prospect Value
```
Base Value = FV_BASE_VALUES[FV grade]  # e.g., FV 60 = $80M
Rank Adjustment = 0.9 - (rank-1) * 0.4/100  # for top 100
Prospect Value = Base Value * Rank Adjustment

# Experience weighting (CRITICAL)
Prospect Weight = max(0, 1 - games_played / threshold)
Final Value = (MLB_Value * MLB_Weight) + (Prospect_Value * Prospect_Weight)
```

Experience thresholds (games to become "established"):
- Batters: 300 games (~2 full seasons)
- Starting Pitchers: 45 starts (~1.5 seasons)
- Relief Pitchers: 65 appearances (~1.5 seasons)

### Trade Value
```
Base_Value = alpha * max(WAR, 0)^beta * inflation(year)
Surplus Value = Base_Value - Contract Value
Trade Value = Σ(Surplus Value from CURRENT_YEAR to FA year)

For arbitration players: floor at $0 (team can non-tender)
For prospects: apply prospect adjustment (FV + experience weight blend)
```

## Input Files

| File | Location | Description |
|------|----------|-------------|
| `batter_predictions.csv` | `data/generated/pipeline/` | Batter projections with wOBA, WAR |
| `pitcher_predictions.csv` | `data/generated/pipeline/` | SP/RP projections with FIP |
| `mlb_salary_data.csv` | `data/salary/` | Contract/salary data |
| `prospects_2014_2026_with_top100.csv` | `data/prospect_data/` | FV grades and rankings |
| `current_rosters.csv` | `data/active_roster/` | Team assignments for park factors |

## Output

**Primary output**: `data/generated/value_by_year/player_values_complete.csv`

Key columns:
- `IDfg`: FanGraphs player ID
- `Name`: Player name
- `Year`: Season
- `WAR`: Projected WAR
- `Base_Value`: Dollar value of WAR
- `contract_value`: Contract cost
- `surplus_value`: Base_Value - contract_value
- `trade_value`: Sum of surplus through FA
- `prospect_adjustment`: Value from prospect grade (if applicable)

## TODO / Known Issues

### ID Migration (Priority: High)
Currently uses FanGraphs ID (`IDfg`) as primary identifier. Plan to migrate to MLB ID (`mlbam_id`):
- [ ] Update data_loader.py to track mlbam_id
- [ ] Update salary_processor.py to match on mlbam_id
- [ ] Update roster matching to prefer mlbam_id
- [ ] Update prospect matching to use mlbam_id

### Testing (Priority: Medium)
- [ ] Add unit tests for calculate_war.py
- [ ] Add unit tests for prospect value calculation
- [ ] Add integration tests for full pipeline

### Validation (Priority: Medium)
- [ ] Compare WAR calculations to FanGraphs published values
- [ ] Validate prospect values against historical trade returns
- [ ] Add sanity checks for extreme values

## Dependencies

- pandas
- numpy
- unidecode
- thefuzz (for name matching)

## Changelog

### 2025-06-11
- **Replaced tier-based WAR valuation with empirically calibrated convex power-law model**
  - Formula: `value = alpha * WAR^beta * inflation(year)` with alpha=$8.59M, beta=1.323
  - Parameters calibrated via Nelder-Mead optimization over 744 MLB trades (2014-2024)
  - Convex shape (beta > 1) captures scarcity premium for elite players
  - Parameters auto-loaded from `convex_calibration.json` with hardcoded fallback defaults
  - Old tiered model kept as `_calculate_war_value_tiered()` for reference
- Fixed two-way player valuation (was hardcoded $10M/WAR, now uses convex model)
- Historical data now uses consistent convex valuation
- Added `ConvexModel` class to `config.py` with full documentation

### 2025-01-28
- Created centralized `config.py` with all settings
- Consolidated WAR calculation in `calculate_war.py`
- Fixed prospect weight to use proper experience thresholds
- Converted all `print()` to `logger` calls
- Added input validation with `validate_input_data()`
- Simplified pipeline from 19 steps to 10
- Added TODO comments for mlbam_id migration

## Author

Niels Christoffersen
