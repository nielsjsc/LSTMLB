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
│      ├── WAR → Dollar value (tiered pricing)                         │
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

### WAR Value Tiers (with 4% annual inflation from 2025):
- Tier 1 (0-2 WAR): $8M per WAR
- Tier 2 (2-4 WAR): $9M per WAR
- Tier 3 (4+ WAR): $10M per WAR

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
Trade Value = Σ(Surplus Value from 2025 to FA year)
Surplus Value = Base Value - Contract Value

For arbitration players: floor at $0
For prospects: apply prospect adjustment
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
