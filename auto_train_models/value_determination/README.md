# Value Determination Module

This module calculates player trade values based on WAR projections and contract status. It is a script-based implementation of the `determine_value.ipynb` notebook.

## Overview

The value determination pipeline combines projected stats and salary data to calculate trade value for MLB players. The main workflow:

1. Load prediction data (batters, starting pitchers, relief pitchers)
2. Load and clean salary data
3. Merge salary data with player IDs
4. Normalize contract statuses (Pre-Arb, Arb-1/2/3, Free Agent, etc.)
5. Generate contract timelines through free agency
6. Calculate WAR-based values using tiered pricing
7. Calculate contract values based on arbitration status
8. Calculate surplus value (Base Value - Contract Value)
9. Integrate historical stats
10. Analyze contract options (player options, team options, opt-outs)
11. Calculate trade values with prospect adjustments
12. Export final data

## Directory Structure

```
value_determination/
├── __init__.py           # Module exports
├── constants.py          # Configuration and constants
├── data_loader.py        # Data loading functions
├── salary_processor.py   # Salary data cleaning and merging
├── contract_processor.py # Contract status normalization and timeline generation
├── value_calculator.py   # WAR and contract value calculations
├── trade_value.py        # Trade value and ranking metrics
├── exporter.py           # Data export functions
├── main.py               # Main pipeline script
└── README.md             # This file
```

## Input Data

The pipeline requires the following input files:

### From `data/generated/pipeline/`:
- `pitcher_predictions.csv` - Pitcher projections with columns: Name, Year, Age, Role, IDfg, FIP, SIERA, ERA, K%, BB%, WAR, etc.
- `batter_predictions.csv` - Batter projections with columns: Name, IDfg, Year, Age, BB%, K%, AVG, OBP, SLG, WAR, etc.

### From `data/salary/`:
- `mlb_salary_data.csv` - Salary data with columns: player_name, player_id, team, year, status, payroll_annual, etc.

### From `data/historic_mlb/`:
- `mlb_batting_data_1950_2025.csv` (or `mlb_batting_data_2000_2024.csv`)
- `mlb_pitching_data_1950_2025.csv` (or `mlb_pitching_data_2000_2024.csv`)

### Optional (for prospect adjustments):
- `data/generated/MiLB/player_histories.csv`

## Output

The pipeline outputs:
- `data/generated/value_by_year/player_values_complete.csv`

## Usage

### Run from project root:
```bash
python run_value_determination.py
```

### Or run the module directly:
```bash
python -m value_determination.main
```

### Or import and use programmatically:
```python
from value_determination.main import main

# Run the full pipeline
export_data = main()

# Or use individual components
from value_determination.data_loader import load_prediction_files
from value_determination.value_calculator import calculate_war_value

sp_data, rp_data, batter_data, salary_data = load_prediction_files()
value = calculate_war_value(war=5.0, year=2025)
```

## Key Calculations

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

### Surplus Value:
```
Surplus Value = Base Value - Contract Value
```

### Trade Value:
Sum of surplus values from current year through free agency, with:
- Floor of $0 for players under team control
- Prospect value adjustments based on MLB experience

## Dependencies

- pandas
- numpy
- unidecode
- thefuzz (for name matching)

## Author

Niels Christoffersen
