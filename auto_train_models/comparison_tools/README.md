# Projection Comparison Tools

This folder contains tools for analyzing and comparing player projections from the pipeline.

## compare_projections.py

An interactive Dash application for visually comparing career trajectories and projections.

### Features

- **Career Trajectory Plot**: View player projections over 15 years (2026-2040)
- **Multi-Stat Radar Chart**: Compare multiple offensive and WAR components
- **WAR Component Analysis**: See breakdown of Off, BsR, Fld, Pos, Def components
- **Counting Stats Comparison**: Compare HR, 2B, RBI, R, SB, CS
- **Team and Position Context**: View player team and defensive position
- **Data Table**: Side-by-side numerical comparison of all statistics

### Requirements

```bash
pip install dash dash-bootstrap-components plotly pandas numpy
```

### Usage

```bash
cd auto_train_models/comparison_tools
python compare_projections.py
```

Then open http://127.0.0.1:8050 in your browser.

### Data Sources

The tool loads data from:
- `data/generated/pipeline/batter_predictions_with_war.csv` - Complete batter projections with WAR
- `data/generated/pipeline/pitcher_predictions.csv` - Pitcher projections
- `data/historic_mlb/mlb_batting_data_1950_2025.csv` - Historical data for comparison

### Stats Displayed

**Offensive:** wOBA, wRC+, OBP, SLG, AVG, BB%, K%

**Counting Stats:** HR, 2B, RBI, R, SB, CS, PA, G

**WAR Components:** Off (Offensive), BsR (Baserunning), Fld (Fielding), Pos (Positional Adj), Def (Defense Total), WAR

**Pitcher Stats:** FIP, SIERA, ERA, K%, BB%, IP, G, GS, WAR

**Player Context:** Name, Team, Position, Age, Year

## generate_report.py

Static HTML report generator for player projections (no server required).

### Usage

```bash
# Generate summary report
python generate_report.py --summary

# Generate report for specific player
python generate_report.py --player "Aaron Judge"

# Generate reports for all players
python generate_report.py --all
```

Reports are saved to `comparison_tools/reports/` directory.
