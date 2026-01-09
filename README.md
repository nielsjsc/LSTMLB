# LongBall Analytics - MLB Player Projection System

LSTM-based player projection system that forecasts MLB player performance 15 years into the future.

## Overview

This project uses Long Short-Term Memory (LSTM) neural networks to project player statistics. Baseball stats are naturally sequential data, making LSTMs well-suited for this task.

### What It Does

- Projects batter and pitcher performance from 2026-2040
- Calculates projected WAR using FanGraphs methodology
- Combines offensive, baserunning, and defensive components
- Supports both classical stats (2000+) and Statcast-enhanced predictions (2015+)

## Live Web Application

The projection system powers [LongBall HQ](https://longballhq.xyz), featuring:

- **Trade Simulator** - Evaluate trades using projected WAR and surplus value calculations
- **Player Projections** - Browse 15-year projections for all MLB players
- **Trade Value Rankings** - Compare players by projected contract value and WAR
- **Prospect Valuations** - Prospect grades converted to trade value metrics

## Project Structure

```
LSTMLB/
├── auto_train_models/          # Core prediction system
│   ├── scripts/
│   │   └── pipeline.py         # Main entry point
│   ├── configs/                # Model configurations
│   ├── core/                   # Training/prediction logic
│   ├── evaluation/             # WAR calculation
│   └── comparison_tools/       # Projection analysis
├── data/
│   ├── historic_mlb/           # Historical player data
│   └── generated/              # Model outputs
│       └── pipeline/           # Final predictions
└── models/                     # Jupyter notebooks for exploration
```

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Generate Projections

Run the unified pipeline:

```bash
cd auto_train_models
python scripts/pipeline.py
```

The pipeline provides an interactive menu:

1. **Train Models** - Train individual or all models
2. **Generate Predictions** - Run trained models on current players
3. **Combine Predictions** - Calculate WAR from component predictions
4. **Full Pipeline** - Train → Predict → Combine in one step

### Output Files

Predictions are saved to `data/generated/pipeline/`:

- `batter_predictions_with_war.csv` - Full batter projections with WAR
- `pitcher_predictions.csv` - Pitcher projections
- `baserunning_predictions.csv` - Baserunning component
- `fielding_predictions.csv` - Defensive component

## Models

| Model | Description | Training Data |
|-------|-------------|---------------|
| Batter (Pretrained) | Classical stats only | 1950-2025 |
| Batter (Finetuned) | Classical + Statcast | 2016-2025 |
| Pitcher SP/RP | Starting/Relief pitchers | 2000-2025 |
| Baserunning | BsR, stolen base value | 2016-2025 |
| Fielding | Position-specific defense | 2016-2025 |

## WAR Calculation

WAR is calculated using FanGraphs methodology:

```
WAR = (Offense + Baserunning + Defense + Positional) / Runs_Per_Win
```

Components:
- **Offense (Off)**: wRAA scaled to playing time
- **Baserunning (BsR)**: Statcast runner runs (XB + SBX)
- **Fielding (Fld)**: Position-specific fielding runs
- **Positional (Pos)**: Position adjustment per 150 games

## Disclaimers

- Hitting projections are generally more reliable than pitching
- Players with limited MLB experience may have unreliable projections
- These are model estimates, not guarantees

## Methodology Notes

### Playing Time Normalization
- Hitters: 150 games
- Catchers: 135 games
- Starting Pitchers: 32 games
- Relief Pitchers: 65 innings

## License

MIT

---

