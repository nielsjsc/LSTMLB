# Projection Comparison Tools

This directory contains a comprehensive suite of tools for hyperparameter experimentation,
model comparison, and projection analysis for the MLB batter prediction pipeline.

## Overview

The comparison tools provide three main capabilities:

1. **Hyperparameter Search**: Systematically train models with different configurations
2. **Multi-Model Comparison**: Compare projections across different trained models
3. **Single Model Analysis**: Deep-dive into individual model projections

## Quick Start

```bash
# 1. Run hyperparameter search (trains models and generates predictions)
python hyperparameter_search.py --run-all

# 2. Compare results across models
python compare_experiments.py

# 3. (Optional) Compare notebook vs pipeline models
python compare_projections.py
```

---

## Tool Reference

### hyperparameter_search.py

**Purpose:** Train multiple batter models with different hyperparameter configurations
and generate 15-year projections for each.

#### Hyperparameters Explored

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `SEQ_LEN` | Number of historical seasons used as input | 1-7 seasons |
| `MIN_PA` | Minimum plate appearances per season for training | 10-200 PA |
| `NUM_LAYERS` | Number of LSTM layers | 1-4 layers |
| `NUM_HEADS` | Number of attention heads | 2-8 heads |
| `HIDDEN_SIZE` | LSTM hidden state dimension | 64-512 |
| `DROPOUT` | Dropout rate for regularization | 0.0-0.5 |
| `BATCH_SIZE` | Training batch size | 16-256 |

#### Predefined Configurations

The tool includes 20+ predefined configurations organized by hypothesis:

**Baseline:**
- `baseline` - Current production configuration

**Sequence Length Variants:**
- `seq_len_1` - Only most recent season
- `seq_len_3` - Three seasons of history
- `seq_len_5` - Five seasons of history
- `seq_len_7` - Maximum historical context

**Data Quality Variants:**
- `min_pa_50` - Moderate PA filter (50)
- `min_pa_100` - Standard PA filter (100)
- `min_pa_200` - Strict PA filter (200)

**Architecture Variants:**
- `shallow_network` - Single layer, small hidden size
- `deep_network` - Four layers, large hidden size
- `wide_network` - Two layers, very large hidden size
- `many_heads` - Eight attention heads

**Regularization Variants:**
- `light_dropout` - Dropout 0.1
- `moderate_dropout` - Dropout 0.2
- `heavy_dropout` - Dropout 0.3
- `aggressive_dropout` - Dropout 0.5

**Batch Size Variants:**
- `small_batch` - Batch size 16
- `medium_batch` - Batch size 64
- `large_batch` - Batch size 256

**Combined Configurations:**
- `long_seq_high_pa` - Long sequences with quality data
- `deep_regularized` - Deep network with dropout
- `small_batch_high_capacity` - Small batches with large network

#### Usage

```bash
# Run all predefined configurations
python hyperparameter_search.py --run-all

# Run specific configurations
python hyperparameter_search.py --configs baseline seq_len_5 deep_network

# List available configurations
python hyperparameter_search.py --list-configs

# Show experiment summary
python hyperparameter_search.py --summary

# Generate predictions only (skip training)
python hyperparameter_search.py --predict-only --configs baseline

# Force retrain existing models
python hyperparameter_search.py --configs baseline --force-retrain
```

#### Output Structure

```
comparison_tools/
    experiments/
        baseline/
            config.json          # Full configuration details
            model.pth            # Trained model checkpoint
            scaler.pkl           # Feature scaler
            predictions.csv      # 15-year projections
            training_log.json    # Training metrics history
        seq_len_5/
            ...
        deep_network/
            ...
```

---

### config_generator.py

**Purpose:** Programmatically generate custom hyperparameter configurations for
grid search, random search, or custom experiments.

#### Usage

```python
from config_generator import ConfigGenerator, HyperparameterSpace

# Define custom search space
space = HyperparameterSpace(
    seq_len=[2, 3, 5],
    min_pa=[50, 100, 200],
    dropout=[0.0, 0.1, 0.2]
)

# Generate grid search configs
generator = ConfigGenerator(space)
configs = generator.grid_search(max_configs=50)

# Or random search
configs = generator.random_search(n_configs=20)

# Save configurations
generator.save_configs(configs, Path('my_configs.json'))
```

#### Predefined Search Spaces

```python
from config_generator import (
    get_sequence_length_space,
    get_data_quality_space,
    get_architecture_space,
    get_regularization_space,
    get_full_space
)

# Focus on sequence length only
space = get_sequence_length_space()

# Full comprehensive search
space = get_full_space()
```

---

### compare_experiments.py

**Purpose:** Interactive dashboard for comparing projections across multiple
trained models from hyperparameter_search.py.

#### Features

- **Multi-Model Selection**: Compare any subset of trained models
- **Career Trajectory Plots**: View how different models project the same player
- **Model Agreement Analysis**: See where models agree/disagree over time
- **Hyperparameter Impact Visualization**: Understand how hyperparameters affect predictions
- **Detailed Comparison Tables**: Year-by-year numerical comparisons
- **Export Functionality**: Export all data to CSV for external analysis

#### Usage

```bash
# Launch dashboard (loads all experiments)
python compare_experiments.py

# Compare specific experiments only
python compare_experiments.py --experiments baseline seq_len_5 deep_network

# Export comparison data
python compare_experiments.py --export

# Use different port
python compare_experiments.py --port 8052
```

Then open http://127.0.0.1:8051 in your browser.

---

### compare_projections.py

**Purpose:** Compare projections between notebook and pipeline models
(original comparison tool).

#### Features

- **Career Trajectory Plot**: View player projections over 15 years (2026-2040)
- **Multi-Stat Radar Chart**: Compare multiple offensive and WAR components
- **WAR Component Analysis**: See breakdown of Off, BsR, Fld, Pos, Def components
- **Counting Stats Comparison**: Compare HR, 2B, RBI, R, SB, CS
- **Team and Position Context**: View player team and defensive position
- **Data Table**: Side-by-side numerical comparison of all statistics

#### Usage

```bash
python compare_projections.py
```

Then open http://127.0.0.1:8050 in your browser.

---

## Requirements

```bash
pip install dash dash-bootstrap-components plotly pandas numpy torch scikit-learn joblib tqdm
```

## Data Sources

| File | Description |
|------|-------------|
| `data/historic_mlb/mlb_batting_data_1950_2025.csv` | Historical batting data |
| `data/generated/pipeline/batter_predictions_with_war.csv` | Production batter projections |
| `data/generated/pipeline/pitcher_predictions.csv` | Production pitcher projections |

## Workflow Example

```bash
# Step 1: Train models with different hyperparameters
python hyperparameter_search.py --configs baseline seq_len_3 seq_len_5 seq_len_7

# Step 2: Check training results
python hyperparameter_search.py --summary

# Step 3: Compare predictions interactively
python compare_experiments.py

# Step 4: Export data for further analysis
python compare_experiments.py --export
```

## Statistics Reference

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
