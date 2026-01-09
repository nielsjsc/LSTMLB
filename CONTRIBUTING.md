# Contributing to LSTMLB

Contributions focused on improving model configurations and training parameters are welcome.

## Testing Model Configurations

The primary area for experimentation is model hyperparameters in `auto_train_models/configs/`.

### Configuration Files

Each model has a dedicated config file:
- `batter_config.py` - Batter projections (classical and Statcast features)
- `pitcher_sp_config.py` / `pitcher_rp_config.py` - Starting and relief pitcher models
- `baserunning_config.py` - Baserunning value predictions
- `defense_infield_config.py` / `defense_outfield_config.py` / `defense_catcher_config.py` - Defensive metrics by position

### Testing Procedure

1. Fork and clone the repository
```bash
git clone https://github.com/your-username/LSTMLB.git
cd LSTMLB
pip install -r requirements.txt
```

2. Modify hyperparameters in a config file:
```python
# Example: auto_train_models/configs/batter_config.py
HIDDEN_SIZE = 128  # Try 64, 256
NUM_LAYERS = 2     # Try 3, 4
DROPOUT = 0.2      # Try 0.1, 0.3
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
```

3. Train the model:
```bash
cd auto_train_models
python scripts/train_models.py --model batter
```

4. Generate predictions and evaluate:
```bash
python scripts/predict_models.py --model-type batter
python evaluation/calculate_war.py
```

5. Compare results using the comparison tool:
```bash
python comparison_tools/compare_projections.py
```

### Evaluation Metrics

Document performance changes:
- Training/validation loss curves
- Comparison to baseline predictions (notebook vs. pipeline)
- Historical accuracy on validation data (2020-2024)
- WAR projection distributions

## Submitting Changes

1. Create a feature branch:
```bash
git checkout -b config/model-improvements
```

2. Document your changes:
- Configuration parameters modified
- Performance improvements observed
- Training time and resource usage

3. Submit a pull request with:
- Clear description of changes
- Performance comparison data
- Justification for parameter choices

## Code Standards

- Follow existing code structure and naming conventions
- Include docstrings for any new functions
- Test changes on multiple model types if modifying core training logic

## Questions

Open an issue for questions about model architecture or training procedures.
