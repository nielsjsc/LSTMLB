# Contributing to LongBall Analytics

## Areas for Contribution

### Projection Methodology

The projection engine lives in `auto_train_models/core/marcel_projections.py`. All seven component models use the Marcel method with empirically-derived aging curves.

Key files:
- `auto_train_models/configs/` — Per-model configuration (features, regression targets, aging parameters)
- `auto_train_models/core/marcel_projections.py` — Marcel weighting, regression, aging application
- `auto_train_models/analysis/marcel_aging_curves.json` — Aging curve parameters by stat

### Value Determination

The trade value pipeline converts projected WAR into dollar values:
- `auto_train_models/value_determination/config.py` — Convex model parameters, WAR constants, contract tables
- `auto_train_models/value_determination/pipelines/` — Surplus calculation, contract processing, history timeline

### Web Application

- `web-app/backend/` — FastAPI routes and data loading
- `web-app/frontend/` — React 18 + TypeScript + TailwindCSS

## Getting Started

```bash
git clone https://github.com/your-username/LSTMLB.git
cd LSTMLB
pip install -r requirements.txt
```

Run the pipeline:
```bash
cd auto_train_models
python scripts/pipeline.py
```

Run the web app:
```bash
cd web-app/backend && uvicorn app.main:app --reload
cd web-app/frontend && npm install && npm run dev
```

## Submitting Changes

1. Create a feature branch
2. Document what you changed and why
3. Include before/after comparisons for any methodology changes
4. Submit a pull request

## Code Conventions

- Follow existing naming and structure
- Python `logging` module for all output (one logger per module)
- Config classes use class attributes, not instances
- Frontend uses TanStack React Query for data fetching
