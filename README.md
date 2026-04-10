# LongBall Analytics

MLB player projection and valuation system. Forecasts performance up to 15 years out, calculates trade value from projected surplus, and powers a full-stack web app with a trade simulator, player projections, and value rankings.

**Live at [longballhq.xyz](https://longballhq.xyz)**

## How It Works

### Projection Engine

Seven independent component models project stats year-by-year using the [Marcel method](https://www.tangotiger.net/marcel/) — a proven baseline in sabermetrics:

1. **3-year weighted average** (5/4/3 weights, sample-size adjusted)
2. **Regression toward league average**
3. **Empirically-derived aging curves** (calibrated from historical MLB data)
4. **Year-by-year forward projection** with compounding age adjustments

| Component | Scope |
|-----------|-------|
| Batting | wOBA, wRC+, OBP, SLG, ISO, BABIP, K%, BB% |
| Pitching (SP) | FIP, ERA, K/9, BB/9, HR/FB, IP |
| Pitching (RP) | Same core stats, RP-specific playing time |
| Fielding (IF) | Statcast range/arm/DP runs, reliability-shrunk (α=0.65) |
| Fielding (OF) | Statcast OAA-based runs (α=0.70) |
| Fielding (C) | Framing, throwing, blocking runs (α=0.70) |
| Baserunning | BsR, sprint speed, SB/CS value |

Fielding projections use **reliability shrinkage multipliers** calibrated via out-of-sample grid search on 2016–2025 data. Infield arm runs are near-zero signal (α=0.10); range runs are the most stable.

### WAR Calculation

Components are combined into WAR following FanGraphs methodology:

- **Batters**: wRAA + BsR + Fielding + Positional Adjustment, divided by runs-per-win
- **Pitchers**: FIP-based WAR with dynamic RPW

### Trade Value Pipeline

Trade value = Σ(surplus) over remaining contract years.

**Surplus** = production value − contract cost, where production value uses a **convex WAR model**:

$$value = \alpha \times \max(WAR, 0)^{\beta}$$

Calibrated on historical trades (2014–2024, minimizing median trade imbalance):
- α ≈ $8.6M, β ≈ 1.18 — meaning star players are valued superlinearly (a 5-WAR player is worth more than five 1-WAR players)

### In-Season Updates

During the season, a daily pipeline blends preseason Marcel projections with actual stats using **Bayesian shrinkage**:

$$\hat{\theta} = \frac{x \cdot n + \mu_0 \cdot n_0}{n + n_0}$$

At 400 PA (batters) or 80 IP (pitchers), the blend is 50/50 between observed and prior. Early in the season, projections dominate; by September, actuals take over.

## Web Application

React 18 + TypeScript frontend, FastAPI + PostgreSQL backend.

- **Trade Simulator** — Evaluate multi-player trades using projected surplus value
- **Player Projections** — 15-year stat projections with component breakdowns
- **Trade Value Rankings** — All players ranked by projected surplus
- **Historical Trade Value Charts** — Track a player's value over their career with transaction annotations
- **Prospect Valuations** — FV grades converted to dollar values, integrated into trade analysis
- **Past Trades** — Historical trades re-evaluated with current projections

## Project Structure

```
auto_train_models/
├── configs/             # Per-model configuration (batting, pitching, fielding, baserunning)
├── core/                # Marcel projections, aging curves, WAR components
├── value_determination/ # Surplus calculation, contract processing, trade value pipeline
├── daily_ros/           # In-season Bayesian rest-of-season updates
├── historical_values/   # Backtesting pipeline (2014–2026)
└── scripts/             # CLI pipeline orchestration
web-app/
├── backend/             # FastAPI + SQLAlchemy API
└── frontend/            # React + Vite + TailwindCSS
data/
├── generated/           # Pipeline outputs (projections, surplus, trade values)
└── ...                  # Historical stats, prospects, salary, player registry
```

## Setup

```bash
pip install -r requirements.txt
```

### Generate Projections

```bash
cd auto_train_models
python scripts/pipeline.py
```

Interactive menu: Predict → Combine → Value Determination → Full Pipeline.

### Run the Web App

```bash
# Backend
cd web-app/backend
uvicorn app.main:app --reload

# Frontend
cd web-app/frontend
npm install && npm run dev
```

## Technical Notes

- Python 3.14, React 18, TypeScript, TailwindCSS
- No NPV discounting on future surplus (design decision)
- LSTM architecture exists in `core/model_architecture.py` but is not active — Marcel outperforms on noisy baseball stats
- Fielding data requires Statcast (2016+); Marcel needs 3 seasons → effective cutoff is 2018
- Positional adjustments: C +12.5, SS +7.5, CF +2.5, 1B −12.5, DH −17.5 (runs/162G)

## License

MIT

