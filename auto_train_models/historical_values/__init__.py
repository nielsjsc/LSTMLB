"""
Historical Trade Value Pipeline
================================

Self-contained pipeline that generates a complete historical trade-value
timeline for every MLB player.  Uses the project's LSTM framework with
classical (pre-Statcast) config files for defense and baserunning so that
projections work for all cutoff years back to 2013.

Inputs:
    - Historical player statistics (FanGraphs batting/pitching/fielding)
    - Cot's salary data by year  (``data/salary/by_year/``)
    - Spotrac transaction data   (``scrapers/data/salary/spotrac_transactions.csv``)
    - Prospect rankings          (``data/prospect_data/``)

Output:
    ``data/generated/value_by_year/trade_value_history.csv``
    One row per (player, year) event — annual surplus snapshots, prospect
    rankings, and free-agency transitions.

Usage:
    python -m auto_train_models.historical_values.main --start 2014 --end 2026
"""
