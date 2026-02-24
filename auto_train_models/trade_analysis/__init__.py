"""
Trade Analysis Pipeline
========================

Historical trade value calibration using backtested projections,
Cot's salary data, and MLB Stats API trade records.

Modules:
    config                — Paths, constants, year ranges
    generate_projections  — Run LSTM predictions for each cutoff year (2013–2024)
    surplus_calculator    — Combine projections + salary → surplus value per player-year
    analyze_trades        — Match trades to surplus, fit nonlinear β, evaluate
    main                  — Orchestrator (generate → surplus → analyze)
"""
