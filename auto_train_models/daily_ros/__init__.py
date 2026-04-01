"""
Daily ROS (Rest-of-Season) Projection Pipeline
===============================================

Phase 1: Data ingestion — scrape current-season stats, merge into
historic files, and update player positions from fielding data.

Phase 2: ROS-aware value determination — blend pre-season predictions
with actual current-season performance via Bayesian shrinkage, prorate
current-year WAR (actual + projected × remaining), then run the full
value determination pipeline (salary, contracts, surplus, trade values).
"""
