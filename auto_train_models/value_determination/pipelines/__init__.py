"""
Value Determination Pipelines
=============================

Pipeline orchestrators that consume the shared value-determination engine.

Modules
-------
current      – Daily production pipeline (WAR → surplus → trade values)
ros          – In-season rest-of-season blending (Bayesian shrinkage)
snapshots    – Daily trade-value snapshot management
trade_history – Historical trade-value timeline builder
"""
