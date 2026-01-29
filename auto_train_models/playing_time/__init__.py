"""
Projection Engine - Playing Time & WAR Calculation
===================================================

Team-constrained playing time allocation and WAR calculation based on:
- Projected rate stats (wOBA for batters, FIP for pitchers)
- Active injury status and history
- Team roster constraints
- Player confidence scores (sample size, prospect ranking, consistency)

The key insight: rate stats don't depend on playing time, so we:
1. Calculate confidence scores based on historical data
2. Use rate stats + confidence to rank players for allocation
3. Allocate playing time based on confidence tiers + injuries
4. Calculate WAR based on allocated games/IP

Author: Niels Christoffersen
Date: January 2026
"""

from .config import Config
from .data_loader import DataLoader
from .injury_processor import InjuryProcessor
from .roster_builder import RosterBuilder
from .allocator import PlayingTimeAllocator
from .value_calculator import ValueCalculator, get_calculator
from .confidence import ConfidenceCalculator, get_confidence_tier

__all__ = [
    'Config',
    'DataLoader', 
    'InjuryProcessor',
    'RosterBuilder',
    'PlayingTimeAllocator',
    'ValueCalculator',
    'get_calculator',
    'ConfidenceCalculator',
    'get_confidence_tier',
]
