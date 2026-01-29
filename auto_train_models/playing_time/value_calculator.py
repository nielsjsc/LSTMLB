"""
Value Calculator - WAR and Counting Stats Calculation
======================================================

Calculates WAR and counting stats based on allocated playing time.
This module is called AFTER playing time allocation to convert rate stats
into actual projected values.

The key insight: rate stats (wOBA, FIP) don't depend on playing time,
so we use them to rank players for allocation. Then we calculate WAR
based on the allocated games/IP.

Author: Niels Christoffersen
Date: January 2026
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Ballpark factors (runs, 100 = neutral)
BALLPARK_FACTORS = {
    'COL': 108,  # Coors Field
    'BOS': 107,  # Fenway Park  
    'CIN': 105,  # Great American Ball Park
    'BAL': 104,  # Oriole Park at Camden Yards
    'TEX': 104,  # Globe Life Field
    'NYY': 103,  # Yankee Stadium
    'MIN': 103,  # Target Field
    'PHI': 103,  # Citizens Bank Park
    'HOU': 102,  # Minute Maid Park
    'TOR': 102,  # Rogers Centre
    'ARI': 101,  # Chase Field
    'ATL': 101,  # Truist Park
    'WSH': 101,  # Nationals Park
    'LAA': 100,  # Angel Stadium
    'CLE': 100,  # Progressive Field
    'DET': 100,  # Comerica Park
    'KC': 100,   # Kauffman Stadium
    'LAD': 100,  # Dodger Stadium
    'NYM': 100,  # Citi Field
    'STL': 100,  # Busch Stadium
    'CHW': 99,   # Guaranteed Rate Field
    'OAK': 99,   # Oakland Coliseum
    'PIT': 99,   # PNC Park
    'SF': 98,    # Oracle Park
    'MIA': 98,   # loanDepot park
    'MIL': 97,   # American Family Field
    'CHC': 97,   # Wrigley Field
    'SD': 96,    # Petco Park
    'TB': 96,    # Tropicana Field
    'SEA': 91    # T-Mobile Park
}

# League constants (2024 baseline)
WOBA_SCALE = 1.23
RPA = 0.117          # League runs per PA
LG_WOBA = 0.309
RPW = 9.8            # Runs per Win
LG_PA = 186188       # League total PA
LG_RUNS_PER_PA = 0.114
LG_WRC_PER_PA = 0.117

# Pitcher constants
LG_FIP = 4.10        # League average FIP
LG_ERA = 4.17        # League average ERA
FIP_CONSTANT = 3.10  # cFIP constant

# Positional adjustments (runs per 162 games)
POSITIONAL_ADJUSTMENTS = {
    'C': 12.5,
    'SS': 7.5,
    '2B': 2.5,
    'CF': 2.5,
    '3B': 2.5,
    'LF': -7.5,
    'RF': -7.5,
    '1B': -12.5,
    'DH': -17.5
}

# Replacement level runs per 600 PA (batters) or 200 IP (pitchers)
REPLACEMENT_LEVEL_RUNS_600PA = 20.0
REPLACEMENT_LEVEL_RUNS_200IP = 17.5


@dataclass
class BatterWARComponents:
    """Components of batter WAR calculation."""
    batting_runs: float
    baserunning_runs: float
    fielding_runs: float
    positional_adj: float
    replacement_runs: float
    war: float
    wrc_plus: float  # wRC+ with park factors
    games: float
    pa: float
    
    # Counting stats
    hr: float = 0.0
    doubles: float = 0.0
    rbi: float = 0.0
    runs: float = 0.0
    sb: float = 0.0
    cs: float = 0.0


@dataclass
class PitcherWARComponents:
    """Components of pitcher WAR calculation."""
    fip_runs: float
    replacement_runs: float
    war: float
    ip: float
    
    # Counting stats
    wins: float = 0.0
    losses: float = 0.0
    saves: float = 0.0
    strikeouts: float = 0.0
    walks: float = 0.0
    era: float = 0.0
    fip: float = 0.0


class ValueCalculator:
    """
    Calculate WAR and counting stats from rate stats and allocated playing time.
    
    This is the final step of the projection pipeline:
    1. Predictions generate rate stats
    2. Playing time is allocated based on rate stat quality
    3. This calculator converts rates to counting stats and WAR
    """
    
    def __init__(self):
        pass
    
    # =========================================================================
    # BATTER WAR CALCULATION
    # =========================================================================
    
    def calculate_wrc_plus(self, woba: float, team: str) -> float:
        """
        Calculate wRC+ using the proper formula with park factors.
        Exactly matches the calculation from calculate_war.py.
        
        Args:
            woba: Projected wOBA
            team: Team abbreviation for park factor
            
        Returns:
            wRC+ value (100 = league average)
        """
        # Calculate wRAA per PA
        wraa_per_pa = (woba - LG_WOBA) / WOBA_SCALE
        
        # Get park factor (default to 100 if no team - meaning no adjustment)
        park_factor = BALLPARK_FACTORS.get(str(team).upper().strip(), 100) / 100
        
        # Calculate Park Adjustment
        park_adjustment = LG_RUNS_PER_PA - (park_factor * LG_RUNS_PER_PA)
        
        # Calculate the numerator for wRC+
        numerator = (wraa_per_pa + LG_RUNS_PER_PA) + park_adjustment
        
        # Calculate wRC+
        wrc_plus = (numerator / LG_WRC_PER_PA) * 100
        
        return wrc_plus
    
    def calculate_batter_war(self,
                             woba: float,
                             games: float,
                             team: str,
                             position: str,
                             baserunning_data: Optional[Dict] = None,
                             fielding_data: Optional[Dict] = None,
                             rate_stats: Optional[Dict] = None) -> BatterWARComponents:
        """
        Calculate batter WAR from rate stats and allocated games.
        
        Args:
            woba: Projected wOBA
            games: Allocated games
            team: Team abbreviation for park factor
            position: Primary position
            baserunning_data: Dict with baserunning rate stats
            fielding_data: Dict with fielding rate stats
            rate_stats: Dict with counting stat rates (HR_rate, etc.)
            
        Returns:
            BatterWARComponents with full breakdown
        """
        # PA estimation
        pa_per_game = 4.2
        pa = games * pa_per_game
        
        # Park factor
        park_factor = BALLPARK_FACTORS.get(str(team).upper().strip(), 100) / 100
        
        # Batting runs (wRAA + park adjustment)
        wraa = ((woba - LG_WOBA) / WOBA_SCALE) * pa
        park_adj = (RPA - (RPA * park_factor)) * pa
        batting_runs = wraa + park_adj
        
        # Baserunning runs
        if baserunning_data:
            # Rate stats are per 150 games
            bsr_xb = baserunning_data.get('sc_baserunning_runner_runs_XB_rate', 0) * (games / 150.0)
            bsr_sbx = baserunning_data.get('sc_baserunning_runner_runs_SBX_rate', 0) * (games / 150.0)
            baserunning_runs = bsr_xb + bsr_sbx
        else:
            baserunning_runs = -0.5 * (games / 150.0)  # Slightly below average
        
        # Fielding runs
        if fielding_data and position != 'DH':
            fielding_runs = self._calculate_fielding_runs(fielding_data, position, games)
        else:
            fielding_runs = 0.0
        
        # Positional adjustment (scale by games)
        pos_adj_per_162 = POSITIONAL_ADJUSTMENTS.get(position, 0.0)
        positional_adj = pos_adj_per_162 * (games / 162.0)
        
        # Replacement level runs
        replacement_runs = REPLACEMENT_LEVEL_RUNS_600PA * (pa / 600.0)
        
        # Total runs above replacement
        rar = batting_runs + baserunning_runs + fielding_runs + positional_adj + replacement_runs
        
        # WAR
        war = rar / RPW
        
        # wRC+ with park factors (matching calculate_war.py methodology)
        wrc_plus = self.calculate_wrc_plus(woba, team)
        
        # Counting stats
        counting = self._calculate_batter_counting_stats(rate_stats, games, baserunning_data)
        
        return BatterWARComponents(
            batting_runs=batting_runs,
            baserunning_runs=baserunning_runs,
            fielding_runs=fielding_runs,
            positional_adj=positional_adj,
            replacement_runs=replacement_runs,
            war=war,
            wrc_plus=wrc_plus,
            games=games,
            pa=pa,
            **counting
        )
    
    def _calculate_fielding_runs(self, fielding_data: Dict, position: str, games: float) -> float:
        """Calculate fielding runs from Statcast FRV metrics."""
        # Rate stats are per 150 games
        scaling = games / 150.0
        
        if position == 'C':
            framing = fielding_data.get('sc_framing_runs/150', 0) * scaling
            throwing = fielding_data.get('sc_throwing_runs/150', 0) * scaling
            blocking = fielding_data.get('sc_blocking_runs/150', 0) * scaling
            return framing + throwing + blocking
        
        elif position in ['1B', '2B', '3B', 'SS']:
            range_runs = fielding_data.get('sc_range_runs/150', 0) * scaling
            arm_runs = fielding_data.get('sc_arm_runs/150', 0) * scaling
            dp_runs = fielding_data.get('sc_dp_runs/150', 0) * scaling
            return range_runs + arm_runs + dp_runs
        
        else:  # Outfielders
            range_runs = fielding_data.get('sc_range_runs/150', 0) * scaling
            arm_runs = fielding_data.get('sc_arm_runs/150', 0) * scaling
            return range_runs + arm_runs
    
    def _calculate_batter_counting_stats(self, 
                                         rate_stats: Optional[Dict],
                                         games: float,
                                         baserunning_data: Optional[Dict]) -> Dict:
        """Convert rate stats to counting stats."""
        counting = {
            'hr': 0.0,
            'doubles': 0.0,
            'rbi': 0.0,
            'runs': 0.0,
            'sb': 0.0,
            'cs': 0.0
        }
        
        if rate_stats:
            # Rate stats are typically per game or per 150 games
            counting['hr'] = rate_stats.get('HR_rate', 0) * games
            counting['doubles'] = rate_stats.get('2B_rate', 0) * games
            counting['rbi'] = rate_stats.get('RBI_rate', 0) * games
            counting['runs'] = rate_stats.get('R_rate', 0) * games
        
        if baserunning_data:
            # SB/CS rates are per 150 games
            counting['sb'] = baserunning_data.get('SB_rate', 0) * (games / 150.0)
            counting['cs'] = baserunning_data.get('CS_rate', 0) * (games / 150.0)
        
        return counting
    
    # =========================================================================
    # PITCHER WAR CALCULATION
    # =========================================================================
    
    def calculate_pitcher_war(self,
                              fip: float,
                              ip: float,
                              team: str,
                              role: str,
                              rate_stats: Optional[Dict] = None) -> PitcherWARComponents:
        """
        Calculate pitcher WAR from FIP and allocated innings.
        
        FIP-based WAR formula:
        - Runs prevented = (LG_FIP - FIP) / 9 * IP
        - Replacement level = IP / 9 * (replacement_runs_per_9)
        - WAR = (Runs prevented + Replacement) / RPW
        
        Args:
            fip: Projected FIP
            ip: Allocated innings pitched
            team: Team abbreviation for park factor
            role: 'SP' or 'RP'
            rate_stats: Dict with rate stats (K%, BB%, etc.)
            
        Returns:
            PitcherWARComponents with full breakdown
        """
        # Park factor adjustment for pitchers (inverse of batters)
        park_factor = BALLPARK_FACTORS.get(str(team).upper().strip(), 100) / 100
        
        # Adjust FIP for park (pitcher in a hitter's park has inflated FIP)
        park_adjusted_fip = fip / park_factor if park_factor != 0 else fip
        
        # FIP runs saved (positive = better than league)
        fip_runs = (LG_FIP - park_adjusted_fip) / 9.0 * ip
        
        # Replacement level runs
        replacement_runs = REPLACEMENT_LEVEL_RUNS_200IP * (ip / 200.0)
        
        # Total runs above replacement
        rar = fip_runs + replacement_runs
        
        # WAR
        war = rar / RPW
        
        # Counting stats
        counting = self._calculate_pitcher_counting_stats(rate_stats, ip, role, fip)
        
        return PitcherWARComponents(
            fip_runs=fip_runs,
            replacement_runs=replacement_runs,
            war=war,
            ip=ip,
            **counting
        )
    
    def _calculate_pitcher_counting_stats(self,
                                          rate_stats: Optional[Dict],
                                          ip: float,
                                          role: str,
                                          fip: float) -> Dict:
        """Convert pitcher rate stats to counting stats."""
        counting = {
            'wins': 0.0,
            'losses': 0.0,
            'saves': 0.0,
            'strikeouts': 0.0,
            'walks': 0.0,
            'era': fip + 0.3,  # ERA typically ~0.3 higher than FIP
            'fip': fip
        }
        
        if rate_stats:
            # K% and BB% are percentages
            # Estimate K/9 and BB/9: K/9 = K% * BF/IP * 9, simplified to K% * ~4 * 9
            k_pct = rate_stats.get('K%', 0.20)
            bb_pct = rate_stats.get('BB%', 0.08)
            
            # Batters faced per IP is roughly 4.3
            bf_per_ip = 4.3
            
            counting['strikeouts'] = k_pct * bf_per_ip * ip
            counting['walks'] = bb_pct * bf_per_ip * ip
            
            # Win/Loss estimation based on IP and role
            if role == 'SP':
                # ~6 IP per start, wins correlate with quality
                games_started = ip / 6.0
                # Win probability roughly correlates with FIP
                win_pct = 0.5 + (LG_FIP - fip) * 0.1
                win_pct = max(0.3, min(0.7, win_pct))
                counting['wins'] = games_started * win_pct
                counting['losses'] = games_started * (1 - win_pct) * 0.8
            else:
                # RP wins/saves
                appearances = ip / 1.0  # ~1 IP per appearance
                if rate_stats.get('SV%', 0) > 0.3:  # Closer
                    counting['saves'] = appearances * 0.5
                counting['wins'] = appearances * 0.05
        
        return counting
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    def calculate_era_minus(self, era: float, team: str) -> float:
        """Calculate ERA- (100 = league average, lower is better)."""
        park_factor = BALLPARK_FACTORS.get(str(team).upper().strip(), 100) / 100
        # Park-adjusted ERA
        adj_era = era / park_factor if park_factor != 0 else era
        return (adj_era / LG_ERA) * 100
    
    def calculate_fip_minus(self, fip: float, team: str) -> float:
        """Calculate FIP- (100 = league average, lower is better)."""
        park_factor = BALLPARK_FACTORS.get(str(team).upper().strip(), 100) / 100
        adj_fip = fip / park_factor if park_factor != 0 else fip
        return (adj_fip / LG_FIP) * 100


# Singleton instance for convenience
_calculator = None

def get_calculator() -> ValueCalculator:
    """Get singleton ValueCalculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = ValueCalculator()
    return _calculator
