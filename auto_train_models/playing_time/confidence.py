"""
Confidence Calculator for Playing Time Allocation

This module computes confidence scores for each player based on:
1. Statistical confidence - derived from backtest error curves and sample size
2. Prospect confidence - based on historical prospect rankings
3. Consistency confidence - based on historical playing time stability

These confidence scores inform playing time allocation by:
- High confidence players get "locked in" to starter roles
- Low confidence players compete for remaining playing time
- Confidence affects the "leash" a team gives to struggling players

Author: Niels Christoffersen
Date: January 2026
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================================================================
# ERROR CURVE PARAMETERS (from backtest_confidence.py results)
# =============================================================================

# Format: {model_type: {metric: {'a': float, 'b': float, 'r_squared': float}}}
# Error curve: RMSE = a / sqrt(sample_size) + b
# 'b' is the irreducible error floor (even with infinite data)

ERROR_CURVE_PARAMS = {
    'batter': {
        'wOBA': {'a': 0.2366, 'b': 0.0280, 'r_squared': 0.586},
        'K%': {'a': 0.3786, 'b': 0.0230, 'r_squared': 0.700},
        'BB%': {'a': 0.1118, 'b': 0.0192, 'r_squared': 0.534},
        'wRC+': {'a': 159.86, 'b': 18.54, 'r_squared': 0.542},
    },
    'SP': {
        'ERA': {'a': 5.495, 'b': 0.808, 'r_squared': 0.386},
        'FIP': {'a': 4.517, 'b': 0.556, 'r_squared': 0.550},
        'K%': {'a': 0.1115, 'b': 0.0274, 'r_squared': 0.420},
        'BB%': {'a': 0.1009, 'b': 0.0124, 'r_squared': 0.572},
    },
    'RP': {
        'ERA': {'a': 1.521, 'b': 1.461, 'r_squared': 0.059},
        'FIP': {'a': 1.398, 'b': 0.958, 'r_squared': 0.093},
        'K%': {'a': 0.1211, 'b': 0.0352, 'r_squared': 0.463},
        'BB%': {'a': 0.0881, 'b': 0.0177, 'r_squared': 0.570},
    }
}

# Reference sample sizes for normalization
# These represent "minimum meaningful sample" and "high confidence sample"
SAMPLE_REFERENCE = {
    'batter': {'min': 100, 'high': 2000, 'col': 'PA', 'seq_length': 5},
    'SP': {'min': 50, 'high': 600, 'col': 'IP', 'seq_length': 3},
    'RP': {'min': 30, 'high': 200, 'col': 'IP', 'seq_length': 3},
}

# Sample size thresholds for prospect confidence decay
# Once a player has enough MLB sample, we trust actual data over prospect pedigree
PROSPECT_SAMPLE_THRESHOLDS = {
    'batter': {'fade_start': 300, 'fade_complete': 1000},  # PA
    'SP': {'fade_start': 100, 'fade_complete': 400},       # IP
    'RP': {'fade_start': 50, 'fade_complete': 150},        # IP
}

# FV Grade to confidence mapping (from FanGraphs scouting scale)
# Higher FV = higher expected MLB contribution
FV_GRADE_MAP = {
    80: 1.00,  # Hall of Fame potential
    75: 0.95,  # Perennial All-Star
    70: 0.90,  # All-Star
    65: 0.85,  # Above-Average Regular
    60: 0.75,  # Plus Regular
    55: 0.60,  # Average Regular
    50: 0.45,  # Fringe Regular / Platoon
    45: 0.30,  # Up-and-down player
    40: 0.15,  # Org depth
    35: 0.05,  # Filler
}


@dataclass
class ConfidenceComponents:
    """Detailed breakdown of confidence score."""
    statistical: float  # 0-1, from backtest error curves (projection accuracy)
    role: float         # 0-1, from GS/G ratio (starter vs bench)
    playing_time: float # 0-1, from PT level (full-time vs part-time)
    prospect: float     # 0-1, from prospect ranking history
    combined: float     # Final weighted score
    
    # Metadata
    sample_size: float
    start_rate: Optional[float] = None  # GS/G ratio from fielding data
    games_played: Optional[int] = None  # Games in most recent season
    expected_rmse_woba: Optional[float] = None  # For batters
    expected_rmse_fip: Optional[float] = None   # For pitchers
    prospect_rank: Optional[int] = None
    years_since_prospect: Optional[int] = None


class ConfidenceCalculator:
    """
    Calculate confidence scores for playing time allocation.
    
    Confidence score (0-1) indicates how much certainty we have in projections.
    Higher confidence = more likely to get locked into a defined role.
    Lower confidence = more likely to compete for playing time.
    """
    
    # Weights for combining confidence components
    # Note: These are BASE weights - prospect weight gets dynamically reduced as sample size increases
    WEIGHTS = {
        'batter': {'statistical': 0.70, 'prospect': 0.20, 'consistency': 0.10},
        'pitcher': {'statistical': 0.70, 'prospect': 0.15, 'consistency': 0.15},
    }
    
    def __init__(self, 
                 historical_batting_path: Optional[Path] = None,
                 historical_pitching_path: Optional[Path] = None,
                 prospects_path: Optional[Path] = None):
        """
        Initialize with paths to historical data.
        
        Args:
            historical_batting_path: Path to MLB batting data
            historical_pitching_path: Path to MLB pitching data
            prospects_path: Path to prospect rankings data
        """
        # Default paths
        data_dir = Path(__file__).parent.parent.parent / 'data'
        
        self.batting_path = historical_batting_path or (
            data_dir / 'historic_mlb' / 'mlb_batting_data_1950_2025.csv'
        )
        self.pitching_path = historical_pitching_path or (
            data_dir / 'historic_mlb' / 'mlb_pitching_data_1950_2025.csv'
        )
        self.prospects_path = prospects_path or (
            data_dir / 'prospect_data' / 'prospects_2014_2026_with_top100.csv'
        )
        self.fielding_path = data_dir / 'historic_mlb' / 'mlb_fielding_data_2000_2025.csv'
        
        # Cached data
        self._batting_df: Optional[pd.DataFrame] = None
        self._pitching_df: Optional[pd.DataFrame] = None
        self._prospects_df: Optional[pd.DataFrame] = None
        self._fielding_df: Optional[pd.DataFrame] = None
        
    def _load_batting_data(self) -> pd.DataFrame:
        """Load historical batting data."""
        if self._batting_df is None:
            if self.batting_path.exists():
                self._batting_df = pd.read_csv(self.batting_path)
                logger.info(f"Loaded {len(self._batting_df)} batting records")
            else:
                logger.warning(f"Batting data not found: {self.batting_path}")
                self._batting_df = pd.DataFrame()
        return self._batting_df
    
    def _load_pitching_data(self) -> pd.DataFrame:
        """Load historical pitching data."""
        if self._pitching_df is None:
            if self.pitching_path.exists():
                self._pitching_df = pd.read_csv(self.pitching_path)
                logger.info(f"Loaded {len(self._pitching_df)} pitching records")
            else:
                logger.warning(f"Pitching data not found: {self.pitching_path}")
                self._pitching_df = pd.DataFrame()
        return self._pitching_df
    
    def _load_prospect_data(self) -> pd.DataFrame:
        """Load prospect rankings data."""
        if self._prospects_df is None:
            if self.prospects_path.exists():
                self._prospects_df = pd.read_csv(self.prospects_path)
                logger.info(f"Loaded {len(self._prospects_df)} prospect records")
            else:
                logger.warning(f"Prospect data not found: {self.prospects_path}")
                self._prospects_df = pd.DataFrame()
        return self._prospects_df
    
    def _load_fielding_data(self) -> pd.DataFrame:
        """Load historical fielding data with games started info."""
        if self._fielding_df is None:
            if self.fielding_path.exists():
                self._fielding_df = pd.read_csv(self.fielding_path)
                logger.info(f"Loaded {len(self._fielding_df)} fielding records")
            else:
                logger.warning(f"Fielding data not found: {self.fielding_path}")
                self._fielding_df = pd.DataFrame()
        return self._fielding_df
    
    # =========================================================================
    # STATISTICAL CONFIDENCE
    # =========================================================================
    
    def _calculate_rmse(self, sample_size: float, model_type: str, metric: str) -> float:
        """
        Calculate expected RMSE given sample size using fitted error curves.
        
        RMSE = a / sqrt(sample_size) + b
        """
        params = ERROR_CURVE_PARAMS.get(model_type, {}).get(metric, {})
        a = params.get('a', 0.3)
        b = params.get('b', 0.03)
        
        if sample_size <= 0:
            return a + b  # Maximum error for zero sample
        
        return a / np.sqrt(sample_size) + b
    
    def _calculate_statistical_confidence(self,
                                          sample_size: float,
                                          model_type: str) -> Tuple[float, float]:
        """
        Calculate statistical confidence from sample size.
        
        Uses a sigmoid-like function that:
        - Starts near 0 for minimal sample
        - Reaches ~0.8 at "high" reference sample (2000 PA)
        - Approaches 0.99 asymptotically for very large samples
        
        For established players (3000+ PA), confidence should be ~0.95+
        
        Returns:
            (confidence_score, expected_rmse)
        """
        ref = SAMPLE_REFERENCE.get(model_type, SAMPLE_REFERENCE['batter'])
        min_sample = ref['min']
        high_sample = ref['high']
        
        # Use wOBA for batters, FIP for pitchers as the primary metric
        metric = 'wOBA' if model_type == 'batter' else 'FIP'
        
        # Calculate expected RMSE at this sample size
        rmse_at_sample = self._calculate_rmse(max(sample_size, 1), model_type, metric)
        
        # Use a sigmoid transformation for confidence
        # This ensures high samples get appropriately high confidence
        # k controls steepness, x0 is the midpoint
        
        if sample_size <= 0:
            confidence = 0.0
        else:
            # Parameters tuned so:
            # - 100 PA → ~0.3 confidence
            # - 500 PA → ~0.7 confidence  
            # - 1000 PA → ~0.85 confidence
            # - 2000 PA → ~0.95 confidence
            # - 3000+ PA → ~0.98 confidence
            
            # Sigmoid: 1 / (1 + exp(-k*(x - x0)))
            # We use log(sample) for better scaling
            if model_type == 'batter':
                # For batters: midpoint at ~400 PA, reaches 0.95 at ~2000 PA
                k = 1.5
                x0 = np.log(400)
                log_sample = np.log(sample_size)
                raw_confidence = 1 / (1 + np.exp(-k * (log_sample - x0)))
            else:
                # For pitchers: midpoint at ~150 IP, reaches 0.95 at ~500 IP
                k = 1.8
                x0 = np.log(150) if model_type == 'SP' else np.log(80)
                log_sample = np.log(sample_size)
                raw_confidence = 1 / (1 + np.exp(-k * (log_sample - x0)))
            
            # Scale to max out around 0.99 (never quite 1.0 - always some uncertainty)
            confidence = raw_confidence * 0.99
        
        return confidence, rmse_at_sample
    
    def calculate_sample_size(self,
                              player_id: int,
                              cutoff_year: int,
                              model_type: str) -> float:
        """
        Calculate cumulative sample size over the model's sequence length.
        
        Args:
            player_id: FanGraphs IDfg
            cutoff_year: Last year of data to include
            model_type: 'batter', 'SP', or 'RP'
        
        Returns:
            Total PA (batters) or IP (pitchers) over sequence window
        """
        ref = SAMPLE_REFERENCE.get(model_type, SAMPLE_REFERENCE['batter'])
        seq_length = ref['seq_length']
        sample_col = ref['col']
        
        # Load appropriate data
        if model_type == 'batter':
            df = self._load_batting_data()
        else:
            df = self._load_pitching_data()
        
        if df.empty:
            return 0.0
        
        # Filter to player and years up to cutoff
        player_data = df[
            (df['IDfg'] == player_id) & 
            (df['Season'] <= cutoff_year)
        ].copy()
        
        if player_data.empty:
            return 0.0
        
        # Get last seq_length seasons
        player_data = player_data.sort_values('Season').tail(seq_length)
        
        # Sum sample size
        if sample_col in player_data.columns:
            return player_data[sample_col].sum()
        
        return 0.0
    
    # =========================================================================
    # PROSPECT CONFIDENCE
    # =========================================================================
    
    def _get_best_prospect_info(self, 
                                player_name: str,
                                player_id: Optional[int] = None) -> Dict:
        """
        Get player's best historical prospect information.
        
        Returns dict with:
            - best_org_rank: Best organizational rank (1-30)
            - best_year: Year of best ranking
            - was_top_100: Whether player was ever top 100 prospect
            - best_fv_grade: Best FV (Future Value) grade if available
        """
        prospects_df = self._load_prospect_data()
        
        default_result = {
            'best_org_rank': None,
            'best_year': None, 
            'was_top_100': False,
            'best_top_100_rank': None,
            'best_fv_grade': None
        }
        
        if prospects_df.empty:
            return default_result
        
        # Try to match by name (normalized)
        player_name_lower = player_name.lower().strip()
        
        # Find matching prospect entries
        matches = prospects_df[
            prospects_df['name'].str.lower().str.strip() == player_name_lower
        ]
        
        if matches.empty:
            # Try fuzzy matching on first/last name
            name_parts = player_name_lower.split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                first_name = name_parts[0]
                # Match on last name and first initial
                matches = prospects_df[
                    (prospects_df['name'].str.lower().str.contains(last_name, na=False)) &
                    (prospects_df['name'].str.lower().str.startswith(first_name[0], na=False))
                ]
        
        if matches.empty:
            return default_result
        
        # Get best (lowest) organization rank
        best_org_rank = matches['rank'].min()
        best_row = matches[matches['rank'] == best_org_rank].iloc[0]
        best_year = int(best_row['year']) if pd.notna(best_row['year']) else None
        
        # Check if was top 100 and get best top 100 rank
        was_top_100 = False
        best_top_100_rank = None
        if 'top_100' in matches.columns:
            top_100_matches = matches[matches['top_100'].notna()]
            if not top_100_matches.empty:
                was_top_100 = True
                best_top_100_rank = int(top_100_matches['top_100'].min())  # Best (lowest) rank
        
        # Get best FV grade if available
        best_fv_grade = None
        if 'grade_overall' in matches.columns:
            fv_values = matches['grade_overall'].dropna()
            if len(fv_values) > 0:
                best_fv_grade = fv_values.max()  # Higher FV = better
        
        return {
            'best_org_rank': best_org_rank,
            'best_year': best_year,
            'was_top_100': was_top_100,
            'best_top_100_rank': best_top_100_rank,
            'best_fv_grade': best_fv_grade
        }
    
    def _calculate_prospect_confidence(self,
                                       player_name: str,
                                       current_year: int,
                                       player_age: int,
                                       sample_size: float,
                                       model_type: str,
                                       player_id: Optional[int] = None) -> Tuple[float, Optional[int], Optional[int]]:
        """
        Calculate confidence boost from prospect pedigree.
        
        CRITICAL: Prospect confidence should FADE as we accumulate MLB sample size.
        Once we have enough data on a player, we know more about them from actual
        performance than any prospect ranking could tell us.
        
        The prospect boost is primarily for young players with limited MLB track record
        who were highly regarded - we give them extended leash based on pedigree.
        
        Args:
            player_name: Player's full name
            current_year: Projection year
            player_age: Player's age in projection year  
            sample_size: Player's MLB sample size (PA or IP)
            model_type: 'batter', 'SP', or 'RP'
            player_id: Optional FanGraphs ID
            
        Returns:
            (prospect_confidence, best_rank, years_since)
        """
        prospect_info = self._get_best_prospect_info(player_name, player_id)
        
        best_rank = prospect_info['best_org_rank']
        rank_year = prospect_info['best_year']
        was_top_100 = prospect_info['was_top_100']
        best_top_100_rank = prospect_info['best_top_100_rank']
        fv_grade = prospect_info['best_fv_grade']
        
        if best_rank is None:
            return 0.0, None, None
        
        years_since = current_year - rank_year if rank_year else 5
        
        # =================================================================
        # Step 1: Calculate base prospect grade from FV, top 100 rank, or org rank
        # =================================================================
        if fv_grade is not None and fv_grade > 0:
            # Use FV grade mapping (higher FV = better prospect)
            # Find closest FV grade in our mapping
            fv_grades = sorted(FV_GRADE_MAP.keys())
            closest_fv = min(fv_grades, key=lambda x: abs(x - fv_grade))
            base_confidence = FV_GRADE_MAP[closest_fv]
        elif best_top_100_rank is not None:
            # Use granular top 100 rank for better confidence differentiation
            if best_top_100_rank <= 5:
                base_confidence = 0.95  # Elite top 5 prospect
            elif best_top_100_rank <= 10:
                base_confidence = 0.90  # Top 10 prospect
            elif best_top_100_rank <= 25:
                base_confidence = 0.85  # Top 25 prospect
            elif best_top_100_rank <= 50:
                base_confidence = 0.80  # Top 50 prospect
            elif best_top_100_rank <= 75:
                base_confidence = 0.75  # Top 75 prospect
            else:
                base_confidence = 0.70  # Back end of top 100
        else:
            # Fallback to org rank only (no top 100 data)
            if best_rank == 1:
                base_confidence = 0.70  # #1 org prospect
            elif best_rank <= 3:
                base_confidence = 0.60
            elif best_rank <= 5:
                base_confidence = 0.50
            elif best_rank <= 10:
                base_confidence = 0.40
            elif best_rank <= 15:
                base_confidence = 0.30
            elif best_rank <= 20:
                base_confidence = 0.20
            else:
                base_confidence = 0.15  # Lower org rank
        
        # =================================================================
        # Step 2: Apply SAMPLE SIZE DECAY - this is the key fix!
        # Once we have enough MLB data, prospect ranking becomes irrelevant
        # =================================================================
        thresholds = PROSPECT_SAMPLE_THRESHOLDS.get(model_type, PROSPECT_SAMPLE_THRESHOLDS['batter'])
        fade_start = thresholds['fade_start']
        fade_complete = thresholds['fade_complete']
        
        if sample_size >= fade_complete:
            # We have enough MLB data - prospect ranking doesn't matter anymore
            sample_decay = 0.0
        elif sample_size <= fade_start:
            # Limited MLB data - prospect ranking is highly relevant
            sample_decay = 1.0
        else:
            # Linear fade between thresholds
            sample_decay = 1.0 - (sample_size - fade_start) / (fade_complete - fade_start)
        
        # =================================================================
        # Step 3: Apply time decay - prospect rankings from long ago matter less
        # =================================================================
        if years_since <= 2:
            time_decay = 1.0
        elif years_since <= 4:
            time_decay = 0.7
        elif years_since <= 6:
            time_decay = 0.4
        else:
            time_decay = 0.2  # Still some residual value for former top prospects
        
        # =================================================================
        # Step 4: Combine all factors
        # =================================================================
        prospect_confidence = base_confidence * sample_decay * time_decay
        
        return prospect_confidence, best_rank, years_since
    
    # =========================================================================
    # ROLE CONFIDENCE (Starter vs Bench)
    # =========================================================================
    
    def _calculate_role_confidence(self,
                                   player_id: int,
                                   cutoff_year: int,
                                   model_type: str) -> Tuple[float, Optional[float], Optional[int]]:
        """
        Calculate role confidence based on games started vs games played.
        
        This is the KEY metric for playing time allocation:
        - Start Rate ~100% = everyday starter (high confidence)
        - Start Rate ~70-90% = platoon or injury fill-in (medium confidence)  
        - Start Rate <70% = bench player (low confidence)
        
        For pitchers:
        - SP: Use GS from pitching data
        - RP: Use appearances (G) - most games = closer/setup role
        
        Returns:
            (role_confidence, start_rate, games_played)
        """
        if model_type == 'batter':
            return self._calculate_batter_role_confidence(player_id, cutoff_year)
        elif model_type == 'SP':
            return self._calculate_sp_role_confidence(player_id, cutoff_year)
        else:  # RP
            return self._calculate_rp_role_confidence(player_id, cutoff_year)
    
    def _calculate_batter_role_confidence(self,
                                          player_id: int,
                                          cutoff_year: int) -> Tuple[float, Optional[float], Optional[int]]:
        """
        Calculate role confidence for position players using fielding + batting data.
        
        Uses GS from fielding data divided by G from BATTING data.
        This avoids double-counting when players switch positions mid-game
        (fielding data has multiple rows per position, batting has one row per player).
        
        Uses 2-year window to capture recent role changes and injuries.
        """
        fielding_df = self._load_fielding_data()
        batting_df = self._load_batting_data()
        
        if fielding_df.empty or batting_df.empty:
            return 0.5, None, None  # Neutral if no data
        
        # Use 2-year window (more responsive to recent role changes/injuries)
        lookback = 2
        
        # Get fielding data for GS
        fielding_data = fielding_df[
            (fielding_df['IDfg'] == player_id) & 
            (fielding_df['Season'] <= cutoff_year) &
            (fielding_df['Season'] > cutoff_year - lookback)
        ]
        
        # Get batting data for G (true games played)
        batting_data = batting_df[
            (batting_df['IDfg'] == player_id) & 
            (batting_df['Season'] <= cutoff_year) &
            (batting_df['Season'] > cutoff_year - lookback)
        ]
        
        if fielding_data.empty or batting_data.empty:
            return 0.4, None, None  # Below average for players without history
        
        # Sum GS from fielding (across all positions)
        total_gs = fielding_data['GS'].sum()
        
        # Sum G from batting (true games played, no double-counting)
        total_g = batting_data['G'].sum() if 'G' in batting_data.columns else 0
        
        if total_g == 0:
            return 0.3, 0.0, 0
        
        start_rate = min(total_gs / total_g, 1.0)  # Cap at 1.0
        
        # Also factor in VOLUME - how many games did they actually play?
        # ~150 games/year for 2 years = 300 games = full starter
        # Injuries show up here as lower game counts
        expected_games = 150 * lookback  # ~300 games over 2 years
        games_factor = min(total_g / expected_games, 1.0)
        
        # Role confidence formula:
        # - Start rate is primary (60% weight) - are they starting when available?
        # - Games played (40% weight) - are they actually available? (captures injuries)
        role_confidence = 0.6 * start_rate + 0.4 * games_factor
        
        return role_confidence, start_rate, int(total_g)
    
    def _calculate_sp_role_confidence(self,
                                      player_id: int,
                                      cutoff_year: int) -> Tuple[float, Optional[float], Optional[int]]:
        """
        Calculate role confidence for starting pitchers.
        
        Uses GS from pitching data - more starts = more established in rotation.
        """
        pitching_df = self._load_pitching_data()
        
        if pitching_df.empty:
            return 0.5, None, None
        
        # Get most recent season
        player_data = pitching_df[
            (pitching_df['IDfg'] == player_id) & 
            (pitching_df['Season'] == cutoff_year)
        ]
        
        if player_data.empty:
            return 0.4, None, None
        
        gs = player_data['GS'].sum() if 'GS' in player_data.columns else 0
        g = player_data['G'].sum() if 'G' in player_data.columns else 0
        
        if g == 0:
            return 0.3, 0.0, 0
        
        # For SP, we want high GS and GS should be most of their appearances
        start_rate = gs / g if g > 0 else 0
        
        # Established starter = 25+ starts
        gs_factor = min(gs / 28, 1.0)  # Normalize to ~28 starts for full season
        
        # If they're making relief appearances too, they're less locked in
        role_confidence = 0.5 * start_rate + 0.5 * gs_factor
        
        return role_confidence, start_rate, int(gs)
    
    def _calculate_rp_role_confidence(self,
                                      player_id: int,
                                      cutoff_year: int) -> Tuple[float, Optional[float], Optional[int]]:
        """
        Calculate role confidence for relief pitchers.
        
        For relievers, more appearances = more trusted by manager.
        Closers/setup men get 60-70 appearances, mop-up guys get 30-40.
        """
        pitching_df = self._load_pitching_data()
        
        if pitching_df.empty:
            return 0.5, None, None
        
        # Get most recent season
        player_data = pitching_df[
            (pitching_df['IDfg'] == player_id) & 
            (pitching_df['Season'] == cutoff_year)
        ]
        
        if player_data.empty:
            return 0.4, None, None
        
        g = player_data['G'].sum() if 'G' in player_data.columns else 0
        gs = player_data['GS'].sum() if 'GS' in player_data.columns else 0
        
        if g == 0:
            return 0.3, 0.0, 0
        
        # Pure reliever should have 0 or very few starts
        relief_purity = 1.0 - (gs / g) if g > 0 else 1.0
        
        # High-leverage relievers get 55-70 appearances
        # Normalize: 60+ games = 1.0, 30 games = 0.5, <20 = low
        appearances_factor = min(g / 55, 1.0)
        
        role_confidence = 0.4 * relief_purity + 0.6 * appearances_factor
        
        return role_confidence, relief_purity, int(g)
    
    # =========================================================================
    # PLAYING TIME LEVEL (Full-time vs Part-time)
    # =========================================================================
    
    def _calculate_consistency_confidence(self,
                                          player_id: int,
                                          cutoff_year: int,
                                          model_type: str,
                                          lookback_years: int = 3) -> float:
        """
        Calculate confidence based on historical playing time consistency.
        
        Players who consistently get playing time are likely to continue.
        High variance in PT = lower confidence.
        
        Returns:
            consistency_confidence (0-1)
        """
        ref = SAMPLE_REFERENCE.get(model_type, SAMPLE_REFERENCE['batter'])
        sample_col = ref['col']
        
        # Load appropriate data
        if model_type == 'batter':
            df = self._load_batting_data()
        else:
            df = self._load_pitching_data()
        
        if df.empty:
            return 0.5  # Neutral confidence
        
        # Filter to player and recent years
        player_data = df[
            (df['IDfg'] == player_id) & 
            (df['Season'] <= cutoff_year) &
            (df['Season'] > cutoff_year - lookback_years)
        ].copy()
        
        if len(player_data) < 2:
            return 0.3  # Low confidence for limited history
        
        if sample_col not in player_data.columns:
            return 0.5
        
        # Calculate coefficient of variation (std / mean)
        mean_sample = player_data[sample_col].mean()
        std_sample = player_data[sample_col].std()
        
        if mean_sample <= 0:
            return 0.3
        
        cv = std_sample / mean_sample
        
        # Also factor in the level of playing time
        # High average PT + low variance = high confidence
        ref_min = ref['min']
        ref_high = ref['high']
        
        # Playing time factor (0 at min, 1 at high)
        pt_factor = np.clip((mean_sample - ref_min) / (ref_high - ref_min), 0, 1)
        
        # Consistency factor (1 at 0 variance, 0 at cv >= 1)
        consistency_factor = np.clip(1 - cv, 0, 1)
        
        # Combine: need both high PT and consistency for high confidence
        consistency_confidence = 0.5 * pt_factor + 0.5 * consistency_factor
        
        return consistency_confidence
    
    # =========================================================================
    # COMBINED CONFIDENCE
    # =========================================================================
    
    def calculate_confidence(self,
                            player_id: int,
                            player_name: str,
                            player_age: int,
                            model_type: str,
                            projection_year: int) -> ConfidenceComponents:
        """
        Calculate combined confidence score for a player.
        
        This score measures TWO distinct concepts:
        1. PROJECTION ACCURACY: How well do we know this player's true talent?
           (Based on sample size, model RMSE curves, prospect pedigree)
        2. ROLE CERTAINTY: Will this player actually get playing time?
           (Based on games started / games played ratio - distinguishes starters from bench)
        
        For playing time allocation, ROLE CERTAINTY is weighted heavily because
        a bench player with 1,000 career PA should NOT get starter-level playing time
        just because we have good data on them.
        
        Args:
            player_id: FanGraphs IDfg
            player_name: Player's full name
            player_age: Player's age in projection year
            model_type: 'batter', 'SP', or 'RP'
            projection_year: Year being projected
        
        Returns:
            ConfidenceComponents with all scores and metadata
        """
        cutoff_year = projection_year - 1
        
        # =====================================================================
        # PROJECTION ACCURACY COMPONENTS
        # =====================================================================
        
        # 1. Statistical confidence (how much sample do we have?)
        sample_size = self.calculate_sample_size(player_id, cutoff_year, model_type)
        stat_conf, expected_rmse = self._calculate_statistical_confidence(sample_size, model_type)
        
        # 2. Prospect confidence (for players with limited sample)
        prospect_conf, prospect_rank, years_since = self._calculate_prospect_confidence(
            player_name=player_name,
            current_year=projection_year,
            player_age=player_age,
            sample_size=sample_size,
            model_type=model_type,
            player_id=player_id
        )
        
        # =====================================================================
        # ROLE CERTAINTY COMPONENTS  
        # =====================================================================
        
        # 3. Role confidence (starter vs bench based on games started ratio)
        role_conf, start_rate, games_played = self._calculate_role_confidence(
            player_id, cutoff_year, model_type
        )
        
        # 4. Playing time level confidence (consistency + level from _calculate_consistency_confidence)
        playing_time_conf = self._calculate_consistency_confidence(
            player_id, cutoff_year, model_type
        )
        
        # =====================================================================
        # COMBINED SCORE
        # =====================================================================
        # Role confidence is the DOMINANT factor for playing time allocation
        # because a bench player (60% start rate) should NOT get full-time PA
        # regardless of how much historical data we have.
        #
        # Weighting philosophy:
        # - Role confidence (50%): Are they a starter or bench player?
        # - Statistical confidence (25%): How accurate is our projection?
        # - Playing time level (15%): What's their typical PT and consistency?
        # - Prospect confidence (10%): For young players with limited sample
        
        # Base combined score with role-dominant weighting
        combined = (
            0.50 * role_conf +
            0.25 * stat_conf +
            0.15 * playing_time_conf +
            0.10 * prospect_conf
        )
        
        # Adjustments for special cases
        
        # Young players with high prospect pedigree but no role yet
        # (e.g., top prospects called up mid-season)
        if player_age <= 23 and prospect_rank is not None and prospect_rank <= 50:
            # Don't penalize too harshly for not having established role
            # They're likely to GET the starting role
            combined = max(combined, 0.55 + (50 - prospect_rank) * 0.005)
        
        # Very established players (3000+ PA batters, 500+ IP pitchers)
        # who might have had injury/rest in recent year
        elite_threshold = 3000 if model_type == 'batter' else 500
        if sample_size >= elite_threshold and stat_conf >= 0.95:
            # Ensure floor for truly elite players
            combined = max(combined, 0.60)
        
        return ConfidenceComponents(
            statistical=stat_conf,
            prospect=prospect_conf,
            role=role_conf,                  # Starter vs bench
            playing_time=playing_time_conf,  # PT level and consistency
            combined=combined,
            sample_size=sample_size,
            expected_rmse_woba=expected_rmse if model_type == 'batter' else None,
            expected_rmse_fip=expected_rmse if model_type != 'batter' else None,
            prospect_rank=prospect_rank,
            years_since_prospect=years_since,
            start_rate=start_rate,           # GS/G ratio
            games_played=games_played        # Recent games
        )
    
    def calculate_batch_confidence(self,
                                   players_df: pd.DataFrame,
                                   model_type: str,
                                   projection_year: int) -> pd.DataFrame:
        """
        Calculate confidence for a batch of players.
        
        Args:
            players_df: DataFrame with columns: IDfg, Name, Age
            model_type: 'batter', 'SP', or 'RP'
            projection_year: Year being projected
        
        Returns:
            DataFrame with confidence scores appended
        """
        results = []
        
        for _, row in players_df.iterrows():
            player_id = row['IDfg']
            player_name = row.get('Name', '')
            player_age = row.get('Age', 27)
            
            conf = self.calculate_confidence(
                player_id=player_id,
                player_name=player_name,
                player_age=int(player_age),
                model_type=model_type,
                projection_year=projection_year
            )
            
            results.append({
                'IDfg': player_id,
                'confidence': conf.combined,
                'conf_statistical': conf.statistical,
                'conf_prospect': conf.prospect,
                'conf_role': conf.role,               # Starter vs bench
                'conf_pt_level': conf.playing_time,   # PT level and consistency
                'sample_size': conf.sample_size,
                'prospect_rank': conf.prospect_rank,
                'start_rate': conf.start_rate,        # GS/G ratio
            })
        
        conf_df = pd.DataFrame(results)
        
        # Merge back
        return players_df.merge(conf_df, on='IDfg', how='left')


# =============================================================================
# CONFIDENCE-BASED ALLOCATION HELPERS
# =============================================================================

def get_confidence_tier(confidence: float) -> str:
    """
    Map confidence score to allocation tier.
    
    Tiers determine how playing time is allocated:
    - 'locked': High confidence, gets starter games guaranteed
    - 'favored': Moderate-high confidence, strong claim to playing time
    - 'competitive': Moderate confidence, competes for time
    - 'fringe': Low confidence, fills remaining games
    """
    if confidence >= 0.7:
        return 'locked'
    elif confidence >= 0.5:
        return 'favored'
    elif confidence >= 0.3:
        return 'competitive'
    else:
        return 'fringe'


def get_tier_games(tier: str, position: str) -> Tuple[int, int]:
    """
    Get (min_games, max_games) for a confidence tier at a position.
    
    Returns:
        (minimum guaranteed games, maximum possible games)
    """
    # Base games by tier
    TIER_GAMES = {
        'locked': {'min_pct': 0.85, 'max_pct': 1.0},
        'favored': {'min_pct': 0.50, 'max_pct': 0.95},
        'competitive': {'min_pct': 0.15, 'max_pct': 0.70},
        'fringe': {'min_pct': 0.0, 'max_pct': 0.40},
    }
    
    # Full season games by position
    FULL_GAMES = {
        'C': 130,  # Catchers play fewer
        'INF': 155,  # Infielders
        'OF': 155,  # Outfielders
        'DH': 155,  # DH
        'SP': 32,  # Starts (translates to ~190 IP)
        'RP': 65,  # Appearances
    }
    
    full = FULL_GAMES.get(position, 150)
    tier_info = TIER_GAMES.get(tier, TIER_GAMES['competitive'])
    
    min_games = int(full * tier_info['min_pct'])
    max_games = int(full * tier_info['max_pct'])
    
    return min_games, max_games
