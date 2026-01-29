"""
Injury data processing and adjustment calculation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, List, NamedTuple
from collections import defaultdict

from .config import Config, InjuryConfig


class InjuryInfo(NamedTuple):
    """Injury information for a player."""
    multiplier: float
    injury_name: Optional[str] = None
    return_date: Optional[str] = None

logger = logging.getLogger(__name__)


class InjuryProcessor:
    """Process injury data and calculate playing time adjustments."""
    
    def __init__(self, injury_data: pd.DataFrame, config: Config = None):
        self.config = config or Config()
        self.injury_config = self.config.injury
        self.raw_data = injury_data
        self._processed: Optional[pd.DataFrame] = None
        self._player_summaries: Optional[pd.DataFrame] = None
        
    def process(self) -> pd.DataFrame:
        """Process raw injury data into structured format."""
        if self._processed is not None:
            return self._processed
        
        if self.raw_data.empty:
            logger.warning("No injury data to process")
            return pd.DataFrame()
        
        df = self.raw_data.copy()
        
        # Parse dates
        df['injury_date_parsed'] = pd.to_datetime(df['injury_date'], errors='coerce')
        df['return_date_parsed'] = pd.to_datetime(df['return_date'], errors='coerce')
        df['il_retro_date_parsed'] = pd.to_datetime(df['il_retro_date'], errors='coerce')
        
        # Classify injury severity
        df['is_major_surgery'] = df['injury_surgery'].apply(self._is_major_surgery)
        df['surgery_type'] = df['injury_surgery'].apply(self._classify_surgery)
        
        # Calculate days lost
        df['days_lost'] = self._calculate_days_lost(df)
        
        # Flag active injuries (only from current projection year)
        current_year = self.config.CURRENT_YEAR
        df['is_active'] = (
            (df['season'] == current_year) & 
            (df['return_date_parsed'].isna() | 
             df['latest_update'].str.contains('No timetable|Out for|Questionable|Projected Injured', case=False, na=False))
        )
        
        self._processed = df
        return df
    
    def _is_major_surgery(self, injury_str: str) -> bool:
        """Check if injury is a major surgery."""
        if pd.isna(injury_str):
            return False
        
        injury_lower = str(injury_str).lower()
        for surgery_type in self.injury_config.MAJOR_SURGERIES.keys():
            if surgery_type in injury_lower:
                return True
        return False
    
    def _classify_surgery(self, injury_str: str) -> Optional[str]:
        """Classify the type of major surgery."""
        if pd.isna(injury_str):
            return None
        
        injury_lower = str(injury_str).lower()
        for surgery_type in self.injury_config.MAJOR_SURGERIES.keys():
            if surgery_type in injury_lower:
                return surgery_type
        return None
    
    def _calculate_days_lost(self, df: pd.DataFrame) -> pd.Series:
        """Calculate days lost to injury."""
        days = pd.Series(index=df.index, dtype=float)
        
        for idx, row in df.iterrows():
            if pd.notna(row['return_date_parsed']) and pd.notna(row['il_retro_date_parsed']):
                delta = row['return_date_parsed'] - row['il_retro_date_parsed']
                days.loc[idx] = max(0, delta.days)
            elif row['latest_update'] == 'Out for 2020 season':
                # Estimate remaining season
                days.loc[idx] = 60  # Approximate
            else:
                days.loc[idx] = np.nan
        
        return days
    
    def get_player_injury_summary(self, fg_id: int = None, 
                                   name: str = None) -> Dict:
        """
        Get injury summary for a player.
        
        Args:
            fg_id: FanGraphs player ID
            name: Player name (fallback if no fg_id)
            
        Returns:
            Dict with injury summary statistics
        """
        df = self.process()
        
        if df.empty:
            return self._empty_summary()
        
        # Filter to player
        if fg_id is not None and not pd.isna(fg_id):
            player_df = df[df['fg_id'] == fg_id]
        elif name is not None:
            from .data_loader import DataLoader
            normalized = DataLoader._normalize_name(name)
            player_df = df[df['name_normalized'] == normalized]
        else:
            return self._empty_summary()
        
        if player_df.empty:
            return self._empty_summary()
        
        current_year = self.config.CURRENT_YEAR
        lookback_start = current_year - self.injury_config.IL_LOOKBACK_YEARS
        
        # Recent injuries (within lookback)
        recent = player_df[player_df['season'] >= lookback_start]
        
        # Count IL stints
        il_stints = len(recent)
        
        # Major surgeries
        major_surgeries = player_df[player_df['is_major_surgery']]
        
        # Most recent major surgery
        latest_surgery = None
        surgery_year = None
        latest_injury_name = None
        latest_return_date = None
        if not major_surgeries.empty:
            latest = major_surgeries.sort_values('season', ascending=False).iloc[0]
            latest_surgery = latest['surgery_type']
            surgery_year = latest['season']
            # Only use injury name/date from current year for display
            if latest['season'] == current_year:
                latest_injury_name = latest.get('injury_surgery')
                # Get return date if available
                if pd.notna(latest.get('return_date')):
                    latest_return_date = str(latest['return_date'])
        
        # Active injury status (only from current year)
        active = player_df[
            (player_df['is_active']) & 
            (player_df['season'] == current_year)
        ]
        has_active_injury = len(active) > 0
        active_injury_type = None
        active_return_date = None
        if has_active_injury:
            active_row = active.iloc[0]
            active_injury_type = active_row['injury_surgery']
            if pd.notna(active_row.get('return_date')):
                active_return_date = str(active_row['return_date'])
            elif pd.notna(active_row.get('latest_update')):
                active_return_date = str(active_row['latest_update'])
        
        # Injury type recurrence
        injury_types = recent['injury_surgery'].dropna().str.lower()
        recurring_types = [t for t in injury_types if injury_types.tolist().count(t) > 1]
        has_recurrence = len(recurring_types) > 0
        
        return {
            'fg_id': fg_id,
            'il_stints_recent': il_stints,
            'has_major_surgery': latest_surgery is not None,
            'latest_surgery_type': latest_surgery,
            'latest_injury_name': latest_injury_name,
            'surgery_year': surgery_year,
            'return_date': active_return_date or latest_return_date,
            'has_active_injury': has_active_injury,
            'active_injury_type': active_injury_type,
            'has_injury_recurrence': has_recurrence,
            'total_days_lost_recent': recent['days_lost'].sum(),
        }
    
    def _empty_summary(self) -> Dict:
        """Return empty injury summary."""
        return {
            'fg_id': None,
            'il_stints_recent': 0,
            'has_major_surgery': False,
            'latest_surgery_type': None,
            'surgery_year': None,
            'has_active_injury': False,
            'active_injury_type': None,
            'has_injury_recurrence': False,
            'total_days_lost_recent': 0,
        }
    
    def calculate_adjustment(self, fg_id: int, projection_year: int,
                              name: str = None) -> InjuryInfo:
        """
        Calculate playing time multiplier for a player.
        
        Args:
            fg_id: FanGraphs player ID
            projection_year: Year being projected
            name: Player name (fallback)
            
        Returns:
            InjuryInfo with multiplier, injury name, and return date
        """
        summary = self.get_player_injury_summary(fg_id, name)
        
        multiplier = 1.0
        injury_name = None
        return_date = None
        
        # 1. Active injury: player is out
        if summary['has_active_injury']:
            logger.debug(f"Player {fg_id} has active injury: {summary['active_injury_type']}")
            injury_name = summary['active_injury_type']
            return_date = summary.get('return_date')
            # Could return 0 or partial based on expected return
            # For now, we'll apply a heavy penalty
            multiplier *= 0.3
            return InjuryInfo(multiplier=multiplier, injury_name=injury_name, return_date=return_date)
        
        # 2. Major surgery recovery
        if summary['has_major_surgery'] and summary['surgery_year'] is not None:
            surgery_type = summary['latest_surgery_type']
            surgery_year = summary['surgery_year']
            years_since = projection_year - surgery_year
            injury_name = summary.get('latest_injury_name', surgery_type)
            return_date = summary.get('return_date')
            
            if surgery_type in self.injury_config.MAJOR_SURGERIES:
                year1_mult, year2_mult = self.injury_config.MAJOR_SURGERIES[surgery_type]
                
                if years_since == 0:
                    multiplier *= year1_mult
                    logger.debug(f"Player {fg_id}: {surgery_type} year 1 -> {year1_mult}")
                elif years_since == 1:
                    multiplier *= year2_mult
                    logger.debug(f"Player {fg_id}: {surgery_type} year 2 -> {year2_mult}")
                # After year 2, no surgery penalty
        
        # 3. Injury history penalty (low weight)
        il_threshold = self.injury_config.IL_STINT_THRESHOLD
        if summary['il_stints_recent'] >= il_threshold:
            penalty = self.injury_config.HISTORY_PENALTY_BASE
            multiplier *= (1.0 - penalty)
            logger.debug(f"Player {fg_id}: injury history penalty -> {penalty}")
            # Set injury name to history if not already set
            if injury_name is None:
                injury_name = "Injury history"
        
        # 4. Recurrence penalty
        if summary['has_injury_recurrence']:
            penalty = self.injury_config.RECURRENCE_PENALTY
            multiplier *= (1.0 - penalty)
            logger.debug(f"Player {fg_id}: recurrence penalty -> {penalty}")
        
        return InjuryInfo(
            multiplier=max(0.0, min(1.0, multiplier)),
            injury_name=injury_name,
            return_date=return_date
        )
    
    def build_adjustment_lookup(self, player_ids: List[int],
                                 projection_year: int) -> Dict[int, InjuryInfo]:
        """
        Build adjustment info for all players.
        
        Args:
            player_ids: List of player FG IDs
            projection_year: Year being projected
            
        Returns:
            Dict mapping IDfg to InjuryInfo (multiplier, injury_name, return_date)
        """
        adjustments = {}
        
        for fg_id in player_ids:
            info = self.calculate_adjustment(fg_id, projection_year)
            adjustments[fg_id] = info
            
        return adjustments
