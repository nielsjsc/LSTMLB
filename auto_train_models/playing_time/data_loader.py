"""
Data loading and ID matching utilities.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from rapidfuzz import fuzz, process

from .config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and merge all data sources for playing time projection."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._batter_predictions: Optional[pd.DataFrame] = None
        self._pitcher_predictions: Optional[pd.DataFrame] = None
        self._fielding_predictions: Optional[pd.DataFrame] = None
        self._baserunning_predictions: Optional[pd.DataFrame] = None
        self._roster: Optional[pd.DataFrame] = None
        self._prospects: Optional[pd.DataFrame] = None
        self._injury_data: Optional[pd.DataFrame] = None
        
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                 pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data sources."""
        logger.info("Loading all data sources...")
        
        batters = self.load_batter_predictions()
        pitchers = self.load_pitcher_predictions()
        fielding = self.load_fielding_predictions()
        roster = self.load_roster()
        prospects = self.load_prospects()
        injuries = self.load_injury_data()
        
        logger.info(f"Loaded: {len(batters)} batter rows, {len(pitchers)} pitcher rows, "
                   f"{len(roster)} roster entries, {len(prospects)} prospects, "
                   f"{len(injuries)} injury records")
        
        return batters, pitchers, fielding, roster, prospects, injuries
    
    def load_batter_predictions(self) -> pd.DataFrame:
        """Load batter projection data."""
        if self._batter_predictions is not None:
            return self._batter_predictions
            
        path = self.config.BATTER_PREDICTIONS
        if not path.exists():
            raise FileNotFoundError(f"Batter predictions not found: {path}")
        
        df = pd.read_csv(path)
        df['IDfg'] = df['IDfg'].astype(int)
        df['Year'] = df['Year'].astype(int)
        # Add normalized name for fuzzy matching
        if 'Name' in df.columns:
            df['name_normalized'] = df['Name'].apply(self._normalize_name)
        self._batter_predictions = df
        return df
    
    def load_pitcher_predictions(self) -> pd.DataFrame:
        """Load pitcher projection data."""
        if self._pitcher_predictions is not None:
            return self._pitcher_predictions
            
        path = self.config.PITCHER_PREDICTIONS
        if not path.exists():
            raise FileNotFoundError(f"Pitcher predictions not found: {path}")
        
        df = pd.read_csv(path)
        df['IDfg'] = df['IDfg'].astype(int)
        df['Year'] = df['Year'].astype(int)
        # Add normalized name for fuzzy matching
        if 'Name' in df.columns:
            df['name_normalized'] = df['Name'].apply(self._normalize_name)
        self._pitcher_predictions = df
        return df
    
    def load_fielding_predictions(self) -> pd.DataFrame:
        """Load fielding projection data (for position inference)."""
        if self._fielding_predictions is not None:
            return self._fielding_predictions
            
        path = self.config.FIELDING_PREDICTIONS
        if not path.exists():
            logger.warning(f"Fielding predictions not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df['IDfg'] = df['IDfg'].astype(int)
        df['Year'] = df['Year'].astype(int)
        # Add normalized name for fuzzy matching
        if 'Name' in df.columns:
            df['name_normalized'] = df['Name'].apply(self._normalize_name)
        self._fielding_predictions = df
        return df
    
    def load_baserunning_predictions(self) -> pd.DataFrame:
        """Load baserunning projection data."""
        if self._baserunning_predictions is not None:
            return self._baserunning_predictions
            
        path = self.config.BASERUNNING_PREDICTIONS
        if not path.exists():
            logger.warning(f"Baserunning predictions not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df['IDfg'] = df['IDfg'].astype(int)
        df['Year'] = df['Year'].astype(int)
        self._baserunning_predictions = df
        return df
    
    def load_roster(self) -> pd.DataFrame:
        """Load current roster data with team assignments."""
        if self._roster is not None:
            return self._roster
            
        path = self.config.ROSTER_DATA
        if not path.exists():
            raise FileNotFoundError(f"Roster data not found: {path}")
        
        df = pd.read_csv(path)
        
        # Normalize team names
        df['team_abbr'] = df['team_name'].map(self.config.TEAM_NAME_TO_ABBR)
        
        # Handle missing fg_id (-1 or NaN)
        df['fg_id'] = df['fg_id'].replace(-1.0, np.nan)
        df['has_fg_id'] = df['fg_id'].notna()
        
        # Create normalized name for matching
        df['name_normalized'] = df['player_name'].apply(self._normalize_name)
        
        self._roster = df
        return df
    
    def load_prospects(self) -> pd.DataFrame:
        """Load prospect data with ETA."""
        if self._prospects is not None:
            return self._prospects
            
        path = self.config.PROSPECT_DATA
        if not path.exists():
            logger.warning(f"Prospect data not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        
        # Parse ETA as integer year
        df['eta_year'] = pd.to_numeric(df['eta'], errors='coerce').astype('Int64')
        
        # Normalize organization to team abbr
        df['team_abbr'] = df['organization'].map(self.config.TEAM_NAME_TO_ABBR)
        
        # Map position to standard groups
        df['position_group'] = df['position'].map(self.config.PROSPECT_POSITION_MAP)
        
        # Normalize name for matching
        df['name_normalized'] = df['name'].apply(self._normalize_name)
        
        self._prospects = df
        return df
    
    def load_injury_data(self) -> pd.DataFrame:
        """Load injury history data."""
        if self._injury_data is not None:
            return self._injury_data
            
        path = self.config.INJURY_DATA
        if not path.exists():
            logger.warning(f"Injury data not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        
        # Ensure fg_id is numeric
        df['fg_id'] = pd.to_numeric(df['fg_id'], errors='coerce')
        
        # Normalize name for matching
        df['name_normalized'] = df['name'].apply(self._normalize_name)
        
        self._injury_data = df
        return df
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize player name for matching."""
        if pd.isna(name):
            return ''
        # Lowercase, strip accents approximation, remove punctuation
        name = str(name).lower().strip()
        # Remove common suffixes
        for suffix in [' jr.', ' jr', ' sr.', ' sr', ' ii', ' iii', ' iv']:
            name = name.replace(suffix, '')
        # Remove punctuation
        name = ''.join(c for c in name if c.isalnum() or c.isspace())
        return name
    
    def match_player_by_name(self, name: str, candidates: pd.DataFrame,
                              name_col: str = 'name_normalized',
                              threshold: int = 85) -> Optional[int]:
        """
        Match a player name to candidates using fuzzy matching.
        
        Args:
            name: Player name to match
            candidates: DataFrame with candidate players
            name_col: Column containing normalized names
            threshold: Minimum match score (0-100)
            
        Returns:
            Index of best match or None if no match found
        """
        if candidates.empty:
            return None
            
        normalized = self._normalize_name(name)
        choices = candidates[name_col].tolist()
        
        result = process.extractOne(
            normalized, 
            choices, 
            scorer=fuzz.token_sort_ratio
        )
        
        if result and result[1] >= threshold:
            match_idx = choices.index(result[0])
            return candidates.index[match_idx]
        
        return None
    
    def build_player_id_map(self, roster: pd.DataFrame, 
                            predictions: pd.DataFrame) -> Dict[str, int]:
        """
        Build mapping from player names to FG IDs.
        Uses direct ID match where available, fuzzy name match otherwise.
        
        Returns:
            Dict mapping player names to IDfg values
        """
        id_map = {}
        
        # Players with valid fg_id
        valid_ids = roster[roster['has_fg_id']]
        for _, row in valid_ids.iterrows():
            id_map[row['player_name']] = int(row['fg_id'])
        
        # Players without fg_id - try to match by name to predictions
        missing_ids = roster[~roster['has_fg_id']]
        
        for _, row in missing_ids.iterrows():
            name = row['player_name']
            match_idx = self.match_player_by_name(name, predictions, 'Name')
            
            if match_idx is not None:
                matched_id = predictions.loc[match_idx, 'IDfg']
                id_map[name] = int(matched_id)
                logger.debug(f"Matched {name} to IDfg {matched_id}")
            else:
                logger.warning(f"Could not match player: {name}")
        
        return id_map
