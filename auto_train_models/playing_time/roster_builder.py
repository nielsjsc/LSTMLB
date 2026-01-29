"""
Roster building and depth chart construction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .config import Config
from .injury_processor import InjuryInfo
from .confidence import ConfidenceCalculator, get_confidence_tier

logger = logging.getLogger(__name__)


@dataclass
class Player:
    """Player representation for depth charts."""
    id: int                    # IDfg
    name: str
    team: str
    position: str              # Primary position
    positions: List[str]       # All positions played
    projected_value: float     # wOBA for batters, FIP for pitchers (rate stat for ranking)
    role: str = None           # SP/RP for pitchers
    injury_multiplier: float = 1.0
    injury_name: str = None    # Name/type of injury
    injury_return_date: str = None  # Expected return date
    
    # Confidence score (0-1) - affects playing time allocation
    confidence: float = 0.5
    confidence_tier: str = 'competitive'  # locked, favored, competitive, fringe
    
    # Full prediction data for WAR calculation later
    prediction_data: Dict = None


class RosterBuilder:
    """Build team rosters with depth charts from multiple data sources."""
    
    def __init__(self, config: Config = None, confidence_calculator: ConfidenceCalculator = None):
        self.config = config or Config()
        self.confidence_calculator = confidence_calculator
        
    def build_team_rosters(self,
                           roster_df: pd.DataFrame,
                           batter_preds: pd.DataFrame,
                           pitcher_preds: pd.DataFrame,
                           fielding_df: pd.DataFrame,
                           injury_adjustments: Dict[int, InjuryInfo],
                           projection_year: int) -> Dict[str, Dict[str, List[Player]]]:
        """
        Build complete team rosters with position assignments.
        
        Args:
            roster_df: Current MLB rosters
            batter_preds: Batter projections
            pitcher_preds: Pitcher projections
            fielding_df: Fielding data (for position inference)
            injury_adjustments: IDfg -> InjuryInfo
            projection_year: Year being projected
            
        Returns:
            Dict[team_abbr, Dict[position_group, List[Player]]]
        """
        team_rosters = {}
        
        # Get unique teams
        teams = roster_df['team_abbr'].dropna().unique()
        
        for team in teams:
            team_roster = self._build_single_team(
                team=team,
                roster_df=roster_df,
                batter_preds=batter_preds,
                pitcher_preds=pitcher_preds,
                fielding_df=fielding_df,
                injury_adjustments=injury_adjustments,
                projection_year=projection_year
            )
            team_rosters[team] = team_roster
            
        return team_rosters
    
    def _build_single_team(self,
                           team: str,
                           roster_df: pd.DataFrame,
                           batter_preds: pd.DataFrame,
                           pitcher_preds: pd.DataFrame,
                           fielding_df: pd.DataFrame,
                           injury_adjustments: Dict[int, InjuryInfo],
                           projection_year: int) -> Dict[str, List[Player]]:
        """Build roster for a single team."""
        
        # Initialize position groups
        roster = {
            'C': [],
            'INF': [],
            'OF': [],
            'DH': [],
            'SP': [],
            'RP': [],
        }
        
        # Filter to team
        team_roster_df = roster_df[roster_df['team_abbr'] == team]
        
        # Process current roster players
        for _, row in team_roster_df.iterrows():
            fg_id = row.get('fg_id')
            if pd.isna(fg_id) or fg_id == -1:
                continue
                
            fg_id = int(fg_id)
            name = row['player_name']
            pos_code = row.get('position_code', '')
            pos_type = row.get('position_type', '')
            
            # Determine if pitcher or position player
            is_pitcher = pos_type == 'Pitcher' or pos_code == '1'
            
            if is_pitcher:
                player = self._build_pitcher_player(
                    fg_id=fg_id,
                    name=name,
                    team=team,
                    pitcher_preds=pitcher_preds,
                    projection_year=projection_year,
                    injury_adjustments=injury_adjustments
                )
            else:
                player = self._build_position_player(
                    fg_id=fg_id,
                    name=name,
                    team=team,
                    pos_code=pos_code,
                    batter_preds=batter_preds,
                    fielding_df=fielding_df,
                    projection_year=projection_year,
                    injury_adjustments=injury_adjustments
                )
            
            if player is not None:
                group = self._get_position_group(player.position, player.role)
                if group in roster:
                    roster[group].append(player)
        
        # Sort each group by projected value (descending)
        for group in roster:
            roster[group] = sorted(roster[group], 
                                   key=lambda p: p.projected_value, 
                                   reverse=True)
        
        return roster
    
    def _build_position_player(self,
                               fg_id: int,
                               name: str,
                               team: str,
                               pos_code: str,
                               batter_preds: pd.DataFrame,
                               fielding_df: pd.DataFrame,
                               projection_year: int,
                               injury_adjustments: Dict[int, InjuryInfo]) -> Optional[Player]:
        """Build Player object for a position player."""
        
        # Get projection
        pred = batter_preds[
            (batter_preds['IDfg'] == fg_id) & 
            (batter_preds['Year'] == projection_year)
        ]
        
        if pred.empty:
            logger.debug(f"No projection for batter {name} ({fg_id})")
            return None
        
        pred_row = pred.iloc[0]
        
        # Use wOBA as the projected value metric for ranking
        # wOBA is a rate stat that doesn't depend on playing time
        woba = pred_row.get('wOBA', 0.300)  # Default to league average
        
        # Store full prediction data for WAR calculation later
        prediction_data = pred_row.to_dict()
        
        # Get positions from fielding data
        positions = self._get_player_positions(fg_id, fielding_df, projection_year)
        
        # Map primary position
        primary_pos = self._map_position_code(pos_code)
        if not positions:
            positions = [primary_pos]
        
        # Get fielding data for this player
        fielding_row = fielding_df[
            (fielding_df['IDfg'] == fg_id) & 
            (fielding_df['Year'] == projection_year)
        ]
        if not fielding_row.empty:
            prediction_data['fielding'] = fielding_row.iloc[0].to_dict()
        
        # Get injury adjustment
        inj_info = injury_adjustments.get(fg_id, InjuryInfo(multiplier=1.0))
        
        # Calculate confidence score if calculator is available
        confidence = 0.5  # Default
        if self.confidence_calculator:
            try:
                # Get player age from prediction data
                player_age = int(pred_row.get('Age', 27))  # Default to prime age
                
                confidence_result = self.confidence_calculator.calculate_confidence(
                    player_id=fg_id,
                    player_name=name,
                    player_age=player_age,
                    model_type='batter',
                    projection_year=projection_year
                )
                confidence = confidence_result.combined
            except Exception as e:
                logger.debug(f"Could not calculate confidence for {name} ({fg_id}): {e}")
        
        confidence_tier = get_confidence_tier(confidence)
        
        return Player(
            id=fg_id,
            name=name,
            team=team,
            position=primary_pos,
            positions=positions,
            projected_value=woba,
            injury_multiplier=inj_info.multiplier,
            injury_name=inj_info.injury_name,
            injury_return_date=inj_info.return_date,
            confidence=confidence,
            confidence_tier=confidence_tier,
            prediction_data=prediction_data
        )
    
    def _build_pitcher_player(self,
                              fg_id: int,
                              name: str,
                              team: str,
                              pitcher_preds: pd.DataFrame,
                              projection_year: int,
                              injury_adjustments: Dict[int, InjuryInfo]) -> Optional[Player]:
        """Build Player object for a pitcher."""
        
        # Get projection
        pred = pitcher_preds[
            (pitcher_preds['IDfg'] == fg_id) & 
            (pitcher_preds['Year'] == projection_year)
        ]
        
        if pred.empty:
            logger.debug(f"No projection for pitcher {name} ({fg_id})")
            return None
        
        pred_row = pred.iloc[0]
        role = pred_row.get('Role', 'RP')
        
        # Use FIP as the projected value metric for ranking
        # FIP is a rate stat that doesn't depend on playing time
        # Lower FIP = better, so we negate it for consistent "higher is better" ranking
        fip = pred_row.get('FIP', 4.50)  # Default to league average
        
        # Store full prediction data for WAR calculation later
        prediction_data = pred_row.to_dict()
        
        # Get injury adjustment
        inj_info = injury_adjustments.get(fg_id, InjuryInfo(multiplier=1.0))
        
        # Calculate confidence score if calculator is available
        confidence = 0.5  # Default
        if self.confidence_calculator:
            try:
                # Get player age from prediction data
                player_age = int(pred_row.get('Age', 27))  # Default to prime age
                
                confidence_result = self.confidence_calculator.calculate_confidence(
                    player_id=fg_id,
                    player_name=name,
                    player_age=player_age,
                    model_type=role,  # 'SP' or 'RP'
                    projection_year=projection_year
                )
                confidence = confidence_result.combined
            except Exception as e:
                logger.debug(f"Could not calculate confidence for {name} ({fg_id}): {e}")
        
        confidence_tier = get_confidence_tier(confidence)
        
        return Player(
            id=fg_id,
            name=name,
            team=team,
            position=role,
            positions=[role],
            projected_value=-fip,  # Negate so higher = better (like wOBA)
            role=role,
            injury_multiplier=inj_info.multiplier,
            injury_name=inj_info.injury_name,
            injury_return_date=inj_info.return_date,
            confidence=confidence,
            confidence_tier=confidence_tier,
            prediction_data=prediction_data
        )
    
    def _get_player_positions(self, fg_id: int, fielding_df: pd.DataFrame,
                              projection_year: int) -> List[str]:
        """Get all positions a player can play from fielding data."""
        if fielding_df.empty:
            return []
        
        # Look at recent years
        player_fielding = fielding_df[
            (fielding_df['IDfg'] == fg_id) &
            (fielding_df['Year'] <= projection_year) &
            (fielding_df['Year'] >= projection_year - 3)
        ]
        
        if player_fielding.empty:
            return []
        
        positions = player_fielding['Pos'].unique().tolist()
        return positions
    
    def _map_position_code(self, code: str) -> str:
        """Map position code to standard position."""
        return self.config.POSITION_CODE_MAP.get(str(code), 'DH')
    
    def _get_position_group(self, position: str, role: str = None) -> str:
        """Map position to allocation group."""
        if role in ['SP', 'RP']:
            return role
        
        if position == 'C':
            return 'C'
        elif position in ['1B', '2B', 'SS', '3B', 'INF']:
            return 'INF'
        elif position in ['LF', 'CF', 'RF', 'OF']:
            return 'OF'
        elif position in ['DH']:
            return 'DH'
        else:
            return 'DH'  # Default
