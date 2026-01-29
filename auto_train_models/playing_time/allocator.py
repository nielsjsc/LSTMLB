"""
Playing Time Allocator - Linear Programming Optimization
=========================================================

Allocates playing time to maximize total team value using linear programming.

Key Features:
- Global optimization (not greedy heuristics)
- Position flexibility with realistic constraints
- Defense estimation for new positions
- Handles position scarcity automatically

Position Hierarchies (movement allowed downward):
    Infield: SS > 3B > 1B > DH
             SS > 2B > 1B > DH
             2B != 3B (arm strength difference)
    
    Outfield: CF > LF = RF > DH
              LF <-> RF (lateral moves allowed)
    
    Catcher: C > DH only (specialized position)

Author: Niels Christoffersen
Date: January 2026
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from scipy.optimize import linprog, OptimizeResult

from .config import Config, TeamBudget
from .roster_builder import Player

logger = logging.getLogger(__name__)


# =============================================================================
# POSITION CONSTANTS
# =============================================================================

# All field positions (9 slots) + DH
POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH']
POS_TO_IDX = {pos: i for i, pos in enumerate(POSITIONS)}
NUM_POSITIONS = len(POSITIONS)

# Position eligibility rules based on historical position
# Key: primary position → Set of positions they can play
# Rules:
#   SS > 3B > 1B > DH
#   SS > 2B > 1B > DH
#   2B != 3B (arm strength)
#   CF > LF = RF > DH
#   LF <-> RF (lateral)
#   C > DH only
POSITION_ELIGIBILITY_RULES = {
    'C':  {'C', 'DH'},                      # Catchers only catch or DH
    'SS': {'SS', '3B', '2B', '1B', 'DH'},   # SS can go anywhere in IF
    '3B': {'3B', '1B', 'DH'},               # 3B cannot go to 2B
    '2B': {'2B', '1B', 'DH'},               # 2B cannot go to 3B
    '1B': {'1B', 'DH'},                     # 1B only to DH
    'CF': {'CF', 'LF', 'RF', 'DH'},         # CF can play corners
    'LF': {'LF', 'RF', 'DH'},               # Corners swap, not CF
    'RF': {'RF', 'LF', 'DH'},               # Corners swap, not CF
    'DH': {'DH'},                           # DH only
    # Generic positions (for roster codes)
    'INF': {'1B', '2B', '3B', 'SS', 'DH'},  # Generic IF can play anywhere
    'OF': {'LF', 'CF', 'RF', 'DH'},         # Generic OF can play anywhere
}

# Positional adjustments (runs per 162 games, from FanGraphs)
POSITIONAL_ADJUSTMENTS = {
    'C': 12.5,
    'SS': 7.5,
    '2B': 3.0,
    'CF': 2.5,
    '3B': 2.5,
    'LF': -7.5,
    'RF': -7.5,
    '1B': -12.5,
    'DH': -17.5,
}

# Position difficulty scores (for defense estimation at new positions)
# Higher = harder defensively
POSITION_DIFFICULTY = {
    'C': 10,   # Hardest - specialized
    'SS': 8,
    'CF': 7,
    '2B': 6,
    '3B': 6,
    'RF': 4,
    'LF': 4,
    '1B': 2,
    'DH': 0,   # No defense
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AllocationResult:
    """Result of playing time allocation for a player."""
    id: int
    name: str
    team: str
    position: str
    role: Optional[str]
    projected_value: float
    injury_multiplier: float
    injury_name: Optional[str] = None
    injury_return_date: Optional[str] = None
    allocated_games: float = 0.0
    allocated_ip: float = 0.0
    allocation_pct: float = 0.0
    
    # Confidence
    confidence: float = 0.5
    confidence_tier: str = 'competitive'
    
    # WAR components (filled after allocation by value_calculator)
    war: Optional[float] = None
    wrc_plus: Optional[float] = None
    batting_runs: Optional[float] = None
    baserunning_runs: Optional[float] = None
    fielding_runs: Optional[float] = None
    positional_adj: Optional[float] = None
    fip_runs: Optional[float] = None
    
    prediction_data: Optional[dict] = None


@dataclass
class TeamAllocation:
    """Team-level allocation summary."""
    team: str
    total_position_games: float = 0.0
    total_sp_ip: float = 0.0
    total_rp_ip: float = 0.0
    players: List[AllocationResult] = field(default_factory=list)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def woba_to_batting_runs_per_pa(woba: float, lg_woba: float = 0.315) -> float:
    """Convert wOBA to batting runs per PA (above average)."""
    woba_scale = 1.23
    return (woba - lg_woba) / woba_scale


def get_eligible_positions(player: Player) -> Set[str]:
    """
    Determine which positions a player is eligible to play.
    
    Uses position hierarchy rules:
    - Players can move "down" the difficulty ladder
    - SS can go to 3B, 2B, 1B (but not OF)
    - CF can go to corner OF (but infielders can't go to OF)
    - 2B and 3B cannot interchange (arm strength)
    
    Examples:
    - Bo Bichette (SS): SS, 3B, 2B, 1B, DH
    - Jorge Polanco (2B): 2B, 1B, DH
    - Aaron Judge (RF): RF, LF, DH
    """
    eligible = set()
    
    # Get all historical positions
    historical = set(player.positions) if player.positions else set()
    
    # Add primary position
    historical.add(player.position)
    
    # For each historical position, add all positions they can play
    for hist_pos in historical:
        if hist_pos in POSITION_ELIGIBILITY_RULES:
            eligible.update(POSITION_ELIGIBILITY_RULES[hist_pos])
    
    # Everyone can DH
    eligible.add('DH')
    
    return eligible


def estimate_defense_at_position(
    player: Player,
    target_pos: str,
    primary_defense: float
) -> float:
    """
    Estimate defensive value at a position the player hasn't played.
    
    Uses their primary position defense and applies a penalty based on
    the difficulty gap between positions.
    
    Args:
        player: Player object
        target_pos: Position to estimate defense for
        primary_defense: sc_total_runs/150 at their primary position
        
    Returns:
        Estimated defensive runs per 150 games at target position
    """
    if target_pos == 'DH':
        return 0.0  # No defense at DH
    
    primary_pos = player.position
    if primary_pos in ['INF', 'OF', 'DH']:
        # Generic position - use first historical position
        primary_pos = player.positions[0] if player.positions else '1B'
    
    # If they've played this position, use actual defense
    if target_pos in (player.positions or []):
        return primary_defense
    
    # Calculate difficulty gap
    primary_diff = POSITION_DIFFICULTY.get(primary_pos, 5)
    target_diff = POSITION_DIFFICULTY.get(target_pos, 5)
    
    # Penalty for moving to an unfamiliar position
    # Moving to easier position: small penalty (unfamiliarity)
    # Moving to harder position: larger penalty (shouldn't happen often)
    if target_diff <= primary_diff:
        # Moving easier: 1.5 runs penalty per difficulty level
        penalty = (primary_diff - target_diff) * 1.5
    else:
        # Moving harder: 3.5 runs penalty (rare case)
        penalty = (target_diff - primary_diff) * 3.5
    
    return primary_defense - penalty


# =============================================================================
# LINEAR PROGRAMMING ALLOCATOR
# =============================================================================

class PlayingTimeAllocator:
    """
    Allocate playing time using linear programming optimization.
    
    Maximizes: Total team value (offense + defense) across all positions
    
    Subject to:
        - Each position gets exactly its PA budget (700, 640 for C)
        - Each player limited by health/age constraints
        - Players can only play eligible positions
    """
    
    def __init__(self, config: Config = None, historical_fielding_path: Path = None):
        self.config = config or Config()
        self.budget = self.config.budget
        
        # Historical data paths
        data_dir = Path(__file__).parent.parent.parent / 'data'
        self.historical_fielding_path = historical_fielding_path or (
            data_dir / 'historic_mlb' / 'mlb_fielding_data_2000_2025.csv'
        )
        self._historical_fielding: Optional[pd.DataFrame] = None
    
    def allocate_all_teams(
        self,
        team_rosters: Dict[str, Dict[str, List[Player]]],
        projection_year: int
    ) -> Dict[str, TeamAllocation]:
        """
        Allocate playing time for all teams.
        
        Args:
            team_rosters: Dict[team, Dict[position_group, List[Player]]]
            projection_year: Year being projected
            
        Returns:
            Dict[team, TeamAllocation]
        """
        allocations = {}
        
        for team, roster in team_rosters.items():
            try:
                allocation = self._allocate_team_lp(team, roster, projection_year)
                allocations[team] = allocation
            except Exception as e:
                logger.error(f"Failed to allocate {team}: {e}", exc_info=True)
                # Fallback to empty allocation
                allocations[team] = TeamAllocation(team=team)
        
        return allocations
    
    def _allocate_team_lp(
        self,
        team: str,
        roster: Dict[str, List[Player]],
        projection_year: int
    ) -> TeamAllocation:
        """
        Allocate playing time for a single team using linear programming.
        
        The LP formulation:
        - Decision variables: PA[player, position] for all player-position pairs
        - Objective: Maximize sum of value[p,pos] * PA[p,pos]
        - Constraints:
            1. Sum of PA at each position = position budget
            2. Sum of PA for each player <= player's max PA
            3. PA[p,pos] = 0 if player not eligible for position
            4. PA >= 0
        """
        allocation = TeamAllocation(team=team)
        
        # Collect all position players (batters)
        batters = self._collect_batters(roster)
        
        if not batters:
            logger.warning(f"{team}: No batters found")
            self._allocate_pitchers(roster, allocation)
            return allocation
        
        # Build optimization matrices
        n_players = len(batters)
        
        # Build eligibility matrix (n_players x n_positions)
        eligibility = self._build_eligibility_matrix(batters)
        
        # Build value matrix (n_players x n_positions)
        value_matrix = self._build_value_matrix(batters, eligibility)
        
        # Build player constraints (max PA per player)
        max_pa = self._build_player_constraints(batters)
        
        # Position budgets (PA per position)
        position_budgets = np.array([
            self.budget.C_PA if pos == 'C' else self.budget.POSITION_PA
            for pos in POSITIONS
        ], dtype=float)
        
        # Solve LP
        pa_allocation = self._solve_lp(
            value_matrix=value_matrix,
            eligibility=eligibility,
            max_pa=max_pa,
            position_budgets=position_budgets,
            team=team
        )
        
        # Convert solution to AllocationResults
        self._create_allocation_results(
            batters=batters,
            pa_allocation=pa_allocation,
            allocation=allocation
        )
        
        # Allocate pitchers
        self._allocate_pitchers(roster, allocation)
        
        return allocation
    
    def _collect_batters(self, roster: Dict[str, List[Player]]) -> List[Player]:
        """Collect unique batters from roster."""
        seen_ids = set()
        batters = []
        
        for group in ['C', 'INF', 'OF', 'DH']:
            for player in roster.get(group, []):
                if player.id not in seen_ids:
                    batters.append(player)
                    seen_ids.add(player.id)
        
        return batters
    
    def _build_eligibility_matrix(self, batters: List[Player]) -> np.ndarray:
        """
        Build eligibility matrix: E[p, pos] = 1 if player p can play position pos.
        
        Uses position hierarchy rules to determine eligibility.
        """
        n_players = len(batters)
        eligibility = np.zeros((n_players, NUM_POSITIONS), dtype=float)
        
        for i, player in enumerate(batters):
            eligible_positions = get_eligible_positions(player)
            
            for pos in eligible_positions:
                if pos in POS_TO_IDX:
                    eligibility[i, POS_TO_IDX[pos]] = 1.0
            
            # Log position flexibility for debugging
            if len(eligible_positions) > 2:
                logger.debug(
                    f"{player.name} ({player.position}): eligible for {sorted(eligible_positions)}"
                )
        
        return eligibility
    
    def _build_value_matrix(
        self,
        batters: List[Player],
        eligibility: np.ndarray
    ) -> np.ndarray:
        """
        Build value matrix: V[p, pos] = expected runs above average per PA.
        
        Value components:
        1. Offense: wOBA → batting runs/PA (same at all positions)
        2. Defense: sc_total_runs/150 scaled to per-PA (varies by position)
        3. Positional adjustment: varies by position
        
        For ineligible positions, value is set to large negative.
        """
        n_players = len(batters)
        value_matrix = np.zeros((n_players, NUM_POSITIONS), dtype=float)
        
        # PA per game approximation
        pa_per_game = self.budget.PA_PER_GAME  # ~4.3
        games_per_season = 162
        pa_per_season = pa_per_game * games_per_season  # ~700
        
        for i, player in enumerate(batters):
            # Offense value (same at all positions)
            woba = player.projected_value
            offense_runs_per_pa = woba_to_batting_runs_per_pa(woba)
            
            # Get fielding data
            fielding_data = {}
            if player.prediction_data and 'fielding' in player.prediction_data:
                fielding_data = player.prediction_data['fielding']
            
            # Primary position defense (sc_total_runs/150)
            primary_defense = fielding_data.get('sc_total_runs/150', 0.0) or 0.0
            
            for j, pos in enumerate(POSITIONS):
                if eligibility[i, j] == 0:
                    # Ineligible: large negative to prevent selection
                    value_matrix[i, j] = -1000.0
                    continue
                
                # Defense value at this position
                if pos == 'DH':
                    defense_runs_150 = 0.0
                elif pos == player.position or pos in (player.positions or []):
                    # Playing their natural position
                    defense_runs_150 = primary_defense
                else:
                    # New position: estimate defense
                    defense_runs_150 = estimate_defense_at_position(
                        player, pos, primary_defense
                    )
                
                # Convert from runs/150 games to runs/PA
                defense_runs_per_pa = defense_runs_150 / (150 * pa_per_game)
                
                # Positional adjustment per PA
                pos_adj = POSITIONAL_ADJUSTMENTS.get(pos, 0.0)
                pos_adj_per_pa = pos_adj / pa_per_season
                
                # Total value per PA
                value_matrix[i, j] = (
                    offense_runs_per_pa + 
                    defense_runs_per_pa + 
                    pos_adj_per_pa
                )
        
        return value_matrix
    
    def _build_player_constraints(self, batters: List[Player]) -> np.ndarray:
        """
        Build max PA constraints for each player.
        
        Considers:
        - Position (catchers capped lower)
        - Injury multiplier
        - Age (older players may need more rest)
        """
        max_pa = np.zeros(len(batters), dtype=float)
        
        for i, player in enumerate(batters):
            # Base max PA
            if player.position == 'C':
                base_max = self.budget.MAX_CATCHER_PA  # 550
            else:
                base_max = self.budget.MAX_PLAYER_PA  # 700
            
            # Apply injury multiplier
            adjusted_max = base_max * player.injury_multiplier
            
            # Age adjustment
            age = player.prediction_data.get('Age', 27) if player.prediction_data else 27
            if age >= 38:
                adjusted_max *= 0.85
            elif age >= 35:
                adjusted_max *= 0.95
            
            max_pa[i] = adjusted_max
        
        return max_pa
    
    def _solve_lp(
        self,
        value_matrix: np.ndarray,
        eligibility: np.ndarray,
        max_pa: np.ndarray,
        position_budgets: np.ndarray,
        team: str
    ) -> np.ndarray:
        """
        Solve the linear programming problem.
        
        Uses slack variables to allow under-filling positions when roster is thin.
        
        Variables: 
            x[i*n_pos + j] = PA for player i at position j
            slack[j] = unfilled PA at position j (penalized in objective)
        
        Objective: maximize sum(value * PA) - penalty * sum(slack)
        
        Constraints:
            sum(PA at position j) + slack[j] = budget[j]  (position filling)
            sum(PA for player i) <= max_pa[i]              (player limits)
            PA[p,pos] = 0 if not eligible                   (eligibility)
            PA >= 0, slack >= 0
        """
        n_players, n_positions = value_matrix.shape
        n_pa_vars = n_players * n_positions
        n_slack_vars = n_positions
        n_vars = n_pa_vars + n_slack_vars
        
        # Objective: minimize -value*PA + penalty*slack
        # Slack penalty should be high enough to prioritize filling positions
        # but not so high that infeasibility breaks things
        slack_penalty = 0.1  # Runs per PA unfilled (encourages filling)
        
        c = np.zeros(n_vars)
        c[:n_pa_vars] = -value_matrix.flatten()  # Maximize value
        c[n_pa_vars:] = slack_penalty  # Minimize slack (unfilled positions)
        
        # Equality constraints: PA at position + slack = budget
        A_eq = np.zeros((n_positions, n_vars))
        for j in range(n_positions):
            for i in range(n_players):
                A_eq[j, i * n_positions + j] = 1.0
            A_eq[j, n_pa_vars + j] = 1.0  # Slack for this position
        b_eq = position_budgets
        
        # Inequality constraints: player PA limits
        A_ub = np.zeros((n_players, n_vars))
        for i in range(n_players):
            for j in range(n_positions):
                A_ub[i, i * n_positions + j] = 1.0
        b_ub = max_pa
        
        # Bounds
        bounds = []
        # PA variables: bounded by eligibility
        for i in range(n_players):
            for j in range(n_positions):
                if eligibility[i, j] > 0:
                    bounds.append((0, max_pa[i]))
                else:
                    bounds.append((0, 0))
        # Slack variables: non-negative, bounded by position budget
        for j in range(n_positions):
            bounds.append((0, position_budgets[j]))
        
        # Solve
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if not result.success:
            logger.warning(f"{team}: LP solver issue: {result.message}")
            # Fallback: greedy allocation
            return self._greedy_fallback(value_matrix, eligibility, max_pa, position_budgets, team)
        
        # Extract PA allocation (ignore slack variables)
        pa_allocation = result.x[:n_pa_vars].reshape(n_players, n_positions)
        
        # Log slack usage (unfilled positions)
        slack = result.x[n_pa_vars:]
        for j, pos in enumerate(POSITIONS):
            if slack[j] > 10:
                logger.info(f"{team} {pos}: {slack[j]:.0f} PA unfilled (roster thin)")
        
        # Clean numerical noise
        pa_allocation[pa_allocation < 1.0] = 0.0
        
        return pa_allocation
    
    def _greedy_fallback(
        self,
        value_matrix: np.ndarray,
        eligibility: np.ndarray,
        max_pa: np.ndarray,
        position_budgets: np.ndarray,
        team: str
    ) -> np.ndarray:
        """
        Greedy fallback allocation when LP fails.
        
        For each position, assign PA to the best eligible player with capacity.
        """
        logger.warning(f"{team}: Using greedy fallback allocation")
        
        n_players, n_positions = value_matrix.shape
        pa_allocation = np.zeros((n_players, n_positions))
        player_pa_used = np.zeros(n_players)
        
        # Process positions by scarcity (fewest eligible players first)
        eligible_counts = eligibility.sum(axis=0)
        position_order = np.argsort(eligible_counts)
        
        for j in position_order:
            budget = position_budgets[j]
            remaining = budget
            
            # Get eligible players sorted by value at this position
            eligible_players = [
                (i, value_matrix[i, j]) 
                for i in range(n_players) 
                if eligibility[i, j] > 0
            ]
            eligible_players.sort(key=lambda x: -x[1])
            
            for player_idx, _ in eligible_players:
                if remaining <= 0:
                    break
                
                available = max_pa[player_idx] - player_pa_used[player_idx]
                if available <= 0:
                    continue
                
                pa_to_assign = min(available, remaining)
                pa_allocation[player_idx, j] = pa_to_assign
                player_pa_used[player_idx] += pa_to_assign
                remaining -= pa_to_assign
            
            if remaining > 10:
                logger.warning(f"{team} {POSITIONS[j]}: {remaining:.0f} PA unfilled")
        
        return pa_allocation
    
    def _create_allocation_results(
        self,
        batters: List[Player],
        pa_allocation: np.ndarray,
        allocation: TeamAllocation
    ) -> None:
        """
        Convert LP solution to AllocationResult objects.
        """
        pa_per_game = self.budget.PA_PER_GAME
        
        player_total_pa: Dict[int, float] = {}
        
        for i, player in enumerate(batters):
            player_total_pa[player.id] = 0.0
            
            for j, pos in enumerate(POSITIONS):
                pa = pa_allocation[i, j]
                
                if pa < 5.0:
                    continue
                
                games = pa / pa_per_game
                player_total_pa[player.id] += pa
                
                result = AllocationResult(
                    id=player.id,
                    name=player.name,
                    team=player.team,
                    position=pos,
                    role=None,
                    projected_value=player.projected_value,
                    injury_multiplier=player.injury_multiplier,
                    injury_name=player.injury_name,
                    injury_return_date=player.injury_return_date,
                    allocated_games=games,
                    allocation_pct=pa / self.budget.TOTAL_PA,
                    confidence=player.confidence,
                    confidence_tier=player.confidence_tier,
                    prediction_data=player.prediction_data
                )
                
                allocation.players.append(result)
                allocation.total_position_games += games
        
        # Log summary
        total_pa = sum(player_total_pa.values())
        total_budget = sum([
            self.budget.C_PA if pos == 'C' else self.budget.POSITION_PA
            for pos in POSITIONS
        ])
        
        logger.debug(
            f"{allocation.team}: Allocated {total_pa:.0f}/{total_budget:.0f} PA "
            f"({100*total_pa/total_budget:.1f}%)"
        )
    
    def _allocate_pitchers(
        self,
        roster: Dict[str, List[Player]],
        allocation: TeamAllocation
    ) -> None:
        """Allocate pitcher innings."""
        self._allocate_starters(roster.get('SP', []), allocation)
        self._allocate_relievers(roster.get('RP', []), allocation)
    
    def _allocate_starters(
        self,
        players: List[Player],
        allocation: TeamAllocation
    ) -> None:
        """Allocate starting pitcher innings by value."""
        if not players:
            return
        
        budget = self.budget.SP_IP
        max_ip = self.budget.MAX_SP_IP
        
        # Sort by value (projected_value is -FIP)
        players = sorted(players, key=lambda p: p.projected_value, reverse=True)
        
        # Rotation slot targets
        slot_targets = [200, 185, 175, 165, 150]
        
        remaining = budget
        
        for i, player in enumerate(players):
            if remaining <= 0:
                break
            
            if i < len(slot_targets):
                target_ip = slot_targets[i]
            else:
                remaining_pitchers = len(players) - i
                target_ip = min(remaining / max(remaining_pitchers, 1), 100)
            
            ip = min(target_ip * player.injury_multiplier, max_ip, remaining)
            
            if ip < 10:
                continue
            
            result = AllocationResult(
                id=player.id,
                name=player.name,
                team=player.team,
                position='SP',
                role='SP',
                projected_value=player.projected_value,
                injury_multiplier=player.injury_multiplier,
                injury_name=player.injury_name,
                injury_return_date=player.injury_return_date,
                allocated_ip=ip,
                allocation_pct=ip / budget,
                confidence=player.confidence,
                confidence_tier=player.confidence_tier,
                prediction_data=player.prediction_data
            )
            
            allocation.players.append(result)
            allocation.total_sp_ip += ip
            remaining -= ip
        
        if remaining > 50:
            logger.warning(f"{allocation.team} SP: {remaining:.0f} IP unallocated")
    
    def _allocate_relievers(
        self,
        players: List[Player],
        allocation: TeamAllocation
    ) -> None:
        """Allocate reliever innings by value."""
        if not players:
            return
        
        budget = self.budget.RP_IP
        max_ip = self.budget.MAX_RP_IP
        
        players = sorted(players, key=lambda p: p.projected_value, reverse=True)
        
        slot_targets = [70, 65, 65, 55, 55, 50, 45, 40]
        
        remaining = budget
        
        for i, player in enumerate(players):
            if remaining <= 0:
                break
            
            if i < len(slot_targets):
                target_ip = slot_targets[i]
            else:
                remaining_relievers = len(players) - i
                target_ip = min(remaining / max(remaining_relievers, 1), 35)
            
            ip = min(target_ip * player.injury_multiplier, max_ip, remaining)
            
            if ip < 5:
                continue
            
            result = AllocationResult(
                id=player.id,
                name=player.name,
                team=player.team,
                position='RP',
                role='RP',
                projected_value=player.projected_value,
                injury_multiplier=player.injury_multiplier,
                injury_name=player.injury_name,
                injury_return_date=player.injury_return_date,
                allocated_ip=ip,
                allocation_pct=ip / budget,
                confidence=player.confidence,
                confidence_tier=player.confidence_tier,
                prediction_data=player.prediction_data
            )
            
            allocation.players.append(result)
            allocation.total_rp_ip += ip
            remaining -= ip
        
        if remaining > 20:
            logger.warning(f"{allocation.team} RP: {remaining:.0f} IP unallocated")
    
    def to_dataframe(self, allocations: Dict[str, TeamAllocation]) -> pd.DataFrame:
        """Convert allocations to DataFrame."""
        rows = []
        
        for team, team_alloc in allocations.items():
            for result in team_alloc.players:
                pred_data = result.prediction_data or {}
                
                row = {
                    'IDfg': result.id,
                    'Name': result.name,
                    'Team': team,
                    'Position': result.position,
                    'Role': result.role,
                    'Age': pred_data.get('Age'),
                    'Confidence': result.confidence,
                    'Confidence_Tier': result.confidence_tier,
                    'Injury_Multiplier': result.injury_multiplier,
                    'Injury_Name': result.injury_name,
                    'Injury_Return_Date': result.injury_return_date,
                    'Allocated_Games': result.allocated_games if result.allocated_games > 0 else None,
                    'Allocated_IP': result.allocated_ip if result.allocated_ip > 0 else None,
                    'Allocation_Pct': result.allocation_pct,
                    'WAR': result.war,
                }
                
                # WAR components
                if result.batting_runs is not None:
                    row['Batting_Runs'] = result.batting_runs
                if result.baserunning_runs is not None:
                    row['BsR'] = result.baserunning_runs
                if result.fielding_runs is not None:
                    row['Fld'] = result.fielding_runs
                if result.positional_adj is not None:
                    row['Pos_Adj'] = result.positional_adj
                if result.fip_runs is not None:
                    row['FIP_Runs'] = result.fip_runs
                
                # Batter stats
                if result.role is None:
                    row['PA'] = (result.allocated_games * self.budget.PA_PER_GAME 
                                 if result.allocated_games else None)
                    row['AVG'] = pred_data.get('AVG')
                    row['OBP'] = pred_data.get('OBP')
                    row['SLG'] = pred_data.get('SLG')
                    row['wOBA'] = pred_data.get('wOBA')
                    row['wRC+'] = result.wrc_plus
                    row['BB%'] = pred_data.get('BB%')
                    row['K%'] = pred_data.get('K%')
                    row['HR'] = pred_data.get('HR')
                    row['SB'] = pred_data.get('SB')
                
                # Pitcher stats
                if result.role in ['SP', 'RP']:
                    row['ERA'] = pred_data.get('ERA')
                    row['FIP'] = pred_data.get('FIP')
                    row['SIERA'] = pred_data.get('SIERA')
                    row['K%'] = pred_data.get('K%')
                    row['BB%'] = pred_data.get('BB%')
                    row['GB%'] = pred_data.get('GB%')
                    row['FBv'] = pred_data.get('FBv')
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def summarize(self, allocations: Dict[str, TeamAllocation]) -> pd.DataFrame:
        """Generate team-level summary."""
        rows = []
        
        for team, alloc in allocations.items():
            rows.append({
                'Team': team,
                'Position_Games': alloc.total_position_games,
                'SP_IP': alloc.total_sp_ip,
                'RP_IP': alloc.total_rp_ip,
                'Total_IP': alloc.total_sp_ip + alloc.total_rp_ip,
                'Players': len(alloc.players),
            })
        
        return pd.DataFrame(rows).sort_values('Team')
