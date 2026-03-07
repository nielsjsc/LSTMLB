import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Set
import logging
from tqdm import tqdm
from pathlib import Path
import json
import warnings

# Suppress sklearn feature name warnings during prediction
warnings.filterwarnings('ignore', message='X does not have valid feature names')

from .data_processing import DataConfig
from .model_architecture import ImprovedLSTM

logger = logging.getLogger(__name__)


# =============================================================================
# PARK FACTOR TOGGLE HELPER
# =============================================================================

def _is_park_factor_enabled(model_type: str) -> bool:
    """
    Check if park factor adjustment is enabled for the given model type.
    
    Reads ENABLE_PARK_FACTOR_ADJUSTMENT from the appropriate config class.
    Returns False (disabled) if the config cannot be loaded or the attribute
    is missing, maintaining backward compatibility.
    
    Args:
        model_type: One of 'batter', 'SP', 'RP', 'baserunning',
                    'defense_infield', 'defense_outfield', 'defense_catcher'
    """
    try:
        if model_type == 'batter':
            from configs.batter_config import BatterConfig
            return getattr(BatterConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)
        elif model_type == 'SP':
            from configs.pitcher_sp_config import PitcherSPConfig
            return getattr(PitcherSPConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)
        elif model_type == 'RP':
            from configs.pitcher_rp_config import PitcherRPConfig
            return getattr(PitcherRPConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)
        else:
            # Fielding/baserunning inherit from batter config
            from configs.batter_config import BatterConfig
            return getattr(BatterConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)
    except (ImportError, AttributeError):
        return False


# =============================================================================
# COUNTING-STAT ADJUSTMENT FOR X-STAT CONSISTENCY
# =============================================================================

def _compute_player_counting_residuals(
    player_data: pd.DataFrame,
    coefficients: dict,
    input_features: list[str],
) -> dict[str, float]:
    """Compute a player's PA-weighted mean residual for each counting stat.

    For each season where the player has both actual counting stats and actual
    rate stats (wOBA, SLG, AVG), we compute:

        OLS_predicted_C = intercept + β_wOBA * wOBA + β_SLG * SLG + β_AVG * AVG
        residual        = actual_C − OLS_predicted_C

    Then the PA-weighted mean residual across seasons is returned.  This is
    the player's "personal offset" — positive means they consistently produce
    more of that counting stat than the OLS predicts (e.g. Ramirez's HR
    relative to his xSLG).

    Returns a dict  ``{stat_name: mean_residual}``.  Stats not present or not
    in input_features are omitted.
    """
    from core.rate_stats_config import BATTING_COUNTING_STATS
    residuals: dict[str, float] = {}

    # Need wOBA/SLG/AVG and PA to compute residuals
    required = {'wOBA', 'SLG', 'AVG', 'PA', 'G'}
    if not required.issubset(player_data.columns):
        return residuals

    # Only use seasons with enough PA and non-NaN rate stats
    mask = (
        (player_data['PA'] >= 50) &
        player_data['wOBA'].notna() &
        player_data['SLG'].notna() &
        player_data['AVG'].notna()
    )
    valid = player_data.loc[mask]
    if len(valid) == 0:
        return residuals

    pa = valid['PA'].values.astype(float)

    for stat, coefs in coefficients.items():
        if stat not in valid.columns or stat not in input_features:
            continue

        # Per-150 counting stat.  If the data has already been through
        # calculate_rate_stats the column is already per-150; otherwise
        # convert raw → per-150 on the fly.
        raw_c = valid[stat].values.astype(float)
        g = valid['G'].values.astype(float)

        # Heuristic: if mean(stat) > 100 and stat is in BATTING_COUNTING_STATS,
        # it's likely still raw (not per-150).  Per-150 HR rarely exceeds 80.
        if stat in BATTING_COUNTING_STATS and np.nanmean(raw_c) > 80 and np.nanmean(g) > 0:
            c_per150 = np.where(g > 0, raw_c / g * 150, 0.0)
        else:
            c_per150 = raw_c

        woba = valid['wOBA'].values.astype(float)
        slg = valid['SLG'].values.astype(float)
        avg = valid['AVG'].values.astype(float)

        intercept = coefs.get('intercept', 0.0)
        ols_pred = intercept + coefs['wOBA'] * woba + coefs['SLG'] * slg + coefs['AVG'] * avg
        season_resid = c_per150 - ols_pred

        # PA-weighted mean
        total_pa = pa.sum()
        if total_pa > 0:
            residuals[stat] = float(np.sum(season_resid * pa) / total_pa)

    return residuals


def _compute_player_woba_gap(
    player_data: pd.DataFrame,
) -> tuple[float, float]:
    """Compute a player's PA-weighted career (wOBA − xwOBA) gap.

    Returns ``(career_gap, statcast_pa)`` where *career_gap* is the
    PA-weighted mean of  ``wOBA − xwOBA``  across all statcast-era seasons
    (those that have a non-null xwOBA).  Positive = player consistently
    outperforms xwOBA (e.g. Ramirez).

    *statcast_pa* is the total PA from those seasons — used downstream
    to blend the gap toward zero for small samples.
    """
    if 'xwOBA' not in player_data.columns or 'wOBA' not in player_data.columns:
        return 0.0, 0.0
    mask = player_data['xwOBA'].notna() & player_data['wOBA'].notna() & (player_data['PA'] >= 50)
    valid = player_data.loc[mask]
    if len(valid) == 0:
        return 0.0, 0.0
    pa = valid['PA'].values.astype(float)
    gaps = (valid['wOBA'].values - valid['xwOBA'].values).astype(float)
    total_pa = pa.sum()
    if total_pa == 0:
        return 0.0, 0.0
    return float(np.sum(gaps * pa) / total_pa), float(total_pa)


def _adjust_counting_stats_for_xstats(
    data: pd.DataFrame,
    orig_woba: np.ndarray | None,
    orig_slg: np.ndarray | None,
    orig_avg: np.ndarray | None,
    input_features: list[str],
    player_data: pd.DataFrame | None = None,
) -> None:
    """Adjust counting stats so they are consistent with x-stat substitutions.

    **Population mode** (``USE_PLAYER_SPECIFIC_XSTAT_ADJUSTMENT = False``)

    Uses additive OLS-derived coefficients::

        Δ = β_wOBA·(new_wOBA − orig_wOBA)
          + β_SLG ·(new_SLG  − orig_SLG)
          + β_AVG ·(new_AVG  − orig_AVG)
        stat_adjusted = max(0, stat + Δ)

    **Player-specific mode** (``USE_PLAYER_SPECIFIC_XSTAT_ADJUSTMENT = True``)

    Replaces the OLS delta with a **proportional wOBA-ratio** adjustment that
    naturally preserves each player's counting-stat mix (e.g. Raleigh's
    HR-heavy / low-2B profile).  The wOBA ratio is softened for players who
    consistently outperform xwOBA (Ramirez)::

        career_gap   = PA-weighted mean(wOBA − xwOBA)  across statcast seasons
        weight       = min(statcast_PA / XSTAT_PA_FULL_WEIGHT, 1.0)
        effective_Δ  = (new_wOBA − orig_wOBA) − weight × career_gap
        ratio        = max(0.5, 1 + effective_Δ / orig_wOBA)
        stat_adjusted = max(0, stat × ratio)

    For a Ramirez-type player with career_gap ≈ +0.02 and full weight:
    the ratio is closer to 1.0, producing a much smaller adjustment than
    the population OLS.  For a 1-year player with no statcast history,
    weight → 0 and the ratio adjustment uses the raw xwOBA delta (roughly
    equivalent to the population model).

    Also blends in an OLS-derived residual offset for each counting stat
    (additive) to capture player-level intercept differences — e.g. Judge
    producing +5.5 HR/150 more than the OLS predicts given his rate stats.

    Modifies *data* **in-place**.  No-op when no substitution deltas exist.
    """
    try:
        from configs.batter_config import BatterConfig
        coefficients = getattr(BatterConfig, 'XSTAT_COUNTING_ADJUSTMENT_COEFFICIENTS', {})
    except (ImportError, AttributeError):
        return

    if not coefficients:
        return

    n = len(data)

    # Compute per-row deltas (current value − original value).  If a stat was
    # not substituted (or not present), delta is 0.
    d_woba = (data['wOBA'].values - orig_woba) if ('wOBA' in data.columns and orig_woba is not None) else np.zeros(n)
    d_slg  = (data['SLG'].values - orig_slg)  if ('SLG' in data.columns and orig_slg is not None)  else np.zeros(n)
    d_avg  = (data['AVG'].values - orig_avg)  if ('AVG' in data.columns and orig_avg is not None)  else np.zeros(n)

    # Short-circuit when nothing was substituted
    if np.allclose(d_woba, 0) and np.allclose(d_slg, 0) and np.allclose(d_avg, 0):
        return

    # ------------------------------------------------------------------
    # Player-specific mode: proportional wOBA-ratio + residual offset
    # ------------------------------------------------------------------
    use_player_specific = False
    try:
        from configs.batter_config import BatterConfig
        use_player_specific = getattr(BatterConfig, 'USE_PLAYER_SPECIFIC_XSTAT_ADJUSTMENT', False)
    except (ImportError, AttributeError):
        pass

    if use_player_specific and player_data is not None and len(player_data) > 0:
        # Career wOBA overperformance gap
        career_gap, statcast_pa = _compute_player_woba_gap(player_data)
        pa_full = getattr(BatterConfig, 'XSTAT_PA_FULL_WEIGHT', 2000)
        gap_weight = min(statcast_pa / pa_full, 1.0)

        # Effective wOBA delta — shrunk for consistent outperformers
        effective_d_woba = d_woba - gap_weight * career_gap  # row-level (n,)

        # Proportional ratio (bounded to avoid extreme or negative values)
        safe_orig_woba = np.where(
            (orig_woba is not None) & (orig_woba > 0.15),
            orig_woba, 0.300,
        )
        ratio = np.clip(1.0 + effective_d_woba / safe_orig_woba, 0.50, 1.50)

        # OLS residuals (intercept-corrected) — for additive player-level offset
        player_residuals = _compute_player_counting_residuals(
            player_data, coefficients, input_features,
        )
        career_pa = float(player_data['PA'].sum()) if 'PA' in player_data.columns else 0.0
        resid_weight = min(career_pa / pa_full, 1.0)

        adjusted_any = False
        for stat, coefs in coefficients.items():
            if stat not in data.columns or stat not in input_features:
                continue

            # Proportional adjustment (preserves player's counting-stat profile)
            stat_vals = data[stat].values.astype(float)
            new_vals = stat_vals * ratio

            # Add player-specific residual offset (additive, captures level diff)
            if stat in player_residuals and resid_weight > 0:
                new_vals += resid_weight * player_residuals[stat]

            data[stat] = np.maximum(0.0, new_vals)
            adjusted_any = True

            logger.debug(
                f"  {stat}: proportional ratio mean={ratio.mean():.4f}, "
                f"career_gap={career_gap:+.4f}, gap_weight={gap_weight:.2f}, "
                f"residual={player_residuals.get(stat, 0):+.2f}"
            )

        if adjusted_any:
            logger.debug(
                f"Player-specific counting adjustment applied "
                f"(effective Δ_wOBA mean={effective_d_woba.mean():.4f}, "
                f"career_gap={career_gap:+.4f}, gap_weight={gap_weight:.2f})"
            )
        return

    # ------------------------------------------------------------------
    # Population mode: additive OLS delta (original behaviour)
    # ------------------------------------------------------------------
    adjusted_any = False
    for stat, coefs in coefficients.items():
        if stat not in data.columns or stat not in input_features:
            continue
        adjustment = (coefs['wOBA'] * d_woba +
                      coefs['SLG'] * d_slg +
                      coefs['AVG'] * d_avg)
        data[stat] = np.maximum(0.0, data[stat].values + adjustment)
        adjusted_any = True

    if adjusted_any:
        logger.debug("Counting stats adjusted for x-stat consistency "
                     f"(Δ_wOBA mean={d_woba.mean():.4f}, Δ_SLG mean={d_slg.mean():.4f}, "
                     f"Δ_AVG mean={d_avg.mean():.4f})")


# =============================================================================
# AGING CONSTRAINT ENFORCEMENT (POST-PREDICTION)
# =============================================================================

class AgingEnforcer:
    """
    Enforces aging constraints on predictions to prevent unrealistic improvements.
    
    This is applied AFTER model prediction to ensure older players don't improve
    on metrics where decline is expected.
    """
    
    # Defensive stats where higher = better (should decline with age)
    DEFENSE_METRICS = [
        'OAA/150', 'DRS/150', 'sc_total_runs/150', 'sc_range_runs/150', 
        'sc_arm_runs/150', 'sc_dp_runs/150', 'sc_framing_runs/150',
        'sc_throwing_runs/150', 'sc_blocking_runs/150'
    ]
    
    def __init__(self, params_path: Optional[Path] = None):
        """Load aging parameters."""
        if params_path is None:
            params_path = Path(__file__).parent.parent / "analysis" / "aging_parameters.json"
        
        self.params = {}
        if params_path.exists():
            with open(params_path) as f:
                self.params = json.load(f)
            logger.info(f"Loaded aging parameters for prediction enforcement")
        else:
            logger.warning(f"Aging parameters not found at {params_path}")
    
    def get_decline_rate(self, category: str, stat: str, age: int) -> float:
        """Get expected decline rate for a stat at given age."""
        # Map category to aging params key
        cat_key = category
        if category in ['fielding_infield', 'fielding_outfield', 'fielding_catcher']:
            cat_key = category
        elif category == 'defense':
            # Try to determine from stat
            if 'framing' in stat.lower() or 'throwing' in stat.lower() or 'blocking' in stat.lower():
                cat_key = 'fielding_catcher'
            elif 'dp_runs' in stat.lower():
                cat_key = 'fielding_infield'
            else:
                cat_key = 'fielding_outfield'  # Default
        
        cat_data = self.params.get(cat_key, {})
        stat_data = cat_data.get(stat, {})
        decline_by_band = stat_data.get('decline_by_age_band', {})
        
        # Find age band
        if 21 <= age <= 25:
            band = '21-25'
        elif 26 <= age <= 30:
            band = '26-30'
        elif 31 <= age <= 35:
            band = '31-35'
        elif 36 <= age <= 40:
            band = '36-40'
        elif 41 <= age <= 45:
            band = '41-45'
        else:
            band = '41-45'  # Use oldest band for very old players
        
        band_data = decline_by_band.get(band, {})
        # Use corrected value
        decline = band_data.get('decline_per_year_corrected')
        if decline is None:
            decline = band_data.get('decline_per_year', 0.0)
        
        return decline if decline is not None else 0.0
    
    def enforce_aging(
        self, 
        predictions: List[Dict], 
        category: str,
        metrics: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Enforce aging constraints on a list of predictions.
        
        For players 30+, prevents improvement on defensive metrics.
        Instead applies expected decline based on empirical aging curves.
        
        Args:
            predictions: List of prediction dicts (must be sorted by year)
            category: 'fielding_infield', 'fielding_outfield', 'fielding_catcher', etc.
            metrics: List of metrics to enforce (defaults to DEFENSE_METRICS)
            
        Returns:
            Adjusted predictions list
        """
        if not predictions or len(predictions) < 2:
            return predictions
        
        if metrics is None:
            metrics = self.DEFENSE_METRICS
        
        # Work on a copy
        adjusted = [p.copy() for p in predictions]
        
        for i in range(1, len(adjusted)):
            prev = adjusted[i - 1]
            curr = adjusted[i]
            age = curr.get('Age', 0)
            
            # Only enforce for players 30+
            if age < 30:
                continue
            
            for metric in metrics:
                if metric not in curr or metric not in prev:
                    continue
                
                prev_val = prev[metric]
                curr_val = curr[metric]
                
                # For defense metrics (higher = better), improvement = increase
                improvement = curr_val - prev_val
                
                if improvement > 0:
                    # Get expected decline
                    expected_decline = self.get_decline_rate(category, metric, age)
                    
                    if expected_decline > 0:
                        # Should be declining, not improving
                        # Apply decline from previous value
                        curr[metric] = prev_val - expected_decline
                    else:
                        # No decline expected (rare), cap improvement at 0
                        curr[metric] = prev_val
                    
                    # Update adjusted
                    adjusted[i] = curr
        
        return adjusted


# Global enforcer instance
_aging_enforcer = None

def get_aging_enforcer() -> AgingEnforcer:
    """Get or create the global aging enforcer."""
    global _aging_enforcer
    if _aging_enforcer is None:
        _aging_enforcer = AgingEnforcer()
    return _aging_enforcer


def generate_batter_names(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Generate unique batter names from the raw dataframe"""
    player_names = pd.DataFrame(raw_df[['Name', 'IDfg']].drop_duplicates()).sort_values('Name')
    return player_names


def load_model_from_checkpoint(checkpoint_path: str, data_config: DataConfig, device: torch.device) -> ImprovedLSTM:
    """Load a model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    model_config = checkpoint['config']
    
    # Check for feature mismatch
    checkpoint_input_size = model_config['input_size']
    config_input_size = len(data_config.input_features)
    if checkpoint_input_size != config_input_size:
        logger.error(f"FEATURE MISMATCH: Checkpoint expects {checkpoint_input_size} features, but config has {config_input_size} features")
        logger.error(f"Config features: {data_config.input_features}")
        raise ValueError(f"Cannot load model trained with {checkpoint_input_size} features using config with {config_input_size} features. "
                        f"Either retrain the model with current config features, or update config to match checkpoint.")
    
    # Detect the actual hidden size from the state dict
    state_dict = checkpoint['model_state_dict']
    
    # Check the input projection layer size to determine the actual hidden size used
    if 'input_projection.0.weight' in state_dict:
        actual_hidden_size = state_dict['input_projection.0.weight'].shape[0]
        logger.info(f"Detected actual hidden size from checkpoint: {actual_hidden_size}")
    else:
        # Fallback to config value
        actual_hidden_size = model_config['hidden_size']
        logger.warning("Could not detect hidden size from state dict, using config value")
    
    # Create model with the actual architecture parameters (no internal modifications anymore)
    model = ImprovedLSTM(
        input_size=model_config['input_size'],
        hidden_size=actual_hidden_size,  # Use actual hidden size directly
        num_layers=model_config['num_layers'],    
        output_size=model_config['output_size'],
        dropout=model_config['dropout'],  # Use actual dropout directly
        bidirectional=model_config.get('bidirectional', True),
        num_heads=model_config.get('num_heads', 4)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


# =============================================================================
# UNIFIED PREDICTION ENGINE
# =============================================================================

def _prepare_player_sequence(
    player_id: str,
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    input_features: List[str],
    seq_length: int,
    cutoff_year: Optional[int] = None
) -> Optional[Dict]:
    """
    Prepare player data and sequence for prediction.
    
    This is the shared preprocessing logic for batter, fielding, and baserunning predictions.
    
    Args:
        player_id: FanGraphs player ID
        raw_df: Historical player data (already filtered to this player if needed)
        player_names: DataFrame mapping IDfg to Name
        input_features: List of feature names for the model
        seq_length: Number of historical seasons to use
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1).
                    If provided, this overrides the player's actual latest_season.
        
    Returns:
        Dict with 'player_name', 'sequence', 'latest_age', 'latest_season', 'n_features'
        or None if player data is invalid
    """
    # Get player data
    player_data = raw_df[raw_df['IDfg'] == player_id].copy()
    if len(player_data) == 0:
        return None
    
    # Get player name
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        logger.warning(f"Player name not found for ID {player_id}")
        return None
    
    # Sort by season
    player_data = player_data.sort_values('Season')
    
    # Handle players with fewer seasons than seq_length by padding
    num_seasons = len(player_data)
    if num_seasons < seq_length:
        recent_data = player_data[input_features].copy()
        while len(recent_data) < seq_length:
            recent_data = pd.concat([recent_data, recent_data.iloc[-1:]], ignore_index=True)
    else:
        recent_data = player_data[input_features].iloc[-seq_length:].copy().reset_index(drop=True)
    
    # =========================================================================
    # STATCAST METRIC SUBSTITUTION (before park neutralization)
    # =========================================================================
    # x-stats replace traditional counterparts first so that park factor
    # neutralization operates on the final values the model will see.
    #
    # Save originals before substitution — needed by counting-stat adjustment.
    _orig_woba = recent_data['wOBA'].values.copy() if 'wOBA' in recent_data.columns else None
    _orig_slg  = recent_data['SLG'].values.copy() if 'SLG' in recent_data.columns else None
    _orig_avg  = recent_data['AVG'].values.copy() if 'AVG' in recent_data.columns else None

    xwoba_substituted = False
    xba_substituted = False
    xslg_substituted = False

    try:
        from configs.batter_config import BatterConfig

        # xwOBA substitution
        if (BatterConfig.USE_XWOBA_FOR_PREDICTIONS and
                'wOBA' in input_features and
                'xwOBA' in player_data.columns):
            original_woba = recent_data['wOBA'].values.copy()

            if num_seasons < seq_length:
                xwoba_candidate = player_data['xwOBA'].copy()
                while len(xwoba_candidate) < seq_length:
                    xwoba_candidate = pd.concat([xwoba_candidate.iloc[:1], xwoba_candidate], ignore_index=True)
                xwoba_candidate = xwoba_candidate.values
            else:
                xwoba_candidate = player_data['xwOBA'].iloc[-seq_length:].values

            # Only substitute if all values are valid — avoids introducing NaN for pre-Statcast seasons
            if not pd.isna(xwoba_candidate).any():
                recent_data['wOBA'] = xwoba_candidate
                xwoba_substituted = True
                logger.debug(f"Player {player_id}: Substituted xwOBA → wOBA position. "
                             f"Original wOBA: {original_woba}, New xwOBA: {recent_data['wOBA'].values}")
            else:
                logger.debug(f"Player {player_id}: Skipped xwOBA substitution — NaN values present for this sequence")

        # xBA substitution (AVG → xBA)
        if (BatterConfig.USE_XBA_FOR_PREDICTIONS and
                'AVG' in input_features and
                'xBA' in player_data.columns):
            original_avg = recent_data['AVG'].values.copy()

            if num_seasons < seq_length:
                xba_candidate = player_data['xBA'].copy()
                while len(xba_candidate) < seq_length:
                    xba_candidate = pd.concat([xba_candidate.iloc[:1], xba_candidate], ignore_index=True)
                xba_candidate = xba_candidate.values
            else:
                xba_candidate = player_data['xBA'].iloc[-seq_length:].values

            # Only substitute if all values are valid — avoids introducing NaN for pre-Statcast seasons
            if not pd.isna(xba_candidate).any():
                recent_data['AVG'] = xba_candidate
                xba_substituted = True
                logger.debug(f"Player {player_id}: Substituted xBA → AVG position. "
                             f"Original AVG: {original_avg}, New xBA: {recent_data['AVG'].values}")
            else:
                logger.debug(f"Player {player_id}: Skipped xBA substitution — NaN values present for this sequence")

        # xSLG substitution (SLG → xSLG)
        if (BatterConfig.USE_XSLG_FOR_PREDICTIONS and
                'SLG' in input_features and
                'xSLG' in player_data.columns):
            original_slg = recent_data['SLG'].values.copy()

            if num_seasons < seq_length:
                xslg_candidate = player_data['xSLG'].copy()
                while len(xslg_candidate) < seq_length:
                    xslg_candidate = pd.concat([xslg_candidate.iloc[:1], xslg_candidate], ignore_index=True)
                xslg_candidate = xslg_candidate.values
            else:
                xslg_candidate = player_data['xSLG'].iloc[-seq_length:].values

            # Only substitute if all values are valid — avoids introducing NaN for pre-Statcast seasons
            if not pd.isna(xslg_candidate).any():
                recent_data['SLG'] = xslg_candidate
                xslg_substituted = True
                logger.debug(f"Player {player_id}: Substituted xSLG → SLG position. "
                             f"Original SLG: {original_slg}, New xSLG: {recent_data['SLG'].values}")
            else:
                logger.debug(f"Player {player_id}: Skipped xSLG substitution — NaN values present for this sequence")

    except (ImportError, AttributeError):
        # Config not available, skip substitution
        pass

    # =========================================================================
    # COUNTING-STAT ADJUSTMENT FOR X-STAT CONSISTENCY
    # =========================================================================
    # When x-stats were substituted above, counting stats (HR, 2B, …) still
    # reflect actual outcomes.  Adjust them so they are consistent with the
    # substituted rate stats, using regression-derived coefficients.
    try:
        from configs.batter_config import BatterConfig
        if getattr(BatterConfig, 'ADJUST_COUNTING_STATS_TO_XSTATS', False):
            _adjust_counting_stats_for_xstats(
                recent_data, _orig_woba, _orig_slg, _orig_avg, input_features,
                player_data=player_data,
            )
    except (ImportError, AttributeError):
        pass

    # =========================================================================
    # PARK FACTOR NEUTRALIZATION (after x-stat substitution)
    # =========================================================================
    # Applied to all adjustable features — including whichever positions now hold
    # x-stat values — so every value in the sequence is park-neutral before the model.
    park_factor_enabled = False
    try:
        from configs.batter_config import BatterConfig
        park_factor_enabled = getattr(BatterConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False)
    except (ImportError, AttributeError):
        pass

    if park_factor_enabled and 'Team' in player_data.columns:
        from core.park_factors import get_park_factor, EXCLUDED_STATS
        adjustable_features = [f for f in input_features if f not in EXCLUDED_STATS]

        # Get the teams for the seasons in the sequence
        if num_seasons < seq_length:
            teams_for_seq = [player_data['Team'].iloc[-1]] * seq_length
        else:
            teams_for_seq = player_data['Team'].iloc[-seq_length:].tolist()

        # Apply per-season neutralization to the DataFrame
        for row_idx, team in enumerate(teams_for_seq):
            pf = get_park_factor(team)
            if pf != 1.0:
                for feat in adjustable_features:
                    if feat in recent_data.columns:
                        recent_data.iloc[row_idx, recent_data.columns.get_loc(feat)] = (
                            recent_data.iloc[row_idx][feat] / pf
                        )

        logger.debug(f"Player {player_id}: Park factor neutralization applied to {len(adjustable_features)} features")
    
    # Check for valid data (no NaN)
    if recent_data.isna().any().any():
        return None
    
    sequence = recent_data.values.astype(np.float64)
    
    latest_age = recent_data['Age'].iloc[-1] if 'Age' in recent_data.columns else player_data['Age'].iloc[-1]
    
    # Use cutoff_year if provided (for consistent projection years),
    # otherwise use player's actual latest season
    if cutoff_year is not None:
        latest_season = cutoff_year
        # Adjust age if cutoff_year is ahead of player's actual latest season
        actual_latest_season = player_data['Season'].max()
        if cutoff_year > actual_latest_season:
            years_ahead = cutoff_year - actual_latest_season
            latest_age = latest_age + years_ahead
    else:
        latest_season = player_data['Season'].max()
    
    return {
        'player_name': player_name,
        'sequence': sequence,
        'latest_age': latest_age,
        'latest_season': latest_season,
        'n_features': len(input_features),
        'xwoba_substituted': xwoba_substituted,
        'xba_substituted': xba_substituted,
        'xslg_substituted': xslg_substituted
    }


def _run_prediction_loop(
    model: ImprovedLSTM,
    scaler: Any,
    sequence: np.ndarray,
    input_features: List[str],
    seq_length: int,
    future_years: int,
    latest_age: float,
    latest_season: int,
    player_name: str,
    player_id: str,
    extra_fields: Optional[Dict] = None,
    non_negative_features: Optional[List[str]] = None
) -> List[Dict]:
    """
    Run the prediction loop for multiple future years.
    
    This is the shared prediction logic for batter, fielding, and baserunning.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted scaler for features
        sequence: Initial sequence array (unscaled)
        input_features: List of feature names
        seq_length: Sequence length
        future_years: Number of years to project
        latest_age: Player's age in their last season
        latest_season: Player's last season year
        player_name: Player's name for output
        player_id: Player's IDfg for output
        extra_fields: Additional fields to add to each prediction dict (e.g., Position_Group)
        non_negative_features: Features that should be clipped to >= 0
        
    Returns:
        List of prediction dictionaries
    """
    # Scale the sequence
    try:
        sequence_scaled = scaler.transform(sequence)
    except Exception as e:
        logger.error(f"Scaling error for player {player_id}: {e}")
        return []
    
    n_features = len(input_features)
    predictions = []
    device = next(model.parameters()).device
    
    for year_offset in range(1, future_years + 1):
        year = latest_season + year_offset
        age = latest_age + year_offset
        
        # Run model prediction
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
            lengths = torch.tensor([seq_length], dtype=torch.int64).to(device)
            output = model(seq_tensor, lengths)
            pred_numpy = output.cpu().numpy()[0]
        
        # Inverse transform to get actual values
        try:
            unscaled_pred = scaler.inverse_transform(pred_numpy.reshape(1, -1))[0]
            
            # Build prediction dictionary
            prediction_dict = {
                'Name': player_name,
                'IDfg': player_id,
                'Year': year,
                'Age': age,
            }
            
            # Add extra fields if provided (e.g., Position_Group, Role)
            if extra_fields:
                prediction_dict.update(extra_fields)
            
            # Add all input features to prediction
            for i, feature in enumerate(input_features):
                if feature == 'Age':
                    prediction_dict[feature] = age
                else:
                    value = unscaled_pred[i]
                    # Apply non-negative constraint if specified
                    if non_negative_features and feature in non_negative_features:
                        value = max(0, value)
                    prediction_dict[feature] = value
            
            predictions.append(prediction_dict)
            
            # Update sequence for next prediction
            # NOTE: pred_numpy is already in scaled space (direct model output)
            # Only the age component needs to be updated with the scaled next year's age
            age_index = input_features.index('Age')
            age_update = np.zeros(n_features)
            age_update[age_index] = age + 1  # Next year's age (unscaled)
            
            # Update just the age in the already-scaled prediction
            pred_numpy[age_index] = scaler.transform(age_update.reshape(1, -1))[0][age_index]
            
            # Slide the sequence window
            sequence_scaled = np.vstack([sequence_scaled[1:], pred_numpy])
            
        except Exception as e:
            logger.error(f"Prediction error for player {player_id}, year {year}: {e}")
            break
    
    return predictions


def predict_future_stats(
    player_id: str,
    input_features: List[str],
    model: ImprovedLSTM,
    scaler: Any,
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    seq_length: int,
    future_years: int = 16,
    extra_fields: Optional[Dict] = None,
    non_negative_features: Optional[List[str]] = None,
    cutoff_year: Optional[int] = None,
    model_type: Optional[str] = None,
    league_priors: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Unified prediction function for batters, fielders, and baserunners.
    
    When model_type is provided and ENABLE_RELIABILITY_REGRESSION is True
    in the corresponding config, this function:
    1. Applies reliability regression to the player's historical sequence
    2. Uses regressed career mean for padding (instead of naive duplication)
    3. Passes age-adjusted priors so padding reflects current talent
    
    Args:
        player_id: FanGraphs player ID
        input_features: List of feature names for the model
        model: Trained LSTM model
        scaler: Fitted scaler for features
        raw_df: Historical player data
        player_names: DataFrame mapping IDfg to Name
        seq_length: Number of historical seasons to use (from config)
        future_years: Number of years to project
        extra_fields: Additional fields to add to each prediction (e.g., {'Position_Group': 'infield'})
        non_negative_features: Features that should be clipped to >= 0 (e.g., ['SB_rate', 'CS_rate'])
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        model_type: Model type for reliability regression ('batter', 'baserunning',
                    'defense_infield', 'defense_outfield', 'defense_catcher').
                    If None, regression is skipped (legacy behavior).
        league_priors: Pre-computed league averages per feature (for regression fallback).
                      If None and regression is enabled, uses player's own data.
        
    Returns:
        List of prediction dictionaries
    """
    # Check if reliability regression is enabled for this model type
    use_regression = False
    if model_type is not None:
        try:
            from core.data_processing import _is_reliability_regression_enabled
            use_regression = _is_reliability_regression_enabled(model_type, context='prediction')
        except ImportError:
            pass
    
    if use_regression:
        return _predict_with_regression(
            player_id=player_id,
            input_features=input_features,
            model=model,
            scaler=scaler,
            raw_df=raw_df,
            player_names=player_names,
            seq_length=seq_length,
            future_years=future_years,
            extra_fields=extra_fields,
            non_negative_features=non_negative_features,
            cutoff_year=cutoff_year,
            model_type=model_type,
            league_priors=league_priors,
        )
    else:
        # Legacy path: no regression
        prep = _prepare_player_sequence(player_id, raw_df, player_names, input_features, seq_length, cutoff_year)
        if prep is None:
            return []
        
        return _run_prediction_loop(
            model=model,
            scaler=scaler,
            sequence=prep['sequence'],
            input_features=input_features,
            seq_length=seq_length,
            future_years=future_years,
            latest_age=prep['latest_age'],
            latest_season=prep['latest_season'],
            player_name=prep['player_name'],
            player_id=player_id,
            extra_fields=extra_fields,
            non_negative_features=non_negative_features
        )


def _predict_with_regression(
    player_id: str,
    input_features: List[str],
    model: ImprovedLSTM,
    scaler: Any,
    raw_df: pd.DataFrame,
    player_names: pd.DataFrame,
    seq_length: int,
    future_years: int,
    extra_fields: Optional[Dict],
    non_negative_features: Optional[List[str]],
    cutoff_year: Optional[int],
    model_type: str,
    league_priors: Optional[Dict[str, float]],
) -> List[Dict]:
    """
    Predict future stats with reliability regression applied to the input sequence.
    
    Mirrors the pitcher pipeline's approach:
    1. Regress each season's rate stats toward career mean (age-adjusted)
    2. Use regressed career mean for padding short sequences
    3. Feed the regressed, properly-padded sequence to the LSTM
    """
    from core.reliability import (
        regress_player_sequence,
        compute_regressed_career_mean,
        get_era_for_features,
    )
    
    # Get player data
    player_data = raw_df[raw_df['IDfg'] == player_id].copy()
    if len(player_data) == 0:
        return []
    
    # Get player name
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        logger.warning(f"Player name not found for ID {player_id}")
        return []
    
    # Sort by season
    player_data = player_data.sort_values('Season')
    
    # Determine latest season and age
    if cutoff_year is not None:
        latest_season = cutoff_year
        actual_latest_season = player_data['Season'].max()
        latest_age = player_data[player_data['Season'] == actual_latest_season]['Age'].iloc[-1]
        if cutoff_year > actual_latest_season:
            latest_age += (cutoff_year - actual_latest_season)
    else:
        latest_season = player_data['Season'].max()
        latest_age = player_data['Age'].iloc[-1]
    
    # Map model_type to the reliability module's key
    reliability_model_type = model_type

    # =========================================================================
    # STATCAST METRIC SUBSTITUTION (batter only) — BEFORE REGRESSION
    # xStats replace their traditional counterparts in player_data so that
    # regression operates on the more-predictive expected metrics.
    # Park neutralization (below) then applies to all adjustable features,
    # including the positions now holding x-stat values.
    # =========================================================================
    if model_type == 'batter':
        # Save originals before substitution — needed by counting-stat adjustment
        _orig_woba = player_data['wOBA'].values.copy() if 'wOBA' in player_data.columns else None
        _orig_slg  = player_data['SLG'].values.copy() if 'SLG' in player_data.columns else None
        _orig_avg  = player_data['AVG'].values.copy() if 'AVG' in player_data.columns else None

        try:
            from configs.batter_config import BatterConfig

            if (BatterConfig.USE_XWOBA_FOR_PREDICTIONS and
                    'wOBA' in input_features and
                    'xwOBA' in player_data.columns):
                mask = player_data['xwOBA'].notna()
                player_data.loc[mask, 'wOBA'] = player_data.loc[mask, 'xwOBA']

            elif (getattr(BatterConfig, 'USE_XWOBA_BLEND_FOR_PREDICTIONS', False) and
                    'wOBA' in input_features and
                    'xwOBA' in player_data.columns):
                mask = player_data['xwOBA'].notna()
                player_data.loc[mask, 'wOBA'] = (
                    player_data.loc[mask, 'wOBA'] + player_data.loc[mask, 'xwOBA']
                ) / 2

            if (BatterConfig.USE_XBA_FOR_PREDICTIONS and
                    'AVG' in input_features and
                    'xBA' in player_data.columns):
                mask = player_data['xBA'].notna()
                player_data.loc[mask, 'AVG'] = player_data.loc[mask, 'xBA']

            if (BatterConfig.USE_XSLG_FOR_PREDICTIONS and
                    'SLG' in input_features and
                    'xSLG' in player_data.columns):
                mask = player_data['xSLG'].notna()
                player_data.loc[mask, 'SLG'] = player_data.loc[mask, 'xSLG']

        except (ImportError, AttributeError):
            pass

        # -----------------------------------------------------------------
        # COUNTING-STAT ADJUSTMENT FOR X-STAT CONSISTENCY
        # -----------------------------------------------------------------
        try:
            from configs.batter_config import BatterConfig
            if getattr(BatterConfig, 'ADJUST_COUNTING_STATS_TO_XSTATS', False):
                _adjust_counting_stats_for_xstats(
                    player_data, _orig_woba, _orig_slg, _orig_avg, input_features
                )
        except (ImportError, AttributeError):
            pass

    # Apply reliability regression to the player's historical data
    era = get_era_for_features(input_features)
    player_data_regressed = regress_player_sequence(
        player_data, input_features,
        model_type=reliability_model_type,
        era=era, league_priors=league_priors
    )

    # Compute regressed career mean for padding
    career_mean = compute_regressed_career_mean(
        player_data, input_features,
        model_type=reliability_model_type,
        era=era, league_priors=league_priors
    )

    # =========================================================================
    # BUILD SEQUENCE (with regressed career mean padding)
    # =========================================================================
    recent_data = player_data_regressed[input_features].iloc[-seq_length:].copy().reset_index(drop=True)
    
    # Pad with regressed career mean if not enough seasons
    num_seasons = len(recent_data)
    if num_seasons < seq_length:
        padding_vector = np.array(
            [career_mean.get(f, 0.0) for f in input_features], dtype=np.float64
        )
        n_pad = seq_length - num_seasons
        padding_df = pd.DataFrame(
            [padding_vector] * n_pad,
            columns=input_features
        )
        recent_data = pd.concat([padding_df, recent_data], ignore_index=True)
    
    # Check for NaN
    if recent_data.isna().any().any():
        logger.debug(f"NaN in sequence for player {player_id} — filling with career mean")
        for feat in input_features:
            if recent_data[feat].isna().any():
                fill_val = career_mean.get(feat, 0.0)
                recent_data[feat] = recent_data[feat].fillna(fill_val)
    
    sequence = recent_data.values.astype(np.float64)
    
    # Park factor neutralization: convert stats to park-neutral (true talent)
    # Only applied when ENABLE_PARK_FACTOR_ADJUSTMENT is True in the relevant config.
    # Applied to all adjustable features including x-stat-substituted positions.
    park_factor_enabled = _is_park_factor_enabled(model_type)

    if park_factor_enabled and 'Team' in player_data.columns:
        from core.park_factors import get_park_factor, EXCLUDED_STATS
        # Apply to all features except always-excluded ones (Age, wRC+, etc.)
        adjustable_indices = [
            i for i, f in enumerate(input_features) if f not in EXCLUDED_STATS
        ]
        # Get per-season teams for the sequence
        num_actual = min(len(player_data_regressed), seq_length)
        season_teams = player_data_regressed['Team'].iloc[-num_actual:].tolist() if 'Team' in player_data_regressed.columns else []
        last_team = player_data['Team'].iloc[-1] if len(player_data) > 0 else None
        n_pad = seq_length - len(season_teams)
        all_teams = [last_team] * n_pad + season_teams
        
        for row_idx, team in enumerate(all_teams):
            pf = get_park_factor(team)
            if pf != 1.0:
                for col_idx in adjustable_indices:
                    sequence[row_idx, col_idx] = sequence[row_idx, col_idx] / pf
    
    # Final NaN/Inf safety
    if np.isnan(sequence).any() or np.isinf(sequence).any():
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        logger.warning(f"NaN/Inf cleaned from sequence for player {player_id}")
    
    # Run prediction loop
    return _run_prediction_loop(
        model=model,
        scaler=scaler,
        sequence=sequence,
        input_features=input_features,
        seq_length=seq_length,
        future_years=future_years,
        latest_age=latest_age,
        latest_season=latest_season,
        player_name=player_name,
        player_id=player_id,
        extra_fields=extra_fields,
        non_negative_features=non_negative_features
    )


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================================================================
# These maintain the old function signatures for existing code

def predict_future_stats_batter(player_id: str, input_features: List[str], model: ImprovedLSTM,
                               scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                               seq_length: int = 5, future_years: int = 16, cutoff_year: Optional[int] = None,
                               league_priors: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Predict future stats for a batter. Wrapper around unified predict_future_stats."""
    return predict_future_stats(
        player_id=player_id,
        input_features=input_features,
        model=model,
        scaler=scaler,
        raw_df=raw_df,
        player_names=player_names,
        seq_length=seq_length,
        future_years=future_years,
        cutoff_year=cutoff_year,
        model_type='batter',
        league_priors=league_priors,
    )


def predict_future_stats_fielding(player_id: str, input_features: List[str], model: ImprovedLSTM,
                                 scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                 position_group: str, seq_length: int = 5, future_years: int = 16,
                                 cutoff_year: Optional[int] = None,
                                 league_priors: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Predict future fielding stats. Wrapper around unified predict_future_stats."""
    # Map position group to model_type for reliability regression
    fielding_model_type = f'defense_{position_group}'
    return predict_future_stats(
        player_id=player_id,
        input_features=input_features,
        model=model,
        scaler=scaler,
        raw_df=raw_df,
        player_names=player_names,
        seq_length=seq_length,
        future_years=future_years,
        extra_fields={'Position_Group': position_group},
        cutoff_year=cutoff_year,
        model_type=fielding_model_type,
        league_priors=league_priors,
    )


def predict_future_stats_baserunning(player_id: str, input_features: List[str], model: ImprovedLSTM,
                                    scaler: Any, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                    seq_length: int = 4, future_years: int = 16, cutoff_year: Optional[int] = None,
                                    league_priors: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Predict future baserunning stats. Wrapper around unified predict_future_stats."""
    return predict_future_stats(
        player_id=player_id,
        input_features=input_features,
        model=model,
        scaler=scaler,
        raw_df=raw_df,
        player_names=player_names,
        seq_length=seq_length,
        future_years=future_years,
        non_negative_features=['SB_rate', 'CS_rate'],
        cutoff_year=cutoff_year,
        model_type='baserunning',
        league_priors=league_priors,
    )


# =============================================================================
# PITCHER PREDICTION (kept separate due to additional complexity)
# =============================================================================

def predict_future_stats_pitcher(player_id: str, input_features: List[str], model, 
                                scaler, raw_df: pd.DataFrame, player_names: pd.DataFrame,
                                role: str, future_years: int = 16, seq_length: int = 4,
                                target_year: int = None,
                                league_priors: Optional[Dict[str, float]] = None) -> List[Dict]:
    """
    Predict future stats for a pitcher with reliability regression.
    
    Key design:
    1. Reliability regression: each season's rate stats are regressed toward the
       player's IP-weighted career mean (or league average for rookies) based on
       sample size and stat-specific stabilization rates. This means a 75 IP season
       keeps its K% mostly intact but its ERA gets pulled toward career norm.
    2. Regressed career mean padding: when the sequence is shorter than seq_length,
       padding uses the player's regressed career average — a principled true-talent
       estimate rather than duplicating the earliest season.
    3. Lower IP threshold for inclusion: since regression handles reliability, we
       can include more seasons (≥20 IP for SP, ≥10 IP for RP) without worrying
       about small-sample noise distorting the sequence.
    
    Args:
        model: Trained ImprovedLSTM model
        scaler: Fitted scaler
        target_year: The year projections should start from (e.g., 2026).
        league_priors: Pre-computed league average priors per feature (from caller).
                      Used as fallback for rookies with <2 career seasons.
    """
    from core.reliability import (
        regress_player_sequence, 
        compute_regressed_career_mean, 
        get_era_for_features,
    )
    
    # Get initial player data
    player_data = raw_df[raw_df['IDfg'] == player_id].sort_values('Season')
    if len(player_data) < 1:
        return []
    
    # Check for required features — skip players with too many NaN values
    required_features = [f for f in input_features if f != 'Age']
    last_valid_season = player_data[player_data['IP'] >= 15].tail(1)
    if last_valid_season.empty:
        last_valid_season = player_data.tail(1)
    
    nan_count = last_valid_season[required_features].isna().sum().sum()
    if nan_count > len(required_features) * 0.3:
        logger.debug(f"Skipping player {player_id} - too many NaN features ({nan_count}/{len(required_features)})")
        return []
        
    # Get player info
    try:
        player_name = player_names[player_names['IDfg'] == player_id]['Name'].iloc[0]
    except IndexError:
        return []
        
    last_season = player_data['Season'].max()
    last_age = player_data[player_data['Season'] == last_season]['Age'].iloc[0]
    
    # Determine the projection start year
    if target_year is not None and last_season < target_year:
        years_missed = target_year - 1 - last_season
        last_age = last_age + years_missed
        last_season = target_year - 1
    
    # Store player context for post-processing
    player_context = {
        'career_high_ip': player_data['IP'].max(),
        'recent_ip': player_data.tail(3)['IP'].mean(),
        'last_fbv': None,
        'last_age': last_age,
        'role': role,
        'recent_surgery': False,
        'recent_performance': {},
    }
    
    # Detect recent surgery/injury
    recent_surgery_detected = False
    if len(player_data) >= 2:
        recent_ip = player_data.tail(2)['IP'].values
        prior_avg = player_data.iloc[:-2]['IP'].mean() if len(player_data) > 2 else player_data.iloc[0]['IP']
        if len(recent_ip) >= 2:
            if (recent_ip[-2] < 50 and prior_avg > 100) or (recent_ip[-1] >= 80 and prior_avg > 130):
                recent_surgery_detected = True
        elif len(recent_ip) > 0 and recent_ip[-1] < 50 and prior_avg > 100:
            recent_surgery_detected = True
    
    player_context['recent_surgery'] = recent_surgery_detected
    
    if recent_surgery_detected:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            most_recent_valid = valid_seasons.iloc[-1]
            player_context['recent_performance'] = {
                'recent_era': most_recent_valid.get('ERA', 4.5),
                'recent_fip': most_recent_valid.get('FIP', 4.5),
                'recent_k_pct': most_recent_valid.get('K%', 0.22),
                'recent_bb_pct': most_recent_valid.get('BB%', 0.09),
            }
    
    if 'FBv' in player_data.columns:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            player_context['last_fbv'] = valid_seasons.iloc[-1]['FBv']
    if 'Stuff+' in player_data.columns:
        valid_seasons = player_data[player_data['IP'] >= 30]
        if not valid_seasons.empty:
            player_context['last_stuff'] = valid_seasons.iloc[-1]['Stuff+']
    
    # =========================================================================
    # QUALIFICATION & IP THRESHOLDS
    # =========================================================================
    # With reliability regression, we can include lower-IP seasons because
    # their noisy rate stats get regressed toward the career mean. The
    # regression itself handles reliability — we no longer need harsh cutoffs.
    #
    # SEQUENCE_THRESHOLD: Minimum IP for inclusion in the input sequence.
    #   Lowered from 50/30 to 20/10 because regression handles sample size.
    #   Still excludes truly meaningless appearances (1-5 IP rehab stints).
    #
    # QUALIFICATION_THRESHOLD: Minimum recent IP to generate predictions at all.
    #   Unchanged — a player needs to have pitched meaningfully recently.
    # =========================================================================
    sequence_ip_threshold = 20 if role == 'SP' else 10
    qualification_ip_threshold = 45 if role == 'SP' else 15
    
    # Check if player qualifies for predictions
    recent_ip = player_data.tail(2)['IP'].max()
    if recent_ip < qualification_ip_threshold:
        logger.debug(f"Skipping {player_name} - insufficient recent IP ({recent_ip:.1f} < {qualification_ip_threshold})")
        return []
    
    # =========================================================================
    # RELIABILITY REGRESSION
    # =========================================================================
    # Regress each season's rate stats toward the player's career mean (or
    # league average for rookies) based on BF and stat-specific stabilization
    # rates. This transforms raw observations into true-talent estimates.
    #
    # Example (200→100→75→200 IP pitcher):
    #   K% (stabilizes at 70 BF): barely changes for any season
    #   ERA (stabilizes at 1320 BF): 75 IP season → heavy regression toward career avg
    # =========================================================================
    era = get_era_for_features(input_features)
    player_data_regressed = regress_player_sequence(
        player_data, input_features, model_type='pitcher', era=era,
        league_priors=league_priors
    )
    
    # Compute regressed career mean for padding
    # This is the best available estimate of the player's true talent level,
    # used when the sequence is shorter than seq_length.
    career_mean = compute_regressed_career_mean(
        player_data, input_features, model_type='pitcher', era=era,
        league_priors=league_priors
    )
    
    # Build sequence from regressed data
    recent_seasons = player_data_regressed.tail(seq_length + 2)  # extra buffer for skipping
    sequence_data = []
    mask = []
    
    for idx, season in recent_seasons.iterrows():
        if season['IP'] >= sequence_ip_threshold:
            base_features = season[input_features].values
            sequence_data.append(base_features)
            mask.append(1)
    
    # Keep only the most recent seq_length valid seasons
    if len(sequence_data) > seq_length:
        sequence_data = sequence_data[-seq_length:]
        mask = mask[-seq_length:]
    
    # Check if we have any valid seasons at all
    if len(sequence_data) == 0:
        logger.debug(f"Skipping pitcher {player_name} - no seasons with IP >= {sequence_ip_threshold}")
        return []
    
    # Pad with regressed career mean if not enough seasons
    if len(sequence_data) < seq_length:
        # Build padding vector from regressed career mean
        padding_vector = np.array(
            [career_mean.get(f, 0.0) for f in input_features], dtype=np.float32
        )
        n_pad = seq_length - len(sequence_data)
        sequence_data = [padding_vector] * n_pad + sequence_data
        mask = [0] * n_pad + mask
    
    # Ensure numeric array
    current_sequence = np.array(sequence_data[-seq_length:], dtype=np.float32)
    mask = np.array(mask[-seq_length:], dtype=np.int64)
    
    # Park factor neutralization: convert stats to park-neutral (true talent)
    # Only applied when ENABLE_PARK_FACTOR_ADJUSTMENT is True in the relevant config
    park_factor_enabled = _is_park_factor_enabled(role)
    
    if park_factor_enabled and 'Team' in player_data.columns:
        from core.park_factors import get_park_factor, EXCLUDED_STATS
        adjustable_indices = [
            i for i, f in enumerate(input_features) if f not in EXCLUDED_STATS
        ]
        # Get the teams for the valid seasons used in the sequence
        valid_seasons_with_team = recent_seasons[recent_seasons['IP'] >= sequence_ip_threshold].tail(seq_length)
        season_teams = valid_seasons_with_team['Team'].tolist() if 'Team' in valid_seasons_with_team.columns else []
        # For padded rows, use the earliest available team
        last_team = player_data['Team'].iloc[-1] if len(player_data) > 0 else None
        n_actual = len(season_teams)
        n_pad = seq_length - n_actual
        all_teams = [last_team] * n_pad + season_teams
        
        for row_idx, team in enumerate(all_teams):
            pf = get_park_factor(team)
            if pf != 1.0:
                for col_idx in adjustable_indices:
                    current_sequence[row_idx, col_idx] = current_sequence[row_idx, col_idx] / pf
    
    # Final NaN/Inf safety check
    if np.isnan(current_sequence).any() or np.isinf(current_sequence).any():
        current_sequence = np.nan_to_num(current_sequence, nan=0.0, posinf=0.0, neginf=0.0)
        logger.warning(f"NaN/Inf cleaned from sequence for {player_name}")
    
    device = next(model.parameters()).device
    predictions_list = []
    
    # Get number of base features (model was trained on base features only)
    n_features = len(input_features)
    
    # Generate predictions
    for year in range(1, future_years + 1):
        # Scale the sequence
        sequence_scaled = scaler.transform(current_sequence)
        
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
        mask_tensor = torch.LongTensor(mask).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(sequence_tensor, mask_tensor.sum(1))
            prediction = prediction.cpu().numpy()
        
        # Inverse transform to get actual values
        prediction_constrained = scaler.inverse_transform(prediction)[0]
        
        # NOTE: Aging curves are applied through the age-adjusted regression
        # prior (in regress_player_sequence / compute_regressed_career_mean),
        # NOT as a post-model adjustment here. This lets the LSTM learn its
        # own aging signal from the Age input feature while the regression
        # prior anchors low-sample seasons to an age-appropriate baseline.
        
        # NOTE: Physical constraints removed for consistency with backtest
        # Previously applied velocity caps, IP limits, and post-surgery constraints
        # These are now handled in post-processing if needed
        
        pred_dict = {
            'Name': player_name,
            'Year': last_season + year,  # Changed from 'Season' to 'Year' for consistency
            'Age': last_age + year,
            'Role': role,
            'IDfg': player_id
        }
        
        # Add predicted stats (except Age)
        for i, feature in enumerate(input_features):
            if feature != 'Age':
                pred_dict[feature] = prediction_constrained[i]
        
        predictions_list.append(pred_dict)
        
        # Update player_context for next year's constraints
        # The constrained velocity becomes the baseline for next year
        if 'FBv' in input_features:
            fbv_idx = input_features.index('FBv')
            player_context['last_fbv'] = prediction_constrained[fbv_idx]
        
        # Update sequence for next prediction - use constrained values for continuity
        next_sequence = prediction_constrained.copy()
        age_index = input_features.index('Age')
        next_sequence[age_index] = last_age + year + 1
        
        # Safety check: ensure no NaN/Inf in prediction before adding to sequence
        if np.isnan(next_sequence).any() or np.isinf(next_sequence).any():
            logger.warning(f"NaN/Inf detected in prediction for {player_name} year {last_season + year} - replacing with last valid values")
            # Use the last row of current_sequence as fallback
            last_valid = current_sequence[-1].copy()
            nan_mask = np.isnan(next_sequence) | np.isinf(next_sequence)
            next_sequence[nan_mask] = last_valid[nan_mask]
            # Update age regardless
            next_sequence[age_index] = last_age + year + 1
        
        # Update sequence
        current_sequence = np.vstack([current_sequence[1:], next_sequence])
        mask = np.ones(seq_length, dtype=np.int64)  # All valid for subsequent predictions
    
    return predictions_list


def predict_all_pitchers(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame, 
    sp_model, 
    rp_model,
    sp_scaler, 
    rp_scaler, 
    sp_input_features: List[str],
    rp_input_features: List[str], 
    seq_length: int, 
    future_years: int = 16, 
    cutoff_year: int = 2024,
    sp_config = None,
    rp_config = None,
    roster_ids: Optional[Set[int]] = None
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all qualified pitchers.
    
    Identifies starting and relief pitchers from the cutoff year (and previous year 
    for returning/injured players), then generates multi-year projections for each.
    
    Args:
        raw_df: Historical pitcher data with Season, IDfg, IP, G, GS columns
        player_names: DataFrame mapping IDfg to Name
        sp_model: Trained LSTM model for starting pitchers
        rp_model: Trained LSTM model for relief pitchers
        sp_scaler: Fitted scaler for SP features
        rp_scaler: Fitted scaler for RP features
        sp_input_features: List of input features for SP model
        rp_input_features: List of input features for RP model
        seq_length: Number of historical seasons used as input sequence
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        roster_ids: Optional set of IDfg values for players on active rosters.
                   Roster pitchers who don't meet normal IP/G thresholds will be
                   recovered from historical data.
        
    Returns:
        DataFrame with predictions for all pitchers, or None if no predictions generated
    """
    logger.info(f"Starting predictions for pitchers from cutoff year {cutoff_year}")
    
    # Pre-compute league average priors for reliability regression.
    # Done once here rather than per-player for efficiency.
    # Used as fallback for rookies with <2 career seasons.
    from core.reliability import compute_league_priors_from_df, get_era_for_features
    sp_era = get_era_for_features(sp_input_features)
    rp_era = get_era_for_features(rp_input_features)
    sp_league_priors = compute_league_priors_from_df(
        raw_df, sp_input_features, model_type='pitcher',
        season=cutoff_year, window=3
    )
    rp_league_priors = compute_league_priors_from_df(
        raw_df, rp_input_features, model_type='pitcher',
        season=cutoff_year, window=3
    )
    logger.info(f"Computed league priors for SP ({len(sp_league_priors)} features) and RP ({len(rp_league_priors)} features)")
    
    # Log park factor adjustment status
    sp_pf = _is_park_factor_enabled('SP')
    rp_pf = _is_park_factor_enabled('RP')
    logger.info(f"Park factor adjustment: SP={'ENABLED' if sp_pf else 'DISABLED'}, RP={'ENABLED' if rp_pf else 'DISABLED'}")
    
    # Get current year and previous year pitchers
    pitchers_current = raw_df[raw_df['Season'] == cutoff_year].copy()
    pitchers_prev = raw_df[raw_df['Season'] == cutoff_year - 1].copy()
    
    # Calculate GS rates
    pitchers_current['GS_rate'] = pitchers_current['GS'] / pitchers_current['G']
    pitchers_prev['GS_rate'] = pitchers_prev['GS'] / pitchers_prev['G']
    
    # Get minimum IP thresholds from config or use defaults
    sp_min_ip = sp_config.MIN_IP_CURRENT if sp_config and hasattr(sp_config, 'MIN_IP_CURRENT') else 25
    rp_min_ip = rp_config.MIN_IP_CURRENT if rp_config and hasattr(rp_config, 'MIN_IP_CURRENT') else 15
    
    # First determine current year roles by GS rate only
    qualified_current_sp = pitchers_current[
        (pitchers_current['IP'] >= sp_min_ip) & 
        (pitchers_current['G'] >= 6)
    ]
    qualified_current_rp = pitchers_current[
        (pitchers_current['IP'] >= rp_min_ip) & 
        (pitchers_current['G'] >= 15)
    ]
    
    # Use current year role if they appear at all
    sp_ids_current = set(qualified_current_sp[qualified_current_sp['GS_rate'] >= 0.8]['IDfg'])
    rp_ids_current = set(qualified_current_rp[qualified_current_rp['GS_rate'] < 0.8]['IDfg'])
    
    # Only look at previous year for players missing from current year
    missing_current = set(pitchers_prev['IDfg']) - set(pitchers_current['IDfg'])
    sp_ids_prev = set(pitchers_prev[
        (pitchers_prev['IDfg'].isin(missing_current)) &
        (pitchers_prev['IP'] >= sp_min_ip) & 
        (pitchers_prev['G'] >= 6) & 
        (pitchers_prev['GS_rate'] >= 0.8)
    ]['IDfg'])
    
    rp_ids_prev = set(pitchers_prev[
        (pitchers_prev['IDfg'].isin(missing_current)) &
        (pitchers_prev['IP'] >= rp_min_ip) & 
        (pitchers_prev['G'] >= 15) & 
        (pitchers_prev['GS_rate'] < 0.8)
    ]['IDfg'])
    
    # Combine IDs
    sp_ids = sp_ids_current.union(sp_ids_prev)
    rp_ids = rp_ids_current.union(rp_ids_prev)
    
    logger.info(f"Found {len(sp_ids_current)} qualified {cutoff_year} SPs and {len(sp_ids_prev)} returning/recovering SPs")
    logger.info(f"Found {len(rp_ids_current)} qualified {cutoff_year} RPs and {len(rp_ids_prev)} returning/recovering RPs")
    
    # =========================================================================
    # ROSTER RECOVERY: recover roster pitchers who didn't meet normal thresholds
    # =========================================================================
    # Catches pitchers who:
    #   - Have cutoff_year data but below IP/G thresholds (e.g. Bieber: 40.1 IP, 4 G)
    #   - Are absent from both cutoff_year AND cutoff_year-1 (e.g. long-term injury)
    # We recover them if they have at least one usable season (IP >= 10) in history.
    if roster_ids is not None:
        all_pitcher_ids = sp_ids | rp_ids
        missing_roster = roster_ids - all_pitcher_ids
        if missing_roster:
            recovered_sp = set()
            recovered_rp = set()
            for pid in missing_roster:
                pitcher_hist = raw_df[
                    (raw_df['IDfg'] == pid) &
                    (raw_df['Season'] <= cutoff_year) &
                    (raw_df['IP'] >= 10)
                ]
                if len(pitcher_hist) == 0:
                    continue
                # Determine role from most recent substantial season
                recent = pitcher_hist.sort_values('Season').iloc[-1]
                gs_rate = recent['GS'] / recent['G'] if recent['G'] > 0 else 0
                if gs_rate >= 0.8:
                    sp_ids.add(pid)
                    recovered_sp.add(pid)
                else:
                    rp_ids.add(pid)
                    recovered_rp.add(pid)
            total_recovered = len(recovered_sp) + len(recovered_rp)
            logger.info(
                f"Roster recovery: {total_recovered} pitchers recovered "
                f"({len(recovered_sp)} SP, {len(recovered_rp)} RP) from historical data "
                f"({len(missing_roster)} roster pitchers were missing)"
            )
    
    # Target year is cutoff_year + 1 (e.g., if cutoff_year=2025, projections start at 2026)
    target_year = cutoff_year + 1
    
    all_predictions = []
    
    # Predict SPs
    logger.info("Generating SP predictions...")
    for player_id in tqdm(sp_ids, desc="Starting Pitchers"):
        # Filter to historical data up to cutoff_year
        player_historical_data = raw_df[
            (raw_df['IDfg'] == player_id) & 
            (raw_df['Season'] <= cutoff_year)
        ].copy()
        
        predictions = predict_future_stats_pitcher(
            player_id=player_id,
            input_features=sp_input_features,
            model=sp_model,
            scaler=sp_scaler,
            raw_df=player_historical_data,
            player_names=player_names,
            role='SP',
            seq_length=seq_length,
            future_years=future_years,
            target_year=target_year,
            league_priors=sp_league_priors
        )
        if predictions:
            all_predictions.extend(predictions)
            
    # Predict RPs
    logger.info("Generating RP predictions...")
    for player_id in tqdm(rp_ids, desc="Relief Pitchers"):
        # Filter to historical data up to cutoff_year
        player_historical_data = raw_df[
            (raw_df['IDfg'] == player_id) & 
            (raw_df['Season'] <= cutoff_year)
        ].copy()
        
        predictions = predict_future_stats_pitcher(
            player_id=player_id,
            input_features=rp_input_features,
            model=rp_model,
            scaler=rp_scaler,
            raw_df=player_historical_data,
            player_names=player_names,
            role='RP',
            seq_length=seq_length,
            future_years=future_years,
            target_year=target_year,
            league_priors=rp_league_priors
        )
        if predictions:
            all_predictions.extend(predictions)
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Sort by Year, Role, and player name
        predictions_df = predictions_df.sort_values(['Year', 'Role', 'Name'], ascending=[True, True, True])
        
        return predictions_df
    else:
        logger.warning("No predictions were generated")
        return None


# =============================================================================
# STATCAST IMPUTATION HELPER
# =============================================================================

def _impute_statcast_from_xwoba(
    df: pd.DataFrame,
    input_features: List[str],
) -> pd.DataFrame:
    """
    Fill missing statcast features using linear regressions on xwOBA.

    Only modifies rows where (a) xwOBA is present and (b) the target statcast
    column is NaN.  Existing values are never overwritten.  The function is a
    no-op when the config toggle ``IMPUTE_STATCAST_FROM_XWOBA`` is ``False``
    or when none of the statcast features appear in *input_features*.

    Returns a **copy** of *df* with imputed values (the original is unchanged).
    """
    try:
        from configs.batter_config import BatterConfig
        if not getattr(BatterConfig, 'IMPUTE_STATCAST_FROM_XWOBA', False):
            return df
        coefficients = getattr(BatterConfig, 'STATCAST_IMPUTATION_COEFFICIENTS', {})
    except (ImportError, AttributeError):
        return df

    if not coefficients:
        return df

    # Only impute features that are actually used by the model
    features_to_impute = [f for f in coefficients if f in input_features and f in df.columns]
    if not features_to_impute or 'xwOBA' not in df.columns:
        return df

    df = df.copy()
    xwoba = df['xwOBA']
    total_imputed = 0

    for feat in features_to_impute:
        coeff = coefficients[feat]
        mask = df[feat].isna() & xwoba.notna()
        n = mask.sum()
        if n > 0:
            df.loc[mask, feat] = coeff['slope'] * xwoba[mask] + coeff['intercept']
            total_imputed += n
            logger.info(
                f"Statcast imputation: filled {n} missing '{feat}' values from xwOBA "
                f"(y = {coeff['slope']:.4f}·xwOBA + {coeff['intercept']:.4f})"
            )

    if total_imputed > 0:
        logger.info(f"Statcast imputation complete: {total_imputed} total values imputed across {len(features_to_impute)} features")
    else:
        logger.debug("Statcast imputation: no missing values needed filling")

    return df


def predict_all_batters(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame,
    model: ImprovedLSTM, 
    scaler: Any, 
    input_features: List[str],
    seq_length: int = 5,
    future_years: int = 16, 
    cutoff_year: int = 2024,
    min_pa_current: int = 100,
    roster_ids: Optional[Set[int]] = None
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all qualified batters.
    
    Identifies batters with sufficient plate appearances in the cutoff year,
    then generates multi-year projections for each.
    
    Note: If BatterConfig.USE_XWOBA_FOR_PREDICTIONS is True and xwOBA data is available,
    the model will receive xwOBA values in place of wOBA for predictions (while still
    being trained on wOBA for broader historical coverage).
    
    Args:
        raw_df: Historical batter data with Season, IDfg, PA columns
        player_names: DataFrame mapping IDfg to Name
        model: Trained LSTM model for batters
        scaler: Fitted scaler for batter features
        input_features: List of input features for the model
        seq_length: Number of historical seasons to use for predictions (from config)
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        min_pa_current: Minimum PA in cutoff year to qualify for predictions
        roster_ids: Optional set of IDfg values for players on active rosters.
                   Players in this set who don't meet normal thresholds will be
                   recovered from historical data if they have any season with PA >= 50.
        
    Returns:
        DataFrame with predictions for all batters, or None if no predictions generated
    """
    # =========================================================================
    # STATCAST IMPUTATION FROM xwOBA (prediction-only)
    # =========================================================================
    # Fill missing statcast features using linear regressions on xwOBA so that
    # players are not excluded from the finetuned model due to absent columns.
    raw_df = _impute_statcast_from_xwoba(raw_df, input_features)

    # Get only current year players (like fielding/baserunning)
    all_players = set(raw_df[
        (raw_df['Season'] == cutoff_year) & 
        (raw_df['PA'] >= min_pa_current)
    ]['IDfg'])
    
    # =========================================================================
    # ROSTER RECOVERY: recover roster players who didn't meet normal thresholds
    # =========================================================================
    # Players on active 40-man rosters should get projections even if they:
    #   - Had a low-PA season in cutoff_year (below min_pa_current)
    #   - Missed cutoff_year entirely (injury, MiLB, etc.)
    # We recover them if they have at least one usable season (PA >= 50) in
    # the historical data. The per-player loop already handles short sequences
    # via padding, and _prepare_player_sequence adjusts age for gap years.
    if roster_ids is not None:
        missing_roster = roster_ids - all_players
        if missing_roster:
            # Find roster players with at least one usable batting season
            historical_batters = set(raw_df[
                (raw_df['IDfg'].isin(missing_roster)) &
                (raw_df['Season'] <= cutoff_year) &
                (raw_df['PA'] >= 50)
            ]['IDfg'].unique())
            recovered = historical_batters
            all_players = all_players | recovered
            logger.info(
                f"Roster recovery: {len(recovered)} batters recovered from historical data "
                f"({len(missing_roster)} roster players were missing, "
                f"{len(missing_roster) - len(recovered)} have no usable batting history)"
            )
    
    # Check if xStat substitutions are enabled
    try:
        from configs.batter_config import BatterConfig
        if BatterConfig.USE_XWOBA_FOR_PREDICTIONS and 'xwOBA' in raw_df.columns:
            xwoba_available_count = raw_df[raw_df['IDfg'].isin(all_players)]['xwOBA'].notna().sum()
            logger.info(f"xwOBA substitution ENABLED - {xwoba_available_count} player-seasons have xwOBA data available")
        if BatterConfig.USE_XBA_FOR_PREDICTIONS and 'xBA' in raw_df.columns:
            xba_available_count = raw_df[raw_df['IDfg'].isin(all_players)]['xBA'].notna().sum()
            logger.info(f"xBA substitution ENABLED - {xba_available_count} player-seasons have xBA data available")
        if BatterConfig.USE_XSLG_FOR_PREDICTIONS and 'xSLG' in raw_df.columns:
            xslg_available_count = raw_df[raw_df['IDfg'].isin(all_players)]['xSLG'].notna().sum()
            logger.info(f"xSLG substitution ENABLED - {xslg_available_count} player-seasons have xSLG data available")
        # Log park factor adjustment status
        if getattr(BatterConfig, 'ENABLE_PARK_FACTOR_ADJUSTMENT', False):
            logger.info("Park factor adjustment ENABLED for batter predictions (neutralize → model → reapply)")
        else:
            logger.info("Park factor adjustment DISABLED for batter predictions")
    except (ImportError, AttributeError):
        pass

    # Pre-compute league priors for reliability regression (if enabled)
    batter_league_priors = None
    try:
        from core.data_processing import _is_reliability_regression_enabled
        if _is_reliability_regression_enabled('batter', context='prediction'):
            from core.reliability import compute_league_priors_from_df
            batter_league_priors = compute_league_priors_from_df(
                raw_df, input_features, model_type='batter',
                season=cutoff_year, window=3
            )
            logger.info(f"Computed league priors for batter regression ({len(batter_league_priors)} features)")
    except ImportError:
        pass

    all_predictions = []
    failed_count = 0
    error_sample = []
    xwoba_substitution_count = 0
    xba_substitution_count = 0
    xslg_substitution_count = 0
    
    for player_id in tqdm(all_players, desc="Generating batter predictions"):
        try:
            # Filter to only historical data up to cutoff_year
            player_data = raw_df[
                (raw_df['IDfg'] == player_id) & 
                (raw_df['Season'] <= cutoff_year)
            ].copy()
            # Additional filtering: Remove seasons with fewer than 50 PA (matching notebook)
            player_data = player_data[player_data['PA'] >= 50].reset_index(drop=True)
            
            if len(player_data) < 1:  # Skip if no valid seasons
                continue
            
            predictions = predict_future_stats_batter(
                player_id=player_id,
                input_features=input_features,
                model=model,
                scaler=scaler,
                raw_df=player_data,  # Use filtered data (up to cutoff_year)
                player_names=player_names,
                seq_length=seq_length,
                future_years=future_years,
                cutoff_year=cutoff_year,  # Ensure predictions start from cutoff_year + 1
                league_priors=batter_league_priors,
            )
            
            if predictions:
                all_predictions.extend(predictions)
                
        except Exception as e:
            failed_count += 1
            # Collect first 5 errors for detailed logging
            if len(error_sample) < 5:
                error_sample.append((player_id, str(e)))
            continue
    
    # Log xStat substitution summary
    total_players = len(all_players)
    if xwoba_substitution_count > 0:
        logger.info(f"✓ xwOBA substitution applied to {xwoba_substitution_count}/{total_players} players ({xwoba_substitution_count/total_players*100:.1f}%)")
    if xba_substitution_count > 0:
        logger.info(f"✓ xBA substitution applied to {xba_substitution_count}/{total_players} players ({xba_substitution_count/total_players*100:.1f}%)")
    if xslg_substitution_count > 0:
        logger.info(f"✓ xSLG substitution applied to {xslg_substitution_count}/{total_players} players ({xslg_substitution_count/total_players*100:.1f}%)")
    if xwoba_substitution_count == 0 and xba_substitution_count == 0 and xslg_substitution_count == 0:
        logger.info("No xStat substitutions applied (all toggles may be False or xStat data not available)")
    
    # Log error details if any failures occurred
    if failed_count > 0:
        logger.error(f"Failed to generate predictions for {failed_count} batters")
        if error_sample:
            logger.error("Sample errors (first 5):")
            for player_id, error in error_sample:
                logger.error(f"  Player {player_id}: {error}")
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df = predictions_df.sort_values(['Year', 'Name'])
        return predictions_df
    else:
        logger.warning("No batter predictions were generated")
        return None


def predict_all_fielders(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame,
    position_models: Dict[str, ImprovedLSTM], 
    position_scalers: Dict[str, Any],
    position_group_map: Dict[str, str],
    input_features_map: Dict[str, List[str]],
    seq_length_map: Dict[str, int],
    future_years: int = 16, 
    cutoff_year: int = 2025,
    use_aging_enforcer: bool = False,
    roster_ids: Optional[Set[int]] = None
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all qualified fielders.
    
    Processes each position group (infield, outfield, catcher) separately using
    position-specific models, then combines results.
    
    Args:
        raw_df: Historical fielding data with Season, IDfg, Pos, Inn columns
        player_names: DataFrame mapping IDfg to Name
        position_models: Dict mapping position group to trained model
        position_scalers: Dict mapping position group to fitted scaler
        position_group_map: Dict mapping specific positions to position groups
        input_features_map: Dict mapping position group to input features
        seq_length_map: Dict mapping position group to sequence length
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        roster_ids: Optional set of IDfg values for players on active rosters.
                   Roster fielders who don't meet the 50-inning threshold will be
                   recovered from historical data.
        
    Returns:
    """
    # Define minimum innings threshold (matches notebook MIN_POSITION_INNINGS)
    MIN_POSITION_INNINGS = 50
    
    all_predictions = []
    
    # Pre-compute league priors for each position group's features
    fielding_league_priors = {}
    try:
        from core.data_processing import _is_reliability_regression_enabled
        from core.reliability import compute_league_priors_from_df
        for model_key in input_features_map:
            defense_type = f'defense_{model_key}'
            if _is_reliability_regression_enabled(defense_type, context='prediction'):
                fielding_league_priors[model_key] = compute_league_priors_from_df(
                    raw_df, input_features_map[model_key],
                    model_type=defense_type,
                    season=cutoff_year, window=3
                )
                logger.info(f"Computed league priors for {defense_type} regression ({len(fielding_league_priors[model_key])} features)")
    except ImportError:
        pass
    
    # Process each position group (like the notebook does)
    for model_key, model in position_models.items():
        scaler = position_scalers[model_key]
        input_features = input_features_map[model_key]
        seq_length = seq_length_map[model_key]
        group_league_priors = fielding_league_priors.get(model_key)
        
        # Get valid positions for this position group
        if model_key == 'infield':
            valid_positions = ['1B', '2B', '3B', 'SS']
        elif model_key == 'outfield':
            valid_positions = ['LF', 'CF', 'RF']
        elif model_key == 'catcher':
            valid_positions = ['C']
        else:
            continue
        
        # Filter data for this position group (matches notebook logic)
        group_df = raw_df[raw_df['Pos'].isin(valid_positions)].copy()
        
        # Get players who played enough innings at any valid position in cutoff year
        # For each player, find their PRIMARY position (most innings played)
        players_current_all = group_df[
            (group_df['Season'] == cutoff_year) & 
            (group_df['Inn'] >= MIN_POSITION_INNINGS) &
            (group_df['Pos'].isin(valid_positions))
        ][['IDfg', 'Pos', 'Inn']].copy()
        
        # Group by player and find position with most innings
        primary_positions = players_current_all.groupby('IDfg').apply(
            lambda x: x.loc[x['Inn'].idxmax(), 'Pos']
        ).reset_index()
        primary_positions.columns = ['IDfg', 'Primary_Pos']
        
        logger.info(f"\nProcessing {model_key} - {len(primary_positions)} players with primary positions")
        
        # =================================================================
        # ROSTER RECOVERY for this position group
        # =================================================================
        # Recover roster players who have historical data at valid positions
        # in this group but didn't qualify in cutoff_year (low innings or absent).
        if roster_ids is not None:
            current_qualified_ids = set(primary_positions['IDfg']) if len(primary_positions) > 0 else set()
            missing_roster = roster_ids - current_qualified_ids
            if missing_roster:
                # Find roster players with at least one usable season at this position group
                recovery_candidates = group_df[
                    (group_df['IDfg'].isin(missing_roster)) &
                    (group_df['Season'] <= cutoff_year) &
                    (group_df['Inn'] >= MIN_POSITION_INNINGS) &
                    (group_df['Pos'].isin(valid_positions))
                ]
                if not recovery_candidates.empty:
                    # For each recovered player, find their primary position (most innings in most recent qualifying season)
                    recovery_primary = recovery_candidates.sort_values(['Season', 'Inn']).groupby('IDfg').apply(
                        lambda x: x.iloc[-1]['Pos']
                    ).reset_index()
                    recovery_primary.columns = ['IDfg', 'Primary_Pos']
                    # Remove any that are already in primary_positions
                    recovery_primary = recovery_primary[~recovery_primary['IDfg'].isin(current_qualified_ids)]
                    if len(recovery_primary) > 0:
                        primary_positions = pd.concat([primary_positions, recovery_primary], ignore_index=True)
                        logger.info(f"  Roster recovery: {len(recovery_primary)} {model_key} fielders recovered from historical data")
        
        # Generate predictions for each player at their PRIMARY position only
        for _, row in tqdm(primary_positions.iterrows(), desc=f"{model_key} predictions"):
            try:
                player_id = row['IDfg']
                primary_position = row['Primary_Pos']  # The position they played most
                
                # Filter to historical data up to cutoff_year, PRIMARY POSITION ONLY
                player_historical_data = group_df[
                    (group_df['IDfg'] == player_id) & 
                    (group_df['Season'] <= cutoff_year) &
                    (group_df['Pos'] == primary_position)  # Only this position's data
                ].copy()
                
                # Skip if not enough historical data at this position
                if len(player_historical_data) == 0:
                    continue

                # Drop seasons with NaN in the input features (e.g. Inn < 100 seasons
                # that were excluded by calculate_rate_stats min_denominator filter).
                # This mirrors what filter_data does during training so the model
                # only sees reliable, full-sample defensive observations.
                player_historical_data = player_historical_data.dropna(
                    subset=[f for f in input_features if f in player_historical_data.columns]
                )
                if len(player_historical_data) == 0:
                    continue
                
                predictions = predict_future_stats_fielding(
                    player_id=player_id,
                    input_features=input_features,
                    model=model,
                    scaler=scaler,
                    raw_df=player_historical_data,
                    player_names=player_names,
                    position_group=model_key,
                    seq_length=seq_length,
                    future_years=future_years,
                    cutoff_year=cutoff_year,  # Ensure predictions start from cutoff_year + 1
                    league_priors=group_league_priors,
                )
                
                if predictions:
                    # Add the primary position to each prediction
                    for pred in predictions:
                        pred['Pos'] = primary_position  # Use their primary position
                        pred['Position_Group'] = model_key
                    
                    all_predictions.extend(predictions)
                    
            except Exception as e:
                logger.error(f"Error predicting for fielder {player_id} at {specific_position}: {str(e)}")
                continue
    
    if all_predictions:
        # Apply aging enforcement to prevent unrealistic improvements for players 30+
        enforcer = get_aging_enforcer()
        
        # Group predictions by player and position group for enforcement
        from collections import defaultdict
        player_predictions = defaultdict(list)
        for pred in all_predictions:
            key = (pred['IDfg'], pred.get('Position_Group', 'unknown'))
            player_predictions[key].append(pred)
        
        # Enforce aging for each player
        enforced_predictions = []
        for (player_id, pos_group), preds in player_predictions.items():
            # Sort by year
            preds_sorted = sorted(preds, key=lambda x: x['Year'])
            
            # Map position group to category
            if pos_group == 'infield':
                category = 'fielding_infield'
                valid_positions = ['1B', '2B', '3B', 'SS']
            elif pos_group == 'outfield':
                category = 'fielding_outfield'
                valid_positions = ['LF', 'CF', 'RF']
            elif pos_group == 'catcher':
                category = 'fielding_catcher'
                valid_positions = ['C']
            else:
                category = 'fielding_outfield'
                valid_positions = ['LF', 'CF', 'RF']
            
            # Get the player's last actual season as baseline for first prediction
            # This is critical: compare 2025 prediction to 2024 actual, not just 2025 to 2026
            # Use only the position they're being predicted for
            first_pred = preds_sorted[0] if preds_sorted else None
            if first_pred and 'Pos' in first_pred:
                player_position = first_pred['Pos']
                player_last_actual = raw_df[
                    (raw_df['IDfg'] == player_id) & 
                    (raw_df['Season'] == cutoff_year) &
                    (raw_df['Pos'] == player_position)  # Only their primary position
                ]
            else:
                player_last_actual = raw_df[
                    (raw_df['IDfg'] == player_id) & 
                    (raw_df['Season'] == cutoff_year) &
                    (raw_df['Pos'].isin(valid_positions))
                ]
            
            if not player_last_actual.empty:
                # Create a baseline record from actual data
                actual_row = player_last_actual.iloc[0]
                baseline = {
                    'IDfg': player_id,
                    'Year': cutoff_year,  # e.g., 2024
                    'Age': actual_row.get('Age', 0),
                    'Name': preds_sorted[0].get('Name', ''),
                    'Position_Group': pos_group,
                    '_is_baseline': True  # Mark so we can remove later
                }
                # Copy defensive metrics from actual data
                for metric in enforcer.DEFENSE_METRICS:
                    if metric in actual_row.index:
                        baseline[metric] = actual_row[metric]
                
                # Prepend baseline so first prediction is compared against actual
                preds_with_baseline = [baseline] + preds_sorted
            else:
                preds_with_baseline = preds_sorted
            
            # Conditionally enforce aging constraints
            if use_aging_enforcer:
                adjusted = enforcer.enforce_aging(preds_with_baseline, category)
                # Remove baseline row (it was just for comparison)
                adjusted = [p for p in adjusted if not p.get('_is_baseline', False)]
                enforced_predictions.extend(adjusted)
            else:
                # Skip aging enforcement - use raw predictions
                # Still remove baseline row if it exists
                preds_no_baseline = [p for p in preds_with_baseline if not p.get('_is_baseline', False)]
                enforced_predictions.extend(preds_no_baseline)
        
        predictions_df = pd.DataFrame(enforced_predictions)
        
        # Clean up the _is_baseline column if it somehow persisted
        if '_is_baseline' in predictions_df.columns:
            predictions_df = predictions_df.drop(columns=['_is_baseline'])
        
        # Sort by Name, Position, and Year (matches notebook output)
        predictions_df = predictions_df.sort_values(['Name', 'Pos', 'Year'])
        
        return predictions_df
    else:
        logger.warning("No fielding predictions were generated")
        return None


def predict_all_baserunners(
    raw_df: pd.DataFrame, 
    player_names: pd.DataFrame,
    model, 
    scaler, 
    input_features: List[str],
    seq_length: int = 4, 
    future_years: int = 16, 
    cutoff_year: int = 2024,
    roster_ids: Optional[Set[int]] = None
) -> Optional[pd.DataFrame]:
    """
    Generate future predictions for all baserunners.
    
    Identifies players from the cutoff year and generates multi-year
    baserunning projections (stolen bases, caught stealing, baserunning runs).
    
    Args:
        raw_df: Historical baserunning data with Season, IDfg columns
        player_names: DataFrame mapping IDfg to Name
        model: Trained LSTM model for baserunning
        scaler: Fitted scaler for baserunning features
        input_features: List of input features for the model
        seq_length: Number of historical seasons used as input sequence
        future_years: Number of years to project into the future
        cutoff_year: Last year of actual data (predictions start from cutoff_year + 1)
        roster_ids: Optional set of IDfg values for players on active rosters.
                   Roster players absent from cutoff_year will be recovered from
                   historical baserunning data.
        
    Returns:
        DataFrame with predictions for all baserunners, or None if no predictions generated
    """
    # Get unique players from cutoff year data
    current_players = set(raw_df[raw_df['Season'] == cutoff_year]['IDfg'].unique())
    
    # =========================================================================
    # ROSTER RECOVERY: recover roster players absent from cutoff_year
    # =========================================================================
    if roster_ids is not None:
        missing_roster = roster_ids - current_players
        if missing_roster:
            # Find roster players with any historical baserunning data
            historical_baserunners = set(raw_df[
                (raw_df['IDfg'].isin(missing_roster)) &
                (raw_df['Season'] <= cutoff_year)
            ]['IDfg'].unique())
            current_players = current_players | historical_baserunners
            logger.info(
                f"Roster recovery: {len(historical_baserunners)} baserunners recovered from historical data "
                f"({len(missing_roster)} roster players were missing)"
            )
    
    all_predictions = []
    
    logger.info(f"Found {len(current_players)} unique players for baserunning predictions")
    
    # Pre-compute league priors for reliability regression (if enabled)
    baserunning_league_priors = None
    try:
        from core.data_processing import _is_reliability_regression_enabled
        if _is_reliability_regression_enabled('baserunning', context='prediction'):
            from core.reliability import compute_league_priors_from_df
            baserunning_league_priors = compute_league_priors_from_df(
                raw_df, input_features, model_type='baserunning',
                season=cutoff_year, window=3
            )
            logger.info(f"Computed league priors for baserunning regression ({len(baserunning_league_priors)} features)")
    except ImportError:
        pass
    
    for player_id in tqdm(current_players, desc="Generating baserunning predictions"):
        try:
            # Filter to historical data up to cutoff_year
            player_historical_data = raw_df[
                (raw_df['IDfg'] == player_id) & 
                (raw_df['Season'] <= cutoff_year)
            ].copy()
            
            predictions = predict_future_stats_baserunning(
                player_id=player_id,
                input_features=input_features,
                model=model,
                scaler=scaler,
                raw_df=player_historical_data,
                player_names=player_names,
                seq_length=seq_length,
                future_years=future_years,
                cutoff_year=cutoff_year,  # Ensure predictions start from cutoff_year + 1
                league_priors=baserunning_league_priors,
            )
            
            if predictions:
                all_predictions.extend(predictions)
            else:
                # Log why no predictions were generated for the first few players
                if len(all_predictions) == 0 and len(current_players) < 10:
                    logger.warning(f"No predictions for player {player_id}")
                
        except Exception as e:
            logger.error(f"Error predicting for baserunner {player_id}: {str(e)}")
            continue
    
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Sort by Year and SB_rate (descending)
        predictions_df = predictions_df.sort_values(['Year', 'SB_rate'], ascending=[True, False])
        
        return predictions_df
    else:
        logger.warning("No baserunning predictions were generated")
        return None


