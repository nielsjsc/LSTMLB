#!/usr/bin/env python3
"""
MLB Prediction Pipeline - Unified Interface
============================================

Single script to handle all pipeline operations:
- Train models (with optional hyperparameter customization)
- Generate predictions
- Run projection engine (playing time + WAR calculation)
- Run full end-to-end pipeline

Author: Niels Christoffersen
Date: December 2025
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPTS_DIR = Path(__file__).parent  # auto_train_models/scripts/
AUTO_TRAIN_DIR = SCRIPTS_DIR.parent  # auto_train_models/ (run commands from here)
DATA_DIR = AUTO_TRAIN_DIR.parent / 'data'  # LSTMLB/data/
GENERATED_DIR = DATA_DIR / 'generated'
PIPELINE_DIR = GENERATED_DIR / 'pipeline'

# Python executable - use the same interpreter running this script
PYTHON_EXE = sys.executable

# Ensure directories exist
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_TYPES = {
    'batter_pretrain': {
        'training_command': 'python scripts/train_models.py --model batter',
        'prediction_command': 'python scripts/predict_models.py --model-type batter --use-pretrained',
        'output_file': 'batter_predictions.csv',
        'description': 'Batter PRE-TRAINING (classical features, 1950-2024)'
    },
    'batter_finetune': {
        'training_command': 'python scripts/train_models.py --model batter --finetune --from-checkpoint checkpoints/batter_pretrained.pth',
        'prediction_command': 'python scripts/predict_models.py --model-type batter',
        'output_file': 'batter_predictions.csv',
        'description': 'Batter FINE-TUNING (adds Statcast features, 2015+) - Requires pre-trained checkpoint'
    },
    # =========================================================================
    # STARTING PITCHER MODELS
    # =========================================================================
    # NOTE: When UNIFIED_PITCHER_MODEL=True in PitcherSPConfig, the SP pretrain
    # step trains on ALL pitcher data (SP+RP combined) and prediction uses the
    # single model for every pitcher.  The RP pretrain/finetune steps are
    # unnecessary in that mode.
    'pitcher_sp_pretrain': {
        'training_command': 'python scripts/train_models.py --model pitcher_sp --pretrain',
        'prediction_command': 'python scripts/predict_models.py --model-type pitcher --use-pretrained',
        'output_file': 'pitcher_predictions.csv',
        'description': 'SP PRE-TRAINING (classical features: K%, BB%, FIP, ERA, 1950-2024)'
    },
    'pitcher_sp_finetune': {
        # NOTE: --freeze-attention is critical - only ~360 training sequences vs 5M+ params
        # With --freeze-attention: ~1.08M trainable params (input_proj + output_proj only)
        # Without: ~5.3M trainable params (causes severe overfitting)
        'training_command': 'python scripts/train_models.py --model pitcher_sp --finetune --from-checkpoint checkpoints/sp/pitcher_sp_pretrained.pth --freeze-lstm --freeze-attention',
        'prediction_command': 'python scripts/predict_models.py --model-type pitcher',
        'output_file': 'pitcher_predictions.csv',
        'description': 'SP FINE-TUNING (adds Stuff+, Location+, Pitching+, xERA, 2020+) - Requires pre-trained checkpoint'
    },
    # =========================================================================
    # RELIEF PITCHER MODELS
    # =========================================================================
    'pitcher_rp_pretrain': {
        'training_command': 'python scripts/train_models.py --model pitcher_rp --pretrain',
        'prediction_command': None,  # Handled by pitcher_sp prediction
        'output_file': None,
        'description': 'RP PRE-TRAINING (classical features: K%, BB%, FIP, ERA, 1950-2024)'
    },
    'pitcher_rp_finetune': {
        # NOTE: Same rationale as SP - limited data requires aggressive freezing
        'training_command': 'python scripts/train_models.py --model pitcher_rp --finetune --from-checkpoint checkpoints/rp/pitcher_rp_pretrained.pth --freeze-lstm --freeze-attention',
        'prediction_command': None,  # Handled by pitcher_sp prediction
        'output_file': None,
        'description': 'RP FINE-TUNING (adds Stuff+, Location+, Pitching+, xERA, 2020+) - Requires pre-trained checkpoint'
    },
    'baserunning': {
        'training_command': 'python scripts/train_models.py --model baserunning',
        'prediction_command': 'python scripts/predict_models.py --model-type baserunning',
        'output_file': 'baserunning_predictions.csv',
        'description': 'Baserunning value (BsR, wSB, UBR, wGDP)'
    },
    'defense_infield': {
        'training_command': 'python scripts/train_models.py --model defense_infield',
        'prediction_command': 'python scripts/predict_models.py --model-type fielding',
        'output_file': 'fielding_predictions.csv',
        'description': 'Infield defense metrics'
    },
    'defense_outfield': {
        'training_command': 'python scripts/train_models.py --model defense_outfield',
        'prediction_command': 'python scripts/predict_models.py --model-type fielding',
        'output_file': 'fielding_predictions.csv',
        'description': 'Outfield defense metrics'
    },
    'defense_catcher': {
        'training_command': 'python scripts/train_models.py --model defense_catcher',
        'prediction_command': 'python scripts/predict_models.py --model-type fielding',
        'output_file': 'fielding_predictions.csv',
        'description': 'Catcher defense metrics'
    },
    'playing_time': {
        'training_command': None,  # No training required
        'prediction_command': 'python -m playing_time.main',
        'output_file': 'playing_time_2026.csv',
        'description': 'Playing time allocation based on WAR projections and injuries'
    }
}


def print_header():
    """Print pipeline header"""
    print("\n" + "=" * 70)
    print("MLB PREDICTION PIPELINE")
    print("=" * 70)
    print()


def print_menu():
    """Display main menu"""
    print("\nMAIN MENU:")
    print("1. Train Models")
    print("2. Generate Predictions")
    print("3. Run Projection Engine (Playing Time + WAR)")
    print("4. Calculate Trade Values (Surplus Value)")
    print("5. Run Full Pipeline (Train → Predict → Project → Values)")
    print("6. Exit")
    print()



def get_user_choice(prompt: str, valid_choices: List[str]) -> str:
    """Get validated user input"""
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def get_projection_year() -> int:
    """Get projection year from user"""
    print("\nEnter projection year (default: 2026):")
    year_input = input("> ").strip()
    if year_input and year_input.isdigit():
        return int(year_input)
    return 2026


def select_models_for_training() -> List[str]:
    """Let user select which models to train"""
    selected_models = []
    
    # Category menu
    print(f"\nSelect model categories to train:")
    print("1. Batting")
    print("2. Pitching")
    print("3. Baserunning")
    print("4. Fielding")
    print("0. All models")
    
    print("\nEnter numbers separated by commas (e.g., 1,2,3) or 0 for all:")
    choice = input("> ").strip()
    
    if choice == '0':
        # All models - use pretrain + finetune for batting/pitching
        return ['batter_pretrain', 'batter_finetune', 
                'pitcher_sp_pretrain', 'pitcher_sp_finetune',
                'pitcher_rp_pretrain', 'pitcher_rp_finetune',
                'baserunning', 'defense_infield', 'defense_outfield', 'defense_catcher']
    
    categories = [x.strip() for x in choice.split(',')]
    
    for cat in categories:
        if cat == '1':  # Batting
            print("\nBatting model options:")
            print("1. Pre-train (classical features, 1950-2024)")
            print("2. Fine-tune (adds Statcast features, 2015+)")
            print("3. Both (pre-train then fine-tune)")
            bat_choice = input("> ").strip()
            if bat_choice == '1':
                selected_models.append('batter_pretrain')
            elif bat_choice == '2':
                selected_models.append('batter_finetune')
            elif bat_choice == '3':
                selected_models.extend(['batter_pretrain', 'batter_finetune'])
            else:
                print("Invalid choice, skipping batting")
                
        elif cat == '2':  # Pitching
            print("\nPitching model options:")
            print("1. Pre-train (classical features, 1950-2024)")
            print("2. Fine-tune (adds Stuff+, Location+, Pitching+, 2020+)")
            print("3. Both (pre-train then fine-tune)")
            pitch_choice = input("> ").strip()
            if pitch_choice == '1':
                selected_models.extend(['pitcher_sp_pretrain', 'pitcher_rp_pretrain'])
            elif pitch_choice == '2':
                selected_models.extend(['pitcher_sp_finetune', 'pitcher_rp_finetune'])
            elif pitch_choice == '3':
                selected_models.extend(['pitcher_sp_pretrain', 'pitcher_sp_finetune',
                                       'pitcher_rp_pretrain', 'pitcher_rp_finetune'])
            else:
                print("Invalid choice, skipping pitching")
                
        elif cat == '3':  # Baserunning
            selected_models.append('baserunning')
            
        elif cat == '4':  # Fielding
            print("\nFielding position options:")
            print("1. Infield")
            print("2. Outfield")
            print("3. Catcher")
            print("4. All positions")
            field_choice = input("> ").strip()
            if field_choice == '1':
                selected_models.append('defense_infield')
            elif field_choice == '2':
                selected_models.append('defense_outfield')
            elif field_choice == '3':
                selected_models.append('defense_catcher')
            elif field_choice == '4':
                selected_models.extend(['defense_infield', 'defense_outfield', 'defense_catcher'])
            else:
                print("Invalid choice, skipping fielding")
    
    if not selected_models:
        print("No valid models selected, using all models")
        return ['batter_pretrain', 'batter_finetune', 
                'pitcher_sp_pretrain', 'pitcher_sp_finetune',
                'pitcher_rp_pretrain', 'pitcher_rp_finetune',
                'baserunning', 'defense_infield', 'defense_outfield', 'defense_catcher']
    
    return selected_models


def select_models_for_prediction() -> List[str]:
    """Let user select which prediction groups to generate"""
    print(f"\nSelect predictions to generate:")
    print("0. All predictions (finetuned models)")
    print("1. Batter predictions (Finetuned - Classical + Statcast)")
    print("2. Batter predictions (Pretrained - Classical only)")
    print("3. Pitcher predictions (Finetuned SP + RP - Stuff+, Location+, Pitching+)")
    print("4. Pitcher predictions (Pretrained SP + RP - Classical only)")
    print("5. Baserunning predictions")
    print("6. Fielding predictions (All positions)")
    
    print("\nEnter numbers separated by commas (e.g., 1,3,5) or 0 for all:")
    choice = input("> ").strip()
    
    # Map user choices to model keys
    prediction_groups = {
        '1': ['batter_finetune'],  # Finetuned model (recommended)
        '2': ['batter_pretrain'],  # Pretrained model (classical only)
        '3': ['pitcher_sp_finetune'],  # Finetuned pitcher model with Statcast
        '4': ['pitcher_sp_pretrain'],  # Pretrained pitcher model (classical only)
        '5': ['baserunning'],
        '6': ['defense_infield']  # Running any defense generates all 3 positions
    }
    
    if choice == '0':
        # All predictions - use finetuned by default
        return ['batter_finetune', 'pitcher_sp_finetune', 'baserunning', 'defense_infield']
    
    try:
        selections = [x.strip() for x in choice.split(',')]
        selected_models = []
        for sel in selections:
            if sel in prediction_groups:
                selected_models.extend(prediction_groups[sel])
        
        if selected_models:
            # Remove duplicates while preserving order
            return list(dict.fromkeys(selected_models))
    except (ValueError, IndexError):
        pass
    
    print("Invalid selection, using all predictions")
    return ['batter_finetune', 'pitcher_sp_finetune', 'baserunning', 'defense_infield']


def get_hyperparameter_overrides() -> Dict[str, Any]:
    """Ask user if they want to customize hyperparameters"""
    print("\nWould you like to customize hyperparameters? (y/n)")
    if input("> ").strip().lower() != 'y':
        return {}
    
    overrides = {}
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER CUSTOMIZATION")
    print("=" * 60)
    
    # Epochs
    print("\n1. EPOCHS")
    print("   Number of training epochs (default: 50)")
    print("   Press Enter to skip:")
    epochs = input("   > ").strip()
    if epochs and epochs.isdigit():
        overrides['epochs'] = int(epochs)
        print(f"   Set epochs to {overrides['epochs']}")
    
    # Loss function selection
    print("\n2. LOSS FUNCTION")
    print("   Choose a loss function for training:")
    print("   ")
    print("   [1] MSE Loss (DEFAULT - Simple mean squared error)")
    print("       → Standard regression loss, no special weighting")
    print("       → Fast, stable, works well in most cases")
    print("   [2] Weighted Loss (Model-specific sample/feature weighting)")
    print("       → Batters: Weights by PA, separate rate/counting stats")
    print("       → Pitchers: Weights by IP for sample importance")
    print("       → Fielding: Weights by innings, position-specific")
    print("   [3] Empirical Aging Loss (Aging-constrained with data-driven params)")
    print("       → Uses historical aging curves from MLB data")
    print("       → Penalizes unrealistic late-career improvements")
    print("       → Addresses survivorship bias in training data")
    print("   ")
    print("   Enter choice (1/2/3, default: 1):")
    loss_choice = input("   > ").strip()
    
    if loss_choice == '1' or not loss_choice:
        overrides['loss_function'] = 'mse'
        print("   Using MSE Loss (default)")
        
    elif loss_choice == '2':
        overrides['loss_function'] = 'weighted'
        print("   Using model-specific weighted loss")
        
    elif loss_choice == '3':
        overrides['loss_function'] = 'empirical'
        
        # Empirical loss strength
        print("\n   EMPIRICAL LOSS STRENGTH:")
        print("   ┌─────────────┬─────────────┬──────────────────────────────────────┐")
        print("   │ Choice      │ Aging Weight│ Description                          │")
        print("   ├─────────────┼─────────────┼──────────────────────────────────────┤")
        print("   │ none        │ 0.00        │ Pure MSE (no aging constraint)       │")
        print("   │ light       │ 0.05        │ Light constraint, mostly MSE         │")
        print("   │ moderate    │ 0.15        │ Balanced (RECOMMENDED)               │")
        print("   │ strong      │ 0.30        │ Prioritize aging plausibility        │")
        print("   │ aggressive  │ 0.50        │ Very strict (for fielding/baserun)   │")
        print("   └─────────────┴─────────────┴──────────────────────────────────────┘")
        print("   Enter choice (default: moderate):")
        strength = input("   > ").strip().lower()
        if strength in ['none', 'light', 'moderate', 'strong', 'aggressive']:
            overrides['empirical_strength'] = strength
        else:
            overrides['empirical_strength'] = 'moderate'
        print(f"   Using empirical aging loss with '{overrides['empirical_strength']}' strength")
    
    else:
        overrides['loss_function'] = 'mse'
        print("   Invalid choice, using MSE Loss (default)")
    
    # Aging enforcer for predictions
    print("\n3. AGING ENFORCER (Fielding Predictions)")
    print("   Apply post-prediction constraints to prevent unrealistic late-career improvements")
    print("   in defensive metrics.")
    print("   ")
    print("   [1] Off (default) - Use raw model predictions")
    print("   [2] On - Apply aging constraints post-prediction")
    print("   ")
    print("   Note: This is separate from empirical aging loss during training.")
    print("   Empirical loss guides training; aging enforcer clips predictions.")
    print("   ")
    print("   Enter choice (1/2, default: 1):")
    aging_choice = input("   > ").strip()
    
    if aging_choice == '2':
        overrides['use_aging_enforcer'] = True
        print("   Aging enforcer enabled for fielding predictions")
    else:
        overrides['use_aging_enforcer'] = False
        print("   Aging enforcer disabled (using raw predictions)")
    
    print("\n" + "=" * 60)
    return overrides


def run_command(command: str, description: str, timeout: int = 7200) -> bool:
    """Execute a command with progress tracking.
    
    Uses the same Python interpreter that's running this script to ensure
    subprocesses use the correct virtual environment.
    """
    # Replace 'python' with the actual interpreter path
    if command.startswith('python '):
        command = f'"{PYTHON_EXE}" ' + command[7:]
    
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    start_time = datetime.now()
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(AUTO_TRAIN_DIR)  # Run from auto_train_models/ so imports work
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait(timeout=timeout)
        
        duration = datetime.now() - start_time
        
        if process.returncode == 0:
            logger.info(f"SUCCESS: {description} completed in {duration}")
            return True
        else:
            logger.error(f"FAILED: {description} failed with code {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: {description} timed out after {timeout}s")
        process.kill()
        return False
    except Exception as e:
        logger.error(f"ERROR: {description} failed: {str(e)}")
        return False


def train_models(selected_models: List[str], hyperparameter_overrides: Dict[str, Any]) -> bool:
    """Train selected models"""
    print(f"\nTraining {len(selected_models)} model(s)...")
    
    # Show loss function info
    loss_func = hyperparameter_overrides.get('loss_function', 'mse')
    if loss_func == 'mse':
        print(f"Using MSE LOSS (default - simple mean squared error)")
    elif loss_func == 'weighted':
        print(f"Using WEIGHTED LOSS (model-specific sample/feature weighting)")
    elif loss_func == 'empirical':
        strength = hyperparameter_overrides.get('empirical_strength', 'moderate')
        print(f"Using EMPIRICAL AGING LOSS with '{strength}' strength")
        print("   (Data-derived aging parameters from aging_parameters_v2.json)")
    
    success_count = 0
    failed_models = []
    
    for model_key in selected_models:
        info = MODEL_TYPES[model_key]
        command = info['training_command']
        
        # Apply hyperparameter overrides
        if 'epochs' in hyperparameter_overrides:
            command += f" --epochs {hyperparameter_overrides['epochs']}"
        
        # Apply loss function selection
        if 'loss_function' in hyperparameter_overrides:
            command += f" --loss-function {hyperparameter_overrides['loss_function']}"
            
            # Add empirical loss parameters if using empirical loss
            if hyperparameter_overrides['loss_function'] == 'empirical':
                if 'empirical_strength' in hyperparameter_overrides:
                    command += f" --empirical-strength {hyperparameter_overrides['empirical_strength']}"
        
        description = f"Training {model_key}"
        
        if run_command(command, description, timeout=7200):
            success_count += 1
        else:
            failed_models.append(model_key)
    
    print(f"\n{'='*70}")
    print(f"Training Summary: {success_count}/{len(selected_models)} successful")
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    print(f"{'='*70}\n")
    
    return len(failed_models) == 0


def generate_predictions(selected_models: List[str], use_aging_enforcer: bool = False) -> bool:
    """Generate predictions for selected models
    
    Args:
        selected_models: List of model keys to generate predictions for
        use_aging_enforcer: Whether to apply aging constraints to fielding predictions
    """
    print(f"\nGenerating predictions for {len(selected_models)} model(s)...")
    
    # Group models by prediction command
    prediction_commands = {}
    for model_key in selected_models:
        info = MODEL_TYPES[model_key]
        cmd = info['prediction_command']
        if cmd:
            if cmd not in prediction_commands:
                prediction_commands[cmd] = model_key
    
    success_count = 0
    failed_predictions = []
    
    for command, model_key in prediction_commands.items():
        # Add aging enforcer flag if this is a fielding prediction
        if 'fielding' in command and use_aging_enforcer:
            command += ' --use-aging-enforcer'
        
        description = f"Generating predictions for {model_key}"
        
        if run_command(command, description, timeout=3600):
            success_count += 1
        else:
            failed_predictions.append(model_key)
    
    print(f"\n{'='*70}")
    print(f"Prediction Summary: {success_count}/{len(prediction_commands)} successful")
    if failed_predictions:
        print(f"Failed predictions: {', '.join(failed_predictions)}")
    print(f"{'='*70}\n")
    
    return len(failed_predictions) == 0


def check_prediction_files() -> bool:
    """Check if all required prediction files exist"""
    required_files = [
        'batter_predictions.csv',
        'pitcher_predictions.csv',
        'baserunning_predictions.csv',
        'fielding_predictions.csv'
    ]
    
    missing = []
    for filename in required_files:
        filepath = PIPELINE_DIR / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        logger.error(f"ERROR: Missing prediction files: {', '.join(missing)}")
        logger.error("Please generate predictions first (option 2) or run full pipeline (option 4)")
        return False
    
    return True


def run_projection_engine(projection_year: int = 2026) -> bool:
    """
    Run the projection engine: allocate playing time and calculate WAR.
    
    This unified step:
    1. Allocates playing time based on rate stats (wOBA/FIP) and injuries
    2. Calculates WAR based on allocated playing time
    3. Exports final projections with all components
    
    Args:
        projection_year: Year to project
        
    Returns:
        True if successful
    """
    print(f"\nRunning Projection Engine for {projection_year}...")
    print("   → Allocating playing time based on wOBA/FIP rankings")
    print("   → Calculating WAR based on allocated games/IP")
    print("   → Incorporating injury adjustments")
    
    # Check if prediction files exist
    if not check_prediction_files():
        return False
    
    command = f"python -m playing_time.main --year {projection_year}"
    description = f"Running projection engine for {projection_year}"
    
    success = run_command(command, description, timeout=600)
    
    if success:
        output_file = DATA_DIR / 'generated' / 'playing_time' / f'projections_{projection_year}.csv'
        if output_file.exists():
            logger.info(f"SUCCESS: Final projections saved to: {output_file}")
        else:
            logger.warning("WARNING: Output file not found at expected location")
    
    return success


def calculate_trade_values() -> bool:
    """
    Calculate trade values for all players.
    
    This runs the value determination pipeline:
    1. Loads predictions (must already exist)
    2. Loads salary/contract data
    3. Calculates WAR with proper park factors
    4. Calculates surplus value (projected WAR value - contract cost)
    5. Exports trade value rankings
    
    Returns:
        True if successful
    """
    print("\nCalculating Trade Values...")
    print("   → Loading predictions and salary data")
    print("   → Calculating WAR with park factors")
    print("   → Computing surplus value (WAR value - salary cost)")
    print("   → Generating trade rankings")
    
    # Check if prediction files exist
    if not check_prediction_files():
        return False
    
    command = "python -m value_determination.main"
    description = "Calculating trade values"
    
    success = run_command(command, description, timeout=600)
    
    if success:
        output_file = DATA_DIR / 'generated' / 'value_by_year' / 'player_values_complete.csv'
        if output_file.exists():
            logger.info(f"SUCCESS: Trade values saved to: {output_file}")
        else:
            logger.warning("WARNING: Output file not found at expected location")
    
    return success


def run_full_pipeline() -> bool:
    """Run complete pipeline: train → predict → project (playing time + WAR) → trade values"""
    print("\nStarting Full Pipeline...")
    print("This will: Train all models → Generate predictions → Run Projection Engine → Calculate Trade Values")
    print("\nPipeline stages:")
    print("  1. Pre-train batter, pitcher_sp, pitcher_rp on classical features")
    print("  2. Fine-tune batter, pitcher_sp, pitcher_rp with Statcast features")
    print("  3. Train baserunning and defense models")
    print("  4. Generate rate stat predictions")
    print("  5. Allocate playing time & calculate WAR (Projection Engine)")
    print("  6. Calculate trade values (surplus value analysis)")
    print("\nContinue? (y/n)")
    
    if input("> ").strip().lower() != 'y':
        print("Pipeline cancelled")
        return False
    
    start_time = datetime.now()
    
    # Get projection year
    projection_year = get_projection_year()
    
    # Get hyperparameter overrides
    overrides = get_hyperparameter_overrides()
    
    # Step 1: Train models in correct order for transfer learning
    # Pre-training phase (must happen first)
    pretrain_models = ['batter_pretrain', 'pitcher_sp_pretrain', 'pitcher_rp_pretrain']
    print("\n" + "="*60)
    print("PHASE 1: PRE-TRAINING (Classical features)")
    print("="*60)
    if not train_models(pretrain_models, overrides):
        logger.error("FAILED: Pre-training phase failed")
        return False
    
    # Fine-tuning phase (requires pre-trained checkpoints)
    finetune_models = ['batter_finetune', 'pitcher_sp_finetune', 'pitcher_rp_finetune']
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING (Statcast features)")
    print("="*60)
    if not train_models(finetune_models, overrides):
        logger.error("FAILED: Fine-tuning phase failed")
        return False
    
    # Other models (no transfer learning)
    other_models = ['baserunning', 'defense_infield', 'defense_outfield', 'defense_catcher']
    print("\n" + "="*60)
    print("PHASE 3: OTHER MODELS (Baserunning, Defense)")
    print("="*60)
    if not train_models(other_models, overrides):
        logger.error("FAILED: Other models training failed")
        return False
    
    # Step 2: Generate predictions (use finetuned models)
    prediction_models = ['batter_finetune', 'pitcher_sp_finetune', 'baserunning', 'defense_infield']
    use_aging_enforcer = overrides.get('use_aging_enforcer', False)
    if not generate_predictions(prediction_models, use_aging_enforcer):
        logger.error("FAILED: Prediction phase failed")
        return False
    
    # Step 3: Run Projection Engine (playing time + WAR)
    if not run_projection_engine(projection_year):
        logger.error("FAILED: Projection engine failed")
        return False
    
    # Step 4: Calculate Trade Values
    if not calculate_trade_values():
        logger.error("FAILED: Trade value calculation failed")
        return False
    
    duration = datetime.now() - start_time
    
    print(f"\n{'='*70}")
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total duration: {duration}")
    print(f"Projection year: {projection_year}")
    print(f"{'='*70}\n")
    
    return True


def display_output_files():
    """Display information about generated files"""
    print("\nGenerated Files:")
    
    playing_time_dir = DATA_DIR / 'generated' / 'playing_time'
    
    value_dir = DATA_DIR / 'generated' / 'value_by_year'
    
    output_files = {
        'Raw Predictions (Rate Stats)': [
            PIPELINE_DIR / 'batter_predictions.csv',
            PIPELINE_DIR / 'pitcher_predictions.csv',
            PIPELINE_DIR / 'baserunning_predictions.csv',
            PIPELINE_DIR / 'fielding_predictions.csv'
        ],
        'Final Projections (with WAR)': [
            playing_time_dir / 'projections_2026.csv',
            playing_time_dir / 'team_summary_2026.csv'
        ],
        'Trade Values (Surplus Value)': [
            value_dir / 'player_values_complete.csv'
        ]
    }
    
    for category, files in output_files.items():
        print(f"\n{category}:")
        for filepath in files:
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"  SUCCESS: {filepath.name}: {size:,} bytes ({size/1024/1024:.1f} MB)")
            else:
                print(f"  MISSING: {filepath.name}: Not found")


def main():
    """Main pipeline interface"""
    print_header()
    
    print("Welcome to the MLB Prediction Pipeline!")
    print(f"Output directory: {PIPELINE_DIR}")
    
    while True:
        print_menu()
        choice = get_user_choice("Select option (1-6): ", ['1', '2', '3', '4', '5', '6'])
        
        if choice == '1':
            # Train Models
            selected = select_models_for_training()
            overrides = get_hyperparameter_overrides()
            train_models(selected, overrides)
            
        elif choice == '2':
            # Generate Predictions
            selected = select_models_for_prediction()
            
            # Ask about aging enforcer for fielding predictions
            use_aging_enforcer = False
            if any('defense' in model for model in selected):
                print("\nApply aging enforcer to fielding predictions? (y/n)")
                print("(Prevents unrealistic late-career defensive improvements)")
                if input("> ").strip().lower() == 'y':
                    use_aging_enforcer = True
            
            generate_predictions(selected, use_aging_enforcer)
            
        elif choice == '3':
            # Run Projection Engine
            projection_year = get_projection_year()
            run_projection_engine(projection_year)
            
        elif choice == '4':
            # Calculate Trade Values
            calculate_trade_values()
            
        elif choice == '5':
            # Run Full Pipeline
            run_full_pipeline()
            
        elif choice == '6':
            # Exit
            print("\nExiting pipeline. Goodbye!")
            display_output_files()
            break
        
        # Ask if user wants to continue
        print("\nReturn to main menu? (y/n)")
        if input("> ").strip().lower() != 'y':
            print("\nExiting pipeline. Goodbye!")
            display_output_files()
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWARNING: Pipeline interrupted by user")
        display_output_files()
        sys.exit(0)
    except Exception as e:
        logger.error(f"ERROR: Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
