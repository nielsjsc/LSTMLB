#!/usr/bin/env python3
"""
MLB Prediction Pipeline - Unified Interface
============================================

Single script to handle all pipeline operations:
- Train models (with optional hyperparameter customization)
- Generate predictions
- Combine predictions into final WAR calculations
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

# Ensure directories exist
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_TYPES = {
    'batter_pretrain': {
        'training_command': 'python scripts/train_models.py --model batter',
        'prediction_command': 'python scripts/predict_models.py --model-type batter --use-pretrained',
        'output_file': 'batter_predictions.csv',
        'description': 'Batter PRE-TRAINING (classical features, 2000-2024)'
    },
    'batter_finetune': {
        'training_command': 'python scripts/train_models.py --model batter --finetune --from-checkpoint checkpoints/batter_pretrained.pth',
        'prediction_command': 'python scripts/predict_models.py --model-type batter',
        'output_file': 'batter_predictions.csv',
        'description': 'Batter FINE-TUNING (adds Statcast features, 2015+) - Requires pre-trained checkpoint'
    },
    'pitcher_sp': {
        'training_command': 'python scripts/train_models.py --model pitcher_sp',
        'prediction_command': 'python scripts/predict_models.py --model-type pitcher',
        'output_file': 'pitcher_predictions.csv',
        'description': 'Starting Pitcher stats'
    },
    'pitcher_rp': {
        'training_command': 'python scripts/train_models.py --model pitcher_rp',
        'prediction_command': None,  # Handled by pitcher_sp prediction
        'output_file': None,
        'description': 'Relief Pitcher stats'
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
    }
}


def print_header():
    """Print pipeline header"""
    print("\n" + "=" * 70)
    print("⚾ MLB PREDICTION PIPELINE")
    print("=" * 70)
    print()


def print_menu():
    """Display main menu"""
    print("\nMAIN MENU:")
    print("1. Train Models")
    print("2. Generate Predictions")
    print("3. Combine Predictions (Calculate WAR)")
    print("4. Run Full Pipeline (Train → Predict → Combine)")
    print("5. Exit")
    print()


def get_user_choice(prompt: str, valid_choices: List[str]) -> str:
    """Get validated user input"""
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")


def select_models_for_training() -> List[str]:
    """Let user select which models to train"""
    print(f"\nSelect models to train:")
    print("0. All models")
    
    model_list = list(MODEL_TYPES.keys())
    for i, (model_key, info) in enumerate(MODEL_TYPES.items(), 1):
        print(f"{i}. {model_key}: {info['description']}")
    
    print("\nEnter numbers separated by commas (e.g., 1,2,3) or 0 for all:")
    choice = input("> ").strip()
    
    if choice == '0':
        return model_list
    
    try:
        indices = [int(x.strip()) - 1 for x in choice.split(',')]
        selected = [model_list[i] for i in indices if 0 <= i < len(model_list)]
        if selected:
            return selected
    except (ValueError, IndexError):
        pass
    
    print("Invalid selection, using all models")
    return model_list


def select_models_for_prediction() -> List[str]:
    """Let user select which prediction groups to generate"""
    print(f"\nSelect predictions to generate:")
    print("0. All predictions")
    print("1. Batter predictions (Finetuned - Classical + Statcast)")
    print("2. Batter predictions (Pretrained - Classical only)")
    print("3. Pitcher predictions (SP + RP combined)")
    print("4. Baserunning predictions")
    print("5. Fielding predictions (All positions)")
    
    print("\nEnter numbers separated by commas (e.g., 1,3,4) or 0 for all:")
    choice = input("> ").strip()
    
    # Map user choices to model keys
    prediction_groups = {
        '1': ['batter_finetune'],  # Finetuned model (recommended)
        '2': ['batter_pretrain'],  # Pretrained model (classical only)
        '3': ['pitcher_sp'],  # Running pitcher_sp generates both SP and RP predictions
        '4': ['baserunning'],
        '5': ['defense_infield']  # Running any defense generates all 3 positions
    }
    
    if choice == '0':
        # All predictions - use finetuned by default
        return ['batter_finetune', 'pitcher_sp', 'baserunning', 'defense_infield']
    
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
    return ['batter_finetune', 'pitcher_sp', 'baserunning', 'defense_infield']


def get_hyperparameter_overrides() -> Dict[str, Any]:
    """Ask user if they want to customize hyperparameters"""
    print("\nWould you like to customize hyperparameters? (y/n)")
    if input("> ").strip().lower() != 'y':
        return {}
    
    overrides = {}
    
    print("\nAvailable hyperparameter overrides:")
    print("- epochs: Number of training epochs (default: 50)")
    print("Enter 'done' when finished, or press Enter to skip")
    
    # Epochs
    print("\nEpochs (default: 50, press Enter to skip):")
    epochs = input("> ").strip()
    if epochs and epochs.isdigit():
        overrides['epochs'] = int(epochs)
    
    return overrides


def run_command(command: str, description: str, timeout: int = 7200) -> bool:
    """Execute a command with progress tracking"""
    logger.info(f"▶ {description}")
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
            logger.info(f"✅ {description} completed in {duration}")
            return True
        else:
            logger.error(f"❌ {description} failed with code {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out after {timeout}s")
        process.kill()
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed: {str(e)}")
        return False


def train_models(selected_models: List[str], hyperparameter_overrides: Dict[str, Any]) -> bool:
    """Train selected models"""
    print(f"\n🏋️ Training {len(selected_models)} model(s)...")
    
    success_count = 0
    failed_models = []
    
    for model_key in selected_models:
        info = MODEL_TYPES[model_key]
        command = info['training_command']
        
        # Apply hyperparameter overrides
        if 'epochs' in hyperparameter_overrides:
            command += f" --epochs {hyperparameter_overrides['epochs']}"
        
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


def generate_predictions(selected_models: List[str]) -> bool:
    """Generate predictions for selected models"""
    print(f"\n🔮 Generating predictions for {len(selected_models)} model(s)...")
    
    # Group models by prediction command
    prediction_commands = {}
    for model_key in selected_models:
        info = MODEL_TYPES[model_key]
        cmd = info['prediction_command']
        if cmd and cmd not in prediction_commands:
            prediction_commands[cmd] = model_key
    
    success_count = 0
    failed_predictions = []
    
    for command, model_key in prediction_commands.items():
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
        'baserunning_predictions.csv',
        'fielding_predictions.csv'
    ]
    
    missing = []
    for filename in required_files:
        filepath = PIPELINE_DIR / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        logger.error(f"❌ Missing prediction files: {', '.join(missing)}")
        logger.error("Please generate predictions first (option 2) or run full pipeline (option 4)")
        return False
    
    return True


def combine_predictions() -> bool:
    """Combine predictions and calculate WAR"""
    print("\n🔗 Combining predictions and calculating WAR...")
    
    # Check if prediction files exist
    if not check_prediction_files():
        return False
    
    command = "python evaluation/calculate_war.py"
    description = "Combining predictions and calculating WAR"
    
    success = run_command(command, description, timeout=600)
    
    if success:
        output_file = PIPELINE_DIR / "batter_predictions_with_war.csv"
        if output_file.exists():
            logger.info(f"✅ Final predictions saved to: {output_file}")
        else:
            logger.warning("⚠️ Output file not found at expected location")
    
    return success


def run_full_pipeline() -> bool:
    """Run complete pipeline: train → predict → combine"""
    print("\n🚀 Starting Full Pipeline...")
    print("This will: Train all models → Generate predictions → Calculate WAR")
    print("\nContinue? (y/n)")
    
    if input("> ").strip().lower() != 'y':
        print("Pipeline cancelled")
        return False
    
    start_time = datetime.now()
    
    # Get hyperparameter overrides
    overrides = get_hyperparameter_overrides()
    
    # Step 1: Train all models
    all_models = list(MODEL_TYPES.keys())
    if not train_models(all_models, overrides):
        logger.error("❌ Training phase failed")
        return False
    
    # Step 2: Generate predictions
    if not generate_predictions(all_models):
        logger.error("❌ Prediction phase failed")
        return False
    
    # Step 3: Combine predictions
    if not combine_predictions():
        logger.error("❌ Combination phase failed")
        return False
    
    duration = datetime.now() - start_time
    
    print(f"\n{'='*70}")
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total duration: {duration}")
    print(f"{'='*70}\n")
    
    return True


def display_output_files():
    """Display information about generated files"""
    print("\n📁 Generated Files:")
    
    output_files = {
        'Raw Predictions': [
            PIPELINE_DIR / 'batter_predictions.csv',
            PIPELINE_DIR / 'pitcher_predictions.csv',
            PIPELINE_DIR / 'baserunning_predictions.csv',
            PIPELINE_DIR / 'fielding_predictions.csv'
        ],
        'Final Output': [
            PIPELINE_DIR / 'batter_predictions_with_war.csv'
        ]
    }
    
    for category, files in output_files.items():
        print(f"\n{category}:")
        for filepath in files:
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"  ✅ {filepath.name}: {size:,} bytes ({size/1024/1024:.1f} MB)")
            else:
                print(f"  ❌ {filepath.name}: Not found")


def main():
    """Main pipeline interface"""
    print_header()
    
    print("Welcome to the MLB Prediction Pipeline!")
    print(f"Output directory: {PIPELINE_DIR}")
    
    while True:
        print_menu()
        choice = get_user_choice("Select option (1-5): ", ['1', '2', '3', '4', '5'])
        
        if choice == '1':
            # Train Models
            selected = select_models_for_training()
            overrides = get_hyperparameter_overrides()
            train_models(selected, overrides)
            
        elif choice == '2':
            # Generate Predictions
            selected = select_models_for_prediction()
            generate_predictions(selected)
            
        elif choice == '3':
            # Combine Predictions
            combine_predictions()
            
        elif choice == '4':
            # Run Full Pipeline
            run_full_pipeline()
            
        elif choice == '5':
            # Exit
            print("\n👋 Exiting pipeline. Goodbye!")
            display_output_files()
            break
        
        # Ask if user wants to continue
        print("\nReturn to main menu? (y/n)")
        if input("> ").strip().lower() != 'y':
            print("\n👋 Exiting pipeline. Goodbye!")
            display_output_files()
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        display_output_files()
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
