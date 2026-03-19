# Model registry and factory

from core.model_architecture import ImprovedLSTM
from core.data_processing import DataConfig
from configs.baserunning_config import BaserunningConfig
from configs.defense_infield_config import DefenseInfieldConfig
from configs.defense_outfield_config import DefenseOutfieldConfig
from configs.defense_catcher_config import DefenseCatcherConfig
from configs.pitcher_sp_config import PitcherSPConfig
from configs.pitcher_rp_config import PitcherRPConfig
from configs.batter_config import BatterConfig
from configs.defense_infield_historical import DefenseInfieldHistoricalConfig
from configs.defense_outfield_historical import DefenseOutfieldHistoricalConfig
from configs.defense_catcher_historical import DefenseCatcherHistoricalConfig
from configs.baserunning_historical import BaserunningHistoricalConfig

class ModelFactory:
    """Factory for creating models based on type"""
    
    CONFIGS = {
        'baserunning': BaserunningConfig,
        'defense_infield': DefenseInfieldConfig,
        'defense_outfield': DefenseOutfieldConfig,
        'defense_catcher': DefenseCatcherConfig,
        'pitcher_sp': PitcherSPConfig,
        'pitcher_rp': PitcherRPConfig,
        'batter': BatterConfig,
        'defense_infield_historical': DefenseInfieldHistoricalConfig,
        'defense_outfield_historical': DefenseOutfieldHistoricalConfig,
        'defense_catcher_historical': DefenseCatcherHistoricalConfig,
        'baserunning_historical': BaserunningHistoricalConfig,
    }
    
    # Data file mapping for each model type
    DATA_FILES = {
        'baserunning': '../data/historic_mlb/mlb_batting_data_1950_2025_with_statcast.csv',
        'defense_infield': '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv',
        'defense_outfield': '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv', 
        'defense_catcher': '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv',
        'pitcher_sp': '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv',
        'pitcher_rp': '../data/historic_mlb/mlb_pitching_data_1950_2025_with_statcast.csv',
        'batter': '../data/historic_mlb/mlb_batting_data_1950_2025_with_statcast.csv',
        'defense_infield_historical': '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv',
        'defense_outfield_historical': '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv',
        'defense_catcher_historical': '../data/historic_mlb/mlb_fielding_data_2000_2025_with_statcast.csv',
        'baserunning_historical': '../data/historic_mlb/mlb_batting_data_1950_2025_with_statcast.csv',
    }
    
    # Convenience mapping for multi-model types
    MULTI_MODEL_GROUPS = {
        'defense': ['defense_infield', 'defense_outfield', 'defense_catcher'],
        'defense_historical': ['defense_infield_historical', 'defense_outfield_historical', 'defense_catcher_historical'],
        'pitcher': ['pitcher_sp', 'pitcher_rp'],
    }
    
    @classmethod
    def get_config(cls, model_type: str):
        """Get configuration for model type"""
        if model_type not in cls.CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.CONFIGS.keys())}")
        
        config = cls.CONFIGS[model_type]
        # Handle both class-based configs (defense) and module-based configs (pitcher, batter)
        if hasattr(config, 'DATA_CONFIG'):
            # Module-based config
            return config
        else:
            # Class-based config
            return config
    
    @classmethod
    def get_data_file(cls, model_type: str):
        """Get data file path for model type"""
        if model_type not in cls.DATA_FILES:
            raise ValueError(f"No data file defined for model type: {model_type}")
        return cls.DATA_FILES[model_type]
    
    @classmethod
    def get_model_group(cls, group_name: str):
        """Get all models in a group (e.g., 'defense' returns all defense models)"""
        if group_name in cls.MULTI_MODEL_GROUPS:
            return cls.MULTI_MODEL_GROUPS[group_name]
        elif group_name in cls.CONFIGS:
            return [group_name]  # Single model
        else:
            raise ValueError(f"Unknown model group: {group_name}")
    
    @classmethod
    def create_model(cls, model_type: str, input_size: int, output_size: int, device):
        """Create model instance for given type"""
        config = cls.get_config(model_type)
        
        # Handle both config types
        if hasattr(config, 'HIDDEN_SIZE'):
            # Simple constant-based config (pitcher, batter) or class-based config (defense)
            model = ImprovedLSTM(
                input_size=input_size,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                output_size=output_size,
                dropout=config.DROPOUT,
                bidirectional=config.BIDIRECTIONAL,
                num_heads=config.NUM_HEADS
            ).to(device)
        else:
            # Module-based config (deprecated path)
            model_config = config.MODEL_CONFIG
            model = ImprovedLSTM(
                input_size=input_size,
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                output_size=output_size,
                dropout=model_config['dropout'],
                bidirectional=model_config['bidirectional'],
                num_heads=model_config['num_heads']
            ).to(device)
        
        return model
