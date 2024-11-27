import os
from typing import Dict

class Config:
    """Configuration for cloud-based ScholarAI"""
    def __init__(self):
        """Initialize configuration with default values"""
        self.hidden_size = 256 
        self.num_attention_heads = 4
        self.num_hidden_layers = 6
        self.intermediate_size = 512
        
        # Cloud storage settings
        self.google_drive_folder = "scholarai_data"
        self.aws_bucket = "scholarai-training"
        self.aws_region = "us-west-2"
        
        # Training settings
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.max_epochs = 100
        self.early_stopping_patience = 5
        
        # Knowledge management
        self.max_concepts = 1000
        self.max_sources = 5000
        
        # Local cache settings
        self.local_cache_size = 1000
        self.cache_dir = os.path.expanduser("~/scholarai_cache")
        
        # Liquid Neural Network settings
        self.num_liquid_layers = 3
        self.dt = 0.1  # Time step for numerical integration
        self.integration_steps = 5  # Number of integration steps
        self.solver = 'euler'  # Can be 'euler' or 'rk4'

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format"""
        config_dict = {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'intermediate_size': self.intermediate_size,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'num_liquid_layers': self.num_liquid_layers,
            'dt': self.dt,
            'integration_steps': self.integration_steps,
            'solver': self.solver
        }
        return config_dict 