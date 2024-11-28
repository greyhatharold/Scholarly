import os
from typing import Dict
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VectorStoreConfig:
    dimensions: int = 768
    index_type: str = 'IVF'
    n_lists: int = 100
    n_probes: int = 10
    quantization_bits: int = 8
    compression_level: int = 9
    use_dimensionality_reduction: bool = True
    target_dimensions: int = 64

@dataclass
class DatabaseConfig:
    db_path: str = "vectors.db"
    connection_timeout: int = 30
    max_connections: int = 5

@dataclass
class EmbeddingConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 32
    target_dims: int = 256
    device: str = "cuda"

@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    pretrain_machine_type: str = 'n1-standard-8'
    pretrain_accelerator_type: str = 'NVIDIA_TESLA_V100'
    pretrain_accelerator_count: int = 2
    pretrain_replica_count: int = 1
    
    finetune_machine_type: str = 'n1-standard-4'
    finetune_accelerator_type: str = 'NVIDIA_TESLA_T4'
    finetune_accelerator_count: int = 1
    finetune_replica_count: int = 1
    
    # Training hyperparameters
    pretrain_learning_rate: float = 1e-4
    pretrain_batch_size: int = 128
    pretrain_epochs: int = 10
    
    finetune_learning_rate: float = 5e-5
    finetune_batch_size: int = 32
    finetune_epochs: int = 3

class Config:
    """Configuration for cloud-based ScholarAI"""
    def __init__(self):
        """Initialize configuration with default values"""
        self.project_root = str(Path(__file__).parent.parent.parent.absolute())
        
        self.hidden_size = 256
        self.num_attention_heads = 4
        self.num_hidden_layers = 6
        self.intermediate_size = 512
        
        # Cloud storage settings
        self.google_drive_folder = os.environ.get('GOOGLE_DRIVE_FOLDER', 'scholarly-data')
        
        # Google Cloud Platform settings
        self.gcp_project = os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.gcp_region = os.environ.get('GOOGLE_CLOUD_REGION', 'us-central1')
        self.gcp_bucket = os.environ.get('GOOGLE_CLOUD_BUCKET')
        if not self.gcp_bucket:
            raise ValueError("GOOGLE_CLOUD_BUCKET environment variable must be set")
        if '-' in self.gcp_bucket:
            print(f"Warning: Bucket name '{self.gcp_bucket}' contains hyphens which may cause issues in some contexts")
        
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
        
        # Initialize vector store before embedding config
        self.vector_store = VectorStoreConfig()
        # Align vector store dimensions with hidden size
        self.vector_store.dimensions = self.hidden_size
        
        # Initialize embedding config
        self.embedding = EmbeddingConfig()
        self.embedding.target_dims = self.hidden_size
        
        # Rest of the existing initialization
        self.database = DatabaseConfig()
        self.training = TrainingConfig()

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
            'solver': self.solver,
            'gcp_project': self.gcp_project,
            'gcp_region': self.gcp_region,
            'gcp_bucket': self.gcp_bucket,
            'training': {
                'pretrain': {
                    'machine_type': self.training.pretrain_machine_type,
                    'accelerator_type': self.training.pretrain_accelerator_type,
                    'accelerator_count': self.training.pretrain_accelerator_count,
                    'replica_count': self.training.pretrain_replica_count,
                    'learning_rate': self.training.pretrain_learning_rate,
                    'batch_size': self.training.pretrain_batch_size,
                    'epochs': self.training.pretrain_epochs,
                },
                'finetune': {
                    'machine_type': self.training.finetune_machine_type,
                    'accelerator_type': self.training.finetune_accelerator_type,
                    'accelerator_count': self.training.finetune_accelerator_count,
                    'replica_count': self.training.finetune_replica_count,
                    'learning_rate': self.training.finetune_learning_rate,
                    'batch_size': self.training.finetune_batch_size,
                    'epochs': self.training.finetune_epochs,
                }
            }
        }
        return config_dict 