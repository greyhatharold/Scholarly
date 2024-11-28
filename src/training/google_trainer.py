from google.cloud import aiplatform
from datetime import datetime
import os
from typing import Dict, Literal

class GoogleTrainer:
    """Handles model training on Google Cloud infrastructure"""
    def __init__(self, config):
        self.config = config
        # Initialize Vertex AI with project and location
        aiplatform.init(
            project=os.environ.get('GOOGLE_CLOUD_PROJECT'),
            location=os.environ.get('GOOGLE_CLOUD_REGION', 'us-central1')
        )
        self.container_registry = os.environ.get('GOOGLE_CONTAINER_REGISTRY')
        
    async def prepare_training_job(self, training_phase: Literal['pretrain', 'finetune']) -> str:
        """Prepares and launches a Vertex AI training job with custom container

        Args:
            training_phase: Phase of training ('pretrain' or 'finetune')
        """
        job_name = self._generate_job_name(training_phase)
        machine_config = self._get_machine_config(training_phase)
        container_uri = await self._prepare_container_image(training_phase)
        
        training_params = {
            'container_uri': container_uri,
            'model_serving_container_image_uri': container_uri,
            'machine_type': machine_config['machine_type'],
            'accelerator_type': machine_config['accelerator_type'],
            'accelerator_count': machine_config['accelerator_count'],
            'replica_count': machine_config['replica_count'],
            'base_output_dir': f'gs://{self.config.gcp_bucket}/training_output/{training_phase}',
            'hyperparameters': self._get_hyperparameters(training_phase)
        }
        
        job = aiplatform.CustomTrainingJob(
            display_name=job_name,
            **training_params
        )
        job.run(sync=False)
        return job_name

    def _generate_job_name(self, training_phase: str) -> str:
        """Generates a unique training job name"""
        return f"scholarai-training-{training_phase}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    def _get_machine_config(self, training_phase: str) -> Dict:
        """Get machine configuration based on training phase"""
        if training_phase == 'pretrain':
            return {
                'machine_type': 'n1-standard-8',
                'accelerator_type': 'NVIDIA_TESLA_V100',
                'accelerator_count': 2,
                'replica_count': 1
            }
        return {
            'machine_type': 'n1-standard-4',
            'accelerator_type': 'NVIDIA_TESLA_T4',
            'accelerator_count': 1,
            'replica_count': 1
        }

    async def _prepare_container_image(self, training_phase: str) -> str:
        """Prepare and return container image URI with all required components"""
        base_image = f"{self.container_registry}/scholarai-training"
        tag = f"{training_phase}-{datetime.now().strftime('%Y%m%d')}"
        
        # Container build happens in CI/CD pipeline
        # Here we just return the appropriate image URI
        return f"{base_image}:{tag}"

    def _get_hyperparameters(self, training_phase: str) -> Dict:
        """Get hyperparameters based on training phase"""
        base_params = {
            # Vector compression params
            'quantization_bits': str(self.config.vector_store.quantization_bits),
            'compression_level': str(self.config.vector_store.compression_level),
            'use_dim_reduction': str(self.config.vector_store.use_dimensionality_reduction),
            
            # FAISS indexing params
            'index_type': 'IVF',
            'n_lists': str(self.config.vector_store.n_lists),
            'n_probes': str(self.config.vector_store.n_probes),
            
            # Liquid NN params
            'num_liquid_layers': str(self.config.num_liquid_layers),
            'dt': str(self.config.dt),
            'integration_steps': str(self.config.integration_steps),
            'hidden_size': str(self.config.hidden_size)
        }
        
        if training_phase == 'pretrain':
            base_params.update({
                'learning_rate': '1e-4',
                'batch_size': '128',
                'epochs': '10'
            })
        else:
            base_params.update({
                'learning_rate': '5e-5',
                'batch_size': '32',
                'epochs': '3'
            })
            
        return base_params