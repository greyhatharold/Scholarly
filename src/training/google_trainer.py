from google.cloud import aiplatform
from datetime import datetime
import os
from typing import Dict, Literal, Optional
import logging
from dataclasses import dataclass
from .deployment_manager import DeploymentManager, ModelValidator, DeploymentMonitor
from src.data.versioning.model_version_manager import ModelVersionManager
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class TrainingVersion:
    """Represents a training job version"""
    version: str
    job_id: str
    training_phase: str
    timestamp: datetime
    hyperparameters: Dict
    machine_config: Dict
    metrics: Optional[Dict] = None

class GoogleTrainer:
    """Handles model training on Google Cloud infrastructure"""
    def __init__(self, config):
        self.config = config
        self.version_manager = ModelVersionManager(config)
        aiplatform.init(
            project=os.environ.get('GOOGLE_CLOUD_PROJECT'),
            location=os.environ.get('GOOGLE_CLOUD_REGION', 'us-central1')
        )
        self.container_registry = os.environ.get('GOOGLE_CONTAINER_REGISTRY')
        asyncio.create_task(self.version_manager.initialize())
        
    async def prepare_training_job(
        self,
        training_phase: Literal['pretrain', 'finetune'],
        version: str
    ) -> str:
        """Prepares and launches a Vertex AI training job with version tracking

        Args:
            training_phase: Phase of training ('pretrain' or 'finetune')
            version: Version string for this training run
        """
        job_name = self._generate_job_name(training_phase)
        machine_config = self._get_machine_config(training_phase)
        container_uri = await self._prepare_container_image(training_phase)
        hyperparameters = self._get_hyperparameters(training_phase)
        
        # Create training version record
        training_version = TrainingVersion(
            version=version,
            job_id=job_name,
            training_phase=training_phase,
            timestamp=datetime.now(),
            hyperparameters=hyperparameters,
            machine_config=machine_config
        )
        
        # Save training version metadata
        await self._save_training_version(training_version)
        
        training_params = {
            'container_uri': container_uri,
            'model_serving_container_image_uri': container_uri,
            'machine_type': machine_config['machine_type'],
            'accelerator_type': machine_config['accelerator_type'],
            'accelerator_count': machine_config['accelerator_count'],
            'replica_count': machine_config['replica_count'],
            'base_output_dir': f'gs://{self.config.gcp_bucket}/training_output/{training_phase}/{version}',
            'hyperparameters': hyperparameters
        }
        
        job = aiplatform.CustomTrainingJob(
            display_name=job_name,
            **training_params
        )
        job.run(sync=False)
        return job_name

    async def _save_training_version(self, training_version: TrainingVersion):
        """Save training version metadata"""
        metadata = {
            'job_id': training_version.job_id,
            'training_phase': training_version.training_phase,
            'timestamp': training_version.timestamp.isoformat(),
            'hyperparameters': training_version.hyperparameters,
            'machine_config': training_version.machine_config,
            'metrics': training_version.metrics
        }
        
        await self.version_manager.save_version(
            model=None,  # No model state yet
            version=training_version.version,
            metadata=metadata
        )

    async def update_training_metrics(self, version: str, metrics: Dict):
        """Update version with training metrics"""
        version_info = await self.version_manager.get_version(version)
        if version_info:
            version_info.metrics = metrics
            await self.version_manager.save_version(
                model=None,
                version=version,
                metadata=version_info.metadata,
                metrics=metrics
            )

    async def automated_deployment(
        self, 
        version: str,
        approve_deployment: bool = False,
        deployment_config: Optional[Dict] = None
    ) -> str:
        """Handle automated model deployment with version tracking and approval

        Args:
            version: Version string to deploy
            approve_deployment: Explicit deployment approval flag
            deployment_config: Optional custom deployment configuration
        
        Raises:
            ValueError: If deployment not approved or version not found
            RuntimeError: If deployment fails
        """
        if not approve_deployment:
            raise ValueError(
                "Deployment requires explicit approval. "
                "Set approve_deployment=True to proceed."
            )

        try:
            # Initialize deployment components
            validator = ModelValidator(self.config)
            monitor = DeploymentMonitor(self.config)
            deployment_manager = DeploymentManager(
                self.config,
                validator,
                monitor,
                require_approval=True
            )
            
            # Get model artifacts for specific version
            model_artifacts = await self._get_version_artifacts(version)
            
            # Add deployment approval metadata
            deployment_metadata = {
                'approved_at': datetime.now().isoformat(),
                'approved_by': os.environ.get('DEPLOYMENT_USER', 'unknown'),
                'deployment_config': deployment_config or {}
            }
            
            # Deploy model with approval metadata
            deployment_id = await deployment_manager.deploy_model(
                model_artifacts,
                deployment_metadata=deployment_metadata
            )
            
            # Update version metadata with deployment info
            await self._update_deployment_info(
                version, 
                deployment_id,
                deployment_metadata
            )
            
            logger.info(f"Successfully deployed approved model version {version}: {deployment_id}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Automated deployment failed for version {version}: {e}")
            raise

    async def _get_version_artifacts(self, version: str) -> Dict:
        """Get artifacts for specific model version"""
        version_info = await self.version_manager.get_version(version)
        if not version_info:
            raise ValueError(f"Version {version} not found")
            
        # Get artifacts from versioned storage
        artifacts = {
            'model_path': f"gs://{self.config.gcp_bucket}/models/v{version}",
            'metadata': version_info.metadata,
            'metrics': version_info.metrics,
            'serving_image': await self._get_serving_image(version)
        }
        return artifacts

    async def _update_deployment_info(
        self, 
        version: str, 
        deployment_id: str,
        deployment_metadata: Dict
    ):
        """Update version metadata with deployment information"""
        version_info = await self.version_manager.get_version(version)
        if version_info:
            metadata = {
                **version_info.metadata,
                'deployment': {
                    'id': deployment_id,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'deployed',
                    **deployment_metadata
                }
            }
            await self.version_manager.save_version(
                model=None,
                version=version,
                metadata=metadata,
                metrics=version_info.metrics
            )

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

    async def prepare_continuous_training(
        self,
        version: str,
        continuous_config: Optional[Dict] = None
    ) -> str:
        """Prepares continuous training job with adaptive learning
        
        Args:
            version: Version string for this training run
            continuous_config: Optional configuration for continuous learning
            
        Returns:
            str: Job name for the continuous training process
        """
        logger.debug("Preparing continuous training job")
        job_name = self._generate_job_name('continuous')
        
        # Use pretrain config as base but adjust for continuous learning
        machine_config = self._get_machine_config('pretrain')
        hyperparameters = self._get_continuous_hyperparameters(continuous_config)
        
        # Create training version record
        training_version = TrainingVersion(
            version=version,
            job_id=job_name,
            training_phase='continuous',
            timestamp=datetime.now(),
            hyperparameters=hyperparameters,
            machine_config=machine_config
        )
        
        await self._save_training_version(training_version)
        
        # Configure continuous training job
        training_params = {
            **self._get_base_training_params(machine_config),
            'hyperparameters': hyperparameters,
            'scheduling': {
                'continuous_eval_frequency': 300,  # Every 5 minutes
                'enable_adaptive_scheduling': True
            }
        }
        
        job = aiplatform.CustomTrainingJob(
            display_name=job_name,
            **training_params
        )
        
        # Start non-blocking continuous training
        job.run(sync=False)
        return job_name
        
    def _get_continuous_hyperparameters(self, continuous_config: Optional[Dict] = None) -> Dict:
        """Get hyperparameters optimized for continuous learning
        
        Args:
            continuous_config: Optional custom configuration
            
        Returns:
            Dict: Hyperparameters for continuous learning
        """
        base_params = self._get_hyperparameters('pretrain')
        continuous_params = {
            # Liquid neural network parameters
            'dt_min': str(self.config.dt / 10),
            'dt_max': str(self.config.dt * 10),
            'adaptive_integration': 'True',
            
            # Continuous learning parameters
            'enable_complexity_tracking': 'True',
            'min_complexity_threshold': '0.1',
            'max_complexity_threshold': '10.0',
            'connection_strength_decay': '0.99',
            
            # Stream processing parameters
            'batch_accumulation_size': '64',
            'min_update_frequency': '100',
            'max_update_frequency': '1000'
        }
        
        # Override with custom config if provided
        if continuous_config:
            continuous_params.update(continuous_config)
            
        return {**base_params, **continuous_params}
        
    def _get_base_training_params(self, machine_config: Dict) -> Dict:
        """Get base training parameters
        
        Args:
            machine_config: Machine configuration dictionary
            
        Returns:
            Dict: Base training parameters
        """
        return {
            'machine_type': machine_config['machine_type'],
            'accelerator_type': machine_config['accelerator_type'],
            'accelerator_count': machine_config['accelerator_count'],
            'replica_count': machine_config['replica_count'],
            'base_output_dir': f'gs://{self.config.gcp_bucket}/continuous_training'
        }