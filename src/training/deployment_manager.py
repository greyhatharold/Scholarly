from google.cloud import aiplatform
from datetime import datetime
from typing import Dict, Optional, List
import logging
import asyncio
from dataclasses import dataclass
from src.config.config import Config, TrainingConfig

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    def __init__(self, training_config: TrainingConfig, phase: str = 'finetune'):
        # Use training config based on phase (pretrain or finetune)
        if phase == 'pretrain':
            self.machine_type = training_config.pretrain_machine_type
            self.accelerator_type = training_config.pretrain_accelerator_type
            self.accelerator_count = training_config.pretrain_accelerator_count
            self.min_nodes = training_config.pretrain_replica_count
        else:
            self.machine_type = training_config.finetune_machine_type
            self.accelerator_type = training_config.finetune_accelerator_type
            self.accelerator_count = training_config.finetune_accelerator_count
            self.min_nodes = training_config.finetune_replica_count
            
        # Set max nodes to 3x min nodes for auto-scaling
        self.max_nodes = self.min_nodes * 3
        self.traffic_split: Dict[str, float] = {"default": 100}

@dataclass
class DeploymentApproval:
    """Tracks deployment approval metadata"""
    approved_at: datetime
    approved_by: str
    deployment_config: Dict
    
class ModelValidator:
    """Handles model validation before deployment"""
    
    def __init__(self, config: Config):
        self.config = config
        
    async def validate_model(self, model_artifacts: Dict) -> bool:
        """Validate model artifacts before deployment"""
        try:
            required_files = ['model.pkl', 'metadata.json']
            
            # Check for required files
            for file in required_files:
                if file not in model_artifacts:
                    logger.error(f"Missing required file: {file}")
                    return False
                    
            # Validate model size
            if not self._validate_model_size(model_artifacts['model.pkl']):
                return False
                
            # Validate model metadata
            if not await self._validate_model_metadata(model_artifacts['metadata.json']):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
            
    def _validate_model_size(self, model_file: bytes) -> bool:
        """Check if model size is within GCP limits"""
        size_mb = len(model_file) / (1024 * 1024)
        # Size limit based on machine type from config
        max_size = 2048 if 'n1-standard' in self.config.training.finetune_machine_type else 4096
        return size_mb <= max_size
        
    async def _validate_model_metadata(self, metadata: Dict) -> bool:
        """Validate model metadata against config"""
        try:
            # Check model architecture parameters
            required_params = {
                'hidden_size': self.config.hidden_size,
                'num_attention_heads': self.config.num_attention_heads,
                'num_hidden_layers': self.config.num_hidden_layers,
                'intermediate_size': self.config.intermediate_size,
                'num_liquid_layers': self.config.num_liquid_layers
            }
            
            for param, expected_value in required_params.items():
                if metadata.get(param) != expected_value:
                    logger.error(f"Model {param} mismatch: expected {expected_value}, "
                               f"got {metadata.get(param)}")
                    return False
                    
            # Validate input/output dimensions
            if metadata.get('input_dims') != self.config.vector_store.dimensions:
                logger.error("Input dimensions mismatch")
                return False
                
            if metadata.get('output_dims') != self.config.hidden_size:
                logger.error("Output dimensions mismatch")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            return False

class DeploymentMonitor:
    """Monitors deployment status and health"""
    
    def __init__(self, config: Config):
        self.config = config
        self.health_checks = {
            'latency': self._check_latency,
            'error_rate': self._check_error_rate,
            'resource_usage': self._check_resource_usage
        }
        
    async def monitor_deployment(self, endpoint_id: str) -> bool:
        """Monitor deployment health metrics"""
        try:
            results = await asyncio.gather(*[
                check(endpoint_id) 
                for check in self.health_checks.values()
            ])
            return all(results)
            
        except Exception as e:
            logger.error(f"Deployment monitoring failed: {e}")
            return False

class DeploymentManager:
    """Manages the entire deployment process with approval requirements"""
    
    def __init__(
        self, 
        config: Config, 
        validator: ModelValidator, 
        monitor: DeploymentMonitor,
        require_approval: bool = True
    ):
        self.config = config
        self.validator = validator
        self.monitor = monitor
        self.deployment_config = DeploymentConfig(config.training)
        self.require_approval = require_approval
        
        aiplatform.init(
            project=self.config.gcp_project,
            location=self.config.gcp_region
        )
        
    async def deploy_model(
        self, 
        model_artifacts: Dict,
        deployment_metadata: Optional[Dict] = None
    ) -> str:
        """Handle end-to-end model deployment process with approval checks
        
        Args:
            model_artifacts: Model artifacts to deploy
            deployment_metadata: Optional deployment approval metadata
            
        Raises:
            ValueError: If deployment approval is missing or invalid
            RuntimeError: If deployment fails
        """
        try:
            # Verify deployment approval if required
            if self.require_approval:
                if not deployment_metadata:
                    raise ValueError("Deployment metadata required for approval")
                    
                self._verify_deployment_approval(deployment_metadata)
            
            # Validate model
            if not await self.validator.validate_model(model_artifacts):
                raise ValueError("Model validation failed")
                
            # Create endpoint if needed
            endpoint = await self._get_or_create_endpoint()
            
            # Deploy model to endpoint
            deployment = await self._deploy_to_endpoint(
                endpoint.name,
                model_artifacts,
                deployment_metadata
            )
            
            # Monitor deployment health
            if not await self.monitor.monitor_deployment(deployment.name):
                await self._rollback_deployment(deployment.name)
                raise RuntimeError("Deployment health checks failed")
                
            return deployment.name
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
            
    def _verify_deployment_approval(self, metadata: Dict):
        """Verify deployment approval metadata
        
        Raises:
            ValueError: If approval metadata is invalid
        """
        required_fields = ['approved_at', 'approved_by']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required approval field: {field}")
                
        # Verify approval timestamp is recent (within last 24 hours)
        approved_at = datetime.fromisoformat(metadata['approved_at'])
        if (datetime.now() - approved_at).total_seconds() > 86400:
            raise ValueError("Deployment approval has expired")
        
    async def _get_or_create_endpoint(self) -> aiplatform.Endpoint:
        """Get existing endpoint or create new one"""
        try:
            endpoint = aiplatform.Endpoint.list(
                filter=f'display_name="{self.config.gcp_bucket}-endpoint"'
            )[0]
        except IndexError:
            endpoint = aiplatform.Endpoint.create(
                display_name=f"{self.config.gcp_bucket}-endpoint"
            )
        return endpoint
        
    async def _deploy_to_endpoint(
        self, 
        endpoint_name: str,
        model_artifacts: Dict,
        deployment_metadata: Optional[Dict] = None
    ) -> aiplatform.Model:
        """Deploy model to endpoint"""
        model = aiplatform.Model.upload(
            display_name=f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            artifact_uri=f"gs://{self.config.gcp_bucket}/models/{model_artifacts['artifact_uri']}",
            serving_container_image_uri=model_artifacts.get('serving_image')
        )
        
        model.deploy(
            endpoint=endpoint_name,
            deployed_model_display_name=model.display_name,
            machine_type=self.deployment_config.machine_type,
            min_replica_count=self.deployment_config.min_nodes,
            max_replica_count=self.deployment_config.max_nodes,
            accelerator_type=self.deployment_config.accelerator_type,
            accelerator_count=self.deployment_config.accelerator_count,
            traffic_split=self.deployment_config.traffic_split
        )
        return model
        
    async def _rollback_deployment(self, deployment_name: str):
        """Rollback failed deployment"""
        try:
            deployment = aiplatform.Model(deployment_name)
            await deployment.undeploy_all()
            logger.info(f"Successfully rolled back deployment: {deployment_name}")
        except Exception as e:
            logger.error(f"Rollback failed: {e}") 