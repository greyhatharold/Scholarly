import boto3
from datetime import datetime
import os
from typing import Dict
class AWSTrainer:
    """Handles model training on AWS infrastructure"""
    def __init__(self, config):
        self.config = config
        self.sagemaker = boto3.client('sagemaker')
        
    async def prepare_training_job(self, model_data: Dict) -> str:
        """Prepares and launches a SageMaker training job"""
        job_name = self._generate_job_name()
        training_params = self._prepare_training_params(job_name)
        
        self.sagemaker.create_training_job(**training_params)
        return job_name
    
    def _generate_job_name(self) -> str:
        """Generates a unique training job name"""
        return f"scholarai-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    def _prepare_training_params(self, job_name: str) -> Dict:
        """Prepares SageMaker training job parameters"""
        return {
            'JobName': job_name,
            'AlgorithmSpecification': {
                'TrainingImage': self.get_training_image(),
                'TrainingInputMode': 'File'
            },
            'RoleArn': os.environ.get('SAGEMAKER_ROLE_ARN'),
            'InputDataConfig': self._get_input_data_config(),
            'OutputDataConfig': {
                'S3OutputPath': f's3://{self.config.aws_bucket}/training_output'
            },
            'ResourceConfig': self._get_resource_config(),
            'StoppingCondition': {'MaxRuntimeInSeconds': 86400},
            'HyperParameters': self._get_hyperparameters()
        } 
    
    def _get_hyperparameters(self) -> Dict:
        """Prepares hyperparameters for liquid neural network training"""
        return {
            'num_liquid_layers': str(self.config.num_liquid_layers),
            'dt': str(self.config.dt),
            'integration_steps': str(self.config.integration_steps),
            'solver': self.config.solver,
            'learning_rate': str(self.config.learning_rate),
            'batch_size': str(self.config.batch_size)
        }