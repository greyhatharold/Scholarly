from datetime import datetime
from typing import Dict, Optional, List, Any, Callable, TYPE_CHECKING
import logging
from dataclasses import dataclass
from src.config.config import Config
from src.data.storage.storage import CloudStorage

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Represents a single model version"""
    version: str
    timestamp: datetime
    filename: str
    metadata: Dict
    metrics: Optional[Dict] = None
    
class ModelVersionRegistry:
    """Manages model version metadata and tracking"""
    def __init__(self, storage: CloudStorage):
        self.storage = storage
        self._registry: Dict[str, ModelVersion] = {}
        
    async def initialize(self):
        """Load existing version registry from storage"""
        try:
            registry_data = await self.storage.load_model_state('model_registry.json')
            for version_data in registry_data.values():
                self._registry[version_data['version']] = ModelVersion(**version_data)
        except FileNotFoundError:
            logger.info("No existing model registry found, creating new one")
            await self._save_registry()
            
    async def _save_registry(self):
        """Save registry state to storage"""
        registry_data = {
            version: {
                'version': mv.version,
                'timestamp': mv.timestamp.isoformat(),
                'filename': mv.filename,
                'metadata': mv.metadata,
                'metrics': mv.metrics
            }
            for version, mv in self._registry.items()
        }
        await self.storage.save_model_state(registry_data, 'model_registry.json')
        
    async def register_version(self, version: ModelVersion):
        """Register a new model version"""
        self._registry[version.version] = version
        await self._save_registry()
        
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get version metadata by version string"""
        return self._registry.get(version)
        
    def list_versions(self) -> List[ModelVersion]:
        """List all registered model versions"""
        return sorted(
            self._registry.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )

class ModelVersionManager:
    """Manages model versioning and storage"""
    def __init__(self, config: Config, model_factory: Optional[Callable[[Config], Any]] = None):
        self.config = config
        self.storage = CloudStorage(config)
        self.registry = ModelVersionRegistry(self.storage)
        self.model_factory = model_factory
        
    async def initialize(self):
        """Initialize version manager"""
        await self.registry.initialize()
        
    async def save_version(
        self,
        model: Any,
        version: str,
        metadata: Dict,
        metrics: Optional[Dict] = None
    ) -> ModelVersion:
        """Save a new model version
        
        Args:
            model: Model instance to save
            version: Version string (e.g. "1.0.0")
            metadata: Version metadata
            metrics: Optional performance metrics
            
        Returns:
            ModelVersion: Created version object
        """
        # Generate version filename
        timestamp = datetime.now()
        filename = f"model_v{version}_{timestamp.strftime('%Y%m%d_%H%M%S')}.pt"
        
        # Save model state
        await self.storage.save_model_state(model.state_dict(), filename)
        
        # Create and register version
        model_version = ModelVersion(
            version=version,
            timestamp=timestamp,
            filename=filename,
            metadata=metadata,
            metrics=metrics
        )
        await self.registry.register_version(model_version)
        
        return model_version
        
    async def load_version(
        self,
        version: str,
        model: Optional[Any] = None
    ) -> Any:
        """Load a specific model version
        
        Args:
            version: Version string to load
            model: Optional existing model instance to load into
            
        Returns:
            ScholarAI: Loaded model instance
        """
        # Get version metadata
        version_info = self.registry.get_version(version)
        if not version_info:
            raise ValueError(f"Version {version} not found")
            
        # Create new model if needed
        if model is None:
            if self.model_factory is None:
                raise ValueError("No model factory provided and no existing model instance")
            model = self.model_factory(self.config)
            
        # Load model state
        state_dict = await self.storage.load_model_state(version_info.filename)
        model.load_state_dict(state_dict)
        
        return model
        
    async def get_latest_version(self) -> Optional[ModelVersion]:
        """Get the latest model version"""
        versions = self.registry.list_versions()
        return versions[0] if versions else None
        
    async def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> Dict:
        """Compare metrics between two versions
        
        Args:
            version_a: First version to compare
            version_b: Second version to compare
            
        Returns:
            Dict: Comparison results
        """
        version_a_info = self.registry.get_version(version_a)
        version_b_info = self.registry.get_version(version_b)
        
        if not version_a_info or not version_b_info:
            raise ValueError("One or both versions not found")
            
        if not version_a_info.metrics or not version_b_info.metrics:
            raise ValueError("Metrics not available for comparison")
            
        # Compare metrics
        comparison = {
            'version_a': version_a_info.version,
            'version_b': version_b_info.version,
            'metrics_diff': {}
        }
        
        # Calculate metric differences
        for metric in version_a_info.metrics:
            if metric in version_b_info.metrics:
                comparison['metrics_diff'][metric] = {
                    'value_a': version_a_info.metrics[metric],
                    'value_b': version_b_info.metrics[metric],
                    'diff': version_b_info.metrics[metric] - version_a_info.metrics[metric]
                }
                
        return comparison 

    async def save_version_with_vectors(
        self,
        model: Any,
        version: str,
        vector_store_id: int,
        metadata: Dict,
        metrics: Optional[Dict] = None
    ) -> ModelVersion:
        """Save model version with associated vector store state
        
        Args:
            model: Model instance
            version: Version string
            vector_store_id: ID of vector store version
            metadata: Version metadata
            metrics: Optional metrics
        """
        # Save model version
        model_version = await self.save_version(
            model=model,
            version=version,
            metadata={
                **metadata,
                'vector_store_version': vector_store_id
            },
            metrics=metrics
        )
        
        # Link vector store version
        model_version.metadata['vector_store_id'] = vector_store_id
        await self.registry.register_version(model_version)
        
        return model_version

    async def load_version_with_vectors(
        self,
        version: str,
        model: Optional[Any] = None,
        data_manager: Optional[Any] = None
    ) -> Any:
        """Load model version and restore associated vector store
        
        Args:
            version: Version to load
            model: Optional existing model
            data_manager: Optional data manager instance
        """
        # Load model version
        model = await self.load_version(version, model)
        
        # Restore vector store if provided
        if data_manager is not None:
            version_info = self.registry.get_version(version)
            vector_store_id = version_info.metadata.get('vector_store_id')
            if vector_store_id:
                await data_manager.restore_vector_store_version(vector_store_id)
                
        return model