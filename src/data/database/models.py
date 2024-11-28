from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
from datetime import datetime
from src.data.vector_store.adaptive_store import AdaptiveVectorStore

@dataclass
class VectorEntry:
    """Represents a vector entry in the database"""
    id: Optional[int]
    vector_data: np.ndarray
    metadata: Dict
    created_at: datetime
    
    @classmethod
    def from_row(cls, row: Dict):
        """Create VectorEntry from database row"""
        return cls(
            id=row['id'],
            vector_data=np.frombuffer(row['vector_data']),
            metadata=row['metadata'],
            created_at=datetime.fromisoformat(row['created_at'])
        )

@dataclass
class VectorBatch:
    """Represents a batch of vectors for bulk operations"""
    vectors: np.ndarray
    metadata: List[Dict]
    created_at: datetime = datetime.now() 

@dataclass
class VectorStoreState:
    """Represents the state of the vector store"""
    version_id: int
    compression_params: Dict
    index_state: Dict
    metadata: Optional[Dict] = None
    created_at: datetime = datetime.now()
    
    @classmethod
    def from_adaptive_store(cls, store: AdaptiveVectorStore, version_id: int):
        """Create state from AdaptiveVectorStore instance"""
        return cls(
            version_id=version_id,
            compression_params=store.get_optimization_stats(),
            index_state=store.index.get_state(),
            metadata={
                'dimensions': store.config.dimensions,
                'index_type': store.config.index_type
            }
        ) 