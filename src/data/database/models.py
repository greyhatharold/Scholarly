from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
from datetime import datetime

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