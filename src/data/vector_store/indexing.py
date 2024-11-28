import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import faiss
from src.config.config import VectorStoreConfig

class VectorIndex:
    """Manages high-dimensional vector indexing using FAISS"""
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index based on configuration"""
        quantizer = faiss.IndexFlatL2(self.config.dimensions)
        if self.config.index_type == 'IVF':
            self.index = faiss.IndexIVFFlat(quantizer, 
                                          self.config.dimensions,
                                          self.config.n_lists)
        else:
            self.index = faiss.IndexFlatL2(self.config.dimensions)
    
    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray):
        """Add vectors to the index"""
        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add_with_ids(vectors, ids)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors"""
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = self.config.n_probes
        distances, indices = self.index.search(query_vector, k)
        return distances, indices 