import numpy as np
from sklearn.decomposition import PCA
from typing import Optional
from src.config.config import EmbeddingConfig

class VectorOptimizer:
    """Optimizes vector representations for storage and retrieval"""
    def __init__(self, config: EmbeddingConfig):
        self.target_dims = config.target_dims
        self.pca = None
        
    def fit(self, vectors: np.ndarray):
        """Fit the optimizer to a set of vectors"""
        if self.target_dims and self.target_dims < vectors.shape[1]:
            self.pca = PCA(n_components=self.target_dims)
            self.pca.fit(vectors)
    
    def optimize(self, vectors: np.ndarray) -> np.ndarray:
        """Optimize vectors for storage"""
        if self.pca is not None:
            vectors = self.pca.transform(vectors)
        return self._normalize(vectors)
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms 