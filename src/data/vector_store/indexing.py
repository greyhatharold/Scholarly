import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import faiss
from src.config.config import VectorStoreConfig
import logging

logger = logging.getLogger(__name__)

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
        """Add vectors to the index
        
        Args:
            vectors: Input vectors of shape (n_vectors, n_dimensions)
            ids: Vector IDs of shape (n_vectors,)
        """
        try:
            # Ensure vectors are 2D and properly shaped
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            
            if vectors.shape[1] != self.config.dimensions:
                raise ValueError(f"Vector dimensions {vectors.shape[1]} do not match index dimensions {self.config.dimensions}")
            
            # Ensure ids are 1D
            ids = np.asarray(ids).reshape(-1)
            if len(ids) != vectors.shape[0]:
                raise ValueError(f"Number of IDs ({len(ids)}) does not match number of vectors ({vectors.shape[0]})")

            # For IVF index, ensure it's trained first
            if isinstance(self.index, faiss.IndexIVFFlat):
                if not self.index.is_trained:
                    if vectors.shape[0] < self.config.n_lists:
                        # Not enough vectors for IVF training, create and use flat index
                        logger.debug("Using flat index due to insufficient training data")
                        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.config.dimensions))
                    else:
                        logger.debug("Training IVF index")
                        self.index.train(vectors)
                        # Wrap trained index with IDMap
                        self.index = faiss.IndexIDMap(self.index)
                else:
                    # Wrap existing trained index if needed
                    if not isinstance(self.index, faiss.IndexIDMap):
                        self.index = faiss.IndexIDMap(self.index)
            else:
                # For flat index, always wrap with IDMap
                if not isinstance(self.index, faiss.IndexIDMap):
                    self.index = faiss.IndexIDMap(self.index)
                
            self.index.add_with_ids(vectors, ids)
            logger.debug(f"Successfully added {len(ids)} vectors to index")
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {e}")
            # Final fallback to flat index with IDMap
            try:
                logger.debug("Attempting fallback to flat index with IDMap")
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.config.dimensions))
                self.index.add_with_ids(vectors, ids)
                logger.debug("Successfully added vectors using flat index fallback")
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                raise
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors"""
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = self.config.n_probes
        distances, indices = self.index.search(query_vector, k)
        return distances, indices 
    
    def get_state(self) -> dict:
        """Get serializable state of the index
        
        Returns:
            dict: Index state containing trained status and vector data
        """
        if not self.index.is_trained:
            return {'trained': False, 'vectors': None}
            
        # Get index data
        index_data = faiss.serialize_index(self.index)
        return {
            'trained': True,
            'vectors': index_data
        }
    
    def load_state(self, state: dict) -> None:
        """Load index state from serialized data
        
        Args:
            state: Dictionary containing index state
        """
        if not state.get('trained', False):
            self._init_index()
            return
            
        if state.get('vectors') is not None:
            self.index = faiss.deserialize_index(state['vectors'])
    
    def get_density_map(self) -> np.ndarray:
        """Calculate density map of current vector space
        
        Returns:
            np.ndarray: Density values across vector space
        """
        if not hasattr(self.index, 'ntotal') or self.index.ntotal == 0:
            return np.ones(1)
            
        # Use FAISS index structure to estimate density
        if isinstance(self.index, faiss.IndexIVFFlat):
            # Get cluster sizes for IVF index
            _, cluster_sizes = self.index.get_list_sizes()
            return np.array(cluster_sizes, dtype=np.float32)
        else:
            # For flat index, use simple distance-based density
            vectors = faiss.vector_to_array(self.index.get_xb())
            vectors = vectors.reshape(-1, self.config.dimensions)
            
            # Compute approximate density using random sampling
            sample_size = min(1000, vectors.shape[0])
            if sample_size < vectors.shape[0]:
                idx = np.random.choice(vectors.shape[0], sample_size, replace=False)
                vectors = vectors[idx]
                
            distances = faiss.pairwise_distances(vectors)
            density = np.mean(distances, axis=1)
            return density