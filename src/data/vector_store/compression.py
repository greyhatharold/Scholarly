import numpy as np
from typing import List
import zlib
from dataclasses import dataclass
from src.config.config import VectorStoreConfig

class VectorCompressor:
    """Handles vector compression using quantization and dimensionality reduction"""
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        
    def compress(self, vectors: np.ndarray) -> bytes:
        """Compress vectors using quantization and zlib"""
        if self.config.use_dimensionality_reduction:
            vectors = self._reduce_dimensions(vectors)
            
        quantized = self._quantize(vectors)
        compressed = zlib.compress(quantized.tobytes(), 
                                 level=self.config.compression_level)
        return compressed
    
    def decompress(self, compressed_data: bytes, original_shape: tuple) -> np.ndarray:
        """Decompress vectors back to original format"""
        decompressed = zlib.decompress(compressed_data)
        vectors = np.frombuffer(decompressed, 
                              dtype=f'uint{self.config.quantization_bits}')
        return self._dequantize(vectors.reshape(original_shape))
    
    def _quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize vectors to reduced bit depth"""
        min_val = vectors.min()
        max_val = vectors.max()
        scale = (2**self.config.quantization_bits - 1) / (max_val - min_val)
        return ((vectors - min_val) * scale).astype(f'uint{self.config.quantization_bits}')
    
    def _dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Restore quantized vectors to original scale"""
        scale = (2**self.config.quantization_bits - 1)
        return quantized.astype('float32') / scale 
    
    def _reduce_dimensions(self, vectors: np.ndarray) -> np.ndarray:
        """Reduce vector dimensions using PCA if configured"""
        try:
            from sklearn.decomposition import PCA
            
            # Handle single vector case
            single_vector = len(vectors.shape) == 1 or vectors.shape[0] == 1
            if single_vector:
                return vectors.reshape(1, -1)
            
            # Use vector store dimensions as target
            target_dims = min(vectors.shape[1], self.config.dimensions)
            if target_dims >= vectors.shape[1]:
                return vectors
                
            pca = PCA(n_components=target_dims)
            return pca.fit_transform(vectors)
        except ImportError:
            print("Warning: sklearn not available for dimension reduction")
            return vectors