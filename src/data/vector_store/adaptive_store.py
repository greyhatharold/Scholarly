from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from src.data.vector_store.indexing import VectorIndex 
from src.data.vector_store.compression import VectorCompressor
from src.config.config import VectorStoreConfig

logger = logging.getLogger(__name__)

# First, let's define interfaces following Interface Segregation Principle
class CompressionStrategy(ABC):
    @abstractmethod
    def determine_optimal_compression(self, data: np.ndarray) -> Dict:
        """Determine optimal compression parameters for given data"""
        pass

class IndexingStrategy(ABC):
    @abstractmethod 
    def adapt_structure(self, data: np.ndarray, patterns: Dict) -> None:
        """Adapt index structure based on data patterns"""
        pass

class ConnectionDiscovery(ABC):
    @abstractmethod
    def discover_connections(self, data: np.ndarray) -> Dict:
        """Discover connections in vector data"""
        pass

class TemporalDynamicsStrategy(ABC):
    @abstractmethod
    def adapt_dynamics(self, data: np.ndarray, temporal_stats: Dict) -> Dict:
        """Adapt temporal processing parameters"""
        pass

# Concrete implementations
class DynamicCompressionStrategy(CompressionStrategy):
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        
    def determine_optimal_compression(self, data: np.ndarray) -> Dict:
        """Analyze data distribution to determine optimal compression"""
        try:
            # Calculate data statistics
            data_variance = np.var(data)
            data_range = np.ptp(data)
            
            # Adjust compression parameters based on data characteristics
            optimal_bits = max(4, min(16, int(np.log2(data_range / data_variance))))
            compression_level = min(9, max(1, int(data_variance * 10)))
            
            return {
                'quantization_bits': optimal_bits,
                'compression_level': compression_level
            }
        except Exception as e:
            logger.error(f"Error determining compression: {e}")
            # Return default values if optimization fails
            return {
                'quantization_bits': self.config.quantization_bits,
                'compression_level': self.config.compression_level
            }

class AdaptiveIndexingStrategy(IndexingStrategy):
    def __init__(self, index: VectorIndex):
        self.index = index
        
    def adapt_structure(self, data: np.ndarray, patterns: Dict) -> None:
        """Adapt index structure based on data patterns"""
        try:
            # Analyze data distribution
            cluster_density = self._analyze_cluster_density(data)
            
            # Adjust number of clusters if needed
            if cluster_density > 0.8:  # High density
                self.index.config.n_lists = min(
                    1000, 
                    self.index.config.n_lists * 2
                )
            elif cluster_density < 0.2:  # Low density
                self.index.config.n_lists = max(
                    10,
                    self.index.config.n_lists // 2
                )
                
            # Adjust probe count based on query patterns
            if patterns.get('query_latency', 0) > patterns.get('target_latency', 0.1):
                self.index.config.n_probes = max(1, self.index.config.n_probes - 1)
            else:
                self.index.config.n_probes = min(20, self.index.config.n_probes + 1)
                
        except Exception as e:
            logger.error(f"Error adapting index structure: {e}")
    
    def _analyze_cluster_density(self, data: np.ndarray) -> float:
        """Calculate cluster density metric"""
        if len(data) < 2:
            return 0.0
        distances = np.linalg.norm(data[1:] - data[:-1], axis=1)
        return float(np.mean(distances < np.median(distances)))

class PatternBasedConnectionDiscovery(ConnectionDiscovery):
    def discover_connections(self, data: np.ndarray) -> Dict:
        """Discover connections between vectors using similarity patterns"""
        try:
            # Calculate pairwise similarities
            similarities = self._compute_similarities(data)
            
            # Find connected components
            connections = self._find_connections(similarities)
            
            return {
                'connections': connections,
                'similarity_matrix': similarities
            }
        except Exception as e:
            logger.error(f"Error discovering connections: {e}")
            return {'connections': [], 'similarity_matrix': None}
            
    def _compute_similarities(self, data: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarities"""
        normalized = data / np.linalg.norm(data, axis=1)[:, np.newaxis]
        return normalized @ normalized.T
        
    def _find_connections(self, similarities: np.ndarray) -> list:
        """Find strongly connected vector pairs"""
        threshold = 0.8  # Similarity threshold
        connections = []
        indices = np.where(similarities > threshold)
        for i, j in zip(*indices):
            if i < j:  # Avoid duplicates
                connections.append((int(i), int(j)))
        return connections

class LiquidTimeStrategy(TemporalDynamicsStrategy):
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.min_dt = config.min_dt
        self.max_dt = config.max_dt
        
    def adapt_dynamics(self, data: np.ndarray, temporal_stats: Dict) -> Dict:
        """Adjust temporal parameters based on data characteristics"""
        try:
            # Calculate temporal complexity
            complexity = self._compute_complexity(data)
            current_dt = temporal_stats.get('dt', self.config.dt)
            
            # Adjust dt based on complexity
            optimal_dt = self._adjust_time_constant(
                complexity,
                current_dt
            )
            
            return {
                'dt': optimal_dt,
                'complexity': complexity,
                'stability_score': self._compute_stability(data)
            }
            
        except Exception as e:
            logger.error(f"Error adapting temporal dynamics: {e}")
            return {'dt': self.config.dt}
            
    def _compute_complexity(self, data: np.ndarray) -> float:
        """Compute temporal complexity metric"""
        # Use gradient approximation as complexity proxy
        try:
            gradients = np.gradient(data, axis=0)
            return float(np.mean(np.linalg.norm(gradients, axis=1)))
        except Exception:
            return 0.0
            
    def _adjust_time_constant(self, complexity: float, current_dt: float) -> float:
        """Adjust time constant based on complexity"""
        adjustment = np.clip(1.0 / (1.0 + complexity), 0.1, 0.9)
        new_dt = current_dt * adjustment
        return float(np.clip(new_dt, self.min_dt, self.max_dt))

class AdaptiveVectorStore:
    """Manages adaptive vector storage with dynamic optimization"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = VectorIndex(config)
        self.compressor = VectorCompressor(config)
        
        # Initialize strategies following Dependency Inversion Principle
        self.compression_strategy = DynamicCompressionStrategy(config)
        self.indexing_strategy = AdaptiveIndexingStrategy(self.index)
        self.connection_discovery = PatternBasedConnectionDiscovery()
        self.temporal_dynamics = LiquidTimeStrategy(config)
        
    async def process_new_information(self, 
                                    input_data: np.ndarray,
                                    patterns: Optional[Dict] = None) -> Dict:
        """Process and store new vector information with adaptive optimization"""
        try:
            logger.debug("Processing new information batch")
            
            # Optimize compression
            compression_params = self.compression_strategy.determine_optimal_compression(
                input_data
            )
            self.compressor.config.quantization_bits = compression_params['quantization_bits']
            self.compressor.config.compression_level = compression_params['compression_level']
            
            # Adapt index structure
            self.indexing_strategy.adapt_structure(
                input_data, 
                patterns or {}
            )
            
            # Discover connections
            connections = self.connection_discovery.discover_connections(input_data)
            
            # Store processed data
            compressed_data = self.compressor.compress(input_data)
            self.index.add_vectors(input_data, np.arange(len(input_data)))
            
            return {
                'compression_params': compression_params,
                'connections': connections,
                'compressed_size': len(compressed_data),
                'original_size': input_data.nbytes
            }
            
        except Exception as e:
            logger.error(f"Error processing new information: {e}")
            raise

    def get_optimization_stats(self) -> Dict:
        """Get current optimization statistics"""
        return {
            'compression_ratio': self.compressor.config.compression_level,
            'quantization_bits': self.compressor.config.quantization_bits,
            'index_clusters': self.index.config.n_lists,
            'probe_count': self.index.config.n_probes
        } 