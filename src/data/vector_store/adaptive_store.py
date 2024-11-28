import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config.config import VectorStoreConfig
from src.data.vector_store.compression import VectorCompressor
from src.data.vector_store.indexing import VectorIndex

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


class MetaLearningStrategy(ABC):
    @abstractmethod
    def learn_pattern(self, query: np.ndarray, results: Dict) -> None:
        """Learn from successful search patterns"""
        pass

    @abstractmethod
    def adapt_search(self, query: np.ndarray) -> Dict:
        """Adapt search strategy based on learned patterns"""
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
                "quantization_bits": optimal_bits,
                "compression_level": compression_level,
            }
        except Exception as e:
            logger.error(f"Error determining compression: {e}")
            # Return default values if optimization fails
            return {
                "quantization_bits": self.config.quantization_bits,
                "compression_level": self.config.compression_level,
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
                self.index.config.n_lists = min(1000, self.index.config.n_lists * 2)
            elif cluster_density < 0.2:  # Low density
                self.index.config.n_lists = max(10, self.index.config.n_lists // 2)

            # Adjust probe count based on query patterns
            if patterns.get("query_latency", 0) > patterns.get("target_latency", 0.1):
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

            return {"connections": connections, "similarity_matrix": similarities}
        except Exception as e:
            logger.error(f"Error discovering connections: {e}")
            return {"connections": [], "similarity_matrix": None}

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
            current_dt = temporal_stats.get("dt", self.config.dt)

            # Adjust dt based on complexity
            optimal_dt = self._adjust_time_constant(complexity, current_dt)

            return {
                "dt": optimal_dt,
                "complexity": complexity,
                "stability_score": self._compute_stability(data),
            }

        except Exception as e:
            logger.error(f"Error adapting temporal dynamics: {e}")
            return {"dt": self.config.dt}

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


class AdaptiveMetaLearning(MetaLearningStrategy):
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.meta_patterns = {}
        self.pattern_weights = {}

    def learn_pattern(self, query: np.ndarray, results: Dict) -> None:
        """Learn from successful search results"""
        try:
            pattern_signature = self._compute_pattern_signature(query)
            success_score = self._evaluate_search_success(results)

            if pattern_signature not in self.meta_patterns:
                self.meta_patterns[pattern_signature] = {
                    "query_prototype": query.copy(),
                    "success_count": 0,
                    "adaptation_weights": np.ones(query.shape[-1]),
                }

            # Update pattern statistics
            pattern = self.meta_patterns[pattern_signature]
            pattern["success_count"] += 1
            pattern["adaptation_weights"] *= 1 - self.config.learning_rate
            pattern["adaptation_weights"] += self.config.learning_rate * success_score

        except Exception as e:
            logger.error(f"Error learning pattern: {e}")

    def adapt_search(self, query: np.ndarray) -> Dict:
        """Adapt search strategy using learned patterns"""
        try:
            similar_patterns = self._find_similar_patterns(query)
            if not similar_patterns:
                return {"weights": np.ones_like(query)}

            # Blend successful search strategies
            adapted_weights = self._blend_strategies(query, similar_patterns)
            return {"weights": adapted_weights, "pattern_count": len(similar_patterns)}

        except Exception as e:
            logger.error(f"Error adapting search: {e}")
            return {"weights": np.ones_like(query)}

    def _compute_pattern_signature(self, query: np.ndarray) -> str:
        """Compute unique signature for search pattern"""
        # Use statistical properties for signature
        stats = [float(np.mean(query)), float(np.std(query)), float(np.median(query))]
        return ";".join(f"{s:.4f}" for s in stats)

    def _evaluate_search_success(self, results: Dict) -> float:
        """Evaluate search result success"""
        if not results:
            return 0.0
        # Consider factors like result count, relevance scores
        relevance_scores = [r.get("distance", 1.0) for r in results]
        return float(np.mean(relevance_scores))

    def _find_similar_patterns(self, query: np.ndarray) -> List[Dict]:
        """Find patterns similar to current query"""
        similar_patterns = []
        for signature, pattern in self.meta_patterns.items():
            similarity = self._compute_similarity(query, pattern["query_prototype"])
            if similarity > self.config.pattern_similarity_threshold:
                similar_patterns.append({"pattern": pattern, "similarity": similarity})
        return similar_patterns

    def _blend_strategies(self, query: np.ndarray, patterns: List[Dict]) -> np.ndarray:
        """Blend multiple search strategies"""
        if not patterns:
            return np.ones_like(query)

        weights = np.zeros_like(query)
        total_similarity = sum(p["similarity"] for p in patterns)

        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            similarity = pattern_info["similarity"]
            weights += (similarity / total_similarity) * pattern["adaptation_weights"]

        return weights / len(patterns)

    def _compute_similarity(self, query: np.ndarray, prototype: np.ndarray) -> float:
        """Compute similarity between query and pattern prototype"""
        return float(np.dot(query, prototype) / (np.linalg.norm(query) * np.linalg.norm(prototype)))


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
        self.meta_learning = AdaptiveMetaLearning(config)

        self.connection_strengths = {}  # Track connection strengths
        self.restructure_threshold = 0.8  # Threshold for triggering restructure

    async def process_new_information(
        self, input_data: np.ndarray, patterns: Optional[Dict] = None
    ) -> Dict:
        """Process and store new vector information with adaptive optimization"""
        try:
            logger.debug("Processing new information batch")

            # Optimize compression
            compression_params = self.compression_strategy.determine_optimal_compression(input_data)
            self.compressor.config.quantization_bits = compression_params["quantization_bits"]
            self.compressor.config.compression_level = compression_params["compression_level"]

            # Adapt index structure
            self.indexing_strategy.adapt_structure(input_data, patterns or {})

            # Discover and strengthen connections
            connections = self.connection_discovery.discover_connections(input_data)
            await self._strengthen_connections(connections["connections"])

            # Check if restructuring is needed
            if self._should_restructure(connections):
                await self._restructure_storage()

            # Store processed data
            compressed_data = self.compressor.compress(input_data)
            self.index.add_vectors(input_data, np.arange(len(input_data)))

            return {
                "compression_params": compression_params,
                "connections": connections,
                "compressed_size": len(compressed_data),
                "original_size": input_data.nbytes,
                "storage_restructured": self._should_restructure(connections),
            }

        except Exception as e:
            logger.error(f"Error processing new information: {e}")
            raise

    async def _strengthen_connections(self, connections: List[Tuple[int, int]]) -> None:
        """Strengthen discovered connections between vectors"""
        for src, dst in connections:
            key = tuple(sorted([src, dst]))  # Ensure consistent ordering
            self.connection_strengths[key] = self.connection_strengths.get(key, 0) + 1

    def _should_restructure(self, connections: Dict) -> bool:
        """Determine if storage restructuring is needed"""
        if not connections.get("similarity_matrix") is not None:
            return False

        # Check connection density and strength
        total_possible = len(self.connection_strengths) * (len(self.connection_strengths) - 1) / 2
        if total_possible == 0:
            return False

        connection_density = len(connections["connections"]) / total_possible
        return connection_density > self.restructure_threshold

    async def _restructure_storage(self) -> None:
        """Restructure storage based on connection patterns"""
        try:
            # Adjust index structure based on connection strengths
            strong_connections = {
                k: v
                for k, v in self.connection_strengths.items()
                if v > np.mean(list(self.connection_strengths.values()))
            }

            if strong_connections:
                # Update index clustering parameters
                self.index.config.n_lists = max(10, min(1000, len(strong_connections) // 2))

                # Reinitialize index with new structure
                self.index._init_index()

                logger.debug(
                    f"Restructured storage with {len(strong_connections)} strong connections"
                )

        except Exception as e:
            logger.error(f"Error restructuring storage: {e}")

    def get_optimization_stats(self) -> Dict:
        """Get current optimization statistics"""
        return {
            "compression_ratio": self.compressor.config.compression_level,
            "quantization_bits": self.compressor.config.quantization_bits,
            "index_clusters": self.index.config.n_lists,
            "probe_count": self.index.config.n_probes,
        }

    async def strengthen_connection_pattern(self, pattern_data: Dict[str, float]) -> None:
        """Strengthen specific connection patterns based on model feedback

        Args:
            pattern_data: Dictionary mapping connection keys to strength scores
        """
        try:
            logger.debug("Strengthening connection patterns")
            for connection_key, strength in pattern_data.items():
                if isinstance(connection_key, str):
                    connection_key = eval(connection_key)  # Convert string tuple to actual tuple
                if isinstance(connection_key, tuple) and len(connection_key) == 2:
                    # Update connection strength with exponential moving average
                    current = self.connection_strengths.get(connection_key, 0)
                    self.connection_strengths[connection_key] = current * 0.7 + strength * 0.3

            # Check if restructuring is needed after strengthening
            if any(
                strength > self.restructure_threshold
                for strength in self.connection_strengths.values()
            ):
                await self._restructure_storage()

        except Exception as e:
            logger.error(f"Error strengthening connections: {e}")

    async def learn_from_search(self, query: np.ndarray, results: Dict) -> None:
        """Learn from successful search results"""
        try:
            logger.debug("Learning from search results")
            self.meta_learning.learn_pattern(query, results)
        except Exception as e:
            logger.error(f"Error learning from search: {e}")

    async def adapt_search(self, query: np.ndarray) -> Dict:
        """Adapt search strategy using learned patterns"""
        try:
            logger.debug("Adapting search strategy")
            adapted_weights = self.meta_learning.adapt_search(query)
            return adapted_weights
        except Exception as e:
            logger.error(f"Error adapting search: {e}")
            return {"weights": np.ones_like(query)}
