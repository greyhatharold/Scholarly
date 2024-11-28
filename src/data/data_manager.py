from typing import Dict, List, Optional
import numpy as np
import io
import pickle
import time
import mmh3
from hashlib import sha256
from src.data.vector_store.compression import VectorCompressor
from src.data.vector_store.indexing import VectorIndex
from src.data.storage.storage import CloudStorage
from src.config.config import Config
import asyncio
import os
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """Manages vector data storage, compression, and retrieval in cloud storage"""
    def __init__(self, config: Config):
        logger.debug("Initializing DataManager")
        self.config = config
        self.storage = CloudStorage(config)
        self.compressor = VectorCompressor(config.vector_store)
        self.index = VectorIndex(config.vector_store)
        asyncio.create_task(self._initialize_storage())
        
    async def _initialize_storage(self):
        """Initialize or load index from cloud storage"""
        try:
            # Initialize required files
            required_files = ['vector_index.pkl', 'hash_mappings.pkl']
            for filename in required_files:
                try:
                    await self.storage.load_model_state(filename)
                except FileNotFoundError:
                    logger.debug(f"Creating new {filename}")
                    initial_state = {} if filename == 'hash_mappings.pkl' else {'trained': False, 'vectors': None}
                    await self.storage.save_model_state(initial_state, filename)

            # Load vector index
            logger.debug("Loading vector index from storage")
            index_state = await self.storage.load_model_state('vector_index.pkl')
            self.index.load_state(index_state)
            logger.debug("Successfully loaded vector index")
        except Exception as e:
            logger.warning(f"Error initializing vector store: {e}")
            self._init_index()

    def _init_index(self):
        """Initialize a fresh index instance"""
        logger.debug("Initializing fresh VectorIndex")
        self.index = VectorIndex(self.config.vector_store)

    async def store_vectors(self, vectors: np.ndarray, metadata: Optional[Dict] = None):
        """Store vectors with metadata in cloud storage"""
        try:
            logger.debug(f"Storing vectors with shape {vectors.shape}")
            
            # Validate vector shape
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors)
            
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)
            
            if vectors.shape[1] != self.config.vector_store.dimensions:
                raise ValueError(
                    f"Vector dimensions {vectors.shape[1]} do not match "
                    f"configured dimensions {self.config.vector_store.dimensions}"
                )
            
            # Ensure vector storage directories exist
            vector_dir = os.path.join(self.config.cache_dir, 'vectors')
            os.makedirs(vector_dir, exist_ok=True)
            
            # Initialize hash mappings if needed
            try:
                await self.storage.load_model_state('hash_mappings.pkl')
            except FileNotFoundError:
                logger.debug("Creating new hash mappings file")
                await self.storage.save_model_state({}, 'hash_mappings.pkl')
            
            # Compress and generate hashes
            try:
                compressed = self.compressor.compress(vectors)
                content_hash = self._generate_content_hash(vectors, metadata)
                vector_id = self._generate_vector_id(content_hash)
                logger.debug(f"Generated vector ID: {vector_id}")
            except Exception as e:
                logger.error(f"Error processing vectors: {e}")
                raise
            
            # Store vector data and metadata
            vector_data = {
                'id': vector_id,
                'compressed_vector': compressed,
                'metadata': metadata or {},
                'content_hash': content_hash,
                'timestamp': time.time(),
                'shape': vectors.shape
            }
            
            # Check for duplicates before storing
            if not await self._check_duplicate(content_hash):
                logger.debug("No duplicate found, proceeding with storage")
                
                # Ensure vectors directory exists in storage
                vector_path = f'vectors/{vector_id}.pkl'
                await self.storage.save_model_state(vector_data, vector_path)
                
                # Update index with new vectors
                try:
                    self.index.add_vectors(vectors, np.array([vector_id]))
                    await self._save_index()
                except Exception as index_error:
                    logger.error(f"Error updating index: {index_error}")
                    # Attempt recovery by reinitializing index
                    self._init_index()
                    self.index.add_vectors(vectors, np.array([vector_id]))
                    await self._save_index()
                
                # Update hash mappings
                await self._store_hash_mapping(content_hash, vector_id)
                logger.debug("Successfully stored vectors and updated index")
                
                return vector_id
                
            logger.debug("Duplicate vector detected, skipping storage")
            return None
            
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise

    async def _check_duplicate(self, content_hash: str) -> bool:
        """Check if content hash already exists"""
        try:
            logger.debug(f"Checking for duplicate hash: {content_hash}")
            hash_mapping = await self.storage.load_model_state('hash_mappings.pkl')
            return content_hash in hash_mapping
        except FileNotFoundError:
            return False

    async def _store_hash_mapping(self, content_hash: str, vector_id: int):
        """Store mapping between content hash and vector ID"""
        logger.debug(f"Storing hash mapping for vector ID: {vector_id}")
        try:
            hash_mapping = await self.storage.load_model_state('hash_mappings.pkl')
        except FileNotFoundError:
            hash_mapping = {}
            
        hash_mapping[content_hash] = vector_id
        await self.storage.save_model_state(hash_mapping, 'hash_mappings.pkl')

    def _generate_vector_id(self, content_hash: str) -> int:
        """Generate deterministic vector ID using MurmurHash3"""
        # Use mmh3 for fast hashing
        vector_id = mmh3.hash64(content_hash.encode())[0]
        logger.debug(f"Generated vector ID {vector_id} from hash {content_hash}")
        return vector_id

    def _generate_content_hash(self, vectors: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """Generate content hash from vector data and metadata"""
        logger.debug("Generating content hash")
        # Create a buffer for the vector data
        vector_buffer = io.BytesIO()
        np.save(vector_buffer, vectors)
        vector_bytes = vector_buffer.getvalue()
        
        # Combine vector data with metadata
        metadata_str = str(sorted(metadata.items())) if metadata else ""
        combined = vector_bytes + metadata_str.encode()
        
        # Generate SHA-256 hash for content verification
        content_hash = sha256(combined).hexdigest()
        logger.debug(f"Generated content hash: {content_hash}")
        return content_hash

    async def search_similar(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search for similar vectors in cloud storage"""
        logger.debug(f"Searching for {k} similar vectors")
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        
        for idx, dist in zip(indices[0], distances[0]):
            try:
                logger.debug(f"Loading vector data for index {idx}")
                vector_data = await self.storage.load_model_state(
                    f'vectors/{int(idx)}.pkl'
                )
                # Verify content hash
                if self._verify_vector_data(vector_data):
                    results.append({
                        'id': vector_data['id'],
                        'distance': float(dist),
                        'metadata': vector_data['metadata'],
                        'content_hash': vector_data['content_hash']
                    })
                    logger.debug(f"Added result with distance {dist}")
            except FileNotFoundError:
                logger.warning(f"Vector data not found for index {idx}")
                continue
                
        logger.debug(f"Found {len(results)} similar vectors")
        return results

    def _verify_vector_data(self, vector_data: Dict) -> bool:
        """Verify vector data integrity using stored hash"""
        logger.debug(f"Verifying vector data for ID: {vector_data.get('id')}")
        if 'content_hash' not in vector_data:
            logger.debug("No content hash found, skipping verification")
            return True  # Skip verification for legacy data
            
        try:
            # Use stored shape for decompression if available
            shape = vector_data.get('shape', (-1,))
            vectors = self.compressor.decompress(vector_data['compressed_vector'], shape)
            computed_hash = self._generate_content_hash(vectors, vector_data['metadata'])
            is_valid = computed_hash == vector_data['content_hash']
            logger.debug(f"Vector data verification {'successful' if is_valid else 'failed'}")
            return is_valid
        except Exception as e:
            logger.error(f"Error verifying vector data: {e}")
            return False

    async def _save_index(self):
        """Save index state to cloud storage"""
        logger.debug("Saving index state to storage")
        index_state = self.index.get_state()
        await self.storage.save_model_state(index_state, 'vector_index.pkl')
        logger.debug("Successfully saved index state")