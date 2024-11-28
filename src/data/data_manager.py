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

class DataManager:
    """Manages vector data storage, compression, and retrieval in cloud storage"""
    def __init__(self, config: Config):
        self.config = config
        self.storage = CloudStorage(config)
        self.compressor = VectorCompressor(config.vector_store)
        self.index = VectorIndex(config.vector_store)
        self._initialize_storage()
        
    async def _initialize_storage(self):
        """Initialize or load index from cloud storage"""
        try:
            index_state = await self.storage.load_model_state('vector_index.pkl')
            self.index.load_state(index_state)
        except FileNotFoundError:
            # New index will be created
            pass

    async def store_vectors(self, vectors: np.ndarray, metadata: Optional[Dict] = None):
        """Store vectors with metadata in cloud storage"""
        compressed = self.compressor.compress(vectors)
        
        # Generate content hash using vector data and metadata
        content_hash = self._generate_content_hash(vectors, metadata)
        vector_id = self._generate_vector_id(content_hash)
        
        # Store vector data and metadata
        vector_data = {
            'id': vector_id,
            'compressed_vector': compressed,
            'metadata': metadata,
            'content_hash': content_hash,
            'timestamp': time.time()
        }
        
        # Check for duplicates before storing
        if not await self._check_duplicate(content_hash):
            # Save to cloud storage
            await self.storage.save_model_state(
                vector_data,
                f'vectors/{vector_id}.pkl'
            )
            
            # Update index
            self.index.add_vectors(vectors, np.array([vector_id]))
            await self._save_index()
            
            # Store hash mapping
            await self._store_hash_mapping(content_hash, vector_id)
            
            return vector_id
        return None

    async def _check_duplicate(self, content_hash: str) -> bool:
        """Check if content hash already exists"""
        try:
            hash_mapping = await self.storage.load_model_state('hash_mappings.pkl')
            return content_hash in hash_mapping
        except FileNotFoundError:
            return False

    async def _store_hash_mapping(self, content_hash: str, vector_id: int):
        """Store mapping between content hash and vector ID"""
        try:
            hash_mapping = await self.storage.load_model_state('hash_mappings.pkl')
        except FileNotFoundError:
            hash_mapping = {}
            
        hash_mapping[content_hash] = vector_id
        await self.storage.save_model_state(hash_mapping, 'hash_mappings.pkl')

    def _generate_vector_id(self, content_hash: str) -> int:
        """Generate deterministic vector ID using MurmurHash3"""
        # Use mmh3 for fast hashing
        return mmh3.hash64(content_hash.encode())[0]

    def _generate_content_hash(self, vectors: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """Generate content hash from vector data and metadata"""
        # Create a buffer for the vector data
        vector_buffer = io.BytesIO()
        np.save(vector_buffer, vectors)
        vector_bytes = vector_buffer.getvalue()
        
        # Combine vector data with metadata
        metadata_str = str(sorted(metadata.items())) if metadata else ""
        combined = vector_bytes + metadata_str.encode()
        
        # Generate SHA-256 hash for content verification
        return sha256(combined).hexdigest()

    async def search_similar(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search for similar vectors in cloud storage"""
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        
        for idx, dist in zip(indices[0], distances[0]):
            try:
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
            except FileNotFoundError:
                continue
                
        return results

    def _verify_vector_data(self, vector_data: Dict) -> bool:
        """Verify vector data integrity using stored hash"""
        if 'content_hash' not in vector_data:
            return True  # Skip verification for legacy data
            
        try:
            vectors = self.compressor.decompress(vector_data['compressed_vector'])
            computed_hash = self._generate_content_hash(vectors, vector_data['metadata'])
            return computed_hash == vector_data['content_hash']
        except Exception as e:
            print(f"Error verifying vector data: {e}")
            return False

    async def _save_index(self):
        """Save index state to cloud storage"""
        index_state = self.index.get_state()
        await self.storage.save_model_state(index_state, 'vector_index.pkl')