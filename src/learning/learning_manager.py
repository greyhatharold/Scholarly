from datetime import datetime
import json
import io
import requests
from typing import Dict, List
from googleapiclient.http import MediaIoBaseUpload
import numpy as np
from src.models.scholar_ai import ScholarAI
from src.data.data_manager import DataManager

class SelfLearningManager:
    """Manages continuous learning and knowledge acquisition"""
    def __init__(self, config):
        self.config = config
        self.data_manager = DataManager(config)
        
    async def acquire_knowledge(self, topic: str) -> Dict:
        """Orchestrates the knowledge acquisition process"""
        # Get knowledge from sources
        sources = await self.search_reliable_sources(topic)
        validated_info = await self.validate_information(sources)
        
        if not validated_info:
            return {}

        # Encode and store knowledge
        try:
            # Convert knowledge to vectors
            vectors = await self._encode_knowledge(validated_info)
            if vectors.size > 0:
                # Store in cloud with metadata
                metadata = {
                    'topic': topic,
                    'timestamp': datetime.now().isoformat(),
                    'sources': sources,
                    'content_summary': validated_info.get('wiki_content', {})
                        .get('query', {})
                        .get('pages', {})
                        .get(next(iter(validated_info.get('wiki_content', {})
                            .get('query', {})
                            .get('pages', {}))), {})
                        .get('extract', '')[:500]  # Store first 500 chars of summary
                }
                
                # Store vectors and metadata in cloud
                await self.data_manager.store_vectors(
                    vectors=vectors,
                    metadata=metadata
                )
                
                return {
                    'status': 'success',
                    'topic': topic,
                    'stored_vectors': vectors.shape[0],
                    'metadata': metadata
                }
                
        except Exception as e:
            print(f"Error storing knowledge: {str(e)}")
            return {
                'status': 'error',
                'message': f"Failed to store knowledge: {str(e)}"
            }
            
        return validated_info

    async def retrieve_knowledge(self, query_text: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant knowledge based on query"""
        try:
            # Convert query to vector
            query_vector = await self._encode_knowledge({'text': query_text})
            
            # Search similar vectors in cloud storage
            results = await self.data_manager.search_similar(
                query_vector=query_vector,
                k=k
            )
            
            return [{
                'topic': result['metadata']['topic'],
                'timestamp': result['metadata']['timestamp'],
                'relevance_score': 1.0 - result['distance'],  # Convert distance to similarity
                'summary': result['metadata'].get('content_summary', ''),
                'sources': result['metadata'].get('sources', [])
            } for result in results]
            
        except Exception as e:
            print(f"Error retrieving knowledge: {str(e)}")
            return []
    
    async def search_reliable_sources(self, topic: str) -> Dict:
        """Searches and retrieves information from reliable sources"""
        sources = {
            'academic_papers': [],
            'educational_sites': [],
            'textbooks': []
        }
        
        wiki_data = await self._fetch_wikipedia_data(topic)
        if wiki_data:
            sources['wiki_content'] = wiki_data
            
        return sources
    
    async def _fetch_wikipedia_data(self, topic: str) -> Dict:
        """Fetches data from Wikipedia API"""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': topic,
            'prop': 'extracts',
            'exintro': True
        }
        
        response = requests.get("https://en.wikipedia.org/w/api.php", params=params)
        return response.json() if response.ok else None 
    
    async def _encode_knowledge(self, knowledge_data: Dict) -> np.ndarray:
        """Convert knowledge into vector representations
        
        Args:
            knowledge_data: Dictionary containing structured knowledge information
            
        Returns:
            np.ndarray: Encoded knowledge vectors with shape (n_samples, embedding_dim)
        """
        model = ScholarAI(self.config)
        await model.initialize()  # Ensure model is initialized
        
        if not knowledge_data:
            return np.array([])
        
        if 'wiki_content' in knowledge_data:
            text_content = await self._extract_text_from_wiki(knowledge_data['wiki_content'])
            if text_content:
                # Reshape the input to match expected dimensions
                encoded = model._prepare_knowledge_tensor({'text': text_content})
                encoded = encoded.view(1, -1)  # Add batch dimension
                return model.encode_knowledge({'text': text_content, 'tensor': encoded})
        
        return model.encode_knowledge(knowledge_data)
    
    async def _extract_text_from_wiki(self, wiki_data: Dict) -> str:
        """Extract meaningful text content from Wikipedia API response
        
        Args:
            wiki_data: Raw Wikipedia API response
            
        Returns:
            str: Extracted text content
        """
        try:
            pages = wiki_data.get('query', {}).get('pages', {})
            if pages:
                page = next(iter(pages.values()))
                return page.get('extract', '')
        except (KeyError, AttributeError):
            return ''
        return ''
    
    async def validate_information(self, sources: Dict) -> Dict:
        """Validates information from sources for accuracy and relevance
        
        Args:
            sources: Dictionary containing information from different source types
            
        Returns:
            Dict of validated information
        """
        validated_info = {}
        
        # Start with basic validation of wiki content as it's currently our main source
        if 'wiki_content' in sources and sources['wiki_content']:
            validated_info['wiki_content'] = sources['wiki_content']
            
        return validated_info