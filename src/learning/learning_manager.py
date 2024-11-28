from datetime import datetime
import logging
import requests
from typing import Dict, List, Optional
import numpy as np
from src.models.scholar_ai import ScholarAI
from src.data.data_manager import DataManager
import torch
from src.training.google_trainer import GoogleTrainer

logger = logging.getLogger(__name__)

class SelfLearningManager:
    """Manages continuous learning and knowledge acquisition"""
    def __init__(self, config):
        logger.debug("Initializing SelfLearningManager")
        self.config = config
        self.data_manager = DataManager(config)
        
    async def acquire_knowledge(self, topic: str) -> Dict:
        """Orchestrates the knowledge acquisition process"""
        logger.info(f"Starting knowledge acquisition for topic: {topic}")
        try:
            # Ensure hash mappings exist
            try:
                await self.data_manager.storage.load_model_state('hash_mappings.pkl')
            except FileNotFoundError:
                logger.debug("Creating new hash mappings file")
                await self.data_manager.storage.save_model_state({}, 'hash_mappings.pkl')

            # Get knowledge from sources
            logger.debug("Searching reliable sources")
            sources = await self.search_reliable_sources(topic)
            logger.debug("Validating information from sources")
            validated_info = await self.validate_information(sources)
            
            if not validated_info:
                logger.debug(f"No valid information found for topic: {topic}")
                return {'status': 'no_data', 'topic': topic}

            # Encode and store knowledge
            try:
                # Convert knowledge to vectors
                logger.debug("Encoding knowledge into vectors")
                vectors = await self._encode_knowledge(validated_info)
                if vectors.size > 0:
                    logger.debug(f"Generated vectors with shape: {vectors.shape}")
                    # Store in cloud with metadata
                    model = ScholarAI(self.config)
                    await model.initialize()
                    complexity = model._estimate_complexity(
                        torch.from_numpy(vectors)
                    ).item()
                    
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
                            .get('extract', '')[:500],  # Store first 500 chars of summary
                        'complexity_score': complexity,
                        'processing_time': datetime.now().isoformat()
                    }
                    
                    try:
                        # Initialize empty hash mappings if not exists
                        logger.debug("Checking hash mappings")
                        try:
                            await self.data_manager.storage.load_model_state('hash_mappings.pkl')
                        except FileNotFoundError:
                            logger.debug("Creating new hash mappings file")
                            await self.data_manager.storage.save_model_state({}, 'hash_mappings.pkl')
                        
                        # Store vectors and metadata in cloud
                        logger.debug("Storing vectors and metadata")
                        vector_id = await self.data_manager.store_vectors(
                            vectors=vectors,
                            metadata=metadata
                        )
                        
                        logger.info(f"Successfully stored knowledge for topic: {topic}")
                        return {
                            'status': 'success',
                            'topic': topic,
                            'stored_vectors': vectors.shape[0],
                            'metadata': metadata,
                            'vector_id': vector_id
                        }
                    except Exception as store_error:
                        logger.error(f"Error storing vectors: {store_error}")
                        return {
                            'status': 'storage_error',
                            'message': str(store_error),
                            'topic': topic
                        }
                        
            except Exception as e:
                logger.error(f"Error processing knowledge: {str(e)}")
                return {
                    'status': 'error',
                    'message': f"Failed to process knowledge: {str(e)}",
                    'topic': topic
                }
                
            logger.debug("No vectors generated from knowledge")
            return {
                'status': 'no_vectors',
                'topic': topic,
                'message': 'No vectors generated from knowledge'
            }
                
        except Exception as e:
            logger.error(f"Error in knowledge acquisition: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'topic': topic
            }

    async def retrieve_knowledge(self, query_text: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant knowledge based on query"""
        logger.info(f"Retrieving knowledge for query: {query_text}")
        try:
            # Convert query to vector
            logger.debug("Converting query to vector")
            query_vector = await self._encode_knowledge({'text': query_text})
            
            # Search similar vectors in cloud storage
            logger.debug(f"Searching for {k} similar vectors")
            results = await self.data_manager.search_similar(
                query_vector=query_vector,
                k=k
            )
            
            logger.debug(f"Found {len(results)} matching results")
            return [{
                'topic': result['metadata']['topic'],
                'timestamp': result['metadata']['timestamp'],
                'relevance_score': 1.0 - result['distance'],  # Convert distance to similarity
                'summary': result['metadata'].get('content_summary', ''),
                'sources': result['metadata'].get('sources', [])
            } for result in results]
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return []
    
    async def search_reliable_sources(self, topic: str) -> Dict:
        """Searches and retrieves information from reliable sources"""
        logger.info(f"Searching reliable sources for topic: {topic}")
        sources = {
            'academic_papers': [],
            'educational_sites': [],
            'textbooks': []
        }
        
        wiki_data = await self._fetch_wikipedia_data(topic)
        if wiki_data:
            logger.debug("Retrieved Wikipedia data")
            sources['wiki_content'] = wiki_data
            
        return sources
    
    async def _fetch_wikipedia_data(self, topic: str) -> Dict:
        """Fetches data from Wikipedia API"""
        logger.debug(f"Fetching Wikipedia data for: {topic}")
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
        logger.debug("Encoding knowledge data")
        model = ScholarAI(self.config)
        await model.initialize()  # Ensure model is initialized
        
        if not knowledge_data:
            logger.debug("No knowledge data to encode")
            return np.array([])
        
        if 'wiki_content' in knowledge_data:
            logger.debug("Processing Wikipedia content")
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
        logger.debug("Extracting text from Wikipedia data")
        try:
            pages = wiki_data.get('query', {}).get('pages', {})
            if pages:
                page = next(iter(pages.values()))
                return page.get('extract', '')
        except (KeyError, AttributeError) as e:
            logger.error(f"Error extracting Wikipedia text: {e}")
            return ''
        return ''
    
    async def validate_information(self, sources: Dict) -> Dict:
        """Validates information from sources for accuracy and relevance
        
        Args:
            sources: Dictionary containing information from different source types
            
        Returns:
            Dict of validated information
        """
        logger.debug("Validating information from sources")
        validated_info = {}
        
        # Start with basic validation of wiki content as it's currently our main source
        if 'wiki_content' in sources and sources['wiki_content']:
            logger.debug("Validated Wikipedia content")
            validated_info['wiki_content'] = sources['wiki_content']
            
        return validated_info
    
    async def start_continuous_learning(self, continuous_config: Optional[Dict] = None) -> None:
        """Start continuous learning process with Google Cloud integration
        
        Args:
            continuous_config: Optional configuration for continuous learning
        """
        logger.info("Starting continuous learning process")
        try:
            # Initialize model and trainer
            model = ScholarAI(self.config)
            await model.initialize()
            
            trainer = GoogleTrainer(self.config)
            
            # Generate version for continuous learning
            version = f"continuous-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Start continuous training job
            job_name = await trainer.prepare_continuous_training(
                version=version,
                continuous_config=continuous_config
            )
            
            logger.info(f"Started continuous learning job: {job_name}")
            
            # Start local continuous processing
            await model.process_continuous_stream()
            
        except Exception as e:
            logger.error(f"Error in continuous learning: {e}")
            raise