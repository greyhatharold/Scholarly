import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
import asyncio
import logging
from src.data.storage.storage import CloudStorage
from src.data.data_manager import DataManager

logger = logging.getLogger(__name__)

class LiquidTimeLayer(nn.Module):
    """Liquid Time-constant layers implementation"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau_network = nn.Linear(hidden_size, hidden_size)
        self.update_network = nn.Linear(hidden_size, hidden_size)
        self.gate_network = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, dt):
        # Compute liquid time constants
        tau = torch.exp(self.tau_network(x))  # Ensure positive time constants
        # Compute state update
        dx = self.update_network(x)
        # Compute gating mechanism
        gate = torch.sigmoid(self.gate_network(x))
        # Apply liquid time-constant update
        x_new = x + dt * gate * (dx / tau)
        return x_new

class ScholarAI(nn.Module):
    """Neural network model for educational AI tasks"""
    def __init__(self, config):
        """Initialize ScholarAI model
        
        Args:
            config: Application configuration
        """
        super().__init__()
        logger.debug("Initializing ScholarAI model")
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        
        # Initialize base model first
        logger.debug(f"Loading base model: {config.embedding.model_name}")
        self.base_model = AutoModel.from_pretrained(config.embedding.model_name)
        
        # Initialize storage after model setup
        logger.debug("Initializing cloud storage")
        self.storage = CloudStorage(config)
        
        # Add DataManager instance
        logger.debug("Initializing data manager")
        self.data_manager = DataManager(config)
        
        # Initialize the rest of the model
        logger.debug("Initializing model layers")
        self._init_model_layers()
    
    async def initialize(self) -> None:
        """Async initialization of model components"""
        try:
            logger.info("Starting async model initialization")
            # Initialize required model files
            required_files = {
                'latest_model.pt': self.state_dict(),
                f"{self.config.embedding.model_name.replace('/', '_')}_base": self.base_model.state_dict()
            }

            for filename, initial_state in required_files.items():
                try:
                    await self.storage.load_model_state(filename)
                    logger.debug(f"Loaded existing {filename}")
                except FileNotFoundError:
                    logger.debug(f"Creating new {filename}")
                    await self.storage.save_model_state(initial_state, filename)
                except Exception as e:
                    logger.warning(f"Error loading {filename}: {e}")
                    # Save current state as fallback
                    await self.storage.save_model_state(initial_state, filename)

            # Load base model weights if available
            try:
                model_path = f"{self.config.embedding.model_name.replace('/', '_')}_base"
                logger.debug(f"Loading base model weights from {model_path}")
                state_dict = await self.storage.load_model_state(model_path)
                if isinstance(state_dict, dict):
                    self.base_model.load_state_dict(state_dict)
                    logger.info("Successfully loaded base model weights")
            except Exception as e:
                logger.warning(f"Using default weights: {e}")

        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
    
    def _init_model_layers(self):
        """Initialize model architecture"""
        logger.debug("Initializing model layers")
        # Replace standard layers with Liquid Time-constant layers
        self.liquid_layers = nn.ModuleList([
            LiquidTimeLayer(self.config.hidden_size)
            for _ in range(self.config.num_liquid_layers)
        ])
        
        # Task-specific layers
        logger.debug("Building task-specific layers")
        self.concept_encoder = self._build_concept_encoder()
        self.knowledge_encoder = self._build_knowledge_encoder()
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.max_concepts)

    def _get_tokenizer(self) -> AutoTokenizer:
        """Lazy initialization of tokenizer
        
        Returns:
            AutoTokenizer: The model's tokenizer
        """
        if self.tokenizer is None:
            logger.debug(f"Initializing tokenizer for {self.config.embedding.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding.model_name)
        return self.tokenizer
    
    def _build_concept_encoder(self):
        """Build the concept encoding layer"""
        logger.debug("Building concept encoder")
        return nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU()
        )
    
    def _build_knowledge_encoder(self):
        """Build the knowledge encoding layer"""
        logger.debug("Building knowledge encoder")
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU()
        )
        
    def forward(self, input_ids, attention_mask):
        """Forward pass through the model"""
        logger.debug("Performing forward pass")
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        concept_encoded = self.concept_encoder(base_outputs.last_hidden_state)
        knowledge_encoded = self.knowledge_encoder(concept_encoded)
        outputs = self.output_layer(knowledge_encoded)
        
        return outputs 
    
    def encode_knowledge(self, knowledge_data: Dict) -> np.ndarray:
        """Encode knowledge data into vector representations"""
        logger.debug("Encoding knowledge data")
        # Use pre-processed tensor if available
        if 'tensor' in knowledge_data:
            logger.debug("Using pre-processed tensor")
            encoded = knowledge_data['tensor']
        else:
            logger.debug("Preparing knowledge tensor")
            encoded = self._prepare_knowledge_tensor(knowledge_data)
            encoded = encoded.view(1, -1)  # Add batch dimension
        
        # Pass through knowledge encoder
        with torch.no_grad():
            knowledge_vectors = self.knowledge_encoder(encoded)
            
        # Convert to numpy for storage
        return knowledge_vectors.cpu().numpy()
    
    def _prepare_knowledge_tensor(self, knowledge_data: Dict) -> torch.Tensor:
        """Prepare knowledge data for encoding
        
        Args:
            knowledge_data: Dictionary containing knowledge information
        
        Returns:
            torch.Tensor: Processed tensor ready for encoding
        """
        logger.debug("Preparing knowledge tensor")
        text_content = ''
        if 'text' in knowledge_data:
            text_content = knowledge_data['text']
        elif 'wiki_content' in knowledge_data:
            wiki_data = knowledge_data['wiki_content']
            pages = wiki_data.get('query', {}).get('pages', {})
            for page_id in pages:
                extract = pages[page_id].get('extract', '')
                text_content += extract + ' '
        
        # Get tokenizer and encode text
        tokenizer = self._get_tokenizer()
        encoded = tokenizer(
            text_content,
            padding='max_length',
            truncation=True,
            max_length=self.config.embedding.max_length,
            return_tensors='pt'
        )
        
        # Get base model embeddings first
        with torch.no_grad():
            base_output = self.base_model(**encoded)
            hidden_states = base_output.last_hidden_state
        
        # Process through concept encoder to match dimensions
        concept_encoded = self.concept_encoder(hidden_states)
        # Average pooling to get single vector
        output = torch.mean(concept_encoded, dim=1)  # Shape: (1, hidden_size)
        
        return output
    
    def _load_cached_model(self, model_name: str) -> Optional[AutoModel]:
        """Attempt to load model from cache/storage
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Optional[AutoModel]: Cached model if available, None otherwise
        """
        logger.debug(f"Attempting to load cached model: {model_name}")
        try:
            model_path = f"{model_name.replace('/', '_')}_base"
            state_dict = asyncio.run(self.storage.load_model_state(model_path))
            model = AutoModel.from_pretrained(model_name)
            model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded cached model: {model_name}")
            return model
        except FileNotFoundError:
            logger.warning(f"No cached model found for: {model_name}")
            return None
        except Exception as e:
            logger.error(f"Error loading cached model: {e}")
            return None
    
    def _cache_model(self, model_name: str) -> None:
        """Cache model to storage
        
        Args:
            model_name: Name of the model to cache
        """
        logger.debug(f"Caching model: {model_name}")
        try:
            model_path = f"{model_name.replace('/', '_')}_base"
            asyncio.run(self.storage.save_model_state(self.base_model, model_path))
            logger.info(f"Successfully cached model: {model_name}")
        except Exception as e:
            logger.error(f"Error caching model: {e}")
    
    async def encode_and_store_knowledge(self, knowledge_data: Dict) -> Optional[int]:
        """Encode knowledge data and store in vector database"""
        logger.debug("Encoding and storing knowledge")
        # Encode knowledge into vectors
        vectors = self.encode_knowledge(knowledge_data)
        
        # Store vectors with metadata
        vector_id = await self.data_manager.store_vectors(
            vectors,
            metadata=knowledge_data
        )
        logger.info(f"Successfully stored knowledge with vector ID: {vector_id}")
        return vector_id
    
    async def find_similar_knowledge(self, query_data: Dict, k: int = 10) -> List[Dict]:
        """Find similar knowledge entries using vector similarity"""
        logger.debug(f"Searching for similar knowledge with k={k}")
        # Encode query into vector
        query_vector = self.encode_knowledge(query_data)
        
        # Search for similar vectors
        results = await self.data_manager.search_similar(query_vector, k)
        logger.info(f"Found {len(results)} similar knowledge entries")
        return results