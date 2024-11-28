import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
import asyncio
from src.data.storage.storage import CloudStorage
from src.data.data_manager import DataManager

class LiquidTimeLayer(nn.Module):
    """Liquid Time-constant layers implementation"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Time-constant networks
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
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        
        # Initialize base model first
        self.base_model = AutoModel.from_pretrained(config.embedding.model_name)
        
        # Initialize storage after model setup
        self.storage = CloudStorage(config)
        
        # Add DataManager instance
        self.data_manager = DataManager(config)
        
        # Initialize the rest of the model
        self._init_model_layers()
    
    async def initialize(self) -> None:
        """Async initialization of model components
        
        This method should be called after construction to load cached weights
        """
        try:
            model_name = self.config.embedding.model_name
            model_path = f"{model_name.replace('/', '_')}_base"
            
            # Try to load cached weights
            try:
                state_dict = await self.storage.load_model_state(model_path)
                self.base_model.load_state_dict(state_dict)
            except FileNotFoundError:
                # Cache current weights if none found
                await self.storage.save_model_state(self.base_model, model_path)
        except Exception as e:
            print(f"Warning: Error during model initialization: {e}")
    
    def _init_model_layers(self):
        """Initialize model architecture"""
        # Replace standard layers with Liquid Time-constant layers
        self.liquid_layers = nn.ModuleList([
            LiquidTimeLayer(self.config.hidden_size)
            for _ in range(self.config.num_liquid_layers)
        ])
        
        # Task-specific layers
        self.concept_encoder = self._build_concept_encoder()
        self.knowledge_encoder = self._build_knowledge_encoder()
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.max_concepts)

    def _get_tokenizer(self) -> AutoTokenizer:
        """Lazy initialization of tokenizer
        
        Returns:
            AutoTokenizer: The model's tokenizer
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding.model_name)
        return self.tokenizer
    
    def _build_concept_encoder(self):
        """Build the concept encoding layer"""
        return nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU()
        )
    
    def _build_knowledge_encoder(self):
        """Build the knowledge encoding layer"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU()
        )
        
    def forward(self, input_ids, attention_mask):
        """Forward pass through the model"""
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
        # Use pre-processed tensor if available
        if 'tensor' in knowledge_data:
            encoded = knowledge_data['tensor']
        else:
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
        try:
            model_path = f"{model_name.replace('/', '_')}_base"
            state_dict = asyncio.run(self.storage.load_model_state(model_path))
            model = AutoModel.from_pretrained(model_name)
            model.load_state_dict(state_dict)
            return model
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Warning: Error loading cached model: {e}")
            return None
    
    def _cache_model(self, model_name: str) -> None:
        """Cache model to storage
        
        Args:
            model_name: Name of the model to cache
        """
        try:
            model_path = f"{model_name.replace('/', '_')}_base"
            asyncio.run(self.storage.save_model_state(self.base_model, model_path))
        except Exception as e:
            print(f"Warning: Error caching model: {e}")
    
    async def encode_and_store_knowledge(self, knowledge_data: Dict) -> Optional[int]:
        """Encode knowledge data and store in vector database"""
        # Encode knowledge into vectors
        vectors = self.encode_knowledge(knowledge_data)
        
        # Store vectors with metadata
        vector_id = await self.data_manager.store_vectors(
            vectors,
            metadata=knowledge_data
        )
        return vector_id
    
    async def find_similar_knowledge(self, query_data: Dict, k: int = 10) -> List[Dict]:
        """Find similar knowledge entries using vector similarity"""
        # Encode query into vector
        query_vector = self.encode_knowledge(query_data)
        
        # Search for similar vectors
        results = await self.data_manager.search_similar(query_vector, k)
        return results