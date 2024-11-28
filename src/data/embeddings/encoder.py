import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np
from transformers import AutoTokenizer
from src.config.config import EmbeddingConfig

class VectorEncoder:
    """Handles encoding of data into vector representations"""
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    async def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text into vector representations"""
        with torch.no_grad():
            inputs = self.tokenize_texts(texts)
            outputs = self.base_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize input texts"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoded.items()} 