import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

class BaseLiquidModel(nn.Module):
    """Base model with liquid neural network architecture for ScholarAI"""
    
    def __init__(self, config):
        """Initialize the base liquid neural network model
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        
        logger.info("Initializing BaseLiquidModel")
        
        # Core architecture components
        self.hidden_size = config.hidden_size
        self.num_liquid_layers = config.num_liquid_layers
        self.dt = config.dt
        self.integration_steps = config.integration_steps
        
        logger.debug(f"Model parameters - hidden_size: {self.hidden_size}, "
                    f"num_liquid_layers: {self.num_liquid_layers}, "
                    f"dt: {self.dt}, integration_steps: {self.integration_steps}")
        
        # Define liquid layers
        logger.debug("Creating liquid layers")
        self.liquid_layers = nn.ModuleList([
            LiquidLayer(
                hidden_size=self.hidden_size,
                dt=self.dt,
                integration_steps=self.integration_steps
            ) for _ in range(self.num_liquid_layers)
        ])
        
        # Transformer components
        logger.debug("Creating attention layers")
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=config.num_attention_heads
            ) for _ in range(config.num_hidden_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        logger.info("BaseLiquidModel initialization complete")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size)
        """
        logger.debug(f"Forward pass input shape: {x.shape}")
        
        # Pass through liquid layers
        for i, liquid_layer in enumerate(self.liquid_layers):
            logger.debug(f"Processing liquid layer {i+1}/{len(self.liquid_layers)}")
            x = liquid_layer(x)
            
        # Pass through attention layers
        for i, attention_layer in enumerate(self.attention_layers):
            logger.debug(f"Processing attention layer {i+1}/{len(self.attention_layers)}")
            attn_output, _ = attention_layer(x, x, x)
            x = x + attn_output
            
        output = self.output_projection(x)
        logger.debug(f"Forward pass output shape: {output.shape}")
        return output

class LiquidLayer(nn.Module):
    """Implementation of a liquid neural network layer"""
    
    def __init__(self, hidden_size: int, dt: float, integration_steps: int):
        """Initialize liquid layer
        
        Args:
            hidden_size: Dimension of hidden states
            dt: Time step for numerical integration
            integration_steps: Number of integration steps
        """
        super().__init__()
        logger.debug(f"Initializing LiquidLayer with hidden_size={hidden_size}, "
                    f"dt={dt}, integration_steps={integration_steps}")
        
        self.hidden_size = hidden_size
        self.dt = dt
        self.integration_steps = integration_steps
        
        # Neural ODE components
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing numerical integration
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Integrated output
        """
        logger.debug(f"LiquidLayer forward pass input shape: {x.shape}")
        h = x
        for step in range(self.integration_steps):
            dh = self.dynamics(h)
            h = h + self.dt * dh
            logger.debug(f"Integration step {step+1}/{self.integration_steps} complete")
        return h

def create_base_model(config) -> BaseLiquidModel:
    """Factory function to create and initialize a base model
    
    Args:
        config: Configuration object containing model parameters
        
    Returns:
        BaseLiquidModel: Initialized model instance
    """
    logger.info("Creating base model")
    
    # Create directory and model path
    model_dir = Path("src/base_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "base_model.pt"
    
    model = BaseLiquidModel(config)
    
    # Initialize weights
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    logger.debug("Initializing model weights")            
    model.apply(_init_weights)
    
    # Save the model
    try:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise
    
    logger.info("Base model creation complete")
    return model 

if __name__ == "__main__":
    # Create a simple config for testing
    from types import SimpleNamespace
    
    config = SimpleNamespace(
        hidden_size=256,
        num_liquid_layers=3,
        dt=0.1,
        integration_steps=5,
        num_attention_heads=8,
        num_hidden_layers=3
    )
    
    # Create and save the model
    try:
        model = create_base_model(config)
        print(f"Model created successfully and saved to src/base_model/base_model.pt")
    except Exception as e:
        print(f"Error creating model: {str(e)}") 