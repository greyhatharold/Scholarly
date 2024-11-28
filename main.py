import os
import sys
from pathlib import Path
import asyncio

# Add project root to Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

from src.models.scholar_ai import ScholarAI
from src.learning.learning_manager import SelfLearningManager
from src.learning.session import StudySession
from src.training.google_trainer import GoogleTrainer
from src.data.storage.storage import CloudStorage
from src.config.config import Config
from src.data.versioning.model_version_manager import ModelVersionManager

async def initialize_components(config: Config):
    """Initialize all system components"""
    # Create model factory function
    def create_scholar_ai(config: Config):
        return ScholarAI(config)
    
    # Initialize version manager with factory
    version_manager = ModelVersionManager(config, model_factory=create_scholar_ai)
    await version_manager.initialize()
    
    # Initialize other components
    model = ScholarAI(config)
    await model.initialize()
    
    storage = CloudStorage(config)
    trainer = GoogleTrainer(config)
    learning_manager = SelfLearningManager(config)
    
    return model, storage, trainer, learning_manager, version_manager

async def load_model_state(model: ScholarAI, storage: CloudStorage):
    """Load existing model state if available"""
    try:
        state_dict = await storage.load_model_state('latest_model.pt')
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Starting with fresh model")

async def main():
    """Main entry point for running the system"""
    config = Config()
    
    # Initialize components
    model, storage, trainer, learning_manager, version_manager = await initialize_components(config)
    
    # Start knowledge acquisition
    topic = "machine learning fundamentals"
    knowledge = await learning_manager.acquire_knowledge(topic)
    
    # Load model state
    await load_model_state(model, storage)
    
    # Create study session
    session = StudySession(model, config)
    
    return model, session, storage, trainer, version_manager

if __name__ == "__main__":
    asyncio.run(main()) 