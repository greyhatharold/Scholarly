from src.models.scholar_ai import ScholarAI
from learning.learning_manager import SelfLearningManager
from src.learning.session import StudySession
from training.google_trainer import GoogleTrainer
from src.data.storage.storage import CloudStorage
from src.config.config import Config

def initialize_components(config: Config):
    """Initialize all system components"""
    model = ScholarAI(config)
    storage = CloudStorage(config)
    trainer = GoogleTrainer(config)
    learning_manager = SelfLearningManager(config)
    
    return model, storage, trainer, learning_manager

def load_model_state(model: ScholarAI, storage: CloudStorage):
    """Load existing model state if available"""
    try:
        state_dict = storage.load_model_state('latest_model.pt')
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Starting with fresh model")

def main():
    """Main entry point for running the system"""
    config = Config()
    
    # Initialize components
    model, storage, trainer, learning_manager = initialize_components(config)
    
    # Load model state
    load_model_state(model, storage)
    
    # Create study session
    session = StudySession(model, config)
    
    return model, session, storage, trainer

if __name__ == "__main__":
    main() 