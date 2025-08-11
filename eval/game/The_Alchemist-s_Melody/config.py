"""
The Alchemist's Melody - Music Agent Game Configuration File
Contains all configuration options for the game environment, agent, and API
"""
import os
from typing import Dict, Any, Optional

class Config:
    """Main configuration class, containing all game and agent settings"""
    
    # =================================
    # Basic Game Configuration
    # =================================
    GAME_NAME = "The Alchemist's Melody"
    GAME_VERSION = "1.0.0"
    
    # Difficulty Settings
    DIFFICULTY_LEVELS = {
        "easy": {"sequence_length": 3, "score_multiplier": 1},
        "normal": {"sequence_length": 5, "score_multiplier": 2}, 
        "hard": {"sequence_length": 7, "score_multiplier": 3}
    }
    DEFAULT_DIFFICULTY = "normal"
    
    # Color to Action Mapping
    COLOR_ID_MAP = {
        "BLUE": 0,     # Sol
        "RED": 1,      # Do
        "GREEN": 2,    # Fa  
        "YELLOW": 3,   # Mi
        "ORANGE": 4,   # Re
        "PURPLE": 5,   # La
        "GREY": 6,     # Ti/Si
    }
    
    # Note Display Names
    NOTE_DISPLAY_NAMES = {
        "do": "Do",
        "re": "Re", 
        "mi": "Mi",
        "fa": "Fa",
        "so": "Sol",
        "la": "La",
        "si": "Si"
    }
    
    # =================================
    # Environment Configuration
    # =================================
    class Environment:
        # Image and Audio Settings
        SCREEN_WIDTH = 800
        SCREEN_HEIGHT = 600
        IMG_SIZE = (224, 224)  # Observation image resolution
        FPS = 60
        
        # Audio Settings
        AUDIO_SR = 16000  # Sample rate 16kHz
        AUDIO_CHANNELS = 1  # Mono channel
        AUDIO_DURATION = 1  # Audio duration per step (seconds)
        
        # Data Saving Settings
        SAVE_DATA = True
        SAVE_SEQUENCE = True
        SAVE_DIR = "game_data/caclu"
        
        # Auto Mode Settings
        AUTO_START_ENABLED = True
        MAX_STEPS_PER_EPISODE = 25
    
    # =================================
    # Agent Configuration
    # =================================
    class Agent:
        # Basic Settings
        VERBOSE = True
        USE_LOCAL_FALLBACK = True
        MAX_RETRIES = 3
        
        # Conversation Strategy Configuration
        CONVERSATION_STRATEGY = "hybrid"  # "native", "rag", "hybrid"
        NATIVE_WINDOW_SIZE = 8  # Number of native conversation rounds to keep
        RAG_RETRIEVAL_COUNT = 3  # Number of relevant rounds for RAG retrieval
        COMPRESS_OLD_ROUNDS = True
        MULTIMODAL_SUMMARY = True
        
        # Memory Management
        MAX_NATIVE_HISTORY = 8
        MAX_TOTAL_MEMORY = 50
        
        # Text Output Management
        SAVE_TEXT_OUTPUTS = True
        TEXT_OUTPUT_DIR = "agent_outputs"
    
    # =================================
    # API Configuration
    # =================================
    class API:
        # Main API Settings (user needs to fill in)
        BASE_URL = ""  # Fill in your API base URL
        API_KEY = ""   # Fill in your API key
        MODEL_CHAT = "gemini-pro-2.5"
        
        # Backup API Settings (Baichuan model)
        BAICHUAN_BASE_URL = ""  # Baichuan API server address
        BAICHUAN_ENABLED = False
        
        # Request Settings
        TIMEOUT = 300  # Request timeout (seconds)
        MAX_TOKENS = 10000
        TEMPERATURE = 0.1
        
        # Retry Settings
        RETRY_ATTEMPTS = 3
        RETRY_DELAY = 1  # Retry delay (seconds)
    
    # =================================
    # Path Configuration
    # =================================
    class Paths:
        # Project Root Directory
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Game Asset Paths
        GAME_DIR = os.path.join(PROJECT_ROOT, "eval", "game", "The_Alchemist-s_Melody")
        ASSETS_DIR = os.path.join(GAME_DIR, "assets")
        MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
        
        # Data Save Paths
        DATA_DIR = os.path.join(GAME_DIR, "game_data")
        LOG_DIR = os.path.join(GAME_DIR, "logs")
        DEBUG_DIR = os.path.join(GAME_DIR, "debug_logs")
        
        # Sequence Data Paths
        SEQUENCE_DIR = os.path.join(DATA_DIR, "sequences")
        SCORES_DIR = os.path.join(DATA_DIR, "scores")
        
        @classmethod
        def ensure_directories(cls):
            """Ensure all necessary directories exist"""
            dirs_to_create = [
                cls.DATA_DIR,
                cls.LOG_DIR, 
                cls.DEBUG_DIR,
                cls.SEQUENCE_DIR,
                cls.SCORES_DIR
            ]
            
            for directory in dirs_to_create:
                os.makedirs(directory, exist_ok=True)
                
            print(f"Ensured directories exist: {dirs_to_create}")
    
    # =================================
    # Scoring System Configuration
    # =================================
    class Scoring:
        # Base Score Settings
        BASE_SCORE = 1000
        
        # Reward System
        POSITIVE_FEEDBACK_REWARD = 0.5  # Correct note reward
        EXPLORATION_REWARD = 0.1        # Exploration reward
        SEQUENCE_ADVANCE_REWARD = 1.0   # Sequence advance reward
        
        # Penalty System
        SEQUENCE_RESET_PENALTY = 1.0    # Sequence reset penalty
        MISTAKE_PENALTY_BASE = 150      # Base penalty for mistakes
        
        # Completion Rewards
        PERFECT_PLAY_BONUS = 500        # Perfect play bonus
        COMPLETION_MULTIPLIER = 100     # Completion reward multiplier
        
        # Score Ratings
        SCORE_RATINGS = {
            "excellent": 0.9,  # Score above 90%
            "great": 0.6,      # Score above 60%
            "good": 0.3,       # Score above 30%
            "practice": 0.0    # Other
        }
    
    # =================================
    # Debugging and Logging Configuration
    # =================================
    class Debug:
        # Log Level
        LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
        
        # Debugging Options
        SAVE_API_REQUESTS = True
        SAVE_GAME_STATES = True
        SAVE_DECISION_HISTORY = True
        
        # Verbose Output Options
        SHOW_MODEL_RESPONSES = True
        SHOW_GAME_STATE_DETAILS = True
        SHOW_MEMORY_STATS = True
        
        # Performance Monitoring
        MONITOR_RESPONSE_TIMES = True
        MONITOR_MEMORY_USAGE = True
    
    # =================================
    # Experiment Configuration
    # =================================
    class Experiment:
        # Experiment Settings
        NUM_EPISODES = 10
        MAX_STEPS_PER_EPISODE = 25
        
        # Evaluation Metrics
        TRACK_COMPLETION_RATE = True
        TRACK_LEARNING_PROGRESS = True
        TRACK_EFFICIENCY_METRICS = True
        
        # Automation Settings
        AUTO_SAVE_RESULTS = True
        AUTO_GENERATE_REPORTS = True

# =================================
# Configuration Validation and Utility Functions
# =================================

def validate_config() -> bool:
    """Validate the configuration's effectiveness"""
    errors = []
    
    # Check API configuration
    if not Config.API.BASE_URL:
        errors.append("API.BASE_URL is not set")
    if not Config.API.API_KEY:
        errors.append("API.API_KEY is not set")
    
    # Check path configuration
    if not os.path.exists(Config.Paths.PROJECT_ROOT):
        errors.append(f"Project root directory does not exist: {Config.Paths.PROJECT_ROOT}")
    
    # Check difficulty configuration
    if Config.DEFAULT_DIFFICULTY not in Config.DIFFICULTY_LEVELS:
        errors.append(f"Default difficulty {Config.DEFAULT_DIFFICULTY} is not in the list of available difficulties")
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Configuration validation passed")
    return True

def load_config_from_env() -> None:
    """Load configuration from environment variables"""
    # API configuration
    Config.API.BASE_URL = os.getenv("ALCHEMIST_API_BASE", Config.API.BASE_URL)
    Config.API.API_KEY = os.getenv("ALCHEMIST_API_KEY", Config.API.API_KEY)
    Config.API.MODEL_CHAT = os.getenv("ALCHEMIST_MODEL", Config.API.MODEL_CHAT)
    
    # Baichuan API configuration
    Config.API.BAICHUAN_BASE_URL = os.getenv("BAICHUAN_API_BASE", Config.API.BAICHUAN_BASE_URL)
    Config.API.BAICHUAN_ENABLED = os.getenv("BAICHUAN_ENABLED", "false").lower() == "true"
    
    # Debugging configuration
    Config.Debug.LOG_LEVEL = os.getenv("LOG_LEVEL", Config.Debug.LOG_LEVEL)
    Config.Agent.VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
    
    print("Configuration loaded from environment variables")

def print_config_summary() -> None:
    """Print a summary of the configuration"""
    print(f"""
=== {Config.GAME_NAME} v{Config.GAME_VERSION} Configuration Summary ===

Game Configuration:
  - Default Difficulty: {Config.DEFAULT_DIFFICULTY}
  - Available Colors: {len(Config.COLOR_ID_MAP)}
  - Auto Start: {Config.Environment.AUTO_START_ENABLED}

Environment Configuration:
  - Image Resolution: {Config.Environment.IMG_SIZE}
  - Audio Sample Rate: {Config.Environment.AUDIO_SR}Hz
  - Save Data: {Config.Environment.SAVE_DATA}

Agent Configuration:
  - Conversation Strategy: {Config.Agent.CONVERSATION_STRATEGY}
  - Memory Window: {Config.Agent.NATIVE_WINDOW_SIZE}
  - Max Retries: {Config.Agent.MAX_RETRIES}

API Configuration:
  - Model: {Config.API.MODEL_CHAT}
  - Timeout: {Config.API.TIMEOUT}s
  - API Configured: {'Yes' if Config.API.API_KEY else 'No'}

Path Configuration:
  - Data Directory: {Config.Paths.DATA_DIR}
  - Debug Directory: {Config.Paths.DEBUG_DIR}

Debug Configuration:
  - Log Level: {Config.Debug.LOG_LEVEL}
  - Verbose Output: {Config.Agent.VERBOSE}
""")

# Initialize configuration
def initialize_config():
    """Initialize the configuration system"""
    print(f"Initializing {Config.GAME_NAME} configuration...")
    
    # Load environment variables
    load_config_from_env()
    
    # Ensure directories exist
    Config.Paths.ensure_directories()
    
    # Validate configuration
    if validate_config():
        print("Configuration initialized successfully")
        if Config.Agent.VERBOSE:
            print_config_summary()
        return True
    else:
        print("Configuration initialization failed")
        return False

# If this file is run directly, execute a configuration test
if __name__ == "__main__":
    initialize_config()