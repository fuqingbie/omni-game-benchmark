"""
Maze Navigation Test Configuration File
"""
import os

# ================================
# Environment Configuration
# ================================
# Headless mode configuration - Set to True in server environments without graphical interface support
HEADLESS_MODE = False  # Set to True to enable headless mode, False to support visualization interface

# If headless mode is enabled, set SDL environment variables
if HEADLESS_MODE:
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

# ================================
# API Configuration - OpenAI Format
# ================================
API_BASE  = ""  # API server address, e.g.: "http://localhost:8000/v1"
API_KEY   = "EMPTY"   # API key, can be set to "EMPTY" for local services
MODEL_CHAT = ""  # Chat model name

# ================================
# Baichuan API Configuration - FastAPI Format
# ================================
BAICHUAN_FASTAPI_BASE_URL = ""  # Baichuan model's FastAPI server address

# ================================
# Game Configuration
# ================================
# Difficulty settings
DEFAULT_DIFFICULTY = "easy"  # Options: "easy", "medium", "hard"

# Test configuration
DEFAULT_ROUNDS = 5          # Default number of test rounds
DEFAULT_MAX_STEPS = 500     # Maximum steps per round
DEFAULT_AUTO_SPEED = 1.0    # Auto-run speed (seconds/step)

# Seed configuration
DEFAULT_SEED = None         # Default seed value, None for random
USE_SEQUENTIAL_SEEDS = True # Whether to use sequential seeds (0, 1, 2, ...) for each round
RANDOM_SEED_RANGE = (0, 9999)  # Range for random seed generation if not sequential

# ================================
# File Path Configuration
# ================================
# Results save directory
RESULTS_DIR = "results"     # Test results save directory
TEMP_DIR_PREFIX = "maze_"   # Temporary file directory prefix

# Game assets path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GAME_DIR = os.path.dirname(CURRENT_DIR)  # Parent directory (game)

# Try to find assets directory
ASSETS_PATHS = [
    os.path.join(GAME_DIR, "assets-necessay"),  # Standard path
    os.path.join(CURRENT_DIR, "..", "assets-necessay"),  # Relative path
    os.path.join(CURRENT_DIR, "assets-necessay"),  # Current directory
    "assets-necessay"  # Simple relative path
]

ASSETS_DIR = None
for path in ASSETS_PATHS:
    if os.path.exists(path):
        ASSETS_DIR = os.path.abspath(path)
        break

if ASSETS_DIR is None:
    print("Warning: assets-necessay directory not found. Audio features may not work properly.")
    # Set a default fallback path
    ASSETS_DIR = os.path.join(GAME_DIR, "assets-necessay")

# ================================
# Display Configuration
# ================================
# Text display window size
TEXT_DISPLAY_SIZE = (600, 400)
TEXT_DISPLAY_POS = (50, 50)
FONT_SIZE = 24

# ================================
# Logging Configuration
# ================================
LOG_LEVEL = "INFO"  # Log level: "DEBUG", "INFO", "WARNING", "ERROR"
ENABLE_DETAILED_LOGS = True  # Whether to enable detailed logs

# ================================
# Difficulty Configuration Mapping
# ================================
DIFFICULTY_MAP = {
    "1": "easy",
    "2": "medium", 
    "3": "hard",
    "easy": "easy",
    "medium": "medium",
    "hard": "hard"
}

DIFFICULTY_DESCRIPTIONS = {
    "easy": "Easy - Fewer walls, wider paths",
    "medium": "Medium - Moderate obstacles, medium difficulty paths", 
    "hard": "Hard - Dense obstacles, narrower passages, larger maze"
}
