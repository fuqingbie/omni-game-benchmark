import pygame
import os
import random
import pickle
import numpy as np
import base64
import mimetypes
try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: FAISS library not found. Audio retrieval for musical notes will use placeholders.")
try:
    from openai import OpenAI, InternalServerError, RateLimitError # Using OpenAI client
except ImportError:
    OpenAI = None
    print("Warning: OpenAI library not found. Audio retrieval will use placeholders.")


# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED_COLOR = (255, 0, 0) # Renamed to avoid conflict with NOTE_RED if used
BLUE_COLOR = (0, 0, 255) # Renamed
YELLOW_COLOR = (255, 255, 0) # Renamed
PURPLE_COLOR = (128, 0, 128) # Renamed
ORANGE_COLOR = (255, 165, 0) # Renamed
GREY_COLOR = (128, 128, 128) # Renamed

LIGHT_GREEN = (170, 255, 170)
LIGHT_RED = (255, 170, 170)
LIGHT_BLUE = (173, 216, 230) # For UI elements
DARK_GREY = (50, 50, 50)     # For button backgrounds or text
BUTTON_HOVER_COLOR = (100, 100, 150)
BUTTON_SELECTED_COLOR = (50, 150, 50) # Greenish tint for selected

# --- Asset Paths ---
# Modified paths to use relative paths
MAIN_DIR = "game"
ASSETS_DIR = os.path.join(MAIN_DIR, "assets-necessay")
KENNEY_DIR = os.path.join(MAIN_DIR, "assets-necessay", "kenney")
KENNEY_AUDIO_DIR = os.path.join(KENNEY_DIR, "Audio")
# ... other KENNEY paths ...
SFX_DIR = os.path.join(ASSETS_DIR, "sfx")
TEXTURES_DIR = os.path.join(ASSETS_DIR, "textures")

# --- RAG Paths & Config (from rag_fps_pipeline.py) ---
INDEX_PATH = "game/assets-necessay/rag-pipeline/kenney.index"
META_PATH = "game/assets-necessay/rag-pipeline/kenney_meta.pkl"

# IMPORTANT: In a real application, use environment variables for API keys.
API_BASE = "" # From rag_fps_pipeline.py
API_KEY = "" # From rag_fps_pipeline.py
USER_TAG = "sound_alchemist_game_rag" # Custom user tag
MODEL_EMB = "text-embedding-3-small" # From rag_fps_pipeline.py

# --- Game States ---
MENU = "menu"
PLAYING = "playing" # General exploration/hub state
PUZZLE_1 = "puzzle_1" # The original placeholder puzzle
PUZZLE_MELODY = "puzzle_melody"
PUZZLE_COMPLETE = "puzzle_complete" # New state for celebrating puzzle completion

# --- Musical Notes Definition ---
NOTE_DO = "do"
NOTE_RE = "re"
NOTE_MI = "mi"
NOTE_FA = "fa"
NOTE_SO = "so"
NOTE_LA = "la" 
NOTE_TI = "si" # Changed to si to match file naming

# Full scale for RAG retrieval and potential future use
FULL_MUSICAL_SCALE_NOTES = [NOTE_DO, NOTE_RE, NOTE_MI, NOTE_FA, NOTE_SO, NOTE_LA, NOTE_TI]

# Notes that are currently interactive in the puzzle (mapped to colored blocks)
INTERACTIVE_MUSICAL_NOTES = [NOTE_DO, NOTE_RE, NOTE_MI, NOTE_FA, NOTE_SO, NOTE_LA, NOTE_TI]


NOTE_DISPLAY_NAMES = {
    NOTE_DO: "Do",
    NOTE_RE: "Re",
    NOTE_MI: "Mi",
    NOTE_FA: "Fa",
    NOTE_SO: "Sol", # Changed to Sol for more standard representation
    NOTE_LA: "La",
    NOTE_TI: "Si", # Changed to Si
}

# Note file paths
MUSIC_DIR = os.path.join(MAIN_DIR, "assets-necessay", "kenney", "music")

# Directly use the specified note files
MUSIC_FILE_PATHS = {
    NOTE_DO: os.path.join(MUSIC_DIR, "note-do.mp3"),
    NOTE_RE: os.path.join(MUSIC_DIR, "note-re.mp3"), 
    NOTE_MI: os.path.join(MUSIC_DIR, "note-mi.mp3"),
    NOTE_FA: os.path.join(MUSIC_DIR, "note-f.mp3"),    # Note: filename is f, not fa
    NOTE_SO: os.path.join(MUSIC_DIR, "note-salt.mp3"), # Note: filename is salt, not so/sol
    NOTE_LA: os.path.join(MUSIC_DIR, "note-la.mp3"),
    NOTE_TI: os.path.join(MUSIC_DIR, "note-c.mp3"),    # Note: filename is c, not si
}

# Removed pre-retrieved sounds, directly using our specified music files
PRE_RETRIEVED_SOUNDS = {
    NOTE_DO: MUSIC_FILE_PATHS[NOTE_DO],
    NOTE_RE: MUSIC_FILE_PATHS[NOTE_RE],
    NOTE_MI: MUSIC_FILE_PATHS[NOTE_MI],
    NOTE_FA: MUSIC_FILE_PATHS[NOTE_FA],
    NOTE_SO: MUSIC_FILE_PATHS[NOTE_SO],
    NOTE_LA: MUSIC_FILE_PATHS[NOTE_LA],
    NOTE_TI: MUSIC_FILE_PATHS[NOTE_TI],
}

# --- Difficulty Levels ---
DIFFICULTY_EASY = "Easy"
DIFFICULTY_MEDIUM = "Medium"
DIFFICULTY_HARD = "Hard"

DIFFICULTY_SETTINGS = {
    DIFFICULTY_EASY: {"sequence_length": 3, "score_multiplier": 1, "name": "Easy"},
    DIFFICULTY_MEDIUM: {"sequence_length": 5, "score_multiplier": 2, "name": "Medium"},
    DIFFICULTY_HARD: {"sequence_length": 7, "score_multiplier": 3, "name": "Hard"} # Allows note repetition
}

# --- Pygame Setup ---
pygame.init()
pygame.mixer.init() # For sound
# Allow more sound channels for simultaneous playback
pygame.mixer.set_num_channels(16)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("The Sound Alchemist's Secret Chamber") # Changed to English
clock = pygame.time.Clock()

# --- Game Variables ---
current_state = MENU
running = True
player_melody_input = [] # Stores the player's current melody sequence
last_puzzle_solved = "" # To know which puzzle led to PUZZLE_COMPLETE
current_difficulty = DIFFICULTY_MEDIUM # Default difficulty
correct_melody_sequence = [] # Will be generated dynamically
melody_puzzle_attempts = 0
player_score = 0
mouse_pos = (0,0) # To store mouse position for hover effects
auto_start_enabled = True
# New: Whether to enable auto-start mode

# Initialize sprite groups
all_game_sprites = pygame.sprite.Group()
note_elements = pygame.sprite.Group()
particles_group = pygame.sprite.Group() 

# Define note_size here, as it's used in PUZZLE_MELODY setup below
note_size = (100, 100)

# --- RAG Setup ---
faiss_index = None
meta_data = None
openai_client = None

if OpenAI:
    try:
        openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        print("OpenAI client initialized for RAG.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}. Audio retrieval will use placeholders.")
        openai_client = None
else:
    print("OpenAI library not available. Using placeholder sounds for RAG.")

def get_text_embedding_openai(text):
    """
    Generates a text embedding using the OpenAI API.
    """
    if openai_client is None:
        print("Error: OpenAI client not initialized. Cannot generate embeddings.")
        return None
    try:
        # Uses the global openai_client and MODEL_EMB
        response = openai_client.embeddings.create(model=MODEL_EMB, input=[text], user=USER_TAG)
        embedding_vector = np.asarray(response.data[0].embedding, "float32")
        return np.expand_dims(embedding_vector, axis=0) # Ensure 2D for FAISS
    except Exception as e:
        print(f"Error generating text embedding for '{text}' via OpenAI: {e}")
        return None

def retrieve_sound_path_from_rag(query_text, index, metadata_list, top_k=1):
    """
    Retrieves the full path to a sound file based on a text query using FAISS and OpenAI embeddings.
    Assumes metadata_list stores paths relative to KENNEY_DIR.
    """
    if index is None or openai_client is None or not metadata_list:
        return None

    query_embedding = get_text_embedding_openai(query_text)
    if query_embedding is None:
        return None
    
    try:
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        if top_k == 1 and len(indices[0]) > 0:
            retrieved_idx = indices[0][0]
            if 0 <= retrieved_idx < len(metadata_list):
                relative_path = metadata_list[retrieved_idx] 
                full_path = os.path.join(KENNEY_DIR, relative_path)

                if os.path.exists(full_path):
                    print(f"Retrieved for '{query_text}': {full_path} (Score: {distances[0][0]})")
                    return full_path
                else:
                    print(f"Retrieved metadata for '{query_text}' but file not found: {full_path}")
                    return None
            else:
                print(f"Retrieved index {retrieved_idx} out of bounds for metadata list.")
                return None
        else:
            return None
    except Exception as e:
        print(f"Error during FAISS search or metadata lookup for '{query_text}': {e}")
        return None

# Load RAG components at startup
if faiss and openai_client: 
    if os.path.exists(INDEX_PATH):
        try:
            faiss_index = faiss.read_index(INDEX_PATH)
            print(f"FAISS index loaded from {INDEX_PATH}. Index size: {faiss_index.ntotal}")
        except Exception as e:
            print(f"Failed to load FAISS index: {e}. Audio retrieval will use placeholders.")
            faiss_index = None
    else:
        print(f"FAISS index file not found at {INDEX_PATH}. Audio retrieval will use placeholders.")
        faiss_index = None

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "rb") as f:
                meta_data = pickle.load(f) 
            print(f"Metadata loaded from {META_PATH}. Number of entries: {len(meta_data)}")
        except Exception as e:
            print(f"Failed to load metadata: {e}. Audio retrieval will use placeholders.")
            meta_data = None
    else:
        print(f"Metadata file not found at {META_PATH}. Audio retrieval will use placeholders.")
        meta_data = None
else:
    if not faiss:
        print("FAISS not available. Using placeholder sounds.")
    if not openai_client:
         print("OpenAI client not initialized. Using placeholder sounds.")


# --- Helper Functions ---
sound_file_paths = {}

def load_sound(name, volume=1.0, directory=SFX_DIR):
    """Load a sound file and record its path"""
    global sound_file_paths
    
    if directory is None: # name is expected to be a full path
        path = name
    else:
        path = os.path.join(directory, name)
    
    # Debug information
    #print(f"Attempting to load audio file: {path}")
    #print(f"Current working directory: {os.getcwd()}")
    #print(f"File exists: {os.path.exists(path)}")
    
    # If the file doesn't exist, try searching from the project root
    if not os.path.exists(path) and path.startswith("game/"):
        # Try to calculate the path based on the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alternative_path = os.path.join(base_dir, path)
        print(f"Trying alternative path: {alternative_path}")
        print(f"Alternative path exists: {os.path.exists(alternative_path)}")
        if os.path.exists(alternative_path):
            path = alternative_path
    
    try:
        # Try to load the audio file, handling potential MP3 ID3 tag issues
        sound = pygame.mixer.Sound(path)
        sound.set_volume(volume)
        # Add the sound object and path to the mapping dictionary, not directly as an attribute
        sound_file_paths[sound] = path
        print(f"Successfully loaded audio: {path}")
        return sound
    except FileNotFoundError: # Specifically catch if the file isn't there
        print(f"Warning: Sound file not found at {path}")
        return None
    except pygame.error as e: # Catch Pygame-specific errors (e.g., unsupported format)
        # If it's an MP3 ID3 tag issue, try to continue loading while ignoring the warning
        if "id3" in str(e).lower() or "comment" in str(e).lower():
            print(f"Warning: MP3 ID3 tag issue (continuing anyway): {path}")
            try:
                # Try to force load, ignoring ID3 tag warnings
                sound = pygame.mixer.Sound(path)
                sound.set_volume(volume)
                # Add the sound object and path to the mapping dictionary
                sound_file_paths[sound] = path
                return sound
            except Exception as inner_e:
                print(f"Failed to load sound even after ignoring ID3 warnings: {path} - Error: {inner_e}")
                return None
        else:
            print(f"Warning: Cannot load sound (pygame error): {path} - {e}")
            return None
    except Exception as e: # Catch any other unexpected errors during loading
        print(f"Warning: An unexpected error occurred while loading sound {path}: {e}")
        return None


def play_sound(sound):
    """Play a sound and record the path of the audio file being played"""
    if sound:
        sound.play()
        # Record the path of the audio file being played (using the mapping dictionary)
        if sound in sound_file_paths:
            pygame.mixer._last_played_sound_file = sound_file_paths[sound]
        else:
            pygame.mixer._last_played_sound_file = None

def get_last_played_audio_data():
    """Get the most recently played audio data (converted to a numpy array)"""
    # Check if there is a record of the most recently played sound
    if not hasattr(pygame.mixer, '_last_played_sound_file') or pygame.mixer._last_played_sound_file is None:
        # If there is no record of a recently played audio file, check if any notes are clicked
        for note_sprite in note_elements:
            if note_sprite.is_highlighted and note_sprite.sound:
                # Return the sound of the currently highlighted note
                if note_sprite.sound in sound_file_paths:
                    print(f"Returning the sound of the currently highlighted note: {sound_file_paths[note_sprite.sound]}")
                    return note_sprite.sound
        # If there are no highlighted notes, return None
        print("No recently played audio and no highlighted note")
        return None
    
    try:
        import librosa
        import soundfile as sf
        
        sound_file_path = pygame.mixer._last_played_sound_file
        
        # Load the audio file using librosa and convert to 16kHz mono
        audio_data, sample_rate = librosa.load(sound_file_path, sr=16000, mono=True)
        
        # Ensure the audio length meets the environment's requirements (1 second = 16000 samples)
        target_length = 16000
        if len(audio_data) > target_length:
            # If the audio is too long, truncate it to the first second
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            # If the audio is too short, pad it with zeros to 1 second
            padding = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
        
        # Clear the record
        pygame.mixer._last_played_sound_file = None
        
        return audio_data.astype(np.float32)
        
    except ImportError:
        print("Warning: librosa not available, cannot convert audio file to numpy array")
        return None
    except Exception as e:
        print(f"Error loading audio data from {sound_file_path}: {e}")
        return None

def load_image(name, directory=TEXTURES_DIR, convert_alpha=True):
    """Loads an image file from the specified directory."""
    path = os.path.join(directory, name)
    try:
        image = pygame.image.load(path)
        if convert_alpha:
            image = image.convert_alpha()
        else:
            image = image.convert()
        return image
    except pygame.error as e:
        print(f"Cannot load image: {path} - {e}")
        return None

# --- UI Button Class ---
class Button:
    def __init__(self, text, x, y, width, height, font, text_color, base_color, hover_color, selected_color, value):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.text_color = text_color
        self.base_color = base_color
        self.hover_color = hover_color
        self.selected_color = selected_color
        self.value = value 
        self.is_hovered = False

    def draw(self, screen, is_selected=False):
        current_color = self.base_color
        if is_selected:
            current_color = self.selected_color
        elif self.is_hovered:
            current_color = self.hover_color
        
        pygame.draw.rect(screen, current_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, WHITE, self.rect, width=2, border_radius=5) 

        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def update_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

# --- Game Objects ---
class PuzzleElement(pygame.sprite.Sprite):
    def __init__(self, image, x, y, sound_name=None, element_id=None, original_color=None):
        super().__init__()
        self.original_image = image 
        self.image = image.copy() 
        self.rect = self.image.get_rect(topleft=(x,y))
        self.sound = None
        self.element_id = element_id
        self.original_color = original_color if original_color else image.get_at((0,0))
        self.is_animating = False
        self.animation_timer = 0
        self.animation_duration = 15 
        
        self.is_highlighted = False
        self.highlight_color_tuple = None
        self.highlight_timer = 0
        self.highlight_duration = 20 

        if sound_name and SFX_DIR: 
            self.sound = load_sound(sound_name)
        elif sound_name: 
             self.sound = load_sound(sound_name, directory=os.path.join(os.path.dirname(__file__), "assets", "sfx"))


    def interact(self):
        print(f"Interacted with element {self.element_id} at {self.rect.topleft}")
        if self.sound:
            play_sound(self.sound)
        
        self.start_click_animation()

        if current_state == PUZZLE_MELODY and self.element_id:
            handle_melody_input(self.element_id, self)

    def start_click_animation(self):
        self.is_animating = True
        self.animation_timer = self.animation_duration
        current_center = self.rect.center
        self.image = pygame.transform.smoothscale(self.original_image, 
                                             (int(self.original_image.get_width() * 1.2),
                                              int(self.original_image.get_height() * 1.2)))
        self.rect = self.image.get_rect(center=current_center)


    def highlight(self, color_tuple, duration=20):
        self.is_highlighted = True
        self.highlight_color_tuple = color_tuple
        self.highlight_timer = duration
        self.image = self.original_image.copy() 
        self.image.fill(self.highlight_color_tuple, special_flags=pygame.BLEND_RGB_MULT)


    def update(self): 
        if self.is_animating:
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self.is_animating = False
                self.image = self.original_image.copy() 
                self.rect = self.image.get_rect(center=self.rect.center) 

        if self.is_highlighted:
            self.highlight_timer -= 1
            if self.highlight_timer > 0:
                if not self.is_animating or (self.is_animating and self.animation_timer <=0) : 
                    temp_image = self.original_image.copy()
                    temp_image.fill(self.highlight_color_tuple, special_flags=pygame.BLEND_RGB_MULT)
                    current_center = self.rect.center
                    self.image = temp_image
                    self.rect = self.image.get_rect(center=current_center)
            else: 
                self.is_highlighted = False
                self.highlight_color_tuple = None
                if not self.is_animating or (self.is_animating and self.animation_timer <=0):
                    self.image = self.original_image.copy()
                    self.rect = self.image.get_rect(center=self.rect.center)
        
        if not self.is_animating and not self.is_highlighted:
            if self.image is not self.original_image: 
                 current_center = self.rect.center
                 self.image = self.original_image.copy()
                 self.rect = self.image.get_rect(center=current_center)


# --- Particle System ---
class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y, color, size_range=(3, 8), vel_range_x=(-2,2), vel_range_y=(-4, -1), grav=0.1, lifespan_frames=60):
        super().__init__()
        size = random.randint(size_range[0], size_range[1])
        self.image = pygame.Surface((size, size))
        self.image.fill(BLACK) 
        pygame.draw.circle(self.image, color, (size//2, size//2), size//2) 
        self.image.set_colorkey(BLACK) 
        self.rect = self.image.get_rect(center=(x, y))
        self.velocity = pygame.math.Vector2(random.uniform(vel_range_x[0], vel_range_x[1]), 
                                           random.uniform(vel_range_y[0], vel_range_y[1]))
        self.gravity = grav
        self.lifespan = lifespan_frames
        self.initial_lifespan = lifespan_frames 

    def update(self):
        self.velocity.y += self.gravity
        self.rect.x += self.velocity.x
        self.rect.y += self.velocity.y
        self.lifespan -= 1
        if self.initial_lifespan > 0:
            alpha = max(0, int(255 * (self.lifespan / self.initial_lifespan)))
            self.image.set_alpha(alpha)
        else:
            self.image.set_alpha(0)

        if self.lifespan <= 0:
            self.kill()


def create_success_particles(num_particles=70, position=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
                             colors=[(255,215,0), (255,255,224), (255,193,37)]): 
    for _ in range(num_particles):
        color = random.choice(colors)
        particle = Particle(position[0] + random.randint(-20,20), 
                            position[1] + random.randint(-20,20), 
                            color, lifespan_frames=random.randint(40,80))
        particles_group.add(particle)

# --- Puzzle Specific Assets & Logic ---

kenney_digital_audio_placeholder_dir = os.path.join(KENNEY_AUDIO_DIR, "Digital Audio", "Audio") 
kenney_ui_audio = os.path.join(KENNEY_AUDIO_DIR, "UI Audio", "Audio")
kenney_jingles_sax = os.path.join(KENNEY_AUDIO_DIR, "Music Jingles", "Audio (Saxophone)")

# Change all feedback messages to English
melody_feedback_sounds = {
    "success": load_sound(MUSIC_FILE_PATHS[NOTE_SO], directory=None, volume=0.7)
}

if melody_feedback_sounds["success"] is None:
    print("Warning: Success sound effect could not be loaded. Game will continue without it.")

placeholder_sounds_map = {
    NOTE_DO: ("powerUp1.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_RE: ("laser1.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_MI: ("highUp.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_FA: ("phaserUp6.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_SO: ("pepSound1.ogg", kenney_digital_audio_placeholder_dir),
    NOTE_LA: ("pepSound2.ogg", kenney_digital_audio_placeholder_dir), 
    NOTE_TI: ("powerUp3.ogg", kenney_digital_audio_placeholder_dir),  
}

# Update note sound loading logic
melody_note_sounds = {}

for note_id in FULL_MUSICAL_SCALE_NOTES:
    note_name_for_query = NOTE_DISPLAY_NAMES.get(note_id, "musical sound")
    sound_path = PRE_RETRIEVED_SOUNDS[note_id]
    
    # Temporarily disable pygame's audio warnings
    import os
    old_stderr = os.dup(2)
    os.close(2)
    os.open(os.devnull, os.O_RDWR)
    
    try:
        sound = load_sound(sound_path, directory=None)
        if sound:
            melody_note_sounds[note_id] = sound
            print(f"Successfully loaded sound effect for note {note_name_for_query}: {sound_path}")
        else:
            print(f"Warning: Could not load sound effect for note {note_name_for_query}: {sound_path}")
            # If MP3 loading fails, try using the original placeholder sound effects
            if note_id in placeholder_sounds_map:
                placeholder_file, placeholder_dir = placeholder_sounds_map[note_id]
                melody_note_sounds[note_id] = load_sound(placeholder_file, directory=placeholder_dir)
                if melody_note_sounds[note_id] is None:
                    print(f"Critical Warning: Placeholder sound effect for {note_name_for_query} ({placeholder_file}) also could not be loaded.")
    finally:
        # Restore stderr
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

# Create custom note buttons without text labels
def create_note_button(color):
    """Creates a custom button surface for musical notes without text"""
    img = pygame.Surface(note_size, pygame.SRCALPHA)
    
    # Fill with base color but slightly transparent
    base_color = (*color[:3], 220)  # RGB + alpha
    pygame.draw.rect(img, base_color, (0, 0, note_size[0], note_size[1]), border_radius=10)
    
    # Add a highlight effect
    highlight_color = (min(color[0] + 50, 255), min(color[1] + 50, 255), min(color[2] + 50, 255), 180)
    pygame.draw.rect(img, highlight_color, (3, 3, note_size[0] - 6, 15), border_radius=7)
    
    # Add border
    border_color = (max(color[0] - 50, 0), max(color[1] - 50, 0), max(color[2] - 50, 0), 255)
    pygame.draw.rect(img, border_color, (0, 0, note_size[0], note_size[1]), width=2, border_radius=10)
    
    return img

# Create note images programmatically - no text labels
note_images = {}
note_colors = {
    "Do": RED_COLOR,
    "Re": ORANGE_COLOR,
    "Mi": YELLOW_COLOR,
    "Fa": GREEN,
    "Sol": BLUE_COLOR,
    "La": PURPLE_COLOR,
    "Si": GREY_COLOR,
}

for note_name, color in note_colors.items():
    note_images[note_name] = create_note_button(color)
    print(f"Created custom button image for note: {note_name}")

# Stores the current color-to-note mapping in the game
current_note_color_mapping = {}

# List of all available colors
ALL_COLORS = {
    "Red": RED_COLOR,
    "Orange": ORANGE_COLOR,
    "Yellow": YELLOW_COLOR,
    "Green": GREEN,
    "Blue": BLUE_COLOR,
    "Purple": PURPLE_COLOR,
    "Grey": GREY_COLOR,
}

# Create initial note elements with custom graphics based on difficulty
def create_note_elements(difficulty=None):
    global note_elements
    
    if difficulty is None:
        difficulty = current_difficulty
    
    # Get the number of notes to display based on difficulty
    settings = DIFFICULTY_SETTINGS[difficulty]
    note_count = settings["sequence_length"]
    
    # Clear existing note elements
    note_elements.empty()  
    all_game_sprites.remove(note_elements)
    
    # Layout parameters
    base_y = 250  # Center vertically
    
    # Only create positions for the correct number of note blocks
    positions = []
    
    # Calculate total width needed for all blocks with spacing
    total_width = note_count * note_size[0] + (note_count - 1) * 20
    start_x = (SCREEN_WIDTH - total_width) // 2
    
    # Create positions based on the actual number of blocks needed
    for i in range(note_count):
        positions.append((start_x + i * (note_size[0] + 20), base_y))
    
    # Randomize positions
    random.shuffle(positions)
    
    # Use the current note-color mapping
    # If the mapping is empty, create a default mapping
    if not current_note_color_mapping:
        default_colors = {
            NOTE_DO: RED_COLOR,
            NOTE_RE: ORANGE_COLOR,
            NOTE_MI: YELLOW_COLOR,
            NOTE_FA: GREEN,
            NOTE_SO: BLUE_COLOR,
            NOTE_LA: PURPLE_COLOR,
            NOTE_TI: GREY_COLOR,
        }
        for note_id, color in default_colors.items():
            current_note_color_mapping[note_id] = color
    
    # Only create sprites for the notes in the current sequence
    for i, note_id in enumerate(correct_melody_sequence):
        if i < len(positions):
            note_name = NOTE_DISPLAY_NAMES.get(note_id, "?")
            color = current_note_color_mapping.get(note_id, (200, 200, 200))
            
            # Create a custom button image based on the current color
            img = create_note_button(color)
            
            element = PuzzleElement(
                img, 
                positions[i][0], 
                positions[i][1], 
                sound_name=None,
                element_id=note_id, 
                original_color=color
            )
            
            if note_id in melody_note_sounds: 
                element.sound = melody_note_sounds[note_id]
            else:
                print(f"Warning: Could not find sound for note {note_id}")
            
            note_elements.add(element)
            all_game_sprites.add(element)

def generate_new_melody_puzzle(difficulty):
    global correct_melody_sequence, player_melody_input, melody_puzzle_attempts, current_note_color_mapping
    
    settings = DIFFICULTY_SETTINGS[difficulty]
    seq_length = settings["sequence_length"]
    
    if not INTERACTIVE_MUSICAL_NOTES:
        print("Error: INTERACTIVE_MUSICAL_NOTES is not defined or empty.")
        correct_melody_sequence = []
        return

    # Use notes in the strict do-re-mi-fa-sol-la-si order
    ordered_notes = [NOTE_DO, NOTE_RE, NOTE_MI, NOTE_FA, NOTE_SO, NOTE_LA, NOTE_TI]
    
    # Randomly select a starting position in the ordered notes
    if seq_length < len(ordered_notes):
        max_start_idx = len(ordered_notes) - seq_length
        start_idx = random.randint(0, max_start_idx)
        correct_melody_sequence = ordered_notes[start_idx:start_idx + seq_length]
    else:
        correct_melody_sequence = ordered_notes[:seq_length]
    
    # Randomize the correspondence between colors and notes - change this logic
    # Create a copy of the lists of notes and colors for random assignment
    available_notes = INTERACTIVE_MUSICAL_NOTES.copy()
    available_colors = list(ALL_COLORS.values())
    
    # Randomly shuffle both lists
    random.shuffle(available_notes)
    random.shuffle(available_colors)
    
    # Create a new random note-color mapping
    current_note_color_mapping = {}
    for i, note_id in enumerate(available_notes):
        if i < len(available_colors):
            current_note_color_mapping[note_id] = available_colors[i]
    
    # Reset player state
    player_melody_input = []
    melody_puzzle_attempts = 0
    
    # Create note elements based on the new melody
    create_note_elements(difficulty)
    
    debug_sequence_names = [NOTE_DISPLAY_NAMES.get(n, n) for n in correct_melody_sequence]
    print(f"New melody puzzle ({difficulty}, {len(correct_melody_sequence)} notes): Sequence is {debug_sequence_names}")
    print("New random note-color mapping:")
    color_name_mapping = {v: k for k, v in ALL_COLORS.items()}
    current_note_color_mapping_name = {}
    
    for note, color in current_note_color_mapping.items():
        color_name = color_name_mapping.get(color, f"RGB{color}")
        print(f"{NOTE_DISPLAY_NAMES.get(note, note)}: {color_name}")
        current_note_color_mapping_name[f"{NOTE_DISPLAY_NAMES.get(note, note)}"] = color_name
    print(f"Current note-color mapping: {current_note_color_mapping_name}")

    # Modified scoring system with more detailed rules
def calculate_score(difficulty, mistakes, completion_time=None):
    """Calculate score with more sophisticated rules"""
    settings = DIFFICULTY_SETTINGS[difficulty]
    seq_length = settings["sequence_length"]
    base_score = 1000
    
    # Base multiplier from difficulty
    multiplier = settings["score_multiplier"]
    
    # Mistake penalties
    mistake_penalty = 150 * multiplier  # Harder difficulties have higher penalties
    mistake_deduction = mistakes * mistake_penalty
    
    # Perfect play bonus
    perfect_bonus = 0
    if mistakes == 0:
        perfect_bonus = 500 * multiplier
    
    # Sequential bonus - rewards completing longer sequences
    sequence_bonus = seq_length * 50 * multiplier
    
    # Calculate final score
    final_score = (base_score * multiplier) + sequence_bonus + perfect_bonus - mistake_deduction
    
    # Ensure score doesn't go negative
    final_score = max(0, final_score)
    
    # Log score calculation details
    print(f"Score Details:")
    print(f"  Base Score: {base_score} × {multiplier} = {base_score * multiplier}")
    print(f"  Sequence Bonus: {sequence_bonus}")
    print(f"  Perfect Play Bonus: {perfect_bonus}")
    print(f"  Mistake Penalty: -{mistake_deduction}")
    print(f"  Final Score: {final_score}")
    
    return final_score

def handle_melody_input(note_id, clicked_element: PuzzleElement):
    global player_melody_input, current_state, last_puzzle_solved, melody_puzzle_attempts, player_score
    
    # Safety check: if no valid element is passed, don't play a sound but continue the logic
    if clicked_element and hasattr(clicked_element, 'sound') and clicked_element.sound:
        play_sound(clicked_element.sound)
    elif note_id in melody_note_sounds:
        # If there is no element but a corresponding sound effect, play it directly
        play_sound(melody_note_sounds[note_id])
    
    if note_id not in INTERACTIVE_MUSICAL_NOTES:
        if clicked_element:
            clicked_element.highlight(LIGHT_RED, duration=30)
        player_melody_input = [] 
        melody_puzzle_attempts += 1
        return

    player_melody_input.append(note_id)
    
    correct_so_far = True
    for i in range(len(player_melody_input)):
        if i >= len(correct_melody_sequence) or player_melody_input[i] != correct_melody_sequence[i]:
            correct_so_far = False
            break
    
    if correct_so_far:
        if clicked_element:
            clicked_element.highlight(LIGHT_GREEN, duration=20)

        if len(player_melody_input) == len(correct_melody_sequence):
            print("Melody correct!")
            if clicked_element:
                create_success_particles(position=clicked_element.rect.center)
            else:
                create_success_particles(position=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            
            # Use the enhanced scoring system
            player_score = calculate_score(current_difficulty, melody_puzzle_attempts)
            print(f"Score: {player_score}, Attempts: {melody_puzzle_attempts}")

            player_melody_input = [] 
            
            last_puzzle_solved = PUZZLE_MELODY
            current_state = PUZZLE_COMPLETE
        
    else:
        # Don't play error sound, just highlight in red if element exists
        if clicked_element:
            clicked_element.highlight(LIGHT_RED, duration=30)
        #player_melody_input = [] 
        melody_puzzle_attempts += 1


# --- UI Elements & Fonts ---
try:
    default_font_name = pygame.font.get_default_font()
    font_large = pygame.font.Font(default_font_name, 74)
    font_medium = pygame.font.Font(default_font_name, 50)
    font_small = pygame.font.Font(default_font_name, 30)
    font_tiny = pygame.font.Font(default_font_name, 24)
except Exception as e:
    print(f"Font loading error: {e}. Using pygame.font.Font(None, size).")
    font_large = pygame.font.Font(None, 74)
    font_medium = pygame.font.Font(None, 50)
    font_small = pygame.font.Font(None, 30)
    font_tiny = pygame.font.Font(None, 24)

# --- Menu Buttons ---
difficulty_buttons = []
def setup_menu_buttons():
    global difficulty_buttons
    difficulty_buttons = [] 
    button_width = 150
    button_height = 50
    button_y_start = 240
    button_spacing = 20 
    button_x = SCREEN_WIDTH // 2 - button_width // 2

    difficulty_options_config = [
        (DIFFICULTY_SETTINGS[DIFFICULTY_EASY]["name"], DIFFICULTY_EASY),
        (DIFFICULTY_SETTINGS[DIFFICULTY_MEDIUM]["name"], DIFFICULTY_MEDIUM),
        (DIFFICULTY_SETTINGS[DIFFICULTY_HARD]["name"], DIFFICULTY_HARD)
    ]

    for i, (text, value) in enumerate(difficulty_options_config):
        button = Button(
            text=text,
            x=button_x,
            y=button_y_start + i * (button_height + button_spacing),
            width=button_width,
            height=button_height,
            font=font_small,
            text_color=WHITE,
            base_color=DARK_GREY,
            hover_color=BUTTON_HOVER_COLOR,
            selected_color=BUTTON_SELECTED_COLOR,
            value=value
        )
        difficulty_buttons.append(button)

setup_menu_buttons() 

# --- Game Logic Placeholder & Other Assets ---
# Create a custom background instead of loading a non-existent image
menu_background_image = None

# Create a custom music-themed background
def create_custom_background():
    bg_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    # Create a gradient from dark blue to dark purple
    for i in range(SCREEN_HEIGHT):
        # Gradient color
        color = (20, max(10, min(20 + i // 30, 40)), max(40, min(60 + i // 10, 120)))
        pygame.draw.line(bg_surface, color, (0, i), (SCREEN_WIDTH, i))
    
    # Add music-themed decorative elements
    for _ in range(30):
        # Randomly positioned note symbols
        x = random.randint(0, SCREEN_WIDTH)
        y = random.randint(0, SCREEN_HEIGHT)
        size = random.randint(5, 15)
        alpha = random.randint(30, 100)
        
        note_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        color = (255, 255, 255, alpha)  # Semi-transparent white
        
        # Simple note shape
        pygame.draw.circle(note_surf, color, (size//2, size//2), size//2)
        pygame.draw.line(note_surf, color, (size-2, size//2), (size-2, size//4), 2)
        
        bg_surface.blit(note_surf, (x, y))
    
    return bg_surface

# Generate custom background
menu_background_image = create_custom_background()

# Do not attempt to load a non-existent image
placeholder_puzzle_image = pygame.Surface((50, 50))
placeholder_puzzle_image.fill(GREEN)
clickable_element = PuzzleElement(placeholder_puzzle_image, 100, 100, None, element_id="green_box") 
# Set sound effect for the green box (using a note sound instead of a non-existent sound file)
if melody_note_sounds and NOTE_DO in melody_note_sounds:
    clickable_element.sound = melody_note_sounds[NOTE_DO]
puzzle1_sprites = pygame.sprite.Group(clickable_element)
all_game_sprites.add(clickable_element)

def start_melody_puzzle_directly(difficulty=None):
    """Start the melody puzzle directly without key interaction"""
    global current_state, current_difficulty, player_score, melody_puzzle_attempts
    
    if difficulty:
        current_difficulty = difficulty
    
    print(f"Auto-starting Melody Puzzle (Difficulty: {current_difficulty})...")
    generate_new_melody_puzzle(current_difficulty) 
    player_score = 0 
    melody_puzzle_attempts = 0 
    current_state = PUZZLE_MELODY
    return True

def set_auto_start_mode(enabled=True):
    """Set the auto-start mode"""
    global auto_start_enabled
    auto_start_enabled = enabled
    print(f"Auto-start mode: {'enabled' if enabled else 'disabled'}")

def get_current_color_note_mapping():
    """Get the current color-to-note mapping (for the agent)"""
    color_note_mapping = {}
    color_name_mapping = {v: k for k, v in ALL_COLORS.items()}
    
    for note_id, color in current_note_color_mapping.items():
        color_name = color_name_mapping.get(color, f"RGB{color}")
        color_note_mapping[color_name.lower()] = NOTE_DISPLAY_NAMES.get(note_id, note_id)
    
    return color_note_mapping

def get_current_color_to_note_id_mapping():
    """Get the current color name to note ID mapping (for the environment)"""
    color_to_note = {}
    color_name_mapping = {v: k for k, v in ALL_COLORS.items()}
    
    for note_id, rgb_color in current_note_color_mapping.items():
        color_name = color_name_mapping.get(rgb_color)
        if color_name:
            color_to_note[color_name.upper()] = note_id
    
    return color_to_note

def get_game_state():
    """Get the current game state information"""
    return {
        "state": current_state,
        "difficulty": current_difficulty,
        "score": player_score,
        "attempts": melody_puzzle_attempts,
        "sequence_length": len(correct_melody_sequence),
        "input_length": len(player_melody_input),
        "game_over": current_state == PUZZLE_COMPLETE,
        "current_note_color_mapping": current_note_color_mapping.copy(),  # Add current mapping information
        "correct_sequence": correct_melody_sequence.copy(),  # Add correct sequence information
        "current_color_to_note_mapping": get_current_color_to_note_id_mapping()  # Add color-to-note mapping
    }

def encode_audio(audio_path):
    # Get the MIME type of the file
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        # Set the default MIME type based on the file extension
        if audio_path.lower().endswith('.wav'):
            mime_type = "audio/wav"
        elif audio_path.lower().endswith('.mp3'):
            mime_type = "audio/mpeg"
        else:
            mime_type = "audio/wav"  # Default to wav
    
    with open(audio_path, "rb") as audio_file:
        base64_data = base64.b64encode(audio_file.read()).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_data}"

def get_last_played_audio_data():
    if player_melody_input[-1]:
        _note_id = player_melody_input[-1]
        _sound_path = PRE_RETRIEVED_SOUNDS[_note_id]
        audio_data_ = encode_audio(_sound_path)
        return audio_data_
    try:
        import librosa
        import soundfile as sf
        
        sound_file_path = pygame.mixer._last_played_sound_file
        
        # Load the audio file using librosa and convert to 16kHz mono
        audio_data, sample_rate = librosa.load(sound_file_path, sr=16000, mono=True)
        
        # Ensure the audio length meets the environment's requirements (1 second = 16000 samples)
        target_length = 16000
        if len(audio_data) > target_length:
            # If the audio is too long, truncate it to the first second
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            # If the audio is too short, pad it with zeros to 1 second
            padding = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
        
        # Clear the record
        pygame.mixer._last_played_sound_file = None
        
        return audio_data.astype(np.float32)
        
    except ImportError:
        print("Warning: librosa not available, cannot convert audio file to numpy array")
        return None
    except Exception as e:
        print(f"Error loading audio data from {sound_file_path}: {e}")
        return None

# --- Main Game Loop ---
# Only run the game loop if this file is executed directly
if __name__ == "__main__":
    while running:
        mouse_pos = pygame.mouse.get_pos() 
        
        # If auto-start mode is enabled and in the menu state, start the game automatically
        if auto_start_enabled and current_state == MENU:
            start_melody_puzzle_directly()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    if current_state == MENU:
                        for button in difficulty_buttons:
                            if button.is_clicked(mouse_pos):
                                current_difficulty = button.value
                                print(f"Difficulty set to {DIFFICULTY_SETTINGS[current_difficulty]['name']} via button click")
                                if melody_feedback_sounds and melody_feedback_sounds.get("correct_input"):
                                    play_sound(melody_feedback_sounds["correct_input"])
                                # If auto-start is enabled, start the game immediately after selecting the difficulty
                                if auto_start_enabled:
                                    start_melody_puzzle_directly()
                                break 
                    elif current_state == PUZZLE_1:
                        for sprite in puzzle1_sprites:
                            if sprite.rect.collidepoint(event.pos):
                                sprite.interact()
                    elif current_state == PUZZLE_MELODY:
                        for note_sprite in note_elements: 
                            if note_sprite.rect.collidepoint(event.pos):
                                note_sprite.interact() 

            elif event.type == pygame.KEYDOWN:
                # Only respond to key presses in non-auto mode
                if not auto_start_enabled:
                    if current_state == MENU:
                        if event.key == pygame.K_p: 
                            start_melody_puzzle_directly()
                        elif event.key == pygame.K_g: 
                             print("Starting Green Box Puzzle from Menu...")
                             current_state = PUZZLE_1
                    elif current_state == PUZZLE_COMPLETE:
                        # Return to menu when puzzle is completed
                        current_state = MENU
                    elif current_state == PUZZLE_MELODY:
                         if event.key == pygame.K_m: 
                            # Return to menu
                            current_state = MENU
                         elif event.key == pygame.K_r:
                            # Reset current puzzle with the same difficulty
                            generate_new_melody_puzzle(current_difficulty)
                            player_score = 0
                            melody_puzzle_attempts = 0
    
        screen.fill(BLACK) 

        if current_state == MENU:
            if menu_background_image:
                screen.blit(menu_background_image, (0,0))
            else:
                # Use a gradient background instead of an image
                for i in range(SCREEN_HEIGHT):
                    # Create a gradient from dark blue to a slightly lighter blue
                    color = (20, 20, max(40, min(40 + i // 3, 90)))
                    pygame.draw.line(screen, color, (0, i), (SCREEN_WIDTH, i))
            
            # Optimize title and button layout to ensure text fits
            title_text = font_large.render("Sound Alchemist's Chamber", True, WHITE)
            title_width = title_text.get_width()
            
            # If the title is too wide, shrink the font
            if title_width > SCREEN_WIDTH - 40:
                scale_factor = (SCREEN_WIDTH - 40) / title_width
                new_size = int(74 * scale_factor)
                font_large_adjusted = pygame.font.Font(default_font_name, new_size)
                title_text = font_large_adjusted.render("Sound Alchemist's Chamber", True, WHITE)
            
            screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 80))
            
            difficulty_title_text = font_medium.render("Select Difficulty:", True, WHITE)
            screen.blit(difficulty_title_text, (SCREEN_WIDTH // 2 - difficulty_title_text.get_width() // 2, 180))

            for button in difficulty_buttons:
                button.update_hover(mouse_pos) 
                button.draw(screen, is_selected=(button.value == current_difficulty))
            
            puzzle_prompts_y_start = (difficulty_buttons[-1].rect.bottom + 40) if difficulty_buttons else 380
            play_prompt_melody = font_small.render("Press 'P' to attempt the Melody Puzzle", True, WHITE)
            screen.blit(play_prompt_melody, (SCREEN_WIDTH // 2 - play_prompt_melody.get_width() // 2, puzzle_prompts_y_start))

            play_prompt_green = font_small.render("Press 'G' for the Green Box (placeholder)", True, WHITE)
            screen.blit(play_prompt_green, (SCREEN_WIDTH // 2 - play_prompt_green.get_width() // 2, puzzle_prompts_y_start + 50))

        elif current_state == PLAYING: 
            screen.fill(WHITE)
            text = font_medium.render("Exploring the Chamber...", True, BLACK) 
            screen.blit(text, (50, 50))
            
        elif current_state == PUZZLE_1:
            screen.fill((50, 50, 50)) 
            text = font_medium.render("Puzzle 1: Click the Green Box", True, WHITE) 
            screen.blit(text, (50, 50))
            puzzle1_sprites.draw(screen) 
            
        elif current_state == PUZZLE_MELODY: 
            screen.fill((30, 30, 70)) 
            
            # Optimize text display to ensure it fits
            title_text = font_medium.render("The Alchemist's Melody", True, WHITE)
            if title_text.get_width() > SCREEN_WIDTH - 40:
                font_medium_adjusted = pygame.font.Font(default_font_name, 40)  # Shrink font
                title_text = font_medium_adjusted.render("The Alchemist's Melody", True, WHITE)
                
            screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 30))

            difficulty_text_str = f"Difficulty: {DIFFICULTY_SETTINGS[current_difficulty]['name']}"
            difficulty_text_surf = font_small.render(difficulty_text_str, True, LIGHT_BLUE)
            screen.blit(difficulty_text_surf, (SCREEN_WIDTH // 2 - difficulty_text_surf.get_width() // 2, 80))

            instruction_text_str = "Click the colored blocks in the correct musical order"
            instruction_text = font_small.render(instruction_text_str, True, YELLOW_COLOR)
            
            # If the text is too long, reduce the font size
            if instruction_text.get_width() > SCREEN_WIDTH - 40:
                max_width = SCREEN_WIDTH - 40
                adjusted_font_size = int(font_small.get_height() * (max_width / instruction_text.get_width()))
                adjusted_font = pygame.font.Font(default_font_name, adjusted_font_size)
                instruction_text = adjusted_font.render(instruction_text_str, True, YELLOW_COLOR)
                
            screen.blit(instruction_text, (SCREEN_WIDTH // 2 - instruction_text.get_width() // 2, 120))
            
            instruction_reset = font_tiny.render("Press 'R' to reset puzzle", True, WHITE)
            screen.blit(instruction_reset, (SCREEN_WIDTH // 2 - instruction_reset.get_width() // 2, 150))
            
            instruction_exit = font_tiny.render("Press 'M' to return to menu", True, WHITE) 
            screen.blit(instruction_exit, (SCREEN_WIDTH // 2 - instruction_exit.get_width() // 2, SCREEN_HEIGHT - 30))

            note_elements.draw(screen) 

            feedback_y_pos = SCREEN_HEIGHT - 70
            input_display_parts = []
            for pid in player_melody_input:
                if "distractor" in pid:
                    input_display_parts.append("X") 
                else:
                    input_display_parts.append(NOTE_DISPLAY_NAMES.get(pid, "?"))
            
            input_text_str = "Your input: " + " - ".join(input_display_parts) 
            
            # If the text is too long, reduce the font size
            input_text_render = font_small.render(input_text_str, True, WHITE)
            if input_text_render.get_width() > SCREEN_WIDTH - 40:
                max_width = SCREEN_WIDTH - 40
                adjusted_font_size = int(font_small.get_height() * (max_width / input_text_render.get_width()))
                adjusted_font = pygame.font.Font(default_font_name, adjusted_font_size)
                input_text_render = adjusted_font.render(input_text_str, True, WHITE)
                
            screen.blit(input_text_render, (SCREEN_WIDTH // 2 - input_text_render.get_width() // 2, feedback_y_pos))

            attempts_text_str = f"Mistakes: {melody_puzzle_attempts}"
            attempts_text_surf = font_tiny.render(attempts_text_str, True, LIGHT_RED)
            screen.blit(attempts_text_surf, (20, SCREEN_HEIGHT - 30))

        elif current_state == PUZZLE_COMPLETE:
            screen.fill((20, 80, 20)) 
            
            main_message = ""
            if last_puzzle_solved == PUZZLE_MELODY:
                main_message = "Melody Puzzle Solved!" 
            elif last_puzzle_solved == PUZZLE_1:
                 main_message = "Green Box Puzzle Solved!" 
            else:
                main_message = "Puzzle Complete!" 

            # Ensure the text fits
            main_message_text = font_large.render(main_message, True, WHITE)
            if main_message_text.get_width() > SCREEN_WIDTH - 40:
                adjusted_font_size = int(font_large.get_height() * ((SCREEN_WIDTH - 40) / main_message_text.get_width()))
                adjusted_font = pygame.font.Font(default_font_name, adjusted_font_size)
                main_message_text = adjusted_font.render(main_message, True, WHITE)
                
            screen.blit(main_message_text, (SCREEN_WIDTH // 2 - main_message_text.get_width() // 2, 
                                             SCREEN_HEIGHT // 2 - main_message_text.get_height() // 2 - 70))
            
            sub_text_str = "Well Done, Alchemist!" 
            if last_puzzle_solved == PUZZLE_MELODY:
                score_text_str = f"Score: {player_score}"
                score_surf = font_medium.render(score_text_str, True, YELLOW_COLOR)
                screen.blit(score_surf, (SCREEN_WIDTH // 2 - score_surf.get_width() // 2, SCREEN_HEIGHT // 2 + 0))

                rating = ""
                if player_score >= DIFFICULTY_SETTINGS[current_difficulty]["score_multiplier"] * 900 : 
                    rating = "Excellent!"
                elif player_score >= DIFFICULTY_SETTINGS[current_difficulty]["score_multiplier"] * 600: 
                    rating = "Great Job!"
                elif player_score > 0:
                    rating = "Good Effort!"
                else:
                    rating = "Keep Practicing!"
                
                rating_surf = font_small.render(rating, True, WHITE)
                screen.blit(rating_surf, (SCREEN_WIDTH // 2 - rating_surf.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
                sub_text_str = "You have a keen ear!"

            sub_text = font_medium.render(sub_text_str, True, YELLOW_COLOR)
            screen.blit(sub_text, (SCREEN_WIDTH // 2 - sub_text.get_width() // 2, SCREEN_HEIGHT // 2 + 90))

            text_continue = font_small.render("Press any key to return to the Menu.", True, WHITE) 
            screen.blit(text_continue, (SCREEN_WIDTH // 2 - text_continue.get_width() // 2, SCREEN_HEIGHT - 100))
            
        all_game_sprites.update() 
        particles_group.update()  
        particles_group.draw(screen) 

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()