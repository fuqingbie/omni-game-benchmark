"""
Classic Bomberman Game
How to run: python classic_bomberman.py

Controls:
Player 1 (Red): WASD to move, Space to place bomb
Player 2 (Blue): Arrow keys to move, Enter to place bomb  
Player 3 (Green): TFGH to move, R to place bomb
Player 4 (Yellow): IJKL to move, U to place bomb

Game Rules:
- Bombs explode in cross-shaped flames
- Flames can destroy soft walls with 20% chance to drop items
- Items include: Fire power up, Bomb count up, Speed up
- Players caught in flames can be rescued by teammates
- Last player standing wins

Dependencies: pygame 2.x, standard library
Resolution: 832 × 704 pixels (13 cols × 11 rows, 64px per tile)
Frame Rate: 60 FPS
"""

import pygame
import random
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set
import os
import sys
from game_difficulty import DifficultyLevel, get_difficulty_config

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 832  # 13 * 64
WINDOW_HEIGHT = 704  # 11 * 64
GRID_WIDTH = 13
GRID_HEIGHT = 11
TILE_SIZE = 64
FPS = 60
MOVE_DELAY = 8  # Added move delay frames, the higher the value, the slower the movement

# Color Definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
BROWN = (139, 69, 19)
LIGHT_BLUE = (173, 216, 230)

class Tile(Enum):
    """Map tile type enum"""
    FLOOR = 0      # Floor
    SOFT = 1       # Soft wall (destructible)
    HARD = 2       # Hard wall (indestructible)
    BOMB = 3       # Bomb
    ITEM = 4       # Item

class Item(Enum):
    """Item type enum"""
    FIRE_UP = 0    # Firepower enhancement
    BOMB_UP = 1    # Bomb count increase
    SPEED_UP = 2   # Speed boost

@dataclass
class Vector2:
    """2D Vector class"""
    x: int
    y: int
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

@dataclass
class Player:
    """Player data class"""
    pos: Vector2                    # Grid coordinates
    fire: int = 1                   # Explosion range
    bombs: int = 1                  # Simultaneous bomb limit
    speed: int = 1                  # Movement speed
    alive: bool = True              # Is alive
    trapped_ticks: int = 0          # Remaining frames while trapped
    input_scheme: Dict[str, int] = field(default_factory=dict)  # Input scheme
    color: Tuple[int, int, int] = WHITE  # Player color
    active_bombs: int = 0           # Current number of active bombs
    player_id: int = 0              # Player ID
    move_cooldown: int = 0          # Movement cooldown

@dataclass
class Bomb:
    """Bomb data class"""
    owner_id: int                   # Owner ID
    grid_pos: Vector2               # Grid position
    timer: int = 300                # Countdown (5 seconds * 60 FPS), will be replaced by steps in the Gym environment
    fire: int = 1                   # Explosion range

@dataclass
class ItemDrop:
    """Dropped item data class"""
    item_type: Item                 # Item type
    grid_pos: Vector2               # Grid position

class AssetManager:
    """Asset Manager - handles image and sound loading"""
    
    def __init__(self):
        self.images = {}
        self.sounds = {}
        # Extended asset path list
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Determine the correct path to assets-necessay
        # Check if we're in the Blasting_Showdown directory
        if "Blasting_Showdown" in script_dir:
            # Go up to the game directory and then to assets-necessay
            assets_base = os.path.join(script_dir, "..", "assets-necessay")
        else:
            # Default fallback
            assets_base = os.path.join(script_dir, "assets-necessay")
        
        self.asset_paths = [
            # Bomb related
            os.path.join(assets_base, "kenney/3D assets/Platformer Kit/Previews/bomb.png"),
            os.path.join(assets_base, "kenney/2D assets/Platformer Assets Base/PNG/Items/bomb.png"),
            os.path.join(assets_base, "kenney/2D assets/Platformer Assets Base/PNG/Items/bombFlash.png"),
            os.path.join(assets_base, "kenney/2D assets/Platformer Pack Redux/PNG/Tiles/bomb.png"),
            os.path.join(assets_base, "kenney/2D assets/Platformer Pack Redux/PNG/Tiles/bombWhite.png"),
            # Character related
            os.path.join(assets_base, "kenney/3D assets/Mini Arcade/Previews/character-gamer.png"),
            os.path.join(assets_base, "kenney/3D assets/Mini Arcade/Previews/character-employee.png"),
            os.path.join(assets_base, "kenney/3D assets/Blocky Characters/Faces/face_robot.png"),
            os.path.join(assets_base, "kenney/2D assets/Abstract Platformer/PNG/Players/Player Green/playerGreen_stand.png"),
            os.path.join(assets_base, "kenney/2D assets/Abstract Platformer/PNG/Players/Player Blue/playerBlue_stand.png"),
            os.path.join(assets_base, "kenney/2D assets/Abstract Platformer/PNG/Players/Player Grey/playerGrey_stand.png"),
            # Explosion effects
            os.path.join(assets_base, "kenney/2D assets/Explosion Pack/PNG/Regular explosion/regularExplosion00.png"),
            os.path.join(assets_base, "kenney/2D assets/Explosion Pack/PNG/Regular explosion/regularExplosion01.png"),
            os.path.join(assets_base, "kenney/2D assets/Explosion Pack/PNG/Regular explosion/regularExplosion02.png"),
            os.path.join(assets_base, "kenney/2D assets/Explosion Pack/PNG/Regular explosion/regularExplosion03.png"),
            os.path.join(assets_base, "kenney/2D assets/Smoke Particles/PNG/Explosion/explosion00.png"),
            os.path.join(assets_base, "kenney/Icons/Board Game Icons/PNG/Double (128px)/fire.png"),
            os.path.join(assets_base, "kenney/2D assets/Particle Pack/PNG (Transparent)/fire_01.png"),
            # Blocks and walls
            os.path.join(assets_base, "kenney/2D assets/Sokoban Pack/PNG/Default size/Blocks/block_01.png"),
            os.path.join(assets_base, "kenney/2D assets/Sokoban Pack/PNG/Default size/Blocks/block_02.png"),
            os.path.join(assets_base, "kenney/2D assets/Sokoban Pack/PNG/Default size/Blocks/block_03.png"),
            os.path.join(assets_base, "kenney/2D assets/Sokoban Pack/PNG/Default size/Blocks/block_04.png"),
            os.path.join(assets_base, "kenney/2D assets/Sokoban Pack/PNG/Default size/Blocks/block_05.png"),
            os.path.join(assets_base, "kenney/2D assets/Sokoban Pack/PNG/Default size/Blocks/block_06.png"),
            os.path.join(assets_base, "kenney/2D assets/Sokoban Pack/PNG/Default size/Blocks/block_07.png"),
            os.path.join(assets_base, "kenney/2D assets/Sokoban Pack/PNG/Default size/Blocks/block_08.png"),
            # Damaged walls
            os.path.join(assets_base, "kenney/3D assets/Retro Urban Kit/Previews/wall-broken-type-a.png"),
            os.path.join(assets_base, "kenney/3D assets/Retro Urban Kit/Previews/wall-broken-type-b.png"),
            os.path.join(assets_base, "kenney/2D assets/Scribble Dungeons/PNG/Double (128px)/wall_damaged.png"),
            os.path.join(assets_base, "kenney/2D assets/Scribble Dungeons/PNG/Double (128px)/wall_demolished.png"),
            os.path.join(assets_base, "kenney/3D assets/Fantasy Town Kit/Previews/wallWoodBroken.png"),
            os.path.join(assets_base, "kenney/3D assets/Fantasy Town Kit/Previews/wallBroken.png"),
            # Audio
            os.path.join(assets_base, "kenney/Audio/Voiceover Pack Fighter/Audio/choose_your_character.ogg"),
            os.path.join(assets_base, "kenney/UI assets/UI Pack/Sounds/click-b.wav"),
            os.path.join(assets_base, "kenney/Audio/Retro Sounds 2/Audio/explosion1.wav"),
            os.path.join(assets_base, "kenney/Audio/Impact Sounds/Audio/footstep_wood_001.wav"),
        ]
        self.load_assets()
    def load_assets(self):
        """Load all available assets"""
        # Try to load image assets
        for path in self.asset_paths:
            if path.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    if os.path.exists(path):
                        image = pygame.image.load(path)
                        filename = os.path.basename(path).lower()
                        
                        # Bomb related
                        if 'bomb' in filename and 'flash' not in filename:
                            if 'white' in filename:
                                self.images['bomb_white'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                            else:
                                self.images['bomb'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                        elif 'bombflash' in filename:
                            self.images['bomb_flash'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                        
                        # Character related - using the unified Abstract Platformer series
                        elif 'playergreen_stand' in filename:
                            self.images['player_base'] = pygame.transform.scale(image, (TILE_SIZE-8, TILE_SIZE-8))
                        elif 'playerblue_stand' in filename:
                            self.images['player_blue_base'] = pygame.transform.scale(image, (TILE_SIZE-8, TILE_SIZE-8))
                        elif 'playergrey_stand' in filename:
                            self.images['player_grey_base'] = pygame.transform.scale(image, (TILE_SIZE-8, TILE_SIZE-8))
                        
                        # Fallback character images
                        elif 'character' in filename or 'gamer' in filename or 'employee' in filename:
                            if 'gamer' in filename:
                                self.images['fallback_player1'] = pygame.transform.scale(image, (TILE_SIZE-8, TILE_SIZE-8))
                            elif 'employee' in filename:
                                self.images['fallback_player2'] = pygame.transform.scale(image, (TILE_SIZE-8, TILE_SIZE-8))
                        elif 'face_robot' in filename:
                            self.images['fallback_player3'] = pygame.transform.scale(image, (TILE_SIZE-8, TILE_SIZE-8))
                        
                        # Explosion effects
                        elif 'explosion' in filename or 'fire' in filename:
                            if 'regularexplosion' in filename:
                                frame_num = filename.split('explosion')[-1].split('.')[0]
                                self.images[f'explosion_{frame_num}'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                            elif filename == 'fire.png':
                                self.images['fire_icon'] = pygame.transform.scale(image, (TILE_SIZE//2, TILE_SIZE//2))
                            elif 'fire_01' in filename:
                                self.images['flame'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                        
                        # Blocks and walls
                        elif 'block_' in filename:
                            block_num = filename.split('block_')[-1].split('.')[0]
                            self.images[f'block_{block_num}'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                        elif 'wall' in filename:
                            if 'broken' in filename:
                                if 'type-a' in filename:
                                    self.images['wall_broken_a'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                                elif 'type-b' in filename:
                                    self.images['wall_broken_b'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                                elif 'wood' in filename:
                                    self.images['wall_wood_broken'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                                else:
                                    self.images['wall_broken'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                            elif 'damaged' in filename:
                                self.images['wall_damaged'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                            elif 'demolished' in filename:
                                self.images['wall_demolished'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                        
                        # Floor tiles
                        elif 'tile_' in filename:
                            tile_num = filename.split('tile_')[-1].split('.')[0]
                            self.images[f'floor_{tile_num}'] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                        
                except Exception as e:
                    print(f"Could not load image {path}: {e}")
        
        # Try to load sound assets
        for path in self.asset_paths:
            if path.endswith(('.ogg', '.wav', '.mp3')):
                try:
                    if os.path.exists(path):
                        sound = pygame.mixer.Sound(path)
                        filename = os.path.basename(path).lower()
                        if 'choose' in filename:
                            self.sounds['game_start'] = sound
                        elif 'click-b' in filename:  # Bomb placement sound effect
                            self.sounds['bomb_place'] = sound
                        elif 'explosion' in filename:  # Bomb explosion sound effect
                            self.sounds['bomb_explode'] = sound
                        elif 'footstep' in filename:  # Player walking sound effect
                            self.sounds['player_walk'] = sound
                except Exception as e:
                    print(f"Could not load sound {path}: {e}")
        
        # Create player sprites from base image
        self.create_player_sprites()
        
        # Create default floor pattern
        self.create_floor_pattern()
    
    def create_player_sprites(self):
        """Create colored player sprites from base image"""
        # Get base player image
        base_image = None
        
        # Prioritize using characters from the Abstract Platformer series
        if 'player_base' in self.images:
            base_image = self.images['player_base']
        elif 'player_blue_base' in self.images:
            base_image = self.images['player_blue_base']
        elif 'player_grey_base' in self.images:
            base_image = self.images['player_grey_base']
        
        if base_image:
            # Create four player sprites with different styles
            self.create_varied_players(base_image)
        else:
            # If no base image is found, create simple geometric players
            self.create_fallback_players()
    
    def create_varied_players(self, base_image):
        """Create four visually distinct player variants from base image"""
        player_configs = [
            {
                'color': (255, 80, 80),     # Red - Player 1
                'brightness': 1.2,
                'contrast': 1.1,
                'hue_shift': 0,
                'outline_color': (200, 0, 0),
                'effect': 'normal'
            },
            {
                'color': (80, 80, 255),     # Blue - Player 2
                'brightness': 1.0,
                'contrast': 1.3,
                'hue_shift': 0,
                'outline_color': (0, 0, 200),
                'effect': 'metallic'
            },
            {
                'color': (80, 255, 80),     # Green - Player 3
                'brightness': 1.1,
                'contrast': 1.0,
                'hue_shift': 0,
                'outline_color': (0, 200, 0),
                'effect': 'glow'
            },
            {
                'color': (255, 255, 80),    # Yellow - Player 4
                'brightness': 1.3,
                'contrast': 0.9,
                'hue_shift': 0,
                'outline_color': (200, 200, 0),
                'effect': 'shadow'
            }
        ]
        
        for i, config in enumerate(player_configs):
            # Create a copy of the base sprite
            sprite = base_image.copy()
            
            # Apply main color tint
            color_overlay = pygame.Surface(sprite.get_size())
            color_overlay.fill(config['color'])
            sprite.blit(color_overlay, (0, 0), special_flags=pygame.BLEND_MULT)
            
            # Apply brightness adjustment
            if config['brightness'] != 1.0:
                brightness_overlay = pygame.Surface(sprite.get_size())
                brightness_value = int(255 * (config['brightness'] - 1.0))
                if brightness_value > 0:
                    brightness_overlay.fill((brightness_value, brightness_value, brightness_value))
                    sprite.blit(brightness_overlay, (0, 0), special_flags=pygame.BLEND_ADD)
                else:
                    brightness_overlay.fill((-brightness_value, -brightness_value, -brightness_value))
                    sprite.blit(brightness_overlay, (0, 0), special_flags=pygame.BLEND_SUB)
            
            # Apply special effect
            sprite = self.apply_player_effect(sprite, config['effect'], config['outline_color'])
            
            # Store the player sprite
            self.images[f'player{i+1}'] = sprite
    
    def apply_player_effect(self, sprite, effect, outline_color):
        """Apply special visual effects to player sprites"""
        if effect == 'metallic':
            # Metallic effect - add highlights and reflections
            enhanced_sprite = sprite.copy()
            highlight = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
            
            # Add top-left highlight
            w, h = sprite.get_size()
            for y in range(h//3):
                for x in range(w//3):
                    alpha = max(0, 100 - (x + y) * 2)
                    highlight.set_at((x, y), (255, 255, 255, alpha))
            
            enhanced_sprite.blit(highlight, (0, 0))
            return enhanced_sprite
            
        elif effect == 'glow':
            # Glow effect - add outer glow
            glowing_sprite = pygame.Surface((sprite.get_width() + 4, sprite.get_height() + 4), pygame.SRCALPHA)
            
            # Create glow background
            for offset in range(1, 3):
                glow_surface = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
                glow_surface.fill(outline_color + (80//offset,))
                for dx in range(-offset, offset+1):
                    for dy in range(-offset, offset+1):
                        if dx*dx + dy*dy <= offset*offset:
                            glowing_sprite.blit(glow_surface, (2+dx, 2+dy))
            
            # Overlay the original sprite
            glowing_sprite.blit(sprite, (2, 2))
            return glowing_sprite
            
        elif effect == 'shadow':
            # Shadow effect - add a dark outline
            shadow_sprite = pygame.Surface((sprite.get_width() + 2, sprite.get_height() + 2), pygame.SRCALPHA)
            
            # Create shadow
            shadow_surface = pygame.Surface(sprite.get_size())
            shadow_surface.fill((30, 30, 30))
            shadow_surface.set_alpha(120)
            
            # Multi-directional shadow
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        shadow_sprite.blit(shadow_surface, (1+dx, 1+dy))
            
            # Overlay the original sprite
            shadow_sprite.blit(sprite, (1, 1))
            return shadow_sprite
            
        else:  # normal
            # Normal effect - add a simple outline
            outlined_sprite = pygame.Surface((sprite.get_width() + 2, sprite.get_height() + 2), pygame.SRCALPHA)
            
            # Create outline
            outline_surface = pygame.Surface(sprite.get_size())
            outline_surface.fill(outline_color)
            outline_surface.set_alpha(150)
            
            # 4-directional outline
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                outlined_sprite.blit(outline_surface, (1+dx, 1+dy))
            
            # Overlay the original sprite
            outlined_sprite.blit(sprite, (1, 1))
            return outlined_sprite

    def create_fallback_players(self):
        """Create fallback geometric player sprites with distinct shapes"""
        player_configs = [
            {
                'color': RED,
                'shape': 'circle',
                'pattern': 'solid',
                'size_modifier': 1.0
            },
            {
                'color': BLUE,
                'shape': 'square',
                'pattern': 'striped',
                'size_modifier': 0.9
            },
            {
                'color': GREEN,
                'shape': 'triangle',
                'pattern': 'dotted',
                'size_modifier': 1.1
            },
            {
                'color': YELLOW,
                'shape': 'diamond',
                'pattern': 'checkered',
                'size_modifier': 1.0
            }
        ]
        
        for i, config in enumerate(player_configs):
            sprite_size = int((TILE_SIZE-8) * config['size_modifier'])
            player_surface = pygame.Surface((TILE_SIZE-8, TILE_SIZE-8), pygame.SRCALPHA)
            center = ((TILE_SIZE-8)//2, (TILE_SIZE-8)//2)
            
            # Draw shadow
            shadow_offset = 2
            if config['shape'] == 'circle':
                pygame.draw.circle(player_surface, (50, 50, 50), 
                                 (center[0]+shadow_offset, center[1]+shadow_offset), sprite_size//3)
            elif config['shape'] == 'square':
                rect = pygame.Rect(center[0] - sprite_size//3 + shadow_offset, 
                                 center[1] - sprite_size//3 + shadow_offset, 
                                 sprite_size//3*2, sprite_size//3*2)
                pygame.draw.rect(player_surface, (50, 50, 50), rect)
            elif config['shape'] == 'triangle':
                points = [
                    (center[0] + shadow_offset, center[1] - sprite_size//3 + shadow_offset),
                    (center[0] - sprite_size//3 + shadow_offset, center[1] + sprite_size//3 + shadow_offset),
                    (center[0] + sprite_size//3 + shadow_offset, center[1] + sprite_size//3 + shadow_offset)
                ]
                pygame.draw.polygon(player_surface, (50, 50, 50), points)
            elif config['shape'] == 'diamond':
                points = [
                    (center[0] + shadow_offset, center[1] - sprite_size//3 + shadow_offset),
                    (center[0] + sprite_size//3 + shadow_offset, center[1] + shadow_offset),
                    (center[0] + shadow_offset, center[1] + sprite_size//3 + shadow_offset),
                    (center[0] - sprite_size//3 + shadow_offset, center[1] + shadow_offset)
                ]
                pygame.draw.polygon(player_surface, (50, 50, 50), points)
            
            # Draw main shape
            if config['shape'] == 'circle':
                pygame.draw.circle(player_surface, config['color'], center, sprite_size//3)
                if config['pattern'] == 'dotted':
                    for angle in range(0, 360, 45):
                        dot_x = center[0] + int((sprite_size//6) * math.cos(math.radians(angle)))
                        dot_y = center[1] + int((sprite_size//6) * math.sin(math.radians(angle)))
                        pygame.draw.circle(player_surface, WHITE, (dot_x, dot_y), 2)
                        
            elif config['shape'] == 'square':
                rect = pygame.Rect(center[0] - sprite_size//3, center[1] - sprite_size//3, 
                                 sprite_size//3*2, sprite_size//3*2)
                pygame.draw.rect(player_surface, config['color'], rect)
                if config['pattern'] == 'striped':
                    for i in range(0, sprite_size//3*2, 4):
                        line_rect = pygame.Rect(rect.x, rect.y + i, rect.width, 2)
                        pygame.draw.rect(player_surface, WHITE, line_rect)
                        
            elif config['shape'] == 'triangle':
                points = [
                    (center[0], center[1] - sprite_size//3),
                    (center[0] - sprite_size//3, center[1] + sprite_size//3),
                    (center[0] + sprite_size//3, center[1] + sprite_size//3)
                ]
                pygame.draw.polygon(player_surface, config['color'], points)
                
            elif config['shape'] == 'diamond':
                points = [
                    (center[0], center[1] - sprite_size//3),
                    (center[0] + sprite_size//3, center[1]),
                    (center[0], center[1] + sprite_size//3),
                    (center[0] - sprite_size//3, center[1])
                ]
                pygame.draw.polygon(player_surface, config['color'], points)
                if config['pattern'] == 'checkered':
                    # Add checkered effect
                    for dx in range(-sprite_size//6, sprite_size//6, 4):
                        for dy in range(-sprite_size//6, sprite_size//6, 4):
                            if (dx//4 + dy//4) % 2:
                                check_rect = pygame.Rect(center[0] + dx, center[1] + dy, 3, 3)
                                pygame.draw.rect(player_surface, WHITE, check_rect)
            
            # Add border
            if config['shape'] == 'circle':
                pygame.draw.circle(player_surface, BLACK, center, sprite_size//3, 2)
            elif config['shape'] == 'square':
                rect = pygame.Rect(center[0] - sprite_size//3, center[1] - sprite_size//3, 
                                 sprite_size//3*2, sprite_size//3*2)
                pygame.draw.rect(player_surface, BLACK, rect, 2)
            elif config['shape'] == 'triangle':
                points = [
                    (center[0], center[1] - sprite_size//3),
                    (center[0] - sprite_size//3, center[1] + sprite_size//3),
                    (center[0] + sprite_size//3, center[1] + sprite_size//3)
                ]
                pygame.draw.polygon(player_surface, BLACK, points, 2)
            elif config['shape'] == 'diamond':
                points = [
                    (center[0], center[1] - sprite_size//3),
                    (center[0] + sprite_size//3, center[1]),
                    (center[0], center[1] + sprite_size//3),
                    (center[0] - sprite_size//3, center[1])
                ]
                pygame.draw.polygon(player_surface, BLACK, points, 2)
            
            # Add player number
            font = pygame.font.Font(None, 16)
            number_text = font.render(str(i+1), True, WHITE)
            number_rect = number_text.get_rect(center=center)
            player_surface.blit(number_text, number_rect)
            
            self.images[f'player{i+1}'] = player_surface
    
    def create_floor_pattern(self):
        """Create dark themed floor texture patterns"""
        # Create dark minimal floor - dark gray
        floor_surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
        floor_surface.fill((45, 45, 45))  # Dark gray
        # Add subtle grid lines
        pygame.draw.line(floor_surface, (55, 55, 55), (TILE_SIZE-1, 0), (TILE_SIZE-1, TILE_SIZE-1), 1)
        pygame.draw.line(floor_surface, (55, 55, 55), (0, TILE_SIZE-1), (TILE_SIZE-1, TILE_SIZE-1), 1)
        self.images['clean_floor'] = floor_surface
        
        # Create alternative dark floor - slightly lighter
        floor_surface2 = pygame.Surface((TILE_SIZE, TILE_SIZE))
        floor_surface2.fill((50, 50, 50))
        pygame.draw.line(floor_surface2, (60, 60, 60), (TILE_SIZE-1, 0), (TILE_SIZE-1, TILE_SIZE-1), 1)
        pygame.draw.line(floor_surface2, (60, 60, 60), (0, TILE_SIZE-1), (TILE_SIZE-1, TILE_SIZE-1), 1)
        self.images['clean_floor_alt'] = floor_surface2
        
        # Create third dark variation
        floor_surface3 = pygame.Surface((TILE_SIZE, TILE_SIZE))
        floor_surface3.fill((40, 40, 40))
        pygame.draw.line(floor_surface3, (50, 50, 50), (TILE_SIZE-1, 0), (TILE_SIZE-1, TILE_SIZE-1), 1)
        pygame.draw.line(floor_surface3, (50, 50, 50), (0, TILE_SIZE-1), (TILE_SIZE-1, TILE_SIZE-1), 1)
        self.images['clean_floor_bright'] = floor_surface3
        
        # Create checkered dark floor pattern
        floor_surface4 = pygame.Surface((TILE_SIZE, TILE_SIZE))
        floor_surface4.fill((42, 42, 42))
        # Add subtle checkered pattern
        for i in range(0, TILE_SIZE, 16):
            for j in range(0, TILE_SIZE, 16):
                if (i//16 + j//16) % 2:
                    check_rect = pygame.Rect(i, j, 16, 16)
                    pygame.draw.rect(floor_surface4, (48, 48, 48), check_rect)
        self.images['clean_floor_checkered'] = floor_surface4
        
        # Create improved hard wall texture
        hard_wall_surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
        hard_wall_surface.fill((90, 90, 90))
        # Add simple brick pattern
        for i in range(0, TILE_SIZE, 16):
            for j in range(0, TILE_SIZE, 8):
                offset = 8 if (j // 8) % 2 else 0
                brick_rect = pygame.Rect(i + offset, j, 14, 6)
                pygame.draw.rect(hard_wall_surface, (110, 110, 110), brick_rect)
                pygame.draw.rect(hard_wall_surface, (70, 70, 70), brick_rect, 1)
        self.images['default_hard_wall'] = hard_wall_surface
        
        # Create soft wall texture
        soft_wall_surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
        soft_wall_surface.fill((139, 90, 43))  # Brown
        # Add wood grain effect
        for i in range(0, TILE_SIZE, 8):
            for j in range(0, TILE_SIZE, 8):
                wood_rect = pygame.Rect(i, j, 6, 6)
                pygame.draw.rect(soft_wall_surface, (160, 110, 60), wood_rect)
                pygame.draw.rect(soft_wall_surface, (120, 70, 30), wood_rect, 1)
        self.images['default_soft_wall'] = soft_wall_surface
    
    def get_image(self, name: str) -> Optional[pygame.Surface]:
        """Get image resource"""
        return self.images.get(name)
    
    def get_sound(self, name: str) -> Optional[pygame.mixer.Sound]:
        """Get sound resource"""
        return self.sounds.get(name)
    
    def get_random_floor_tile(self, x: int, y: int) -> pygame.Surface:
        """Get random dark floor tile based on position"""
        # Use position as seed to ensure same tile at same position
        random.seed(x * 1000 + y)
        
        # Use only dark floor variants
        dark_floors = ['clean_floor', 'clean_floor_alt', 'clean_floor_bright', 'clean_floor_checkered']
        tile_name = random.choice(dark_floors)
        return self.images[tile_name]
    
    def get_wall_texture(self, wall_type: str, x: int, y: int) -> pygame.Surface:
        """Get wall texture based on type and position"""
        random.seed(x * 1000 + y)
        
        if wall_type == 'soft':
            # Soft walls use wood or block textures
            soft_wall_options = ['wall_wood_broken', 'block_01', 'block_02', 'block_03']
            for option in soft_wall_options:
                if option in self.images:
                    return self.images[option]
            return self.images['default_soft_wall']
        elif wall_type == 'hard':
            # Hard walls use stone textures
            hard_wall_options = ['block_04', 'block_05', 'block_06', 'block_07', 'block_08']
            for option in hard_wall_options:
                if option in self.images:
                    return self.images[option]
            return self.images['default_hard_wall']
        
        # Default texture
        return self.images.get('default_hard_wall')

class GameState:
    """Game State Manager"""
    
    def __init__(self, difficulty=DifficultyLevel.NORMAL):
        # Save difficulty setting
        self.difficulty = difficulty
        self.difficulty_config = get_difficulty_config(difficulty)
        
        # Set grid size based on difficulty config
        global GRID_WIDTH, GRID_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT
        GRID_WIDTH = self.difficulty_config.grid_width
        GRID_HEIGHT = self.difficulty_config.grid_height
        WINDOW_WIDTH = GRID_WIDTH * TILE_SIZE
        WINDOW_HEIGHT = GRID_HEIGHT * TILE_SIZE
        
        self.grid = [[Tile.FLOOR for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.players: List[Player] = []
        self.bombs: List[Bomb] = []
        self.items: List[ItemDrop] = []
        self.flames = {}  # Changed to a dictionary: (x, y) -> remaining time
        self.flame_duration = 60  # Flame duration in frames
        self.game_over = False
        self.winner = None
        self.asset_manager = AssetManager()
        
        # Explosion callback function
        self.register_explosion_callback = None
    
        # Initialize map
        self.init_arena()
        
        # Initialize players
        self.init_players()
    
    def init_arena(self):
        """Initialize the arena map"""
        # Place hard walls at the intersections of odd rows and columns
        for y in range(1, GRID_HEIGHT, 2):
            for x in range(1, GRID_WIDTH, 2):
                self.grid[y][x] = Tile.HARD
        
        # Randomly place soft walls according to difficulty config
        soft_wall_chance = self.difficulty_config.soft_wall_chance
        clear_radius = self.difficulty_config.clear_radius
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] == Tile.FLOOR:
                    # Skip the clear_radius x clear_radius areas in the four corners (player spawn points)
                    if ((x < clear_radius and y < clear_radius) or 
                        (x >= GRID_WIDTH-clear_radius and y < clear_radius) or
                        (x < clear_radius and y >= GRID_HEIGHT-clear_radius) or
                        (x >= GRID_WIDTH-clear_radius and y >= GRID_HEIGHT-clear_radius)):
                        continue
                    
                    if random.random() < soft_wall_chance:
                        self.grid[y][x] = Tile.SOFT
    
    def init_players(self):
        """Initialize players"""
        # Player spawn positions (four corners)
        spawn_positions = [
            Vector2(0, 0),                              # Top-left
            Vector2(GRID_WIDTH-1, 0),                   # Top-right
            Vector2(0, GRID_HEIGHT-1),                  # Bottom-left
            Vector2(GRID_WIDTH-1, GRID_HEIGHT-1)        # Bottom-right
        ]
        
        # Input schemes
        input_schemes = [
            {'up': pygame.K_w, 'down': pygame.K_s, 'left': pygame.K_a, 'right': pygame.K_d, 'bomb': pygame.K_SPACE},
            {'up': pygame.K_UP, 'down': pygame.K_DOWN, 'left': pygame.K_LEFT, 'right': pygame.K_RIGHT, 'bomb': pygame.K_RETURN},
            {'up': pygame.K_t, 'down': pygame.K_g, 'left': pygame.K_f, 'right': pygame.K_h, 'bomb': pygame.K_r},
            {'up': pygame.K_i, 'down': pygame.K_k, 'left': pygame.K_j, 'right': pygame.K_l, 'bomb': pygame.K_u}
        ]
        
        # Player colors
        colors = [RED, BLUE, GREEN, YELLOW]
        
        for i in range(4):
            player = Player(
                pos=spawn_positions[i],
                input_scheme=input_schemes[i],
                color=colors[i],
                player_id=i,
                speed=self.difficulty_config.player_speed  # Use speed from difficulty config
            )
            self.players.append(player)

    def handle_input(self, keys):
        """Handle player input"""
        for player in self.players:
            if not player.alive:
                continue
            
            # Update movement cooldown
            if player.move_cooldown > 0:
                player.move_cooldown -= 1
                continue
            
            # Movement input
            new_pos = Vector2(player.pos.x, player.pos.y)
            moved = False
            
            if keys[player.input_scheme['up']]:
                new_pos.y -= 1
                moved = True
                print("PLAYER_MOVED")
            elif keys[player.input_scheme['down']]:
                new_pos.y += 1
                moved = True
                print("PLAYER_MOVED")
            elif keys[player.input_scheme['left']]:
                new_pos.x -= 1
                moved = True
                print("PLAYER_MOVED")
            elif keys[player.input_scheme['right']]:
                new_pos.x += 1
                moved = True
                print("PLAYER_MOVED")
            
            # Check if move is valid
            if moved and self.is_valid_move(new_pos):
                player.pos = new_pos
                # Set movement cooldown, adjusted based on player speed
                base_cooldown = MOVE_DELAY
                speed_modifier = max(1, player.speed)
                player.move_cooldown = max(1, base_cooldown - (speed_modifier - 1) * 2)
                
                # Play walking sound effect
                player_walk_sound = self.asset_manager.get_sound('player_walk')
                if player_walk_sound:
                    player_walk_sound.play()
                
                # Check for item pickup
                self.check_item_pickup(player)
            
            # Bomb placement input
            if keys[player.input_scheme['bomb']]:
                self.place_bomb(player)
    
    def is_valid_move(self, pos: Vector2) -> bool:
        """Check if move is valid"""
        # Boundary check
        if pos.x < 0 or pos.x >= GRID_WIDTH or pos.y < 0 or pos.y >= GRID_HEIGHT:
            return False
        
        # Obstacle check
        tile = self.grid[pos.y][pos.x]
        if tile in [Tile.HARD, Tile.SOFT, Tile.BOMB]:
            return False
        
        return True
    
    def place_bomb(self, player: Player):
        """Place a bomb"""
        # Check bomb count limit
        if player.active_bombs >= player.bombs:
            return
        
        # Check if a bomb is already at the current position
        bomb_pos = player.pos
        for bomb in self.bombs:
            if bomb.grid_pos.x == bomb_pos.x and bomb.grid_pos.y == bomb_pos.y:
                return
        
        # Place bomb
        bomb = Bomb(
            owner_id=player.player_id,
            grid_pos=Vector2(bomb_pos.x, bomb_pos.y),
            fire=player.fire
        )
        self.bombs.append(bomb)
        self.grid[bomb_pos.y][bomb_pos.x] = Tile.BOMB
        player.active_bombs += 1
        print("BOMB_PLACED")
        
        # Play bomb placement sound effect
        bomb_place_sound = self.asset_manager.get_sound('bomb_place')
        if bomb_place_sound:
            bomb_place_sound.play()
    
    def update_bombs(self):
        """Update bomb state"""
        bombs_to_remove = []
        
        for bomb in self.bombs:
            bomb.timer -= 1
            
            if bomb.timer <= 0:
                self.explode_bomb(bomb)
                bombs_to_remove.append(bomb)
        
        # Remove exploded bombs
        for bomb in bombs_to_remove:
            self.bombs.remove(bomb)
            self.grid[bomb.grid_pos.y][bomb.grid_pos.x] = Tile.FLOOR
            # Decrease player's active bomb count
            if bomb.owner_id < len(self.players):
                self.players[bomb.owner_id].active_bombs -= 1
    
    def explode_bomb(self, bomb: Bomb):
        """Handle bomb explosion"""
        print("BOMB_EXPLODED")
        
        # Play explosion sound effect
        bomb_explode_sound = self.asset_manager.get_sound('bomb_explode')
        if bomb_explode_sound:
            bomb_explode_sound.play()
        
        # Explosion center
        center_x, center_y = bomb.grid_pos.x, bomb.grid_pos.y
        self.flames[(center_x, center_y)] = self.flame_duration
        
        # Record all affected positions
        affected_positions = {(center_x, center_y)}
        
        # Flame propagation in four directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Down, Up, Right, Left
        
        for dx, dy in directions:
            for i in range(1, bomb.fire + 1):
                flame_x = center_x + dx * i
                flame_y = center_y + dy * i
                
                # Boundary check
                if flame_x < 0 or flame_x >= GRID_WIDTH or flame_y < 0 or flame_y >= GRID_HEIGHT:
                    break
                
                tile = self.grid[flame_y][flame_x]
                
                # Hard wall blocks flame
                if tile == Tile.HARD:
                    break
                
                # Add flame
                self.flames[(flame_x, flame_y)] = self.flame_duration
                affected_positions.add((flame_x, flame_y))
                
                # Soft wall is destroyed and blocks flame
                if tile == Tile.SOFT:
                    self.grid[flame_y][flame_x] = Tile.FLOOR
                    # 20% chance to drop an item
                    if random.random() < 0.2:
                        item_type = random.choice(list(Item))
                        self.items.append(ItemDrop(item_type, Vector2(flame_x, flame_y)))
                    break
                
                # Bomb chain reaction
                if tile == Tile.BOMB:
                    for other_bomb in self.bombs:
                        if (other_bomb.grid_pos.x == flame_x and 
                            other_bomb.grid_pos.y == flame_y):
                            other_bomb.timer = 0  # Explode immediately
                            break
        
        # If an explosion callback exists, call it
        if self.register_explosion_callback:
            self.register_explosion_callback(bomb, affected_positions)
        
        # Check if players are hit by flames
        self.check_flame_damage()
    
    def update_flames(self):
        """Update flame state"""
        expired_flames = []
        
        # Update the timer for each flame position
        for pos, timer in self.flames.items():
            # Decrease timer
            self.flames[pos] = timer - 1
            
            # If timer reaches zero, mark as expired
            if self.flames[pos] <= 0:
                expired_flames.append(pos)
        
        # Remove expired flames
        for pos in expired_flames:
            self.flames.pop(pos)
    
    def check_flame_damage(self):
        """Check for flame damage"""
        for player in self.players:
            if not player.alive:
                continue
            
            player_pos = (player.pos.x, player.pos.y)
            if player_pos in self.flames:
                if player.trapped_ticks > 0:
                    # Already trapped, die immediately
                    player.alive = False
                else:
                    # Trapped for the first time
                    player.trapped_ticks = 150  # 2.5 seconds
                    
    def update_players(self):
        """Update player state"""
        for player in self.players:
            if not player.alive:
                continue
            
            # Update trapped status
            if player.trapped_ticks > 0:
                player.trapped_ticks -= 1
                if player.trapped_ticks <= 0:
                    # Check for teammate rescue
                    rescued = False
                    for other_player in self.players:
                        if (other_player != player and other_player.alive and
                            other_player.pos.x == player.pos.x and
                            other_player.pos.y == player.pos.y):
                            rescued = True
                            break
                    
                    if not rescued:
                        player.alive = False
    
    def check_item_pickup(self, player: Player):
        """Check for item pickup"""
        items_to_remove = []
        
        for item in self.items:
            if (item.grid_pos.x == player.pos.x and 
                item.grid_pos.y == player.pos.y):
                
                # Apply item effect
                if item.item_type == Item.FIRE_UP:
                    player.fire = min(player.fire + 1, 8)
                elif item.item_type == Item.BOMB_UP:
                    player.bombs = min(player.bombs + 1, 8)
                elif item.item_type == Item.SPEED_UP:
                    player.speed = min(player.speed + 1, 8)  # Speed boost will increase max move distance
            
                items_to_remove.append(item)
                print("ITEM_COLLECTED")
                
                # Play pickup sound effect
                pickup_sound = self.asset_manager.get_sound('pickup')
                if pickup_sound:
                    pickup_sound.play()
        
        # Remove picked-up items
        for item in items_to_remove:
            self.items.remove(item)
    
    def check_win_condition(self):
        """Check for win condition"""
        alive_players = [p for p in self.players if p.alive]
        
        if len(alive_players) <= 1:
            self.game_over = True
            self.winner = alive_players[0] if alive_players else None
            print("GAME_OVER")
    
    def update(self):
        """Update game state"""
        if self.game_over:
            return
        
        self.update_bombs()
        self.update_flames()
        self.update_players()
        self.check_win_condition()
    
    def reset(self, difficulty=None):
        """Reset the game"""
        if difficulty is None:
            difficulty = self.difficulty
        self.__init__(difficulty=difficulty)

    def to_dict(self):
        """Convert game state to a dictionary for serialization"""
        state = {
            "grid": [[tile.value for tile in row] for row in self.grid],
            "players": [],
            "bombs": [],
            "items": [],
            "flames": list(self.flames),
            "flame_timer": self.flame_timer,
            "game_over": self.game_over,
            "winner": self.winner.player_id if self.winner else None
        }
        
        # Serialize players
        for player in self.players:
            player_data = {
                "pos_x": player.pos.x,
                "pos_y": player.pos.y,
                "fire": player.fire,
                "bombs": player.bombs,
                "speed": player.speed,
                "alive": player.alive,
                "trapped_ticks": player.trapped_ticks,
                "input_scheme": player.input_scheme,
                "color": player.color,
                "active_bombs": player.active_bombs,
                "player_id": player.player_id,
                "move_cooldown": player.move_cooldown
            }
            state["players"].append(player_data)
        
        # Serialize bombs
        for bomb in self.bombs:
            bomb_data = {
                "owner_id": bomb.owner_id,
                "grid_pos_x": bomb.grid_pos.x,
                "grid_pos_y": bomb.grid_pos.y,
                "timer": bomb.timer,
                "fire": bomb.fire
            }
            state["bombs"].append(bomb_data)
        
        # Serialize items
        for item in self.items:
            item_data = {
                "item_type": item.item_type.value,
                "grid_pos_x": item.grid_pos.x,
                "grid_pos_y": item.grid_pos.y
            }
            state["items"].append(item_data)
            
        return state
    
    @classmethod
    def from_dict(cls, state_dict, asset_manager=None):
        """Create a game state instance from a dictionary"""
        state = cls()
        
        # If an asset_manager is provided, use it
        if asset_manager:
            state.asset_manager = asset_manager
            
        # Restore grid
        for y, row in enumerate(state_dict["grid"]):
            for x, cell in enumerate(row):
                state.grid[y][x] = Tile(cell)
        
        # Clear current player, bomb, and item lists
        state.players = []
        state.bombs = []
        state.items = []
        
        # Restore players
        for player_data in state_dict["players"]:
            player = Player(
                pos=Vector2(player_data["pos_x"], player_data["pos_y"]),
                fire=player_data["fire"],
                bombs=player_data["bombs"],
                speed=player_data["speed"],
                alive=player_data["alive"],
                trapped_ticks=player_data["trapped_ticks"],
                input_scheme=player_data["input_scheme"],
                color=player_data["color"],
                active_bombs=player_data["active_bombs"],
                player_id=player_data["player_id"],
                move_cooldown=player_data["move_cooldown"]
            )
            state.players.append(player)
        
        # Restore bombs
        for bomb_data in state_dict["bombs"]:
            bomb = Bomb(
                owner_id=bomb_data["owner_id"],
                grid_pos=Vector2(bomb_data["grid_pos_x"], bomb_data["grid_pos_y"]),
                timer=bomb_data["timer"],
                fire=bomb_data["fire"]
            )
            state.bombs.append(bomb)
        
        # Restore items
        for item_data in state_dict["items"]:
            item = ItemDrop(
                item_type=Item(item_data["item_type"]),
                grid_pos=Vector2(item_data["grid_pos_x"], item_data["grid_pos_y"])
            )
            state.items.append(item)
        
        # Restore other state
        state.flames = set(tuple(pos) for pos in state_dict["flames"])
        state.flame_timer = state_dict["flame_timer"]
        state.game_over = state_dict["game_over"]
        
        # Restore winner
        if state_dict["winner"] is not None:
            for player in state.players:
                if player.player_id == state_dict["winner"]:
                    state.winner = player
                    break
        else:
            state.winner = None
            
        return state

    # To support step-based countdown instead of frame-based, add a variant of the bomb update method
    def update_bombs_steps(self):
        """Update bomb state (using step-based timing)"""
        bombs_to_remove = []
        
        for bomb in self.bombs:
            bomb.timer -= 1
            
            if bomb.timer <= 0:
                self.explode_bomb(bomb)
                bombs_to_remove.append(bomb)
        
        # Remove exploded bombs
        for bomb in bombs_to_remove:
            self.bombs.remove(bomb)
            self.grid[bomb.grid_pos.y][bomb.grid_pos.x] = Tile.FLOOR
            # Decrease player's active bomb count
            if bomb.owner_id < len(self.players):
                self.players[bomb.owner_id].active_bombs -= 1

class Renderer:
    """Renderer Class - responsible for game rendering"""
    
    def __init__(self, screen, game_state: GameState):
        self.screen = screen
        self.game_state = game_state
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.explosion_frame = 0  # Explosion animation frame
    
    def render(self):
        """Render the game screen"""
        # Dark background to match dark floor theme
        self.screen.fill((25, 25, 25))  # Very dark gray background
        
        # Render map
        self.render_grid()
        
        # Render items
        self.render_items()
        
        # Render bombs
        self.render_bombs()
        
        # Render flames
        self.render_flames()
        
        # Render players
        self.render_players()
        
        # Render UI
        self.render_ui()
        
        # Render game over screen
        if self.game_state.game_over:
            self.render_game_over()
        
        # Update animation frame
        self.explosion_frame = (self.explosion_frame + 1) % 240  # 4 second loop
        
        pygame.display.flip()
    
    def render_grid(self):
        """Render the map grid"""
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                tile = self.game_state.grid[y][x]
                
                if tile == Tile.FLOOR:
                    # Use random floor tiles
                    floor_tile = self.game_state.asset_manager.get_random_floor_tile(x, y)
                    self.screen.blit(floor_tile, (x * TILE_SIZE, y * TILE_SIZE))
                elif tile == Tile.SOFT:
                    # Soft walls use wood texture
                    soft_wall_texture = self.game_state.asset_manager.get_wall_texture('soft', x, y)
                    self.screen.blit(soft_wall_texture, (x * TILE_SIZE, y * TILE_SIZE))
                elif tile == Tile.HARD:
                    # Hard walls use stone texture
                    hard_wall_texture = self.game_state.asset_manager.get_wall_texture('hard', x, y)
                    self.screen.blit(hard_wall_texture, (x * TILE_SIZE, y * TILE_SIZE))
    
    def render_items(self):
        """Render items"""
        for item in self.game_state.items:
            item_image = self.game_state.asset_manager.get_image(item.item_type.name.lower())
            if item_image:
                self.screen.blit(item_image, (item.grid_pos.x * TILE_SIZE, item.grid_pos.y * TILE_SIZE))
    
    def render_bombs(self):
        """Render bombs"""
        for bomb in self.game_state.bombs:
            bomb_image = self.game_state.asset_manager.get_image('bomb')
            if bomb_image:
                self.screen.blit(bomb_image, (bomb.grid_pos.x * TILE_SIZE, bomb.grid_pos.y * TILE_SIZE))
    
    def render_flames(self):
        """Render flames"""
        flame_images = [self.game_state.asset_manager.get_image(f'explosion_{i:02}') for i in range(4)]
        
        for (flame_x, flame_y) in self.game_state.flames:
            for image in flame_images:
                if image:
                    self.screen.blit(image, (flame_x * TILE_SIZE, flame_y * TILE_SIZE))
    
    def render_players(self):
        """Render players"""
        for player in self.game_state.players:
            if not player.alive:
                continue
            
            player_image = self.game_state.asset_manager.get_image(f'player{player.player_id+1}')
            if player_image:
                self.screen.blit(player_image, (player.pos.x * TILE_SIZE, player.pos.y * TILE_SIZE))
    
    def render_ui(self):
        """Render the user interface"""
        for i, player in enumerate(self.game_state.players):
            if not player.alive:
                continue
            
            # Display player info
            info_text = f"Player {i+1}: Bombs={player.bombs} Fire={player.fire} Speed={player.speed}"
            text_surface = self.font.render(info_text, True, WHITE)
            self.screen.blit(text_surface, (10, 10 + i * 30))
        
        # Display game over info
        if self.game_state.game_over:
            if self.game_state.winner:
                winner_text = f"Player {self.game_state.winner.player_id+1} Wins!"
                text_surface = self.font.render(winner_text, True, WHITE)
                self.screen.blit(text_surface, (WINDOW_WIDTH//2 - text_surface.get_width()//2, WINDOW_HEIGHT//2 - 20))
            
            restart_text = "Press R to Restart"
            text_surface = self.font.render(restart_text, True, WHITE)
            self.screen.blit(text_surface, (WINDOW_WIDTH//2 - text_surface.get_width()//2, WINDOW_HEIGHT//2 + 20))
    
    def render_game_over(self):
        """Render the game over screen"""
        game_over_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        game_over_surface.set_alpha(200)  # Semi-transparent black background
        
        # Draw game over background
        pygame.draw.rect(game_over_surface, (0, 0, 0), (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Draw text
        font = pygame.font.Font(None, 74)
        text = "Game Over"
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 50))
        game_over_surface.blit(text_surface, text_rect)
        
        if self.game_state.winner:
            winner_text = f"Player {self.game_state.winner.player_id+1} Wins!"
            winner_surface = font.render(winner_text, True, WHITE)
            winner_rect = winner_surface.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 10))
            game_over_surface.blit(winner_surface, winner_rect)
        
        # Draw restart prompt
        font_small = pygame.font.Font(None, 36)
        restart_text = "Press R to Restart"
        restart_surface = font_small.render(restart_text, True, WHITE)
        restart_rect = restart_surface.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 70))
        game_over_surface.blit(restart_surface, restart_rect)
        
        # Render to the screen
        self.screen.blit(game_over_surface, (0, 0))

def main():
    """Main function"""
    # Allow difficulty selection
    difficulty = DifficultyLevel.NORMAL  # Default difficulty
    
    # Simple console selection interface
    print("Please select the game difficulty:")
    print("1. Easy (Small map, few obstacles)")
    print("2. Normal (Standard map)")
    print("3. Hard (Large map, more obstacles)")
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        difficulty = DifficultyLevel.EASY
    elif choice == '3':
        difficulty = DifficultyLevel.HARD
    
    # Initialize Pygame
    pygame.init()
    
    # Initialize game state and renderer (using the selected difficulty)
    game_state = GameState(difficulty=difficulty)
    
    # Adjust window size according to the selected difficulty
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Classic Bomberman - {game_state.difficulty_config.name} Mode")
    
    clock = pygame.time.Clock()
    renderer = Renderer(screen, game_state)
    
    # Main game loop
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_state.game_over:
                    # Restart game
                    game_state.reset()
                    renderer.game_state = game_state
        
        # Input handling
        keys = pygame.key.get_pressed()
        game_state.handle_input(keys)
        
        # Update game state
        game_state.update()
        
        # Render
        renderer.render()
        
        # Control frame rate
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == '__main__':
    main()