import pygame
import random
import math
import time
import json
import base64
import os
import numpy as np
import gym
from gym import spaces
import io
from PIL import Image
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import sys
import importlib.util

# Add game difficulty import
from game_difficulty import DifficultyLevel, get_difficulty_config

# Fix hyphen issue in filename
# Rename the file to a simpler name
classic_bomberman_file = os.path.join(os.path.dirname(__file__), 'classic_bomberman-daiceshi.py')
module_name = "classic_bomberman_module"

# Import the module directly
spec = importlib.util.spec_from_file_location(module_name, classic_bomberman_file)
game_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = game_module
spec.loader.exec_module(game_module)

# Get the required classes and constants from the module
Tile = game_module.Tile
Item = game_module.Item
Vector2 = game_module.Vector2
Player = game_module.Player
Bomb = game_module.Bomb
ItemDrop = game_module.ItemDrop
AssetManager = game_module.AssetManager
GameState = game_module.GameState
Renderer = game_module.Renderer
WINDOW_WIDTH = game_module.WINDOW_WIDTH
WINDOW_HEIGHT = game_module.WINDOW_HEIGHT
GRID_WIDTH = game_module.GRID_WIDTH
GRID_HEIGHT = game_module.GRID_HEIGHT
TILE_SIZE = game_module.TILE_SIZE
FPS = game_module.FPS
RED = game_module.RED
BLUE = game_module.BLUE
GREEN = game_module.GREEN
YELLOW = game_module.YELLOW
BLACK = game_module.BLACK
WHITE = game_module.WHITE

class BombermanAction(Enum):
    """Action enum type"""
    MOVE = 0      # Move action
    PLACE_BOMB = 1  # Place bomb action

class BombermanEnv(gym.Env):
    """Bomberman Gym Environment"""
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': FPS
    }
    
    def __init__(self, render_mode=None, num_players=4, max_steps=1000, difficulty=DifficultyLevel.NORMAL):
        super().__init__()
        
        # Initialize Pygame
        if not pygame.get_init():
            pygame.init()
        
        # Environment parameters
        self.render_mode = render_mode
        self.num_players = num_players
        self.max_steps = max_steps
        self.current_step = 0
        self.difficulty = difficulty
        self.difficulty_config = get_difficulty_config(difficulty)
        
        # Update parameters based on difficulty
        global GRID_WIDTH, GRID_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT
        GRID_WIDTH = self.difficulty_config.grid_width
        GRID_HEIGHT = self.difficulty_config.grid_height
        WINDOW_WIDTH = GRID_WIDTH * TILE_SIZE
        WINDOW_HEIGHT = GRID_HEIGHT * TILE_SIZE
        
        # Set max move distance and bomb countdown
        self.max_move_distance = self.difficulty_config.max_move_distance
        self.bomb_countdown_steps = 3  # bomb explosion countdown (steps)
        
        # Create game state and renderer
        self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.game_state = GameState(difficulty=difficulty)
        self.renderer = Renderer(self.screen, self.game_state)
        
        # If it needs to be displayed in a window
        if self.render_mode == 'human':
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption(f"Bomberman Gym - {self.difficulty_config.name} Mode")
        
        # Change bomb timer to step-based timing
        self._convert_bomb_timers_to_steps()
        
        # Define action space and observation space
        # Adjust observation space to accommodate different grid sizes
        self.action_space = spaces.Dict({
            player_id: spaces.Dict({
                'action_type': spaces.Discrete(2),  # 0: Move, 1: Place bomb
                'target_x': spaces.Discrete(GRID_WIDTH),  # Target x-coordinate
                'target_y': spaces.Discrete(GRID_HEIGHT),  # Target y-coordinate
            })
            for player_id in range(num_players)
        })
        
        # Observation space: Game state + Image + Audio
        self.observation_space = spaces.Dict({
            'state': spaces.Dict({
                'grid': spaces.Box(low=0, high=4, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.int8),
                'players': spaces.Dict({
                    player_id: spaces.Dict({
                        'position_x': spaces.Discrete(GRID_WIDTH),
                        'position_y': spaces.Discrete(GRID_HEIGHT),
                        'alive': spaces.Discrete(2),
                        'fire_power': spaces.Box(low=1, high=8, shape=(), dtype=np.int8),
                        'bomb_count': spaces.Box(low=1, high=8, shape=(), dtype=np.int8),
                        'speed': spaces.Box(low=1, high=8, shape=(), dtype=np.int8),
                        'active_bombs': spaces.Box(low=0, high=8, shape=(), dtype=np.int8),
                        'trapped': spaces.Discrete(2),
                    })
                    for player_id in range(num_players)
                }),
                'bombs': spaces.Dict({
                    'positions_x': spaces.Box(low=0, high=GRID_WIDTH-1, shape=(20,), dtype=np.int8),
                    'positions_y': spaces.Box(low=0, high=GRID_HEIGHT-1, shape=(20,), dtype=np.int8),
                    'countdown': spaces.Box(low=0, high=self.bomb_countdown_steps, shape=(20,), dtype=np.int8),
                    'owner': spaces.Box(low=0, high=num_players-1, shape=(20,), dtype=np.int8),
                    'fire_power': spaces.Box(low=1, high=8, shape=(20,), dtype=np.int8),
                    'count': spaces.Discrete(21),  # Max 20 bombs
                }),
                'items': spaces.Dict({
                    'positions_x': spaces.Box(low=0, high=GRID_WIDTH-1, shape=(50,), dtype=np.int8),
                    'positions_y': spaces.Box(low=0, high=GRID_HEIGHT-1, shape=(50,), dtype=np.int8),
                    'types': spaces.Box(low=0, high=2, shape=(50,), dtype=np.int8),
                    'count': spaces.Discrete(51),  # Max 50 items
                }),
                'flames': spaces.Dict({
                    'positions_x': spaces.Box(low=0, high=GRID_WIDTH-1, shape=(100,), dtype=np.int8),
                    'positions_y': spaces.Box(low=0, high=GRID_HEIGHT-1, shape=(100,), dtype=np.int8),
                    'count': spaces.Discrete(101),  # Max 100 flame tiles
                }),
                'game_over': spaces.Discrete(2),
            }),
            'image': spaces.Text(max_length=1000000),  # Base64 encoded image
            'audio': spaces.Text(max_length=1000000),  # Base64 encoded audio
            'step': spaces.Box(low=0, high=max_steps, shape=(), dtype=np.int32),
        })
        
        # Directory for storing temporary files
        self.temp_dir = "/tmp/bomberman_gym"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Audio event log - modified to a more detailed format, including player information
        self.audio_events = []  # Format: (event_type, player_id, other_params)
        
        # Add a counter for invalid actions
        self.invalid_actions_count = {i: 0 for i in range(num_players)}
    
    def _convert_bomb_timers_to_steps(self):
        """Converts bomb timers to step-based timing"""
        for bomb in self.game_state.bombs:
            # Convert frame-based timing to step-based timing
            bomb.timer = self.bomb_countdown_steps
    
    def reset(self, seed=None, options=None, difficulty=None):
        """Resets the environment, with an option to choose a new difficulty"""
        super().reset(seed=seed)
        
        # If a new difficulty is provided, update the difficulty settings
        if difficulty is not None:
            self.difficulty = difficulty
            self.difficulty_config = get_difficulty_config(difficulty)
            
            # Update global variables
            global GRID_WIDTH, GRID_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT
            GRID_WIDTH = self.difficulty_config.grid_width
            GRID_HEIGHT = self.difficulty_config.grid_height
            WINDOW_WIDTH = GRID_WIDTH * TILE_SIZE
            WINDOW_HEIGHT = GRID_HEIGHT * TILE_SIZE
            
            # Update move distance
            self.max_move_distance = self.difficulty_config.max_move_distance
            
            # Recreate the screen and window
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            if self.render_mode == 'human':
                self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption(f"Bomberman Gym - {self.difficulty_config.name} Mode")
        
        self.current_step = 0
        self.game_state.reset(difficulty=self.difficulty)
        self.renderer = Renderer(self.screen, self.game_state)
        self._convert_bomb_timers_to_steps()
        self.audio_events = []
        
        # Reset invalid action count
        self.invalid_actions_count = {i: 0 for i in range(self.num_players)}
        
        # Return initial observation
        obs = self._get_observation()
        info = {"difficulty": self.difficulty.value}
        
        return obs, info
    
    def step(self, actions):
        """Executes one step of action"""
        self.current_step += 1
        self.audio_events = []  # Clear audio events from the previous step
        
        # Process each player's action
        for player_id, action in actions.items():
            if player_id >= len(self.game_state.players):
                continue
                
            player = self.game_state.players[player_id]
            if not player.alive:
                continue
            
            try:
                # Try to get action parameters
                action_type = action.get('action_type')
                target_x = action.get('target_x')
                target_y = action.get('target_y')
                
                # Check for missing necessary parameters
                if action_type is None or (action_type == BombermanAction.MOVE.value and (target_x is None or target_y is None)):
                    # Record invalid action
                    self.invalid_actions_count[player_id] += 1
                    print(f"Warning: Player {player_id} provided an invalid action: {action}. Applying default action (stay still). Total invalid actions: {self.invalid_actions_count[player_id]}")
                    
                    # Provide default action - player stays still
                    action_type = BombermanAction.MOVE.value
                    target_x = player.pos.x
                    target_y = player.pos.y
                
                # Execute action
                if action_type == BombermanAction.MOVE.value:
                    self._handle_move_action(player, target_x, target_y)
                elif action_type == BombermanAction.PLACE_BOMB.value:
                    self._handle_place_bomb_action(player)
            except Exception as e:
                # Catch all exceptions to ensure the game continues even if action processing fails
                self.invalid_actions_count[player_id] += 1
                print(f"Error: An error occurred while processing action for player {player_id}: {e}. Applying default action (stay still). Total invalid actions: {self.invalid_actions_count[player_id]}")
                # Game continues, player stays still
        
        # When updating game state, pass the explosion event callback
        # Update bombs using step countdown
        self.game_state.update_bombs = self.game_state.update_bombs_steps
        self.game_state.register_explosion_callback = self.register_explosion_event
        self.game_state.update()
        
        # Get rewards, observation, done status, and info
        obs = self._get_observation()
        rewards = self._get_rewards()
        terminated = self.game_state.game_over
        truncated = self.current_step >= self.max_steps
        info = {
            "invalid_actions": self.invalid_actions_count.copy()
        }
        
        # If it needs to be displayed in a window
        if self.render_mode == 'human':
            self.render()
        
        return obs, rewards, terminated, truncated, info
    
    def register_explosion_event(self, bomb, affected_positions):
        """Registers an explosion sound event"""
        self.audio_events.append(('bomb_explode', bomb.owner_id, {
            'pos': (bomb.grid_pos.x, bomb.grid_pos.y),
            'fire': bomb.fire,
            'affected_positions': list(affected_positions)
        }))
        
        # Ensure the flame duration after explosion is correct (using steps instead of frames)
        for pos in affected_positions:
            self.game_state.flames[pos] = 2  # Flame lasts for 2 steps
    
    def _handle_move_action(self, player, target_x, target_y):
        """Handles a move action"""
        # Check if the move distance is within the allowed range
        current_x, current_y = player.pos.x, player.pos.y
        distance = abs(target_x - current_x) + abs(target_y - current_y)  # Manhattan distance
        max_distance = self.max_move_distance + (player.speed - 1)  # base move distance + speed bonus
        
        if distance > max_distance:
            # Out of range, truncate to the maximum allowed distance
            if target_x > current_x:
                dx = min(target_x - current_x, max_distance)
            else:
                dx = max(target_x - current_x, -max_distance)
                
            # Adjust the move distance in the y-direction
            remaining_distance = max_distance - abs(dx)
            if target_y > current_y:
                dy = min(target_y - current_y, remaining_distance)
            else:
                dy = max(target_y - current_y, -remaining_distance)
                
            target_x = current_x + dx
            target_y = current_y + dy
        
        # Check if the target position is valid
        path = self._find_path(current_x, current_y, target_x, target_y, max_distance)
        if path:
            # Move to the furthest valid point on the path
            final_pos = path[-1]
            player.pos = Vector2(final_pos[0], final_pos[1])
            
            # Check for item pickup
            self.game_state.check_item_pickup(player)
            
            # Record move sound event, including player ID
            self.audio_events.append(('player_walk', player.player_id, {
                'from_pos': (current_x, current_y),
                'to_pos': (final_pos[0], final_pos[1])
            }))
    
    def _find_path(self, start_x, start_y, target_x, target_y, max_steps):
        """Finds a path from start to target, returns the furthest reachable point if unreachable"""
        # Simplified A* pathfinding algorithm
        open_set = [(start_x, start_y)]
        closed_set = set()
        g_score = {(start_x, start_y): 0}
        came_from = {}
        
        while open_set:
            current = min(open_set, key=lambda pos: g_score[pos] + abs(pos[0] - target_x) + abs(pos[1] - target_y))
            
            if current == (target_x, target_y) or g_score[current] >= max_steps:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            # Check four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check boundaries
                if (neighbor[0] < 0 or neighbor[0] >= GRID_WIDTH or 
                    neighbor[1] < 0 or neighbor[1] >= GRID_HEIGHT):
                    continue
                
                # Check for obstacles
                if self.game_state.grid[neighbor[1]][neighbor[0]] in [Tile.HARD, Tile.SOFT, Tile.BOMB]:
                    continue
                
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in open_set:
                    open_set.append(neighbor)
                elif tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
        
        # If no path is found, return the point closest to the target
        if came_from:
            best_pos = max(closed_set, key=lambda pos: g_score[pos])
            path = [best_pos]
            while best_pos in came_from:
                best_pos = came_from[best_pos]
                path.append(best_pos)
            path.reverse()
            return path
        
        # Completely unable to move
        return [(start_x, start_y)]
    
    def _handle_place_bomb_action(self, player):
        """Handles a place bomb action"""
        # Check bomb count limit
        if player.active_bombs >= player.bombs:
            return
        
        # Check if a bomb is already at the current position
        bomb_pos = player.pos
        for bomb in self.game_state.bombs:
            if bomb.grid_pos.x == bomb_pos.x and bomb.grid_pos.y == bomb_pos.y:
                return
        
        # Place bomb
        bomb = Bomb(
            owner_id=player.player_id,
            grid_pos=Vector2(bomb_pos.x, bomb_pos.y),
            fire=player.fire,
            timer=self.bomb_countdown_steps  # Use step-based timing
        )
        self.game_state.bombs.append(bomb)
        self.game_state.grid[bomb_pos.y][bomb_pos.x] = Tile.BOMB
        player.active_bombs += 1
        
        # Record place bomb sound event, including player ID
        self.audio_events.append(('bomb_place', player.player_id, {
            'pos': (bomb_pos.x, bomb_pos.y),
            'fire': player.fire
        }))
    
    def render(self):
        """Renders the game screen"""
        self.renderer.render()
        
        if self.render_mode == 'human':
            # Copy to the window for display
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            pygame.event.pump()  # Process event queue to prevent the program from becoming unresponsive
        
        return self.screen
    
    def _get_observation(self):
        """Gets the observation state"""
        # Game state
        state = self._get_game_state()
        
        # Game image
        image_base64 = self._get_image_base64()
        
        # Game audio
        audio_base64 = self._get_audio_base64()
        
        return {
            'state': state,
            'image': image_base64,
            'audio': audio_base64,
            'step': self.current_step
        }
    
    def _get_game_state(self):
        """Gets a structured representation of the game state"""
        # Map grid
        grid = np.array([[tile.value for tile in row] for row in self.game_state.grid], dtype=np.int8)
        
        # Player information
        players_info = {}
        for i, player in enumerate(self.game_state.players):
            if i >= self.num_players:
                break
            players_info[i] = {
                'position_x': player.pos.x,
                'position_y': player.pos.y,
                'alive': int(player.alive),
                'fire_power': player.fire,
                'bomb_count': player.bombs,
                'speed': player.speed,
                'active_bombs': player.active_bombs,
                'trapped': int(player.trapped_ticks > 0),
            }
        
        # Bomb information
        bomb_positions_x = np.zeros(20, dtype=np.int8)
        bomb_positions_y = np.zeros(20, dtype=np.int8)
        bomb_countdown = np.zeros(20, dtype=np.int8)
        bomb_owner = np.zeros(20, dtype=np.int8)
        bomb_fire_power = np.zeros(20, dtype=np.int8)
        
        for i, bomb in enumerate(self.game_state.bombs[:20]):
            bomb_positions_x[i] = bomb.grid_pos.x
            bomb_positions_y[i] = bomb.grid_pos.y
            bomb_countdown[i] = bomb.timer
            bomb_owner[i] = bomb.owner_id
            bomb_fire_power[i] = bomb.fire
        
        # Item information
        item_positions_x = np.zeros(50, dtype=np.int8)
        item_positions_y = np.zeros(50, dtype=np.int8)
        item_types = np.zeros(50, dtype=np.int8)
        
        for i, item in enumerate(self.game_state.items[:50]):
            item_positions_x[i] = item.grid_pos.x
            item_positions_y[i] = item.grid_pos.y
            item_types[i] = item.item_type.value
        
        # Flame information
        flame_positions_x = np.zeros(100, dtype=np.int8)
        flame_positions_y = np.zeros(100, dtype=np.int8)
        
        # Adapt to the new flame data structure
        for i, (x, y) in enumerate(list(self.game_state.flames.keys())[:100]):
            flame_positions_x[i] = x
            flame_positions_y[i] = y
        
        return {
            'grid': grid,
            'players': players_info,
            'bombs': {
                'positions_x': bomb_positions_x,
                'positions_y': bomb_positions_y,
                'countdown': bomb_countdown,
                'owner': bomb_owner,
                'fire_power': bomb_fire_power,
                'count': len(self.game_state.bombs),
            },
            'items': {
                'positions_x': item_positions_x,
                'positions_y': item_positions_y,
                'types': item_types,
                'count': len(self.game_state.items),
            },
            'flames': {
                'positions_x': flame_positions_x,
                'positions_y': flame_positions_y,
                'count': len(self.game_state.flames),
            },
            'game_over': int(self.game_state.game_over),
        }
    
    def _get_image_base64(self):
        """Gets the base64 encoding of the game screen"""
        # Render to a surface
        self.renderer.render()
        
        # Convert pygame surface to PIL Image
        image_str = pygame.image.tostring(self.screen, 'RGB')
        image = Image.frombytes('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), image_str)
        
        # Save as a temporary file
        temp_image_path = os.path.join(self.temp_dir, f"frame_{self.current_step}.png")
        image.save(temp_image_path)
        
        # Encode to base64
        return self.encode_image(temp_image_path)
    
    def _get_audio_base64(self):
        """Gets the base64 encoding of the game audio"""
        # Return all audio events, not just the first one
        if not self.audio_events:
            return ""
        
        audio_data = []
        asset_manager = self.game_state.asset_manager
        
        for event_type, player_id, params in self.audio_events:
            sound = asset_manager.get_sound(event_type)
            
            # Map event types to specific file names - updated to WAV files
            event_file_mapping = {
                'player_walk': 'footstep_wood_001.wav',
                'bomb_place': 'click-b.wav',
                'bomb_explode': 'explosion1.wav'
            }
            
            # Use the mapping to find the specific audio file
            audio_path = None
            if event_type in event_file_mapping:
                target_file = event_file_mapping[event_type]
                for path in asset_manager.asset_paths:
                    if path.endswith(target_file):
                        if os.path.exists(path):
                            audio_path = path
                            break
                
                if audio_path:
                    # Create a data structure containing event information
                    player_name = f"Player {player_id + 1}"
                    event_description = f"{player_name} - {event_type}"
                    
                    # Add more description based on the event type
                    if event_type == 'player_walk':
                        from_pos = params['from_pos']
                        to_pos = params['to_pos']
                        event_description += f" from ({from_pos[0]},{from_pos[1]}) to ({to_pos[0]},{to_pos[1]})"
                    elif event_type == 'bomb_place':
                        pos = params['pos']
                        fire = params['fire']
                        event_description += f" at ({pos[0]},{pos[1]}) with fire power {fire}"
                    elif event_type == 'bomb_explode':
                        pos = params['pos']
                        fire = params['fire']
                        affected_positions = params['affected_positions']
                        affected_desc = ", ".join([f"({x},{y})" for x, y in affected_positions])
                        event_description += f" at ({pos[0]},{pos[1]}) with fire power {fire}, affected: {affected_desc}"
                    
                    # Encode audio
                    encoded_audio = self.encode_audio_vllm(audio_path)
                    
                    # Add to the result list
                    audio_data.append({
                        'event_type': event_type,
                        'player_id': player_id,
                        'description': event_description,
                        'audio_base64': encoded_audio,
                        'params': params
                    })
        
        # Convert the result to a JSON string and return it
        return json.dumps(audio_data)
    
    def encode_audio_vllm(self, audio_path):
        """Encodes audio to base64 format"""
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    def encode_image(self, image_path):
        """Encodes an image to base64 format"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _get_rewards(self):
        """Gets the reward for each player"""
        rewards = {i: 0.0 for i in range(self.num_players)}
        
        # Complex reward calculation logic can be implemented here
        # For example: destroying walls, collecting items, defeating enemies, etc.
        
        # Simple reward: survival reward
        for i, player in enumerate(self.game_state.players):
            if i >= self.num_players:
                break
                
            if player.alive:
                rewards[i] += 0.1  # Survival reward
        
        # Victory reward
        if self.game_state.game_over and self.game_state.winner:
            winner_id = self.game_state.winner.player_id
            if winner_id < self.num_players:
                rewards[winner_id] += 10.0  # Large victory reward
        
        return rewards
    
    def close(self):
        """Closes the environment"""
        if hasattr(self, 'window') and self.window is not None:
            pygame.display.quit()
            pygame.quit()

# Example usage
if __name__ == "__main__":
    # Add difficulty selection menu
    print("Please select the game difficulty:")
    print("1. Easy (Small map, few obstacles)")
    print("2. Normal (Standard map)")
    print("3. Hard (Large map, more obstacles)")
    
    choice = input("Enter your choice (1/2/3, default 2): ")
    difficulty = DifficultyLevel.NORMAL
    
    if choice == '1':
        difficulty = DifficultyLevel.EASY
        print("Selected Easy difficulty - Small map, few obstacles")
    elif choice == '3':
        difficulty = DifficultyLevel.HARD
        print("Selected Hard difficulty - Large map, more obstacles")
    else:
        print("Selected Normal difficulty - Standard map")
    
    env = BombermanEnv(render_mode='human', difficulty=difficulty)
    obs, info = env.reset()
    
    for _ in range(1000):
        # Random actions
        actions = {}
        for player_id in range(env.num_players):
            if random.random() < 0.2:  # 20% chance to place a bomb
                actions[player_id] = {
                    'action_type': BombermanAction.PLACE_BOMB.value,
                    'target_x': 0,  # Target coordinates not needed for placing a bomb
                    'target_y': 0
                }
            else:  # 80% chance to move
                actions[player_id] = {
                    'action_type': BombermanAction.MOVE.value,
                    'target_x': random.randint(0, GRID_WIDTH-1),
                    'target_y': random.randint(0, GRID_HEIGHT-1)
                }
        
        # Execute actions and get results
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # If the game is over, reset the environment
        if terminated or truncated:
            print("Game over, resetting environment")
            obs, info = env.reset()
            # Wait a short time for observation
            time.sleep(0.5)
        
        # Add a little delay to make the demonstration easier to observe
        time.sleep(0.1)
    
    # Close the environment
    env.close()