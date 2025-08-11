"""
Wrap sound_alchemist_game.py into a Gymnasium Env.

Core Features
---------
1. reset(difficulty) directly jumps to the specified difficulty and starts the game;
2. step(action):
   • Injects a click action (discrete: color id)
   • Advances the internal game logic by 1 second (60 frames) and synchronizes logic
   • Captures the screen to get the final RGB frame (224×224)
   • Records 1 second of system audio / game audio (16 kHz mono)
   • Returns obs = {"image", "audio", "state"}
"""
from __future__ import annotations
import os
import time
import json
import traceback
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("Warning: sounddevice not available. Audio capture disabled.")

try:
    import soundfile as sf
except ImportError:
    sf = None
    print("Warning: soundfile not available. Audio saving disabled.")

try:
    import librosa
except ImportError:
    librosa = None
    print("Warning: librosa not available. Audio resampling disabled.")

from typing import List, Dict, Any

# === Simplified Game Wrapper =====
class Game:
    """Simplified game wrapper to adapt the original game logic"""
    def __init__(self, screen):
        self.screen = screen
        self.score = 0
        self.lives = 3
        self.solved_blocks = 0
        self.game_over = False
        
        # Audio capture related
        self.last_played_sound = None
        self.sound_played_this_step = False
        
        # Add verbose attribute
        self._verbose = False
        
        # Import and initialize the original game's global variables
        try:
            from . import sound_alchemist_game as game_module
            self.game_module = game_module
            # Set the game's screen
            game_module.screen = screen
            # Enable auto-start mode
            game_module.set_auto_start_mode(True)
        except ImportError:
            print("Warning: Could not import sound_alchemist_game module")
            self.game_module = None

    @property
    def verbose(self):
        """Get the verbose attribute"""
        return getattr(self, '_verbose', False)

    @verbose.setter
    def verbose(self, value):
        """Set the verbose attribute"""
        self._verbose = value

    def reset_audio_state(self):
        """Reset the audio state"""
        self.last_played_sound = None
        self.sound_played_this_step = False

    def get_state_info(self):
        """Get detailed game state information"""
        if self.game_module:
            try:
                return self.game_module.get_game_state()
            except Exception as e:
                print(f"Error getting game state from module: {e}")
                # Return a default state
                return {
                    "state": "unknown",
                    "difficulty": "normal",
                    "score": self.score,
                    "attempts": 0,
                    "sequence_length": 0,
                    "input_length": 0,
                    "game_over": self.game_over
                }
        return {
            "state": "unknown",
            "difficulty": "normal",
            "score": self.score,
            "attempts": 0,
            "sequence_length": 0,
            "input_length": 0,
            "game_over": self.game_over
        }

    def reset(self, difficulty="normal"):
        """Reset the game state and start the game automatically"""
        self.score = 0
        self.lives = 3
        self.solved_blocks = 0
        self.game_over = False
        
        if self.game_module:
            # Reset the original game's global state
            self.game_module.current_state = self.game_module.MENU
            self.game_module.player_score = 0
            self.game_module.melody_puzzle_attempts = 0
            self.game_module.player_melody_input = []
            
            # Set up the game according to difficulty
            difficulty_map = {
                "easy": self.game_module.DIFFICULTY_EASY,
                "normal": self.game_module.DIFFICULTY_MEDIUM,
                "hard": self.game_module.DIFFICULTY_HARD
            }
            target_difficulty = difficulty_map.get(difficulty, self.game_module.DIFFICULTY_MEDIUM)
            
            # Start the melody puzzle directly
            self.game_module.start_melody_puzzle_directly(target_difficulty)
            
            print(f"Game reset with difficulty: {difficulty}, state: {self.game_module.current_state}")
    
    def update(self):
        """Update the game logic for one frame"""
        if self.game_module:
            try:
                # Handle pygame events to prevent the window from becoming unresponsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                
                # Update game state
                self.score = self.game_module.player_score
                
                # Manually call the game rendering logic
                self._render_game_frame()
                
                # Get game state information
                game_state = self.game_module.get_game_state()
                
                # Update game over condition
                self.game_over = game_state["game_over"]
                
                # If the game is completed, automatically reset to a new round
                if self.game_over and self.game_module.current_state == self.game_module.PUZZLE_COMPLETE:
                    print("Puzzle completed! Auto-resetting for next round...")
                    # Delay a short time for the player to see the completion state
                    time.sleep(0.1)
                    # Reset to a new round, maintaining the current difficulty
                    current_diff_name = {
                        self.game_module.DIFFICULTY_EASY: "easy",
                        self.game_module.DIFFICULTY_MEDIUM: "normal", 
                        self.game_module.DIFFICULTY_HARD: "hard"
                    }.get(self.game_module.current_difficulty, "normal")
                    self.reset(current_diff_name)
            except Exception as e:
                print(f"Error in game update: {e}")
                # Reset the game state to prevent crashing
                self.game_over = False

    def _render_game_frame(self):
        """Manually render a game frame"""
        if not self.game_module:
            return
            
        try:
            # Get the current game state
            current_state = self.game_module.current_state
            
            # Clear the screen
            self.screen.fill((0, 0, 0))
            
            if current_state == self.game_module.PUZZLE_MELODY:
                # Render the melody puzzle interface
                self._render_melody_puzzle()
            elif current_state == self.game_module.MENU:
                # Render the menu interface
                self._render_menu()
            elif current_state == self.game_module.PUZZLE_COMPLETE:
                # Render the completion interface
                self._render_puzzle_complete()
            else:
                # Default rendering
                self._render_default()
            
            # Update the display
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error in render game frame: {e}")
            # At least fill with a non-black background to show rendering is working
            self.screen.fill((50, 50, 100))
            pygame.display.flip()

    def _render_melody_puzzle(self):
        """Render the melody puzzle interface"""
        # Background
        self.screen.fill((30, 30, 70))
        
        # Get fonts
        try:
            font_medium = pygame.font.Font(None, 50)
            font_small = pygame.font.Font(None, 30)
            font_tiny = pygame.font.Font(None, 24)
        except:
            font_medium = pygame.font.Font(None, 50)
            font_small = pygame.font.Font(None, 30)
            font_tiny = pygame.font.Font(None, 24)
        
        # Title
        title_text = font_medium.render("The Alchemist's Melody", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.screen.get_width() // 2, 50))
        self.screen.blit(title_text, title_rect)
        
        # Difficulty display
        difficulty_name = self.game_module.DIFFICULTY_SETTINGS[self.game_module.current_difficulty]['name']
        difficulty_text = font_small.render(f"Difficulty: {difficulty_name}", True, (173, 216, 230))
        difficulty_rect = difficulty_text.get_rect(center=(self.screen.get_width() // 2, 90))
        self.screen.blit(difficulty_text, difficulty_rect)
        
        # Instruction text
        instruction_text = font_small.render("Click the colored blocks in the correct musical order", True, (255, 255, 0))
        instruction_rect = instruction_text.get_rect(center=(self.screen.get_width() // 2, 130))
        self.screen.blit(instruction_text, instruction_rect)
        
        # Draw note elements
        if hasattr(self.game_module, 'note_elements'):
            self.game_module.note_elements.draw(self.screen)
            # Update note element animations
            self.game_module.note_elements.update()
        
        # Display player input
        input_display = []
        for note_id in self.game_module.player_melody_input:
            note_name = self.game_module.NOTE_DISPLAY_NAMES.get(note_id, "?")
            input_display.append(note_name)
        
        input_text = font_small.render(f"Your input: {' - '.join(input_display)}", True, (255, 255, 255))
        input_rect = input_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() - 80))
        self.screen.blit(input_text, input_rect)
        
        # Display number of mistakes
        attempts_text = font_tiny.render(f"Mistakes: {self.game_module.melody_puzzle_attempts}", True, (255, 170, 170))
        self.screen.blit(attempts_text, (20, self.screen.get_height() - 30))
        
        # Draw particle effects
        if hasattr(self.game_module, 'particles_group'):
            self.game_module.particles_group.update()
            self.game_module.particles_group.draw(self.screen)

    def _render_menu(self):
        """Render the menu interface"""
        # Use a gradient background
        for i in range(self.screen.get_height()):
            color = (20, 20, max(40, min(40 + i // 3, 90)))
            pygame.draw.line(self.screen, color, (0, i), (self.screen.get_width(), i))
        
        # Get fonts
        try:
            font_large = pygame.font.Font(None, 74)
            font_medium = pygame.font.Font(None, 50)
        except:
            font_large = pygame.font.Font(None, 74)
            font_medium = pygame.font.Font(None, 50)
        
        # Title
        title_text = font_large.render("Sound Alchemist's Chamber", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.screen.get_width() // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # Auto mode prompt
        auto_text = font_medium.render("Auto Mode - Game Starting...", True, (255, 255, 0))
        auto_rect = auto_text.get_rect(center=(self.screen.get_width() // 2, 300))
        self.screen.blit(auto_text, auto_rect)

    def _render_puzzle_complete(self):
        """Render the puzzle completion interface"""
        self.screen.fill((20, 80, 20))
        
        try:
            font_large = pygame.font.Font(None, 74)
            font_medium = pygame.font.Font(None, 50)
        except:
            font_large = pygame.font.Font(None, 74)
            font_medium = pygame.font.Font(None, 50)
        
        # Completion message
        complete_text = font_large.render("Melody Puzzle Solved!", True, (255, 255, 255))
        complete_rect = complete_text.get_rect(center=(self.screen.get_width() // 2, 200))
        self.screen.blit(complete_text, complete_rect)
        
        # Score display
        score_text = font_medium.render(f"Score: {self.game_module.player_score}", True, (255, 255, 0))
        score_rect = score_text.get_rect(center=(self.screen.get_width() // 2, 300))
        self.screen.blit(score_text, score_rect)
        
        # Congratulations text
        congrats_text = font_medium.render("Well Done, Alchemist!", True, (255, 255, 0))
        congrats_rect = congrats_text.get_rect(center=(self.screen.get_width() // 2, 400))
        self.screen.blit(congrats_text, congrats_rect)

    def _render_default(self):
        """Default rendering"""
        self.screen.fill((50, 50, 100))
        try:
            font = pygame.font.Font(None, 36)
            text = font.render("Game Loading...", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
            self.screen.blit(text, text_rect)
        except:
            pass

    def get_last_played_audio_data(self):
        """Get the most recently played audio data"""
        if self.game_module and hasattr(self.game_module, 'get_last_played_audio_data'):
            try:
                return self.game_module.get_last_played_audio_data()
            except Exception as e:
                print(f"Error getting audio data from game module: {e}")
                return None
        return None

    def click(self, color_name):
        """Handle a color click and record the played sound"""
        if not self.game_module:
            return
            
        try:
            # Dynamically get the current color-to-note mapping
            color_to_note = self._get_dynamic_color_to_note_mapping()
            
            note_id = color_to_note.get(color_name)
            if note_id:
                print(f"Color {color_name} mapped to note {note_id} ({self.game_module.NOTE_DISPLAY_NAMES.get(note_id, note_id)})")
                
                # Record the sound that is about to be played
                self.last_played_sound = note_id
                self.sound_played_this_step = True
                
                # Find the corresponding note element and trigger interaction
                found_element = False
                for note_sprite in self.game_module.note_elements:
                    if note_sprite.element_id == note_id:
                        note_sprite.interact()
                        found_element = True
                        break
                
                if not found_element:
                    print(f"Warning: No visual element found for note {note_id}")
                    # Create a temporary element object to simulate interaction
                    class TempElement:
                        def __init__(self, element_id):
                            self.element_id = element_id
                            self.sound = None
                            # Add a simulated rect attribute
                            self.rect = type('Rect', (), {'center': (400, 300)})()
                        
                        def highlight(self, color_tuple, duration=20):
                            """Simulate the highlight method, does nothing in practice"""
                            print(f"TempElement highlight called with color {color_tuple}, duration {duration}")
                            pass
                    
                    # Get the corresponding sound effect and play it
                    if hasattr(self.game_module, 'melody_note_sounds') and note_id in self.game_module.melody_note_sounds:
                        # Directly call the play_sound function to ensure the file path is recorded
                        sound_obj = self.game_module.melody_note_sounds[note_id]
                        self.game_module.play_sound(sound_obj)
                    
                    # Create a temporary element and call the game logic
                    temp_element = TempElement(note_id)
                    temp_element.sound = self.game_module.melody_note_sounds.get(note_id) if hasattr(self.game_module, 'melody_note_sounds') else None
                    
                    # Directly call the game logic to handle the input
                    self.game_module.handle_melody_input(note_id, temp_element)
            else:
                print(f"Warning: Color {color_name} not mapped to any note in current game configuration")
                
        except Exception as e:
            print(f"Error in click handling: {e}")
            traceback.print_exc()

    def _get_dynamic_color_to_note_mapping(self):
        """Dynamically get the current color-to-note mapping from the game"""
        color_to_note = {}
        
        try:
            if not self.game_module:
                return {}
            
            # Get the current note-to-color mapping
            current_note_color_mapping = getattr(self.game_module, 'current_note_color_mapping', {})
            all_colors = getattr(self.game_module, 'ALL_COLORS', {})
            
            if current_note_color_mapping and all_colors:
                # Create an RGB color to color name mapping
                color_rgb_to_name = {v: k for k, v in all_colors.items()}
                
                # Convert from note->color mapping to color_name->note mapping
                for note_id, rgb_color in current_note_color_mapping.items():
                    color_name = color_rgb_to_name.get(rgb_color)
                    if color_name:
                        color_to_note[color_name.upper()] = note_id
                
                if self.verbose and color_to_note:
                    print(f"Dynamic color-to-note mapping: {color_to_note}")
                
                return color_to_note
            
            # If the dynamic mapping cannot be obtained, log a warning
            if self.verbose:
                print("Warning: Could not get dynamic color-to-note mapping from game module")
                
        except Exception as e:
            if self.verbose:
                print(f"Error getting dynamic color-to-note mapping: {e}")
        
        # Return an empty mapping for the caller to handle
        return {}

    def get_block_pos(self, color_name):
        """Get the position of the specified color block"""
        if not self.game_module:
            return (400, 300)  # Default center position
            
        try:
            # Use dynamic mapping to get the note ID
            color_to_note = self._get_dynamic_color_to_note_mapping()
            note_id = color_to_note.get(color_name)
            
            if note_id:
                for note_sprite in self.game_module.note_elements:
                    if note_sprite.element_id == note_id:
                        return note_sprite.rect.center
        except Exception as e:
            print(f"Error getting block position: {e}")
        
        return (400, 300)  # Default position

    def get_available_colors(self):
        """Get information about the available color blocks for the current round"""
        if not self.game_module:
            # Force wait for the game module to initialize
            max_retries = 10
            retry_count = 0
            while not self.game_module and retry_count < max_retries:
                print(f"Waiting for game module initialization... (retry {retry_count + 1}/{max_retries})")
                time.sleep(0.1)
                retry_count += 1
                # Re-try getting the game module
                if hasattr(self.game, 'game_module'):
                    self.game_module = self.game.game_module
            
            if not self.game_module:
                raise RuntimeError("Game module not available after maximum retries")
        
        return self._get_available_colors_from_game_module(self.game_module)

    def _get_available_colors_from_game_module(self, game_module) -> List[Dict[str, Any]]:
        """Get the list of available colors from the game module"""
        retry_count = 0
        max_retries = 50  # Increase retry count
        
        while retry_count < max_retries:
            try:
                # Get from the current note sequence and color mapping
                correct_sequence = getattr(game_module, 'correct_melody_sequence', [])
                current_mapping = getattr(game_module, 'current_note_color_mapping', {})
                
                if not correct_sequence:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"correct_melody_sequence is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                    
                if not current_mapping:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"current_note_color_mapping is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                # Get color name mapping
                all_colors = getattr(game_module, 'ALL_COLORS', {})
                note_display_names = getattr(game_module, 'NOTE_DISPLAY_NAMES', {})
                
                if not all_colors:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"ALL_COLORS is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                    
                if not note_display_names:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"NOTE_DISPLAY_NAMES is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                    
                color_name_mapping = {v: k for k, v in all_colors.items()}
                available_colors = []
                
                # Validate the integrity of all necessary data
                incomplete_data = False
                for note_id in correct_sequence:
                    if note_id not in current_mapping:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Note {note_id} not in current_mapping, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                    
                    rgb_color = current_mapping[note_id]
                    color_name = color_name_mapping.get(rgb_color)
                    if not color_name:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Color name not found for RGB {rgb_color}, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                    
                    note_name = note_display_names.get(note_id)
                    if not note_name:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Note display name not found for {note_id}, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                    
                    available_colors.append({
                        "note_id": note_id,
                        "color_name": color_name,
                        "note_name": note_name,
                        "color_rgb": rgb_color
                    })
                
                if incomplete_data:
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                if len(available_colors) == len(correct_sequence):
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"Successfully got {len(available_colors)} colors from game module sequence")
                        for color_info in available_colors:
                            print(f"  - {color_info['color_name']}: {color_info['note_name']}")
                    return available_colors
                else:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"Incomplete color list ({len(available_colors)}/{len(correct_sequence)}), retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
            except Exception as e:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"Error getting available colors from game module (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(0.1)
        
        raise RuntimeError(f"Failed to get real available colors from game module after {max_retries} retries")

    def _get_color_note_mapping_from_game_module(self, game_module) -> Dict[str, str]:
        """Get the color-to-note mapping from the game module"""
        retry_count = 0
        max_retries = 50  # Increase retry count
        
        while retry_count < max_retries:
            try:
                # Get the current mapping and display names
                current_mapping = getattr(game_module, 'current_note_color_mapping', {})
                note_display_names = getattr(game_module, 'NOTE_DISPLAY_NAMES', {})
                
                if not current_mapping:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"current_note_color_mapping is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                if not note_display_names:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"NOTE_DISPLAY_NAMES is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                all_colors = getattr(game_module, 'ALL_COLORS', {})
                if not all_colors:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"ALL_COLORS is empty, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                    
                color_name_mapping = {v: k for k, v in all_colors.items()}
                color_note_mapping = {}
                
                # Validate the integrity of all data
                incomplete_data = False
                for note_id, rgb_color in current_mapping.items():
                    color_name = color_name_mapping.get(rgb_color)
                    if not color_name:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Color name not found for RGB {rgb_color}, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                        
                    note_name = note_display_names.get(note_id)
                    if not note_name:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Note display name not found for {note_id}, retrying... ({retry_count + 1}/{max_retries})")
                        incomplete_data = True
                        break
                        
                    color_note_mapping[color_name.lower()] = note_name
                
                if incomplete_data:
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                if len(color_note_mapping) == len(current_mapping):
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"Built color-note mapping from game module: {color_note_mapping}")
                    return color_note_mapping
                else:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"Incomplete color-note mapping ({len(color_note_mapping)}/{len(current_mapping)}), retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
            except Exception as e:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"Error getting color-note mapping from game module (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(0.1)
        
        raise RuntimeError(f"Failed to get real color-note mapping from game module after {max_retries} retries")

    def _convert_note_id_to_color_name(self, note_id, game_module) -> str:
        """Convert a note ID to a color name"""
        retry_count = 0
        max_retries = 30  # Increase retry count
        
        while retry_count < max_retries:
            try:
                # Get the current note-to-color mapping
                if hasattr(game_module, 'current_note_color_mapping'):
                    note_color_mapping = game_module.current_note_color_mapping
                    if not note_color_mapping:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"note_color_mapping is empty, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.1)
                        continue
                        
                    if note_id in note_color_mapping:
                        rgb_color = note_color_mapping[note_id]
                        
                        # Must find the corresponding color name from ALL_COLORS
                        if not hasattr(game_module, 'ALL_COLORS') or not game_module.ALL_COLORS:
                            if hasattr(self, '_verbose') and self._verbose:
                                print(f"ALL_COLORS not available for note {note_id}, retrying... ({retry_count + 1}/{max_retries})")
                            retry_count += 1
                            time.sleep(0.1)
                            continue
                            
                        all_colors = game_module.ALL_COLORS
                        color_name_mapping = {v: k for k, v in all_colors.items()}
                        color_name = color_name_mapping.get(rgb_color)
                        if color_name:
                            if hasattr(self, '_verbose') and self._verbose:
                                print(f"Successfully converted note {note_id} to color {color_name}")
                            return color_name.lower()
                        else:
                            if hasattr(self, '_verbose') and self._verbose:
                                print(f"Color name not found for RGB {rgb_color}, retrying... ({retry_count + 1}/{max_retries})")
                            retry_count += 1
                            time.sleep(0.1)
                            continue
                    else:
                        if hasattr(self, '_verbose') and self._verbose:
                            print(f"Note {note_id} not in mapping, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.1)
                        continue
                else:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"current_note_color_mapping attribute not found, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
            except Exception as e:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"Error converting note ID {note_id} to color name (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(0.1)
        
        raise RuntimeError(f"Failed to convert note ID {note_id} to color name after {max_retries} retries")

# ---- Constants ----
FPS = 60                 # Game logic frame rate
SEC_PER_STEP = 1         # Env advances 1 second per step
AUDIO_SR = 16_000        # Sampling rate 16 kHz
AUDIO_CHANNELS = 1       # Mono
IMG_SIZE = (224, 224)    # Resolution for output observation
# Extend color map to support all notes
COLOR_ID_MAP = {
    "BLUE": 0,     # Sol
    "RED": 1,      # Do
    "GREEN": 2,    # Fa  
    "YELLOW": 3,   # Mi
    "ORANGE": 4,   # Re
    "PURPLE": 5,   # La
    "GREY": 6,     # Ti/Si
}

class SoundAlchemistEnv(gym.Env):
    """Gymnasium-style environment"""
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        difficulty: str = "normal",
        render_mode: str | None = None,
        capture_audio: bool = True,
        audio_device: str | int | None = None,
        save_data: bool = False,
        save_dir: str = "game_data/caclu",
        save_sequence: bool = True
    ):
        super().__init__()
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.capture_audio = capture_audio
        self.audio_device = audio_device
        self.save_data = save_data
        self.save_dir = save_dir
        self.save_sequence = save_sequence
        
        # Data saving related
        self.episode_count = 0
        self.step_count_total = 0
        
        # Sequence data saving related
        self.current_episode_sequence: List[Dict[str, Any]] = []
        self.sequence_save_dir = os.path.join(save_dir, "sequences")
        
        self.current_correct_sequence = []
        
        # Create save directories
        if self.save_data:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "audio"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "metadata"), exist_ok=True)
            print(f"Data will be saved to: {self.save_dir}")
        
        if self.save_sequence:
            os.makedirs(self.sequence_save_dir, exist_ok=True)
            print(f"Sequence data will be saved to: {self.sequence_save_dir}")

        # Action: which color to click
        self.action_space = spaces.Discrete(len(COLOR_ID_MAP))

        # Observation: Dict(image, audio, state)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(*IMG_SIZE, 3), dtype=np.uint8
                ),
                "audio": spaces.Box(
                    low=-1.0, high=1.0, shape=(AUDIO_SR * SEC_PER_STEP,), dtype=np.float32
                ),
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                ),
            }
        )

        # ---- Initialize PyGame and the original game ----
        pygame.init()
        pygame.mixer.init()
        # Solve window opening issue in headless mode: HIDDEN
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Sound-Alchemist Gym Env")
        self.clock = pygame.time.Clock()
        self.game = Game(screen=self.screen)
        
        # Add a direct reference to the game module
        self.game_module = self.game.game_module if hasattr(self.game, 'game_module') else None
        
        self.tick = 0
        self._last_score = 0

        # New: Multimodal scoring system
        self.episode_performance_log: List[Dict[str, Any]] = []
        self.current_episode_metrics = {
            "positive_rewards": 0,
            "negative_rewards": 0,
            "audio_events": 0,
            "visual_changes": 0,
            "sequence_progress": 0,
            "sequence_resets": 0,
            "decision_quality": [],
            "response_times": [],
            "correct_actions": 0,
            "total_actions": 0
        }
        
        # Scores save directory
        self.scores_save_dir = os.path.join(save_dir, "scores")
        if self.save_data:
            os.makedirs(self.scores_save_dir, exist_ok=True)
            print(f"Episode scores will be saved to: {self.scores_save_dir}")

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if options and "difficulty" in options:
            self.difficulty = options["difficulty"]

        # Save the previous episode's sequence data
        if self.save_sequence and self.current_episode_sequence:
            self._save_episode_sequence()

        # Reset sequence data
        self.current_episode_sequence = []

        # 1) Reset the game
        self.game.reset(self.difficulty)

        self.tick = 0
        self._last_score = 0
        self.episode_count += 1

        # Force multiple updates to ensure the game initializes and renders correctly
        for _ in range(10):
            self.game.update()
            self.clock.tick(FPS)

        # 2) Grab the first frame
        frame = self._grab_frame()
        audio = np.zeros(AUDIO_SR * SEC_PER_STEP, dtype=np.float32)  # No sound on the first frame
        state_vec = self._get_state_vec()

        # If the frame is still all black, try a manual render
        if np.all(frame == 0):
            print("Warning: Initial frame is black, attempting manual render...")
            self.game._render_game_frame()
            frame = self._grab_frame()

        # 3) Save data (if enabled)
        if self.save_data:
            self._save_step_data(frame, audio, state_vec, action=-1, reward=0, is_reset=True)


        observation = {"image": frame, "audio": audio, "state": state_vec}
        info = {"tick": self.tick, "episode": self.episode_count}
        return observation, info

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action id"

        # Reset audio state
        self.game.reset_audio_state()
        
        # Record state before the step
        prev_score = self.game.score
        prev_attempts = getattr(self.game.game_module, 'melody_puzzle_attempts', 0) if self.game.game_module else 0
        prev_input_length = len(getattr(self.game.game_module, 'player_melody_input', [])) if self.game.game_module else 0

        # 1) Convert discrete action to color string
        color_name = list(COLOR_ID_MAP.keys())[action]
        self._inject_click(color_name)

        # 3) Advance the game by SEC_PER_STEP seconds
        n_frames = int(FPS * SEC_PER_STEP)
        for _ in range(n_frames):
            self.game.update()
            self.clock.tick(FPS)

        self.tick += 1
        self.step_count_total += 1
        
        # 2) Get internal game audio data
        audio_block = self._get_game_audio(duration=SEC_PER_STEP)

        # 4) Construct the return
        frame = self._grab_frame()
        state_vec = self._get_state_vec()

        # Smarter reward calculation, considering learning process and UI feedback
        cur_score = self.game.score
        cur_attempts = getattr(self.game.game_module, 'melody_puzzle_attempts', 0) if self.game.game_module else 0
        cur_input_length = len(getattr(self.game.game_module, 'player_melody_input', [])) if self.game.game_module else 0
        
        # Base reward: score increment
        reward = cur_score - self._last_score
        
        # Progress reward: successfully added to sequence (input length increased)
        if cur_input_length > prev_input_length:
            reward += 0.5  # Positive progress reward
        
        # Reset penalty: sequence was reset (input length reduced to 0)
        elif prev_input_length > 0 and cur_input_length == 0 and cur_attempts > prev_attempts:
            reward -= 1.0  # Larger penalty for sequence reset
        
        # Exploration reward: successful click (sound played but no error)
        elif cur_attempts == prev_attempts and self.game.sound_played_this_step:
            reward += 0.1  # Small positive reward to encourage exploration
            
        self._last_score = cur_score

        terminated = self.game.game_over
        truncated = False
        
        # Enhanced info dictionary, including UI feedback information
        info = {
            "tick": self.tick, 
            "episode": self.episode_count,
            "color_clicked": color_name,
            "sound_played": self.game.sound_played_this_step,
            "note_played": self.game.last_played_sound,
            "attempts_increased": cur_attempts > prev_attempts,
            "score_increased": cur_score > prev_score,
            "sequence_progress": cur_input_length,
            "sequence_reset": prev_input_length > 0 and cur_input_length == 0,
            "sequence_advanced": cur_input_length > prev_input_length,
            "total_sequence_length": len(getattr(self.game.game_module, 'correct_melody_sequence', [])) if self.game.game_module else 0
        }

        # 5) Save data (if enabled)
        if self.save_data:
            self._save_step_data(frame, audio_block, state_vec, action, reward)


        observation = {"image": frame, "audio": audio_block, "state": state_vec}
        
        # 6) Record step to sequence
        if self.save_sequence:
            self._add_to_sequence(
                step_type="step",
                observation=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info
            )

        if self.render_mode == "rgb_array":
            self.render()
        return observation, reward, terminated, truncated, info

    def close(self):
        # Save the last episode's sequence data
        if self.save_sequence and self.current_episode_sequence:
            self._save_episode_sequence()
        pygame.quit()

    # ---------- Internal Utilities ----------
    def _grab_frame(self) -> np.ndarray:
        """Grab screen -> 224x224 RGB"""
        raw = pygame.surfarray.array3d(self.screen)  # (W,H,3)
        raw = np.transpose(raw, (1, 0, 2))           # (H,W,3)
        # resize -> 224x224
        surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(raw), IMG_SIZE)
        arr = pygame.surfarray.array3d(surf)
        arr = np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        return arr

    def _get_game_audio(self, duration: float) -> np.ndarray:
        """Get internal game audio data"""
        # Try to get the internally played audio from the game
        if hasattr(self.game, 'get_last_played_audio_data'):
            game_audio = self.game.get_last_played_audio_data()
            if game_audio is not None:
                return game_audio
        
        # If no game audio, use the original recording logic or generate silence
        if self.capture_audio and sd:
            try:
                # Try to record real audio
                audio = sd.rec(int(duration * AUDIO_SR), samplerate=AUDIO_SR,
                              channels=AUDIO_CHANNELS, dtype="float32",
                              device=self.audio_device)
                sd.wait()
                return audio.flatten()
            except Exception as e:
                print(f"Audio recording failed: {e}, using silence")
        
        # Generate silent data
        print("Using silence for audio data")
        return np.zeros(int(duration * AUDIO_SR), dtype=np.float32)

    def _record_audio(self, duration: float) -> np.ndarray:
        """Backward compatible audio recording method"""
        return self._get_game_audio(duration)

    def _inject_click(self, color_name: str):
        """Inject a click event into PyGame; find coordinates based on color"""
        print(f"Injecting click for color: {color_name}")
        
        # If the game has an API for specific clicks, call it directly
        if hasattr(self.game, "click"):
            self.game.click(color_name)
            return

        # Fallback solution: find the center coordinates of the color block and inject a mouse event
        pos = self.game.get_block_pos(color_name)
        print(f"Click position for {color_name}: {pos}")
        
        ev_down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": 1})
        ev_up   = pygame.event.Event(pygame.MOUSEBUTTONUP,   {"pos": pos, "button": 1})
        pygame.event.post(ev_down)
        pygame.event.post(ev_up)

    def _get_state_vec(self) -> np.ndarray:
        """Concatenate key internal information into a fixed-length vector"""
        score = getattr(self.game, "score", 0)
        lives = getattr(self.game, "lives", 0)
        solved = getattr(self.game, "solved_blocks", 0)
        tick = self.tick
        return np.array([score, lives, solved, tick], dtype=np.float32)

    def _save_step_data(self, image: np.ndarray, audio: np.ndarray, state: np.ndarray, 
                       action: int, reward: float, is_reset: bool = False):
        """Save step data to files"""
        try:
            from PIL import Image
            
            # Generate filename base
            if is_reset:
                filename_base = f"ep{self.episode_count:04d}_reset"
            else:
                filename_base = f"ep{self.episode_count:04d}_step{self.tick:04d}"
            
            # Save image
            img_path = os.path.join(self.save_dir, "images", f"{filename_base}.png")
            Image.fromarray(image).save(img_path)
            
            # Save audio
            if sf:
                audio_path = os.path.join(self.save_dir, "audio", f"{filename_base}.wav")
                sf.write(audio_path, audio, AUDIO_SR)
            
            # Get detailed game state information
            game_state_info = self._build_detailed_game_state(action, is_reset)
            
            # Save reconstructed metadata (strictly following the required format)
            metadata = {
                "episode": self.episode_count,
                "step": self.tick,
                "step_total": self.step_count_total,
                "action": action if action is not None else 0,
                "is_reset": is_reset,
                "difficulty": self.difficulty,
                "timestamp": time.time(),
                "audio_info": {
                    "sample_rate": AUDIO_SR,
                    "duration": len(audio) / AUDIO_SR,
                    "has_sound": not np.allclose(audio, 0),
                    "rms_level": float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
                },
                "game_state": game_state_info
            };
            
            metadata_path = os.path.join(self.save_dir, "metadata", f"{filename_base}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            if is_reset:
                print(f"Saved reset data: {filename_base}")
            elif self.tick % 5 == 0:  # Print every 5 steps
                sound_info = "with sound" if not np.allclose(audio, 0) else "silent"
                print(f"Saved step data: {filename_base} ({sound_info})")
                
        except Exception as e:
            print(f"Error saving step data: {e}")

    def _build_detailed_game_state(self, action: int = None, is_reset: bool = False) -> Dict[str, Any]:
        """Build detailed game state information (enhanced version for the agent)"""
        try:
            # Get base game state
            base_state_info = self.game.get_state_info() if self.game else {}
            
            end_game = False
            
            # Get game module reference
            game_module = self.game.game_module if self.game and hasattr(self.game, 'game_module') else None
            
            if not game_module:
                print("Could not get game module, returning base state")
                return {
                    "current_score": base_state_info.get("score", 0),
                    "attempts": base_state_info.get("attempts", 0),
                    "sequence_length": base_state_info.get("sequence_length", 0),
                    "input_length": base_state_info.get("input_length", 0),
                    "game_over": base_state_info.get("game_over", False),
                    "currently_in_correct_sequence": False,
                    "needs_restart_from_beginning": False,
                    "current_correct_sequence": [],
                    "previous_clicks": [],
                    "last_clicked_block_color": "",
                    "last_last_clicked_block_color": "",
                    "last_clicked_action": 0,
                    "available_colors": [],
                    "color_note_mapping": {}
                }
            
            # Get detailed information from the game module
            player_input = getattr(game_module, 'player_melody_input', [])
            correct_sequence = getattr(game_module, 'correct_melody_sequence', [])
            attempts = getattr(game_module, 'melody_puzzle_attempts', 0)
            current_state = getattr(game_module, 'current_state', 'unknown')
            player_score = getattr(game_module, 'player_score', 0)
            current_note_color_mapping = getattr(game_module, 'current_note_color_mapping', {})

            if len(player_input) > 0:  # First check if player_input has elements
                if len(self.current_correct_sequence) != 0:
                    if len(self.current_correct_sequence) < len(correct_sequence) and player_input[-1] == correct_sequence[len(self.current_correct_sequence)]:
                        self.current_correct_sequence.append(self.game._convert_note_id_to_color_name(player_input[-1], game_module))
                    else:
                        self.current_correct_sequence = []
                else:
                    if player_input[-1] == correct_sequence[0]:
                        self.current_correct_sequence.append(self.game._convert_note_id_to_color_name(player_input[-1], game_module))

            # Analyze sequence correctness
            if len(self.current_correct_sequence) != 0:
                currently_in_correct_sequence = True       
            else:
                currently_in_correct_sequence = False
            
            needs_restart_from_beginning = not currently_in_correct_sequence 
            
            # Build history of recent clicks
            current_difficulty = getattr(game_module, 'current_difficulty', 'normal')
            
            if current_difficulty == "Hard":
                previous_clicks = []
                if len(self.current_correct_sequence)== 7:
                    end_game = True
                    
                if len(player_input) > 0:
                    recent_inputs = player_input[-7:] if len(player_input) > 7 else player_input
                    for note_id in reversed(recent_inputs):
                        color_name = self.game._convert_note_id_to_color_name(note_id, game_module)
                        if color_name:
                            previous_clicks.append(color_name)
            elif current_difficulty == "Medium":
                previous_clicks = []
                if len(player_input) > 0:
                    if len(self.current_correct_sequence)== 5:
                        end_game = True                  
                    recent_inputs = player_input[-5:] if len(player_input) > 5 else player_input
                    for note_id in reversed(recent_inputs):
                        color_name = self.game._convert_note_id_to_color_name(note_id, game_module)
                        if color_name:
                            previous_clicks.append(color_name)
            else: 
                previous_clicks = []
                if len(player_input) > 0:
                    if len(self.current_correct_sequence)== 3:
                            end_game = True   
                    recent_inputs = player_input[-3:] if len(player_input) > 3 else player_input
                    for note_id in reversed(recent_inputs):
                        color_name = self.game._convert_note_id_to_color_name(note_id, game_module)
                        if color_name:
                            previous_clicks.append(color_name)
            
            # Information about the last click
            last_clicked_block_color = ""
            last_last_clicked_block_color = ""
            last_clicked_action = 0
            
            if len(player_input) > 0:
                last_note_id = player_input[-1]
                last_clicked_block_color = self.game._convert_note_id_to_color_name(last_note_id, game_module) or ""
                last_clicked_action = last_note_id
            
            if len(player_input) > 1:
                last_last_note_id = player_input[-2]
                last_last_clicked_block_color = self.game._convert_note_id_to_color_name(last_last_note_id, game_module) or ""
            
            # Get available colors information
            available_colors = self.game._get_available_colors_from_game_module(game_module)
            
            # Get color to note mapping
            color_note_mapping = self.game._get_color_note_mapping_from_game_module(game_module)
            
            # Build detailed state
            detailed_state = {
                "current_score": player_score,
                "attempts": attempts,
                "sequence_length": len(correct_sequence),
                "input_length": len(player_input),
                "current_state": current_state,
                "game_over": base_state_info.get("game_over", False),
                "currently_in_correct_sequence": currently_in_correct_sequence,
                "needs_restart_from_beginning": needs_restart_from_beginning,
                "current_correct_sequence": self.current_correct_sequence.copy(),
                "previous_clicks": previous_clicks,
                "last_clicked_block_color": last_clicked_block_color,
                "last_last_clicked_block_color": last_last_clicked_block_color,
                "last_clicked_action": last_clicked_action,
                "available_colors": available_colors,
                "color_note_mapping": color_note_mapping,
                # Extra information
                "player_input_sequence": player_input.copy(),
                "correct_sequence": correct_sequence.copy(),
                "current_note_color_mapping": current_note_color_mapping.copy(),
                "difficulty": getattr(game_module, 'current_difficulty', 'normal'),
            }
            
            if self.verbose and action is not None:
                print(f"Built detailed state for action {action}:")
                print(f"  Currently correct: {currently_in_correct_sequence}")
                print(f"  Needs restart: {needs_restart_from_beginning}")
                print(f"  Input/Sequence: {len(player_input)}/{len(correct_sequence)}")
                print(f"  Last color: {last_clicked_block_color}")
                print(f"  Available colors: {len(available_colors)}")
            
            return detailed_state, end_game
            
        except Exception as e:
            if self.verbose:
                print(f"Error building detailed game state: {e}")
                import traceback
                traceback.print_exc()
            return {}
        
    def get_available_colors(self):
            if hasattr(self.game_module, 'get_available_colors'):
                available_colors = self.game_module.get_available_colors()
                if available_colors:
                    return available_colors
            
            # Fallback solution: get from the current sequence and color mapping
            if (hasattr(self.game_module, 'correct_melody_sequence') and 
                hasattr(self.game_module, 'current_note_color_mapping')):
                
                current_sequence = self.game_module.correct_melody_sequence
                current_mapping = self.game_module.current_note_color_mapping
                
                if current_sequence and current_mapping:
                    available_colors = []
                    from game.sound_alchemist.sound_alchemist_game import ALL_COLORS
                    color_name_mapping = {v: k for k, v in ALL_COLORS.items()}
                    
                    for note_id in current_sequence:
                        if note_id in current_mapping:
                            color = current_mapping[note_id]
                            color_name = color_name_mapping.get(color, f"RGB{color}")
                            note_name = self.game_module.NOTE_DISPLAY_NAMES.get(note_id, note_id)
                            available_colors.append({
                                "color_name": color_name,
                                "note_name": note_name,
                                "note_id": note_id,
                                "color_rgb": color
                            })
                    
                    return available_colors
            

    def _analyze_sequence_correctness(self, player_input: List, correct_sequence: List) -> bool:
        """Analyze if the player's input sequence is correct"""
        if not player_input or not correct_sequence:
            return len(player_input) == 0  # Empty input is correct
        
        # Check if each position matches
        for i in range(len(player_input)):
            if i >= len(correct_sequence) or player_input[i] != correct_sequence[i]:
                return False
        
        return True
    
    def get_detailed_game_state_for_agent(self) -> Dict[str, Any]:
        """Specialized method to get detailed game state for the agent"""
        
        return self._build_detailed_game_state(action=None, is_reset=False)
    
    
    def _add_to_sequence(self, step_type: str, **kwargs):
        """Add a step to the current episode sequence"""
        if not self.save_sequence:
            return
            
        try:
            step_data = {
                "step_type": step_type,
                "timestamp": time.time(),
                **kwargs
            }
            
            # Clean non-serializable data
            step_data = self._clean_sequence_data(step_data)
            
            self.current_episode_sequence.append(step_data)
            
        except Exception as e:
            if self.verbose:
                print(f"Error adding to sequence: {e}")

    @property
    def verbose(self):
        """Get the verbose attribute"""
        return getattr(self, '_verbose', False)

    @verbose.setter
    def verbose(self, value):
        """Set the verbose attribute"""
        self._verbose = value

    def _save_episode_sequence(self):
        """Save the episode sequence data"""
        if not self.current_episode_sequence:
            return
            
        try:
            timestamp = time.time()
            filename = f"episode_{self.episode_count:04d}_{int(timestamp)}.json"
            filepath = os.path.join(self.sequence_save_dir, filename)
            
            # Clean sequence data
            cleaned_sequence = self._clean_sequence_data(self.current_episode_sequence)
            
            episode_data = {
                "episode": self.episode_count,
                "timestamp": timestamp,
                "difficulty": self.difficulty,
                "total_steps": len(cleaned_sequence),
                "sequence": cleaned_sequence
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            if hasattr(self, '_verbose') and self._verbose:
                print(f"Saved episode sequence to: {filepath}")
                
        except Exception as e:
            if hasattr(self, '_verbose') and self._verbose:
                print(f"Error saving episode sequence: {e}")

    def _clean_sequence_data(self, data: Any) -> Any:
        """Clean non-serializable content from sequence data"""
        if isinstance(data, dict):
            return {k: self._clean_sequence_data(v) for k, v in data.items() if k not in ['image', 'audio']}
        elif isinstance(data, list):
            return [self._clean_sequence_data(item) for item in data]
        elif isinstance(data, (np.ndarray, np.integer, np.floating)):
            return data.tolist() if hasattr(data, 'tolist') else str(data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

    def _json_serializer(self, obj):
        """JSON serializer helper function"""
        if isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)