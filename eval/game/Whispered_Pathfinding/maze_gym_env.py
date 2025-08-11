import gym
import numpy as np
import pygame
import math
import os
import random
import base64
from gym import spaces
from gtts import gTTS
import tempfile
from datetime import datetime
import threading
from maze_3d import MazeGame

class MazeGymEnv(gym.Env):
    """
    Gym environment based on the 3D maze game
    
    Action Space:
    - Forward distance: [-1.0, 3.0], negative value means backward, positive means forward
    - Rotation angle: [-180.0, 180.0] degrees, negative value means turn left, positive means turn right
    
    Observation Space:
    - Player position (x, y)
    - Player facing angle
    - Distance to goal
    - Angle to goal (relative to player's facing direction)
    - Wall distances in 8 directions
    - Screenshot (RGB image array)
    - Audio data (base64 encoded string, returned via info)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode='rgb_array', voice_guidance=True, difficulty='easy'):
        super(MazeGymEnv, self).__init__()
        
        # Save difficulty setting
        self.difficulty = difficulty
        
        # Define action space: [forward_distance, rotation_angle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -180.0]), 
            high=np.array([3.0, 180.0]), 
            dtype=np.float32
        )
        
        # Define observation space as a Dict, including state vector and screen image
        self.observation_space = spaces.Dict({
            # State vector: [x_pos, y_pos, angle, dist_to_goal, angle_to_goal, wall_dist_in_8_directions]
            'vector': spaces.Box(
                low=np.array([0, 0, -np.pi, 0, -np.pi] + [0] * 8),
                high=np.array([MazeGame.MAZE_SIZE, MazeGame.MAZE_SIZE, np.pi, 
                              np.sqrt(2) * MazeGame.MAZE_SIZE, np.pi] + [MazeGame.MAX_DEPTH] * 8),
                dtype=np.float32
            ),
            # Screen image: RGB format
            'screen': spaces.Box(
                low=0, 
                high=255, 
                shape=(MazeGame.SCREEN_HEIGHT, MazeGame.SCREEN_WIDTH, 3),
                dtype=np.uint8
            )
        })
        
        self.render_mode = render_mode
        self.game = None  # Will be initialized in reset
        self.window_surface = None
        
        # Voice guidance related settings
        self.voice_guidance = voice_guidance
        self.voice_temp_dir = tempfile.mkdtemp(prefix="maze_voice_")
        self.last_guidance_angle = None
        self.guidance_cooldown = 0  # Voice guidance cooldown
        self.last_audio_path = None  # Store the path of the last played audio file
        
        # Cache for frequently used voice guidance
        self.voice_cache = {}
        
    def encode_audio_vllm(self, audio_path):
        """
        Encode an audio file into a base64 string
        
        Parameters:
            audio_path: Path to the audio file
        
        Returns:
            A base64 encoded string
        """
        if not audio_path or not os.path.exists(audio_path):
            return None
            
        try:
            with open(audio_path, "rb") as audio_file:
                return base64.b64encode(audio_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding audio: {e}")
            return None
    
    def _pregenerate_voice_guidance(self):
        """Pregenerate and cache frequently used voice guidance"""
        # No longer pregenerating fixed phrases, as each voice prompt is now dynamically generated based on the current position
        pass
        
    def _get_direction_guidance(self, relative_angle_deg, distance):
        """
        Generate precise direction advice based on the relative angle and distance
        
        Parameters:
            relative_angle_deg: Angle of the target relative to the player's facing direction (in degrees)
            distance: Distance to the target
        """
        # Convert the angle to the range (-180, 180]
        while relative_angle_deg > 180:
            relative_angle_deg -= 360
        while relative_angle_deg <= -180:
            relative_angle_deg += 360
        
        # Round the angle and distance to make the voice more natural
        rounded_angle = int(round(abs(relative_angle_deg) / 5) * 5)  # Round to the nearest 5 degrees
        rounded_distance = round(distance * 10) / 10  # Round to 1 decimal place
            
        # Generate prompts based on angle range and distance
        if abs(relative_angle_deg) <= 15:
            # Almost straight ahead
            return f"The exit is {rounded_distance} meters straight ahead."
        elif abs(relative_angle_deg) <= 45:
            # Slightly to the left/right
            direction = "right" if relative_angle_deg > 0 else "left"
            return f"The exit is {rounded_distance} meters ahead, {rounded_angle} degrees to your {direction}."
        elif abs(relative_angle_deg) <= 135:
            # Turn sharply left/right
            direction = "right" if relative_angle_deg > 0 else "left"
            return f"Turn {direction} about {rounded_angle} degrees. The exit is {rounded_distance} meters away."
        else:
            # Almost behind
            return f"The exit is {rounded_distance} meters behind you. Turn around."
        
    def _generate_voice_guidance(self, text, cache_only=False):
        """Generate and play voice guidance"""
        if not self.voice_guidance:
            return
            
        # Check if it's cached
        if text in self.voice_cache:
            voice_file = self.voice_cache[text]
        else:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            voice_file = os.path.join(self.voice_temp_dir, f"guidance_{timestamp}.wav")
            
            # Use gTTS to generate speech
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(voice_file)
                self.voice_cache[text] = voice_file
            except Exception as e:
                print(f"Error generating voice guidance: {e}")
                return
        
        # Save the path of the last audio file
        self.last_audio_path = voice_file
        
        # If only caching, do not play
        if cache_only:
            return
            
        # Play the voice in a background thread to avoid blocking the main thread
        def play_sound():
            try:
                sound = pygame.mixer.Sound(voice_file)
                sound.play()
            except Exception as e:
                print(f"Error playing voice guidance: {e}")
                
        threading.Thread(target=play_sound).start()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state and return the initial observation"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize or reset the game, using the specified difficulty
        if self.game is None:
            self.game = MazeGame(difficulty=self.difficulty)
            # Disable original game sound effects to avoid interference
            self.game.sound_go = None
        else:
            self.game.reset_game(difficulty=self.difficulty)
            
        # Reset voice guidance state
        self.last_guidance_angle = None
        self.guidance_cooldown = 0
        self.last_audio_path = None
        
        # Update the upper bounds of the vector in the environment's observation space based on the current difficulty
        current_maze_size = self.game.current_maze_size
        self.observation_space['vector'] = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, -np.pi] + [0] * 8),
            high=np.array([current_maze_size, current_maze_size, np.pi, 
                          np.sqrt(2) * current_maze_size, np.pi] + [MazeGame.MAX_DEPTH] * 8),
            dtype=np.float32
        )
        
        # Initial voice guidance
        if self.voice_guidance:
            self._generate_voice_guidance(f"Welcome to the {self.difficulty} maze. Find the exit!")
        
        # Get the initial observation
        observation = self._get_observation()
        
        # Generate the info dictionary, including audio data
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute an action and return the next state, reward, whether terminated, and extra info
        
        Parameters:
            action: [forward_distance, rotation_angle]
        """
        # Cache the current position to calculate the reward
        prev_pos = self.game.player_pos.copy()
        
        # Execute the action
        self._execute_action(action)
        
        # Get the new observation
        observation = self._get_observation()
        
        # Check if the goal is reached
        dist_to_goal = np.linalg.norm(self.game.player_pos - self.game.goal_pos)
        # Set different goal achievement distances based on difficulty
        threshold = 1.0 if self.difficulty == 'hard' else 0.7
        terminated = dist_to_goal < threshold
        
        # Calculate the reward
        reward = self._calculate_reward(dist_to_goal, prev_pos, terminated)
        
        # Check if a wall was hit
        truncated = False  # Currently not considering early termination
        
        # Provide voice direction guidance - now a new prompt is generated at each step
        self._provide_voice_guidance(observation['vector'])
        
        # Get the info dictionary, including audio data
        info = self._get_info()
        info['dist_to_goal'] = dist_to_goal
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self):
        """Get the info dictionary containing audio data"""
        info = {}
        
        # Add audio data (if any)
        audio_data = self.encode_audio_vllm(self.last_audio_path)
        if audio_data:
            info['audio'] = audio_data
            
        return info
    
    def _provide_voice_guidance(self, observation):
        """Provide voice navigation prompts based on the current state"""
        if not self.voice_guidance:
            return
            
        # Get the distance and angle to the goal (relative to the player's facing direction)
        dist_to_goal = observation[3]
        relative_angle_rad = observation[4]
        relative_angle_deg = math.degrees(relative_angle_rad)
        
        # Remove the cooldown check to ensure a prompt is generated every time
        
        # Generate a voice prompt based on position and direction
        if dist_to_goal < 2.0:
            # Special prompt when very close to the goal
            guidance_text = f"The exit is only {round(dist_to_goal, 1)} meters away."
        else:
            # Normal direction guidance
            guidance_text = self._get_direction_guidance(relative_angle_deg, dist_to_goal)
        
        # Generate a new voice prompt
        self._generate_voice_guidance(guidance_text)
        
        # Update the last guided angle (still recorded but not used for conditional checks)
        self.last_guidance_angle = relative_angle_deg
        
        # Set cooldown to 0 to disable the cooldown mechanism
        self.guidance_cooldown = 0

    def _execute_action(self, action):
        """
        Execute the given continuous action
        
        Parameters:
            action: [forward_distance, rotation_angle]
        """
        move_distance, rotation_angle = action
        
        # Convert angle from degrees to radians
        rotation_radians = math.radians(rotation_angle)
        
        # Apply rotation
        self.game.player_angle += rotation_radians
        
        # Normalize the angle to the range [-pi, pi]
        while self.game.player_angle > math.pi:
            self.game.player_angle -= 2 * math.pi
        while self.game.player_angle <= -math.pi:
            self.game.player_angle += 2 * math.pi
        
        # Calculate movement direction
        dir_x = math.cos(self.game.player_angle)
        dir_y = math.sin(self.game.player_angle)
        
        # Apply movement (forward/backward)
        new_x = self.game.player_pos[0] + dir_x * move_distance * self.game.MOVE_SPEED * 10.0
        new_y = self.game.player_pos[1] + dir_y * move_distance * self.game.MOVE_SPEED * 10.0
        
        # Check for collision and update position
        if not self.game.is_wall(new_x, self.game.player_pos[1]):
            self.game.player_pos[0] = new_x
        if not self.game.is_wall(self.game.player_pos[0], new_y):
            self.game.player_pos[1] = new_y
    
    def _get_observation(self):
        """Get the current environment's observation vector and screen image"""
        # Basic observation: player position and angle
        x, y = self.game.player_pos
        angle = self.game.player_angle
        
        # Calculate distance and angle to the goal
        goal_dir = self.game.goal_pos - self.game.player_pos
        distance_to_goal = np.linalg.norm(goal_dir)
        goal_angle = math.atan2(goal_dir[1], goal_dir[0])
        
        # Calculate the angle difference of the goal relative to the player's facing direction
        angle_diff = goal_angle - angle
        # Normalize the angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Get wall distances in 8 directions
        wall_distances = []
        for i in range(8):
            ray_angle = angle + i * math.pi / 4  # 45-degree intervals, covering 360 degrees
            wall_dist, _, _ = self.game.cast_ray(ray_angle)
            wall_distances.append(wall_dist)
        
        # Combine the observation vector
        vector_obs = np.array([x, y, angle, distance_to_goal, angle_diff] + wall_distances, 
                              dtype=np.float32)
        
        # Get the screen image
        # First, render the current screen
        self.game.render_3d_view()
        self.game.render_mini_map()
        self.game.render_ui()
        
        # Get a screenshot as an RGB array
        screen_image = pygame.surfarray.array3d(self.game.screen)
        # Adjust the array shape to (height, width, 3) and convert to uint8 type
        screen_image = np.transpose(screen_image, (1, 0, 2)).astype(np.uint8)
        
        # Return the observation in Dict format
        return {
            'vector': vector_obs,
            'screen': screen_image
        }
    
    def _calculate_reward(self, dist_to_goal, prev_pos, reached_goal):
        """
        Calculate the reward function
        
        Here we use a simple reward function:
        - Reaching the goal: large reward +10
        - Getting closer to the goal: small reward, inversely proportional to the distance to the goal
        - Moving away or hitting a wall: small penalty
        """
        if reached_goal:
            return 10.0  # Reward for reaching the goal
        
        # Calculate the reward for getting closer to the goal
        prev_dist = np.linalg.norm(self.game.goal_pos - prev_pos)
        reward = prev_dist - dist_to_goal  # Positive value means getting closer, negative means moving away
        
        # Add a small reward based on distance to encourage getting closer to the goal
        distance_reward = 0.01 / (0.1 + dist_to_goal)
        
        return reward + distance_reward
    
    def render(self):
        """Render the current game screen"""
        if self.render_mode == "human" and self.window_surface is None:
            pygame.init()
            pygame.display.init()
            self.window_surface = pygame.display.set_mode(
                (MazeGame.SCREEN_WIDTH, MazeGame.SCREEN_HEIGHT)
            )
            
        # Use the game's rendering functions
        self.game.render_3d_view()
        self.game.render_mini_map()
        self.game.render_ui()
        
        if self.render_mode == "human":
            pygame.display.flip()
            return None
        elif self.render_mode == "rgb_array":
            # Convert the rendered result to an RGB array
            return pygame.surfarray.array3d(
                self.game.screen
            ).swapaxes(0, 1)
    
    def close(self):
        """Close the environment and release resources"""
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.window_surface = None
        if self.game is not None:
            self.game.running = False
            
        # Clean up temporary voice files
        import shutil
        try:
            if os.path.exists(self.voice_temp_dir):
                shutil.rmtree(self.voice_temp_dir)
        except Exception as e:
            print(f"Error cleaning up voice files: {e}")
        # Clean up temporary voice files
        import shutil
        try:
            if os.path.exists(self.voice_temp_dir):
                shutil.rmtree(self.voice_temp_dir)
        except Exception as e:
            print(f"Error cleaning up voice files: {e}")