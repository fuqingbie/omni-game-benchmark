import gym
import numpy as np
import pygame
import io
import base64
import json
import os
import tempfile
import time
import requests
from PIL import Image
from maze_gym_env import MazeGymEnv
import datetime
import pathlib
import argparse

# Import configuration
from config import (
    API_BASE, API_KEY, MODEL_CHAT,
    DEFAULT_DIFFICULTY, DEFAULT_ROUNDS, DEFAULT_MAX_STEPS, DEFAULT_AUTO_SPEED,
    RESULTS_DIR, TEMP_DIR_PREFIX, TEXT_DISPLAY_SIZE, TEXT_DISPLAY_POS, FONT_SIZE,
    DIFFICULTY_MAP, DIFFICULTY_DESCRIPTIONS,
    DEFAULT_SEED, USE_SEQUENTIAL_SEEDS, RANDOM_SEED_RANGE
)

class ModelMazeRunner:
    """An automatic maze runner that uses a large model for analysis"""
    
    def __init__(self, difficulty=DEFAULT_DIFFICULTY, auto_speed=DEFAULT_AUTO_SPEED, 
                 max_steps=DEFAULT_MAX_STEPS, results_dir=RESULTS_DIR,
                 seed=None, use_sequential_seeds=USE_SEQUENTIAL_SEEDS):
        # Initialize Pygame (for audio playback and display)
        pygame.init()
        pygame.mixer.init()
        
        # Store game settings
        self.difficulty = difficulty
        self.auto_speed = auto_speed
        self.max_steps = max_steps
        self.results_dir = results_dir
        
        # Seed configuration
        self.seed = seed
        self.use_sequential_seeds = use_sequential_seeds
        self.current_seed = None
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {API_KEY}"})
        
        # Get the correct path to game assets
        current_dir = os.path.dirname(os.path.abspath(__file__))
        game_dir = os.path.dirname(current_dir)  # Go up one level to (game)
        assets_dir = os.path.join(game_dir, "assets-necessay")
        
        # Check if the assets directory exists, and try other possible paths if it doesn't
        if not os.path.exists(assets_dir):
            # Try a relative path
            assets_dir = os.path.join("..", "assets-necessay")
            if not os.path.exists(assets_dir):
                # Try the assets directory in the current directory
                assets_dir = "assets-necessay"
                if not os.path.exists(assets_dir):
                    print(f"Warning: assets directory not found, audio functions may not work correctly")
                    assets_dir = None
        
        print(f"Using assets directory: {assets_dir}")
        
        # Create the environment, enable voice navigation, set the difficulty, and pass the assets path
        if assets_dir:
            # Set an environment variable or pass parameters to MazeGymEnv
            os.environ['MAZE_ASSETS_PATH'] = assets_dir
        
        self.env = MazeGymEnv(render_mode="human", voice_guidance=True, difficulty=self.difficulty)
        
        # Create a temporary directory to store audio files
        self.temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
        
        # Initialize an additional display window (optional, for displaying model analysis results)
        self.text_display_size = TEXT_DISPLAY_SIZE
        self.text_display = pygame.Surface(self.text_display_size)
        self.text_display_pos = TEXT_DISPLAY_POS
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        
        # Track total reward and the current suggested action
        self.total_reward = 0
        self.current_suggested_action = None
        self.current_step = 0
        
        # Game statistics
        self.stats = {
            "steps": 0,
            "total_reward": 0, 
            "invalid_actions": 0  # New: record the number of invalid actions
        }
        
        # Set the system prompt
        self.system_prompt = """
        You are a professional maze navigation intelligent agent.

        Observation information:
        1. Image - Shows a 3D view of the maze and a mini-map
        2. Audio - Provides voice navigation guidance
        3. State vector - Contains position, orientation, and target information

        Your task is to provide optimal navigation suggestions.

        Executable actions:
        - Forward distance: [-1.0, 3.0], negative values mean moving backward, positive values mean moving forward
        - Rotation angle: [-180.0, 180.0] degrees, negative values mean rotating left, positive values mean rotating right, relative to the current orientation

        Analyze each observation and provide clear action recommendations, including:
        1. A brief description of the current position and surrounding environment
        2. Suggested action (forward/backward distance and rotation angle)
        3. Reasoning for this action (e.g., avoiding walls, facing the target, etc.)

        [IMPORTANT] Your response must end with the following exact format: "Suggested action: [number] [number]"
        For example: "Suggested action: 1.0 45" or "Suggested action: 0.5 -30"
        Do not use any other formats, such as "Suggested action: move forward 1.0, rotate 45", only use the number pair without units.
        """
    
    def play_audio(self, audio_base64):
        """Play audio prompts"""
        if not audio_base64:
            return
            
        try:
            # Decode base64 to an audio file
            audio_data = base64.b64decode(audio_base64)
            audio_file = os.path.join(self.temp_dir, "current_guidance.wav")
            
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            
            # Play the audio
            sound = pygame.mixer.Sound(audio_file)
            sound.play()
            
            # Wait for the audio to finish playing (optional)
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def get_model_suggestion(self, observation, audio=None):
        """Get an action suggestion from the model"""
        try:
            # Convert the image to base64 encoding
            image = observation['screen']
            img_pil = Image.fromarray(image)
            img_io = io.BytesIO()
            img_pil.save(img_io, format="JPEG")
            img_bytes = img_io.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # Build the vector state description
            vector_obs = observation['vector']
            state_description = (
                f"Current position: x={vector_obs[0]:.2f}, y={vector_obs[1]:.2f}\n"
                f"Current orientation: {np.degrees(vector_obs[2]):.1f}°\n"
                f"Distance to target: {vector_obs[3]:.2f}m\n"
                f"Direction to target: {np.degrees(vector_obs[4]):.2f}°\n"
                "Distance to walls: " + 
                ", ".join([f"{i*45}°: {d:.1f}m" for i, d in enumerate(vector_obs[5:])])
            )
            
            # Construct user prompt
            user_content = f"""
Please analyze the current maze environment and provide navigation suggestions.

Environment state information:
{state_description}

Please provide the following:
1. Environment analysis: Describe the current position, orientation, and relationship to the target position
2. Suggested action: Provide specific forward distance and rotation angle
3. Navigation rationale: Explain why you chose this action

Remember to end your response with the format "Suggested action: [forward distance] [rotation angle]".
For example: "Suggested action: 1.0 -45"
"""
            
            # Build the messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ]
                }
            ]
            
            # If there is audio data, add it to the messages
            if audio is not None:
                messages[1]["content"].append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio,
                        "format": "wav",
                    },
                })
            
            # Build the complete request payload
            payload = {
                "model": MODEL_CHAT,
                "messages": messages,
                "modalities": ["text", "audio"],
            }
            
            # API endpoint URL
            url = f"{API_BASE}/chat/completions"
            
            print("Requesting model analysis...")
            # Send the API request using the session.post() method
            r = self.session.post(url, json=payload, timeout=300)
            
            # Check the response status
            if r.status_code != 200:
                print(f"API request failed, status code: {r.status_code}")
                print(f"Response content: {r.text}")
                return f"API request failed: {r.status_code} - {r.text}", None
            
            # Parse the response JSON
            response_data = r.json()
            
            # Extract the text content
            try:
                model_response = response_data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                print(f"Could not extract text from the response: {response_data}")
                return "Could not extract text from the response", None
            
            print("\nModel analysis result:")
            print("-" * 60)
            print(model_response)
            print("-" * 60)
            
            # Extract the suggested action from the response
            action = self.extract_action_from_response(model_response)
            return model_response, action
            
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return f"API request error: {e}", None
        except Exception as e:
            print(f"Model request error: {e}")
            return f"Error while requesting the model: {e}", None
    
    def extract_action_from_response(self, response):
        """Extract the action from the model's response"""
        try:
            # Standard format matching: "Suggested action: 1.0 45"
            import re
            action_match = re.search(r"Suggested action:\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", response)
            if not action_match:
                action_match = re.search(r"Suggested action：\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", response)
            
            if action_match:
                forward = float(action_match.group(1))
                rotation = float(action_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # Match the format with "forward" and "rotate" keywords: "Suggested action: forward 0.7, rotate 45"
            front_turn_match = re.search(r"Suggested action:[\s]*forward\s*([-+]?\d*\.?\d+)[,，]?\s*rotate\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE) 
            if not front_turn_match:
                front_turn_match = re.search(r"Suggested action：[\s]*forward\s*([-+]?\d*\.?\d+)[,，]?\s*rotate\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE)
            
            if front_turn_match:
                forward = float(front_turn_match.group(1))
                rotation = float(front_turn_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # Match comma-separated format: "Suggested action: 0.7, 45"
            comma_match = re.search(r"Suggested action:[\s]*([-+]?\d*\.?\d+)[,，]\s*([-+]?\d*\.?\d+)", response)
            if not comma_match:
                comma_match = re.search(r"Suggested action：[\s]*([-+]?\d*\.?\d+)[,，]\s*([-+]?\d*\.?\d+)", response)
            
            if comma_match:
                forward = float(comma_match.group(1))
                rotation = float(comma_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
                
            # Match format with units: "Suggested action: 0.7m 45deg"
            unit_match = re.search(r"Suggested action:[\s]*([-+]?\d*\.?\d+)\s*m?\s*([-+]?\d*\.?\d+)\s*deg?", response)
            if not unit_match:
                unit_match = re.search(r"Suggested action：[\s]*([-+]?\d*\.?\d+)\s*m?\s*([-+]?\d*\.?\d+)\s*deg?", response)
            
            if unit_match:
                forward = float(unit_match.group(1))
                rotation = float(unit_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # If none of the above match, try to find a pair of numbers after the suggested action
            action_line_match = re.search(r"Suggested action:(.+)$|Suggested action：(.+)$", response, re.MULTILINE)
            if action_line_match:
                action_line = action_line_match.group(1) or action_line_match.group(2)
                # Extract numbers from this line
                nums = re.findall(r"[-+]?\d*\.?\d+", action_line)
                if len(nums) >= 2:
                    forward = float(nums[0])
                    rotation = float(nums[1])
                    # Check if the range is reasonable
                    if -1.0 <= forward <= 3.0 and -180.0 <= rotation <= 180.0:
                        return np.array([forward, rotation], dtype=np.float32)
            
            # Fallback matching, find the last two numbers in the text
            numbers = re.findall(r"[-+]?\d*\.?\d+", response)
            if len(numbers) >= 2:
                forward = float(numbers[-2])
                rotation = float(numbers[-1])
                # Check if the range is reasonable
                if -1.0 <= forward <= 3.0 and -180.0 <= rotation <= 180.0:
                    return np.array([forward, rotation], dtype=np.float32)
            
            print("Could not extract a valid action from the response, using the default action")
            # Record invalid action
            self.stats["invalid_actions"] += 1
            # Provide a safe default action
            return np.array([0.0, 0.0], dtype=np.float32)
            
        except Exception as e:
            print(f"Error while extracting the action: {e}")
            # Record invalid action
            self.stats["invalid_actions"] += 1
            # Provide a safe default action
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def get_user_input(self):
    
        print(f"Using the model's suggested action: {self.current_suggested_action[0]:.2f} {self.current_suggested_action[1]:.2f}")
        return self.current_suggested_action

    
    def print_observation_info(self, observation):
        """Print observation information"""
        vector_obs = observation['vector']
        
        print("\n===== Current State =====")
        print(f"Position: ({vector_obs[0]:.2f}, {vector_obs[1]:.2f})")
        print(f"Orientation: {np.degrees(vector_obs[2]):.1f}°")
        print(f"Distance to target: {vector_obs[3]:.2f}m")
        print(f"Angle to target: {np.degrees(vector_obs[4]):.1f}°")
        
        # Output main wall distance information
        directions = ["Front", "Front-Right", "Right", "Back-Right", "Back", "Back-Left", "Left", "Front-Left"]
        for i, direction in enumerate(directions):
            print(f"Wall distance {direction}: {vector_obs[5+i]:.2f}m")
    
    def save_stats(self, round_num, terminated, truncated):
        """Save statistics to a file"""
        # Create the save directory (if it doesn't exist)
        save_dir = pathlib.Path(self.results_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the current time as part of the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Determine the current code filename (without the .py extension)
        script_name = os.path.basename(__file__).replace('.py', '')
        
        # Create the filename
        filename = f"{script_name}_round{round_num}_{timestamp}.txt"
        filepath = save_dir / filename
        
        # Write statistics
        with open(filepath, 'w') as f:
            f.write("===== Navigation Statistics =====\n")
            f.write(f"Difficulty level: {self.difficulty}\n")
            f.write(f"Seed value: {self.current_seed}\n")
            f.write(f"Total steps: {self.stats['steps']}\n")
            f.write(f"Total reward: {self.stats['total_reward']:.2f}\n")
            f.write(f"Number of invalid actions: {self.stats['invalid_actions']}\n")
            f.write(f"Successfully reached the goal: {'Yes' if terminated else 'No'}\n")
            if truncated:
                f.write("Round was truncated: Yes\n")
            f.write(f"Save time: {timestamp}\n")
        
        print(f"Statistics saved to: {filepath}")
    
    def run_single_round(self, round_num):
        """Run a single round of maze navigation"""
        print(f"\n===== Starting round {round_num} =====")
        print(f"Difficulty level: {self.difficulty}")
        
        # Set seed for this round
        if self.seed is not None:
            # Use fixed seed
            self.current_seed = self.seed
        elif self.use_sequential_seeds:
            # Use sequential seed: 0, 1, 2, ...
            self.current_seed = round_num - 1
        else:
            # Use random seed
            import random
            self.current_seed = random.randint(*RANDOM_SEED_RANGE)
        
        print(f"Using seed: {self.current_seed}")
        
        # Reset the environment
        observation, info = self.env.reset(seed=self.current_seed)
        
        # Reset statistics
        self.total_reward = 0
        self.current_step = 0
        self.stats = {
            "steps": 0,
            "total_reward": 0,
            "invalid_actions": 0
        }
        
        # Display initial state information
        self.print_observation_info(observation)
        
        # Play initial audio prompt
        if 'audio' in info:
            print("Playing voice navigation...")
            self.play_audio(info['audio'])
        
        # Autorun loop
        done = False
        audio_data = None
        
        while not done and self.current_step < self.max_steps:
            # Get model suggestion
            model_response, suggested_action = self.get_model_suggestion(observation, audio_data)
            self.current_suggested_action = suggested_action
            
            if suggested_action is not None:
                print(f"\nModel suggested action: Forward={suggested_action[0]:.2f}, Rotate={suggested_action[1]:.2f}°")
            else:
                self.current_suggested_action = np.array([0.0, 0.0], dtype=np.float32)
                print("\nThe model could not provide a clear action suggestion, using the default action")
       
            # Execute action
            action = self.get_user_input()
            if action is None:  # User interrupted
                return False, False
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            self.current_step += 1
            
            # Update the environment display
            self.env.render()
            
            # Display status information
            print(f"\nStep {self.current_step}/{self.max_steps}, Total reward: {self.total_reward:.2f}")
            print(f"Distance to target: {observation['vector'][3]:.2f}m")
            
            # Play audio prompt (if any)
            if 'audio' in info:
                try:
                    # Decode audio data
                    audio_data = base64.b64decode(info['audio'])
                    audio_file = os.path.join(self.temp_dir, "last_guidance.wav")
                    
                    with open(audio_file, "wb") as f:
                        f.write(audio_data)
                    
                    # Play the audio
                    self.play_audio(info['audio'])
                    audio_data = info['audio']
                except Exception as e:
                    print(f"Error processing audio data: {e}")
                    audio_data = None
            
            # Check if finished
            done = terminated or truncated
            if terminated:
                print("\nCongratulations! You have successfully reached the goal!")
            elif truncated:
                print("\nRound ended!")
        
        # Update and display statistics
        self.stats["steps"] = self.current_step
        self.stats["total_reward"] = self.total_reward
        
        print("\n===== Navigation Statistics =====")
        print(f"Difficulty level: {self.difficulty}")
        print(f"Total steps: {self.stats['steps']}")
        print(f"Total reward: {self.stats['total_reward']:.2f}")
        print(f"Number of invalid actions: {self.stats['invalid_actions']}")
        print(f"Successfully reached the goal: {'Yes' if terminated else 'No'}")
        
        # Save statistics
        self.save_stats(round_num, terminated, truncated)
        
        return terminated, truncated
    
    def run(self, total_rounds=DEFAULT_ROUNDS):
        """Run multiple rounds of maze navigation"""
        print(f"Starting automatic maze navigation - Difficulty: {self.difficulty}")
        print(f"Auto speed: {self.auto_speed}s/step, Max steps: {self.max_steps}")
        print(f"Running {total_rounds} rounds, statistics will be saved after each round")
        print(f"Results save directory: {self.results_dir}")
        
        completed_rounds = 0
        success_rounds = 0
        
        try:
            for round_num in range(1, total_rounds + 1):
                terminated, truncated = self.run_single_round(round_num)
                completed_rounds += 1
                if terminated:
                    success_rounds += 1
                
                # Pause for a short time between rounds
                if round_num < total_rounds:
                    print(f"\nWaiting 2 seconds before starting the next round...")
                    time.sleep(2)
            
            # Print overall statistics
            print("\n===== Overall Statistics =====")
            print(f"Completed rounds: {completed_rounds}/{total_rounds}")
            print(f"Successfully reached goal rounds: {success_rounds}/{total_rounds}")
            print(f"Success rate: {(success_rounds/total_rounds)*100:.1f}%")
            
        except KeyboardInterrupt:
            print("\nUser interrupted the test")
            print(f"Completed rounds: {completed_rounds}/{total_rounds}")
        finally:
            # Clean up resources
            self.env.close()
            pygame.quit()
            
            # Clean up temporary files
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Error while cleaning up temporary files: {e}")
            
            print("\nAutomatic navigation test ended!")


def select_difficulty():
    """Let the user select the maze difficulty"""
    print("Please select the maze difficulty:")
    for key, desc in DIFFICULTY_DESCRIPTIONS.items():
        if key in ["1", "2", "3"]:
            continue
        number = {"easy": "1", "medium": "2", "hard": "3"}[key]
        print(f"{number}. {desc}")
    
    while True:
        choice = input("Please enter your choice (1-3): ")
        if choice in DIFFICULTY_MAP:
            return DIFFICULTY_MAP[choice]
        else:
            print("Invalid choice, please re-enter.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Maze Navigation Test - OpenAI Model")
    parser.add_argument("--difficulty", type=str, default=DEFAULT_DIFFICULTY, 
                       choices=["easy", "medium", "hard"], help="Maze difficulty")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Number of test rounds")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum steps per round")
    parser.add_argument("--speed", type=float, default=DEFAULT_AUTO_SPEED, help="Autorun speed")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, help="Results save directory")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed value")
    parser.add_argument("--no-sequential-seeds", action="store_true", help="Disable sequential seed mode")
    parser.add_argument("--interactive", action="store_true", help="Interactively select difficulty")
    
    args = parser.parse_args()
    
    # Interactively select difficulty
    if args.interactive:
        difficulty = select_difficulty()
    else:
        difficulty = args.difficulty
    
    # Create the autorunner and run the test
    runner = ModelMazeRunner(
        difficulty=difficulty, 
        auto_speed=args.speed, 
        max_steps=args.max_steps,
        results_dir=args.results_dir,
        seed=args.seed,
        use_sequential_seeds=not args.no_sequential_seeds
    )
    runner.run(total_rounds=args.rounds)

if __name__ == "__main__":
    main()