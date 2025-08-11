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
    BAICHUAN_FASTAPI_BASE_URL,
    DEFAULT_DIFFICULTY, DEFAULT_ROUNDS, DEFAULT_MAX_STEPS, DEFAULT_AUTO_SPEED,
    RESULTS_DIR, TEMP_DIR_PREFIX, TEXT_DISPLAY_SIZE, TEXT_DISPLAY_POS, FONT_SIZE,
    DIFFICULTY_MAP, DIFFICULTY_DESCRIPTIONS,
    DEFAULT_SEED, USE_SEQUENTIAL_SEEDS, RANDOM_SEED_RANGE
)

class ModelMazeRunner:
    """Automatic maze runner that uses the Baichuan model for analysis."""
    
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
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session_id = None  # Session ID for the Baichuan model
        
        # Create the environment, enable voice guidance, and set difficulty
        self.env = MazeGymEnv(render_mode="human", voice_guidance=True, difficulty=self.difficulty)
        
        # Create a temporary directory to store files
        self.temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
        
        # Initialize an additional display window (optional, for displaying model analysis results)
        self.text_display_size = TEXT_DISPLAY_SIZE
        self.text_display = pygame.Surface(self.text_display_size)
        self.text_display_pos = TEXT_DISPLAY_POS
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        
        # Track total reward and current suggested action
        self.total_reward = 0
        self.current_suggested_action = None
        self.current_step = 0
        
        # Game statistics
        self.stats = {
            "steps": 0,
            "total_reward": 0, 
            "invalid_actions": 0  # New: Record the number of invalid actions
        }
        
        # Set system prompt
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

        # Seed configuration
        self.seed = seed
        self.use_sequential_seeds = use_sequential_seeds
        self.current_seed = None
    
    def clear_session(self):
        """Clear the current session."""
        if self.session_id:
            try:
                url = f"{BAICHUAN_FASTAPI_BASE_URL}/clear_session"
                data = {"session_id": self.session_id}
                response = self.session.post(url, data=data, timeout=10)
                if response.status_code == 200:
                    print("✅ Session cleared")
                else:
                    print(f"⚠️ Failed to clear session: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Error clearing session: {e}")
        self.session_id = None

    def play_audio(self, audio_base64):
        """Play audio guidance."""
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
            
            # Wait for audio playback to complete (optional)
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def get_model_suggestion(self, observation, audio=None):
        """Get action suggestion from the Baichuan model."""
        try:
            # Save the image to a temporary file
            image = observation['screen']
            img_pil = Image.fromarray(image)
            image_path = os.path.join(self.temp_dir, "current_view.jpg")
            img_pil.save(image_path)
            
            # Build vector state description
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
            
            # Prepare Baichuan API request data
            data = {
                "query": user_content,
                "system_prompt": self.system_prompt,
                "audiogen_flag": False,
                "session_id": self.session_id
            }
            
            # Prepare file upload
            files = []
            if os.path.exists(image_path):
                files.append(('image_files', ('current_view.jpg', open(image_path, 'rb'), 'image/jpeg')))
            
            # If there is audio data, save and upload it
            if audio is not None:
                audio_path = os.path.join(self.temp_dir, "guidance_audio.wav")
                if isinstance(audio, str):
                    # If it's a base64 string, decode and save
                    audio_data = base64.b64decode(audio)
                    with open(audio_path, 'wb') as f:
                        f.write(audio_data)
                else:
                    # If it's binary data, save it directly
                    with open(audio_path, 'wb') as f:
                        f.write(audio)
                files.append(('audio_file', ('guidance_audio.wav', open(audio_path, 'rb'), 'audio/wav')))
            
            url = f"{BAICHUAN_FASTAPI_BASE_URL}/chat"
            
            print("Requesting analysis from the Baichuan model...")
            
            try:
                response = self.session.post(url, data=data, files=files, timeout=300)
            finally:
                # Close file handles
                for _, file_tuple in files:
                    file_tuple[1].close()
            
            # Check response status
            if response.status_code != 200:
                print(f"API request failed, status code: {response.status_code}")
                print(f"Response content: {response.text}")
                return f"API request failed: {response.status_code} - {response.text}", None
            
            # Parse response JSON
            response_data = response.json()
            
            # Extract text content and session ID
            try:
                model_response = response_data.get("text", "")
                self.session_id = response_data.get("session_id")
            except (KeyError, IndexError):
                print(f"Could not extract text from response: {response_data}")
                return "Could not extract text from response", None
            
            print("\nBaichuan Model Analysis Result:")
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
            return f"Error requesting model: {e}", None
    
    def extract_action_from_response(self, response):
        """Extract action from the model's response."""
        try:
            # Standard format match: "Suggested action: 1.0 45"
            import re
            action_match = re.search(r"Suggested action:\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", response)
            if not action_match:
                action_match = re.search(r"Suggested action:\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", response)
            
            if action_match:
                forward = float(action_match.group(1))
                rotation = float(action_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # Match format with "forward" and "rotation" keywords: "Suggested action: forward 0.7, rotate 45"
            front_turn_match = re.search(r"Suggested action:[\s]*forward\s*([-+]?\d*\.?\d+)[,，]?\s*rotate\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE) 
            if not front_turn_match:
                front_turn_match = re.search(r"Suggested action:[\s]*forward\s*([-+]?\d*\.?\d+)[,，]?\s*rotate\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE)
            
            if front_turn_match:
                forward = float(front_turn_match.group(1))
                rotation = float(front_turn_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # Match comma-separated format: "Suggested action: 0.7, 45"
            comma_match = re.search(r"Suggested action:[\s]*([-+]?\d*\.?\d+)[,，]\s*([-+]?\d*\.?\d+)", response)
            if not comma_match:
                comma_match = re.search(r"Suggested action:[\s]*([-+]?\d*\.?\d+)[,，]\s*([-+]?\d*\.?\d+)", response)
            
            if comma_match:
                forward = float(comma_match.group(1))
                rotation = float(comma_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
                
            # Match format with units: "Suggested action: 0.7m 45deg"
            unit_match = re.search(r"Suggested action:[\s]*([-+]?\d*\.?\d+)\s*m?\s*([-+]?\d*\.?\d+)\s*deg?", response)
            if not unit_match:
                unit_match = re.search(r"Suggested action:[\s]*([-+]?\d*\.?\d+)\s*m?\s*([-+]?\d*\.?\d+)\s*deg?", response)
            
            if unit_match:
                forward = float(unit_match.group(1))
                rotation = float(unit_match.group(2))
                return np.array([forward, rotation], dtype=np.float32)
            
            # If none of the above match, try to find the number pair after the suggested action text
            action_line_match = re.search(r"Suggested action:(.+)$|Suggested action:(.+)$", response, re.MULTILINE)
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
            
            # Fallback match, find the last two numbers in the text
            numbers = re.findall(r"[-+]?\d*\.?\d+", response)
            if len(numbers) >= 2:
                forward = float(numbers[-2])
                rotation = float(numbers[-1])
                # Check if the range is reasonable
                if -1.0 <= forward <= 3.0 and -180.0 <= rotation <= 180.0:
                    return np.array([forward, rotation], dtype=np.float32)
            
            print("Could not extract a valid action from the response, using default action")
            # Record invalid action
            self.stats["invalid_actions"] += 1
            # Provide a safe default action
            return np.array([0.0, 0.0], dtype=np.float32)
            
        except Exception as e:
            print(f"Error while extracting action: {e}")
            # Record invalid action
            self.stats["invalid_actions"] += 1
            # Provide a safe default action
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def get_user_input(self):
    
        print(f"Using model's suggested action: {self.current_suggested_action[0]:.2f} {self.current_suggested_action[1]:.2f}")
        return self.current_suggested_action

    
    def print_observation_info(self, observation):
        """Print observation information."""
        vector_obs = observation['vector']
        
        print("\n===== Current State =====")
        print(f"Position: ({vector_obs[0]:.2f}, {vector_obs[1]:.2f})")
        print(f"Orientation: {np.degrees(vector_obs[2]):.1f}°")
        print(f"Distance to Target: {vector_obs[3]:.2f}m")
        print(f"Angle to Target: {np.degrees(vector_obs[4]):.1f}°")
        
        # Output main wall distance information
        directions = ["Front", "Front-Right", "Right", "Back-Right", "Back", "Back-Left", "Left", "Front-Left"]
        for i, direction in enumerate(directions):
            print(f"{direction} wall distance: {vector_obs[5+i]:.2f}m")
    
    def save_stats(self, round_num, terminated, truncated):
        """Save statistics to a file."""
        # Create the save directory (if it doesn't exist)
        save_dir = pathlib.Path(self.results_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the current time as part of the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Determine the current script filename (without the .py extension)
        script_name = os.path.basename(__file__).replace('.py', '')
        
        # Create the filename
        filename = f"{script_name}_round{round_num}_{timestamp}.txt"
        filepath = save_dir / filename
        
        # Write statistics
        with open(filepath, 'w') as f:
            f.write("===== Navigation Statistics =====\n")
            f.write(f"Difficulty Level: {self.difficulty}\n")
            f.write(f"Seed Value: {self.current_seed}\n")
            f.write(f"Total Steps: {self.stats['steps']}\n")
            f.write(f"Total Reward: {self.stats['total_reward']:.2f}\n")
            f.write(f"Number of Invalid Actions: {self.stats['invalid_actions']}\n")
            f.write(f"Successfully Reached Target: {'Yes' if terminated else 'No'}\n")
            if truncated:
                f.write("Round was truncated: Yes\n")
            f.write(f"Save Time: {timestamp}\n")
        
        print(f"Statistics saved to: {filepath}")
    
    def run_single_round(self, round_num):
        """Run a single round of maze navigation."""
        print(f"\n===== Starting Round {round_num} =====")
        print(f"Difficulty Level: {self.difficulty}")
        
        # Clear the previous session
        self.clear_session()
        
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
        
        # Reset environment with seed
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
        
        # Play initial audio guidance
        if 'audio' in info:
            print("Playing voice guidance...")
            self.play_audio(info['audio'])
        
        # Automatic run loop
        done = False
        audio_data = None
        
        while not done and self.current_step < self.max_steps:
            # Get model suggestion
            model_response, suggested_action = self.get_model_suggestion(observation, audio_data)
            self.current_suggested_action = suggested_action
            
            if suggested_action is not None:
                print(f"\nModel suggested action: Forward={suggested_action[0]:.2f}, Rotation={suggested_action[1]:.2f}°")
            else:
                self.current_suggested_action = np.array([0.0, 0.0], dtype=np.float32)
                print("\nModel could not provide a clear action suggestion, using default action")
            
            # Execute the action
            action = self.get_user_input()
            if action is None:  # User interrupted
                return False, False
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            self.current_step += 1
            
            # Update environment display
            self.env.render()
            
            # Display status information
            print(f"\nStep {self.current_step}/{self.max_steps}, Total Reward: {self.total_reward:.2f}")
            print(f"Distance to Target: {observation['vector'][3]:.2f}m")
            
            # Play audio guidance (if any)
            if 'audio' in info:
                try:
                    # Play the audio
                    self.play_audio(info['audio'])
                    audio_data = info['audio']
                except Exception as e:
                    print(f"Error processing audio data: {e}")
                    audio_data = None
            
            # Check if finished
            done = terminated or truncated
            if terminated:
                print("\nCongratulations! Successfully reached the target!")
            elif truncated:
                print("\nRound ended!")
        
        # Update and display statistics
        self.stats["steps"] = self.current_step
        self.stats["total_reward"] = self.total_reward
        
        print("\n===== Navigation Statistics =====")
        print(f"Difficulty Level: {self.difficulty}")
        print(f"Total Steps: {self.stats['steps']}")
        print(f"Total Reward: {self.stats['total_reward']:.2f}")
        print(f"Number of Invalid Actions: {self.stats['invalid_actions']}")
        print(f"Successfully Reached Target: {'Yes' if terminated else 'No'}")
        
        # Save statistics
        self.save_stats(round_num, terminated, truncated)
        
        return terminated, truncated
    
    def run(self, total_rounds=DEFAULT_ROUNDS):
        """Run multiple rounds of maze navigation."""
        print(f"Starting automatic maze navigation - Difficulty: {self.difficulty}")
        print(f"Auto Speed: {self.auto_speed}s/step, Max Steps: {self.max_steps}")
        print(f"Will run for {total_rounds} rounds, statistics will be saved after each round")
        print(f"Results save directory: {self.results_dir}")
        
        # Check Baichuan API connection
        try:
            response = self.session.get(f"{BAICHUAN_FASTAPI_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Baichuan Model API connection is normal")
            else:
                print("❌ Baichuan Model API connection is abnormal")
                return
        except Exception as e:
            print(f"❌ Cannot connect to Baichuan Model API: {e}")
            return
        
        total_rounds = 5
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
            print("\n===== All Rounds Statistics =====")
            print(f"Completed Rounds: {completed_rounds}/{total_rounds}")
            print(f"Successful Rounds: {success_rounds}/{total_rounds}")
            print(f"Success Rate: {(success_rounds/total_rounds)*100:.1f}%")
            
        except KeyboardInterrupt:
            print("\nUser interrupted the test")
            print(f"Completed rounds: {completed_rounds}/{total_rounds}")
        finally:
            # Clean up session and resources
            self.clear_session()
            self.env.close()
            pygame.quit()
            
            # Clean up temporary files
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Error while cleaning up temporary files: {e}")
            
            print("\nAutomatic navigation test finished!")


def select_difficulty():
    """Let the user select the maze difficulty."""
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
            print("Invalid choice, please enter again.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Maze Navigation Test - Baichuan Model")
    parser.add_argument("--difficulty", type=str, default=DEFAULT_DIFFICULTY, 
                       choices=["easy", "medium", "hard"], help="Maze difficulty")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Number of test rounds")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum steps per round")
    parser.add_argument("--speed", type=float, default=DEFAULT_AUTO_SPEED, help="Auto-run speed")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed value")
    parser.add_argument("--no-sequential-seeds", action="store_true", help="Disable sequential seed mode")
    parser.add_argument("--interactive", action="store_true", help="Interactively select difficulty")
    
    args = parser.parse_args()
    
    # Interactive difficulty selection
    if args.interactive:
        difficulty = select_difficulty()
    else:
        difficulty = args.difficulty
    
    # Create the auto-runner and run the test
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