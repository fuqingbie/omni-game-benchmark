import gym
import numpy as np
import pygame
import matplotlib
matplotlib.use('Agg')  # Use a backend that does not require a GUI
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image
import tempfile
import os
from maze_gym_env import MazeGymEnv

class InteractiveMazeRunner:
    """Interactive Maze Runner, allows the user to control the character through terminal input."""
    
    def __init__(self, difficulty='easy'):
        # Initialize Pygame (for audio playback)
        pygame.init()
        pygame.mixer.init()
        
        # Store difficulty
        self.difficulty = difficulty
        
        # Create the environment, enable voice guidance, and set the difficulty
        self.env = MazeGymEnv(render_mode="human", voice_guidance=True, difficulty=self.difficulty)
        
        # Create a temporary directory to store audio files
        self.temp_dir = tempfile.mkdtemp(prefix="interactive_maze_")
        
        # Track total reward
        self.total_reward = 0
    
    def play_audio(self, audio_base64):
        """Play audio guidance."""
        if not audio_base64:
            return
            
        try:
            # Decode base64 into an audio file
            audio_data = base64.b64decode(audio_base64)
            audio_file = os.path.join(self.temp_dir, "current_guidance.wav")
            
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            
            # Play the audio
            sound = pygame.mixer.Sound(audio_file)
            sound.play()
            
            # Wait for the audio to finish playing
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def get_user_input(self):
        """Get the user's input action."""
        print("\n===== Please Enter Action =====")
        print("Action format: [Forward Distance] [Rotation Angle]")
        print("Forward distance range: [-1.0, 3.0], negative value means backward")
        print("Rotation angle range: [-180.0, 180.0], negative value means turn left")
        print("Example: '1.0 45' means move forward 1.0 unit and rotate 45 degrees to the right")
        print("Enter 'q' to quit the game")
        print("Enter 'r' to reset the game")
        print("Enter 'd' to change difficulty")
        
        while True:
            try:
                user_input = input("> ")
                
                # Check for special commands
                if user_input.lower() == 'q':
                    return None
                elif user_input.lower() == 'r':
                    return 'reset'
                elif user_input.lower() == 'd':
                    return 'change_difficulty'
                    
                # Parse input
                values = user_input.split()
                if len(values) != 2:
                    print("Error: Please enter two values, separated by a space")
                    continue
                    
                forward = float(values[0])
                rotation = float(values[1])
                
                # Validate range
                if not (-1.0 <= forward <= 3.0):
                    print("Error: Forward distance must be in the range [-1.0, 3.0]")
                    continue
                    
                if not (-180.0 <= rotation <= 180.0):
                    print("Error: Rotation angle must be in the range [-180.0, 180.0]")
                    continue
                    
                # Return the valid action
                return np.array([forward, rotation], dtype=np.float32)
                
            except ValueError:
                print("Error: Please enter valid numbers")
                continue
    
    def print_observation_info(self, observation):
        """Print observation information."""
        vector_obs = observation['vector']
        
        print("\n===== Current State =====")
        print(f"Difficulty: {self.difficulty}")
        print(f"Position: ({vector_obs[0]:.2f}, {vector_obs[1]:.2f})")
        print(f"Orientation: {np.degrees(vector_obs[2]):.1f}°")
        print(f"Distance to Goal: {vector_obs[3]:.2f} meters")
        print(f"Angle to Goal: {np.degrees(vector_obs[4]):.1f}°")
    
    def change_difficulty(self):
        """Change the game difficulty."""
        print("\nSelect new difficulty:")
        print("1. Easy")
        print("2. Medium")
        print("3. Hard")
        
        while True:
            choice = input("Please select (1-3): ")
            if choice == "1":
                new_difficulty = "easy"
                break
            elif choice == "2":
                new_difficulty = "medium"
                break
            elif choice == "3":
                new_difficulty = "hard"
                break
            else:
                print("Invalid choice, please enter again")
        
        # Update difficulty and reset the environment
        self.difficulty = new_difficulty
        observation = self.env.set_difficulty(new_difficulty)
        if observation:
            print(f"\nDifficulty changed to: {new_difficulty}")
            self.total_reward = 0
            return observation
        return None
    
    def run(self):
        """Run the interactive maze environment."""
        print(f"Welcome to the Interactive Maze Environment! Difficulty: {self.difficulty}")
        print("Loading environment...")
        
        # Reset the environment
        observation, info = self.env.reset()
        
        # Display initial state information
        self.print_observation_info(observation)
        
        # Play initial audio guidance
        if 'audio' in info:
            print("Playing voice guidance...")
            self.play_audio(info['audio'])
        
        # Interactive loop
        done = False
        while not done:
            # Get user's input action
            action = self.get_user_input()
            
            if action is None:  # User chose to quit
                break
            elif action == 'reset':  # User chose to reset
                observation, info = self.env.reset()
                self.total_reward = 0
                self.print_observation_info(observation)
                if 'audio' in info:
                    self.play_audio(info['audio'])
                continue
            elif action == 'change_difficulty':  # User chose to change difficulty
                new_observation = self.change_difficulty()
                if new_observation:
                    observation = new_observation
                    self.print_observation_info(observation)
                continue
                
            # Execute the action
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            
            # Render the current environment
            self.env.render()
            
            # Print information
            self.print_observation_info(observation)
            print(f"Executed Action: Forward={action[0]:.2f}, Rotation={action[1]:.2f}°")
            print(f"Reward: {reward:.2f}, Total Reward: {self.total_reward:.2f}")
            
            # Play audio guidance (if any)
            if 'audio' in info:
                print("Playing voice guidance...")
                self.play_audio(info['audio'])
            
            # Check if the episode is finished
            done = terminated or truncated
            if terminated:
                print("\nCongratulations! You have reached the goal!")
            elif truncated:
                print("\nEpisode finished!")
        
        # Clean up resources
        self.env.close()
        pygame.quit()
        
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        
        print(f"\nGame Over! Total Reward: {self.total_reward:.2f}")

def select_difficulty():
    """Let the user select the maze difficulty."""
    print("Please select the maze difficulty:")
    print("1. Easy - Fewer walls, wide paths")
    print("2. Medium - Moderate obstacles, medium difficulty paths")
    print("3. Hard - Dense obstacles, narrower passages, larger maze")
    
    while True:
        choice = input("Please enter your choice (1-3): ")
        if choice == "1":
            return "easy"
        elif choice == "2":
            return "medium"
        elif choice == "3":
            return "hard"
        else:
            print("Invalid choice, please enter again.")

if __name__ == "__main__":
    # Select difficulty level
    difficulty = select_difficulty()
    
    # Create and run the interactive maze
    runner = InteractiveMazeRunner(difficulty=difficulty)
    runner.run()
    # Create and run the interactive maze
    runner = InteractiveMazeRunner(difficulty=difficulty)
    runner.run()
    runner.run()
    # Create and run the interactive maze
    runner = InteractiveMazeRunner(difficulty=difficulty)
    runner.run()