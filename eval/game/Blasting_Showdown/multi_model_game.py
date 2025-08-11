import os
import time
import json
import argparse
from typing import Dict, List
import copy

# Import game environment and AI controller
from bomberman_gym import BombermanEnv, BombermanAction
from ai_player_controller import AIPlayerController
from game_difficulty import DifficultyLevel

class MultiModelGame:
    """Multi-Model Bomberman Game Controller"""
    
    def __init__(self, model_configs, episodes=3, steps_per_episode=300, delay=0.3, difficulty=DifficultyLevel.NORMAL):
        """
        Initialize the multi-model game
        
        Args:
            model_configs: List of model configurations [{"api_base": "...", "api_key": "...", "model": "...", "player_id": 0}, ...]
            episodes: Number of game episodes
            steps_per_episode: Maximum steps per episode
            delay: Delay per step (seconds)
            difficulty: Game difficulty
        """
        self.model_configs = model_configs
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.delay = delay
        self.difficulty = difficulty
        
        # Validate configuration
        assert len(model_configs) <= 4, "A maximum of 4 models are supported"
        player_ids = [config.get("player_id") for config in model_configs]
        assert len(player_ids) == len(set(player_ids)), "Player IDs cannot be repeated"
        assert all(0 <= pid <= 3 for pid in player_ids), "Player ID must be between 0-3"
        
        # Create game environment
        self.env = BombermanEnv(render_mode='human', difficulty=difficulty)
        
        # Create AI controllers
        self.controllers = {}
        for config in model_configs:
            player_id = config["player_id"]
            self.controllers[player_id] = AIPlayerController(
                config["api_base"], 
                config["api_key"],
                config["model"]
            )
    
    def run_game(self):
        """Run the game"""
        try:
            stats = {"wins": {pid: 0 for pid in self.controllers.keys()}}
            
            for episode in range(self.episodes):
                print(f"\n====== Game {episode+1}/{self.episodes} ======")
                obs, info = self.env.reset()
                
                ep_start_time = time.time()
                for step in range(self.steps_per_episode):
                    step_start_time = time.time()
                    
                    # Get decisions for all active players
                    actions = {}
                    for player_id, controller in self.controllers.items():
                        # Check if the player is alive
                        if obs['state']['players'][player_id]['alive'] == 0:
                            continue
                        
                        # Get AI decision
                        action = controller.get_decision(player_id, obs)
                        actions[player_id] = action
                    
                    # Execute actions
                    obs, rewards, terminated, truncated, info = self.env.step(actions)
                    
                    # Calculate step duration
                    step_time = time.time() - step_start_time
                    print(f"Step {step+1} took: {step_time:.2f} seconds")

                    # Add a delay in the GUI for observation
                    if step_time < self.delay:
                        time.sleep(self.delay - step_time)
                    
                    # Check if the game is over
                    if terminated:
                        # Find the winner
                        winner_id = None
                        for pid, p_info in obs['state']['players'].items():
                            if p_info['alive'] == 1:
                                winner_id = int(pid)
                                break
                        
                        if winner_id is not None:
                            # Find the model name corresponding to the winner_id
                            winner_model_name = "Unknown"
                            for config in self.model_configs:
                                if config['player_id'] == winner_id:
                                    winner_model_name = config['model']
                                    break
                            
                            print(f"Game Over! Player {winner_id+1} (Model: {winner_model_name}) wins!")
                            
                            # Record win
                            if winner_id in stats["wins"]:
                                stats["wins"][winner_id] += 1
                        else:
                            print("Game Over! It's a draw!")
                        break
                    elif truncated:
                        print(f"Maximum steps {self.steps_per_episode} reached!")
                        break
                
                # Calculate episode time
                ep_time = time.time() - ep_start_time
                print(f"Game {episode+1} completed, took: {ep_time:.2f} seconds")
            
            # Output statistics
            print("\n====== Game Statistics ======")
            for player_id, wins in stats["wins"].items():
                # Find the model name corresponding to the player_id
                model_name = "Unknown"
                for config in self.model_configs:
                    if config['player_id'] == player_id:
                        model_name = config['model']
                        break
                print(f"Player {player_id+1} ({model_name}): Won {wins}/{self.episodes} games ({wins/self.episodes*100:.1f}%)")
        
        except KeyboardInterrupt:
            print("User interrupted, ending game")
        except Exception as e:
            print(f"An error occurred during the game: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure the environment is closed properly
            self.env.close()

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Bomberman Battle")
    parser.add_argument("--config", type=str, default="model_config_example.json", help="Path to the model configuration file (JSON format)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of game episodes")
    parser.add_argument("--steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay per step (seconds)")
    parser.add_argument("--difficulty", type=str, choices=['easy', 'normal', 'hard'], 
                       default='normal', help="Game difficulty: easy, normal, hard")
    
    args = parser.parse_args()
    
    # Convert difficulty string to enum type
    difficulty_map = {
        'easy': DifficultyLevel.EASY,
        'normal': DifficultyLevel.NORMAL,
        'hard': DifficultyLevel.HARD
    }
    difficulty = difficulty_map.get(args.difficulty, DifficultyLevel.NORMAL)
    
    # Read configuration file
    try:
        with open(args.config, 'r') as f:
            model_configs = json.load(f)
    except Exception as e:
        print(f"Failed to read configuration file: {e}")
        print("Example configuration file format:")
        print("""
        [
            {
                "api_base": "https://api.example.com",
                "api_key": "your_api_key_1",
                "model": "model_name_1",
                "player_id": 0
            },
            {
                "api_base": "https://api.example.com",
                "api_key": "your_api_key_2",
                "model": "model_name_2",
                "player_id": 1
            }
        ]
        """)
        exit(1)
    
    # Create and run the game
    game = MultiModelGame(
        model_configs=model_configs,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        delay=args.delay,
        difficulty=difficulty
    )
    
    game.run_game()