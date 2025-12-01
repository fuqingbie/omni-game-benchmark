import os
import time
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime
import copy
import pathlib

# Import game environment and AI controller
from bomberman_gym import BombermanEnv, BombermanAction
from ai_player_controller import AIPlayerController
from game_difficulty import DifficultyLevel

class MultiModelGame:
    """Multi-Model Bomberman Game Controller"""
    
    def __init__(self, model_configs, episodes=3, steps_per_episode=300, delay=0.3, 
                 difficulty=DifficultyLevel.NORMAL, results_dir: str = "results/bomberman"):
        """
        Initialize the multi-model game
        
        Args:
            model_configs: List of model configurations [{"api_base": "...", "api_key": "...", "model": "...", "player_id": 0}, ...]
            episodes: Number of game episodes
            steps_per_episode: Maximum steps per episode
            delay: Delay per step (seconds)
            difficulty: Game difficulty
            results_dir: Directory to save results
        """
        self.model_configs = model_configs
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.delay = delay
        self.difficulty = difficulty
        self.results_dir = results_dir
        
        # Create results directory
        pathlib.Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
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
        
        # Initialize detailed statistics
        self.detailed_stats = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "difficulty": difficulty.value if hasattr(difficulty, 'value') else str(difficulty),
                "episodes": episodes,
                "steps_per_episode": steps_per_episode,
                "num_players": len(model_configs)
            },
            "players": {},
            "episodes": [],
            "summary": {}
        }
        
        # Initialize player stats
        for config in model_configs:
            pid = config["player_id"]
            self.detailed_stats["players"][pid] = {
                "model": config["model"],
                "api_base": config.get("api_base", ""),
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "total_kills": 0,
                "total_deaths": 0,
                "total_items_collected": 0,
                "total_bombs_placed": 0,
                "total_steps_survived": 0
            }
    
    def run_game(self):
        """Run the game"""
        try:
            stats = {"wins": {pid: 0 for pid in self.controllers.keys()}}
            
            for episode in range(self.episodes):
                print(f"\n====== Game {episode+1}/{self.episodes} ======")
                obs, info = self.env.reset()
                
                # Track episode data
                episode_data = {
                    "episode": episode + 1,
                    "start_time": datetime.now().isoformat(),
                    "steps": 0,
                    "winner": None,
                    "result": "incomplete",
                    "player_stats": {}
                }
                
                # Initialize per-episode player stats
                for config in self.model_configs:
                    pid = config["player_id"]
                    episode_data["player_stats"][pid] = {
                        "model": config["model"],
                        "kills": 0,
                        "deaths": 0,
                        "items_collected": 0,
                        "bombs_placed": 0,
                        "final_alive": False
                    }
                
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
                        
                        # Track bomb placement
                        if action == 5:  # BOMB action
                            episode_data["player_stats"][player_id]["bombs_placed"] += 1
                    
                    # Execute actions
                    obs, rewards, terminated, truncated, info = self.env.step(actions)
                    episode_data["steps"] = step + 1
                    
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
                                episode_data["player_stats"][int(pid)]["final_alive"] = True
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
                            
                            # Update episode data
                            episode_data["winner"] = winner_id
                            episode_data["result"] = "win"
                            
                            # Update detailed stats
                            self.detailed_stats["players"][winner_id]["wins"] += 1
                            for pid in self.controllers.keys():
                                if pid != winner_id:
                                    self.detailed_stats["players"][pid]["losses"] += 1
                        else:
                            print("Game Over! It's a draw!")
                            episode_data["result"] = "draw"
                            for pid in self.controllers.keys():
                                self.detailed_stats["players"][pid]["draws"] += 1
                        break
                    elif truncated:
                        print(f"Maximum steps {self.steps_per_episode} reached!")
                        episode_data["result"] = "truncated"
                        # Check alive players for truncated games
                        for pid, p_info in obs['state']['players'].items():
                            if p_info['alive'] == 1:
                                episode_data["player_stats"][int(pid)]["final_alive"] = True
                        break
                
                # Calculate episode time
                ep_time = time.time() - ep_start_time
                episode_data["duration_seconds"] = ep_time
                episode_data["end_time"] = datetime.now().isoformat()
                
                # Update cumulative stats
                for pid in self.controllers.keys():
                    ep_stats = episode_data["player_stats"][pid]
                    self.detailed_stats["players"][pid]["total_bombs_placed"] += ep_stats["bombs_placed"]
                    self.detailed_stats["players"][pid]["total_steps_survived"] += episode_data["steps"]
                
                # Add episode to detailed stats
                self.detailed_stats["episodes"].append(episode_data)
                
                print(f"Game {episode+1} completed, took: {ep_time:.2f} seconds")
                
                # Save intermediate results after each episode
                self._save_results()
            
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
            
            # Final save
            self._finalize_and_save_results()
        
        except KeyboardInterrupt:
            print("User interrupted, ending game")
            self._finalize_and_save_results()
        except Exception as e:
            print(f"An error occurred during the game: {e}")
            import traceback
            traceback.print_exc()
            self._finalize_and_save_results()
        finally:
            # Ensure the environment is closed properly
            self.env.close()
    
    def _save_results(self):
        """Save intermediate results to file"""
        timestamp = self.detailed_stats["experiment_info"]["start_time"].replace(":", "-").replace(".", "-")
        filename = f"results_{timestamp}.json"
        filepath = pathlib.Path(self.results_dir) / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_stats, f, indent=2, ensure_ascii=False)
    
    def _finalize_and_save_results(self):
        """Finalize and save all results"""
        # Calculate summary statistics
        total_episodes = len(self.detailed_stats["episodes"])
        
        summary = {
            "total_episodes_completed": total_episodes,
            "total_episodes_planned": self.episodes,
            "end_time": datetime.now().isoformat(),
            "player_summaries": {}
        }
        
        for pid, pstats in self.detailed_stats["players"].items():
            win_rate = pstats["wins"] / total_episodes if total_episodes > 0 else 0
            summary["player_summaries"][pid] = {
                "model": pstats["model"],
                "win_rate": win_rate,
                "wins": pstats["wins"],
                "losses": pstats["losses"],
                "draws": pstats["draws"],
                "avg_bombs_placed": pstats["total_bombs_placed"] / total_episodes if total_episodes > 0 else 0,
                "avg_steps_survived": pstats["total_steps_survived"] / total_episodes if total_episodes > 0 else 0
            }
        
        self.detailed_stats["summary"] = summary
        
        # Save final results
        self._save_results()
        
        # Print summary
        print("\n====== Final Results Saved ======")
        print(f"Results saved to: {self.results_dir}")
        for pid, psummary in summary["player_summaries"].items():
            print(f"Player {pid+1} ({psummary['model']}): Win Rate = {psummary['win_rate']*100:.1f}%")

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Bomberman Battle")
    parser.add_argument("--config", type=str, default="model_config_example.json", help="Path to the model configuration file (JSON format)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of game episodes")
    parser.add_argument("--steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay per step (seconds)")
    parser.add_argument("--difficulty", type=str, choices=['easy', 'normal', 'hard'], 
                       default='normal', help="Game difficulty: easy, normal, hard")
    parser.add_argument("--results-dir", type=str, default="results/bomberman",
                       help="Directory to save results (default: results/bomberman)")
    
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
        difficulty=difficulty,
        results_dir=args.results_dir
    )
    
    game.run_game()