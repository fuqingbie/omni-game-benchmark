#!/usr/bin/env python
import os
import sys
import argparse
import subprocess

def main():
    """Launch the AI Bomberman game"""
    parser = argparse.ArgumentParser(description="Launch the AI Bomberman game")
    parser.add_argument("--config", type=str, default="model_config_example.json", 
                      help="Path to the model configuration file, defaults to model_config_example.json")
    parser.add_argument("--episodes", type=int, default=3, help="Number of game episodes, defaults to 3")
    parser.add_argument("--steps", type=int, default=300, help="Maximum steps per episode, defaults to 300")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay per step (seconds), defaults to 0.5")
    parser.add_argument("--difficulty", type=str, choices=['easy', 'normal', 'hard'], 
                       default='easy', help="Game difficulty: easy, normal, hard")

    args = parser.parse_args()
    
    # Check if the file exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)
    multi_model_game_path = os.path.join(script_dir, "multi_model_game.py")
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found")
        return
        
    if not os.path.exists(multi_model_game_path):
        print(f"Error: multi_model_game.py not found. Please ensure this file has been created.")
        return
    
    # Build the command
    cmd = [
        sys.executable,
        multi_model_game_path,
        "--config", config_path,
        "--episodes", str(args.episodes),
        "--steps", str(args.steps),
        "--delay", str(args.delay),
        "--difficulty", args.difficulty
    ]
    
    print("Launching AI Bomberman game...")
    print(f"Configuration file: {config_path}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.steps}")
    print(f"Delay per step: {args.delay} seconds")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nGame interrupted")
    except Exception as e:
        print(f"An error occurred during runtime: {e}")

if __name__ == "__main__":
    main()