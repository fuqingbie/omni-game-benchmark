"""
Runner - Unified evaluation runner.

This module provides the main benchmark runner that orchestrates
game evaluations across different models and games.
"""

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base_env import BaseGameEnv
from .game_registry import GameRegistry, GameEntry, get_game_registry
from .model_registry import ModelConfig, ModelRegistry, get_default_registry
from .result_schema import (
    EpisodeResult,
    GameResult,
    BenchmarkResult,
    ResultWriter,
    StepResult,
)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    # Games and models to evaluate
    games: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    
    # Evaluation parameters
    episodes_per_game: int = 10
    max_steps_per_episode: int = 1000
    timeout_per_episode: float = 300.0  # seconds
    
    # Environment settings
    difficulty: str = "normal"
    render: bool = False
    
    # Output settings
    output_dir: str = "benchmark_results"
    save_episodes: bool = True
    save_steps: bool = False  # Detailed step-by-step data
    save_video: bool = False
    
    # Execution settings
    parallel: bool = False
    num_workers: int = 1
    seed: Optional[int] = None
    
    # Retry settings
    retry_on_error: bool = True
    max_retries: int = 3
    
    # Agent settings
    agent_class: Optional[Type] = None
    agent_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "games": self.games,
            "models": self.models,
            "episodes_per_game": self.episodes_per_game,
            "max_steps_per_episode": self.max_steps_per_episode,
            "timeout_per_episode": self.timeout_per_episode,
            "difficulty": self.difficulty,
            "render": self.render,
            "output_dir": self.output_dir,
            "save_episodes": self.save_episodes,
            "save_steps": self.save_steps,
            "save_video": self.save_video,
            "parallel": self.parallel,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "retry_on_error": self.retry_on_error,
            "max_retries": self.max_retries,
            "agent_config": self.agent_config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """Create from dictionary."""
        return cls(
            games=data.get("games", []),
            models=data.get("models", []),
            episodes_per_game=data.get("episodes_per_game", 10),
            max_steps_per_episode=data.get("max_steps_per_episode", 1000),
            timeout_per_episode=data.get("timeout_per_episode", 300.0),
            difficulty=data.get("difficulty", "normal"),
            render=data.get("render", False),
            output_dir=data.get("output_dir", "benchmark_results"),
            save_episodes=data.get("save_episodes", True),
            save_steps=data.get("save_steps", False),
            save_video=data.get("save_video", False),
            parallel=data.get("parallel", False),
            num_workers=data.get("num_workers", 1),
            seed=data.get("seed"),
            retry_on_error=data.get("retry_on_error", True),
            max_retries=data.get("max_retries", 3),
            agent_config=data.get("agent_config", {}),
        )
    
    @classmethod
    def from_file(cls, filepath: str) -> "RunConfig":
        """Load configuration from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                data = json.load(f)
        return cls.from_dict(data)


class BenchmarkRunner:
    """
    Main benchmark runner that orchestrates evaluations.
    """
    
    def __init__(
        self,
        config: RunConfig,
        game_registry: Optional[GameRegistry] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Run configuration
            game_registry: Game registry (uses default if None)
            model_registry: Model registry (uses default if None)
        """
        self.config = config
        self.game_registry = game_registry or get_game_registry()
        self.model_registry = model_registry or get_default_registry()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Result writer
        self.result_writer = ResultWriter(str(self.output_dir))
        
        # Callbacks
        self._on_episode_start: Optional[Callable] = None
        self._on_episode_end: Optional[Callable] = None
        self._on_game_complete: Optional[Callable] = None
        
        # Results
        self.benchmark_result: Optional[BenchmarkResult] = None
    
    def run(self) -> BenchmarkResult:
        """
        Run the complete benchmark.
        
        Returns:
            BenchmarkResult with all results
        """
        # Initialize benchmark result
        self.benchmark_result = BenchmarkResult(
            benchmark_name="omni-game-benchmark",
            config=self.config.to_dict(),
            system_info=self._get_system_info(),
        )
        
        print(f"Starting benchmark with {len(self.config.games)} games and {len(self.config.models)} models")
        print(f"Output directory: {self.output_dir}")
        
        # Run for each game-model combination
        for game_name in self.config.games:
            for model_name in self.config.models:
                try:
                    game_result = self.run_game(game_name, model_name)
                    self.benchmark_result.add_game_result(game_result)
                except Exception as e:
                    print(f"Error running {game_name} with {model_name}: {e}")
                    # Create error result
                    error_result = GameResult(
                        game_name=game_name,
                        model_name=model_name,
                    )
                    error_result.config["error"] = str(e)
                    self.benchmark_result.add_game_result(error_result)
        
        # Finalize
        self.benchmark_result.finalize()
        
        # Save results
        self.result_writer.write_benchmark(self.benchmark_result)
        self.result_writer.write_summary_csv(self.benchmark_result)
        
        print(f"Benchmark complete. Results saved to {self.output_dir}")
        
        return self.benchmark_result
    
    def run_game(self, game_name: str, model_name: str) -> GameResult:
        """
        Run evaluation for a single game-model combination.
        
        Args:
            game_name: Name of the game
            model_name: Name of the model
            
        Returns:
            GameResult with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {game_name} with {model_name}")
        print(f"{'='*60}")
        
        # Get game entry
        game_entry = self.game_registry.get(game_name)
        if game_entry is None:
            raise ValueError(f"Game '{game_name}' not registered")
        
        # Get model config
        model_config = self.model_registry.get(model_name)
        if model_config is None:
            raise ValueError(f"Model '{model_name}' not registered")
        
        # Initialize game result
        game_result = GameResult(
            game_name=game_name,
            model_name=model_name,
            config={
                "difficulty": self.config.difficulty,
                "episodes": self.config.episodes_per_game,
                "max_steps": self.config.max_steps_per_episode,
            }
        )
        
        # Create agent
        agent = self._create_agent(model_config)
        
        # Create environment
        env = game_entry.create_env(
            difficulty=self.config.difficulty,
            render_mode="rgb_array" if self.config.render else None,
        )
        
        try:
            # Run episodes
            for episode_num in range(self.config.episodes_per_game):
                print(f"\nEpisode {episode_num + 1}/{self.config.episodes_per_game}")
                
                episode_result = self.run_episode(
                    env=env,
                    agent=agent,
                    game_name=game_name,
                    model_name=model_name,
                    episode_id=episode_num + 1,
                )
                
                game_result.add_episode(episode_result)
                
                # Print episode summary
                print(f"  Reward: {episode_result.total_reward:.2f}")
                print(f"  Steps: {episode_result.total_steps}")
                print(f"  Status: {episode_result.status}")
                
                # Save episode if configured
                if self.config.save_episodes:
                    self.result_writer.write_episode(
                        episode_result,
                        include_steps=self.config.save_steps
                    )
        
        finally:
            env.close()
        
        # Finalize game result
        game_result.finalize()
        
        # Save game result
        self.result_writer.write_game(game_result)
        
        # Print summary
        print(f"\nGame Summary:")
        print(f"  Mean Score: {game_result.mean_score:.2f} ± {game_result.std_score:.2f}")
        print(f"  Win Rate: {game_result.win_rate:.1%}")
        print(f"  Total Duration: {game_result.total_duration_seconds:.1f}s")
        
        # Callback
        if self._on_game_complete:
            self._on_game_complete(game_result)
        
        return game_result
    
    def run_episode(
        self,
        env: BaseGameEnv,
        agent: Any,
        game_name: str,
        model_name: str,
        episode_id: int,
    ) -> EpisodeResult:
        """
        Run a single episode.
        
        Args:
            env: Game environment
            agent: Agent to use
            game_name: Name of the game
            model_name: Name of the model
            episode_id: Episode number
            
        Returns:
            EpisodeResult
        """
        # Initialize episode result
        episode_result = EpisodeResult(
            episode_id=episode_id,
            game_name=game_name,
            model_name=model_name,
        )
        
        # Callback
        if self._on_episode_start:
            self._on_episode_start(episode_result)
        
        try:
            # Reset environment
            obs, info = env.reset(seed=self._get_episode_seed(episode_id))
            
            # Episode loop
            for step_num in range(self.config.max_steps_per_episode):
                # Get action from agent
                action = agent.act(obs)
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record step
                if self.config.save_steps:
                    step_result = StepResult(
                        step=step_num + 1,
                        action=action,
                        reward=reward,
                        info=info,
                    )
                    episode_result.add_step(step_result)
                else:
                    episode_result.total_reward += reward
                    episode_result.total_steps = step_num + 1
                
                # Check termination
                if terminated or truncated:
                    break
            
            # Get final state
            episode_result.final_score = env.get_score()
            episode_result.win = env.is_win()
            
            # Get game-specific metrics
            if hasattr(env, 'get_state_for_agent'):
                episode_result.game_metrics = env.get_state_for_agent()
            
            episode_result.mark_complete("success")
        
        except Exception as e:
            episode_result.error_message = str(e)
            episode_result.mark_complete("error")
            
            if self.config.retry_on_error:
                # Could implement retry logic here
                pass
        
        # Callback
        if self._on_episode_end:
            self._on_episode_end(episode_result)
        
        return episode_result
    
    def _create_agent(self, model_config: ModelConfig) -> Any:
        """Create an agent for the model."""
        if self.config.agent_class:
            # Use custom agent class
            return self.config.agent_class(
                model_config=model_config,
                **self.config.agent_config
            )
        
        # Use default API agent
        return DefaultAPIAgent(model_config, **self.config.agent_config)
    
    def _get_episode_seed(self, episode_id: int) -> Optional[int]:
        """Get seed for an episode."""
        if self.config.seed is not None:
            return self.config.seed + episode_id
        return None
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Callback setters
    def on_episode_start(self, callback: Callable):
        """Set callback for episode start."""
        self._on_episode_start = callback
    
    def on_episode_end(self, callback: Callable):
        """Set callback for episode end."""
        self._on_episode_end = callback
    
    def on_game_complete(self, callback: Callable):
        """Set callback for game completion."""
        self._on_game_complete = callback


class DefaultAPIAgent:
    """
    Default agent that uses API calls for decisions.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the agent.
        
        Args:
            model_config: Model configuration
            verbose: Whether to print debug info
            **kwargs: Additional configuration
        """
        self.model_config = model_config
        self.verbose = verbose
        self.extra_config = kwargs
        
        # Initialize session
        self._session = None
    
    def act(self, obs: Dict[str, Any]) -> Any:
        """
        Get action from model.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Action to take
        """
        # This is a placeholder - actual implementation would call the model API
        # and parse the response to get an action
        
        # For now, return random action
        import random
        return random.randint(0, 6)
    
    def reset(self):
        """Reset agent state for new episode."""
        pass


def run_benchmark(config_path: str) -> BenchmarkResult:
    """
    Convenience function to run a benchmark from a config file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        BenchmarkResult
    """
    config = RunConfig.from_file(config_path)
    runner = BenchmarkRunner(config)
    return runner.run()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run game benchmark")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--games", nargs="+", help="Override games to evaluate")
    parser.add_argument("--models", nargs="+", help="Override models to use")
    parser.add_argument("--episodes", type=int, help="Override episodes per game")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load config
    config = RunConfig.from_file(args.config)
    
    # Apply overrides
    if args.games:
        config.games = args.games
    if args.models:
        config.models = args.models
    if args.episodes:
        config.episodes_per_game = args.episodes
    if args.output:
        config.output_dir = args.output
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    result = runner.run()
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Games evaluated: {result.games_evaluated}")
    print(f"Models evaluated: {result.models_evaluated}")
    print(f"Total episodes: {result.total_episodes}")
    print(f"Total duration: {result.total_duration_seconds:.1f}s")
    
    # Print leaderboard
    print("\nLeaderboard (by mean score):")
    for i, entry in enumerate(result.get_leaderboard("mean_score")[:10], 1):
        print(f"  {i}. {entry['game']} / {entry['model']}: {entry['mean_score']:.2f}")


if __name__ == "__main__":
    main()
