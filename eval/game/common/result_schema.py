"""
Result Schema - Standardized result format across all games.

This module defines the standard result format for game evaluations,
ensuring consistency and comparability across different games and models.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ResultStatus(Enum):
    """Status of an evaluation result."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    INCOMPLETE = "incomplete"


@dataclass
class StepResult:
    """Result of a single step in an episode."""
    step: int
    action: Any
    reward: float = 0.0
    observation_summary: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    episode_id: int
    game_name: str
    model_name: str
    
    # Core metrics
    total_reward: float = 0.0
    total_steps: int = 0
    status: str = "incomplete"
    
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    
    # Episode-specific data
    final_score: float = 0.0
    win: bool = False
    
    # Detailed step data (optional)
    steps: List[StepResult] = field(default_factory=list)
    
    # Game-specific metrics
    game_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error info if any
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
    
    def mark_complete(self, status: str = "success"):
        """Mark episode as complete."""
        self.end_time = datetime.now().isoformat()
        self.status = status
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            self.duration_seconds = (end - start).total_seconds()
    
    def add_step(self, step_result: StepResult):
        """Add a step result."""
        self.steps.append(step_result)
        self.total_steps = len(self.steps)
        self.total_reward += step_result.reward
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "game_name": self.game_name,
            "model_name": self.model_name,
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "final_score": self.final_score,
            "win": self.win,
            "game_metrics": self.game_metrics,
            "error_message": self.error_message,
            "steps_count": len(self.steps),
        }
    
    def to_dict_full(self) -> Dict[str, Any]:
        """Convert to dictionary with all step data."""
        result = self.to_dict()
        result["steps"] = [asdict(s) for s in self.steps]
        return result


@dataclass
class GameResult:
    """Aggregated result for a game across multiple episodes."""
    game_name: str
    model_name: str
    
    # Episode results
    episodes: List[EpisodeResult] = field(default_factory=list)
    
    # Aggregate metrics
    total_episodes: int = 0
    completed_episodes: int = 0
    successful_episodes: int = 0
    
    # Score statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    
    mean_score: float = 0.0
    std_score: float = 0.0
    
    # Win rate
    win_rate: float = 0.0
    
    # Timing
    total_duration_seconds: float = 0.0
    mean_episode_duration: float = 0.0
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
    
    def add_episode(self, episode: EpisodeResult):
        """Add an episode result and update aggregates."""
        self.episodes.append(episode)
        self._update_aggregates()
    
    def _update_aggregates(self):
        """Update aggregate metrics from episodes."""
        if not self.episodes:
            return
        
        self.total_episodes = len(self.episodes)
        
        # Count completed and successful
        self.completed_episodes = sum(
            1 for e in self.episodes if e.status in ["success", "completed"]
        )
        self.successful_episodes = sum(
            1 for e in self.episodes if e.win
        )
        
        # Reward statistics
        rewards = [e.total_reward for e in self.episodes]
        self.mean_reward = sum(rewards) / len(rewards)
        
        if len(rewards) > 1:
            variance = sum((r - self.mean_reward) ** 2 for r in rewards) / len(rewards)
            self.std_reward = variance ** 0.5
        else:
            self.std_reward = 0.0
        
        self.min_reward = min(rewards)
        self.max_reward = max(rewards)
        
        # Score statistics
        scores = [e.final_score for e in self.episodes]
        self.mean_score = sum(scores) / len(scores)
        
        if len(scores) > 1:
            variance = sum((s - self.mean_score) ** 2 for s in scores) / len(scores)
            self.std_score = variance ** 0.5
        else:
            self.std_score = 0.0
        
        # Win rate
        self.win_rate = self.successful_episodes / self.total_episodes
        
        # Timing
        self.total_duration_seconds = sum(e.duration_seconds for e in self.episodes)
        self.mean_episode_duration = self.total_duration_seconds / self.total_episodes
    
    def finalize(self):
        """Finalize the game result."""
        self.end_time = datetime.now().isoformat()
        self._update_aggregates()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_name": self.game_name,
            "model_name": self.model_name,
            "total_episodes": self.total_episodes,
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "win_rate": self.win_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "mean_episode_duration": self.mean_episode_duration,
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "episodes": [e.to_dict() for e in self.episodes],
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result across multiple games and models."""
    benchmark_name: str
    benchmark_version: str = "1.0"
    
    # Game results
    game_results: List[GameResult] = field(default_factory=list)
    
    # Summary metrics
    games_evaluated: int = 0
    models_evaluated: int = 0
    total_episodes: int = 0
    
    # Overall timing
    total_duration_seconds: float = 0.0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    # System info
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
    
    def add_game_result(self, game_result: GameResult):
        """Add a game result."""
        self.game_results.append(game_result)
        self._update_summary()
    
    def _update_summary(self):
        """Update summary metrics."""
        if not self.game_results:
            return
        
        self.games_evaluated = len(set(gr.game_name for gr in self.game_results))
        self.models_evaluated = len(set(gr.model_name for gr in self.game_results))
        self.total_episodes = sum(gr.total_episodes for gr in self.game_results)
        self.total_duration_seconds = sum(gr.total_duration_seconds for gr in self.game_results)
    
    def finalize(self):
        """Finalize the benchmark result."""
        self.end_time = datetime.now().isoformat()
        self._update_summary()
    
    def get_leaderboard(self, metric: str = "mean_score") -> List[Dict[str, Any]]:
        """Get a leaderboard sorted by metric."""
        entries = []
        for gr in self.game_results:
            value = getattr(gr, metric, 0)
            entries.append({
                "game": gr.game_name,
                "model": gr.model_name,
                metric: value,
            })
        
        entries.sort(key=lambda x: x[metric], reverse=True)
        return entries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "games_evaluated": self.games_evaluated,
            "models_evaluated": self.models_evaluated,
            "total_episodes": self.total_episodes,
            "total_duration_seconds": self.total_duration_seconds,
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "system_info": self.system_info,
            "game_results": [gr.to_dict() for gr in self.game_results],
        }


class ResultWriter:
    """Utility class for writing results to files."""
    
    def __init__(self, output_dir: str):
        """Initialize result writer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_episode(self, episode: EpisodeResult, include_steps: bool = False):
        """Write episode result to file."""
        filename = f"episode_{episode.game_name}_{episode.model_name}_{episode.episode_id}.json"
        filepath = self.output_dir / filename
        
        data = episode.to_dict_full() if include_steps else episode.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def write_game(self, game_result: GameResult):
        """Write game result to file."""
        filename = f"game_{game_result.game_name}_{game_result.model_name}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game_result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def write_benchmark(self, benchmark: BenchmarkResult):
        """Write complete benchmark result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{benchmark.benchmark_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(benchmark.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def write_summary_csv(self, benchmark: BenchmarkResult):
        """Write summary as CSV file."""
        import csv
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{benchmark.benchmark_name}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Game", "Model", "Episodes", "Win Rate", 
                "Mean Score", "Std Score", "Mean Reward", "Duration (s)"
            ])
            
            # Data rows
            for gr in benchmark.game_results:
                writer.writerow([
                    gr.game_name,
                    gr.model_name,
                    gr.total_episodes,
                    f"{gr.win_rate:.2%}",
                    f"{gr.mean_score:.2f}",
                    f"{gr.std_score:.2f}",
                    f"{gr.mean_reward:.2f}",
                    f"{gr.total_duration_seconds:.1f}",
                ])
        
        return filepath


def load_result(filepath: str) -> Union[EpisodeResult, GameResult, BenchmarkResult]:
    """Load a result from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Detect type based on fields
    if "game_results" in data:
        # Benchmark result
        result = BenchmarkResult(
            benchmark_name=data["benchmark_name"],
            benchmark_version=data.get("benchmark_version", "1.0"),
        )
        result.config = data.get("config", {})
        result.start_time = data.get("start_time")
        result.end_time = data.get("end_time")
        result.system_info = data.get("system_info", {})
        # Note: game_results would need recursive loading
        return result
    
    elif "episodes" in data:
        # Game result
        result = GameResult(
            game_name=data["game_name"],
            model_name=data["model_name"],
        )
        result.config = data.get("config", {})
        # Copy aggregates
        for key in ["total_episodes", "completed_episodes", "successful_episodes",
                    "mean_reward", "std_reward", "win_rate", "mean_score"]:
            if key in data:
                setattr(result, key, data[key])
        return result
    
    else:
        # Episode result
        return EpisodeResult(
            episode_id=data["episode_id"],
            game_name=data["game_name"],
            model_name=data["model_name"],
            total_reward=data.get("total_reward", 0),
            total_steps=data.get("total_steps", 0),
            status=data.get("status", "unknown"),
        )
