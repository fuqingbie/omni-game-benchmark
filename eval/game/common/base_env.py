"""
Base Environment - Common interface for game environments.

This module defines the base class and interfaces that all game environments
should implement for consistency across the benchmark.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class GameInfo:
    """Information about a game environment."""
    name: str
    version: str = "1.0"
    
    # Game description
    description: str = ""
    
    # Action space info
    action_type: str = "discrete"  # "discrete", "continuous", "multi-discrete"
    num_actions: int = 0
    action_names: List[str] = field(default_factory=list)
    
    # Observation space info
    observation_type: str = "dict"  # "dict", "array", "image"
    has_image: bool = False
    has_audio: bool = False
    has_text: bool = False
    image_size: Optional[Tuple[int, int, int]] = None
    audio_sample_rate: int = 16000
    
    # Game parameters
    max_steps: int = 1000
    default_difficulty: str = "normal"
    available_difficulties: List[str] = field(default_factory=lambda: ["easy", "normal", "hard"])
    
    # Scoring info
    min_score: float = 0.0
    max_score: float = float('inf')
    score_interpretation: str = "higher_better"  # "higher_better", "lower_better"
    
    # Modality requirements
    requires_vision: bool = True
    requires_audio: bool = False
    requires_reasoning: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "action_type": self.action_type,
            "num_actions": self.num_actions,
            "action_names": self.action_names,
            "observation_type": self.observation_type,
            "has_image": self.has_image,
            "has_audio": self.has_audio,
            "has_text": self.has_text,
            "image_size": self.image_size,
            "audio_sample_rate": self.audio_sample_rate,
            "max_steps": self.max_steps,
            "default_difficulty": self.default_difficulty,
            "available_difficulties": self.available_difficulties,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "score_interpretation": self.score_interpretation,
            "requires_vision": self.requires_vision,
            "requires_audio": self.requires_audio,
            "requires_reasoning": self.requires_reasoning,
        }


class BaseGameEnv(ABC):
    """
    Abstract base class for game environments.
    
    All game environments should inherit from this class to ensure
    a consistent interface across the benchmark.
    """
    
    def __init__(
        self,
        difficulty: str = "normal",
        render_mode: Optional[str] = None,
        save_data: bool = False,
        save_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the game environment.
        
        Args:
            difficulty: Game difficulty level
            render_mode: Rendering mode ("rgb_array", "human", None)
            save_data: Whether to save episode data
            save_dir: Directory to save data
            **kwargs: Additional environment-specific parameters
        """
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.save_data = save_data
        self.save_dir = save_dir
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        
        # State
        self._is_initialized = False
        self._current_observation = None
    
    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        pass
    
    @abstractmethod
    def step(
        self,
        action: Any
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the environment and release resources."""
        pass
    
    @abstractmethod
    def get_game_info(self) -> GameInfo:
        """Get information about the game."""
        pass
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", else None
        """
        return None
    
    def get_state_for_agent(self) -> Dict[str, Any]:
        """
        Get the current state formatted for an agent.
        
        This method should return a dictionary containing all relevant
        state information that an agent might need to make decisions.
        
        Returns:
            Dictionary with state information
        """
        return {
            "episode": self.episode_count,
            "step": self.step_count,
            "total_reward": self.total_reward,
        }
    
    def get_available_actions(self) -> List[Any]:
        """
        Get list of available actions in current state.
        
        Returns:
            List of valid actions
        """
        info = self.get_game_info()
        if info.action_type == "discrete":
            return list(range(info.num_actions))
        return []
    
    def get_action_description(self, action: Any) -> str:
        """
        Get a human-readable description of an action.
        
        Args:
            action: The action to describe
            
        Returns:
            Description string
        """
        info = self.get_game_info()
        if info.action_type == "discrete" and info.action_names:
            if 0 <= action < len(info.action_names):
                return info.action_names[action]
        return str(action)
    
    def validate_action(self, action: Any) -> bool:
        """
        Validate if an action is legal in current state.
        
        Args:
            action: The action to validate
            
        Returns:
            True if action is valid
        """
        info = self.get_game_info()
        if info.action_type == "discrete":
            return 0 <= action < info.num_actions
        return True
    
    def get_score(self) -> float:
        """
        Get the current game score.
        
        Returns:
            Current score
        """
        return self.total_reward
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            True if game is over
        """
        return False
    
    def is_win(self) -> bool:
        """
        Check if the game was won.
        
        Returns:
            True if game was won
        """
        return False
    
    @property
    def observation(self) -> Optional[Dict[str, Any]]:
        """Get the current observation."""
        return self._current_observation
    
    def set_difficulty(self, difficulty: str):
        """
        Set the game difficulty.
        
        Args:
            difficulty: New difficulty level
        """
        info = self.get_game_info()
        if difficulty in info.available_difficulties:
            self.difficulty = difficulty
        else:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. "
                f"Available: {info.available_difficulties}"
            )


class MultiPlayerGameEnv(BaseGameEnv):
    """
    Base class for multi-player game environments.
    
    Extends BaseGameEnv to support multiple players/agents.
    """
    
    def __init__(
        self,
        num_players: int = 2,
        **kwargs
    ):
        """
        Initialize multi-player environment.
        
        Args:
            num_players: Number of players
            **kwargs: Base environment parameters
        """
        super().__init__(**kwargs)
        self.num_players = num_players
        self.current_player = 0
        self.player_scores = {i: 0.0 for i in range(num_players)}
    
    @abstractmethod
    def step_player(
        self,
        player_id: int,
        action: Any
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute a step for a specific player.
        
        Args:
            player_id: The player taking the action
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass
    
    def get_player_observation(self, player_id: int) -> Dict[str, Any]:
        """
        Get observation for a specific player.
        
        Args:
            player_id: The player ID
            
        Returns:
            Player-specific observation
        """
        return self._current_observation or {}
    
    def get_player_score(self, player_id: int) -> float:
        """
        Get score for a specific player.
        
        Args:
            player_id: The player ID
            
        Returns:
            Player's score
        """
        return self.player_scores.get(player_id, 0.0)
    
    def get_winner(self) -> Optional[int]:
        """
        Get the winning player ID.
        
        Returns:
            Winner's player ID or None if no winner yet
        """
        if not self.is_game_over():
            return None
        
        # Default: highest score wins
        max_score = max(self.player_scores.values())
        winners = [p for p, s in self.player_scores.items() if s == max_score]
        
        return winners[0] if len(winners) == 1 else None


class RecordingWrapper:
    """
    Wrapper to record environment interactions.
    
    Wraps a game environment to record all observations, actions, and rewards.
    """
    
    def __init__(
        self,
        env: BaseGameEnv,
        record_video: bool = False,
        record_audio: bool = False,
        output_dir: str = "recordings"
    ):
        """
        Initialize recording wrapper.
        
        Args:
            env: The environment to wrap
            record_video: Whether to record video
            record_audio: Whether to record audio
            output_dir: Directory for recordings
        """
        self.env = env
        self.record_video = record_video
        self.record_audio = record_audio
        self.output_dir = output_dir
        
        self._frames: List[np.ndarray] = []
        self._audio_chunks: List[np.ndarray] = []
        self._episode_data: List[Dict[str, Any]] = []
        self._current_episode: List[Dict[str, Any]] = []
    
    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset and start recording new episode."""
        # Save previous episode if any
        if self._current_episode:
            self._episode_data.append(self._current_episode)
        
        self._current_episode = []
        self._frames = []
        self._audio_chunks = []
        
        obs, info = self.env.reset(**kwargs)
        
        # Record initial observation
        self._record_step(obs, None, 0, False, False, info, is_reset=True)
        
        return obs, info
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute step and record."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._record_step(obs, action, reward, terminated, truncated, info)
        
        return obs, reward, terminated, truncated, info
    
    def _record_step(
        self,
        obs: Dict[str, Any],
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
        is_reset: bool = False
    ):
        """Record a single step."""
        step_data = {
            "step": len(self._current_episode),
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "is_reset": is_reset,
        }
        
        # Record frame
        if self.record_video and "image" in obs:
            self._frames.append(obs["image"].copy())
            step_data["frame_index"] = len(self._frames) - 1
        
        # Record audio
        if self.record_audio and "audio" in obs:
            self._audio_chunks.append(obs["audio"].copy())
            step_data["audio_index"] = len(self._audio_chunks) - 1
        
        self._current_episode.append(step_data)
    
    def close(self):
        """Close and save recordings."""
        # Save final episode
        if self._current_episode:
            self._episode_data.append(self._current_episode)
        
        # Save recorded data
        self._save_recordings()
        
        self.env.close()
    
    def _save_recordings(self):
        """Save all recordings to disk."""
        import os
        import json
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save episode data
        data_path = os.path.join(self.output_dir, "episodes.json")
        with open(data_path, 'w') as f:
            json.dump(self._episode_data, f, indent=2, default=str)
        
        # Save video if recorded
        if self.record_video and self._frames:
            self._save_video()
        
        # Save audio if recorded
        if self.record_audio and self._audio_chunks:
            self._save_audio()
    
    def _save_video(self):
        """Save recorded frames as video."""
        try:
            from moviepy.editor import ImageSequenceClip
            
            video_path = os.path.join(self.output_dir, "recording.mp4")
            clip = ImageSequenceClip(self._frames, fps=30)
            clip.write_videofile(video_path, codec='libx264', audio=False)
            clip.close()
        except ImportError:
            print("moviepy not available, skipping video save")
    
    def _save_audio(self):
        """Save recorded audio."""
        try:
            import soundfile as sf
            
            audio_path = os.path.join(self.output_dir, "recording.wav")
            audio_data = np.concatenate(self._audio_chunks)
            sf.write(audio_path, audio_data, 16000)
        except ImportError:
            print("soundfile not available, skipping audio save")
    
    # Delegate other methods to wrapped env
    def __getattr__(self, name):
        return getattr(self.env, name)
