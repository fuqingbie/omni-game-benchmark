"""
Simplified Cooperative Command Game Package

A streamlined implementation with normalized scores built into environment feedback
and simplified logging. This package provides:

- CoopCommandEnv: Main game environment with built-in normalized scoring
- CoopCommandGymEnv: OpenAI Gym wrapper for RL integration
- Core game logic, team management, audio, and visualization systems

Key features:
- Normalized scores (0-100) included in environment feedback
- Round-based gameplay with automatic optimal round calculation
- Multi-difficulty support (normal, medium, hard)
- Fixed seed support for reproducible testing
- Audio and visual observation support
- Simplified logging without complex evaluation infrastructure
"""

from .env import CoopCommandEnv, GameConfig, GameDifficulty, GameState
from .gym_wrapper import CoopCommandGymEnv
from .team_member import TeamMember, TeamMemberManager, Position, VoiceAsset
from .game_logic import GameLogic
from .audio_system import AudioSystem
from .visualization import GameVisualization

__version__ = "1.0.0"

__all__ = [
    # Main environment classes
    'CoopCommandEnv',
    'CoopCommandGymEnv',
    
    # Configuration and enums
    'GameConfig',
    'GameDifficulty',
    'GameState',
    
    # Core game components
    'TeamMember',
    'TeamMemberManager',
    'GameLogic',
    'AudioSystem',
    'GameVisualization',
    
    # Utility classes
    'Position',
    'VoiceAsset'
]

# Quick start function for testing
def create_simple_game(difficulty="normal", seed_index=0, max_rounds=None, deterministic_commands=True):
    """Create a simple game environment for quick testing
    
    Args:
        difficulty: Game difficulty ("normal", "medium", "hard")
        seed_index: Fixed seed index (0-9) for reproducible games
        max_rounds: Maximum rounds (auto-calculated if None)
        deterministic_commands: If True, all valid commands succeed (default: deterministic)
    
    Returns:
        CoopCommandEnv: Configured game environment
    """
    config = GameConfig(
        difficulty=GameDifficulty(difficulty),
        seed_index=seed_index,
        max_rounds=max_rounds,
        enable_audio=True,
        deterministic_commands=deterministic_commands
    )
    return CoopCommandEnv(config=config, enable_assets=True)

def create_gym_env(difficulty="normal", seed_index=0, enable_visual=False, enable_audio=True):
    """Create a Gym environment for RL experiments
    
    Args:
        difficulty: Game difficulty ("normal", "medium", "hard")
        seed_index: Fixed seed index (0-9) for reproducible games
        enable_visual: Enable visual observations
        enable_audio: Enable audio observations
    
    Returns:
        CoopCommandGymEnv: Gym-compatible environment
    """
    return CoopCommandGymEnv(
        difficulty=difficulty,
        seed_index=seed_index,
        enable_visual=enable_visual,
        enable_audio=enable_audio
    ) 