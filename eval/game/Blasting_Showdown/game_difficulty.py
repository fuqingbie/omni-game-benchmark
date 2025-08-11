"""
Bomberman Game Difficulty Configuration Module
Defines game parameters for different difficulty levels
"""
from enum import Enum
from dataclasses import dataclass


class DifficultyLevel(Enum):
    """Game Difficulty Levels"""
    EASY = 'easy'       # Easy
    NORMAL = 'normal'   # Normal
    HARD = 'hard'       # Hard


@dataclass
class DifficultyConfig:
    """Difficulty Configuration Dataclass"""
    name: str           # Difficulty name
    grid_width: int     # Grid width
    grid_height: int    # Grid height
    soft_wall_chance: float  # Soft wall generation probability 
    clear_radius: int   # Clear radius around player's initial position
    player_speed: int   # Player base speed
    max_move_distance: int  # Maximum move distance


# Predefined difficulty configurations
DIFFICULTY_CONFIGS = {
    DifficultyLevel.EASY: DifficultyConfig(
        name="Easy",
        grid_width=9,           # Smaller map: 9x7
        grid_height=7,
        soft_wall_chance=0.3,   # Reduced probability of generating soft walls (obstacles)
        clear_radius=2,         # Larger initial clear area
        player_speed=2,         # Faster base speed
        max_move_distance=6,    # Greater move distance
    ),
    DifficultyLevel.NORMAL: DifficultyConfig(
        name="Normal", 
        grid_width=13,
        grid_height=11,
        soft_wall_chance=0.6,
        clear_radius=1,
        player_speed=1,
        max_move_distance=5,
    ),
    DifficultyLevel.HARD: DifficultyConfig(
        name="Hard",
        grid_width=15,
        grid_height=13,
        soft_wall_chance=0.7,
        clear_radius=1,
        player_speed=1,
        max_move_distance=4,
    )
}


def get_difficulty_config(difficulty: DifficultyLevel = DifficultyLevel.NORMAL) -> DifficultyConfig:
    """Get the configuration for a specified difficulty"""
    return DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS[DifficultyLevel.NORMAL])