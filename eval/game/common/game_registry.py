"""
Game Registry - Game registration and CLI tools.

This module provides a registry for games and CLI tools for managing
game registrations and running evaluations.
"""

import os
import json
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base_env import BaseGameEnv, GameInfo


@dataclass
class GameEntry:
    """Entry for a registered game."""
    name: str
    env_class: Optional[Type[BaseGameEnv]] = None
    env_factory: Optional[Callable[..., BaseGameEnv]] = None
    
    # Module path for dynamic loading
    module_path: Optional[str] = None
    class_name: Optional[str] = None
    
    # Game metadata
    info: Optional[GameInfo] = None
    description: str = ""
    version: str = "1.0"
    
    # Paths
    game_dir: Optional[str] = None
    config_path: Optional[str] = None
    
    # Default configuration
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    # Tags for filtering
    tags: List[str] = field(default_factory=list)
    
    # Evaluation settings
    default_episodes: int = 10
    default_max_steps: int = 1000
    
    def create_env(self, **kwargs) -> BaseGameEnv:
        """
        Create an instance of the game environment.
        
        Args:
            **kwargs: Environment configuration parameters
            
        Returns:
            Instance of the game environment
        """
        # Merge default config with provided kwargs
        config = {**self.default_config, **kwargs}
        
        if self.env_class is not None:
            return self.env_class(**config)
        
        if self.env_factory is not None:
            return self.env_factory(**config)
        
        if self.module_path and self.class_name:
            # Dynamic loading
            module = importlib.import_module(self.module_path)
            env_class = getattr(module, self.class_name)
            return env_class(**config)
        
        raise ValueError(f"No environment class or factory registered for game '{self.name}'")
    
    def get_info(self) -> GameInfo:
        """Get game information."""
        if self.info:
            return self.info
        
        # Try to get from environment
        try:
            env = self.create_env()
            info = env.get_game_info()
            env.close()
            return info
        except Exception:
            pass
        
        # Return basic info
        return GameInfo(
            name=self.name,
            version=self.version,
            description=self.description,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "module_path": self.module_path,
            "class_name": self.class_name,
            "description": self.description,
            "version": self.version,
            "game_dir": self.game_dir,
            "config_path": self.config_path,
            "default_config": self.default_config,
            "tags": self.tags,
            "default_episodes": self.default_episodes,
            "default_max_steps": self.default_max_steps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameEntry":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            module_path=data.get("module_path"),
            class_name=data.get("class_name"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            game_dir=data.get("game_dir"),
            config_path=data.get("config_path"),
            default_config=data.get("default_config", {}),
            tags=data.get("tags", []),
            default_episodes=data.get("default_episodes", 10),
            default_max_steps=data.get("default_max_steps", 1000),
        )


class GameRegistry:
    """Registry for game environments."""
    
    def __init__(self):
        """Initialize the registry."""
        self._games: Dict[str, GameEntry] = {}
    
    def register(
        self,
        name: str,
        env_class: Optional[Type[BaseGameEnv]] = None,
        env_factory: Optional[Callable[..., BaseGameEnv]] = None,
        module_path: Optional[str] = None,
        class_name: Optional[str] = None,
        info: Optional[GameInfo] = None,
        description: str = "",
        version: str = "1.0",
        game_dir: Optional[str] = None,
        default_config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        default_episodes: int = 10,
        default_max_steps: int = 1000,
    ) -> GameEntry:
        """
        Register a game.
        
        Args:
            name: Unique game identifier
            env_class: Environment class
            env_factory: Factory function to create environment
            module_path: Module path for dynamic loading
            class_name: Class name for dynamic loading
            info: Game information
            description: Game description
            version: Game version
            game_dir: Directory containing game files
            default_config: Default configuration
            tags: Tags for filtering
            default_episodes: Default number of episodes
            default_max_steps: Default max steps per episode
            
        Returns:
            The registered GameEntry
        """
        entry = GameEntry(
            name=name,
            env_class=env_class,
            env_factory=env_factory,
            module_path=module_path,
            class_name=class_name,
            info=info,
            description=description,
            version=version,
            game_dir=game_dir,
            default_config=default_config or {},
            tags=tags or [],
            default_episodes=default_episodes,
            default_max_steps=default_max_steps,
        )
        
        self._games[name] = entry
        return entry
    
    def get(self, name: str) -> Optional[GameEntry]:
        """Get a registered game by name."""
        return self._games.get(name)
    
    def list_games(self) -> List[str]:
        """List all registered game names."""
        return list(self._games.keys())
    
    def list_games_with_tags(self, tags: List[str]) -> List[str]:
        """List games with specific tags."""
        result = []
        for name, entry in self._games.items():
            if any(tag in entry.tags for tag in tags):
                result.append(name)
        return result
    
    def create_env(self, name: str, **kwargs) -> BaseGameEnv:
        """
        Create an environment for a game.
        
        Args:
            name: Game name
            **kwargs: Environment parameters
            
        Returns:
            Game environment instance
        """
        entry = self.get(name)
        if entry is None:
            raise ValueError(f"Game '{name}' not registered")
        return entry.create_env(**kwargs)
    
    def load_from_file(self, filepath: str):
        """Load game registrations from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for game_data in data.get("games", []):
            entry = GameEntry.from_dict(game_data)
            self._games[entry.name] = entry
    
    def save_to_file(self, filepath: str):
        """Save game registrations to a JSON file."""
        data = {
            "games": [entry.to_dict() for entry in self._games.values()]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary."""
        return {
            "games": {name: entry.to_dict() for name, entry in self._games.items()}
        }
    
    def discover_games(self, base_dir: str):
        """
        Discover and register games from a directory.
        
        Looks for game directories with a game_config.json or env.py file.
        
        Args:
            base_dir: Base directory to search
        """
        base_path = Path(base_dir)
        
        for item in base_path.iterdir():
            if not item.is_dir():
                continue
            
            # Skip common non-game directories
            if item.name.startswith(('_', '.', 'common', 'assets')):
                continue
            
            # Check for game config
            config_path = item / "game_config.json"
            if config_path.exists():
                self._load_game_from_config(config_path)
                continue
            
            # Check for environment file
            env_path = item / "env.py"
            if env_path.exists():
                self._register_from_env_file(item.name, item, env_path)
                continue
            
            # Check for gym wrapper
            gym_path = item / "gym_wrapper.py"
            if gym_path.exists():
                self._register_from_gym_wrapper(item.name, item, gym_path)
    
    def _load_game_from_config(self, config_path: Path):
        """Load game from config file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        entry = GameEntry.from_dict(config)
        entry.config_path = str(config_path)
        entry.game_dir = str(config_path.parent)
        
        self._games[entry.name] = entry
    
    def _register_from_env_file(self, name: str, game_dir: Path, env_path: Path):
        """Register game from env.py file."""
        # Construct module path
        rel_path = env_path.relative_to(game_dir.parent)
        module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
        
        self.register(
            name=name,
            module_path=module_path,
            class_name="GameEnv",  # Common convention
            game_dir=str(game_dir),
            description=f"Game from {game_dir.name}",
        )
    
    def _register_from_gym_wrapper(self, name: str, game_dir: Path, wrapper_path: Path):
        """Register game from gym_wrapper.py file."""
        rel_path = wrapper_path.relative_to(game_dir.parent)
        module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
        
        self.register(
            name=name,
            module_path=module_path,
            class_name="GameGymWrapper",  # Common convention
            game_dir=str(game_dir),
            description=f"Game from {game_dir.name}",
        )


# Global default registry
_default_registry: Optional[GameRegistry] = None


def get_game_registry() -> GameRegistry:
    """Get or create the default game registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = GameRegistry()
    return _default_registry


def register_game(
    name: str,
    env_class: Optional[Type[BaseGameEnv]] = None,
    env_factory: Optional[Callable[..., BaseGameEnv]] = None,
    **kwargs
) -> GameEntry:
    """
    Convenience function to register a game in the default registry.
    """
    return get_game_registry().register(
        name=name,
        env_class=env_class,
        env_factory=env_factory,
        **kwargs
    )


# Pre-register known games
def register_builtin_games():
    """Register built-in games from the benchmark."""
    registry = get_game_registry()
    
    # Blasting Showdown (Bomberman)
    registry.register(
        name="blasting_showdown",
        module_path="game.Blasting_Showdown.bomberman_gym",
        class_name="BombermanGymEnv",
        description="Multi-player bomberman game with AI opponents",
        tags=["multiplayer", "strategy", "action"],
        default_episodes=5,
        default_max_steps=300,
    )
    
    # Myriad Echoes (Rhythm Memory)
    registry.register(
        name="myriad_echoes",
        module_path="game.Myriad_Echoes.rhythm_memory_gym_env",
        class_name="RhythmMemoryEnv",
        description="Audio-visual rhythm memory game",
        tags=["audio", "memory", "rhythm"],
        default_episodes=10,
        default_max_steps=100,
    )
    
    # Phantom Soldiers in the Fog
    registry.register(
        name="phantom_soldiers",
        module_path="game.Phantom_Soldiers_in_the_Fog.gym_wrapper",
        class_name="PhantomSoldiersEnv",
        description="Fog of war tactical game with audio cues",
        tags=["audio", "strategy", "fog_of_war"],
        default_episodes=10,
        default_max_steps=200,
    )
    
    # The Alchemist's Melody
    registry.register(
        name="alchemist_melody",
        module_path="game.The_Alchemist-s_Melody.sound_alchemist_env",
        class_name="SoundAlchemistEnv",
        description="Musical puzzle game with color-sound matching",
        tags=["audio", "puzzle", "music"],
        default_episodes=10,
        default_max_steps=50,
    )
    
    # Whispered Pathfinding (Maze)
    registry.register(
        name="whispered_pathfinding",
        module_path="game.Whispered_Pathfinding.maze_gym_env",
        class_name="MazeGymEnv",
        description="3D maze navigation with audio guidance",
        tags=["audio", "navigation", "3d"],
        default_episodes=5,
        default_max_steps=100,
    )


# CLI implementation
def cli_main():
    """CLI entry point for game management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Game Registry CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List registered games")
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show details")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show game information")
    info_parser.add_argument("game", help="Game name")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new game")
    register_parser.add_argument("name", help="Game name")
    register_parser.add_argument("--module", required=True, help="Module path")
    register_parser.add_argument("--class", dest="class_name", required=True, help="Class name")
    register_parser.add_argument("--description", default="", help="Game description")
    register_parser.add_argument("--tags", nargs="+", default=[], help="Game tags")
    
    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover games in directory")
    discover_parser.add_argument("directory", help="Directory to search")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export registry to file")
    export_parser.add_argument("output", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize registry with built-in games
    register_builtin_games()
    registry = get_game_registry()
    
    if args.command == "list":
        if args.tags:
            games = registry.list_games_with_tags(args.tags)
        else:
            games = registry.list_games()
        
        print(f"Registered games ({len(games)}):")
        for name in games:
            entry = registry.get(name)
            if args.verbose and entry:
                print(f"  {name}: {entry.description}")
                if entry.tags:
                    print(f"    Tags: {', '.join(entry.tags)}")
            else:
                print(f"  - {name}")
    
    elif args.command == "info":
        entry = registry.get(args.game)
        if entry:
            print(f"Game: {entry.name}")
            print(f"Version: {entry.version}")
            print(f"Description: {entry.description}")
            print(f"Module: {entry.module_path}.{entry.class_name}")
            print(f"Tags: {', '.join(entry.tags) if entry.tags else 'None'}")
            print(f"Default Episodes: {entry.default_episodes}")
            print(f"Default Max Steps: {entry.default_max_steps}")
            
            if entry.default_config:
                print(f"Default Config: {json.dumps(entry.default_config, indent=2)}")
        else:
            print(f"Game '{args.game}' not found")
    
    elif args.command == "register":
        registry.register(
            name=args.name,
            module_path=args.module,
            class_name=args.class_name,
            description=args.description,
            tags=args.tags,
        )
        print(f"Registered game: {args.name}")
    
    elif args.command == "discover":
        registry.discover_games(args.directory)
        print(f"Discovered games: {registry.list_games()}")
    
    elif args.command == "export":
        registry.save_to_file(args.output)
        print(f"Exported registry to: {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    cli_main()
