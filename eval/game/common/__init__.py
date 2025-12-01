"""
Common modules for unified game benchmark framework.

This package provides:
- Model Registry: Register and manage AI model capabilities
- Video Processor: Unified video processing for different model types
- Result Schema: Standardized result format across all games
- Base Environment: Common interface for game environments
- Game Registry: Game registration and CLI tools
- Runner: Unified evaluation runner
- Statistics: Unified statistics analysis
"""

from .model_registry import (
    ModelCapability,
    ModelConfig,
    ModelRegistry,
    VideoCapability,
    AudioCapability,
    ImageCapability,
    get_default_registry,
)

from .video_processor import (
    VideoProcessor,
    DirectVideoProcessor,
    FrameExtractProcessor,
    FileUploadProcessor,
    get_processor_for_model,
)

from .result_schema import (
    EpisodeResult,
    GameResult,
    BenchmarkResult,
    ResultWriter,
)

from .base_env import (
    BaseGameEnv,
    GameInfo,
)

from .game_registry import (
    GameEntry,
    GameRegistry,
    get_game_registry,
    register_game,
)

from .runner import (
    BenchmarkRunner,
    RunConfig,
)

from .statistics import (
    StatisticsAnalyzer,
    compute_game_metrics,
)

__all__ = [
    # Model Registry
    "ModelCapability",
    "ModelConfig", 
    "ModelRegistry",
    "VideoCapability",
    "AudioCapability",
    "ImageCapability",
    "get_default_registry",
    # Video Processor
    "VideoProcessor",
    "DirectVideoProcessor",
    "FrameExtractProcessor",
    "FileUploadProcessor",
    "get_processor_for_model",
    # Result Schema
    "EpisodeResult",
    "GameResult",
    "BenchmarkResult",
    "ResultWriter",
    # Base Environment
    "BaseGameEnv",
    "GameInfo",
    # Game Registry
    "GameEntry",
    "GameRegistry",
    "get_game_registry",
    "register_game",
    # Runner
    "BenchmarkRunner",
    "RunConfig",
    # Statistics
    "StatisticsAnalyzer",
    "compute_game_metrics",
]

__version__ = "0.1.0"
