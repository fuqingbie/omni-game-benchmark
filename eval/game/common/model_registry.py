"""
Model Registry - Register and manage AI model capabilities.

This module provides a registry for AI models with their capabilities,
including video processing modes, audio support, and API configurations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
import json
import os


class VideoCapability(Enum):
    """Video processing capabilities for models."""
    DIRECT_VIDEO = auto()      # Model accepts video files directly (e.g., base64 MP4)
    FRAME_EXTRACT = auto()     # Model accepts extracted frames as images
    FILE_UPLOAD = auto()        # Model requires file upload (e.g., Baichuan multipart)
    NO_VIDEO = auto()          # Model does not support video input


class AudioCapability(Enum):
    """Audio processing capabilities for models."""
    DIRECT_AUDIO = auto()      # Model accepts audio files directly
    AUDIO_FEATURES = auto()    # Model accepts extracted audio features
    NO_AUDIO = auto()          # Model does not support audio input


class ImageCapability(Enum):
    """Image processing capabilities for models."""
    BASE64_IMAGE = auto()      # Model accepts base64 encoded images
    URL_IMAGE = auto()         # Model accepts image URLs
    NO_IMAGE = auto()          # Model does not support image input


@dataclass
class ModelCapability:
    """Describes the capabilities of an AI model."""
    video: VideoCapability = VideoCapability.NO_VIDEO
    audio: AudioCapability = AudioCapability.NO_AUDIO
    image: ImageCapability = ImageCapability.BASE64_IMAGE
    
    # Video processing parameters
    frame_interval: float = 0.5  # Seconds between frames for FRAME_EXTRACT
    max_frames: int = 10         # Maximum number of frames to extract
    video_format: str = "mp4"    # Preferred video format
    
    # Audio processing parameters
    audio_sample_rate: int = 16000
    audio_format: str = "wav"
    
    # Image processing parameters
    image_format: str = "jpeg"
    max_image_size: tuple = (1024, 1024)
    
    # API parameters
    max_tokens: int = 4096
    supports_streaming: bool = False
    supports_function_calling: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary."""
        return {
            "video": self.video.name,
            "audio": self.audio.name,
            "image": self.image.name,
            "frame_interval": self.frame_interval,
            "max_frames": self.max_frames,
            "video_format": self.video_format,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_format": self.audio_format,
            "image_format": self.image_format,
            "max_image_size": self.max_image_size,
            "max_tokens": self.max_tokens,
            "supports_streaming": self.supports_streaming,
            "supports_function_calling": self.supports_function_calling,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCapability":
        """Create capability from dictionary."""
        return cls(
            video=VideoCapability[data.get("video", "NO_VIDEO")],
            audio=AudioCapability[data.get("audio", "NO_AUDIO")],
            image=ImageCapability[data.get("image", "BASE64_IMAGE")],
            frame_interval=data.get("frame_interval", 0.5),
            max_frames=data.get("max_frames", 10),
            video_format=data.get("video_format", "mp4"),
            audio_sample_rate=data.get("audio_sample_rate", 16000),
            audio_format=data.get("audio_format", "wav"),
            image_format=data.get("image_format", "jpeg"),
            max_image_size=tuple(data.get("max_image_size", [1024, 1024])),
            max_tokens=data.get("max_tokens", 4096),
            supports_streaming=data.get("supports_streaming", False),
            supports_function_calling=data.get("supports_function_calling", False),
        )


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    name: str                          # Model identifier
    api_base: str                      # API endpoint base URL
    api_key: str = ""                  # API key (can be loaded from env)
    model_id: str = ""                 # Model ID for API calls
    capability: ModelCapability = field(default_factory=ModelCapability)
    
    # Optional custom parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Load API key from environment if not provided
        if not self.api_key:
            env_key = f"{self.name.upper().replace('-', '_')}_API_KEY"
            self.api_key = os.environ.get(env_key, "")
        
        # Default model_id to name if not provided
        if not self.model_id:
            self.model_id = self.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (without sensitive data)."""
        return {
            "name": self.name,
            "api_base": self.api_base,
            "model_id": self.model_id,
            "capability": self.capability.to_dict(),
            "extra_params": self.extra_params,
            "has_api_key": bool(self.api_key),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        capability = ModelCapability.from_dict(data.get("capability", {}))
        return cls(
            name=data["name"],
            api_base=data["api_base"],
            api_key=data.get("api_key", ""),
            model_id=data.get("model_id", ""),
            capability=capability,
            extra_params=data.get("extra_params", {}),
        )


# Pre-defined capabilities for common model families
OPENAI_CAPABILITY = ModelCapability(
    video=VideoCapability.DIRECT_VIDEO,
    audio=AudioCapability.NO_AUDIO,
    image=ImageCapability.BASE64_IMAGE,
    max_tokens=4096,
    supports_streaming=True,
    supports_function_calling=True,
)

OPENAI_FRAME_CAPABILITY = ModelCapability(
    video=VideoCapability.FRAME_EXTRACT,
    audio=AudioCapability.NO_AUDIO,
    image=ImageCapability.BASE64_IMAGE,
    frame_interval=0.5,
    max_frames=10,
    max_tokens=4096,
)

BAICHUAN_CAPABILITY = ModelCapability(
    video=VideoCapability.FILE_UPLOAD,
    audio=AudioCapability.NO_AUDIO,
    image=ImageCapability.BASE64_IMAGE,
    max_tokens=4096,
)

QWEN_CAPABILITY = ModelCapability(
    video=VideoCapability.FRAME_EXTRACT,
    audio=AudioCapability.DIRECT_AUDIO,
    image=ImageCapability.BASE64_IMAGE,
    frame_interval=0.5,
    max_frames=8,
    max_tokens=8192,
)

GEMINI_CAPABILITY = ModelCapability(
    video=VideoCapability.DIRECT_VIDEO,
    audio=AudioCapability.DIRECT_AUDIO,
    image=ImageCapability.BASE64_IMAGE,
    max_tokens=8192,
    supports_streaming=True,
)


class ModelRegistry:
    """Registry for AI models and their capabilities."""
    
    def __init__(self):
        """Initialize the registry."""
        self._models: Dict[str, ModelConfig] = {}
        self._capability_presets: Dict[str, ModelCapability] = {
            "openai": OPENAI_CAPABILITY,
            "openai-frames": OPENAI_FRAME_CAPABILITY,
            "baichuan": BAICHUAN_CAPABILITY,
            "qwen": QWEN_CAPABILITY,
            "gemini": GEMINI_CAPABILITY,
        }
    
    def register(
        self,
        name: str,
        api_base: str,
        api_key: str = "",
        model_id: str = "",
        capability: Optional[ModelCapability] = None,
        capability_preset: Optional[str] = None,
        **extra_params
    ) -> ModelConfig:
        """
        Register a model with its configuration.
        
        Args:
            name: Unique model identifier
            api_base: API endpoint base URL
            api_key: API key (or set via environment variable)
            model_id: Model ID for API calls
            capability: Model capability configuration
            capability_preset: Use a pre-defined capability preset
            **extra_params: Additional model-specific parameters
            
        Returns:
            The registered ModelConfig
        """
        # Determine capability
        if capability is not None:
            cap = capability
        elif capability_preset and capability_preset in self._capability_presets:
            cap = self._capability_presets[capability_preset]
        else:
            cap = ModelCapability()
        
        config = ModelConfig(
            name=name,
            api_base=api_base,
            api_key=api_key,
            model_id=model_id,
            capability=cap,
            extra_params=extra_params,
        )
        
        self._models[name] = config
        return config
    
    def get(self, name: str) -> Optional[ModelConfig]:
        """Get a registered model by name."""
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def list_models_with_capability(
        self,
        video: Optional[VideoCapability] = None,
        audio: Optional[AudioCapability] = None,
        image: Optional[ImageCapability] = None,
    ) -> List[str]:
        """List models with specific capabilities."""
        result = []
        for name, config in self._models.items():
            cap = config.capability
            if video and cap.video != video:
                continue
            if audio and cap.audio != audio:
                continue
            if image and cap.image != image:
                continue
            result.append(name)
        return result
    
    def add_capability_preset(self, name: str, capability: ModelCapability):
        """Add a custom capability preset."""
        self._capability_presets[name] = capability
    
    def load_from_file(self, filepath: str):
        """Load model configurations from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for model_data in data.get("models", []):
            self.register(
                name=model_data["name"],
                api_base=model_data["api_base"],
                api_key=model_data.get("api_key", ""),
                model_id=model_data.get("model_id", ""),
                capability_preset=model_data.get("capability_preset"),
                **model_data.get("extra_params", {})
            )
    
    def save_to_file(self, filepath: str):
        """Save model configurations to a JSON file."""
        data = {
            "models": [
                config.to_dict() for config in self._models.values()
            ]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary."""
        return {
            "models": {name: config.to_dict() for name, config in self._models.items()},
            "capability_presets": {
                name: cap.to_dict() for name, cap in self._capability_presets.items()
            },
        }


# Global default registry
_default_registry: Optional[ModelRegistry] = None


def get_default_registry() -> ModelRegistry:
    """Get or create the default model registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry


def register_model(
    name: str,
    api_base: str,
    api_key: str = "",
    model_id: str = "",
    capability: Optional[ModelCapability] = None,
    capability_preset: Optional[str] = None,
    **extra_params
) -> ModelConfig:
    """
    Convenience function to register a model in the default registry.
    """
    return get_default_registry().register(
        name=name,
        api_base=api_base,
        api_key=api_key,
        model_id=model_id,
        capability=capability,
        capability_preset=capability_preset,
        **extra_params
    )
