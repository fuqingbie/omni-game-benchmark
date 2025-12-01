"""
Video Processor - Unified video processing for different model types.

This module provides processors for converting video data into formats
suitable for different AI models:
- DirectVideoProcessor: For models that accept video files directly
- FrameExtractProcessor: For models that accept extracted frames
- FileUploadProcessor: For models that require file upload
"""

import base64
import io
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Optional imports
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from moviepy.editor import VideoFileClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

from .model_registry import (
    ModelCapability,
    ModelConfig,
    VideoCapability,
)


@dataclass
class ProcessedVideo:
    """Container for processed video data."""
    # Original video path or data
    source: Union[str, bytes, np.ndarray]
    
    # Processed outputs (only one will be populated based on processor type)
    base64_video: Optional[str] = None          # For DIRECT_VIDEO
    frames: Optional[List[np.ndarray]] = None    # For FRAME_EXTRACT
    frame_timestamps: Optional[List[float]] = None
    base64_frames: Optional[List[str]] = None   # Base64 encoded frames
    upload_file_path: Optional[str] = None       # For FILE_UPLOAD
    upload_url: Optional[str] = None             # Uploaded file URL
    
    # Metadata
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    
    def to_api_content(self, capability: ModelCapability) -> List[Dict[str, Any]]:
        """Convert to API content format based on capability."""
        content = []
        
        if capability.video == VideoCapability.DIRECT_VIDEO and self.base64_video:
            content.append({
                "type": "video",
                "video_url": {
                    "url": f"data:video/{capability.video_format};base64,{self.base64_video}"
                }
            })
        
        elif capability.video == VideoCapability.FRAME_EXTRACT and self.base64_frames:
            for i, frame_b64 in enumerate(self.base64_frames):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{capability.image_format};base64,{frame_b64}"
                    }
                })
        
        elif capability.video == VideoCapability.FILE_UPLOAD and self.upload_url:
            content.append({
                "type": "file_url",
                "file_url": {
                    "url": self.upload_url
                }
            })
        
        return content


class VideoProcessor(ABC):
    """Abstract base class for video processors."""
    
    @abstractmethod
    def process(
        self,
        source: Union[str, bytes, np.ndarray],
        **kwargs
    ) -> ProcessedVideo:
        """
        Process video from various input formats.
        
        Args:
            source: Video source - can be:
                   - str: file path to video
                   - bytes: raw video data
                   - np.ndarray: video frames array (T, H, W, C)
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedVideo with the appropriate output format
        """
        pass
    
    @staticmethod
    def _load_video_clip(source: Union[str, bytes]) -> "VideoFileClip":
        """Load video as MoviePy clip."""
        if not HAS_MOVIEPY:
            raise ImportError("moviepy is required for video processing")
        
        if isinstance(source, bytes):
            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(source)
                temp_path = f.name
            return VideoFileClip(temp_path)
        else:
            return VideoFileClip(source)
    
    @staticmethod
    def _frame_to_base64(frame: np.ndarray, format: str = "jpeg") -> str:
        """Convert frame to base64 string."""
        if not HAS_PIL:
            raise ImportError("PIL is required for image processing")
        
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format=format.upper())
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


class DirectVideoProcessor(VideoProcessor):
    """
    Processor for models that accept video files directly.
    Converts video to base64 encoded string.
    """
    
    def __init__(self, video_format: str = "mp4"):
        self.video_format = video_format
    
    def process(
        self,
        source: Union[str, bytes, np.ndarray],
        **kwargs
    ) -> ProcessedVideo:
        """Process video for direct video input models."""
        result = ProcessedVideo(source=source)
        
        if isinstance(source, str):
            # Read file and encode
            with open(source, 'rb') as f:
                video_data = f.read()
            result.base64_video = base64.b64encode(video_data).decode('utf-8')
            
            # Get metadata
            if HAS_MOVIEPY:
                clip = VideoFileClip(source)
                result.duration = clip.duration
                result.fps = clip.fps
                result.width = clip.w
                result.height = clip.h
                result.frame_count = int(clip.fps * clip.duration)
                clip.close()
        
        elif isinstance(source, bytes):
            result.base64_video = base64.b64encode(source).decode('utf-8')
        
        elif isinstance(source, np.ndarray):
            # Convert frames to video file
            if HAS_MOVIEPY:
                from moviepy.editor import ImageSequenceClip
                
                fps = kwargs.get('fps', 30)
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                    temp_path = f.name
                
                clip = ImageSequenceClip(list(source), fps=fps)
                clip.write_videofile(temp_path, codec='libx264', audio=False, verbose=False, logger=None)
                
                with open(temp_path, 'rb') as f:
                    video_data = f.read()
                result.base64_video = base64.b64encode(video_data).decode('utf-8')
                
                result.duration = clip.duration
                result.fps = fps
                result.height, result.width = source.shape[1:3]
                result.frame_count = len(source)
                
                clip.close()
                os.unlink(temp_path)
        
        return result


class FrameExtractProcessor(VideoProcessor):
    """
    Processor for models that accept extracted frames.
    Extracts frames at specified intervals and converts to base64.
    """
    
    def __init__(
        self,
        frame_interval: float = 0.5,
        max_frames: int = 10,
        image_format: str = "jpeg",
        resize: Optional[tuple] = None
    ):
        """
        Initialize frame extractor.
        
        Args:
            frame_interval: Seconds between frames
            max_frames: Maximum number of frames to extract
            image_format: Output image format (jpeg, png)
            resize: Optional (width, height) to resize frames
        """
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.image_format = image_format
        self.resize = resize
    
    def process(
        self,
        source: Union[str, bytes, np.ndarray],
        **kwargs
    ) -> ProcessedVideo:
        """Process video by extracting frames."""
        result = ProcessedVideo(source=source)
        
        frame_interval = kwargs.get('frame_interval', self.frame_interval)
        max_frames = kwargs.get('max_frames', self.max_frames)
        resize = kwargs.get('resize', self.resize)
        
        if isinstance(source, np.ndarray):
            # Source is already frames array
            frames = list(source)
            fps = kwargs.get('fps', 30)
            duration = len(frames) / fps
        else:
            # Load video and extract frames
            if not HAS_MOVIEPY:
                raise ImportError("moviepy is required for video processing")
            
            clip = self._load_video_clip(source)
            duration = clip.duration
            fps = clip.fps
            
            # Calculate frame times
            num_frames = min(int(duration / frame_interval) + 1, max_frames)
            frame_times = [i * frame_interval for i in range(num_frames)]
            frame_times = [t for t in frame_times if t < duration]
            
            # Extract frames
            frames = []
            for t in frame_times:
                frame = clip.get_frame(t)
                frames.append(frame)
            
            result.frame_timestamps = frame_times
            result.width = clip.w
            result.height = clip.h
            clip.close()
        
        # Resize frames if needed
        if resize and HAS_PIL:
            resized_frames = []
            for frame in frames:
                img = Image.fromarray(frame.astype(np.uint8))
                img = img.resize(resize, Image.Resampling.LANCZOS)
                resized_frames.append(np.array(img))
            frames = resized_frames
            result.width, result.height = resize
        
        # Convert to base64
        base64_frames = []
        for frame in frames:
            b64 = self._frame_to_base64(frame, self.image_format)
            base64_frames.append(b64)
        
        result.frames = frames
        result.base64_frames = base64_frames
        result.duration = duration
        result.fps = fps
        result.frame_count = len(frames)
        
        return result


class FileUploadProcessor(VideoProcessor):
    """
    Processor for models that require file upload.
    Prepares video file and handles upload via custom upload function.
    """
    
    def __init__(
        self,
        upload_func: Optional[callable] = None,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize file upload processor.
        
        Args:
            upload_func: Optional function to upload file and return URL
                        Signature: (file_path: str) -> str (url)
            temp_dir: Directory for temporary files
        """
        self.upload_func = upload_func
        self.temp_dir = temp_dir or tempfile.gettempdir()
    
    def process(
        self,
        source: Union[str, bytes, np.ndarray],
        **kwargs
    ) -> ProcessedVideo:
        """Process video for file upload models."""
        result = ProcessedVideo(source=source)
        
        # Get file path
        if isinstance(source, str):
            file_path = source
        elif isinstance(source, bytes):
            # Write to temp file
            with tempfile.NamedTemporaryFile(
                suffix='.mp4',
                dir=self.temp_dir,
                delete=False
            ) as f:
                f.write(source)
                file_path = f.name
        elif isinstance(source, np.ndarray):
            # Convert frames to video
            if HAS_MOVIEPY:
                from moviepy.editor import ImageSequenceClip
                
                fps = kwargs.get('fps', 30)
                with tempfile.NamedTemporaryFile(
                    suffix='.mp4',
                    dir=self.temp_dir,
                    delete=False
                ) as f:
                    file_path = f.name
                
                clip = ImageSequenceClip(list(source), fps=fps)
                clip.write_videofile(
                    file_path,
                    codec='libx264',
                    audio=False,
                    verbose=False,
                    logger=None
                )
                clip.close()
            else:
                raise ImportError("moviepy is required for video processing")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        result.upload_file_path = file_path
        
        # Get metadata
        if HAS_MOVIEPY:
            clip = VideoFileClip(file_path)
            result.duration = clip.duration
            result.fps = clip.fps
            result.width = clip.w
            result.height = clip.h
            result.frame_count = int(clip.fps * clip.duration)
            clip.close()
        
        # Upload if function provided
        upload_func = kwargs.get('upload_func', self.upload_func)
        if upload_func:
            result.upload_url = upload_func(file_path)
        
        return result
    
    def set_upload_function(self, func: callable):
        """Set the upload function."""
        self.upload_func = func


def get_processor_for_model(
    model_config: ModelConfig
) -> VideoProcessor:
    """
    Get the appropriate video processor for a model.
    
    Args:
        model_config: Model configuration with capabilities
        
    Returns:
        VideoProcessor instance suitable for the model
    """
    cap = model_config.capability
    
    if cap.video == VideoCapability.DIRECT_VIDEO:
        return DirectVideoProcessor(video_format=cap.video_format)
    
    elif cap.video == VideoCapability.FRAME_EXTRACT:
        return FrameExtractProcessor(
            frame_interval=cap.frame_interval,
            max_frames=cap.max_frames,
            image_format=cap.image_format,
        )
    
    elif cap.video == VideoCapability.FILE_UPLOAD:
        return FileUploadProcessor()
    
    else:
        # Return a no-op processor for NO_VIDEO
        return DirectVideoProcessor()


def process_video_for_model(
    video_source: Union[str, bytes, np.ndarray],
    model_config: ModelConfig,
    **kwargs
) -> ProcessedVideo:
    """
    Convenience function to process video for a specific model.
    
    Args:
        video_source: Video source (path, bytes, or frames array)
        model_config: Model configuration
        **kwargs: Additional processing parameters
        
    Returns:
        ProcessedVideo ready for the model
    """
    processor = get_processor_for_model(model_config)
    return processor.process(video_source, **kwargs)
