#!/usr/bin/env python3
"""
Enhanced Video Recorder with Audio Integration

Provides improved video recording capabilities with better audio-video synchronization
and enhanced audio file matching for the cooperative command game.
"""

import os
import json
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Check for required libraries
try:
    from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ MoviePy not available. Enhanced video features will be limited.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedVideoRecorder:
    """Enhanced video recorder with better audio integration capabilities."""
    
    def __init__(self, fps: float = 0.5, output_dir: str = "recordings", 
                 audio_duration_per_frame: float = 3.0):
        """
        Initialize the enhanced video recorder.
        
        Args:
            fps: Frames per second for video recording (lower values allow more time for audio)
            output_dir: Directory to save video files
            audio_duration_per_frame: Expected duration of audio per frame in seconds
        """
        self.fps = fps
        self.audio_duration_per_frame = audio_duration_per_frame
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Recording state
        self.frames = []
        self.audio_events = []
        self.frame_timestamps = []
        self.recording = False
        self.session_name = None
        
        # Audio assets directory paths to search
        self.audio_search_dirs = [
            Path("audio_assets"),
            Path("eval/game/coop_game_new/audio_assets"),
            Path("audio_assets/bytedance_audio_assets")
        ]
        
        # Load audio manifest for better file matching
        self.audio_manifest = self._load_audio_manifest()
        
        # Check available libraries
        self.use_moviepy = MOVIEPY_AVAILABLE
        self.use_cv2 = CV2_AVAILABLE
        
        if not (self.use_moviepy or self.use_cv2):
            logger.warning("Enhanced video recording disabled: Neither moviepy nor opencv available")
    
    def _load_audio_manifest(self) -> Dict:
        """Load the audio manifest file for better audio file matching."""
        for search_dir in self.audio_search_dirs:
            manifest_path = search_dir / "audio_manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        logger.info(f"Loaded audio manifest from {manifest_path}")
                        return manifest
                except Exception as e:
                    logger.error(f"Failed to load audio manifest from {manifest_path}: {e}")
        
        logger.warning("No audio manifest found")
        return {}
    
    def start_recording(self, session_name: str = None) -> bool:
        """Start a new recording session."""
        if not (self.use_moviepy or self.use_cv2):
            return False
        
        if session_name is None:
            session_name = f"enhanced_recording_{int(time.time())}"
        
        self.session_name = session_name
        self.frames = []
        self.audio_events = []
        self.frame_timestamps = []
        self.recording = True
        
        logger.info(f"Started enhanced video recording: {session_name}")
        logger.info(f"Settings - FPS: {self.fps}, Audio per frame: {self.audio_duration_per_frame}s")
        
        return True
    
    def add_frame(self, frame: np.ndarray, audio_events: List[str] = None) -> None:
        """Add a frame and associated audio events to the recording."""
        if not self.recording:
            return
        
        timestamp = time.time()
        
        # Store frame
        self.frames.append(frame.copy())
        self.frame_timestamps.append(timestamp)
        
        # Store audio events with enhanced metadata
        if audio_events:
            for audio_event in audio_events:
                self.audio_events.append({
                    'timestamp': timestamp,
                    'event': audio_event,
                    'frame_index': len(self.frames) - 1,
                    'frame_time': (len(self.frames) - 1) / self.fps,
                    'audio_file': self._find_best_audio_file(audio_event)
                })
        
        logger.debug(f"Added frame {len(self.frames)} with {len(audio_events) if audio_events else 0} audio events")
    
    def _find_best_audio_file(self, event_text: str) -> Optional[str]:
        """Find the best matching audio file for the given event text."""
        if not event_text or not self.audio_manifest:
            return None
        
        # Extract keywords from event text for matching
        event_lower = event_text.lower()
        keywords = event_lower.replace(":", "").replace(",", "").replace(".", "").split()
        
        # Search through manifest for matches
        generated_files = self.audio_manifest.get("generated_files", {})
        
        best_match = None
        best_score = 0
        
        for role, assets in generated_files.items():
            for asset in assets:
                asset_text = asset.get("text", "").lower()
                
                # Calculate match score based on keyword overlap
                score = 0
                for keyword in keywords:
                    if len(keyword) > 2 and keyword in asset_text:
                        score += len(keyword)  # Longer keywords get higher scores
                
                if score > best_score:
                    best_score = score
                    best_match = asset
        
        if best_match and best_score > 0:
            # Try different path variations to find the actual file
            audio_path = best_match.get("audio_path")
            filename = best_match.get("filename")
            
            for search_dir in self.audio_search_dirs:
                # Try with filename
                if filename:
                    file_path = search_dir / filename
                    if file_path.exists():
                        logger.debug(f"Found audio match for '{event_text}': {file_path}")
                        return str(file_path)
                
                # Try with full path in search directory
                if audio_path:
                    path_filename = Path(audio_path).name
                    file_path = search_dir / path_filename
                    if file_path.exists():
                        logger.debug(f"Found audio match for '{event_text}': {file_path}")
                        return str(file_path)
        
        # Fallback: search for files with keyword matches
        for search_dir in self.audio_search_dirs:
            if search_dir.exists():
                for audio_file in search_dir.glob("*.wav"):
                    filename_lower = audio_file.name.lower()
                    for keyword in keywords:
                        if len(keyword) > 3 and keyword in filename_lower:
                            logger.debug(f"Found fallback audio match for '{event_text}': {audio_file}")
                            return str(audio_file)
        
        logger.debug(f"No audio file found for event: '{event_text}'")
        return None
    
    def stop_recording_and_save(self, output_path: str = None) -> Tuple[bool, str]:
        """Stop recording and save the enhanced video file with audio."""
        if not self.recording or not self.frames:
            return False, "No recording in progress or no frames captured"
        
        self.recording = False
        
        if output_path is None:
            output_path = self.output_dir / f"{self.session_name}_enhanced.mp4"
        else:
            output_path = Path(output_path)
        
        try:
            if self.use_moviepy:
                return self._save_enhanced_with_moviepy(output_path)
            elif self.use_cv2:
                return self._save_with_cv2_fallback(output_path)
            else:
                return False, "No video library available"
        except Exception as e:
            logger.error(f"Enhanced video save error: {e}")
            return False, f"Enhanced video save error: {e}"
    
    def _save_enhanced_with_moviepy(self, output_path: Path) -> Tuple[bool, str]:
        """Save enhanced video using MoviePy with improved audio integration."""
        if not self.frames:
            return False, "No frames to save"
        
        try:
            logger.info(f"Creating enhanced video with {len(self.frames)} frames and {len(self.audio_events)} audio events")
            
            # Create video clip from frames
            video_clip = ImageSequenceClip(self.frames, fps=self.fps)
            logger.info(f"Video duration: {video_clip.duration:.2f} seconds")
            
            # Process audio events for integration
            audio_clips = []
            successful_audio = 0
            
            if self.audio_events:
                logger.info(f"Processing {len(self.audio_events)} audio events...")
                
                for i, event in enumerate(self.audio_events):
                    audio_file = event.get('audio_file')
                    frame_time = event.get('frame_time', 0)
                    
                    if audio_file and os.path.exists(audio_file):
                        try:
                            # Load audio file
                            audio_clip = AudioFileClip(audio_file)
                            
                            # Position audio at the correct time
                            # Add slight overlap for smoother transitions
                            start_time = max(0, frame_time - 0.1)
                            audio_clip = audio_clip.set_start(start_time)
                            
                            # Limit audio duration to not overlap too much with next frame
                            max_duration = min(audio_clip.duration, self.audio_duration_per_frame + 0.5)
                            audio_clip = audio_clip.subclip(0, max_duration)
                            
                            audio_clips.append(audio_clip)
                            successful_audio += 1
                            
                            logger.debug(f"Added audio {i+1}: {audio_file} at {start_time:.2f}s (duration: {max_duration:.2f}s)")
                            
                        except Exception as ae:
                            logger.warning(f"Failed to load audio file {audio_file}: {ae}")
                
                # Composite all audio clips
                if audio_clips:
                    try:
                        logger.info(f"Compositing {len(audio_clips)} audio clips...")
                        final_audio = CompositeAudioClip(audio_clips)
                        
                        # Ensure audio doesn't exceed video duration
                        if final_audio.duration > video_clip.duration:
                            final_audio = final_audio.subclip(0, video_clip.duration)
                        
                        video_clip = video_clip.set_audio(final_audio)
                        logger.info(f"✅ Successfully integrated {successful_audio} audio clips")
                        
                    except Exception as ae:
                        logger.error(f"Audio composition failed: {ae}")
                        successful_audio = 0
                        
                else:
                    logger.warning("No valid audio files found for integration")
            
            # Write the video file
            logger.info(f"Writing video to {output_path}...")
            if successful_audio > 0:
                video_clip.write_videofile(
                    str(output_path), 
                    codec='libx264', 
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                result_msg = f"Enhanced video saved: {output_path} (with {successful_audio} audio clips)"
            else:
                video_clip.write_videofile(
                    str(output_path), 
                    codec='libx264', 
                    audio=False,
                    verbose=False,
                    logger=None
                )
                result_msg = f"Video saved: {output_path} (no audio integrated)"
            
            # Save detailed metadata
            metadata = {
                'session_name': self.session_name,
                'frames_count': len(self.frames),
                'audio_events_count': len(self.audio_events),
                'successful_audio_integrations': successful_audio,
                'fps': self.fps,
                'audio_duration_per_frame': self.audio_duration_per_frame,
                'video_duration': video_clip.duration,
                'frame_timestamps': self.frame_timestamps,
                'audio_events': self.audio_events,
                'enhanced_features': True
            }
            
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            result_msg += f", Metadata: {metadata_path}"
            logger.info(result_msg)
            
            return True, result_msg
            
        except Exception as e:
            logger.error(f"Enhanced MoviePy save error: {e}")
            return False, f"Enhanced MoviePy save error: {e}"
    
    def _save_with_cv2_fallback(self, output_path: Path) -> Tuple[bool, str]:
        """Fallback video save using OpenCV (without audio integration)."""
        if not self.frames:
            return False, "No frames to save"
        
        logger.warning("Using OpenCV fallback - audio integration not available")
        
        try:
            # Get frame dimensions
            height, width = self.frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
            
            # Write frames
            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
            
            video_writer.release()
            
            # Save metadata with audio information
            metadata = {
                'session_name': self.session_name,
                'frames_count': len(self.frames),
                'audio_events_count': len(self.audio_events),
                'fps': self.fps,
                'frame_timestamps': self.frame_timestamps,
                'audio_events': self.audio_events,
                'video_created_with': 'opencv_fallback',
                'audio_integrated': False
            }
            
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return True, f"Video saved (OpenCV fallback): {output_path}, Metadata: {metadata_path}"
            
        except Exception as e:
            logger.error(f"OpenCV save error: {e}")
            return False, f"OpenCV save error: {e}"
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current recording status and statistics."""
        return {
            'recording': self.recording,
            'session_name': self.session_name,
            'frames_count': len(self.frames),
            'audio_events_count': len(self.audio_events),
            'duration_seconds': len(self.frames) / self.fps if self.frames else 0,
            'fps': self.fps,
            'audio_duration_per_frame': self.audio_duration_per_frame,
            'enhanced_features': True,
            'audio_manifest_loaded': bool(self.audio_manifest)
        }


def create_enhanced_video_from_files(frames_dir: str, audio_events_file: str, 
                                   output_path: str, fps: float = 0.5) -> Tuple[bool, str]:
    """
    Create an enhanced video from a directory of frame images and audio events file.
    
    Args:
        frames_dir: Directory containing numbered frame images (e.g., frame_001.jpg)
        audio_events_file: JSON file containing audio events with timestamps
        output_path: Output path for the enhanced video
        fps: Frames per second for the output video
        
    Returns:
        Tuple of (success, message)
    """
    try:
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return False, f"Frames directory not found: {frames_dir}"
        
        # Load audio events
        with open(audio_events_file, 'r') as f:
            audio_data = json.load(f)
        
        audio_events = audio_data.get('audio_events', [])
        
        # Get frame files
        frame_files = sorted(frames_path.glob("*.jpg")) + sorted(frames_path.glob("*.png"))
        if not frame_files:
            return False, f"No frame images found in {frames_dir}"
        
        # Create enhanced recorder
        recorder = EnhancedVideoRecorder(fps=fps, output_dir=Path(output_path).parent)
        recorder.start_recording(f"batch_enhanced_{int(time.time())}")
        
        # Load frames and add to recorder
        for i, frame_file in enumerate(frame_files):
            from PIL import Image
            img = Image.open(frame_file)
            frame_array = np.array(img)
            
            # Find audio events for this frame
            frame_audio = []
            for event in audio_events:
                if event.get('frame_index') == i:
                    frame_audio.append(event.get('event', ''))
            
            recorder.add_frame(frame_array, frame_audio)
        
        # Save the enhanced video
        success, message = recorder.stop_recording_and_save(output_path)
        return success, message
        
    except Exception as e:
        return False, f"Batch processing error: {e}"


if __name__ == "__main__":
    # Example usage
    print("🎥 Enhanced Video Recorder Test")
    
    # Create a test recorder
    recorder = EnhancedVideoRecorder(fps=0.5, audio_duration_per_frame=3.0)
    
    # Print status
    status = recorder.get_current_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    print("✅ Enhanced Video Recorder initialized successfully!") 