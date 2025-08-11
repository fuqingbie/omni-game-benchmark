"""
Simplified OpenAI Gym Wrapper for Cooperative Command Game

This wrapper provides a clean interface with normalized scores built into
the observation and info feedback, with optional video recording capabilities.
"""

import gym
import numpy as np
import pygame
import json
import time
import base64
import io
import tempfile
import os
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
from gym import spaces
from pathlib import Path

from env import CoopCommandEnv, GameConfig, GameDifficulty

# Try to import video processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
    import moviepy.config as moviepy_config
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Try to import enhanced video recorder
try:
    from enhanced_video_recorder import EnhancedVideoRecorder
    ENHANCED_VIDEO_AVAILABLE = True
except ImportError:
    ENHANCED_VIDEO_AVAILABLE = False

# Try to import GUI for enhanced visual rendering
try:
    from gui_interface import GameGUI, GUIConfig, Colors
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️ GUI interface not available. Using basic visual rendering.")


class VideoRecorder:
    """Records game sessions as video files with integrated audio."""
    
    def __init__(self, fps: float = 30, output_dir: str = "recordings"):
        self.fps = max(fps, 0.1)  # Ensure fps is never 0 or negative
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Frame and audio buffers
        self.frames = []
        self.audio_events = []
        self.frame_timestamps = []
        
        # Video writer setup
        self.video_writer = None
        self.temp_video_path = None
        self.recording = False
        
        # Check available libraries
        self.use_moviepy = MOVIEPY_AVAILABLE
        self.use_cv2 = CV2_AVAILABLE
        
        if not (self.use_moviepy or self.use_cv2):
            print("⚠️ Video recording disabled: Neither moviepy nor opencv available")
    
    def start_recording(self, session_name: str = None):
        """Start a new recording session."""
        if not (self.use_moviepy or self.use_cv2):
            return False
        
        if session_name is None:
            session_name = f"recording_{int(time.time())}"
        
        self.session_name = session_name
        self.frames = []
        self.audio_events = []
        self.frame_timestamps = []
        self.recording = True
        
        # Create temporary video file path
        self.temp_video_path = self.output_dir / f"{session_name}_temp.avi"
        
        return True
    
    def add_frame(self, frame: np.ndarray, audio_events: List[str] = None):
        """Add a frame and associated audio to the recording."""
        if not self.recording:
            return
        
        timestamp = time.time()
        
        # Store frame
        self.frames.append(frame.copy())
        self.frame_timestamps.append(timestamp)
        
        # Store audio events for this frame (but don't integrate into video)
        if audio_events:
            for audio_event in audio_events:
                self.audio_events.append({
                    'timestamp': timestamp,
                    'event': audio_event,
                    'frame_index': len(self.frames) - 1
                })
    
    def stop_recording_and_save(self, output_path: str = None) -> Tuple[bool, str]:
        """Stop recording and save the video file (visual only)."""
        if not self.recording or not self.frames:
            return False, "No recording in progress or no frames captured"
        
        self.recording = False
        
        if output_path is None:
            output_path = self.output_dir / f"{self.session_name}.mp4"
        else:
            output_path = Path(output_path)
        
        try:
            if self.use_cv2:
                return self._save_with_cv2(output_path)
            elif self.use_moviepy:
                return self._save_with_moviepy(output_path)
            else:
                return False, "No video library available"
        except Exception as e:
            return False, f"Video save error: {e}"
    
    def _save_with_cv2(self, output_path: Path) -> Tuple[bool, str]:
        """Save video using OpenCV (visual only)."""
        if not self.frames:
            return False, "No frames to save"
        
        # Get frame dimensions
        height, width = self.frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        # Write frames (visual only)
        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)
        
        video_writer.release()
        
        # Save audio events as separate metadata file
        audio_path = output_path.with_suffix('.json')
        with open(audio_path, 'w') as f:
            json.dump({
                'audio_events': self.audio_events,
                'frame_timestamps': self.frame_timestamps,
                'fps': self.fps
            }, f, indent=2)
        
        return True, f"Video saved (visual only): {output_path}, Audio metadata: {audio_path}"
    
    def _save_with_moviepy(self, output_path: Path) -> Tuple[bool, str]:
        """Save video using MoviePy (visual only)."""
        if not self.frames:
            return False, "No frames to save"
        
        try:
            from moviepy.editor import ImageSequenceClip
            
            # Create video from frames (visual only)
            video_clip = ImageSequenceClip(self.frames, fps=self.fps)
            
            # Write video file without audio
            video_clip.write_videofile(str(output_path), codec='libx264', audio=False)
            
            # Save audio events as separate metadata
            audio_path = output_path.with_suffix('.json')
            with open(audio_path, 'w') as f:
                json.dump({
                    'audio_events': self.audio_events,
                    'frame_timestamps': self.frame_timestamps,
                    'fps': self.fps,
                    'integrated_audio': False  # Audio is separate
                }, f, indent=2)
            
            result_msg = f"Video saved (visual only): {output_path}, Audio metadata: {audio_path}"
            
            return True, result_msg
            
        except Exception as e:
            return False, f"MoviePy save error: {e}"
    
    def _find_audio_file_for_event(self, event_text: str) -> Optional[str]:
        """Try to find an actual audio file for the given event text."""
        if not event_text:
            return None
        
        # Common audio asset directories to check
        audio_dirs = [
            Path("audio_assets"),
            Path("eval/game/coop_game_new/audio_assets"), 
            Path("audio_assets/bytedance_audio_assets")
        ]
        
        # Extract potential keywords from the event text for file matching
        keywords = event_text.lower().replace(":", "").replace(",", "").split()
        
        for audio_dir in audio_dirs:
            if audio_dir.exists():
                # Look for WAV files that might match
                for audio_file in audio_dir.glob("*.wav"):
                    filename_lower = audio_file.name.lower()
                    # Simple matching: if any keyword appears in filename
                    if any(keyword in filename_lower for keyword in keywords if len(keyword) > 3):
                        return str(audio_file)
        
        return None
    
    def get_current_video_data(self) -> Dict[str, Any]:
        """Get current recording data without saving."""
        return {
            'frames': len(self.frames),
            'duration_seconds': len(self.frames) / self.fps if self.frames and self.fps > 0 else 0,
            'audio_events': len(self.audio_events),
            'recording': self.recording
        }


class CoopCommandGymEnv(gym.Env):
    """Simplified OpenAI Gym wrapper around CoopCommandEnv with normalized scoring."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, difficulty: str = "normal", max_rounds: Optional[int] = None, 
                 enable_audio: bool = True, enable_visual: bool = False, seed_index: int = 0,
                 deterministic_commands: bool = True, 
                 recording_mode: str = "individual",  # "individual", "video", or "both"
                 video_fps: float = 30, recordings_dir: str = "recordings",
                 enhanced_video: bool = False, audio_duration_per_frame: float = 3.0):
        super().__init__()
        
        # Configuration
        self.max_rounds = max_rounds
        self.seed_index = seed_index
        self.enable_audio = enable_audio
        self.enable_visual = enable_visual
        self.deterministic_commands = deterministic_commands
        self.recording_mode = recording_mode
        self.video_fps = video_fps
        self.enhanced_video = enhanced_video
        self.audio_duration_per_frame = audio_duration_per_frame
        
        # 🎯 新增：TTS引擎初始化
        self.tts_available = False
        self.tts_engine = None
        
        if enable_audio:
            # 尝试初始化TTS引擎
            self._init_tts_engine()
        
        # Video recording setup
        self.video_recorder = None
        self.enhanced_video_recorder = None
        self.enable_video_recording = recording_mode in ["video", "both"]
        
        if self.enable_video_recording:
            if self.enhanced_video and ENHANCED_VIDEO_AVAILABLE:
                # Use enhanced video recorder with audio integration
                self.enhanced_video_recorder = EnhancedVideoRecorder(
                    fps=video_fps, 
                    output_dir=recordings_dir,
                    audio_duration_per_frame=audio_duration_per_frame
                )
                print(f"🎥 Enhanced video recording enabled (FPS: {video_fps}, Audio per frame: {audio_duration_per_frame}s)")
            else:
                # Use standard video recorder (visual only)
                self.video_recorder = VideoRecorder(fps=video_fps, output_dir=recordings_dir)
                if self.enhanced_video and not ENHANCED_VIDEO_AVAILABLE:
                    print("⚠️ Enhanced video requested but not available, using standard video recording")
            
            # Enable visual rendering if video recording is requested
            if not self.enable_visual:
                self.enable_visual = True
                print("🎥 Auto-enabled visual rendering for video recording")
        
        # Video input mode setup - buffer recent frames for creating video clips
        self.video_input_mode = recording_mode == "video"
        
        # Also enable frame buffering when enhanced video is enabled (for API input video creation)
        self.enable_frame_buffer = self.video_input_mode or self.enhanced_video
        if self.enable_frame_buffer:
            self.frame_buffer = []  # Buffer for recent frames
            self.frame_audio_buffer = []  # Buffer for audio events corresponding to frames
            self.max_buffer_frames = 6  # Changed from 30 to 6 for 6-second clips
            if not self.enable_visual:
                self.enable_visual = True
                if self.video_input_mode:
                    print("🎥 Auto-enabled visual rendering for video input mode")
                else:
                    print("🎥 Auto-enabled visual rendering for enhanced video with frame buffering")
        
        # Create underlying game environment
        diff_enum = GameDifficulty[difficulty.upper()] if isinstance(difficulty, str) else difficulty
        
        # Get noise level from difficulty configuration
        difficulty_configs = {
            "normal": {"noise_level": 0.0},
            "medium": {"noise_level": 0.2},
            "hard": {"noise_level": 0.4}
        }
        difficulty_name = diff_enum.value
        noise_level = difficulty_configs.get(difficulty_name, {"noise_level": 0.0})["noise_level"]
        
        config = GameConfig(
            difficulty=diff_enum, 
            max_rounds=max_rounds,
            seed_index=seed_index,
            enable_audio=enable_audio,
            noise_level=noise_level,
            deterministic_commands=deterministic_commands
        )
        self._game = CoopCommandEnv(config=config, enable_assets=True)
        
        # Enhanced visual rendering setup
        if self.enable_visual:
            if GUI_AVAILABLE:
                # Use sophisticated GUI rendering
                try:
                    print("🎯 Attempting to initialize GUI rendering for evaluation...")
                    
                    # Initialize GUI with proper configuration
                    gui_config = GUIConfig(
                        window_width=800,
                        window_height=600,
                        map_width=500,  # Leave space for panel
                        map_height=600,
                        panel_width=300,
                        tile_size=8
                    )
                    
                    # Create GUI renderer (without starting the game loop)
                    self._gui_renderer = GameGUI(gui_config)
                    
                    # Important: Don't create a display window for evaluation
                    # This prevents pygame from trying to open a window during evaluation
                    if hasattr(self._gui_renderer, 'screen') and self._gui_renderer.screen:
                        print("✅ GUI renderer screen created successfully")
                    else:
                        raise Exception("Failed to create GUI screen surface")
                    
                    # Link the existing game environment to the GUI instead of creating a new one
                    self._gui_renderer.env = self._game
                    self._gui_renderer.visualization = self._game.visualization
                    self._gui_renderer.game_started = False  # Will be set to True when needed
                    
                    self._use_gui_rendering = True
                    print("✅ Enhanced GUI rendering initialized successfully for evaluation")
                    
                except Exception as e:
                    print(f"⚠️ Failed to initialize GUI renderer: {e}")
                    print("🔄 Falling back to enhanced basic rendering...")
                    import traceback
                    traceback.print_exc()
                    self._use_gui_rendering = False
                    self._init_basic_pygame()
            else:
                print("⚠️ GUI interface not available (import failed)")
                print("🔄 Using enhanced basic rendering...")
                self._use_gui_rendering = False
                self._init_basic_pygame()
        
        # Start game to get initial state and team information
        initial_state = self._game.start_game()
        
        # Cache member order for consistent action encoding
        self.member_ids = list(initial_state["team_status"]["members"].keys())
        self.num_members = len(self.member_ids)
        
        # Command types supported by this wrapper
        self.command_types = [
            "move",   # Move to coordinates
            "attack", # Attack position
            "defend", # Defend position
            "recon",  # Recon area
            "status", # Ask for status report
        ]

        # Define action and observation spaces
        # Action: [member_idx, command_type_idx, x_coord, y_coord]
        # member_idx: 0 to num_members-1 (individuals), num_members (all team), num_members+1 (multiple)
        self.action_space = spaces.MultiDiscrete([
            self.num_members + 2,  # 0 to num_members-1 for individuals, num_members for all, num_members+1 for multiple
            len(self.command_types), 
            101,  # x coordinate (0-100)
            101   # y coordinate (0-100)
        ])

        # Status mapping for observation encoding
        self._status_mapping = {
            "idle": 0, "moving": 1, "attacking": 2, "defending": 3,
            "reconnaissance": 4, "dead": 5, "injured": 6
        }

        # Vector observation space: [member_health, member_status, member_x, member_y] * num_members + [rounds_remaining, score_normalized]
        obs_dim = self.num_members * 4 + 2
        
        # Define observation space based on recording mode
        if self.recording_mode == "video":
            # Video recording mode: return actual video content and separate audio
            self.observation_space = spaces.Dict({
                'vector': spaces.Box(low=0.0, high=100.0, shape=(obs_dim,), dtype=np.float32),
                'video': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),  # Placeholder for base64 string
                'audio': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),  # Placeholder for base64 string
                'video_data': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)  # Placeholder for JSON string
            })
        elif self.recording_mode == "both":
            # Both modes: return individual components AND video data with separate audio
            self.observation_space = spaces.Dict({
                'vector': spaces.Box(low=0.0, high=100.0, shape=(obs_dim,), dtype=np.float32),
                'image': spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),
                'audio': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),  # Placeholder for base64 string
                'video': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),  # Placeholder for base64 string
                'video_data': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)  # Placeholder for JSON string
            })
        elif self.enable_visual or self.enable_audio:
            # Traditional multi-modal observation space (individual mode)
            obs_dict = {'vector': spaces.Box(low=0.0, high=100.0, shape=(obs_dim,), dtype=np.float32)}
            
            if self.enable_visual:
                obs_dict['image'] = spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8)
            
            if self.enable_audio:
                obs_dict['audio'] = spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)  # Placeholder for base64 string
                
            self.observation_space = spaces.Dict(obs_dict)
        else:
            # Simple vector observation
            self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(obs_dim,), dtype=np.float32)

        # Internal state tracking
        self._last_score = 0
        self._done = False
        self._cached_obs = None
        
        # Multi-member command support
        self._pending_member_list = None

    def _init_basic_pygame(self):
        """Initialize basic pygame rendering as fallback"""
        try:
            pygame.init()
            pygame.font.init()
            self._display_size = (800, 600)
            self._font = pygame.font.SysFont(None, 24)
            self._render_surface = pygame.Surface(self._display_size)
            print("✅ Using basic pygame rendering for visual observations")
        except Exception as e:
            print(f"⚠️ Failed to initialize pygame for visual rendering: {e}")
            self.enable_visual = False

    def _init_tts_engine(self):
        """初始化TTS引擎"""
        try:
            # 优先尝试gTTS（需要网络）
            try:
                from gtts import gTTS
                # 测试gTTS是否可用
                test_tts = gTTS(text="test", lang='en', slow=False)
                self.tts_engine = "gtts"
                self.tts_available = True
                print("✅ gTTS引擎初始化成功")
                return
            except ImportError:
                print("gTTS不可用，尝试pyttsx3...")
            except Exception as e:
                print(f"gTTS初始化失败: {e}，尝试其他引擎...")
            
            # 尝试pyttsx3（本地TTS）
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                self.tts_engine = engine
                self.tts_available = True
                print("✅ pyttsx3引擎初始化成功")
                return
            except ImportError:
                print("pyttsx3不可用，尝试edge-tts...")
            except Exception as e:
                print(f"pyttsx3初始化失败: {e}，尝试其他引擎...")
            
            # 尝试edge-tts
            try:
                import edge_tts
                self.tts_engine = "edge-tts"
                self.tts_available = True
                print("✅ edge-tts引擎初始化成功")
                return
            except ImportError:
                print("edge-tts不可用")
            except Exception as e:
                print(f"edge-tts初始化失败: {e}")
                pass
            
            print("⚠️ 没有可用的TTS引擎，音频将以文本形式提供")
            self.tts_available = False
            self.tts_engine = None
            
        except Exception as e:
            print(f"TTS引擎初始化错误: {e}")
            self.tts_available = False
            self.tts_engine = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        # Set seed manually if provided (don't call super().reset() as it's not implemented)
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)
        
        state = self._game.start_game()
        self._last_score = 0
        self._done = False
        
        # Clear frame buffer for video input mode or enhanced video
        if self.enable_frame_buffer:
            self.frame_buffer = []
            self.frame_audio_buffer = []
            print("🔄 Cleared frame and audio buffers for new episode")
        
        # Start video recording if enabled
        if self.enable_video_recording:
            session_name = f"game_seed_{self.seed_index}_{int(time.time())}"
            if options and 'session_name' in options:
                session_name = options['session_name']
            
            if self.enhanced_video_recorder:
                self.enhanced_video_recorder.start_recording(session_name)
                print(f"🎥 Started enhanced video recording: {session_name}")
            elif self.video_recorder:
                self.video_recorder.start_recording(session_name)
                print(f"🎥 Started video recording: {session_name}")
        
        self._cached_obs = self._state_to_obs(state)
        
        # Create comprehensive info with normalized scores
        info = self._create_info_dict(state)
        
        return self._cached_obs, info

    def set_multi_member_list(self, member_list: List[int]):
        """Set the member list for the next multi-member command."""
        self._pending_member_list = member_list

    def step(self, action):
        """Execute one step in the environment"""
        if self._done:
            return self._cached_obs, 0.0, True, False, {}

        # Decode action
        member_idx, cmd_idx, x_coord, y_coord = action
        
        # Validate command type index
        if not (0 <= cmd_idx < len(self.command_types)):
            return self._cached_obs, 0.0, True, False, {"error": "Invalid command type index"}
        
        cmd_type = self.command_types[int(cmd_idx)]
        tgt_position = (float(x_coord), float(y_coord))

        # Build natural-language command string
        if cmd_type == "move":
            cmd_str = f"move to {int(x_coord)}, {int(y_coord)}"
        elif cmd_type == "attack":
            cmd_str = f"attack position {int(x_coord)}, {int(y_coord)}"
        elif cmd_type == "defend":
            cmd_str = f"defend position {int(x_coord)}, {int(y_coord)}"
        elif cmd_type == "recon":
            cmd_str = f"scout area {int(x_coord)}, {int(y_coord)}"
        elif cmd_type == "status":
            cmd_str = "report status"
            tgt_position = None
        else:
            raise ValueError("Invalid command type index")

        # Handle different member selection modes
        member_idx = int(member_idx)
        
        if member_idx == self.num_members:
            # Team-wide command: execute for all team members
            total_reward = 0.0
            responses = []
            
            for m_id in self.member_ids:
                result = self._game.execute_command(cmd_str, target_member=m_id, target_position=tgt_position)
                if result.get("success", False):
                    responses.append(f"{m_id}: {result.get('response', 'Acknowledged')}")
                else:
                    responses.append(f"{m_id}: {result.get('message', 'Failed')}")
            
            print(f"Team command executed: {cmd_str}")
            print(f"Responses: {'; '.join(responses[:3])}...")  # Show first 3 responses
                
        elif member_idx == self.num_members + 1:
            # Multiple specific members command - use pending member list if available
            if self._pending_member_list:
                # Use the specific member indices that were parsed
                target_member_indices = self._pending_member_list
                target_members = [self.member_ids[i] for i in target_member_indices if 0 <= i < len(self.member_ids)]
                self._pending_member_list = None  # Clear after use
            else:
                # Fallback to first two members if no list was provided
                target_members = self.member_ids[:min(2, len(self.member_ids))]
            
            responses = []
            
            for m_id in target_members:
                result = self._game.execute_command(cmd_str, target_member=m_id, target_position=tgt_position)
                if result.get("success", False):
                    responses.append(f"{m_id}: {result.get('response', 'Acknowledged')}")
                else:
                    responses.append(f"{m_id}: {result.get('message', 'Failed')}")
            
            print(f"Multi-member command executed for {len(target_members)} members: {cmd_str}")
            print(f"Responses: {'; '.join(responses)}")
                
        else:
            # Individual member command (original behavior)
            if not (0 <= member_idx < len(self.member_ids)):
                return self._cached_obs, 0.0, True, False, {"error": "Invalid member index"}
            
            member_id = self.member_ids[member_idx]
            result = self._game.execute_command(cmd_str, target_member=member_id, target_position=tgt_position)
            
            print(f"Individual command: {member_id} - {cmd_str}")
            if result.get("success", False):
                print(f"Response: {result.get('response', 'Acknowledged')}")
            else:
                print(f"Failed: {result.get('message', 'Command failed')}")
        
        # Advance simulation by one round
        state = self._game.step()

        # Compute reward based on normalized score change
        current_score = state.get("score_normalized", 0)
        reward = float(current_score - self._last_score)
        self._last_score = current_score

        # Termination conditions
        terminated = state["state"] in ("completed", "failed")
        truncated = False
        self._done = terminated

        obs = self._state_to_obs(state)
        self._cached_obs = obs
        
        # Handle video recording
        if self.enable_video_recording:
            active_recorder = self.enhanced_video_recorder if self.enhanced_video_recorder else self.video_recorder
            if active_recorder and (
                (hasattr(active_recorder, 'recording') and active_recorder.recording) or
                (hasattr(active_recorder, 'video_recorder') and active_recorder.video_recorder and active_recorder.video_recorder.recording)
            ):
                # Add frame to video recording
                if self.enable_visual:
                    try:
                        image_array = self._render_to_array()
                        audio_events = [msg.get("message", "") for msg in state.get("audio_messages", [])]
                        active_recorder.add_frame(image_array, audio_events)
                    except Exception as e:
                        print(f"Video recording frame error: {e}")
                
                # Stop recording if game is terminated
                if terminated:
                    try:
                        success, message = active_recorder.stop_recording_and_save()
                        if success:
                            if self.enhanced_video_recorder:
                                print(f"🎥 Enhanced {message}")
                            else:
                                print(f"🎥 {message}")
                        else:
                            print(f"⚠️ Video save failed: {message}")
                    except Exception as e:
                        print(f"Video save error: {e}")
        
        # Create comprehensive info with normalized scores
        info = self._create_info_dict(state)
        
        return obs, reward, terminated, truncated, info

    def _normalize_team_member_info(self, info, member_id: str) -> Dict:
        """Normalize team member info to consistent dictionary format."""
        try:
            if isinstance(info, dict):
                # Already in correct format, just validate and fill defaults
                result = {
                    "health": float(info.get("health", 100)),
                    "status": info.get("status", "idle"),
                    "position": {
                        "x": float(info.get("position", {}).get("x", 0) if isinstance(info.get("position"), dict) else 0),
                        "y": float(info.get("position", {}).get("y", 0) if isinstance(info.get("position"), dict) else 0)
                    },
                    "name": info.get("name", member_id),
                    "role": info.get("role", "unknown"),
                    "ammo": float(info.get("ammo", 100)),
                    "current_task": info.get("current_task"),
                    "task_progress": float(info.get("task_progress", 0.0))
                }
                return result
            elif isinstance(info, (list, tuple)) and len(info) >= 4:
                # Convert tuple/list format to dictionary
                result = {
                    "health": float(info[0]) if len(info) > 0 else 100.0,
                    "status": str(info[1]) if len(info) > 1 else "idle",
                    "position": {
                        "x": float(info[2]) if len(info) > 2 else 0.0,
                        "y": float(info[3]) if len(info) > 3 else 0.0
                    },
                    "name": str(info[4]) if len(info) > 4 else member_id,
                    "role": str(info[5]) if len(info) > 5 else "unknown",
                    "ammo": float(info[6]) if len(info) > 6 else 100.0,
                    "current_task": str(info[7]) if len(info) > 7 else None,
                    "task_progress": float(info[8]) if len(info) > 8 else 0.0
                }
                return result
            else:
                # Fallback for completely unexpected format
                print(f"Warning: Unexpected team member info format for {member_id}: {type(info)} - {info}")
                result = {
                    "health": 100.0,
                    "status": "idle", 
                    "position": {"x": 0.0, "y": 0.0},
                    "name": member_id,
                    "role": "unknown",
                    "ammo": 100.0,
                    "current_task": None,
                    "task_progress": 0.0
                }
                return result
        except Exception as norm_error:
            print(f"Error normalizing member {member_id}: {norm_error}")
            # Return safe fallback
            return {
                "health": 100.0,
                "status": "idle", 
                "position": {"x": 0.0, "y": 0.0},
                "name": member_id,
                "role": "unknown",
                "ammo": 100.0,
                "current_task": None,
                "task_progress": 0.0
            }

    def _state_to_obs(self, state: dict):
        """Convert game state to observation"""
        try:
            team_members = state["team_status"]["members"]
            obs_list = []
            
            # Encode team member states
            for m_id in self.member_ids:
                try:
                    raw_info = team_members[m_id]
                    # Always normalize to dictionary format first
                    info = self._normalize_team_member_info(raw_info, m_id)
                    
                    # Extract data from normalized dictionary
                    health = info["health"]
                    status = info["status"]
                    status_code = float(self._status_mapping.get(status, 0))
                    x = info["position"]["x"]
                    y = info["position"]["y"]
                    
                    obs_list.extend([health, status_code, x, y])
                    
                except Exception as member_error:
                    print(f"Error processing member {m_id}: {member_error}")
                    # Use safe defaults
                    obs_list.extend([100.0, 0.0, 0.0, 0.0])
        
        except Exception as state_error:
            print(f"Error in _state_to_obs: {state_error}")
            print(f"State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
            # Return minimal safe observation
            obs_list = [100.0, 0.0, 0.0, 0.0] * self.num_members

        # Add global game state safely
        try:
            rounds_rem = float(state.get("rounds_remaining", 0))
            score_normalized = float(state.get("score_normalized", 0))
            obs_list.extend([rounds_rem, score_normalized])
        except Exception as global_error:
            print(f"Error processing global state: {global_error}")
            obs_list.extend([0.0, 0.0])  # Safe defaults
        
        # Ensure obs_list has the expected length
        expected_length = self.num_members * 4 + 2
        if len(obs_list) != expected_length:
            print(f"Warning: obs_list length {len(obs_list)} != expected {expected_length}")
            obs_list = obs_list[:expected_length] + [0.0] * max(0, expected_length - len(obs_list))
        
        vector_obs = np.array(obs_list, dtype=np.float32)
        
        # Return observation based on recording mode with separate audio
        if self.recording_mode == "video":
            # Video mode: return vector data and visual video + separate audio
            observation = {'vector': vector_obs}
            
            if self.enable_visual:
                try:
                    # Get current frame for buffer
                    current_frame = self._render_to_array()
                    
                    # Add current frame to buffer for video creation
                    if self.enable_frame_buffer:
                        self.frame_buffer.append(current_frame.copy())
                        # Maintain buffer size limit
                        if len(self.frame_buffer) > self.max_buffer_frames:
                            self.frame_buffer.pop(0)
                        
                        # Create video clip from recent frames (visual only)
                        if len(self.frame_buffer) >= 1:
                            video_bytes = self._create_video_clip_from_buffer()
                            if video_bytes:
                                # Base64 encode the visual-only video
                                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                                observation['video'] = video_base64
                                print(f"🎬 Created visual-only video: {len(video_bytes)} bytes, {len(self.frame_buffer)} frames")
                            else:
                                # Fallback to single frame as JPEG
                                observation['video'] = self._create_single_frame_fallback()
                        else:
                            # Not enough frames yet, use single frame
                            observation['video'] = self._create_single_frame_fallback()
                    else:
                        # Not in video input mode, use single frame
                        observation['video'] = self._create_single_frame_fallback()
                        
                except Exception as e:
                    print(f"Video creation error: {e}")
                    observation['video'] = ""
            else:
                observation['video'] = ""
            
            # 🎯 新增：生成独立的音频内容
            if self.enable_audio:
                step_count = state.get("current_round", 0)
                
                # 尝试生成真实的音频文件
                audio_bytes = self._generate_step_audio_file(state, step_count)
                
                if audio_bytes:
                    # 将音频字节编码为base64 - 使用全局导入的base64模块
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    observation['audio'] = audio_base64
                    print(f"🎤 生成独立音频: {len(audio_bytes)} 字节")
                else:
                    # 回退到JSON格式的文字指导
                    enhanced_audio = self._generate_enhanced_audio_guidance(state, step_count)
                    audio_messages = state.get("audio_messages", [])
                    combined_audio = {
                        "guidance": enhanced_audio,
                        "team_communications": [msg.get("message", "") for msg in audio_messages[-3:]],
                        "audio_type": "text"  # 标识这是文字而非音频
                    }
                    observation['audio'] = json.dumps(combined_audio)
                    print(f"📝 回退到文字音频指导")
            else:
                observation['audio'] = ""
            
            # Also add video recording metadata
            if self.video_recorder:
                video_data = self.video_recorder.get_current_video_data()
                video_data['audio_messages'] = state.get("audio_messages", [])
                observation['video_data'] = json.dumps(video_data)
            else:
                observation['video_data'] = json.dumps({})
                
            return observation
            
        elif self.recording_mode == "both":
            # Both modes: return individual components AND visual video + separate audio
            observation = {'vector': vector_obs}
            
            # Add visual observation
            if self.enable_visual:
                try:
                    image_array = self._render_to_array()
                    observation['image'] = image_array
                    
                    # 也提供base64编码的纯视觉视频
                    if self.enable_frame_buffer:
                        self.frame_buffer.append(image_array.copy())
                        if len(self.frame_buffer) > self.max_buffer_frames:
                            self.frame_buffer.pop(0)
                    
                    # Create visual-only video for compatibility
                    observation['video'] = self._create_single_frame_fallback()
                    
                except Exception as e:
                    print(f"Visual rendering error: {e}")
                    # Fallback to black image
                    observation['image'] = np.zeros((600, 800, 3), dtype=np.uint8)
                    observation['video'] = ""
            else:
                observation['video'] = ""
            
            # 🎯 新增：独立音频观察
            if self.enable_audio:
                step_count = state.get("current_round", 0)
                audio_bytes = self._generate_step_audio_file(state, step_count)
                
                if audio_bytes:
                    # 返回base64编码的音频 - 使用全局导入的base64模块
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    observation['audio'] = audio_base64
                else:
                    # 回退到文字格式
                    enhanced_audio = self._generate_enhanced_audio_guidance(state, step_count)
                    audio_messages = state.get("audio_messages", [])
                    combined_audio = {
                        "guidance": enhanced_audio,
                        "team_communications": [msg.get("message", "") for msg in audio_messages[-3:]],
                        "audio_type": "text"
                    }
                    observation['audio'] = json.dumps(combined_audio)
            
            # Add video recording data
            if self.video_recorder:
                video_data = self.video_recorder.get_current_video_data()
                video_data['audio_messages'] = state.get("audio_messages", [])
                observation['video_data'] = json.dumps(video_data)
            else:
                observation['video_data'] = json.dumps({})
                
            return observation
            
        elif self.enable_visual or self.enable_audio:
            # Traditional individual mode: return multi-modal observation with separate audio
            observation = {'vector': vector_obs}
            
            # Add visual observation
            if self.enable_visual:
                try:
                    image_array = self._render_to_array()
                    observation['image'] = image_array
                    
                    # Add frame to buffer for enhanced video creation (if enabled)
                    if self.enable_frame_buffer:
                        self.frame_buffer.append(image_array.copy())
                        if len(self.frame_buffer) > self.max_buffer_frames:
                            self.frame_buffer.pop(0)
                            
                except Exception as e:
                    print(f"Visual rendering error: {e}")
                    # Fallback to black image
                    observation['image'] = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # 🎯 新增：独立音频观察（不再与视频混合）
            if self.enable_audio:
                step_count = state.get("current_round", 0)
                
                # 尝试生成真实音频
                audio_bytes = self._generate_step_audio_file(state, step_count)
                
                if audio_bytes:
                    # 返回base64编码的音频 - 使用全局导入的base64模块
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    observation['audio'] = audio_base64
                else:
                    # 回退到文字格式
                    enhanced_audio = self._generate_enhanced_audio_guidance(state, step_count)
                    audio_messages = state.get("audio_messages", [])
                    combined_audio = {
                        "guidance": enhanced_audio,
                        "team_communications": [msg.get("message", "") for msg in audio_messages[-3:]],
                        "audio_type": "text"
                    }
                    observation['audio'] = json.dumps(combined_audio)
                
            return observation
        else:
            # Simple vector observation
            return vector_obs

    def _create_info_dict(self, state: dict) -> Dict[str, Any]:
        """Create comprehensive info dictionary with normalized scores"""
        return {
            # Core scores and metrics
            "main_score": state.get("main_score", 0),
            "score_normalized": state.get("score_normalized", 0.0),
            "max_possible_score": state.get("max_possible_score", 0),
            
            # Round information
            "current_round": state.get("current_round", 0),
            "max_rounds": state.get("max_rounds", 50),
            "optimal_rounds": state.get("optimal_rounds", 10),
            "naive_optimal_rounds": state.get("naive_optimal_rounds", 10),
            "worst_case_optimal_rounds": state.get("worst_case_optimal_rounds", 10),
            "rounds_remaining": state.get("rounds_remaining", 0),
            
            # Objective tracking
            "objectives_completed": len(state.get("completed_objectives", [])),
            "total_objectives": len(state.get("mission_objectives", [])),
            "objectives_failed": len(state.get("failed_objectives", [])),
            
            # Performance metrics
            "success_rate": len(state.get("completed_objectives", [])) / max(1, len(state.get("mission_objectives", []))) * 100,
            "efficiency": min(100, (state.get("optimal_rounds", 10) / max(1, state.get("current_round", 1))) * 100),
            
            # Game state
            "game_state": state.get("state", "unknown"),
            "game_seed": state.get("game_seed", 0),
            
            # Command tracking
            "recent_commands": len(state.get("recent_commands", [])),
            "auxiliary_command_score": state.get("auxiliary_command_score", 0)
        }

    def _render_to_array(self) -> np.ndarray:
        """Render game state to numpy array for visual observation with enhanced GUI rendering"""
        if not self.enable_visual:
            return np.zeros((600, 800, 3), dtype=np.uint8)
        
        try:
            if hasattr(self, '_use_gui_rendering') and self._use_gui_rendering:
                # Use sophisticated GUI rendering
                return self._render_with_gui()
            else:
                # Fallback to basic rendering
                return self._render_basic()
        except Exception as e:
            print(f"Visual rendering error: {e}")
            return np.zeros((600, 800, 3), dtype=np.uint8)

    def _render_with_gui(self) -> np.ndarray:
        """Render using the sophisticated GUI system"""
        try:
            # Get current game state
            state = self._game.get_game_state()
            
            # Update GUI state
            self._gui_renderer.current_state = state
            self._gui_renderer.game_started = True
            
            # Get render data from visualization system
            render_data = self._game.visualization.get_render_data(
                state['team_status'],
                state.get('hidden_objectives', []),
                state.get('mission_objectives', [])  # Pass visible objectives correctly
            )
            
            # Clear the GUI surface
            self._gui_renderer.screen.fill(Colors.BACKGROUND)
            
            # Render using GUI methods - this includes map and full info panel
            self._gui_renderer._render_map(render_data)
            self._gui_renderer._render_info_panel(render_data)
            
            # Convert GUI surface to numpy array
            array = pygame.surfarray.array3d(self._gui_renderer.screen)
            return array.transpose(1, 0, 2)  # Convert to (height, width, 3)
            
        except Exception as e:
            print(f"GUI rendering error: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Falling back to enhanced basic rendering...")
            return self._render_basic()

    def _add_info_overlay(self, state: Dict):
        """Add key information overlay to the GUI rendering"""
        try:
            # Add compact text overlay in top-right corner
            y_offset = 10
            x_start = 600  # Right side of map
            
            # Score information
            score_text = f"Score: {state.get('score_normalized', 0):.1f}/100"
            score_surface = self._gui_renderer.font_small.render(score_text, True, Colors.TEXT_HIGHLIGHT)
            self._gui_renderer.screen.blit(score_surface, (x_start, y_offset))
            y_offset += 20
            
            # Round information
            round_text = f"Round: {state.get('current_round', 0)}/{state.get('max_rounds', 0)}"
            round_surface = self._gui_renderer.font_small.render(round_text, True, Colors.TEXT)
            self._gui_renderer.screen.blit(round_surface, (x_start, y_offset))
            y_offset += 20
            
            # Objectives count
            completed = len(state.get('completed_objectives', []))
            total = len(state.get('mission_objectives', []))
            obj_text = f"Objectives: {completed}/{total}"
            obj_surface = self._gui_renderer.font_small.render(obj_text, True, Colors.TEXT)
            self._gui_renderer.screen.blit(obj_surface, (x_start, y_offset))
            
        except Exception as e:
            print(f"Info overlay error: {e}")

    def _create_enhanced_video_clip_from_buffer(self) -> Optional[bytes]:
        """Create a short video clip with audio integration from recent frames in buffer."""
        if not self.enable_frame_buffer or not self.frame_buffer:
            return None
        
        # Only create enhanced video if enhanced video recording is enabled
        if not self.enhanced_video or not ENHANCED_VIDEO_AVAILABLE:
            return self._create_video_clip_from_buffer()  # Fall back to regular video
        
        try:
            from enhanced_video_recorder import EnhancedVideoRecorder
            import tempfile
            
            # Create temporary enhanced video recorder
            temp_recorder = EnhancedVideoRecorder(
                fps=max(0.1, min(self.video_fps, 2.0)),  # Use appropriate FPS for API input
                output_dir=tempfile.gettempdir(),
                audio_duration_per_frame=self.audio_duration_per_frame
            )
            
            # Start recording
            session_name = f"api_input_{int(time.time())}"
            temp_recorder.start_recording(session_name)
            
            # Add frames and corresponding audio events
            for i, frame in enumerate(self.frame_buffer):
                # Get audio events for this frame
                frame_audio = []
                if i < len(self.frame_audio_buffer):
                    frame_audio = self.frame_audio_buffer[i]
                
                temp_recorder.add_frame(frame, frame_audio)
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_output_path = temp_file.name
            
            # Stop recording and save enhanced video
            success, message = temp_recorder.stop_recording_and_save(temp_output_path)
            
            if success:
                # Read the enhanced video file as bytes
                with open(temp_output_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Clean up temporary file
                os.unlink(temp_output_path)
                
                print(f"✅ Created enhanced API video with audio from {len(self.frame_buffer)} frames")
                return video_bytes
            else:
                print(f"⚠️ Enhanced video creation failed: {message}")
                return self._create_video_clip_from_buffer()  # Fall back to regular video
                
        except Exception as e:
            print(f"Enhanced video clip creation error: {e}")
            return self._create_video_clip_from_buffer()  # Fall back to regular video

    def _create_video_clip_from_buffer(self) -> Optional[bytes]:
        """Create a short video clip from recent frames in buffer."""
        if not self.enable_frame_buffer or not self.frame_buffer:
            return None
        
        try:
            # Import necessary libraries for video creation
            if MOVIEPY_AVAILABLE:
                from moviepy.editor import ImageSequenceClip
                import tempfile
                
                # Create video clip from buffered frames
                # Use API-compatible frame rate (allow low FPS for audio integration)
                clip_fps = max(0.1, min(self.video_fps, 10))  # Minimum 0.1 FPS, max 10 FPS for model input
                
                # Ensure we have enough frames for a meaningful video (minimum 2 seconds)
                min_frames_needed = max(2, int(clip_fps * 2))  # At least 2 seconds worth of frames
                frames_to_use = self.frame_buffer.copy()
                
                # If we don't have enough frames, duplicate the last frame
                while len(frames_to_use) < min_frames_needed:
                    frames_to_use.append(frames_to_use[-1] if frames_to_use else self._render_to_array())
                
                video_clip = ImageSequenceClip(frames_to_use, fps=clip_fps)
                
                # Create temporary file for the video
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Write video to temporary file
                video_clip.write_videofile(
                    temp_path, 
                    codec='libx264', 
                    audio=False,  # No audio for model input clips
                    verbose=False,
                    logger=None  # Suppress moviepy logs
                )
                
                # Read the video file as bytes
                with open(temp_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                print(f"✅ Created API-compatible video from {len(frames_to_use)} frames ({video_clip.duration:.1f}s @ {clip_fps}fps)")
                return video_bytes
                
            elif CV2_AVAILABLE:
                import tempfile
                
                # Fallback to OpenCV video creation
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                if self.frame_buffer:
                    height, width = self.frame_buffer[0].shape[:2]
                    
                    # Use API-compatible frame rate (allow low FPS for audio integration)
                    clip_fps = max(0.1, min(self.video_fps, 10))  # Minimum 0.1 FPS, max 10 FPS
                    
                    # Ensure we have enough frames for a meaningful video (minimum 2 seconds)
                    min_frames_needed = max(2, int(clip_fps * 2))  # At least 2 seconds worth of frames
                    frames_to_use = self.frame_buffer.copy()
                    
                    # If we don't have enough frames, duplicate the last frame
                    while len(frames_to_use) < min_frames_needed:
                        frames_to_use.append(frames_to_use[-1] if frames_to_use else self._render_to_array())
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(temp_path, fourcc, clip_fps, (width, height))
                    
                    # Write frames
                    for frame in frames_to_use:
                        # Convert RGB to BGR for OpenCV
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(bgr_frame)
                    
                    video_writer.release()
                    
                    # Read the video file as bytes
                    with open(temp_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    # Clean up temporary file
                    os.unlink(temp_path)
                    
                    print(f"✅ Created API-compatible video using OpenCV from {len(frames_to_use)} frames ({len(frames_to_use)/clip_fps:.1f}s @ {clip_fps}fps)")
                    return video_bytes
            
            else:
                print("⚠️ No video creation library available (MoviePy or OpenCV)")
                return None
                
        except Exception as e:
            print(f"Video clip creation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _render_basic(self) -> np.ndarray:
        """Fallback basic rendering with objectives display"""
        if not hasattr(self, '_render_surface'):
            return np.zeros((600, 800, 3), dtype=np.uint8)
        
        try:
            # Clear surface with green background (similar to GUI)
            self._render_surface.fill((144, 238, 144))  # Light green background
            
            # Get current game state
            state = self._game.get_game_state()
            team_status = state["team_status"]["members"]
            
            # Create a right panel area for information
            panel_rect = pygame.Rect(500, 0, 300, 600)
            pygame.draw.rect(self._render_surface, (50, 50, 50), panel_rect)  # Dark panel
            pygame.draw.rect(self._render_surface, (100, 100, 100), panel_rect, 2)  # Border
            
            # Panel content
            panel_x = 510
            y_offset = 10
            
            # Title
            title_surface = self._font.render("Game Status", True, (255, 255, 0))
            self._render_surface.blit(title_surface, (panel_x, y_offset))
            y_offset += 30
            
            # Score information
            score_text = f"Score: {state['main_score']}"
            score_surface = self._font.render(score_text, True, (255, 255, 255))
            self._render_surface.blit(score_surface, (panel_x, y_offset))
            y_offset += 20
            
            normalized_score = state.get('score_normalized', 0)
            norm_text = f"Normalized: {normalized_score:.1f}/100"
            norm_surface = self._font.render(norm_text, True, (255, 255, 255))
            self._render_surface.blit(norm_surface, (panel_x, y_offset))
            y_offset += 25
            
            # Round information
            round_text = f"Round: {state['current_round']}/{state['max_rounds']}"
            round_surface = self._font.render(round_text, True, (255, 255, 255))
            self._render_surface.blit(round_surface, (panel_x, y_offset))
            y_offset += 30
            
            # Team members section
            team_title = self._font.render("Team Members", True, (255, 255, 0))
            self._render_surface.blit(team_title, (panel_x, y_offset))
            y_offset += 25
            
            # Render team members on map and in panel
            member_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 255), (255, 255, 0), (128, 0, 128), (255, 165, 0)]
            for i, (member_id, raw_member_info) in enumerate(team_status.items()):
                # Normalize member info to consistent format
                member_info = self._normalize_team_member_info(raw_member_info, member_id)
                
                # Member on map - use proper coordinate scaling
                padding = 10
                pos_x = int(padding + (member_info['position']['x'] / 100.0) * (500 - 2 * padding))
                pos_y = int(padding + (member_info['position']['y'] / 100.0) * (600 - 2 * padding))
                if 0 <= pos_x <= 500 and 0 <= pos_y <= 600:
                    color = member_colors[i % len(member_colors)]
                    pygame.draw.circle(self._render_surface, color, (pos_x, pos_y), 8)
                    pygame.draw.circle(self._render_surface, (255, 255, 255), (pos_x, pos_y), 8, 2)
                
                # Member in panel
                name = member_info['name']
                role = member_info['role']
                member_text = f"{name} ({role})"
                if len(member_text) > 20:
                    member_text = member_text[:17] + "..."
                member_surface = self._font.render(member_text, True, (255, 255, 255))
                self._render_surface.blit(member_surface, (panel_x, y_offset))
                y_offset += 18
                
                # Health and position
                health = member_info['health']
                status = member_info['status']
                info_text = f"  {health}% HP, {status}"
                info_surface = pygame.font.Font(None, 16).render(info_text, True, (200, 200, 200))
                self._render_surface.blit(info_surface, (panel_x, y_offset))
                y_offset += 15
                
                pos_text = f"  ({member_info['position']['x']:.1f}, {member_info['position']['y']:.1f})"
                pos_surface = pygame.font.Font(None, 16).render(pos_text, True, (200, 200, 200))
                self._render_surface.blit(pos_surface, (panel_x, y_offset))
                y_offset += 20
            
            # Objectives section
            y_offset += 10
            obj_title = self._font.render("Objectives", True, (255, 255, 0))
            self._render_surface.blit(obj_title, (panel_x, y_offset))
            y_offset += 25
            
            # Check remaining space for objectives
            panel_bottom = 600 - 20  # Reserve bottom space
            remaining_height = panel_bottom - y_offset - 30  # Reserve space at bottom
            
            # Visible objectives
            objectives = state.get('mission_objectives', [])
            if objectives:
                objectives_shown = 0
                estimated_height_per_obj = 35  # Updated estimate for two-line objectives
                max_objectives_that_fit = max(2, remaining_height // estimated_height_per_obj)  # Show at least 2
                
                for i, obj in enumerate(objectives):
                    if objectives_shown >= max_objectives_that_fit:
                        # Show truncation indicator
                        if len(objectives) > objectives_shown:
                            truncate_text = f"... and {len(objectives) - objectives_shown} more objectives"
                            truncate_surface = pygame.font.Font(None, 16).render(truncate_text, True, (128, 128, 128))
                            self._render_surface.blit(truncate_surface, (panel_x, y_offset))
                        break
                    
                    # Check if we have enough space for this objective (two lines)
                    if y_offset + estimated_height_per_obj > panel_bottom:
                        break
                    
                    # Status circle
                    status = obj.get('status', 'pending')
                    status_colors = {
                        'completed': (0, 255, 0),
                        'in_progress': (255, 165, 0),
                        'failed': (255, 0, 0),
                        'pending': (255, 255, 0)
                    }
                    status_color = status_colors.get(status, (255, 255, 255))
                    pygame.draw.circle(self._render_surface, status_color, (panel_x + 5, y_offset + 8), 4)
                    
                    # Objective text - smart wrapping for full visibility
                    obj_desc = obj.get('description', obj.get('obj_type', 'Unknown'))
                    
                    # Smart text wrapping to ensure coordinates are always visible
                    max_line_length = 35  # Characters that fit in the panel width
                    
                    if len(obj_desc) <= max_line_length:
                        # Single line - fits completely
                        obj_surface = pygame.font.Font(None, 16).render(obj_desc, True, (255, 255, 255))
                        self._render_surface.blit(obj_surface, (panel_x + 15, y_offset))
                        y_offset += 18
                    else:
                        # Two lines needed - smart split to preserve coordinates
                        if " at (" in obj_desc:
                            # Split at coordinate boundary to preserve full coordinates
                            parts = obj_desc.split(" at (", 1)
                            main_part = parts[0].strip()
                            coord_part = "at (" + parts[1] if len(parts) > 1 else ""
                            
                            # If main part is too long, break it intelligently
                            if len(main_part) > max_line_length:
                                # Find a good break point (space, comma, etc.)
                                break_point = max_line_length - 3
                                for i in range(break_point, max(0, break_point - 10), -1):
                                    if main_part[i] in [' ', ',', '-', '_']:
                                        break_point = i
                                        break
                                
                                line1 = main_part[:break_point].strip()
                                line2 = main_part[break_point:].strip()
                                if coord_part:
                                    line2 += " " + coord_part
                            else:
                                # Main part fits on first line
                                line1 = main_part
                                line2 = coord_part
                            
                            # Render first line
                            line1_surface = pygame.font.Font(None, 16).render(line1, True, (255, 255, 255))
                            self._render_surface.blit(line1_surface, (panel_x + 15, y_offset))
                            y_offset += 16
                            
                            # Render second line with slight indent and smaller font for coordinates
                            if line2:
                                line2_surface = pygame.font.Font(None, 14).render(line2, True, (200, 200, 200))
                                self._render_surface.blit(line2_surface, (panel_x + 20, y_offset))
                                y_offset += 16
                            else:
                                y_offset += 2  # Small gap if no second line
                        else:
                            # No coordinates - break at word boundary
                            break_point = max_line_length - 3
                            for i in range(break_point, max(0, break_point - 10), -1):
                                if obj_desc[i] in [' ', ',', '-', '_']:
                                    break_point = i
                                    break
                            
                            line1 = obj_desc[:break_point].strip()
                            line2 = obj_desc[break_point:].strip()
                            
                            # Render first line
                            line1_surface = pygame.font.Font(None, 16).render(line1, True, (255, 255, 255))
                            self._render_surface.blit(line1_surface, (panel_x + 15, y_offset))
                            y_offset += 16
                            
                            # Render second line with slight indent
                            if line2:
                                line2_surface = pygame.font.Font(None, 16).render(line2, True, (255, 255, 255))
                                self._render_surface.blit(line2_surface, (panel_x + 20, y_offset))
                                y_offset += 16
                            else:
                                y_offset += 2
                    
                    # Add small gap between objectives
                    y_offset += 3
                    objectives_shown += 1
                    
                    # Render objective on map if position available
                    if 'target_position' in obj and obj['target_position']:
                        target = obj['target_position']
                        # Use proper coordinate scaling: world coords (0-100) to map area (0-500)
                        padding = 10
                        map_x = int(padding + (target['x'] / 100.0) * (500 - 2 * padding))
                        map_y = int(padding + (target['y'] / 100.0) * (600 - 2 * padding))
                        if 0 <= map_x <= 500 and 0 <= map_y <= 600:
                            pygame.draw.circle(self._render_surface, status_color, (map_x, map_y), 6)
                            pygame.draw.circle(self._render_surface, (255, 255, 255), (map_x, map_y), 6, 2)
            else:
                no_obj_surface = pygame.font.Font(None, 16).render("No visible objectives", True, (200, 200, 200))
                self._render_surface.blit(no_obj_surface, (panel_x, y_offset))
                y_offset += 18
            
            # Hidden objectives indicator - only if space allows
            hidden_count = state.get('hidden_objectives_count', 0)
            total_objectives = state.get('total_objectives_count', len(objectives))
            if hidden_count > 0 and y_offset + 35 <= panel_bottom:
                y_offset += 10
                hidden_text = f"🔍 {hidden_count} hidden objectives"
                hidden_surface = pygame.font.Font(None, 16).render(hidden_text, True, (128, 128, 128))
                self._render_surface.blit(hidden_surface, (panel_x, y_offset))
                y_offset += 15
                
                hint_text = "Explore to discover!"
                hint_surface = pygame.font.Font(None, 16).render(hint_text, True, (128, 128, 128))
                self._render_surface.blit(hint_surface, (panel_x, y_offset))
                y_offset += 15
            elif total_objectives > len(objectives) and y_offset + 20 <= panel_bottom:
                y_offset += 10
                unknown_text = f"? More objectives may exist"
                unknown_surface = pygame.font.Font(None, 16).render(unknown_text, True, (128, 128, 128))
                self._render_surface.blit(unknown_surface, (panel_x, y_offset))
            
            # Discovery hints - only if space allows
            if state.get('discovery_hints') and y_offset + 40 <= panel_bottom:
                y_offset += 15
                hint_title = self._font.render("Discovery Hints", True, (255, 255, 0))
                self._render_surface.blit(hint_title, (panel_x, y_offset))
                y_offset += 20
                
                for hint in state['discovery_hints'][:2]:  # Show first 2 hints
                    if y_offset + 15 > panel_bottom:
                        break
                    if len(hint) > 25:
                        hint = hint[:22] + "..."
                    hint_surface = pygame.font.Font(None, 16).render(hint, True, (255, 255, 128))
                    self._render_surface.blit(hint_surface, (panel_x, y_offset))
                    y_offset += 15
            
            # Convert surface to numpy array
            array = pygame.surfarray.array3d(self._render_surface)
            return array.transpose(1, 0, 2)  # Convert to (height, width, 3)
            
        except Exception as e:
            print(f"Basic rendering error: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((600, 800, 3), dtype=np.uint8)

    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human' and self.enable_visual:
            try:
                if hasattr(self, '_use_gui_rendering') and self._use_gui_rendering:
                    # Use GUI rendering
                    if not hasattr(self._gui_renderer, '_display'):
                        self._gui_renderer._display = pygame.display.set_mode((800, 600))
                        pygame.display.set_caption("Coop Command Game")
                    
                    self._render_to_array()  # This updates the render surface
                    self._gui_renderer._display.blit(self._gui_renderer.screen, (0, 0))
                    pygame.display.flip()
                else:
                    # Basic pygame display rendering
                    if not hasattr(self, '_display'):
                        self._display = pygame.display.set_mode(self._display_size)
                        pygame.display.set_caption("Coop Command Game")
                    
                    self._render_to_array()  # This updates the render surface
                    self._display.blit(self._render_surface, (0, 0))
                    pygame.display.flip()
                
            except Exception as e:
                print(f"Display rendering error: {e}")

    def close(self):
        """Close the environment"""
        # Clean up video recording if active
        if self.enable_video_recording and self.video_recorder and self.video_recorder.recording:
            try:
                success, message = self.video_recorder.stop_recording_and_save()
                if success:
                    print(f"🎥 Final save: {message}")
                else:
                    print(f"⚠️ Final video save failed: {message}")
            except Exception as e:
                print(f"Video cleanup error: {e}")
        
        # Clean up frame buffer for video input mode or enhanced video
        if hasattr(self, 'frame_buffer'):
            self.frame_buffer = []
        
        if hasattr(self, '_game'):
            self._game.shutdown()
        
        if self.enable_visual:
            # Cleanup GUI renderer if available
            if hasattr(self, '_gui_renderer'):
                try:
                    if hasattr(self._gui_renderer, 'env') and self._gui_renderer.env:
                        self._gui_renderer.env.shutdown()
                    pygame.quit()
                except:
                    pass
            else:
                try:
                    pygame.quit()
                except:
                    pass

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current game for evaluation purposes"""
        return self._game.get_evaluation_metrics()

    def capture_screen(self) -> Tuple[bool, str]:
        """Capture screenshot as base64 string"""
        if not self.enable_visual:
            return False, ""
        
        try:
            image_array = self._render_to_array()
            image = Image.fromarray(image_array)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return True, image_base64
        except Exception as e:
            print(f"Screenshot capture error: {e}")
            return False, ""
    
    def start_video_recording(self, session_name: str = None) -> bool:
        """Manually start video recording."""
        if not self.enable_video_recording:
            print("⚠️ Video recording not enabled")
            return False
        
        active_recorder = self.enhanced_video_recorder if self.enhanced_video_recorder else self.video_recorder
        if not active_recorder:
            print("⚠️ No video recorder available")
            return False
        
        return active_recorder.start_recording(session_name)
    
    def stop_video_recording(self, output_path: str = None) -> Tuple[bool, str]:
        """Manually stop video recording and save."""
        if not self.enable_video_recording:
            return False, "Video recording not enabled"
        
        active_recorder = self.enhanced_video_recorder if self.enhanced_video_recorder else self.video_recorder
        if not active_recorder:
            return False, "No video recorder available"
        
        return active_recorder.stop_recording_and_save(output_path)
    
    def get_video_status(self) -> Dict[str, Any]:
        """Get current video recording status."""
        if not self.enable_video_recording:
            return {"recording": False, "available": False, "enhanced": False}
        
        active_recorder = self.enhanced_video_recorder if self.enhanced_video_recorder else self.video_recorder
        if not active_recorder:
            return {"recording": False, "available": False, "enhanced": False}
        
        if self.enhanced_video_recorder:
            status = self.enhanced_video_recorder.get_current_status()
            status["available"] = True
            status["enhanced"] = True
        else:
            status = self.video_recorder.get_current_video_data()
            status["available"] = True
            status["enhanced"] = False
        
        return status
    
    def save_current_session_video(self, output_path: str = None) -> Tuple[bool, str]:
        """Save current video session without stopping recording."""
        if not self.enable_video_recording or not self.video_recorder or not self.video_recorder.recording:
            return False, "No active video recording"
        
        try:
            # Create a copy of current recording data
            temp_recorder = VideoRecorder(fps=self.video_fps)
            temp_recorder.frames = self.video_recorder.frames.copy()
            temp_recorder.audio_events = self.video_recorder.audio_events.copy()
            temp_recorder.frame_timestamps = self.video_recorder.frame_timestamps.copy()
            temp_recorder.recording = False  # Mark as stopped for saving
            # Save the copy
            if output_path is None:
                output_path = self.video_recorder.output_dir / f"{self.video_recorder.session_name}_partial.mp4"
            print(f"临时保存路径: {output_path}")
            return temp_recorder._save_with_cv2(Path(output_path)) if temp_recorder.use_cv2 else temp_recorder._save_with_moviepy(Path(output_path))
            
        except Exception as e:
            return False, f"Partial save error: {e}"
    
    def _generate_step_audio_file(self, state: dict, step_count: int) -> Optional[bytes]:
        """生成step的音频文件"""
        try:
            # 生成文字内容
            text_guidance = self._generate_enhanced_audio_guidance(state, step_count)
            
            if not text_guidance:
                return None
            
            # 使用TTS转换为音频
            return self._text_to_speech(text_guidance)
            
        except Exception as e:
            print(f"音频文件生成错误: {e}")
            return None
    
    def _text_to_speech(self, text: str) -> Optional[bytes]:
        """将文字转换为语音"""
        try:
            # 方案1：使用gTTS
            try:
                from gtts import gTTS
                import io
                
                tts = gTTS(text=text, lang='en', slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                return audio_buffer.read()
            except ImportError:
                print("gTTS not available, trying pyttsx3...")
                pass
            
            # 如果所有TTS都不可用，返回None
            print("⚠️ No TTS engine available. Audio will be provided as text.")
            return None
        
        except Exception as e:
            print(f"TTS转换失败: {e}")
            return None

    def _generate_enhanced_audio_guidance(self, state: dict, step_count: int) -> str:
        """生成增强的音频战术指导"""
        try:
            guidance_components = []
            
            # 1. 战术评估
            tactical_guidance = self._generate_tactical_audio_guidance(state, step_count)
            if tactical_guidance:
                guidance_components.append(tactical_guidance)
            
            # 2. 移动和位置建议
            movement_guidance = self._generate_movement_audio_guidance(state)
            if movement_guidance:
                guidance_components.append(movement_guidance)
            
            
            # 3. 资源管理建议
            resource_guidance = self._generate_resource_audio_guidance(state)
            if resource_guidance:
                guidance_components.append(resource_guidance)
            
            # 4. 情报和发现提示
            intelligence_guidance = self._generate_intelligence_audio_guidance(state)
            if intelligence_guidance:
                guidance_components.append(intelligence_guidance)
            
            # 5. 协作建议
            coordination_guidance = self._generate_coordination_audio_guidance(state)
            if coordination_guidance:
                guidance_components.append(coordination_guidance)
            
            # 6. 风险评估
            risk_guidance = self._generate_risk_assessment_audio(state)
            if risk_guidance:
                guidance_components.append(risk_guidance)
            
            # 组合所有指导内容
            if guidance_components:
                intro = f"Step {step_count} tactical guidance: "
                full_guidance = intro + " ... ".join(guidance_components) + " ... End guidance."
                
                # 🎯 修改：优化音频长度，适合TTS
                # 控制长度，避免过长（TTS有时间限制）
                if len(full_guidance) > 400:  # 增加长度限制
                    # 优先保留高优先级信息
                    priority_components = [tactical_guidance, movement_guidance, risk_guidance]
                    priority_components = [comp for comp in priority_components if comp and len(comp) < 100]
                    if priority_components:
                        full_guidance = intro + " ".join(priority_components) + " End priority guidance."
                    else:
                        # 如果优先级组件也太长，只保留最重要的
                        if tactical_guidance and len(tactical_guidance) < 150:
                            full_guidance = intro + tactical_guidance + " End guidance."
                        else:
                            full_guidance = f"Step {step_count} guidance: Mission status updated. Continue operations."
                
                return full_guidance
            
            return f"Step {step_count} guidance: All systems nominal. Proceed with mission."
            
        except Exception as e:
            print(f"音频指导生成错误: {e}")
            return f"Step {step_count} guidance: Tactical analysis in progress."

    def _generate_tactical_audio_guidance(self, state: dict, step_count: int) -> str:
        """生成战术音频指导"""
        try:
            guidance_parts = []
            
            # 威胁级别评估
            threat_level = self._assess_current_threat_level(state)
            if threat_level == "high":
                guidance_parts.append("WARNING: High threat environment detected. Recommend defensive positioning.")
            elif threat_level == "medium":
                guidance_parts.append("CAUTION: Moderate threats present. Maintain tactical awareness.")
            elif threat_level == "low":
                guidance_parts.append("CLEAR: Low threat level. Good opportunity for objective advancement.")
            
            # 时间压力提醒
            urgency = self._calculate_time_urgency(state)
            if urgency > 0.8:
                guidance_parts.append("URGENT: Time running critical. Prioritize high-value objectives.")
            elif urgency > 0.6:
                guidance_parts.append("NOTICE: Time pressure increasing. Consider acceleration of mission pace.")
            
            return " ".join(guidance_parts)
            
        except Exception as e:
            print(f"战术指导生成错误: {e}")
            return ""

    def _generate_movement_audio_guidance(self, state: dict) -> str:
        """生成移动相关的音频指导"""
        try:
            guidance_parts = []
            
            # 队形分析
            formation_status = self._analyze_team_formation(state)
            if formation_status.get("too_clustered"):
                guidance_parts.append("FORMATION: Spread out to reduce risk.")
            elif formation_status.get("too_spread"):
                guidance_parts.append("FORMATION: Regroup for support.")
            
            # 地形优势提醒
            terrain_advantages = self._identify_terrain_advantages(state)
            if terrain_advantages:
                guidance_parts.append(f"TERRAIN: Move to {terrain_advantages.get('best_position', 'unknown')}.")
            
            return " ".join(guidance_parts)
            
        except Exception as e:
            print(f"移动指导生成错误: {e}")
            return ""

    def _generate_resource_audio_guidance(self, state: dict) -> str:
        """生成资源管理音频指导"""
        try:
            guidance_parts = []
            
            # 健康状态汇总
            health_status = self._analyze_team_health_status(state)
            if health_status.get("critical_members"):
                members = ", ".join(health_status["critical_members"][:2])
                guidance_parts.append(f"MEDICAL CRITICAL: {members} require immediate attention.")
            elif health_status.get("injured_members"):
                members = ", ".join(health_status["injured_members"][:2])
                guidance_parts.append(f"MEDICAL: {members} showing reduced combat effectiveness.")
            
            # 特殊能力可用性
            special_abilities = self._check_special_abilities_available(state)
            if special_abilities:
                ability_list = ", ".join(special_abilities[:2])
                guidance_parts.append(f"CAPABILITIES: {ability_list} ready for deployment.")
            
            return " ".join(guidance_parts)
            
        except Exception as e:
            print(f"资源指导生成错误: {e}")
            return ""

    def _generate_intelligence_audio_guidance(self, state: dict) -> str:
        """生成情报相关的音频指导"""
        try:
            guidance_parts = []
            
            # 目标发现概率
            discovery_probability = self._calculate_discovery_probability(state)
            if discovery_probability > 0.7:
                guidance_parts.append(f"DISCOVERY: High probability objective zone identified. Success rate {discovery_probability*100:.0f}%.")
            elif discovery_probability > 0.4:
                guidance_parts.append(f"DISCOVERY: Potential objective area detected. Investigate recommended.")
            
            # 未探索区域提醒
            unexplored_areas = self._identify_unexplored_high_value_areas(state)
            if unexplored_areas:
                guidance_parts.append(f"INTEL: High-value unexplored sector detected. Recommend reconnaissance.")
            
            return " ".join(guidance_parts)
            
        except Exception as e:
            print(f"情报指导生成错误: {e}")
            return ""

    def _generate_coordination_audio_guidance(self, state: dict) -> str:
        """生成协作相关的音频指导"""
        try:
            guidance_parts = []
            
            # 团队协作机会
            cooperation_opportunities = self._identify_cooperation_opportunities(state)
            if cooperation_opportunities:
                op = cooperation_opportunities[0]
                guidance_parts.append(f"COORDINATION: Joint operation opportunity detected. Estimated success rate {op.get('success_rate', 0)*100:.0f}%.")
            
            # 支援可用性
            support_analysis = self._analyze_support_availability(state)
            if support_analysis.get("fire_support_available"):
                guidance_parts.append("SUPPORT: Fire support available for target designation.")
            if support_analysis.get("overwatch_positions"):
                guidance_parts.append("OVERWATCH: Optimal sniper position identified for area coverage.")
            
            return " ".join(guidance_parts)
            
        except Exception as e:
            print(f"协作指导生成错误: {e}")
            return ""

    def _generate_risk_assessment_audio(self, state: dict) -> str:
        """生成风险评估音频"""
        try:
            guidance_parts = []
            
            # 即时风险警告
            immediate_risks = self._assess_immediate_risks(state)
            for risk in immediate_risks[:2]:  # 最多报告2个最高风险
                if risk.get("severity") == "critical":
                    guidance_parts.append(f"DANGER: {risk.get('description', 'Unknown threat')} poses immediate threat.")
                elif risk.get("severity") == "high":
                    guidance_parts.append(f"RISK: {risk.get('description', 'Potential hazard')} detected. Exercise caution.")
            
            # 预测性风险
            predicted_risks = self._predict_future_risks(state)
            if predicted_risks:
                next_risk = predicted_risks[0]
                guidance_parts.append(f"FORECAST: {next_risk.get('description', 'Future risk')} anticipated in {next_risk.get('estimated_turns', 'unknown')} moves.")
            
            return " ".join(guidance_parts)
            
        except Exception as e:
            print(f"风险评估生成错误: {e}")
            return ""

    # 辅助分析方法
    def _assess_current_threat_level(self, state: dict) -> str:
        """评估当前威胁级别"""
        try:
            # 简化的威胁评估逻辑
            team_health = []
            team_members = state.get("team_status", {}).get("members", {})
            
            for member_id, member_info in team_members.items():
                normalized_info = self._normalize_team_member_info(member_info, member_id)
                team_health.append(normalized_info["health"])
            
            if team_health:
                avg_health = sum(team_health) / len(team_health)
                if avg_health < 50:
                    return "high"
                elif avg_health < 75:
                    return "medium"
                else:
                    return "low"
            
            return "medium"
            
        except Exception as e:
            print(f"威胁级别评估错误: {e}")
            return "medium"

    def _calculate_time_urgency(self, state: dict) -> float:
        """计算时间紧迫性 (0.0-1.0)"""
        try:
            current_round = state.get("current_round", 0)
            max_rounds = state.get("max_rounds", 50)
            
            if max_rounds > 0:
                return min(1.0, current_round / max_rounds)
            return 0.5
            
        except Exception as e:
            print(f"时间紧迫性计算错误: {e}")
            return 0.5

    def _analyze_team_formation(self, state: dict) -> Dict:
        """分析团队队形"""
        try:
            team_members = state.get("team_status", {}).get("members", {})
            positions = []
            
            for member_id, member_info in team_members.items():
                normalized_info = self._normalize_team_member_info(member_info, member_id)
                pos = normalized_info["position"]
                positions.append((pos["x"], pos["y"]))
            
            if len(positions) < 2:
                return {"optimal": True}
            
            # 计算成员间距离
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    distances.append(dist)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                if avg_distance < 15:
                    return {"too_clustered": True}
                elif avg_distance > 50:
                    return {"too_spread": True}
                else:
                    return {"optimal": True}
            
            return {"optimal": True}
            
        except Exception as e:
            print(f"队形分析错误: {e}")
            return {"optimal": True}

    def _identify_terrain_advantages(self, state: dict) -> Dict:
        """识别地形优势"""
        try:
            # 简化的地形分析
            return {"best_position": "North-East sector"}
            
        except Exception as e:
            print(f"地形优势分析错误: {e}")
            return {}

    def _analyze_team_health_status(self, state: dict) -> Dict:
        """分析团队健康状态"""
        try:
            team_members = state.get("team_status", {}).get("members", {})
            critical_members = []
            injured_members = []
            
            for member_id, member_info in team_members.items():
                normalized_info = self._normalize_team_member_info(member_info, member_id)
                health = normalized_info["health"]
                name = normalized_info["name"]
                
                if health < 25:
                    critical_members.append(name)
                elif health < 60:
                    injured_members.append(name)
            
            return {
                "critical_members": critical_members,
                "injured_members": injured_members
            }
            
        except Exception as e:
            print(f"健康状态分析错误: {e}")
            return {}

    def _check_special_abilities_available(self, state: dict) -> List[str]:
        """检查可用的特殊能力"""
        try:
            abilities = []
            team_members = state.get("team_status", {}).get("members", {})
            
            for member_id, member_info in team_members.items():
                normalized_info = self._normalize_team_member_info(member_info, member_id)
                role = normalized_info["role"]
                health = normalized_info["health"]
                
                if health > 50:  # 只有健康状况良好的成员能使用特殊能力
                    if role == "scout":
                        abilities.append("reconnaissance")
                    elif role == "heavy":
                        abilities.append("fire support")
                    elif role == "medic":
                        abilities.append("medical treatment")
                    elif role == "engineer":
                        abilities.append("equipment repair")
                    elif role == "sniper":
                        abilities.append("overwatch")
            
            return abilities
            
        except Exception as e:
            print(f"特殊能力检查错误: {e}")
            return []

    def _calculate_discovery_probability(self, state: dict) -> float:
        """计算发现目标的概率"""
        try:
            # 简化的发现概率计算
            completed_objectives = len(state.get("completed_objectives", []))
            total_objectives = len(state.get("mission_objectives", []))
            
            if total_objectives > 0:
                completion_rate = completed_objectives / total_objectives
                # 根据完成率调整发现新目标的概率
                return max(0.3, 0.8 - completion_rate * 0.5)
            
            return 0.5
            
        except Exception as e:
            print(f"发现概率计算错误: {e}")
            return 0.5

    def _identify_unexplored_high_value_areas(self, state: dict) -> List[Dict]:
        """识别未探索的高价值区域"""
        try:
            # 简化实现 - 实际应该基于地图探索数据
            return [{"area": "North sector", "value": "high"}]
            
        except Exception as e:
            print(f"未探索区域识别错误: {e}")
            return []

    def _identify_cooperation_opportunities(self, state: dict) -> List[Dict]:
        """识别协作机会"""
        try:
            opportunities = []
            team_members = state.get("team_status", {}).get("members", {})
            
            # 检查是否有多个成员可以协作
            available_members = []
            for member_id, member_info in team_members.items():
                normalized_info = self._normalize_team_member_info(member_info, member_id)
                if normalized_info["health"] > 60:
                    available_members.append(normalized_info)
            
            if len(available_members) >= 2:
                opportunities.append({
                    "description": "Multi-member coordinated assault",
                    "success_rate": 0.75
                })
            
            return opportunities
            
        except Exception as e:
            print(f"协作机会识别错误: {e}")
            return []

    def _analyze_support_availability(self, state: dict) -> Dict:
        """分析支援可用性"""
        try:
            team_members = state.get("team_status", {}).get("members", {})
            support_info = {
                "fire_support_available": False,
                "overwatch_positions": False
            }
            
            for member_id, member_info in team_members.items():
                normalized_info = self._normalize_team_member_info(member_info, member_id)
                role = normalized_info["role"]
                health = normalized_info["health"]
                
                if health > 50:
                    if role == "heavy":
                        support_info["fire_support_available"] = True
                    elif role == "sniper":
                        support_info["overwatch_positions"] = True
            
            return support_info
            
        except Exception as e:
            print(f"支援分析错误: {e}")
            return {}

    def _assess_immediate_risks(self, state: dict) -> List[Dict]:
        """评估即时风险"""
        try:
            risks = []
            team_members = state.get("team_status", {}).get("members", {})
            
            # 检查团队健康状况风险
            critical_health_count = 0
            for member_id, member_info in team_members.items():
                normalized_info = self._normalize_team_member_info(member_info, member_id)
                if normalized_info["health"] < 30:
                    critical_health_count += 1
            
            if critical_health_count > 0:
                risks.append({
                    "description": f"Team member casualty risk",
                    "severity": "high" if critical_health_count > 1 else "medium"
                })
            
            # 检查时间压力风险
            urgency = self._calculate_time_urgency(state)
            if urgency > 0.8:
                risks.append({
                    "description": "Mission time expiration",
                    "severity": "critical"
                })
            
            return risks
            
        except Exception as e:
            print(f"即时风险评估错误: {e}")
            return []

    def _predict_future_risks(self, state: dict) -> List[Dict]:
        """预测未来风险"""
        try:
            risks = []
            
            # 预测基于当前趋势的风险
            current_round = state.get("current_round", 0)
            max_rounds = state.get("max_rounds", 50)
            
            if current_round > max_rounds * 0.7:
                remaining_rounds = max_rounds - current_round
                risks.append({
                    "description": "Mission timeout approaching",
                    "estimated_turns": remaining_rounds
                })
            
            return risks
            
        except Exception as e:
            print(f"未来风险预测错误: {e}")
            return []