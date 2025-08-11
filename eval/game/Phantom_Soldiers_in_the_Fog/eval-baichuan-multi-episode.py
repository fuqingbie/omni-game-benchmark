import json
import logging
import os
import re
import time
import base64
import numpy as np
import argparse
import requests
import traceback

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gym_wrapper import CoopCommandGymEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baichuan_eval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

FASTAPI_BASE_URL = ""

class MultiProviderEvaluator:
    """Multi-provider evaluator for cooperative command game supporting Baichuan."""
    
    def __init__(self, difficulty: str = "normal", seed_index: int = 0, 
                 max_rounds: Optional[int] = 100, enable_stream: bool = True,
                 save_media: bool = True, deterministic_commands: bool = True,
                 api_provider: str = "baichuan", model_name: Optional[str] = None,
                 input_mode: str = "video", include_vector_text: bool = True,
                 enhanced_video: bool = False, video_fps: float = 0.5, 
                 audio_duration_per_frame: float = 3.0, num_episodes: int = 10):
        """
        Initialize the multi-provider evaluator.
        """
        self.difficulty = difficulty
        self.seed_index = seed_index
        self.max_rounds = max_rounds
        self.enable_stream = enable_stream
        self.save_media = save_media
        self.deterministic_commands = deterministic_commands
        self.api_provider = api_provider.lower()
        self.input_mode = input_mode.lower()
        self.include_vector_text = include_vector_text
        self.enhanced_video = enhanced_video
        self.video_fps = video_fps
        self.audio_duration_per_frame = audio_duration_per_frame
        self.num_episodes = num_episodes  # 新增：episode数量
        
        # Validate input mode
        if self.input_mode not in ["image_audio", "video"]:
            raise ValueError(f"Unsupported input mode: {input_mode}. Choose from: ['image_audio', 'video']")
        
        # 百川模型配置
        self.model_name = model_name or "baichuan"
        
        # 初始化session和session_id用于百川模型
        self.session = requests.Session()
        self.session_id = None
        
        # Create output directory for this evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_root = "outputs"
        os.makedirs(output_root, exist_ok=True)
        self.output_dir = Path(f"{output_root}/{self.api_provider}_eval_{difficulty}_seed{seed_index}_{num_episodes}ep_{timestamp}")
        if self.save_media:
            self.output_dir.mkdir(exist_ok=True)
            self.images_dir = self.output_dir / "images"
            self.audio_dir = self.output_dir / "audio"
            self.videos_dir = self.output_dir / "videos"
            self.responses_dir = self.output_dir / "responses"
            self.images_dir.mkdir(exist_ok=True)
            self.audio_dir.mkdir(exist_ok=True)
            self.videos_dir.mkdir(exist_ok=True)
            self.responses_dir.mkdir(exist_ok=True)
        
        # 新增：为每个episode创建子目录
        self.episode_dirs = {}
        if self.save_media:
            for ep in range(num_episodes):
                ep_dir = self.output_dir / f"episode_{ep:02d}"
                ep_dir.mkdir(exist_ok=True)
                self.episode_dirs[ep] = {
                    "root": ep_dir,
                    "images": ep_dir / "images",
                    "audio": ep_dir / "audio", 
                    "videos": ep_dir / "videos",
                    "responses": ep_dir / "responses"
                }
                # 创建子目录
                for subdir in ["images", "audio", "videos", "responses"]:
                    self.episode_dirs[ep][subdir].mkdir(exist_ok=True)
        
        # 不在初始化时创建环境，将在每个episode开始时创建
        self.env = None
        
        # Results tracking - 修改为支持多episode
        self.results = {
            "config": {
                "difficulty": difficulty,
                "seed_index": seed_index,
                "max_rounds": max_rounds,
                "api_provider": self.api_provider,
                "model": self.model_name,
                "input_mode": self.input_mode,
                "include_vector_text": self.include_vector_text,
                "enhanced_video": self.enhanced_video,
                "video_fps": self.video_fps,
                "audio_duration_per_frame": self.audio_duration_per_frame,
                "vision_support": True,
                "num_episodes": num_episodes,  # 新增
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(self.output_dir) if self.save_media else None
            },
            "episodes": [],  # 修改：存储每个episode的结果
            "summary_stats": {},  # 新增：汇总统计
            "media_files": {
                "images": [],
                "audio": [],
                "videos": [],
                "responses": []
            },
            "command_compliance": {
                "total_turns": 0,
                "valid_single_commands": 0,
                "multiple_command_violations": 0,
                "no_command_found": 0,
                "compliance_rate": 0.0
            }
        }
        
        # Command types for reference
        self.command_types = ["move", "attack", "defend", "recon", "status"]
        
        # Store member list for multi-member commands
        self._last_member_list = []
        
        # Enhanced system prompt
        command_reliability_note = """
⚙️ COMMAND EXECUTION:
- Commands execute deterministically - all valid commands will succeed
- Focus on strategic positioning and optimal task assignment
- No need to account for random command failures

""" if deterministic_commands else """
⚙️ COMMAND EXECUTION:
- Commands may randomly fail based on team member capability and status
- Scout: 85% success rate, Heavy: 75%, Medic: 90%, Engineer: 80%, Sniper: 70%
- Injured or low-health members have reduced success rates
- Plan for potential command failures and have backup strategies

"""
        
        # Build media description based on input mode
        if self.input_mode == "video":
            if self.enhanced_video:
                actual_fps = self.video_fps if self.enhanced_video else 1
                media_description = f"""- Video: Visual sequence with integrated audio (FPS: {actual_fps}, Audio per frame: {self.audio_duration_per_frame}s)
  Shows game state progression and team member movements with synchronized voice communications"""
            else:
                media_description = "- Video: Visual sequence showing game state progression and team member movements over time"
        else:
            media_description = """- Audio: Voice reports from team members about discoveries, status changes
- Image: Visual representation of current game state"""

        # Build information sources section
        if self.include_vector_text:
            info_sources = f"""📊 INFORMATION PROVIDED:
- Vector: Team member states (health, status, position) + global info (rounds remaining, normalized score)
{media_description}
- Discovery hints: Clues about nearby hidden objectives"""
        else:
            info_sources = f"""📊 INFORMATION PROVIDED:
{media_description}
- Discovery hints: Clues about nearby hidden objectives

⚠️ NOTE: Vector information is available visually - interpret team states from the visual input."""

        self.system_prompt = f"""
You are commanding a military team in a cooperative mission. You MUST provide EXACTLY ONE command per turn.

❌ FORBIDDEN: Multiple commands like "COMMAND: 0 move 20 30" AND "COMMAND: 1 recon 40 50"
✅ CORRECT: Only one command like "COMMAND: 0 move 20 30"

If you provide more than one command, the system will ERROR and use a default command instead.

KEY GAME MECHANICS:

{command_reliability_note}

🎯 HIDDEN OBJECTIVES:
- Some objectives are HIDDEN and not visible initially
- You must EXPLORE different areas to discover hidden objectives
- Scout team members have higher discovery probability (80% vs 40%)
- Send scouts to unexplored areas to find new objectives
- Discovery hints may indicate "unusual activity" in areas with hidden objectives

⚠️ MOVEMENT UNCERTAINTY:
- Team members DO NOT move to exact coordinates you specify
- Movement has ERROR based on:
  * Role precision (Scout: low error, Heavy: high error)
  * Health status (injured = more error)
  * Movement distance (longer moves = more error)
- Expect actual positions to deviate from your targets
- Plan for imprecise movement in your strategy

{info_sources}

🎮 STRATEGIC CONSIDERATIONS:
- Balance exploration (finding hidden objectives) vs completion (finishing known objectives)
- Use scouts for exploration and discovery
- Account for movement errors in positioning
- Monitor team health and status for optimal assignment
- Hidden objectives may have high score values - worth discovering!

🚨 COMMAND FORMAT - PROVIDE EXACTLY ONE OF THESE:

**Individual Command (one member):**
COMMAND: [member_id] [action] [x] [y]

**Team Command (all members together):**
COMMAND: all [action] [x] [y]

**Multi-member Command (specific members together):**
COMMAND: 0,1,2 [action] [x] [y]

**Available Actions:** move, attack, defend, recon, status
**Coordinates:** x, y: 0-100 (actual position will vary due to movement error)

EXAMPLES OF CORRECT RESPONSES:
✅ "Based on the current situation, I'll send the scout to explore. COMMAND: 0 recon 25 30"
✅ "The team should move together to the objective. COMMAND: all move 45 20"
✅ "Two scouts should explore this area. COMMAND: 0,1 recon 70 80"

EXAMPLES OF INCORRECT RESPONSES (WILL CAUSE ERRORS):
❌ "COMMAND: 0 move 25 30" followed by "COMMAND: 1 recon 45 20"
❌ Multiple command lines in any form
❌ Suggesting multiple commands for "efficient coordination"

🚨 FINAL REMINDER: ONE COMMAND ONLY! 🚨
- Analyze the situation thoroughly
- Choose the SINGLE most important action
- Provide exactly ONE command
- Plan step-by-step across multiple turns, not all at once

Provide your strategic analysis, then end with exactly ONE command.
"""

    def encode_file_to_base64(self, file_path: str) -> str:
        """编码文件为base64字符串"""
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"文件编码失败 {file_path}: {e}")
            return ""

    def _get_state_description(self, observation) -> str:
        """Create human-readable state description from observation."""
        try:
            # Handle both dict and array observation formats
            if isinstance(observation, dict):
                # Multi-modal observation format
                vector_obs = observation['vector']
                # Handle audio data safely - check if it's JSON or base64
                audio_data = []
                if isinstance(observation.get('audio'), str):
                    audio_str = observation.get('audio', '[]')
                    if audio_str.startswith('{') or audio_str.startswith('['):
                        try:
                            audio_data = json.loads(audio_str)
                        except json.JSONDecodeError:
                            audio_data = []
                    # If it's base64 encoded audio, skip JSON parsing
                    else:
                        audio_data = []
                has_image = 'image' in observation
            else:
                # Simple vector observation format
                vector_obs = observation
                audio_data = []
                has_image = False
            
            # Ensure vector_obs is a numpy array
            if hasattr(vector_obs, 'shape'):
                vector_obs = vector_obs.flatten()  # Flatten if multi-dimensional
            
            # Parse vector observation
            num_members = (len(vector_obs) - 2) // 4
            description = [f"Team size: {num_members}"]
            
            # Member states
            for i in range(num_members):
                base_idx = i * 4
                health = float(vector_obs[base_idx])
                status_code = int(float(vector_obs[base_idx + 1]))
                x, y = float(vector_obs[base_idx + 2]), float(vector_obs[base_idx + 3])
                
                status_names = ["idle", "moving", "attacking", "defending", "recon", "dead", "injured"]
                status = status_names[status_code] if status_code < len(status_names) else "unknown"
                
                description.append(f"Member {i}: {health:.0f}% health, {status}, at ({x:.0f},{y:.0f})")
            
            # Global state
            rounds_remaining = int(float(vector_obs[-2]))
            score_normalized = float(vector_obs[-1])
            description.append(f"Rounds remaining: {rounds_remaining}")
            description.append(f"Score: {score_normalized:.1f}/100")
            
            # Audio events
            if audio_data:
                description.append(f"Audio: {', '.join(str(msg) for msg in audio_data)}")
            
            # Visual info
            if has_image:
                description.append("Visual: Game state image available")
            
            # Check for video in observation
            if isinstance(observation, dict) and observation.get('video') is not None:
                description.append("Video: Game state video sequence available")
            
            return "\n".join(description)
            
        except Exception as e:
            logger.error(f"Error creating state description: {e}")
            logger.error(f"Observation type: {type(observation)}")
            if hasattr(observation, 'shape'):
                logger.error(f"Observation shape: {observation.shape}")
            elif isinstance(observation, dict):
                logger.error(f"Observation keys: {list(observation.keys())}")
            return "State parsing failed"


    def _build_messages(self, observation, step: int, video_path: Optional[str] = None, 
                       audio_path: Optional[str] = None, image_path: Optional[str] = None) -> List[Dict]:
        """构建包含多模态内容的消息"""
        try:
            # 构建基础文本内容
            if self.include_vector_text:
                state_desc = self._get_state_description(observation)
                base_text = f"""Current game state:
{state_desc}

🚨🚨🚨 CRITICAL REMINDER: EXACTLY ONE COMMAND ONLY! 🚨🚨🚨

You MUST provide exactly ONE command in your response. Multiple commands will cause SYSTEM ERRORS!

❌ DO NOT DO THIS: Provide multiple "COMMAND:" lines
✅ DO THIS: Provide exactly one "COMMAND:" line

Choose the SINGLE most important action for this turn. You can plan additional moves for future turns.

Available inputs:
- Vector: Team member states (health, status, position) + global info (rounds remaining, normalized score)
- Video: Visual sequence showing game state progression
- Audio: Tactical guidance and team communications
- Discovery hints: Clues about nearby hidden objectives

Analyze the situation and provide your ONE command."""
            else:
                base_text = """🚨🚨🚨 CRITICAL REMINDER: EXACTLY ONE COMMAND ONLY! 🚨🚨🚨

You MUST provide exactly ONE command in your response. Multiple commands will cause SYSTEM ERRORS!

❌ DO NOT DO THIS: Provide multiple "COMMAND:" lines
✅ DO THIS: Provide exactly one "COMMAND:" line

Choose the SINGLE most important action for this turn. You can plan additional moves for future turns.

Available inputs:
- Video: Visual sequence showing game state progression
- Audio: Tactical guidance and team communications
- Discovery hints: Clues about nearby hidden objectives

⚠️ NOTE: Vector information is available visually - interpret team states from the visual input.

Analyze the situation and provide your ONE command."""

            content = [{
                "type": "text",
                "text": base_text
            }]
            
            # 添加视频 - 直接使用base64数据而不是文件路径
            if isinstance(observation, dict) and observation.get('video'):
                video_data = observation['video']
                if isinstance(video_data, str) and video_data:
                    try:
                        # 检测视频格式
                        video_bytes_test = base64.b64decode(video_data[:100])
                        is_actual_video = (video_bytes_test[4:12] == b'ftypmp4' or 
                                         video_bytes_test[4:12] == b'ftypisom' or
                                         video_bytes_test[4:8] == b'ftyp')
                        
                        if is_actual_video:
                            content.append({
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{video_data}"
                                }
                            })
                            logger.info("✅ 视频数据已添加到消息中 (MP4)")
                        else:
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{video_data}"
                                }
                            })
                            logger.info("✅ 图像数据已添加到消息中 (JPEG)")
                    except Exception as e:
                        logger.warning(f"视频数据处理失败: {e}")
            
            # 添加图像（如果有且没有视频）
            elif isinstance(observation, dict) and observation.get('image') is not None:
                image_data = observation['image']
                if isinstance(image_data, str):
                    image_base64 = image_data
                elif hasattr(image_data, 'shape'):
                    from PIL import Image
                    import io
                    
                    if len(image_data.shape) == 3:
                        image_pil = Image.fromarray(image_data.astype(np.uint8))
                        buffer = io.BytesIO()
                        image_pil.save(buffer, format='JPEG', quality=85)
                        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    else:
                        image_base64 = None
                else:
                    image_base64 = None
                
                if image_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
                    logger.info("✅ 图像数据已添加到消息中")
            
            # 构建消息列表
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]
            
            # 🎯 添加音频数据 - 直接处理base64音频数据
            if isinstance(observation, dict) and observation.get('audio'):
                audio_data = observation['audio']
                
                try:
                    if isinstance(audio_data, str):
                        # 尝试作为base64音频数据处理
                        if not audio_data.startswith('{') and len(audio_data) > 1000:
                            try:
                                # 验证是否为有效的base64音频
                                test_decode = base64.b64decode(audio_data[:100])
                                
                                # 添加音频到消息中
                                messages[1]["content"].append({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_data,
                                        "format": "mp3",
                                    },
                                })
                                logger.info("✅ Base64音频数据已添加到消息中")
                                
                            except Exception as audio_error:
                                logger.warning(f"音频数据处理失败: {audio_error}")
                                # 作为文本处理
                                messages[1]["content"][0]["text"] += f"\n\n🎤 AUDIO INFO: {audio_data[:200]}..."
                        
                        elif audio_data.startswith('{'):
                            # JSON格式的音频指导
                            try:
                                audio_json = json.loads(audio_data)
                                if audio_json.get("guidance"):
                                    guidance_text = audio_json["guidance"]
                                    messages[1]["content"][0]["text"] += f"\n\n🎤 AUDIO GUIDANCE: {guidance_text}"
                                    
                                    # 如果有团队通信，也添加
                                    if audio_json.get("team_communications"):
                                        comms = audio_json["team_communications"]
                                        if comms:
                                            comm_texts = []
                                            for comm in comms:
                                                if isinstance(comm, dict) and comm.get("message"):
                                                    comm_texts.append(comm["message"])
                                                elif isinstance(comm, str):
                                                    comm_texts.append(comm)
                                            
                                            if comm_texts:
                                                messages[1]["content"][0]["text"] += f"\n🗣️ TEAM COMMUNICATIONS: {'; '.join(comm_texts[:3])}"
                                
                                logger.info("✅ 音频指导已添加为文本")
                            except json.JSONDecodeError as e:
                                logger.warning(f"音频JSON解析失败: {e}")
                        else:
                            # 普通文本音频信息
                            messages[1]["content"][0]["text"] += f"\n\n🎤 AUDIO INFO: {audio_data}"
                            logger.info("✅ 音频文本信息已添加")
                
                except Exception as e:
                    logger.warning(f"音频处理错误: {e}")
            
            return messages
            
        except Exception as e:
            logger.error(f"消息构建失败: {e}")
            # 返回基本消息
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "text", "text": "Analyze the situation and provide your ONE command."}]}
            ]

    def _query_model(self, observation, step: int) -> Tuple[str, np.ndarray, Dict]:
        """Query the Baichuan model using file upload format"""
        try:
            # Initialize media_paths
            media_paths = {
                "video": None,
                "audio": None,
                "image": None,
                "response": None,
                "api_input": None
            }
            
            # Save media files to disk for file upload
            video_path = None
            audio_path = None
            image_path = None
            
            # Handle video content
            if isinstance(observation, dict) and 'video' in observation:
                video_path = self._save_video(observation['video'], step)
                if video_path:
                    media_paths['video'] = video_path
            
            # Handle audio content
            if isinstance(observation, dict) and 'audio' in observation:
                audio_path = self._save_audio(observation['audio'], step)
                if audio_path:
                    media_paths['audio'] = audio_path
            
            # Handle image content (for non-video modes)
            if isinstance(observation, dict) and 'image' in observation:
                image_path = self._save_image(observation['image'], step)
                if image_path:
                    media_paths['image'] = image_path
            
            # Build user query text
            if self.include_vector_text:
                state_desc = self._get_state_description(observation)
                user_query = f"""Current game state:
{state_desc}

🚨🚨🚨 CRITICAL REMINDER: EXACTLY ONE COMMAND ONLY! 🚨🚨🚨

You MUST provide exactly ONE command in your response. Multiple commands will cause SYSTEM ERRORS!

❌ DO NOT DO THIS: Provide multiple "COMMAND:" lines
✅ DO THIS: Provide exactly one "COMMAND:" line

Choose the SINGLE most important action for this turn. You can plan additional moves for future turns.

Available inputs:
- Vector: Team member states (health, status, position) + global info (rounds remaining, normalized score)
- Video: Visual sequence showing game state progression
- Audio: Tactical guidance and team communications
- Discovery hints: Clues about nearby hidden objectives

Analyze the situation and provide your ONE command."""
            else:
                user_query = """🚨🚨🚨 CRITICAL REMINDER: EXACTLY ONE COMMAND ONLY! 🚨🚨🚨

You MUST provide exactly ONE command in your response. Multiple commands will cause SYSTEM ERRORS!

❌ DO NOT DO THIS: Provide multiple "COMMAND:" lines
✅ DO THIS: Provide exactly one "COMMAND:" line

Choose the SINGLE most important action for this turn. You can plan additional moves for future turns.

Available inputs:
- Video: Visual sequence showing game state progression
- Audio: Tactical guidance and team communications
- Discovery hints: Clues about nearby hidden objectives

⚠️ NOTE: Vector information is available visually - interpret team states from the visual input.

Analyze the situation and provide your ONE command."""

            # Prepare data for Baichuan API
            data = {
                "query": user_query,
                "system_prompt": self.system_prompt,
                "audiogen_flag": False,
                "session_id": self.session_id
            }
            
            # Prepare file uploads
            files = []
            if video_path and os.path.exists(video_path):
                files.append(('video_files', ('sequence_video.mp4', open(video_path, 'rb'), 'video/mp4')))
            if audio_path and os.path.exists(audio_path):
                files.append(('audio_file', ('sequence_audio.wav', open(audio_path, 'rb'), 'audio/wav')))
            if image_path and os.path.exists(image_path):
                files.append(('image_files', ('screen_capture.jpg', open(image_path, 'rb'), 'image/jpeg')))
            
            url = f"{FASTAPI_BASE_URL}/chat"
            
            logger.info(f"Sending request to {url}")
            logger.debug(f"Files being uploaded: {[f[0] for f in files]}")
            
            try:
                response = self.session.post(url, data=data, files=files, timeout=300)
            finally:
                # Close file handles
                for _, file_tuple in files:
                    file_tuple[1].close()
            
            if response.status_code == 200:
                response_data = response.json()
                model_response = response_data.get("text", "")
                self.session_id = response_data.get("session_id")
                
                logger.info(f"Model response received: {model_response[:100]}...")
                
                # Save the response
                media_paths["response"] = self._save_model_response(model_response, step)
                
                # Save API input summary
                media_paths["api_input"] = self._save_api_input_video(observation, step)
                
                # Extract command from response
                command = self._extract_command(model_response)
                
                return model_response, command, media_paths
            else:
                error_msg = f"API请求失败: {response.status_code}"
                logger.error(f"{error_msg}\n错误信息: {response.text}")
                return error_msg, np.array([0, 0, 50, 50], dtype=np.int32), media_paths
                
        except requests.exceptions.Timeout:
            error_msg = "API请求超时"
            logger.error(error_msg)
            return error_msg, np.array([0, 0, 50, 50], dtype=np.int32), media_paths
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求异常: {e}"
            logger.error(error_msg)
            return error_msg, np.array([0, 0, 50, 50], dtype=np.int32), media_paths
        except Exception as e:
            logger.error(f"Model query failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return safe media_paths dict with all expected keys
            safe_media_paths = {
                "video": None,
                "audio": None,
                "image": None,
                "response": None,
                "api_input": None
            }
            return f"Error: {e}", np.array([0, 0, 50, 50], dtype=np.int32), safe_media_paths

    def _run_single_episode(self, episode_idx: int) -> Dict:
        """运行单个episode的评估"""
        # 每个episode开始时清除会话
        self.clear_session()
        
        # 重置当前episode的追踪变量
        current_episode_compliance = {
            "total_turns": 0,
            "valid_single_commands": 0,
            "multiple_command_violations": 0,
            "no_command_found": 0,
            "compliance_rate": 0.0
        }
        
        # 重置环境
        try:
            observation, info = self.env.reset()
            logger.debug(f"Episode {episode_idx} environment reset successful")
        except Exception as e:
            logger.error(f"Episode {episode_idx} environment reset failed: {e}")
            raise
        
        episode_seed = self.seed_index + episode_idx * 1000
        logger.info(f"Episode {episode_idx + 1} - Score: {info.get('score_normalized', 0):.1f}/100, "
                   f"Max rounds: {info.get('max_rounds', 0)}, Seed: {episode_seed}")
        
        step_count = 0
        total_reward = 0
        steps = []
        
        while True:
            step_count += 1
            logger.debug(f"Episode {episode_idx + 1}, Step {step_count}")
            
            # 获取当前episode的媒体保存目录
            current_images_dir = self.episode_dirs[episode_idx]["images"] if episode_idx in self.episode_dirs else self.images_dir
            current_audio_dir = self.episode_dirs[episode_idx]["audio"] if episode_idx in self.episode_dirs else self.audio_dir
            current_videos_dir = self.episode_dirs[episode_idx]["videos"] if episode_idx in self.episode_dirs else self.videos_dir
            current_responses_dir = self.episode_dirs[episode_idx]["responses"] if episode_idx in self.episode_dirs else self.responses_dir
            
            # 临时修改保存目录
            orig_dirs = None
            if self.save_media:
                orig_dirs = (self.images_dir, self.audio_dir, self.videos_dir, self.responses_dir)
                self.images_dir = current_images_dir
                self.audio_dir = current_audio_dir  
                self.videos_dir = current_videos_dir
                self.responses_dir = current_responses_dir
            
            try:
                # Query model and save media files
                model_response, command, media_paths = self._query_model(observation, step_count)
                
                # 恢复原始目录
                if orig_dirs:
                    self.images_dir, self.audio_dir, self.videos_dir, self.responses_dir = orig_dirs
                
                # Validate single command requirement and track compliance
                current_episode_compliance["total_turns"] += 1
                
                # Use improved command detection that filters out markdown formatting
                valid_command_lines = []
                all_command_lines = re.findall(r"COMMAND:\s*[^\n]+", model_response, re.IGNORECASE)
                
                for cmd_line in all_command_lines:
                    # Filter out markdown formatting that isn't a real command
                    if not re.match(r"COMMAND:\s*[\*\#\-\`]+", cmd_line, re.IGNORECASE):
                        # Check if it contains actual command content
                        if re.search(r"COMMAND:\s*(?:\w+(?:,\w+)*|all)\s+\w+", cmd_line, re.IGNORECASE):
                            valid_command_lines.append(cmd_line)
                
                command_count = len(valid_command_lines)
                
                if command_count > 1:
                    current_episode_compliance["multiple_command_violations"] += 1
                    logger.warning(f"⚠️ Multiple valid commands detected ({command_count}). Using first valid command.")
                elif command_count == 0:
                    current_episode_compliance["no_command_found"] += 1
                    logger.warning("⚠️ No valid COMMAND found in response. Using default command.")
                else:
                    current_episode_compliance["valid_single_commands"] += 1
                    logger.debug(f"✅ Valid single command detected")
                
                # Set member list for multi-member commands before execution
                if hasattr(self, '_last_member_list') and self._last_member_list and command[0] == self.env.num_members + 1:
                    self.env.set_multi_member_list(self._last_member_list)
                
                # Execute command
                obs, reward, terminated, truncated, info = self.env.step(command)
                print(info)
                total_reward += reward
                
                # Generate proper command description based on command type
                try:
                    member_idx = int(command[0]) if len(command) > 0 else 0
                    cmd_idx = int(command[1]) if len(command) > 1 else 0
                    x = int(command[2]) if len(command) > 2 else 50
                    y = int(command[3]) if len(command) > 3 else 50
                    
                    # Validate command index
                    if 0 <= cmd_idx < len(self.command_types):
                        cmd_type = self.command_types[cmd_idx]
                    else:
                        cmd_type = self.command_types[0]  # Default to first command type
                        
                except (IndexError, ValueError, TypeError) as e:
                    logger.error(f"Error parsing command array: {e}, command: {command}")
                    member_idx, cmd_idx, x, y = 0, 0, 50, 50
                    cmd_type = self.command_types[0]
                
                if member_idx == self.env.num_members:
                    # Team-wide command
                    command_desc = f"{cmd_type} to ({x},{y}) by all team members"
                elif member_idx == self.env.num_members + 1:
                    # Multi-member command
                    command_desc = f"{cmd_type} to ({x},{y}) by multiple members"
                else:
                    # Individual member command
                    command_desc = f"{cmd_type} to ({x},{y}) by member {member_idx}"
                
                # Log step results with media paths
                step_result = {
                    "step": step_count,
                    "command": command.tolist(),
                    "command_desc": command_desc,
                    "reward": float(reward),
                    "total_reward": float(total_reward),
                    "score_normalized": float(info.get('score_normalized', 0)),
                    "rounds_remaining": info.get('rounds_remaining', 0),
                    "objectives_completed": info.get('objectives_completed', 0),
                    "model_response_length": len(model_response),
                    "terminated": terminated,
                    "truncated": truncated,
                    "media_paths": media_paths
                }
                
                steps.append(step_result)
                
                logger.debug(f"Episode {episode_idx + 1}, Step {step_count}: {command_desc}")
                logger.debug(f"Reward: {reward:.2f}, Total: {total_reward:.2f}, Score: {info.get('score_normalized', 0):.1f}/100")
                
                # Update observation
                observation = obs
                
                # Check termination
                if terminated or truncated:
                    logger.info(f"Episode {episode_idx + 1} ended - Terminated: {terminated}, Truncated: {truncated}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in episode {episode_idx + 1}, step {step_count}: {e}")
                # 恢复目录（如果出错）
                if orig_dirs:
                    self.images_dir, self.audio_dir, self.videos_dir, self.responses_dir = orig_dirs
                raise
        
        # Calculate episode compliance rate
        if current_episode_compliance["total_turns"] > 0:
            current_episode_compliance["compliance_rate"] = (
                current_episode_compliance["valid_single_commands"] / 
                current_episode_compliance["total_turns"]
            )
        
        # Final statistics for this episode
        final_stats = {
            "total_steps": step_count,
            "total_reward": float(total_reward),
            "final_score_normalized": float(info.get('score_normalized', 0)),
            "objectives_completed": info.get('objectives_completed', 0),
            "total_objectives": info.get('total_objectives', 0),
            "success_rate": info.get('objectives_completed', 0) / max(1, info.get('total_objectives', 1)),
            "terminated": terminated,
            "truncated": truncated
        }
        
        return {
            "episode_index": episode_idx,
            "episode_seed": episode_seed,
            "status": "completed",
            "steps": steps,
            "final_stats": final_stats,
            "command_compliance": current_episode_compliance
        }

    def _calculate_summary_stats(self, episode_results: List[Dict]) -> Dict:
        """计算所有episode的汇总统计"""
        if not episode_results:
            return {}
        
        # 收集所有已完成episode的统计数据
        completed_episodes = [ep for ep in episode_results if ep.get("status") == "completed"]
        failed_episodes = [ep for ep in episode_results if ep.get("status") == "failed"]
        
        if not completed_episodes:
            return {
                "total_episodes": len(episode_results),
                "completed_episodes": 0,
                "failed_episodes": len(failed_episodes),
                "success_rate": 0.0
            }
        
        # 提取各项指标
        scores = [ep["final_stats"]["final_score_normalized"] for ep in completed_episodes]
        steps = [ep["final_stats"]["total_steps"] for ep in completed_episodes]
        objectives_completed = [ep["final_stats"]["objectives_completed"] for ep in completed_episodes]
        total_objectives = [ep["final_stats"]["total_objectives"] for ep in completed_episodes]
        episode_success_rates = [ep["final_stats"]["success_rate"] for ep in completed_episodes]
        
        return {
            "total_episodes": len(episode_results),
            "completed_episodes": len(completed_episodes),
            "failed_episodes": len(failed_episodes),
            "completion_rate": len(completed_episodes) / len(episode_results),
            
            # 分数统计
            "score_stats": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores)
            },
            
            # 步数统计
            "steps_stats": {
                "mean": np.mean(steps),
                "std": np.std(steps),
                "min": np.min(steps),
                "max": np.max(steps),
                "median": np.median(steps)
            },
            
            # 目标完成统计
            "objectives_stats": {
                "total_completed": sum(objectives_completed),
                "total_available": sum(total_objectives),
                "mean_completed_per_episode": np.mean(objectives_completed),
                "mean_success_rate": np.mean(episode_success_rates),
                "episodes_with_100_percent": sum(1 for rate in episode_success_rates if rate >= 1.0),
                "episodes_with_50_percent_plus": sum(1 for rate in episode_success_rates if rate >= 0.5)
            },
            
            # 整体成功指标
            "overall_success_rate": np.mean(episode_success_rates),
            "consistency": 1.0 - (np.std(scores) / 100.0),  # 一致性指标 (分数标准差的倒数)
        }

    def _log_final_report(self, summary_stats: Dict, overall_command_compliance: Dict):
        """输出最终评估报告"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 MULTI-EPISODE EVALUATION REPORT")
        logger.info(f"{'='*80}")
        
        # 基础信息
        logger.info(f"API Provider: {self.api_provider.upper()}, Model: {self.model_name}")
        logger.info(f"Episodes: {summary_stats.get('total_episodes', 0)} total, "
                   f"{summary_stats.get('completed_episodes', 0)} completed, "
                   f"{summary_stats.get('failed_episodes', 0)} failed")
        logger.info(f"Completion Rate: {summary_stats.get('completion_rate', 0):.1%}")
        
        # 分数统计
        if "score_stats" in summary_stats:
            score_stats = summary_stats["score_stats"]
            logger.info(f"\n📊 SCORE STATISTICS:")
            logger.info(f"  Mean Score: {score_stats['mean']:.1f}/100 (±{score_stats['std']:.1f})")
            logger.info(f"  Score Range: {score_stats['min']:.1f} - {score_stats['max']:.1f}")
            logger.info(f"  Median Score: {score_stats['median']:.1f}/100")
        
        # 目标完成统计
        if "objectives_stats" in summary_stats:
            obj_stats = summary_stats["objectives_stats"]
            logger.info(f"\n🎯 OBJECTIVES STATISTICS:")
            logger.info(f"  Overall Success Rate: {obj_stats['mean_success_rate']:.1%}")
            logger.info(f"  Total Objectives Completed: {obj_stats['total_completed']}/{obj_stats['total_available']}")
            logger.info(f"  Episodes with 100% Success: {obj_stats['episodes_with_100_percent']}/{summary_stats['total_episodes']}")
            logger.info(f"  Episodes with 50%+ Success: {obj_stats['episodes_with_50_percent_plus']}/{summary_stats['total_episodes']}")
        
        # 效率统计
        if "steps_stats" in summary_stats:
            steps_stats = summary_stats["steps_stats"]
            logger.info(f"\n⚡ EFFICIENCY STATISTICS:")
            logger.info(f"  Mean Steps per Episode: {steps_stats['mean']:.1f} (±{steps_stats['std']:.1f})")
            logger.info(f"  Steps Range: {steps_stats['min']:.0f} - {steps_stats['max']:.0f}")
        
        # 一致性评估
        consistency = summary_stats.get('consistency', 0)
        logger.info(f"\n🎲 CONSISTENCY ANALYSIS:")
        logger.info(f"  Performance Consistency: {consistency:.1%}")
        if consistency >= 0.8:
            logger.info(f"  ✅ High consistency - Stable performance across episodes")
        elif consistency >= 0.6:
            logger.info(f"  ⚠️ Moderate consistency - Some performance variation")
        else:
            logger.info(f"  ❌ Low consistency - High performance variation")
        
        # 命令合规性报告
        compliance = overall_command_compliance
        logger.info(f"\n🔧 COMMAND COMPLIANCE REPORT:")
        logger.info(f"  Total Turns: {compliance['total_turns']}")
        logger.info(f"  Valid Single Commands: {compliance['valid_single_commands']}")
        logger.info(f"  Multiple Command Violations: {compliance['multiple_command_violations']}")
        logger.info(f"  No Command Found: {compliance['no_command_found']}")
        logger.info(f"  Overall Compliance Rate: {compliance['compliance_rate']:.1%}")
        
        if compliance['compliance_rate'] < 1.0:
            violation_rate = 100 - compliance['compliance_rate']*100
            logger.warning(f"  ⚠️ Model violated single command constraint in {violation_rate:.1f}% of turns!")
        else:
            logger.info(f"  ✅ Perfect command compliance achieved across all episodes!")
        
        logger.info(f"{'='*80}")

    
    def _convert_numpy_types(self, obj):
        """递归转换NumPy类型为Python原生类型，使其可以JSON序列化"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save multi-episode results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.api_provider}_eval_{self.difficulty}_seed{self.seed_index}_{self.num_episodes}ep_{timestamp}.json"
        
        # Save in output directory if media saving is enabled
        if self.save_media:
            filepath = self.output_dir / "results.json"
        else:
            filepath = Path(filename)
        
        # Add summary of media files to results
        if self.save_media:
            total_images = sum(len(ep.get("steps", [])) for ep in self.results["episodes"])
            total_videos = len([ep for ep in self.results["episodes"] if ep.get("status") == "completed"])
            
            self.results["media_summary"] = {
                "total_episodes": len(self.results["episodes"]),
                "total_images_across_episodes": total_images,
                "total_videos_across_episodes": total_videos,
                "output_directory": str(self.output_dir),
                "episode_directories": {
                    f"episode_{i:02d}": str(self.episode_dirs[i]["root"]) 
                    for i in range(len(self.results["episodes"]))
                    if i in self.episode_dirs
                }
            }
        # 转换NumPy类型为JSON可序列化类型
        json_safe_results = self._convert_numpy_types(self.results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Multi-episode results saved to: {filepath.absolute()}")
        if self.save_media:
            logger.info(f"Media files directory structure:")
            logger.info(f"  Root: {self.output_dir.absolute()}")
            logger.info(f"  Episodes: {len(self.results['episodes'])} episode subdirectories")
            logger.info(f"  Total Steps: {sum(len(ep.get('steps', [])) for ep in self.results['episodes'])}")
        
        return str(filepath)

    def clear_session(self):
        """清除当前会话"""
        if self.session_id:
            try:
                url = f"{FASTAPI_BASE_URL}/clear_session"
                data = {"session_id": self.session_id}
                response = self.session.post(url, data=data, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ 会话已清除")
                else:
                    logger.warning(f"⚠️ 清除会话失败: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ 清除会话错误: {e}")
        self.session_id = None

    def close(self):
        """Clean up resources."""
        self.clear_session()
        if self.env:
            self.env.close()

    def _save_video(self, video_data: str, step: int) -> Optional[str]:
        """Save video data to file"""
        if not self.save_media or not video_data:
            return None
        
        try:
            # Decode base64 video data
            video_bytes = base64.b64decode(video_data)
            
            # Save to file
            video_path = self.videos_dir / f"step_{step:03d}_video.mp4"
            with open(video_path, 'wb') as f:
                f.write(video_bytes)
            
            return str(video_path)
        except Exception as e:
            logger.error(f"Failed to save video for step {step}: {e}")
            return None

    def _save_audio(self, audio_data: str, step: int) -> Optional[str]:
        """Save audio data to file"""
        if not self.save_media or not audio_data:
            return None
        
        try:
            # Check if it's base64 encoded audio or JSON text
            if audio_data.startswith('{'):
                # JSON format - save as text file
                audio_path = self.audio_dir / f"step_{step:03d}_audio.json"
                with open(audio_path, 'w', encoding='utf-8') as f:
                    f.write(audio_data)
            else:
                # Base64 encoded audio - save as audio file
                audio_bytes = base64.b64decode(audio_data)
                audio_path = self.audio_dir / f"step_{step:03d}_audio.mp3"
                with open(audio_path, 'wb') as f:
                    f.write(audio_bytes)
            
            return str(audio_path)
        except Exception as e:
            logger.error(f"Failed to save audio for step {step}: {e}")
            return None

    def _save_image(self, image_data, step: int) -> Optional[str]:
        """Save image data to file"""
        if not self.save_media or image_data is None:
            return None
        
        try:
            from PIL import Image
            import io
            
            if isinstance(image_data, str):
                # Base64 encoded image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif hasattr(image_data, 'shape'):
                # Numpy array
                image = Image.fromarray(image_data.astype(np.uint8))
            else:
                return None
            
            # Save to file
            image_path = self.images_dir / f"step_{step:03d}_image.jpg"
            image.save(image_path, 'JPEG', quality=85)
            
            return str(image_path)
        except Exception as e:
            logger.error(f"Failed to save image for step {step}: {e}")
            return None

    def _save_model_response(self, response_text: str, step: int) -> Optional[str]:
        """Save model response to file"""
        if not self.save_media or not response_text:
            return None
        
        try:
            response_path = self.responses_dir / f"step_{step:03d}_response.txt"
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            
            return str(response_path)
        except Exception as e:
            logger.error(f"Failed to save response for step {step}: {e}")
            return None

    def _save_api_input_video(self, observation, step: int) -> Optional[str]:
        """Save API input video content"""
        if not self.save_media:
            return None
        
        try:
            # Create a summary of what would be sent to the API
            api_input_summary = {
                "step": step,
                "observation_type": type(observation).__name__,
                "modalities_included": []
            }
            
            if isinstance(observation, dict):
                if 'video' in observation and observation['video']:
                    api_input_summary["modalities_included"].append("video")
                    api_input_summary["video_size_bytes"] = len(observation['video'])
                
                if 'audio' in observation and observation['audio']:
                    api_input_summary["modalities_included"].append("audio")
                    api_input_summary["audio_size_bytes"] = len(observation['audio'])
                
                if 'image' in observation:
                    api_input_summary["modalities_included"].append("image")
                
                if 'vector' in observation:
                    api_input_summary["modalities_included"].append("vector")
                    api_input_summary["vector_shape"] = list(observation['vector'].shape)
            
            # Save summary
            api_input_path = self.output_dir / "api_inputs" / f"step_{step:03d}_api_input.json"
            api_input_path.parent.mkdir(exist_ok=True)
            
            with open(api_input_path, 'w', encoding='utf-8') as f:
                json.dump(api_input_summary, f, indent=2)
            
            return str(api_input_path)
        except Exception as e:
            logger.error(f"Failed to save API input summary for step {step}: {e}")
            return None

    def _extract_command(self, response_text: str) -> np.ndarray:
        """Extract command from model response"""
        try:
            # Look for COMMAND: pattern
            command_pattern = r"COMMAND:\s*([^\n]+)"
            matches = re.findall(command_pattern, response_text, re.IGNORECASE)
            
            if not matches:
                logger.warning("No COMMAND found in response, using default")
                return np.array([0, 0, 50, 50], dtype=np.int32)
            
            # Use first command found
            command_str = matches[0].strip()
            logger.debug(f"Found command: {command_str}")
            
            # Parse command string
            parts = command_str.split()
            if len(parts) < 3:
                logger.warning(f"Invalid command format: {command_str}")
                return np.array([0, 0, 50, 50], dtype=np.int32)
            
            # Handle different member selection formats
            member_part = parts[0]
            action = parts[1]
            
            # Parse member selection
            if member_part.lower() == "all":
                member_idx = self.env.num_members  # Team command
            elif "," in member_part:
                # Multi-member command like "0,1,2"
                member_indices = [int(x.strip()) for x in member_part.split(",") if x.strip().isdigit()]
                self._last_member_list = member_indices
                member_idx = self.env.num_members + 1  # Multi-member command
            else:
                # Single member
                try:
                    member_idx = int(member_part)
                except ValueError:
                    logger.warning(f"Invalid member index: {member_part}")
                    member_idx = 0
            
            # Parse action
            action_map = {cmd: i for i, cmd in enumerate(self.command_types)}
            cmd_idx = action_map.get(action.lower(), 0)
            
            # Parse coordinates
            if len(parts) >= 4:
                try:
                    x = max(0, min(100, int(float(parts[2]))))
                    y = max(0, min(100, int(float(parts[3]))))
                except ValueError:
                    x, y = 50, 50
            else:
                x, y = 50, 50
            
            return np.array([member_idx, cmd_idx, x, y], dtype=np.int32)
            
        except Exception as e:
            logger.error(f"Error extracting command: {e}")
            return np.array([0, 0, 50, 50], dtype=np.int32)

    def _create_env_for_episode(self, episode_idx: int) -> CoopCommandGymEnv:
        """为指定episode创建新的环境实例"""
        try:
            # 计算该episode的seed：基础seed + episode索引，确保每个episode独立
            episode_seed = self.seed_index + episode_idx * 1000
            
            # 确定录制目录
            if self.save_media and episode_idx in self.episode_dirs:
                recordings_dir = str(self.episode_dirs[episode_idx]["videos"])
            else:
                recordings_dir = "recordings"
            
            # 确定录制模式
            if self.input_mode == "video":
                recording_mode = "video"
            elif self.enhanced_video:
                recording_mode = "both"
            else:
                recording_mode = "individual"
            
            actual_fps = self.video_fps if self.enhanced_video else 1
            
            # 创建新的环境实例
            env = CoopCommandGymEnv(
                difficulty=self.difficulty,
                seed_index=episode_seed,  # 使用计算出的episode seed
                max_rounds=self.max_rounds,
                enable_audio=True,
                enable_visual=True,
                deterministic_commands=self.deterministic_commands,
                recording_mode=recording_mode,
                video_fps=actual_fps,
                enhanced_video=self.enhanced_video,
                audio_duration_per_frame=self.audio_duration_per_frame,
                recordings_dir=recordings_dir
            )
            
            logger.info(f"Created environment for episode {episode_idx} with seed {episode_seed}")
            return env
            
        except Exception as e:
            logger.error(f"Failed to create environment for episode {episode_idx}: {e}")
            raise

    def run_evaluation(self) -> Dict:
        """Run the multi-episode evaluation."""
        logger.info(f"Starting multi-episode evaluation - Provider: BAICHUAN, Model: {self.model_name}")
        logger.info(f"Episodes: {self.num_episodes}, Difficulty: {self.difficulty}, Base Seed: {self.seed_index}")
        if self.save_media:
            logger.info(f"Media files will be saved to: {self.output_dir}")

        # Initialize results structure
        self.results = {
            "config": {
                "difficulty": self.difficulty,
                "seed_index": self.seed_index,
                "max_rounds": self.max_rounds,
                "api_provider": self.api_provider,
                "model": self.model_name,
                "input_mode": self.input_mode,
                "include_vector_text": self.include_vector_text,
                "enhanced_video": self.enhanced_video,
                "video_fps": self.video_fps,
                "audio_duration_per_frame": self.audio_duration_per_frame,
                "vision_support": True,
                "num_episodes": self.num_episodes,
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(self.output_dir) if self.save_media else None
            },
            "episodes": [],
            "summary_stats": {},
            "media_files": {
                "images": [],
                "audio": [],
                "videos": [],
                "responses": []
            },
            "command_compliance": {
                "total_turns": 0,
                "valid_single_commands": 0,
                "multiple_command_violations": 0,
                "no_command_found": 0,
                "compliance_rate": 0.0
            }
        }

        for ep in range(self.num_episodes):
            logger.info(f"\n🚀 Starting Episode {ep + 1}/{self.num_episodes}")
            
            # Create a new environment instance for this episode
            self.env = self._create_env_for_episode(ep)
            
            # Run the episode
            episode_result = self._run_single_episode(ep)
            self.results["episodes"].append(episode_result)
        
        # 计算汇总统计信息
        self.results["summary_stats"] = self._calculate_summary_stats(self.results["episodes"])
        
        # 记录最终报告
        self._log_final_report(self.results["summary_stats"], self.results["episodes"][-1]["command_compliance"])
        
        return self.results

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Episode Cooperative Command Game Evaluation")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["normal", "medium", "hard"])
    parser.add_argument("--seed_index", type=int, default=0)
    parser.add_argument("--max_rounds", type=int, default=100, help="Maximum rounds for the game (default: 100)")
    parser.add_argument("--save_media", action="store_true", default=True, help="Save media files (default: True)")
    parser.add_argument("--probabilistic_commands", action="store_true")
    # 移除不再需要的API provider选项，固定使用baichuan
    parser.add_argument("--model_name", type=str, default="baichuan",
                        help="Model name (defaults to baichuan)")
    parser.add_argument("--no_stream", action="store_true",
                        help="Disable streaming responses")
    parser.add_argument("--input_mode", type=str, default="video", 
                        choices=["image_audio", "video"],
                        help="Input modality mode: 'image_audio' for separate image and audio inputs, 'video' for video input")
    parser.add_argument("--no_vector_text", action="store_true",
                        help="Exclude vector information from text prompt (rely on visual interpretation only)")
    parser.add_argument("--enhanced_video", action="store_true",
                        help="Enable enhanced video recording with integrated audio")
    parser.add_argument("--video_fps", type=float, default=0.5,
                        help="Frames per second for video recording (default: 0.5 for audio integration)")
    parser.add_argument("--audio_duration_per_frame", type=float, default=3.0,
                        help="Expected audio duration per frame in seconds (default: 3.0)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: 10)")
    return parser.parse_args()

def main():
    args = parse_args()
    difficulty = args.difficulty
    seed_index = args.seed_index
    max_rounds = args.max_rounds
    save_media = args.save_media
    probabilistic_commands = args.probabilistic_commands
    model_name = args.model_name
    enable_stream = not args.no_stream
    input_mode = args.input_mode
    include_vector_text = not args.no_vector_text
    enhanced_video = args.enhanced_video
    video_fps = args.video_fps
    audio_duration_per_frame = args.audio_duration_per_frame
    num_episodes = args.num_episodes
    
    # Display configuration
    print(f"\n🚀 Multi-Episode Evaluation Configuration")
    print(f"📍 API Provider: BAICHUAN")
    print(f"🤖 Model: {model_name}")
    print(f"🎮 Difficulty: {difficulty.upper()}, Base Seed: {seed_index}")
    print(f"🔄 Episodes: {num_episodes}")
    print(f"🎯 Max Rounds per Episode: {max_rounds}")
    print(f"💾 Save Media: {'Yes' if save_media else 'No'}")
    print(f"🎯 Command Execution: {'Probabilistic' if probabilistic_commands else 'Deterministic'}")
    print(f"🔄 Streaming: {'Enabled' if enable_stream else 'Disabled'}")
    print(f"🎬 Input Mode: {input_mode.upper()}")
    print(f"📊 Include Vector Text: {'Yes' if include_vector_text else 'No'}")
    print(f"🎥 Enhanced Video: {'Yes' if enhanced_video else 'No'}")
    print(f"🎥 Video FPS: {video_fps}")
    print(f"🎤 Audio Duration per Frame: {audio_duration_per_frame}s")
    
    print(f"👁️  Vision: Supported ({input_mode} input)")
    print("=" * 60)
    
    # Run evaluation
    evaluator = MultiProviderEvaluator(
        difficulty=difficulty,
        seed_index=seed_index,
        max_rounds=max_rounds,
        enable_stream=enable_stream,
        save_media=save_media,
        deterministic_commands=not probabilistic_commands,
        api_provider="baichuan",
        model_name=model_name,
        input_mode=input_mode,
        include_vector_text=include_vector_text,
        enhanced_video=enhanced_video,
        video_fps=video_fps,
        audio_duration_per_frame=audio_duration_per_frame,
        num_episodes=num_episodes
    )

    try:
        # 检查百川API连接
        try:
            response = evaluator.session.get(f"{FASTAPI_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ 百川模型 API 连接正常")
            else:
                print("❌ 百川模型 API 连接异常")
                return
        except Exception as e:
            print(f"❌ 无法连接到百川模型 API: {e}")
            return
        
        results = evaluator.run_evaluation()
        filepath = evaluator.save_results()
        
        print(f"\n✅ Multi-episode evaluation completed!")
        print(f"📄 Results saved to: {filepath}")
        print(f"📈 Episodes: {results['summary_stats'].get('completed_episodes', 0)}/{results['summary_stats'].get('total_episodes', 0)}")
        print(f"🏆 Mean Score: {results['summary_stats'].get('score_stats', {}).get('mean', 0):.1f}/100")
        print(f"🎯 Overall Success Rate: {results['summary_stats'].get('objectives_stats', {}).get('mean_success_rate', 0):.1%}")
        print(f"🎮 Provider: BAICHUAN")
        
    except KeyboardInterrupt:
        print("\n⛔ Multi-episode evaluation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Multi-episode evaluation failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
    finally:
        evaluator.close()

if __name__ == "__main__":
    main()