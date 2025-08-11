"""
Full-modal model inference client based on the Baichuan model
---------------------------------------------------
• Takes Env observations (image np.array, audio np.array, state vec)
  → Constructs a multimodal prompt compatible with the Baichuan API
• Calls the Baichuan interface
• Parses the model's reply into a discrete action id
"""
from __future__ import annotations
import time
import base64
import io
import json
import math
import os
import wave
import re
import numpy as np
import requests
from PIL import Image
from typing import Optional, Dict, List, Any
from datetime import datetime
import random

# --- Baichuan FastAPI Server Configuration ---
FASTAPI_BASE_URL = ""  # FastAPI server address for the Baichuan model

# New: Multi-turn Conversation Strategy Configuration
CONVERSATION_STRATEGY = {
    "mode": "hybrid",  # "native", "rag", "hybrid"
    "native_window_size": 8,  # Number of native conversation rounds to keep
    "rag_retrieval_count": 3,  # Number of relevant rounds for RAG retrieval
    "compress_old_rounds": True,  # Whether to compress old rounds
    "multimodal_summary": True,  # Whether to summarize multimodal data
}

# ---- Action Mapping to be consistent with Env ----
COLOR_ID_MAP = {
    "BLUE": 0,     # Sol
    "RED": 1,      # Do
    "GREEN": 2,    # Fa  
    "YELLOW": 3,   # Mi
    "ORANGE": 4,   # Re
    "PURPLE": 5,   # La
    "GREY": 6,     # Ti/Si
}

def save(data: Any, filename: str, indent: int = 2):
    """Helper function to save data to a JSON file"""
    try:
        # Create debug directory
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        timestamped_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(debug_dir, timestamped_filename)
        
        # Clean large binary content from data for saving
        cleaned_data = _clean_data_for_json(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=indent, ensure_ascii=False, default=str)
        
        print(f"Debug data saved to: {filepath}")
        
    except Exception as e:
        print(f"Warning: Failed to save debug data to {filename}: {e}")

def _clean_data_for_json(data: Any) -> Any:
    """Clean non-JSON serializable content from data"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key in ["image_url", "audio"] and isinstance(value, dict):
                # For image and audio data, only keep metadata
                if "url" in value and value["url"].startswith("data:"):
                    cleaned[key] = {"type": "base64_data", "size": len(value["url"])}
                elif "data" in value:
                    cleaned[key] = {"type": "binary_data", "size": len(str(value["data"]))}
                else:
                    cleaned[key] = _clean_data_for_json(value)
            else:
                cleaned[key] = _clean_data_for_json(value)
        return cleaned
    elif isinstance(data, list):
        return [_clean_data_for_json(item) for item in data]
    elif isinstance(data, (np.ndarray, np.integer, np.floating)):
        return data.tolist() if hasattr(data, 'tolist') else str(data)
    else:
        return data

class SimpleLocalAgent:
    """A simple local agent as a fallback for network failures"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.random_seed = int(time.time()) % 1000
        self.action_count = 0
        
    def act(self, obs: dict) -> int:
        """Simple random policy with some heuristics"""
        self.action_count += 1
        
        # Simple policy based on game state
        state = obs.get("state", [0, 0, 0, 0])
        audio = obs.get("audio", np.zeros(16000))
        
        # If there is an audio signal, try to select an action based on audio intensity
        if not np.allclose(audio, 0):
            audio_energy = np.sum(audio ** 2)
            action = int(audio_energy * 1000) % len(COLOR_ID_MAP)
        else:
            # Otherwise, use a cyclic policy
            action = self.action_count % len(COLOR_ID_MAP)
        
        if self.verbose:
            print(f"Local agent chose action {action}")
        
        return action

class ConversationMemoryManager:
    """Multimodal Conversation Memory Manager"""
    
    def __init__(self, max_native_history: int = 8, max_total_memory: int = 50):
        self.max_native_history = max_native_history
        self.max_total_memory = max_total_memory
        
        # Store native conversation rounds
        self.conversation_rounds: List[Dict[str, Any]] = []
        
        # Store compressed history summaries
        self.compressed_summaries: List[Dict[str, Any]] = []
        
        # Current episode information
        self.current_episode = 1
        
        self.learned_color_note_mapping: Dict[str, str] = {}
        
    def reset_for_new_episode(self, episode_number: int):
        """Reset memory manager for a new episode"""
        # Compress the memory of the current episode
        if self.conversation_rounds:
            episode_summary = self._create_episode_summary()
            self.compressed_summaries.append(episode_summary)
        
        # Clear native history, prepare for a new episode
        self.conversation_rounds = []
        self.current_episode = episode_number
        
        # Limit the number of compressed summaries
        if len(self.compressed_summaries) > 10:
            self.compressed_summaries = self.compressed_summaries[-10:]

    def add_round(self, round_data: Dict[str, Any]):
        """Add a new conversation round"""
        # Compress multimodal data
        compressed_round = self._compress_multimodal_data(round_data)
        
        self.conversation_rounds.append(compressed_round)
        
        # Limit native history length
        if len(self.conversation_rounds) > self.max_native_history:
            # Compress and remove the oldest round
            old_round = self.conversation_rounds.pop(0)
            summary = self._create_round_summary(old_round)
            self.compressed_summaries.append(summary)
            
            # Limit the number of compressed summaries
            if len(self.compressed_summaries) > 20:
                self.compressed_summaries = self.compressed_summaries[-20:]
    
    def get_recent_native_history(self) -> List[Dict[str, Any]]:
        """Get recent native history"""
        return self.conversation_rounds.copy()
    
    def retrieve_relevant_rounds(self, current_context: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve historical rounds based on relevance"""
        if not self.compressed_summaries:
            return []
        
        # Simple relevance calculation
        scored_rounds = []
        for summary in self.compressed_summaries:
            relevance = self._calculate_relevance_score(current_context, summary)
            scored_rounds.append((relevance, summary))
        
        # Sort by relevance and return top_k
        scored_rounds.sort(key=lambda x: x[0], reverse=True)
        return [round_data for _, round_data in scored_rounds[:top_k]]
    
    def _compress_multimodal_data(self, round_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress multimodal data, keeping key information"""
        compressed = round_data.copy()
        
        # Keep image analysis but remove raw data
        if "image_analysis" in compressed:
            compressed["image_analysis"] = {
                "dominant_colors": compressed["image_analysis"].get("dominant_colors", []),
                "brightness": compressed["image_analysis"].get("brightness", 0),
                "detected_blocks": compressed["image_analysis"].get("detected_blocks", {})
            }
        
        # Keep audio analysis but remove raw data
        if "audio_analysis" in compressed:
            compressed["audio_analysis"] = {
                "has_sound": compressed["audio_analysis"].get("has_sound", False),
                "rms_level": compressed["audio_analysis"].get("rms_level", 0),
                "dominant_frequency": compressed["audio_analysis"].get("dominant_frequency", 0)
            }
        
        return compressed
    
    def _create_round_summary(self, round_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a round summary"""
        return {
            "type": "round_summary",
            "timestamp": time.time(),
            "game_state": round_data.get("game_state", {}),
            "action_taken": round_data.get("action_taken", ""),
            "decision_reasoning": round_data.get("decision_reasoning", ""),
            "key_observations": {
                "had_audio": round_data.get("audio_analysis", {}).get("has_sound", False),
                "visual_change": round_data.get("image_analysis", {}).get("brightness", 0) > 50
            }
        }
    
    def _create_episode_summary(self) -> Dict[str, Any]:
        """Create an episode summary"""
        if not self.conversation_rounds:
            return {"type": "episode_summary", "episode": self.current_episode, "rounds": 0}
        
        # Summarize key information in the episode
        total_rounds = len(self.conversation_rounds)
        actions_taken = [r.get("action_taken", "") for r in self.conversation_rounds]
        
        return {
            "type": "episode_summary", 
            "episode": self.current_episode,
            "rounds": total_rounds,
            "common_actions": actions_taken[-5:] if actions_taken else [],
            "timestamp": time.time()
        }
    
    def _calculate_relevance_score(self, current_context: Dict[str, Any], historical_round: Dict[str, Any]) -> float:
        """Calculate relevance score"""
        score = 0.0
        
        # Similarity based on game state
        if "game_state" in current_context and "game_state" in historical_round:
            current_score = current_context["game_state"].get("score", 0)
            hist_score = historical_round["game_state"].get("score", 0)
            score += 1.0 / (1.0 + abs(current_score - hist_score) / 100.0)
        
        # Similarity based on audio state
        current_audio = current_context.get("audio_analysis", {}).get("has_sound", False)
        hist_audio = historical_round.get("audio_analysis", {}).get("has_sound", False)
        if current_audio == hist_audio:
            score += 0.5
        
        return score
    
    def get_conversation_context_for_api(self, strategy: str = "hybrid") -> str:
        """Get conversation context based on strategy"""
        if strategy == "native":
            return self._build_native_context()
        elif strategy == "rag":
            return self._build_rag_context()
        else:  # hybrid
            return self._build_hybrid_context()
    
    def _build_native_context(self) -> str:
        """Build native-style context"""
        context_parts = []
        
        for i, round_data in enumerate(self.conversation_rounds):
            round_summary = f"Round {i+1}:\n"
    
            # Check if in the correct sequence
            correct_sequence = round_data.get('currently_in_correct_sequence', False)
            round_summary += f"  Game State: Currently in correct sequence={correct_sequence}\n"
            
            # Add action and reasoning information
            action_taken = round_data.get('action_taken', 'Unknown')
            round_summary += f"  Action: {action_taken}\n"
            context_parts.append(round_summary)
        
        return "\n".join(context_parts)
    
    def _build_rag_context(self) -> str:
        """Build RAG-style context"""
        context_parts = []
        
        # Add relevant historical summaries
        for summary in self.compressed_summaries[-3:]:  # Last 3 summaries
            if summary.get("type") == "episode_summary":
                context_parts.append(f"Previous Episode {summary.get('episode', '?')}: {summary.get('rounds', 0)} rounds")
            elif summary.get("type") == "round_summary":
                context_parts.append(f"Similar situation: {summary.get('action_taken', 'Unknown')} ...")
        
        return "\n".join(context_parts)
    
    def _build_hybrid_context(self) -> str:
        """Build hybrid strategy context"""
        context_parts = []
        
        # Add recent native history (simplified version)
        recent_context = self._build_native_context()
        if recent_context:
            context_parts.append("RECENT HISTORY:")
            context_parts.append(recent_context)
        
        # Add relevant historical summaries
        #rag_context = self._build_rag_context()
        #if rag_context:
        #    context_parts.append("\nRELEVANT PAST EXPERIENCE:")
        #    context_parts.append(rag_context)
        
        return "\n".join(context_parts)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "active_rounds": len(self.conversation_rounds),
            "compressed_summaries": len(self.compressed_summaries),
            "current_episode": self.current_episode,
            "total_memory_items": len(self.conversation_rounds) + len(self.compressed_summaries)
        }

class MultimodalAgent:
    def __init__(self, verbose: bool = False, use_local_fallback: bool = True, max_retries: int = 3,
                 conversation_strategy: str = "hybrid"):
        self.verbose = verbose
        self.use_local_fallback = use_local_fallback
        self.max_retries = max_retries
        
        # Baichuan API session management
        self.session = requests.Session()
        self.session_id = None  # Session ID for the Baichuan model
        
        # Conversation Strategy
        self.conversation_strategy = conversation_strategy
        self.memory_manager = ConversationMemoryManager(
            max_native_history=CONVERSATION_STRATEGY["native_window_size"],
            max_total_memory=25
        )
        
        # Local fallback agent
        self.local_agent = SimpleLocalAgent(verbose=verbose) if use_local_fallback else None
        
        # Connection statistics
        self.api_call_count = 0
        self.api_fail_count = 0
        
        # Decision history - extended to include full model output
        self.decision_history: List[Dict[str, Any]] = []
        
        # New: Full model output history
        self.model_output_history: List[Dict[str, Any]] = []
        
        # New: Learned color-note mapping - changed to model-learned mapping
        self.learned_color_note_mapping: Dict[str, str] = {}
        self.color_feedback_history: Dict[str, List[bool]] = {}
        
        # New: Game state tracking
        self.current_episode = 1
        self.current_step = 0
        self.previous_clicks: List[str] = []
        self.current_correct_sequence: List[str] = []
        self.last_action_result = None
        self.is_in_correct_sequence = False
        self.needs_restart_from_beginning = False

        # New: Text information logging
        self.text_outputs = []  # Record text output for each step
        self.current_step_text = ""  # Text information for the current step
        
        # New: Game environment connection
        self.game_environment = None

        # New: Game completion and score records
        self.game_completion_history: List[Dict[str, Any]] = []
        self.current_round_start_time = time.time()
        self.last_game_completed = False
        self.completion_scores: List[int] = []
        self.total_rounds_played = 0
        self.total_successful_rounds = 0

    def set_game_environment(self, env):
        """Set the game environment connection to get real-time game status"""
        self.game_environment = env
        if self.verbose:
            print("Game environment connected to agent")

    def clear_session(self):
        """Clear current session"""
        if self.session_id:
            try:
                url = f"{FASTAPI_BASE_URL}/clear_session"
                data = {"session_id": self.session_id}
                response = self.session.post(url, data=data, timeout=10)
                if response.status_code == 200:
                    print("✅ Session cleared")
                else:
                    print(f"⚠️ Failed to clear session: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Session clearing error: {e}")
        self.session_id = None

    def act(self, obs: dict) -> int:
        """
        obs = {"image": np.uint8 [H,W,3], "audio": np.float32 [N], "state": np.float32 [k]}
        return: action id (int)
        """
        decision_start_time = time.time()
        
        # Update step count
        self.current_step += 1
         
        # Analyze observation data
        obs_analysis = self._analyze_observation(obs)
        
        # Prepare current round data
        current_round_data = {
            "game_state": {
                "score": float(obs["state"][0]),
                "lives": float(obs["state"][1]), 
                "solved": float(obs["state"][2]),
                "tick": float(obs["state"][3])
            },
            "image_analysis": obs_analysis["image"],
        }
        
        action_id = None
        decision_method = None
        llm_response = None
        error_info = None
        full_api_response = None
        end_game = False  # Initialize end_game variable
        
        # First, try using the Baichuan API
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    print(f"Attempting Baichuan API call {attempt + 1}/{self.max_retries}...")
                
                payload, payload_end_game = self._build_payload(obs, current_round_data)
                end_game = payload_end_game  # Get end_game value from payload
                
                # Record API call start time
                api_start_time = time.time()
                
                # Call Baichuan API and get full response
                full_response, response_text = self._query_baichuan_api_with_full_response(payload)
                
                # Record API call end time
                api_end_time = time.time()
                
                # Parse action
                action_id = self._parse_action(response_text)
                decision_method = "Baichuan_API"
                llm_response = response_text
                full_api_response = {
                    "success": True,
                    "response_time": api_end_time - api_start_time,
                    "response_text": response_text,
                    "full_response": full_response,
                    "attempt_number": attempt + 1,
                    "payload_summary": self._sanitize_payload_for_storage(payload),
                    "timestamp": api_end_time
                }
                
                self.api_call_count += 1
                break
                
            except Exception as e:
                error_info = str(e)
                self.api_fail_count += 1
                
                # Record failed API call
                failed_response = {
                    "success": False,
                    "error": error_info,
                    "attempt_number": attempt + 1,
                    "timestamp": time.time()
                }
                self.model_output_history.append(failed_response)
                
                if self.verbose:
                    print(f"Baichuan API call {attempt + 1} failed: {error_info}")
                
                if attempt == self.max_retries - 1:
                    break
        
        # Ensure we have a valid action_id
        if action_id is None:
            if self.local_agent:
                action_id = self.local_agent.act(obs)
                decision_method = "Local_Fallback"
                if self.verbose:
                    print("Using local fallback agent")
            else:
                # Final fallback: random selection
                action_id = self.current_step % len(COLOR_ID_MAP)
                decision_method = "Random_Fallback"
                if self.verbose:
                    print("Using random fallback")
        
        # Record successful API response to history
        if full_api_response:
            self.model_output_history.append(full_api_response)
        
        # Record decision process and add to memory manager
        decision_time = time.time() - decision_start_time
        current_round_data["action_taken"] = list(COLOR_ID_MAP.keys())[action_id]
        current_round_data["decision_reasoning"] = llm_response or f"Method: {decision_method}"
        current_round_data["full_model_output"] = full_api_response
        
        self.memory_manager.add_round(current_round_data)
        
        self._record_decision(obs, obs_analysis, action_id, decision_method, 
                             llm_response, error_info, decision_time, full_api_response)
        
        return action_id, end_game

    def _sanitize_payload_for_storage(self, payload: dict) -> dict:
        """Clean the payload for storage (remove large binary data)"""
        sanitized = payload.copy()
        
        # Process multimodal content in messages
        if "messages" in sanitized:
            sanitized_messages = []
            for msg in sanitized["messages"]:
                sanitized_msg = msg.copy()
                if isinstance(msg.get("content"), list):
                    sanitized_content = []
                    for content_item in msg["content"]:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "image":
                                sanitized_content.append({"type": "image", "data": "[IMAGE_DATA_REMOVED]"})
                            elif content_item.get("type") == "audio":
                                sanitized_content.append({"type": "audio", "data": "[AUDIO_DATA_REMOVED]"})
                            else:
                                sanitized_content.append(content_item)
                        else:
                            sanitized_content.append(content_item)
                    sanitized_msg["content"] = sanitized_content
                sanitized_messages.append(sanitized_msg)
            sanitized["messages"] = sanitized_messages
        
        return sanitized

    def _query_baichuan_api_with_full_response(self, payload: dict) -> tuple:
        """Query the Baichuan API and return the full response and text content"""
        # Record request start time
        request_start_time = time.time()
        
        try:
            # Save request payload for debugging
            if self.verbose:
                print(f"Sending API request to Baichuan model...")
                try:
                    save(payload, "baichuan_payload.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save payload for debugging: {save_error}")
            
            # Prepare Baichuan API request data
            data = {
                "query": payload["query"],
                "system_prompt": payload["system_prompt"],
                "audiogen_flag": False,
                "session_id": self.session_id
            }
            
            # Prepare file uploads
            files = []
            
            # Process image file
            if payload.get("image_path"):
                files.append(('image_files', ('game_image.jpg', open(payload["image_path"], 'rb'), 'image/jpeg')))
            
            # Process audio file
            if payload.get("audio_path"):
                files.append(('audio_file', ('game_audio.wav', open(payload["audio_path"], 'rb'), 'audio/wav')))
            
            url = f"{FASTAPI_BASE_URL}/chat"
            
            # Send request
            try:
                r = self.session.post(url, data=data, files=files, timeout=300)
            finally:
                # Close file handles
                for _, file_tuple in files:
                    file_tuple[1].close()
            
            request_end_time = time.time()
            request_duration = request_end_time - request_start_time
            
            # Check HTTP status code
            if r.status_code != 200:
                error_msg = f"HTTP {r.status_code}: {r.text[:500]}"
                if self.verbose:
                    print(f"Baichuan API request failed with status {r.status_code}")
                    try:
                        save({"error": error_msg, "status_code": r.status_code, "response_text": r.text}, 
                             "baichuan_api_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise requests.exceptions.HTTPError(error_msg)
            
            # Parse JSON response
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON response: {str(e)[:200]}... Response text: {r.text[:500]}"
                if self.verbose:
                    try:
                        save({"error": error_msg, "raw_response": r.text}, "baichuan_json_parse_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise ValueError(error_msg)
            
            # Validate response structure
            if not isinstance(data, dict):
                error_msg = f"Invalid response format: expected dict, got {type(data)}"
                if self.verbose:
                    try:
                        save({"error": error_msg, "response_data": data}, "baichuan_invalid_format_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise ValueError(error_msg)
            
            # Check for error messages
            if "error" in data:
                error_info = data["error"]
                error_msg = f"Baichuan API returned error: {error_info.get('message', 'Unknown error')}"
                if self.verbose:
                    try:
                        save({"api_error": error_info, "full_response": data}, "baichuan_api_returned_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise RuntimeError(error_msg)
            
            # Extract response text
            response_text = data.get("text", "")
            self.session_id = data.get("session_id")
            
            # Validate response text
            if not isinstance(response_text, str):
                error_msg = f"Invalid response text type: expected str, got {type(response_text)}"
                if self.verbose:
                    try:
                        save({"error": error_msg, "response_text": response_text}, "baichuan_invalid_text_type_error.json", indent=2)
                    except Exception as save_error:
                        print(f"Warning: Could not save error for debugging: {save_error}")
                raise ValueError(error_msg)
            
            # Log success information
            if self.verbose:
                print(f"Baichuan API request successful:")
                print(f"  - Duration: {request_duration:.3f}s")
                print(f"  - Response length: {len(response_text)} chars")
                print(f"  - Session ID: {self.session_id}")
                
                # Save successful response
                response_summary = {
                    "success": True,
                    "request_duration": request_duration,
                    "response_length": len(response_text),
                    "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "timestamp": request_end_time,
                    "session_id": self.session_id
                }
                try:
                    save(response_summary, "baichuan_successful_response.json", indent=2)
                    save(data, "baichuan_response.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save response for debugging: {save_error}")
            
            return data, response_text
            
        except requests.exceptions.Timeout:
            error_msg = "Baichuan API request timed out after 300 seconds"
            if self.verbose:
                print(f"Baichuan API request timeout")
                try:
                    save({"error": error_msg, "url": url, "timeout": 300}, "baichuan_timeout_error.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save error for debugging: {save_error}")
            raise TimeoutError(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)[:200]}"
            if self.verbose:
                print(f"Baichuan API connection error: {error_msg}")
                try:
                    save({"error": error_msg, "url": url}, "baichuan_connection_error.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save error for debugging: {save_error}")
            raise ConnectionError(error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)[:200]}"
            if self.verbose:
                print(f"Baichuan API request error: {error_msg}")
                try:
                    save({"error": error_msg, "url": url}, "baichuan_request_error.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save error for debugging: {save_error}")
            raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error in Baichuan API call: {str(e)[:200]}"
            if self.verbose:
                print(f"Unexpected Baichuan API error: {error_msg}")
                try:
                    save({"error": error_msg, "url": url, "exception_type": type(e).__name__}, 
                         "baichuan_unexpected_error.json", indent=2)
                except Exception as save_error:
                    print(f"Warning: Could not save error for debugging: {save_error}")
            raise RuntimeError(error_msg)

    # ---------- Decision Recording and Analysis ----------
    def _record_decision(self, obs: dict, obs_analysis: Dict[str, Any], action_id: int, 
                        decision_method: str, llm_response: str = None, 
                        error_info: str = None, decision_time: float = 0,
                        full_model_output: Dict[str, Any] = None):
        """Record the decision process"""
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "episode": self.current_episode,
            "step": self.current_step,
            "action_id": action_id,
            "action_name": list(COLOR_ID_MAP.keys())[action_id],
            "decision_method": decision_method,
            "decision_time": decision_time,
            "game_state": {
                "score": float(obs["state"][0]),
                "lives": float(obs["state"][1]),
                "solved": float(obs["state"][2]),
                "tick": float(obs["state"][3])
            },
            "observation_analysis": obs_analysis,
            "llm_response": llm_response,
            "full_model_output": full_model_output,
            "error_info": error_info
        }
        
        self.decision_history.append(decision_record)

        # Build complete text information
        text_info = {
            "decision_method": decision_method,
            "llm_response": llm_response or "",
            "error_info": error_info or "",
            "reasoning": self._extract_reasoning_from_response(llm_response) if llm_response else "",
            "timestamp": decision_record["timestamp"]
        }
        
        # Save text information
        self.text_outputs.append(text_info)
        self.current_step_text = self._format_step_text(text_info)

    def _extract_reasoning_from_response(self, response_text: str) -> str:
        """Extract reasoning process from model response"""
        if not response_text:
            return ""
        
        # Find the decision section
        reasoning_keywords = ["DECISION:", "REASONING:", "ANALYSIS:", "THINKING:"]
        for keyword in reasoning_keywords:
            start_idx = response_text.upper().find(keyword)
            if start_idx >= 0:
                # Extract content starting from the keyword
                reasoning_section = response_text[start_idx:start_idx+300]  # Limit length
                return reasoning_section.strip()
        
        # If no keyword is found, return the last 200 characters as reasoning
        return response_text[-200:].strip() if len(response_text) > 200 else response_text.strip()

    def _format_step_text(self, text_info: Dict[str, Any]) -> str:
        """Format step text information into a readable format"""
        formatted_text = f"Method: {text_info['decision_method']}\n"
        
        if text_info['reasoning']:
            formatted_text += f"Reasoning: {text_info['reasoning']}\n"
        
        if text_info['error_info']:
            formatted_text += f"Error: {text_info['error_info']}\n"
        
        
        return formatted_text

    def get_current_step_text(self) -> str:
        """Get the text information for the current step"""
        return self.current_step_text

    def get_all_text_outputs(self) -> List[Dict[str, Any]]:
        """Get all text output history"""
        return self.text_outputs.copy()
    
    def _analyze_observation(self, obs: dict) -> Dict[str, Any]:
        """Analyze observation data"""
        image = obs["image"]
        audio = obs["audio"]
        state = obs["state"]
        
        # Image analysis
        image_analysis = {
            "shape": image.shape,
            "brightness": float(np.mean(image)),
            "dominant_colors": self._get_dominant_colors(image),
            "detected_blocks": self._detect_colored_blocks(image),
            "progress_indicators": self._detect_progress_indicators(image)
        }
        

        # State analysis
        state_analysis = {
            "score": float(state[0]),
            "lives": float(state[1]),
            "solved": float(state[2]),
            "tick": float(state[3])
        }
        
        return {
            "image": image_analysis,
            "state": state_analysis
        }

    def _get_dominant_colors(self, image: np.ndarray, top_k: int = 5) -> List[List[int]]:
        """Get dominant colors in the image"""
        # Reshape the image to a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Simple color clustering (simplified version of k-means)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by pixel count
        sorted_indices = np.argsort(counts)[::-1]
        
        # Return the top_k colors
        top_colors = unique_colors[sorted_indices[:top_k]]
        return [color.tolist() for color in top_colors]

    def _detect_colored_blocks(self, image: np.ndarray) -> Dict[str, bool]:
        """Detect colored blocks in the image"""
        detected_blocks = {}
        
        # Define color ranges (simplified detection)
        color_ranges = {
            "RED": ([150, 0, 0], [255, 100, 100]),
            "GREEN": ([0, 150, 0], [100, 255, 100]),
            "BLUE": ([0, 0, 150], [100, 100, 255]),
            "YELLOW": ([150, 150, 0], [255, 255, 100]),
            "ORANGE": ([200, 100, 0], [255, 200, 100]),
            "PURPLE": ([100, 0, 150], [200, 100, 255]),
            "GREY": ([100, 100, 100], [200, 200, 200])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            
            # Check for pixels within this color range
            mask = np.all((image >= lower_np) & (image <= upper_np), axis=2)
            detected_blocks[color_name] = np.any(mask)
        
        return detected_blocks

    def _detect_progress_indicators(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect game progress indicators"""
        # Simplified progress detection
        brightness = np.mean(image)
        
        return {
            "overall_brightness": float(brightness),
            "has_bright_areas": brightness > 100,
            "has_dark_areas": brightness < 50
        }

    def _estimate_dominant_frequency(self, audio: np.ndarray) -> float:
        """Estimate the dominant frequency of the audio"""
        if np.allclose(audio, 0):
            return 0.0
        
        # Simple frequency estimation (based on zero-crossing rate)
        zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
        if len(zero_crossings) > 1:
            # Estimate frequency
            sample_rate = 16000  # Assume sample rate
            freq = len(zero_crossings) * sample_rate / (2 * len(audio))
            return float(freq)
        
        return 0.0

    def _build_exploration_strategy(self) -> str:
        """Build exploration strategy description"""
        if not self.color_feedback_history:
            return "Initial exploration: Try different colors to learn sound patterns."
        
        # Analyze attempted colors
        successful_colors = []
        unsuccessful_colors = []
        
        for color, results in self.color_feedback_history.items():
            success_rate = sum(results) / len(results) if results else 0
            if success_rate > 0.5:
                successful_colors.append(color)
            else:
                unsuccessful_colors.append(color)
        
        strategy_parts = []
        if successful_colors:
            strategy_parts.append(f"Previously successful colors: {', '.join(successful_colors)}")
        if unsuccessful_colors:
            strategy_parts.append(f"Previously unsuccessful: {', '.join(unsuccessful_colors)}")
        
        return " | ".join(strategy_parts) if strategy_parts else "Continue systematic exploration."

    def _parse_action(self, text: str) -> int:
        """Parse action from LLM response"""
        text_upper = text.upper()
        
        # Directly search for color names
        for color_name, color_id in COLOR_ID_MAP.items():
            if color_name in text_upper:
                return color_id
        
        # If no explicit color is found, try to parse from numbers
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            action_num = int(numbers[0]) % len(COLOR_ID_MAP)
            return action_num
        
        # Finally, fallback to random selection
        return self.current_step % len(COLOR_ID_MAP)

    def update_color_feedback(self, color_name: str, success: bool):
        """Update color feedback history"""
        if color_name not in self.color_feedback_history:
            self.color_feedback_history[color_name] = []
        self.color_feedback_history[color_name].append(success)
        
        # Limit history length
        if len(self.color_feedback_history[color_name]) > 10:
            self.color_feedback_history[color_name] = self.color_feedback_history[color_name][-10:]

    def update_learned_mapping(self, color_name: str, note_description: str):
        """Update learned color-note mapping"""
        self.learned_color_note_mapping[color_name] = note_description

    def _get_difficulty_sequence_length(self, difficulty: str) -> int:
        """Get sequence length for the given difficulty"""
        difficulty_lengths = {"easy": 3, "normal": 5, "hard": 7}
        return difficulty_lengths.get(difficulty, 5)

    def _color_id_to_english_name(self, color_id: int) -> str:
        """Convert color ID to English name"""
        color_names = list(COLOR_ID_MAP.keys())
        if 0 <= color_id < len(color_names):
            return color_names[color_id]
        return "Unknown"

    def _analyze_game_progress(self, obs: dict, action_taken: int = None) -> Dict[str, Any]:
        """Analyze game progress"""
        state = obs["state"]
        audio = obs["audio"]
        
        # Basic game information
        progress_info = {
            "current_score": float(state[0]),
            "lives_remaining": float(state[1]),
            "blocks_solved": float(state[2]),
            "game_tick": float(state[3])
        }
        
        # Audio feedback analysis
        has_audio_feedback = not np.allclose(audio, 0)
        progress_info["audio_feedback_present"] = has_audio_feedback
        
        if has_audio_feedback:
            progress_info["audio_intensity"] = float(np.sqrt(np.mean(audio**2)))
        
        # If an action was taken, record the color choice
        if action_taken is not None:
            progress_info["last_action_color"] = self._color_id_to_english_name(action_taken)
        
        return progress_info

    def _build_structured_game_info(self, obs: dict, action_taken: int = None) -> Dict[str, Any]:
        """Build structured game information"""
        # Basic game state
        game_info = self._analyze_game_progress(obs, action_taken)
        
        # Add learned mapping information
        game_info["learned_mappings"] = self.learned_color_note_mapping.copy()
        
        # Add color feedback history
        game_info["color_performance"] = {}
        for color, results in self.color_feedback_history.items():
            if results:
                success_rate = sum(results) / len(results)
                game_info["color_performance"][color] = {
                    "success_rate": success_rate,
                    "attempts": len(results),
                    "recent_success": results[-1] if results else False
                }
        
        # Add exploration suggestions
        game_info["exploration_strategy"] = self._build_exploration_strategy()
        
        return game_info

    def _build_payload(self, obs: dict, current_round_data: dict) -> dict:
        """Build the Baichuan API request payload"""
        # Save image to a temporary file
        image = obs["image"]
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            if isinstance(image, np.ndarray):
                image = image.astype(np.uint8)
            else:
                image = np.array(image, dtype=np.uint8)
        
        # Convert to PIL image and save
        img_pil = Image.fromarray(image)
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, f"game_image_{self.current_step}.jpg")
        img_pil.save(image_path)
        
        audio = obs["audio"]
        audio_path = os.path.join(temp_dir, f"game_audio_{self.current_step}.wav")
        
        try:
            # Process base64 encoded audio data
            if isinstance(audio, str) and audio.startswith("data:"):
                try:
                    # Parse data URL format: data:mime_type;base64,base64_data
                    header, base64_data = audio.split(',', 1)
                    mime_type = header.split(':')[1].split(';')[0]
                    
                    # Decode base64 data
                    audio_bytes = base64.b64decode(base64_data)
                    
                    # Save audio file directly
                    with open(audio_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    if self.verbose:
                        print(f"Base64 audio saved successfully: {len(audio_bytes)} bytes, type: {mime_type}")
                    
                except Exception as audio_error:
                    if self.verbose:
                        print(f"Base64 audio save error: {audio_error}")
                    audio_path = None
            else:
                try:
                    import wave
                    
                    # Check audio data type and preprocess
                    if isinstance(audio, str):
                        # If audio is a string, skip saving audio
                        if self.verbose:
                            print(f"Audio data is string type, skipping audio save")
                        audio_path = None
                    elif not isinstance(audio, np.ndarray):
                        # Try to convert to numpy array
                        try:
                            audio = np.array(audio, dtype=np.float32)
                        except (ValueError, TypeError) as e:
                            if self.verbose:
                                print(f"Cannot convert audio to numpy array: {e}")
                            audio_path = None
                    
                    if audio_path is not None and isinstance(audio, np.ndarray):
                        # Ensure audio data is a valid numerical array
                        if audio.size == 0:
                            if self.verbose:
                                print("Audio array is empty, skipping audio save")
                            audio_path = None
                        elif not np.isfinite(audio).all():
                            if self.verbose:
                                print("Audio contains invalid values (inf/nan), cleaning...")
                            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        if audio_path is not None:
                            with wave.open(audio_path, 'wb') as wav_file:
                                wav_file.setnchannels(1)  # Mono
                                wav_file.setsampwidth(2)  # 16-bit
                                wav_file.setframerate(16000)  # 16kHz sample rate
                                
                                # Convert float32 audio to int16
                                if audio.dtype == np.float32 or audio.dtype == np.float64:
                                    # Ensure audio values are within [-1, 1]
                                    audio = np.clip(audio, -1.0, 1.0)
                                    audio_int16 = (audio * 32767).astype(np.int16)
                                elif audio.dtype in [np.int16, np.int32]:
                                    audio_int16 = audio.astype(np.int16)
                                else:
                                    # For other types, first convert to float32 and then process
                                    audio_float = audio.astype(np.float32)
                                    audio_float = np.clip(audio_float, -1.0, 1.0)
                                    audio_int16 = (audio_float * 32767).astype(np.int16)
                                
                                wav_file.writeframes(audio_int16.tobytes())
                                
                                if self.verbose:
                                    print(f"Audio saved successfully: {audio.shape}, dtype={audio.dtype}")
                                    
                except Exception as audio_error:
                    if self.verbose:
                        print(f"Audio save error: {audio_error}")
                        print(f"Audio type: {type(audio)}")
                        if hasattr(audio, 'dtype'):
                            print(f"Audio dtype: {audio.dtype}")
                        if hasattr(audio, 'shape'):
                            print(f"Audio shape: {audio.shape}")
                    audio_path = None
        except:
            print("none")
        
            
        

        # Use the modified detailed game state information
        detailed_game_state, detailed_state, end_game = self._build_detailed_game_state_for_api(obs)
        
        # Get conversation context
        conversation_context = self.memory_manager.get_conversation_context_for_api(self.conversation_strategy)
    
        # Get current available color block information
        available_colors_info, available_colors_map = self._get_available_colors_info_dynamic()
        
        # Update color-note mapping
        if not detailed_state.get("last_clicked_block_color"):
            # First step, initialize mapping
            if isinstance(available_colors_map, list):
                for color_info in available_colors_map:
                    color_name = color_info.get("color_name", "")
                    if color_name:
                        self.learned_color_note_mapping[color_name] = "Unknown"
                
                if self.verbose:
                    print(f"Initialized color mapping with available colors: {self.learned_color_note_mapping}")
        else:
            # Update known mapping
            last_color = detailed_state.get("last_clicked_block_color")
            if last_color:
                last_color = last_color.capitalize()
            for color_info in available_colors_map if isinstance(available_colors_map, list) else []:
                if color_info.get("color_name", "") == last_color:
                    note_name = color_info.get("note_name", "Unknown")
                    self.learned_color_note_mapping[last_color] = note_name
                    break
            
            if self.verbose:
                print(f"Updated color-note mapping: {last_color} → {self.learned_color_note_mapping.get(last_color, 'Unknown')}")

        # System prompt
        system_prompt = (
    "You are a MULTIMODAL AI agent playing a musical color-matching game.\n"
    "\n"
    "## ROLE\n"
    "Click exactly ONE coloured block per turn to reproduce the target melody.\n"
    "\n"
    "## GAME RULES\n"
    "1. Musical order (ascending): do → re → mi → fa → sol → la → si.\n"
    "2. At the start of each round, the FIRST note is chosen at random; it may be any notes.\n"
    "3. After the first note, you must continue in the same ascending order **without skipping any note** until the melody is complete (wrap around if needed).\n"
    "4. After any wrong click, the sequence resets to this round's first note.\n"
    "5. Colour-to-note mapping is RANDOMIZED **each round**; learn it anew from feedback.\n"
    "\n"
    "## OBSERVATION FIELDS\n"
    "• `image`  – current board frame (colours & highlights).\n"
    "• `audio`  – sound from **your previous click**.\n"
    "• `currently_in_correct_sequence` (bool)\n"
    "• `needs_restart_from_beginning` (bool)\n"
    "• `current_correct_sequence` (list of colours already correct)\n"
    "• `input_length` (int)\n"
    "\n"
    f"{available_colors_info}\n"
    "*Clicking any other colour is invalid.*\n"
    "*The order of these colors has no significance; it's completely random.*\n"
    "\n"
    "## DECISION CHECKLIST\n"
    "1. If `needs_restart_from_beginning` is true → restart with this round's first note.\n"
    "2. Otherwise pick the next consecutive note based on `current_correct_sequence`—do **not** skip any note.\n"
    "3. Identify the NOTE you just heard by pairing your last action with the `audio` feedback.\n"
    "4. Choose the colour that plays the required next note.\n"
    "\n"
    "## OUTPUT FORMAT\n"
    "Reply with **ONLY** two uppercase tokens separated by a comma and a space:\n"
    "<COLOUR>, <NOTE>\n"
    "• <COLOUR>  e.g. `BLUE`.\n"
    "• <NOTE>    ∈ {DO, RE, MI, FA, SOL, LA, SI}.\n"
    "No other text, punctuation, or line breaks."
)

        # Build user query
        user_query = f"""Current game observation and detailed environment state:

{detailed_game_state}

Based on the detailed game state above, what color block should I click next?
- If currently_in_correct_sequence is True: Continue the musical sequence
- If needs_restart_from_beginning is True: Start from the beginning note
- If currently_in_correct_sequence is False: Choose a different color than the last clicked one

{conversation_context}

Remember to follow the ascending musical order without skipping notes."""

        return {
            "query": user_query,
            "system_prompt": system_prompt,
            "image_path": image_path,
            "audio_path": audio_path
        }, end_game

    def _get_available_colors_info_dynamic(self) -> str:
        """Dynamically get available color block information for the current round from the game environment"""
        retry_count = 0
        max_retries = 50
        
        while retry_count < max_retries:
            try:
                if not self.game_environment:
                    if self.verbose:
                        print(f"Game environment not available, retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.2)
                    continue
                
                # Try to get available colors from the game environment
                if hasattr(self.game_environment, 'get_available_colors'):
                    try:
                        available_colors = self.game_environment.get_available_colors()
                        shuffled_colors = available_colors.copy()
                        random.shuffle(shuffled_colors)
                        available_colors = shuffled_colors
                        
                        if available_colors and len(available_colors) > 0:
                            color_list = []
                            incomplete_info = False
                            
                            for color_info in available_colors:
                                color_name = color_info.get("color_name", "")
                                note_name = color_info.get("note_name", "")
                                
                                if not color_name or not note_name:
                                    if self.verbose:
                                        print(f"Incomplete color info: {color_info}, retrying... ({retry_count + 1}/{max_retries})")
                                    incomplete_info = True
                                    break
                                    
                                color_list.append(f"- {color_name.upper()}")
                            
                            if not incomplete_info and len(color_list) == len(available_colors):
                                colors_text = "\n".join(color_list)
                                if self.verbose:
                                    print(f"Successfully got available colors info: {len(color_list)} colors")
                                return colors_text, available_colors
                            else:
                                if self.verbose:
                                    print(f"Color info incomplete, retrying... ({retry_count + 1}/{max_retries})")
                                retry_count += 1
                                time.sleep(0.2)
                                continue
                        else:
                            if self.verbose:
                                print(f"No available colors returned, retrying... ({retry_count + 1}/{max_retries})")
                            retry_count += 1
                            time.sleep(0.2)
                            continue
                            
                    except Exception as env_error:
                        if self.verbose:
                            print(f"Error calling get_available_colors: {env_error}, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                
                # Fallback method: Get current color mapping from the game state
                if hasattr(self.game_environment, 'game_module') and self.game_environment.game_module:
                    game_module = self.game_environment.game_module
                    
                    if not hasattr(game_module, 'current_note_color_mapping'):
                        if self.verbose:
                            print(f"Game module missing current_note_color_mapping, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                    
                    current_note_color_mapping = getattr(game_module, 'current_note_color_mapping', {})
                    note_display_names = getattr(game_module, 'NOTE_DISPLAY_NAMES', {})
                    all_colors = getattr(game_module, 'ALL_COLORS', {})
                    
                    if not current_note_color_mapping or not note_display_names or not all_colors:
                        if self.verbose:
                            print(f"Game module data incomplete, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                    
                    color_name_mapping = {v: k for k, v in all_colors.items()}
                    color_list = []
                    incomplete_mapping = False
                    
                    for note_id, rgb_color in current_note_color_mapping.items():
                        color_name = color_name_mapping.get(rgb_color)
                        if not color_name:
                            incomplete_mapping = True
                            break
                            
                        note_display = note_display_names.get(note_id)
                        if not note_display:
                            incomplete_mapping = True
                            break
                            
                        color_list.append(f"- {color_name.upper()}")
                    
                    if not incomplete_mapping and len(color_list) == len(current_note_color_mapping):
                        colors_text = "\n".join(color_list)
                        if self.verbose:
                            print(f"Successfully got colors from game module: {len(color_list)} colors")
                        return colors_text, []
                    else:
                        if self.verbose:
                            print(f"Incomplete mapping from game module, retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(0.2)
                        continue
                
                if self.verbose:
                    print(f"All methods failed, retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(0.2)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error getting dynamic color info (retry {retry_count + 1}): {e}")
                retry_count += 1
                time.sleep(0.2)
        
        # If all retries fail, raise an exception
        raise RuntimeError(f"Failed to get real color information after {max_retries} retries. Game environment may not be properly initialized.")

    def _build_detailed_game_state_for_api(self, obs: dict) -> str:
        """Build detailed game state information, getting full state from the game environment in real-time"""
        state = obs["state"]
        
        try:
            # Get basic game state
            game_state_dict = {
                "current_score": float(state[0]),
                "lives_remaining": float(state[1]),
                "blocks_solved": float(state[2]),
                "current_tick": float(state[3])
            }
            
            # Get detailed state information from the game environment - add retry mechanism
            detailed_state = None
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries and not detailed_state:
                detailed_state, end_game = self._extract_detailed_state_from_environment()
                
                if not detailed_state:
                    retry_count += 1
                    if self.verbose:
                        print(f"Failed to get detailed state, retry {retry_count}/{max_retries}")
                    
                    # Short wait before retrying
                    if retry_count < max_retries:
                        import time
                        time.sleep(0.1)
                else:
                    break
            
            if detailed_state:
                # Extract key information from the detailed state
                last_action = detailed_state.get("last_clicked_action", 0)
                last_color = detailed_state.get("last_clicked_block_color", "")
                in_sequence = detailed_state.get("currently_in_correct_sequence", False)
                needs_restart = detailed_state.get("needs_restart_from_beginning", False)
                current_sequence = detailed_state.get("current_correct_sequence", [])
                last_last_color = detailed_state.get("last_last_clicked_block_color", "")
                previous_clicks = detailed_state.get("previous_clicks", [])
                sequence_length = detailed_state.get("sequence_length", 5)
                input_length = detailed_state.get("input_length", 0)
                attempts = detailed_state.get("attempts", 0)
                game_over = detailed_state.get("game_over", False)
                
                # New: check data consistency, if attempts > 0 but previous_clicks is empty, retry getting state
                if attempts > 0 and not previous_clicks:
                    if self.verbose:
                        print(f"Data inconsistency detected: attempts={attempts} but previous_clicks is empty. Retrying...")
                    
                    # Additional retry mechanism for data consistency issues
                    additional_retries = 10
                    for extra_retry in range(additional_retries):
                        if self.verbose:
                            print(f"Consistency retry {extra_retry + 1}/{additional_retries}")
                        
                        # Short wait before getting state again
                        import time
                        time.sleep(0.2)
                        
                        retry_state, end_game = self._extract_detailed_state_from_environment()
                        if retry_state:
                            retry_previous_clicks = retry_state.get("previous_clicks", [])
                            retry_attempts = retry_state.get("attempts", 0)
                            
                            # If data consistency improves after retry, use the new state
                            if retry_attempts > 0 and retry_previous_clicks:
                                if self.verbose:
                                    print(f"Consistency improved: attempts={retry_attempts}, previous_clicks={retry_previous_clicks}")
                                detailed_state = retry_state
                                # Re-extract key information
                                last_action = detailed_state.get("last_clicked_action", 0)
                                last_color = detailed_state.get("last_clicked_block_color", "")
                                in_sequence = detailed_state.get("currently_in_correct_sequence", False)
                                needs_restart = detailed_state.get("needs_restart_from_beginning", False)
                                current_sequence = detailed_state.get("current_correct_sequence", [])
                                last_last_color = detailed_state.get("last_last_clicked_block_color", "")
                                previous_clicks = retry_previous_clicks
                                sequence_length = detailed_state.get("sequence_length", 5)
                                input_length = detailed_state.get("input_length", 0)
                                attempts = retry_attempts
                                game_over = detailed_state.get("game_over", False)
                                break
                            elif retry_attempts == 0:
                                # If attempts becomes 0 after retry, the game may have been reset
                                if self.verbose:
                                    print(f"Game appears to have reset: attempts={retry_attempts}")
                                detailed_state = retry_state
                                # Re-extract key information
                                last_action = detailed_state.get("last_clicked_action", 0)
                                last_color = detailed_state.get("last_clicked_block_color", "")
                                in_sequence = detailed_state.get("currently_in_correct_sequence", False)
                                needs_restart = detailed_state.get("needs_restart_from_beginning", False)
                                current_sequence = detailed_state.get("current_correct_sequence", [])
                                last_last_color = detailed_state.get("last_last_clicked_block_color", "")
                                previous_clicks = retry_previous_clicks
                                sequence_length = detailed_state.get("sequence_length", 5)
                                input_length = detailed_state.get("input_length", 0)
                                attempts = retry_attempts
                                game_over = detailed_state.get("game_over", False)
                                break
                    else:
                        if self.verbose:
                            print(f"Consistency retries exhausted, using original state with warning")
                
                # Build detailed state description text
                state_description = f"""DETAILED GAME STATE (from environment):

LAST ACTION INFO:
- Last Clicked Action ID: {last_action}
- Last Clicked Block Color: {last_color}
- Previous Block Color (last_last): {last_last_color}

SEQUENCE STATUS (CRITICAL FOR DECISION):
- Currently in Correct Sequence: {in_sequence}
- Needs Restart from Beginning: {needs_restart}
- Current Correct Sequence: {current_sequence}
- Previous Clicks History: {previous_clicks}
- Sequence Length: {sequence_length}
- Current Input Length: {input_length}

GAME STATUS:
- Game Over: {game_over}
- Current Tick: {game_state_dict['current_tick']}
- Attempts: {attempts}

STRATEGY HINTS:
- If 'current_correct_sequence' has items: These are the correct colors so far
- If 'previous_clicks' shows history: Learn from past click patterns
- Use sequence position to determine next required musical note"""

                # Save game state data for debugging and analysis
                try:
                    game_state_data = {
                        "timestamp": datetime.now().isoformat(),
                        "episode": self.current_episode,
                        "step": self.current_step,
                        "last_clicked_color": last_color,
                        "currently_in_correct_sequence": in_sequence,
                        "current_correct_sequence": current_sequence,
                        "last_last_clicked_color": last_last_color,
                        "previous_clicks": previous_clicks,
                        "sequence_length": sequence_length,
                        "input_length": input_length,
                        "needs_restart": needs_restart,
                        "game_score": game_state_dict['current_score'],
                        "lives_remaining": game_state_dict['lives_remaining'],
                        "game_over": game_over,
                        "attempts": attempts,
                        "retry_count": retry_count,
                        "data_consistency_check": {
                            "attempts_gt_zero": attempts > 0,
                            "previous_clicks_empty": len(previous_clicks) == 0,
                            "inconsistency_detected": attempts > 0 and len(previous_clicks) == 0
                        }
                    }
                    
                    game_data_dir = "game_data/caclu"
                    os.makedirs(game_data_dir, exist_ok=True)
                    
                    # Name file using timestamp and step information
                    filename = f"game_state_ep{self.current_episode}_step{self.current_step}.json"
                    
                    # Create the full file path
                    full_filepath = os.path.join(game_data_dir, filename)
                    
                    # Save directly to the specified path, without using the save function's automatic path handling
                    with open(full_filepath, 'w', encoding='utf-8') as f:
                        json.dump(game_state_data, f, indent=2, ensure_ascii=False, default=str)
                    
                    if self.verbose:
                        print(f"Game state saved to: {full_filepath}")
                        
                except Exception as save_error:
                    if self.verbose:
                        print(f"Warning: Failed to save game state data: {save_error}")
                
                return state_description, detailed_state, end_game
            else:
                # All retries failed, use basic state information
                if self.verbose:
                    print(f"Failed to get detailed state after {max_retries} retries, using basic state")
                
                # Build basic state description
                fallback_description = f"""BASIC GAME STATE (detailed state unavailable after {max_retries} retries):

BASIC INFO:
- Score: {game_state_dict['current_score']}
- Lives: {game_state_dict['lives_remaining']}
- Blocks Solved: {game_state_dict['blocks_solved']}
- Game Tick: {game_state_dict['current_tick']}

SEQUENCE STATUS (LIMITED INFO):
- Episode: {self.current_episode}
- Step: {self.current_step}
- Last Action Taken: {list(COLOR_ID_MAP.keys())[self.current_step % len(COLOR_ID_MAP)] if self.current_step > 0 else "None"}

FALLBACK STRATEGY:
- Use audio feedback to learn color-note mappings
- Try systematic exploration of available colors
- Listen for musical sequence patterns in audio feedback
- Use visual feedback (green/red highlights) to confirm correct/incorrect choices

NOTE: Detailed game state from environment is temporarily unavailable. 
Making decisions based on basic state information and audio/visual feedback."""
                
                # Save failed state information for debugging
                try:
                    fallback_data = {
                        "timestamp": datetime.now().isoformat(),
                        "episode": self.current_episode,
                        "step": self.current_step,
                        "error": "Failed to get detailed state",
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                        "basic_state": game_state_dict,
                        "has_game_environment": self.game_environment is not None
                    }
                    
                    game_data_dir = "game_data/caclu"
                    os.makedirs(game_data_dir, exist_ok=True)
                    filename = f"fallback_state_ep{self.current_episode}_step{self.current_step}.json"
                    full_filepath = os.path.join(game_data_dir, filename)
                    with open(full_filepath, 'w', encoding='utf-8') as f:
                        json.dump(fallback_data, f, indent=2, ensure_ascii=False, default=str)
                    
                    if self.verbose:
                        print(f"Fallback state saved to: {full_filepath}")
                        
                except Exception as save_error:
                    if self.verbose:
                        print(f"Warning: Failed to save fallback state data: {save_error}")
                
                return fallback_description, {}, False
            
        except Exception as e:
            if self.verbose:
                print(f"Error building detailed game state: {e}")
            
            # Final error handling: return the most basic state information
            error_description = f"""EMERGENCY FALLBACK STATE (Error occurred):

ERROR: {str(e)}

MINIMAL GAME INFO:
- Score: {float(state[0])}
- Lives: {float(state[1])}
- Blocks Solved: {float(state[2])}
- Game Tick: {float(state[3])}
- Episode: {self.current_episode}
- Step: {self.current_step}

EMERGENCY STRATEGY:
- Use random exploration if no other information available
- Try to listen to audio feedback from previous actions
- Look for visual patterns in the game image
- Make conservative choices to preserve lives"""
            
            return error_description, {}, False

    def _extract_detailed_state_from_environment(self) -> Dict[str, Any]:
        """Extract detailed state information from the game environment"""
        if not self.game_environment:
            if self.verbose:
                print("No game environment connected")
            return {}, False
        
        try:
            # Use the environment's dedicated method to get the detailed state
            if hasattr(self.game_environment, 'get_detailed_game_state_for_agent'):
                try:
                    detailed_state, end_game = self.game_environment.get_detailed_game_state_for_agent()
                    if detailed_state and isinstance(detailed_state, dict):
                        
                        # Check data consistency
                        attempts = detailed_state.get("attempts", 0)
                        previous_clicks = detailed_state.get("previous_clicks", [])
                        
                        if attempts > 0 and not previous_clicks:
                            if self.verbose:
                                print(f"Data consistency issue detected: attempts={attempts}, previous_clicks={previous_clicks}")
                            
                            # Internal retry mechanism
                            max_internal_retries = 15
                            retry_delay = 0.1
                            
                            for retry_attempt in range(max_internal_retries):
                                if self.verbose:
                                    print(f"Internal retry {retry_attempt + 1}/{max_internal_retries}")
                                
                                time.sleep(retry_delay)
                                retry_state, end_game = self.game_environment.get_detailed_game_state_for_agent()
                                
                                if retry_state and isinstance(retry_state, dict):
                                    retry_attempts = retry_state.get("attempts", 0)
                                    retry_previous_clicks = retry_state.get("previous_clicks", [])
                                    
                                    if retry_attempts > 0 and retry_previous_clicks:
                                        if self.verbose:
                                            print(f"Data consistency restored: attempts={retry_attempts}")
                                        detailed_state = retry_state
                                        break
                                    elif retry_attempts == 0:
                                        if self.verbose:
                                            print(f"Game state reset: attempts={retry_attempts}")
                                        detailed_state = retry_state
                                        break
                                    
                                    retry_delay = min(retry_delay * 1.2, 0.5)
                            else:
                                if self.verbose:
                                    print(f"Internal retries exhausted. Using original state with warning.")
                        
                        if self.verbose:
                            current_attempts = detailed_state.get("attempts", 0)
                            current_previous_clicks = detailed_state.get("previous_clicks", [])
                            
                            print(f"Successfully got detailed state from environment:")
                            print(f"  Current state: {detailed_state.get('current_state', 'unknown')}")
                            print(f"  Score: {detailed_state.get('current_score', 0)}")
                            print(f"  Sequence: {detailed_state.get('input_length', 0)}/{detailed_state.get('sequence_length', 0)}")
                            print(f"  In correct sequence: {detailed_state.get('currently_in_correct_sequence', False)}")
                            print(f"  Needs restart: {detailed_state.get('needs_restart_from_beginning', False)}")
                            print(f"  Available colors: {len(detailed_state.get('available_colors', []))}")
                            print(f"  Attempts: {current_attempts}")
                            print(f"  Previous clicks: {current_previous_clicks}")
                            
                            # Check data consistency status
                            if current_attempts > 0 and not current_previous_clicks:
                                print("  WARNING: Data consistency issue still present!")
                            elif current_attempts > 0 and current_previous_clicks:
                                print("  INFO: Data consistency verified")
                            
                            # Check if fallback was used
                            debug_info = detailed_state.get('debug_info', {})
                            if debug_info.get('fallback_used', False):
                                print("  Warning: Using fallback state")
                            elif debug_info.get('game_module_available', False):
                                print("  Success: Got real-time game state")
                            
                            # Check for consistency warnings
                            if "data_consistency_warning" in detailed_state:
                                print("  WARNING: Data consistency could not be resolved")
                        
                        return detailed_state, end_game
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error using environment's get_detailed_game_state_for_agent: {e}")
            
            else:
                if self.verbose:
                    print("Environment does not have get_detailed_game_state_for_agent method")
                    
        except Exception as e:
            if self.verbose:
                print(f"Error in _extract_detailed_state_from_environment: {e}")
                import traceback
                traceback.print_exc()
        
        return {}, False
    
    def _check_sequence_match(self, player_input: List, correct_sequence: List) -> bool:
        """Check if the player's input matches the correct sequence"""
        if not player_input:
            return True
        if not correct_sequence:
            return False
        
        for i in range(len(player_input)):
            if i >= len(correct_sequence) or player_input[i] != correct_sequence[i]:
                return False
        return True

    def _extract_detailed_state_from_environment(self) -> tuple:
        """Extract detailed state information from the game environment"""
        if not self.game_environment:
            if self.verbose:
                print("No game environment connected")
            return {}, False
        
        try:
            # Use the environment's dedicated method to get the detailed state
            if hasattr(self.game_environment, 'get_detailed_game_state_for_agent'):
                try:
                    detailed_state, end_game = self.game_environment.get_detailed_game_state_for_agent()
                    if detailed_state and isinstance(detailed_state, dict):
                        
                        # Check data consistency
                        attempts = detailed_state.get("attempts", 0)
                        previous_clicks = detailed_state.get("previous_clicks", [])
                        
                        if attempts > 0 and not previous_clicks:
                            if self.verbose:
                                print(f"Data consistency issue detected: attempts={attempts}, previous_clicks={previous_clicks}")
                            
                            # Internal retry mechanism
                            max_internal_retries = 15
                            retry_delay = 0.1
                            
                            for retry_attempt in range(max_internal_retries):
                                if self.verbose:
                                    print(f"Internal retry {retry_attempt + 1}/{max_internal_retries}")
                                
                                time.sleep(retry_delay)
                                retry_state, end_game = self.game_environment.get_detailed_game_state_for_agent()
                                
                                if retry_state and isinstance(retry_state, dict):
                                    retry_attempts = retry_state.get("attempts", 0)
                                    retry_previous_clicks = retry_state.get("previous_clicks", [])
                                    
                                    if retry_attempts > 0 and retry_previous_clicks:
                                        if self.verbose:
                                            print(f"Data consistency restored: attempts={retry_attempts}")
                                        detailed_state = retry_state
                                        break
                                    elif retry_attempts == 0:
                                        if self.verbose:
                                            print(f"Game state reset: attempts={retry_attempts}")
                                        detailed_state = retry_state
                                        break
                                    
                                    retry_delay = min(retry_delay * 1.2, 0.5)
                            else:
                                if self.verbose:
                                    print(f"Internal retries exhausted. Using original state with warning.")
                        
                        # Verbose output processing
                        if self.verbose:
                            current_attempts = detailed_state.get("attempts", 0)
                            current_previous_clicks = detailed_state.get("previous_clicks", [])
                            
                            print(f"Successfully got detailed state from environment:")
                            print(f"  Current state: {detailed_state.get('current_state', 'unknown')}")
                            print(f"  Score: {detailed_state.get('current_score', 0)}")
                            print(f"  Sequence: {detailed_state.get('input_length', 0)}/{detailed_state.get('sequence_length', 0)}")
                            print(f"  In correct sequence: {detailed_state.get('currently_in_correct_sequence', False)}")
                            print(f"  Needs restart: {detailed_state.get('needs_restart_from_beginning', False)}")
                            print(f"  Available colors: {len(detailed_state.get('available_colors', []))}")
                            print(f"  Attempts: {current_attempts}")
                            print(f"  Previous clicks: {current_previous_clicks}")
                            
                            # Check data consistency status
                            if current_attempts > 0 and not current_previous_clicks:
                                print("  WARNING: Data consistency issue still present!")
                            elif current_attempts > 0 and current_previous_clicks:
                                                               print("  INFO: Data consistency verified")
                            
                            # Check if fallback was used
                            debug_info = detailed_state.get('debug_info', {})
                            if debug_info.get('fallback_used', False):
                                print("  Warning: Using fallback state")
                            elif debug_info.get('game_module_available', False):
                                print("  Success: Got real-time game state")
                            
                            # Check for consistency warnings
                            if "data_consistency_warning" in detailed_state:
                                print("  WARNING: Data consistency could not be resolved")
                        
                        return detailed_state, end_game
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error using environment's get_detailed_game_state_for_agent: {e}")
            
            else:
                if self.verbose:
                    print("Environment does not have get_detailed_game_state_for_agent method")
                    
        except Exception as e:
            if self.verbose:
                print(f"Error in _extract_detailed_state_from_environment: {e}")
                import traceback
                traceback.print_exc()
        
        return {}, False