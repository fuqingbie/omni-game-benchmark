"""
Audio System for Cooperative Command Game

Handles audio message generation, TTS, and audio event management with support for
multiple voice models and languages.
"""

import os
import time
import json
import random
import threading
import logging
from queue import Queue
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Try to import TTS libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Set TTS availability based on available libraries
TTS_AVAILABLE = PYTTSX3_AVAILABLE or GTTS_AVAILABLE

# Try to import audio effects (optional)
try:
    from audio_effects import AudioEffects, get_noise_effects_for_difficulty
except ImportError:
    AudioEffects = None
    get_noise_effects_for_difficulty = lambda *args: None

# Simple voice asset class
@dataclass
class VoiceAsset:
    """Represents a voice message or audio asset."""
    text: str
    priority: int = 1  # 1-5, higher is more urgent
    voice_id: Optional[str] = None
    emotion: str = "neutral"
    duration: float = 0.0
    file_path: Optional[str] = None
    audio_path: Optional[str] = None  # Add audio_path field for compatibility

@dataclass 
class AudioMessage:
    voice_asset: VoiceAsset
    member_id: str
    member_role: str
    timestamp: float
    priority: int
    processed: bool = False
    audio_file: Optional[str] = None  # Add audio_file field


# Role to callsign mapping to ensure correct audio matching
ROLE_TO_CALLSIGN = {
    "scout": "Alpha",
    "heavy": "Bravo", 
    "medic": "Charlie",
    "engineer": "Delta",
    "sniper": "Echo",
    "support": "Foxtrot"
}


class AudioSystem:
    """Manages TTS generation and audio playback for team communications"""
    
    def __init__(self, audio_assets_dir: str = "code/coop_game/audio_assets", enable_effects: bool = True):
        self.audio_assets_dir = Path(audio_assets_dir)
        
        # Audio queue and processing
        self.audio_queue: Queue = Queue()
        self.current_audio: Optional[AudioMessage] = None
        self.is_playing = False
        
        # Audio effects system
        self.enable_effects = enable_effects and AudioEffects is not None
        self.audio_effects = AudioEffects() if self.enable_effects else None
        
        # Audio playback setup (removed TTS engine initialization)
        self.pygame_available = PYGAME_AVAILABLE
        self.tts_available = TTS_AVAILABLE  # Add this for compatibility
        if self.pygame_available:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            except Exception as e:
                logging.error(f"Failed to initialize pygame mixer: {e}")
                self.pygame_available = False
        
        # Load audio manifest for pre-generated files
        self.audio_manifest = self._load_audio_manifest()
        
        # Audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self.audio_thread_running = True
        self.audio_thread.start()
    
    def _load_audio_manifest(self) -> Dict:
        """Load the audio manifest file that maps voice lines to audio files"""
        manifest_path = self.audio_assets_dir / "audio_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load audio manifest: {e}")
                return {}
        else:
            logging.warning(f"Audio manifest not found at {manifest_path}")
            return {}
    
    def get_audio_path(self, text: str, member_role: str) -> Optional[str]:
        """Get the audio file path for given text and member role"""
        if not self.audio_manifest:
            return None
        
        # Look for assets in the "generated_files" structure
        generated_files = self.audio_manifest.get("generated_files", {})
        role_assets = generated_files.get(member_role, [])
        
        # First try exact match
        for asset in role_assets:
            if asset.get("text") == text:
                audio_path = asset.get("audio_path")
                filename = asset.get("filename")
                
                if audio_path:
                    # Try the original path first
                    if Path(audio_path).exists():
                        return audio_path
                    
                    # Try removing the "code/coop_game/" prefix
                    alt_audio_path = audio_path.replace("code/coop_game/", "")
                    if Path(alt_audio_path).exists():
                        return alt_audio_path
                    
                    # Try just the filename in the audio_assets directory
                    if filename:
                        filename_path = self.audio_assets_dir / filename
                        if filename_path.exists():
                            return str(filename_path)
                    
                    # Extract filename from the path and try in audio_assets directory
                    audio_filename = Path(audio_path).name
                    filename_path = self.audio_assets_dir / audio_filename
                    if filename_path.exists():
                        return str(filename_path)
                break
        
        # If no exact match, try fuzzy matching for dynamic messages
        best_match = self._find_best_audio_match(text, role_assets, member_role)
        if best_match:
            audio_path = best_match.get("audio_path")
            filename = best_match.get("filename")
            
            if audio_path:
                # Try the various path options
                for path_option in [
                    audio_path,
                    audio_path.replace("code/coop_game/", ""),
                    str(self.audio_assets_dir / filename) if filename else None,
                    str(self.audio_assets_dir / Path(audio_path).name)
                ]:
                    if path_option and Path(path_option).exists():
                        logging.debug(f"Found fuzzy audio match for '{text}': {path_option}")
                        return path_option
        
        logging.warning(f"Audio file not found for text: '{text}' (role: {member_role})")
        return None
    
    def _find_best_audio_match(self, text: str, role_assets: List[Dict], member_role: str) -> Optional[Dict]:
        """Find the best matching audio asset for dynamic text using fuzzy matching"""
        if not role_assets:
            return None
        
        text_lower = text.lower()
        best_match = None
        best_score = 0
        
        # Get expected callsign for this role
        expected_callsign = ROLE_TO_CALLSIGN.get(member_role, "").lower()
        if not expected_callsign:
            logging.warning(f"Unknown role for callsign validation: {member_role}")
            return None
        
        # Common text patterns and their keywords
        pattern_keywords = {
            "position": ["moving", "position", "coordinates"],
            "status": ["ready", "standing by", "all clear", "position"],
            "damage": ["taking damage", "hit", "health", "critical"],
            "enemy": ["enemy", "contact", "engaging", "target"],
            "command": ["command", "acknowledged", "roger", "affirmative"],
            "negative": ["negative", "cannot", "unable", "blocked"],
            "ammo": ["ammo", "ammunition", "low on", "depleted"],
            "recon": ["reconnaissance", "scouting", "area", "survey"],
        }
        
        for asset in role_assets:
            asset_text = asset.get("text", "").lower()
            score = 0
            
            # CRITICAL: Validate that this audio contains the correct callsign
            if expected_callsign not in asset_text:
                logging.debug(f"Rejecting audio match - wrong callsign. Expected '{expected_callsign}' in '{asset_text}'")
                continue  # Skip this asset if it doesn't contain the correct callsign
            
            # Check for key phrase matches
            for pattern, keywords in pattern_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    if any(keyword in asset_text for keyword in keywords):
                        score += 10
                        break
            
            # Check for specific message type matches
            if "moving to position" in text_lower and "position" in asset_text:
                score += 15
            if "adjusted to stay within" in text_lower and "position" in asset_text:
                score += 10
            if "taking damage" in text_lower and ("damage" in asset_text or "taking" in asset_text):
                score += 15
            if "enemy fire" in text_lower and ("enemy" in asset_text or "fire" in asset_text):
                score += 15
            if "command acknowledged" in text_lower and "command" in asset_text:
                score += 15
            
            # Word overlap bonus
            text_words = set(text_lower.split())
            asset_words = set(asset_text.split())
            common_words = text_words.intersection(asset_words)
            score += len(common_words) * 2
            
            # Bonus for exact callsign match
            if expected_callsign in asset_text:
                score += 5
            
            if score > best_score:
                best_score = score
                best_match = asset
        
        # Only return match if score is reasonable and callsign was validated
        if best_match and best_score >= 5:
            logging.debug(f"Found valid audio match for '{text}' with callsign '{expected_callsign}': {best_match.get('text', '')}")
            return best_match
        else:
            logging.debug(f"No valid audio match found for '{text}' with callsign '{expected_callsign}' (best score: {best_score})")
            return None
    
    def queue_audio_message(self, voice_asset: VoiceAsset, member_id: str, member_role: str, 
                           noise_level: float = 0.0, difficulty: str = "normal", custom_effects=None):
        """Queue an audio message for playback"""
        # Use pre-generated audio file or fallback to the one specified in voice_asset
        audio_file = voice_asset.audio_path
        if not audio_file or not Path(audio_file).exists():
            # Try to find the audio file using the manifest
            audio_file = self.get_audio_path(voice_asset.text, member_role)
        
        message = AudioMessage(
            voice_asset=voice_asset,
            member_id=member_id,
            member_role=member_role,  # Add missing member_role parameter
            timestamp=time.time(),
            priority=voice_asset.priority,  # Add missing priority parameter
            audio_file=audio_file
        )
        
        # Apply audio effects if enabled and noise_level > 0
        if self.enable_effects and self.audio_effects and noise_level > 0 and audio_file:
            effects = get_noise_effects_for_difficulty(difficulty, noise_level)
            if effects:
                try:
                    processed_file = self.audio_effects.process_audio_file(audio_file, effects)
                    message.audio_file = processed_file
                except Exception as e:
                    logging.warning(f"Failed to apply audio effects: {e}")
        
        # Add to queue with priority handling
        self.audio_queue.put((voice_asset.priority, message))
    
    def _audio_processing_loop(self):
        """Main audio processing loop running in separate thread"""
        while self.audio_thread_running:
            try:
                if not self.audio_queue.empty():
                    # Get highest priority message (highest number = highest priority)
                    priority_queue = []
                    
                    # Collect all messages currently in queue
                    while not self.audio_queue.empty():
                        try:
                            priority_queue.append(self.audio_queue.get_nowait())
                        except:
                            break
                    
                    if priority_queue:
                        # Sort by priority (descending) and timestamp (ascending)
                        priority_queue.sort(key=lambda x: (-x[0], x[1].timestamp))
                        
                        # Play highest priority message
                        _, message = priority_queue[0]
                        self._play_audio_message(message)
                        
                        # Put remaining messages back in queue
                        for priority, msg in priority_queue[1:]:
                            self.audio_queue.put((priority, msg))
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logging.error(f"Audio processing error: {e}")
                time.sleep(1)  # Longer delay on error
    
    def _play_audio_message(self, message: AudioMessage):
        """Play a single audio message"""
        self.current_audio = message
        self.is_playing = True
        
        try:
            if self.pygame_available and message.audio_file and os.path.exists(message.audio_file):
                # Play using pygame
                pygame.mixer.music.load(message.audio_file)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            else:
                # Fallback: display text only
                print(f"[AUDIO] {message.voice_asset.text}")
                # Simulate audio duration based on text length
                duration = max(2.0, len(message.voice_asset.text) * 0.05)
                time.sleep(duration)
                
        except Exception as e:
            logging.error(f"Audio playback error: {e}")
            # Fallback to text display
            print(f"[AUDIO] {message.voice_asset.text}")
            time.sleep(2.0)
        
        finally:
            self.current_audio = None
            self.is_playing = False
    
    def stop_current_audio(self):
        """Stop currently playing audio"""
        if self.pygame_available and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        self.is_playing = False
        self.current_audio = None
    
    def clear_audio_queue(self):
        """Clear all pending audio messages"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
    
    def get_audio_status(self) -> Dict:
        """Get current audio system status"""
        return {
            "is_playing": self.is_playing,
            "current_audio": {
                "text": self.current_audio.voice_asset.text if self.current_audio else None,
                "member_id": self.current_audio.member_id if self.current_audio else None,
                "priority": self.current_audio.voice_asset.priority if self.current_audio else None
            } if self.current_audio else None,
            "queue_size": self.audio_queue.qsize(),
            "tts_available": self.tts_available,
            "audio_available": self.pygame_available
        }
    
    def shutdown(self):
        """Shutdown audio system"""
        self.audio_thread_running = False
        self.stop_current_audio()
        self.clear_audio_queue()
        
        if self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.pygame_available:
            pygame.mixer.quit()


class AudioLibraryManager:
    """Manages pre-recorded audio assets and retrieval"""
    
    def __init__(self, retrieval_function=None):
        self.retrieval_function = retrieval_function
        self.audio_cache = {}
    
    def get_audio_assets(self, description: str, max_count: int = 10) -> List[str]:
        """Get relevant audio assets using retrieval system"""
        if not self.retrieval_function:
            return []
        
        try:
            # Use the retrieval system to find relevant audio files
            from ..retrieval_sprites import retrieve_relevant_files_from_store
            _, audio_files = retrieve_relevant_files_from_store(
                description, 
                top_k_audio=max_count,
                remove_loop=True
            )
            return audio_files
        except Exception as e:
            logging.error(f"Audio retrieval error: {e}")
            return []
    
    def enrich_voice_assets(self, voice_assets: List[VoiceAsset], member_role: str) -> List[VoiceAsset]:
        """Enrich voice assets with relevant audio files"""
        enriched_assets = []
        
        for asset in voice_assets:
            # Try to find matching audio for this text
            search_description = f"{member_role} military communication {asset.text}"
            audio_files = self.get_audio_assets(search_description, max_count=3)
            
            if audio_files:
                # Create variant with audio file
                enriched_asset = VoiceAsset(
                    text=asset.text,
                    audio_path=audio_files[0],  # Use best match
                    priority=asset.priority
                )
                enriched_assets.append(enriched_asset)
            else:
                # Keep original asset (will use TTS)
                enriched_assets.append(asset)
        
        return enriched_assets