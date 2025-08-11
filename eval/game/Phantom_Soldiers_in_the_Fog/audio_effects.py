"""
Audio Effects Module

Provides real-time audio effects for military communication simulation,
including static, crackle, fade, and radio interference effects.
"""

import numpy as np
import pygame
import io
import wave
import random
from typing import Optional, Tuple
from pathlib import Path
import tempfile
import os

class AudioEffects:
    """Generate and apply audio effects for realistic communication simulation"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
        # Pre-generated noise patterns for efficiency
        self._static_noise = self._generate_static_pattern(duration=1.0)
        self._crackle_noise = self._generate_crackle_pattern(duration=1.0)
        
    def _generate_static_pattern(self, duration: float = 1.0) -> np.ndarray:
        """Generate white noise static pattern"""
        samples = int(self.sample_rate * duration)
        # White noise with some filtering for radio-like static
        noise = np.random.normal(0, 0.3, samples)
        
        # Apply basic filtering to make it sound more like radio static
        # Simple low-pass filtering effect
        for i in range(1, len(noise)):
            noise[i] = 0.7 * noise[i] + 0.3 * noise[i-1]
            
        return noise
    
    def _generate_crackle_pattern(self, duration: float = 1.0) -> np.ndarray:
        """Generate crackle/pop noise pattern"""
        samples = int(self.sample_rate * duration)
        noise = np.zeros(samples)
        
        # Random pops and crackles
        num_pops = random.randint(3, 8)
        for _ in range(num_pops):
            pop_start = random.randint(0, samples - 100)
            pop_duration = random.randint(10, 50)
            pop_intensity = random.uniform(0.5, 1.0)
            
            # Create a quick pop sound
            pop_samples = np.linspace(0, pop_intensity, pop_duration // 2)
            pop_samples = np.concatenate([pop_samples, pop_samples[::-1]])
            
            end_idx = min(pop_start + len(pop_samples), samples)
            noise[pop_start:end_idx] += pop_samples[:end_idx - pop_start]
            
        return noise
    
    def apply_static_effect(self, audio_data: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Apply static noise to audio data"""
        if len(audio_data) == 0:
            return audio_data
            
        # Extend static pattern to match audio length
        audio_length = len(audio_data)
        static_length = len(self._static_noise)
        
        if audio_length > static_length:
            # Repeat static pattern
            repetitions = (audio_length // static_length) + 1
            extended_static = np.tile(self._static_noise, repetitions)[:audio_length]
        else:
            extended_static = self._static_noise[:audio_length]
        
        # Mix with original audio
        return audio_data + (extended_static * intensity)
    
    def apply_crackle_effect(self, audio_data: np.ndarray, intensity: float = 0.4) -> np.ndarray:
        """Apply crackle/pop effects to audio data"""
        if len(audio_data) == 0:
            return audio_data
            
        # Generate crackle for the audio length
        audio_length = len(audio_data)
        crackle = self._generate_crackle_pattern(audio_length / self.sample_rate)
        
        if len(crackle) != audio_length:
            # Adjust crackle length to match audio
            if len(crackle) > audio_length:
                crackle = crackle[:audio_length]
            else:
                crackle = np.pad(crackle, (0, audio_length - len(crackle)), 'constant')
        
        return audio_data + (crackle * intensity)
    
    def apply_fade_effect(self, audio_data: np.ndarray, fade_type: str = "out") -> np.ndarray:
        """Apply fade in/out effect"""
        if len(audio_data) == 0:
            return audio_data
            
        fade_samples = min(len(audio_data) // 4, self.sample_rate // 2)  # Max 0.5 second fade
        
        if fade_type == "out":
            fade_curve = np.linspace(1.0, 0.0, fade_samples)
            audio_data[-fade_samples:] *= fade_curve
        elif fade_type == "in":
            fade_curve = np.linspace(0.0, 1.0, fade_samples)
            audio_data[:fade_samples] *= fade_curve
        
        return audio_data
    
    def apply_radio_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply radio-like frequency filtering"""
        if len(audio_data) == 0:
            return audio_data
            
        # Simple high-pass filter simulation (removes some low frequencies)
        filtered = np.copy(audio_data)
        
        # Apply basic filtering by reducing low frequency components
        for i in range(1, len(filtered)):
            filtered[i] = 0.8 * filtered[i] + 0.2 * filtered[i-1]
        
        return filtered
    
    def apply_distance_effect(self, audio_data: np.ndarray, distance_factor: float = 0.5) -> np.ndarray:
        """Simulate distance by reducing volume and adding slight reverb"""
        if len(audio_data) == 0:
            return audio_data
            
        # Reduce volume
        audio_data = audio_data * (1.0 - distance_factor * 0.7)
        
        # Add simple reverb effect
        if len(audio_data) > self.sample_rate // 10:
            delay_samples = self.sample_rate // 20  # 50ms delay
            reverb = np.zeros_like(audio_data)
            reverb[delay_samples:] = audio_data[:-delay_samples] * 0.3
            audio_data = audio_data + reverb
        
        return audio_data
    
    def process_audio_file(self, input_path: str, effects: list, output_path: Optional[str] = None) -> str:
        """
        Process an audio file with specified effects and return path to processed file
        
        Args:
            input_path: Path to input audio file
            effects: List of effect dictionaries, e.g.:
                     [{"type": "static", "intensity": 0.3}, 
                      {"type": "crackle", "intensity": 0.2}]
            output_path: Optional output path (if None, creates temp file)
        
        Returns:
            Path to processed audio file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        try:
            # Load audio file
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=512)
            sound = pygame.mixer.Sound(input_path)
            
            # Convert to numpy array
            audio_array = pygame.sndarray.array(sound)
            
            # Handle stereo to mono conversion
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Convert to float for processing
            audio_data = audio_array.astype(np.float32) / 32768.0
            
            # Apply effects
            for effect in effects:
                effect_type = effect.get("type", "")
                intensity = effect.get("intensity", 0.3)
                
                if effect_type == "static":
                    audio_data = self.apply_static_effect(audio_data, intensity)
                elif effect_type == "crackle":
                    audio_data = self.apply_crackle_effect(audio_data, intensity)
                elif effect_type == "fade":
                    fade_type = effect.get("fade_type", "out")
                    audio_data = self.apply_fade_effect(audio_data, fade_type)
                elif effect_type == "radio":
                    audio_data = self.apply_radio_filter(audio_data)
                elif effect_type == "distance":
                    distance_factor = effect.get("distance_factor", 0.5)
                    audio_data = self.apply_distance_effect(audio_data, distance_factor)
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Convert back to 16-bit integers
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create output file
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                output_path = temp_file.name
                temp_file.close()
            
            # Save processed audio
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {e}")

# Convenience functions for common noise types
def get_noise_effects_for_difficulty(difficulty: str, noise_level: float) -> list:
    """Get appropriate noise effects based on difficulty and noise level"""
    if noise_level <= 0:
        return []
    
    effects = []
    
    # Adjust effects based on difficulty
    if difficulty == "normal":
        # Minimal effects
        if random.random() < noise_level:
            effects.append({"type": "static", "intensity": 0.1})
    
    elif difficulty == "medium":
        # Moderate effects
        if random.random() < noise_level:
            effects.append({"type": "static", "intensity": 0.2})
            effects.append({"type": "crackle", "intensity": 0.1})
    
    elif difficulty == "hard":
        # Heavy effects
        if random.random() < noise_level:
            effects.append({"type": "static", "intensity": 0.3})
            effects.append({"type": "crackle", "intensity": 0.2})
            effects.append({"type": "radio_filter", "intensity": 1.0})
    
    return effects