"""
Audio Processing System for Messi Assistant
----------------------------------------

This module provides centralized audio processing with standardized settings
and quality control. It handles all audio transformations and analysis for
the assistant's audio pipeline.

Key Features:
    1. Audio Processing:
        - Sample rate conversion
        - Channel management
        - Level normalization
        - Quality metrics
    
    2. Quality Control:
        - Voice activity detection
        - Signal-to-noise analysis
        - Level monitoring
        - Duration validation
    
    3. Audio Analysis:
        - RMS level calculation
        - Peak detection
        - Voice percentage
        - SNR calculation
    
    4. Format Management:
        - WAV file handling
        - Buffer management
        - Temporary file handling
        - Multi-rate support

Settings:
    - Sample rates (input/processing/output)
    - Audio formats (bit depth, channels)
    - Quality thresholds
    - Processing parameters

Usage:
    processor = AudioProcessor()
    
    # Process audio
    audio_array, metrics = await processor.process_input(audio_data)
    
    # Save processed audio
    wav_path = await processor.save_wav(audio_array, sample_rate, "output.wav")

Integration:
    Works with:
    - AudioInterface for I/O
    - SpeechManager for STT
    - TTS for speech synthesis

Author: Your Name
Created: 2024-01-24
"""

import numpy as np
from typing import Optional, Tuple
import wave
from pathlib import Path
import pyaudio

class AudioProcessor:
    """Centralized audio processing with standardized settings"""
    
    # Standardized audio settings
    SETTINGS = {
        "sample_rates": {
            "input": 44100,      # Hardware input
            "processing": 16000,  # Wake word and Whisper
            "output": 24000      # TTS output
        },
        "formats": {
            "width": 2,          # 16-bit
            "channels": 1,       # Mono
            "chunk_size": 4096
        },
        "quality": {
            "min_duration": 0.5,    # Seconds
            "max_duration": 30.0,    # Seconds
            "min_rms": 50,          # Minimum RMS level
            "target_rms": 2000,     # Target RMS after normalization
            "min_voice_percent": 5,  # Minimum voice activity
            "noise_floor": 100      # Background noise threshold
        }
    }

    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Track peak levels for auto-adjustment
        self.peak_rms = 0
        self.noise_floor = self.SETTINGS["quality"]["noise_floor"]
        
    async def process_input(self, audio_data: bytes, 
                          source_rate: int = SETTINGS["sample_rates"]["input"],
                          target_rate: int = SETTINGS["sample_rates"]["processing"]) -> Tuple[np.ndarray, dict]:
        """Process input audio with quality metrics"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Get initial metrics
            metrics = self._analyze_audio(audio_array, source_rate)
            
            # Check if audio meets quality standards
            if not self._check_quality(metrics):
                return None, metrics
            
            # Normalize levels
            audio_array = self._normalize_audio(audio_array, metrics["rms"])
            
            # Resample if needed
            if source_rate != target_rate:
                audio_array = self._resample_audio(audio_array, source_rate, target_rate)
            
            # Update metrics after processing
            metrics.update(self._analyze_audio(audio_array, target_rate))
            metrics["processed"] = True
            
            return audio_array, metrics
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None, {"error": str(e)}

    def _analyze_audio(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """Analyze audio and return metrics"""
        try:
            rms = np.sqrt(np.mean(np.square(audio_array)))
            peak = np.max(np.abs(audio_array))
            duration = len(audio_array) / sample_rate
            
            # Calculate voice activity
            is_voice = np.abs(audio_array) > self.noise_floor
            voice_percent = np.mean(is_voice) * 100
            
            # Update peak RMS for adaptive thresholds
            self.peak_rms = max(self.peak_rms, rms)
            
            # Calculate signal-to-noise ratio
            noise_mask = np.abs(audio_array) <= self.noise_floor
            if np.any(noise_mask):
                noise_level = np.mean(np.abs(audio_array[noise_mask]))
                snr = 20 * np.log10(peak / noise_level) if noise_level > 0 else 0
            else:
                snr = 0
            
            return {
                "rms": rms,
                "peak": peak,
                "duration": duration,
                "voice_percent": voice_percent,
                "snr": snr,
                "sample_rate": sample_rate
            }
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return {}

    def _check_quality(self, metrics: dict) -> bool:
        """Check if audio meets quality standards"""
        if not metrics:
            return False
            
        settings = self.SETTINGS["quality"]
        
        # Print detailed metrics
        print("\nAudio Quality Metrics:")
        print(f"- Duration: {metrics['duration']:.2f}s")
        print(f"- RMS Level: {metrics['rms']:.0f}")
        print(f"- Peak Level: {metrics['peak']:.0f}")
        print(f"- Voice Activity: {metrics['voice_percent']:.1f}%")
        print(f"- SNR: {metrics['snr']:.1f}dB")
        
        # Check duration
        if metrics["duration"] < settings["min_duration"]:
            print("❌ Audio too short")
            return False
        if metrics["duration"] > settings["max_duration"]:
            print("❌ Audio too long")
            return False
            
        # Check levels
        if metrics["rms"] < settings["min_rms"]:
            print("⚠️  Low audio level - will apply gain")
        
        # Check voice activity
        if metrics["voice_percent"] < settings["min_voice_percent"]:
            print("❌ Insufficient voice activity")
            return False
            
        return True

    def _normalize_audio(self, audio_array: np.ndarray, current_rms: float) -> np.ndarray:
        """Normalize audio with better gain control"""
        try:
            if current_rms > 0:
                target = self.SETTINGS["quality"]["target_rms"]
                gain = min(target / current_rms, 20.0)  # Limit maximum gain
                
                print(f"Applying gain: {gain:.1f}x")
                
                # Apply gain with soft clipping
                normalized = audio_array * gain
                # Soft clip to prevent harsh distortion
                normalized = np.tanh(normalized / 32767.0) * 32767.0
                
                return normalized.astype(np.int16)
            return audio_array
            
        except Exception as e:
            print(f"Error normalizing audio: {e}")
            return audio_array

    def _resample_audio(self, audio_array: np.ndarray, 
                       source_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio with high quality"""
        try:
            if source_rate == target_rate:
                return audio_array
                
            # Calculate resampling ratio
            ratio = target_rate / source_rate
            
            # Generate time bases
            original_length = len(audio_array)
            new_length = int(original_length * ratio)
            
            # Use linear interpolation for simple resampling
            resampled = np.interp(
                np.linspace(0, original_length - 1, new_length, endpoint=False),
                np.arange(original_length),
                audio_array
            ).astype(np.int16)
            
            return resampled
            
        except Exception as e:
            print(f"Error resampling audio: {e}")
            return audio_array

    async def save_wav(self, audio_array: np.ndarray, 
                      sample_rate: int, filename: str) -> Optional[Path]:
        """Save audio to WAV file"""
        try:
            file_path = self.temp_dir / filename
            with wave.open(str(file_path), 'wb') as wav_file:
                wav_file.setnchannels(self.SETTINGS["formats"]["channels"])
                wav_file.setsampwidth(self.SETTINGS["formats"]["width"])
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            return file_path
            
        except Exception as e:
            print(f"Error saving WAV: {e}")
            return None 