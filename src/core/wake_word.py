import pvporcupine
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class WakeWordDetector:
    def __init__(self, settings):
        self.settings = settings
        self.porcupine = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.initialize()

    def initialize(self):
        """Initialize the wake word detector"""
        try:
            # Use more sensitive defaults
            self.threshold = getattr(self.settings, 'WAKE_WORD_THRESHOLD', 0.5)  # More sensitive default
            self.min_volume = getattr(self.settings, 'WAKE_WORD_MIN_VOLUME', 100)  # Lower minimum
            self.max_volume = getattr(self.settings, 'WAKE_WORD_MAX_VOLUME', 4000)  # Higher maximum
            
            print("\nWake Word Configuration:")
            print(f"  Sensitivity threshold: {self.threshold}")
            print(f"  Volume range: {self.min_volume} - {self.max_volume}")
            
            # Initialize Porcupine
            self.porcupine = pvporcupine.create(
                access_key=self.settings.PICOVOICE_ACCESS_KEY,
                keyword_paths=[str(self.settings.WAKE_WORD_MODEL_PATH)],
                sensitivities=[self.threshold]
            )
            
            print(f"\nPorcupine Configuration:")
            print(f"  Frame length: {self.porcupine.frame_length}")
            print(f"  Version: {self.porcupine.version}")
            
        except Exception as e:
            print(f"Error initializing wake word detector: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            self.porcupine = None

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Process an audio frame for wake word detection"""
        try:
            if self.porcupine is None:
                return False
                
            # Check audio level
            level = np.abs(audio_frame).mean()
            if level < self.min_volume:
                return False
                
            # Resample from 48kHz to 16kHz for Porcupine
            resampled = np.interp(
                np.linspace(0, len(audio_frame), self.porcupine.frame_length),
                np.arange(len(audio_frame)),
                audio_frame
            ).astype(np.int16)
                
            # Process with Porcupine
            result = self.porcupine.process(resampled)
            return result >= 0
            
        except Exception as e:
            print(f"Error processing audio frame: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            return False

    def __del__(self):
        """Clean up resources"""
        if self.porcupine:
            self.porcupine.delete()
        if self.executor:
            self.executor.shutdown()