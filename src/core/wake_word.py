"""
Wake Word Detector using Porcupine
--------------------------------
Detects the wake word "Hey Messy" using Picovoice's Porcupine engine.
"""

import pvporcupine
import struct
import numpy as np
from typing import Optional, Dict, Callable
from pathlib import Path

class WakeWordDetector:
    def __init__(self, config: Dict):
        """Initialize wake word detector"""
        self.config = config
        
        # Get the project root directory (parent of src/core)
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models" / "hey-messy_en_raspberry-pi_v3_0_0.ppn"
        
        self.porcupine = pvporcupine.create(
            access_key=config["access_key"],
            keyword_paths=[str(model_path)],
            sensitivities=[config["sensitivity"]]
        )
        
        # Audio settings
        self.frame_length = self.porcupine.frame_length
        self.sample_rate = self.porcupine.sample_rate
        
        print(f"\nWake Word Configuration:")
        print(f"  Frame Length: {self.frame_length}")
        print(f"  Sample Rate: {self.sample_rate}")
        print(f"  Sensitivity: {config['sensitivity']}")
        print(f"  Model Path: {model_path}")

    def process_frame(self, pcm: bytes) -> bool:
        """
        Process a single frame of audio
        Returns True if wake word detected
        """
        try:
            # Convert bytes to int16 array
            pcm = struct.unpack_from("h" * self.frame_length, pcm)
            
            # Process with Porcupine
            result = self.porcupine.process(pcm)
            return result == 0  # Wake word detected
            
        except Exception as e:
            print(f"Error processing audio frame: {e}")
            return False

    def cleanup(self):
        """Clean up resources"""
        if self.porcupine:
            self.porcupine.delete()