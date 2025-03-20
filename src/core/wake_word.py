"""
Wake Word Detector using Porcupine
--------------------------------
Detects the wake word "Hey Messy" using Picovoice's Porcupine engine.
"""

import os
import pvporcupine
import struct
import numpy as np
from typing import Optional, Dict, Callable
from pathlib import Path
import sounddevice as sd
import threading
import queue
import time

class WakeWordDetector:
    def __init__(self, model_path, sensitivity=0.5):
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.porcupine = None
        self.stream = None
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.callback = None
        self.detection_thread = None
        self.frame_length = 512
        self.sample_rate = 16000

    def initialize(self):
        try:
            # Use the environment variable directly instead of a template string
            access_key = os.getenv('PICOVOICE_ACCESS_KEY')
            if not access_key:
                raise ValueError("PICOVOICE_ACCESS_KEY environment variable is not set")
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=['hey google'],
                sensitivities=[self.sensitivity]
            )
            return True
        except Exception as e:
            print(f"❌ Failed to initialize wake word detector: {str(e)}")
            return False

    def start(self, callback):
        if not self.porcupine:
            print("❌ Wake word detector not initialized")
            return False

        self.callback = callback
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.start()
        return True

    def stop(self):
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join()
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.porcupine:
            self.porcupine.delete()

    def _detection_loop(self):
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.frame_length,
                callback=self._audio_callback
            )
            self.stream.start()
            while self.is_running:
                try:
                    pcm = self.audio_queue.get(timeout=0.1)
                    keyword_index = self.porcupine.process(pcm)
                    if keyword_index >= 0:
                        print("🎯 Wake word detected!")
                        if self.callback:
                            self.callback()
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"❌ Error in detection loop: {str(e)}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.flatten())

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