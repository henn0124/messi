import pyaudio
import numpy as np
import time
import asyncio
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.config import Settings

class MicCalibration:
    def __init__(self):
        self.settings = Settings()
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Test durations
        self.ambient_duration = 5
        self.speech_duration = 5
        self.test_phrases = 3
        
        # Metrics storage
        self.ambient_levels = []
        self.speech_levels = []
        self.peak_level = 0
        
        # Wake word specific metrics
        self.wake_word_detections = []
        self.detection_gaps = []
        self.false_triggers = 0

    async def run_calibration(self):
        """Run complete microphone calibration"""
        try:
            print("\n=== Microphone Calibration Tool ===")
            
            # List available devices
            print("\nAvailable Audio Devices:")
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                print(f"Index {i}: {dev['name']}")
                print(f"  Max Input Channels: {dev['maxInputChannels']}")
                print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
            
            # Find Maono mic
            mic_index = None
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                if "maono" in dev['name'].lower():
                    mic_index = i
                    print(f"\nFound Maono mic at index {i}")
                    break
            
            if mic_index is None:
                print("Maono microphone not found!")
                return
            
            # Initialize audio stream
            self.stream = self.p.open(
                rate=44100,  # Native rate
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=2048
            )
            
            # Test 1: Ambient Noise
            print("\n=== Testing Ambient Noise ===")
            print(f"Please remain quiet for {self.ambient_duration} seconds...")
            await self._measure_levels(self.ambient_duration, self.ambient_levels)
            
            # Test 2: Speech Levels
            print("\n=== Testing Speech Levels ===")
            print(f"Please speak 'Hey Messy' several times at your normal volume")
            print(f"Recording for {self.speech_duration} seconds...")
            await self._measure_levels(self.speech_duration, self.speech_levels)
            
            # Calculate and display results
            await self._analyze_results()
            
        except Exception as e:
            print(f"Calibration error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    async def _measure_levels(self, duration: int, level_storage: list):
        """Measure audio levels with visual feedback"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Read audio
                data = self.stream.read(2048, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Calculate levels
                level = np.abs(audio_array).mean()
                peak = np.abs(audio_array).max()
                
                # Store metrics
                level_storage.append(level)
                self.peak_level = max(self.peak_level, peak)
                
                # Visual feedback
                level_bar = "#" * int(level / 100)
                print(f"\rLevel: {level_bar}", end="", flush=True)
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"Error measuring levels: {e}")
                break
        
        print()  # New line after progress

    async def _analyze_results(self):
        """Analyze and display calibration results"""
        if not self.ambient_levels or not self.speech_levels:
            print("No data collected!")
            return
            
        # Calculate metrics
        ambient_mean = np.mean(self.ambient_levels)
        ambient_std = np.std(self.ambient_levels)
        speech_mean = np.mean(self.speech_levels)
        speech_std = np.std(self.speech_levels)
        
        # Calculate recommended settings
        min_volume = max(speech_mean * 0.4, 150)
        silence_threshold = max(ambient_mean * 2, 80)
        max_volume = min(self.peak_level * 1.2, 3351)
        
        print("\n=== Calibration Results ===")
        
        print("\nAmbient Noise:")
        print(f"  Mean Level: {ambient_mean:.1f}")
        print(f"  Std Dev: {ambient_std:.1f}")
        print(f"  Range: {min(self.ambient_levels):.1f} - {max(self.ambient_levels):.1f}")
        
        print("\nSpeech Levels:")
        print(f"  Mean Level: {speech_mean:.1f}")
        print(f"  Std Dev: {speech_std:.1f}")
        print(f"  Range: {min(self.speech_levels):.1f} - {max(self.speech_levels):.1f}")
        
        print(f"\nPeak Level: {self.peak_level:.1f}")
        
        print("\nRecommended Settings:")
        print(f"WAKE_WORD_MIN_VOLUME = {min_volume:.0f}")
        print(f"AUDIO_SILENCE_THRESHOLD = {silence_threshold:.0f}")
        print(f"WAKE_WORD_MAX_VOLUME = {max_volume:.0f}")
        
        # Save results
        results_file = project_root / "calibration_results.txt"
        with open(results_file, "w") as f:
            f.write("=== Microphone Calibration Results ===\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Raw Metrics:\n")
            f.write(f"Ambient Noise Mean: {ambient_mean:.1f}\n")
            f.write(f"Ambient Noise Std Dev: {ambient_std:.1f}\n")
            f.write(f"Speech Level Mean: {speech_mean:.1f}\n")
            f.write(f"Speech Level Std Dev: {speech_std:.1f}\n")
            f.write(f"Peak Level: {self.peak_level:.1f}\n\n")
            
            f.write("Recommended Settings:\n")
            f.write(f"WAKE_WORD_MIN_VOLUME = {min_volume:.0f}\n")
            f.write(f"AUDIO_SILENCE_THRESHOLD = {silence_threshold:.0f}\n")
            f.write(f"WAKE_WORD_MAX_VOLUME = {max_volume:.0f}\n")
        
        print(f"\nResults saved to {results_file}")

    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

if __name__ == "__main__":
    calibrator = MicCalibration()
    asyncio.run(calibrator.run_calibration()) 