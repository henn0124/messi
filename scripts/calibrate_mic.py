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
print(f"Added to Python path: {project_root}")

try:
    from src.core.config import Settings
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

class MicCalibration:
    def __init__(self):
        self.settings = Settings()
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Test durations
        self.ambient_duration = 5    # seconds for ambient noise
        self.speech_duration = 10    # increased to 10 seconds for better wake word testing
        self.test_phrases = 3        # number of "Hey Messy" test phrases
        
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
            
            # Initialize audio stream
            self.stream = self.p.open(
                rate=44100,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=1,  # Maono mic
                frames_per_buffer=1024
            )
            
            # Test 1: Ambient Noise
            print("\n=== Testing Ambient Noise ===")
            print(f"Please remain quiet for {self.ambient_duration} seconds...")
            await self._measure_levels(self.ambient_duration, self.ambient_levels)
            
            # Test 2: Wake Word Testing
            print(f"\n=== Testing Wake Word Detection ===")
            print(f"Please say 'Hey Messy' {self.test_phrases} times")
            print("Wait for the prompt between each phrase")
            
            for i in range(self.test_phrases):
                print(f"\nPhrase {i+1}/{self.test_phrases}: Say 'Hey Messy' now...")
                start_time = time.time()
                await self._measure_levels(3, self.speech_levels)  # 3 seconds per phrase
                self.detection_gaps.append(time.time() - start_time)
                await asyncio.sleep(2)  # Pause between phrases
            
            # Calculate and display results
            await self._analyze_results()
            
        except Exception as e:
            print(f"Calibration error: {e}")
        finally:
            self.cleanup()

    async def _measure_levels(self, duration: int, level_storage: list):
        """Measure audio levels for specified duration"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Read audio
                data = self.stream.read(1024, exception_on_overflow=False)
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
        """Analyze and display calibration results with wake word recommendations"""
        if not self.ambient_levels or not self.speech_levels:
            print("No data collected!")
            return
            
        # Calculate metrics
        ambient_mean = np.mean(self.ambient_levels)
        ambient_std = np.std(self.ambient_levels)
        speech_mean = np.mean(self.speech_levels)
        speech_std = np.std(self.speech_levels)
        
        # Calculate wake word specific recommendations
        avg_detection_time = np.mean(self.detection_gaps)
        detection_std = np.std(self.detection_gaps)
        
        # Calculate recommended settings
        recommended = {
            "WAKE_WORD_MIN_VOLUME": max(speech_mean * 0.4, 150),
            "AUDIO_SILENCE_THRESHOLD": max(ambient_mean * 2, 100),
            "WAKE_WORD_MAX_VOLUME": min(self.peak_level * 1.2, 5000),
            "WAKE_WORD_THRESHOLD": 0.85,  # Start more sensitive
            "WAKE_WORD_DETECTION_WINDOW": max(avg_detection_time * 0.8, 0.3),
            "WAKE_WORD_CONSECUTIVE_FRAMES": 1 if speech_std < 100 else 2
        }
        
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
        
        print("\nWake Word Timing:")
        print(f"  Average Detection Time: {avg_detection_time:.2f}s")
        print(f"  Detection Time Std Dev: {detection_std:.2f}s")
        
        print("\nRecommended Settings:")
        for key, value in recommended.items():
            if isinstance(value, float):
                print(f"{key} = {value:.2f}")
            else:
                print(f"{key} = {value}")
        
        # Save comprehensive results
        results_file = project_root / "calibration_results.txt"
        with open(results_file, "w") as f:
            f.write("=== Microphone Calibration Results ===\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Raw Metrics:\n")
            f.write(f"Ambient Noise Mean: {ambient_mean:.1f}\n")
            f.write(f"Ambient Noise Std Dev: {ambient_std:.1f}\n")
            f.write(f"Speech Level Mean: {speech_mean:.1f}\n")
            f.write(f"Speech Level Std Dev: {speech_std:.1f}\n")
            f.write(f"Peak Level: {self.peak_level:.1f}\n")
            f.write(f"Average Detection Time: {avg_detection_time:.2f}s\n\n")
            
            f.write("Add these to your .env file:\n")
            f.write("# Wake Word Settings (calibrated)\n")
            for key, value in recommended.items():
                if isinstance(value, float):
                    f.write(f"{key}={value:.2f}\n")
                else:
                    f.write(f"{key}={value}\n")
        
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