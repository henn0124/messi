"""
Audio Test Suite for Messi Assistant
----------------------------------
Tests audio input/output functionality using ALSA.
"""

import alsaaudio
import numpy as np
import wave
import asyncio
import time
import os
import sys
from datetime import datetime
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.config import Settings
from src.core.wake_word import WakeWordDetector

class AudioTester:
    def __init__(self):
        # Initialize settings
        self.settings = Settings()
        
        # Audio configuration from settings
        self.audio_config = self.settings.audio
        
        # Audio settings
        self.input_rate = self.audio_config["input"]["rate"]
        self.output_rate = self.audio_config["output"]["rate"]
        self.processing_rate = self.audio_config["input"]["processing_rate"]
        self.period_size = self.audio_config["input"]["period_size"]
        self.channels = self.audio_config["input"]["channels"]
        
        # Volume thresholds
        self.min_volume = self.settings.WAKE_WORD_MIN_VOLUME
        self.max_volume = self.settings.WAKE_WORD_MAX_VOLUME
        
        # Initialize streams
        self.input_stream = None
        self.output_stream = None
        
        # Initialize wake word detector
        self.wake_word_detector = WakeWordDetector(self.settings)
        self.frame_length = self.wake_word_detector.porcupine.frame_length
        self.samples_per_frame = int(self.frame_length * self.input_rate / self.processing_rate)
        
        # Test state
        self.running = False
        self.test_duration = 10  # Default test duration in seconds
        
        # Create test output directory
        os.makedirs("tests/output", exist_ok=True)

    async def cleanup(self):
        """Clean up audio resources"""
        if self.input_stream:
            try:
                self.input_stream.close()
            except:
                pass
            self.input_stream = None
            
        if self.output_stream:
            try:
                self.output_stream.close()
            except:
                pass
            self.output_stream = None

    def _try_open_input_device(self, device_name):
        """Try to open an input device with fallbacks"""
        try:
            return alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,
                device=device_name,
                format=alsaaudio.PCM_FORMAT_S16_LE,
                channels=self.channels,
                rate=self.input_rate,
                periodsize=self.period_size
            )
        except alsaaudio.ALSAAudioError as e:
            print(f"Could not open {device_name}: {e}")
            return None

    def _find_available_input_device(self):
        """Find an available input device"""
        # First try the configured device
        primary_device = self.audio_config["input"]["device"]
        stream = self._try_open_input_device(primary_device)
        if stream:
            print(f"Using primary device: {primary_device}")
            return stream, primary_device

        # Try alternative device names for TONOR TM20
        alt_devices = [
            "hw:CARD=Device_1,DEV=0",
            "plughw:CARD=Device_1,DEV=0",
            "sysdefault:CARD=Device_1",
            "default"
        ]
        
        for device in alt_devices:
            print(f"Trying alternative device: {device}")
            stream = self._try_open_input_device(device)
            if stream:
                print(f"Using alternative device: {device}")
                return stream, device

        # Last resort: try any available capture device
        for device in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
            if device not in ["null", "default"]:
                print(f"Trying fallback device: {device}")
                stream = self._try_open_input_device(device)
                if stream:
                    print(f"Using fallback device: {device}")
                    return stream, device

        return None, None

    async def initialize_audio(self):
        """Initialize audio devices"""
        try:
            await self.cleanup()
            
            print("\nInitializing audio devices...")
            
            # List available devices
            print("\nAvailable PCM capture devices:")
            for device in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
                print(f"  {device}")
            
            print("\nAvailable PCM playback devices:")
            for device in alsaaudio.pcms(alsaaudio.PCM_PLAYBACK):
                print(f"  {device}")
            
            # Try to find an available input device
            self.input_stream, input_device = self._find_available_input_device()
            if not self.input_stream:
                print("❌ Could not find any available input devices!")
                return False
            
            # Initialize output stream
            try:
                self.output_stream = alsaaudio.PCM(
                    type=alsaaudio.PCM_PLAYBACK,
                    mode=alsaaudio.PCM_NORMAL,
                    device="default",  # Use default output device
                    format=alsaaudio.PCM_FORMAT_S16_LE,
                    channels=self.channels,
                    rate=self.output_rate,
                    periodsize=self.period_size
                )
            except alsaaudio.ALSAAudioError as e:
                print(f"Warning: Could not open output device: {e}")
                print("Continuing without audio output...")
            
            print("\n✓ Audio devices initialized")
            print(f"Input device: {input_device}")
            print(f"Output device: {'default' if self.output_stream else 'None'}")
            return True
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            traceback.print_exc()
            return False

    def _resample_audio(self, audio_48k):
        """Resample 48kHz audio to 16kHz using high-quality interpolation"""
        try:
            # Calculate exact ratio
            ratio = self.processing_rate / self.input_rate
            
            # Calculate target length
            target_length = int(len(audio_48k) * ratio)
            
            # Use numpy's efficient interpolation
            resampled = np.interp(
                np.linspace(0, len(audio_48k) - 1, target_length),
                np.arange(len(audio_48k)),
                audio_48k
            ).astype(np.int16)
            
            # Ensure exact frame length
            if len(resampled) > self.frame_length:
                return resampled[:self.frame_length]
            elif len(resampled) < self.frame_length:
                return np.pad(resampled, (0, self.frame_length - len(resampled)))
            return resampled
            
        except Exception as e:
            print(f"Error in resampling: {e}")
            return np.zeros(self.frame_length, dtype=np.int16)

    async def test_wake_word(self):
        """Test wake word detection"""
        try:
            print("\nStarting wake word detection test...")
            print("Say 'Hey Messy' to test detection")
            print("Test will run for", self.test_duration, "seconds")
            
            self.running = True
            start_time = time.time()
            detections = 0
            frames_processed = 0
            buffer = np.array([], dtype=np.int16)
            
            while time.time() - start_time < self.test_duration and self.running:
                # Read audio frame
                length, data = self.input_stream.read()
                if length > 0:
                    # Convert to numpy array and add to buffer
                    audio = np.frombuffer(data, dtype=np.int16)
                    buffer = np.concatenate([buffer, audio])
                    
                    # Calculate and display volume level
                    volume = np.abs(audio).mean()
                    
                    # Visual volume meter
                    if volume > self.max_volume:
                        meter = "LOUD! 🔊"
                    elif volume > self.min_volume:
                        meter_length = int((volume - self.min_volume) / (self.max_volume - self.min_volume) * 30)
                        meter = "▮" * meter_length
                    else:
                        meter = "quiet"
                    
                    print(f"\rLevel: {meter:<32} ({volume:>4.0f})", end="", flush=True)
                    
                    # Process complete frames from buffer
                    while len(buffer) >= self.samples_per_frame:
                        # Extract frame
                        frame = buffer[:self.samples_per_frame]
                        buffer = buffer[self.samples_per_frame:]
                        
                        # Resample from 48kHz to 16kHz
                        resampled = self._resample_audio(frame)
                        
                        # Process with wake word detector
                        result = self.wake_word_detector.process_frame(resampled)
                        frames_processed += 1
                        
                        if result:
                            detections += 1
                            print(f"\n🎤 Wake word detected! ({detections} total)")
                
                await asyncio.sleep(0.001)
            
            print(f"\n\nTest complete!")
            print(f"Processed {frames_processed} frames")
            print(f"Detected wake word {detections} times")
            
        except Exception as e:
            print(f"Error in wake word test: {e}")
            traceback.print_exc()
        finally:
            self.running = False

    async def test_recording(self):
        """Test audio recording"""
        try:
            print("\nStarting recording test...")
            print("Recording will run for", self.test_duration, "seconds")
            
            self.running = True
            start_time = time.time()
            chunks = []
            
            while time.time() - start_time < self.test_duration and self.running:
                # Read audio frame
                length, data = self.input_stream.read()
                if length > 0:
                    chunks.append(data)
                    
                    # Calculate and display volume level
                    audio = np.frombuffer(data, dtype=np.int16)
                    volume = np.abs(audio).mean()
                    
                    # Visual volume meter
                    if volume > self.max_volume:
                        meter = "LOUD! 🔊"
                    elif volume > self.min_volume:
                        meter_length = int((volume - self.min_volume) / (self.max_volume - self.min_volume) * 30)
                        meter = "▮" * meter_length
                    else:
                        meter = "quiet"
                    
                    print(f"\rLevel: {meter:<32} ({volume:>4.0f})", end="", flush=True)
                
                await asyncio.sleep(0.001)
            
            print("\n\nSaving recording...")
            
            # Save as WAV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/output/test_recording_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.input_rate)
                wf.writeframes(b''.join(chunks))
            
            print(f"Recording saved to: {filename}")
            
            # Play back the recording
            print("\nPlaying back recording...")
            
            with wave.open(filename, 'rb') as wf:
                data = wf.readframes(wf.getnframes())
                
                # Write audio data in chunks
                chunk_size = self.period_size * 2  # 2 bytes per sample
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    if chunk:
                        self.output_stream.write(chunk)
            
            print("Playback complete!")
            
        except Exception as e:
            print(f"Error in recording test: {e}")
            traceback.print_exc()
        finally:
            self.running = False

    async def test_playback(self):
        """Test audio playback"""
        try:
            print("\nStarting playback test...")
            
            # Generate test tone (1kHz sine wave)
            duration = 2.0  # seconds
            frequency = 1000.0  # Hz
            samples = np.arange(int(duration * self.output_rate))
            tone = (32767 * np.sin(2 * np.pi * frequency * samples / self.output_rate)).astype(np.int16)
            
            print("Playing test tone (1kHz)...")
            
            # Write audio data in chunks
            chunk_size = self.period_size * 2  # 2 bytes per sample
            for i in range(0, len(tone) * 2, chunk_size):
                chunk = tone[i//2:i//2 + self.period_size].tobytes()
                if chunk:
                    self.output_stream.write(chunk)
            
            print("Playback complete!")
            
        except Exception as e:
            print(f"Error in playback test: {e}")
            traceback.print_exc()

    async def calibrate_microphone(self):
        """Calibrate microphone levels and thresholds"""
        try:
            print("\nStarting microphone calibration...")
            print("This test will help determine optimal audio thresholds.")
            
            # Initialize arrays for collecting samples
            ambient_samples = []
            speech_samples = []
            
            # Step 1: Measure ambient noise
            print("\n1. Measuring ambient noise")
            print("Please remain quiet for 5 seconds...")
            
            start_time = time.time()
            while time.time() - start_time < 5:
                length, data = self.input_stream.read()
                if length > 0:
                    audio = np.frombuffer(data, dtype=np.int16)
                    volume = np.abs(audio).mean()
                    ambient_samples.append(volume)
                    
                    # Visual feedback
                    meter_length = int(volume / 100)
                    meter = "▮" * min(meter_length, 30)
                    print(f"\rLevel: {meter:<32} ({volume:>4.0f})", end="", flush=True)
                await asyncio.sleep(0.001)
            
            # Calculate ambient noise statistics
            ambient_mean = np.mean(ambient_samples)
            ambient_std = np.std(ambient_samples)
            
            # Step 2: Measure normal speech
            print("\n\n2. Measuring normal speech")
            print("Please speak normally for 5 seconds...")
            print("Say something like: 'This is my normal speaking voice'")
            
            start_time = time.time()
            while time.time() - start_time < 5:
                length, data = self.input_stream.read()
                if length > 0:
                    audio = np.frombuffer(data, dtype=np.int16)
                    volume = np.abs(audio).mean()
                    speech_samples.append(volume)
                    
                    # Visual feedback
                    meter_length = int(volume / 100)
                    meter = "▮" * min(meter_length, 30)
                    print(f"\rLevel: {meter:<32} ({volume:>4.0f})", end="", flush=True)
                await asyncio.sleep(0.001)
            
            # Calculate speech statistics
            speech_mean = np.mean(speech_samples)
            speech_std = np.std(speech_samples)
            
            # Calculate recommended thresholds
            silence_threshold = ambient_mean + (2 * ambient_std)
            wake_word_threshold = silence_threshold * 1.2
            max_volume = speech_mean + (3 * speech_std)
            
            # Print results
            print("\n\nCalibration Results:")
            print("-" * 40)
            print(f"Ambient Noise:")
            print(f"  Mean: {ambient_mean:.1f}")
            print(f"  Std Dev: {ambient_std:.1f}")
            print(f"\nNormal Speech:")
            print(f"  Mean: {speech_mean:.1f}")
            print(f"  Std Dev: {speech_std:.1f}")
            
            print("\nRecommended Settings:")
            print("-" * 40)
            print("Add these to your config.yaml:")
            print("\naudio:")
            print("  input:")
            print(f"    silence_threshold: {silence_threshold:.0f}    # Volume level to detect silence")
            print(f"    silence_duration: 0.5     # Duration of silence to end recording")
            print("\nwake_word:")
            print(f"  volume_threshold: {wake_word_threshold:.0f}    # Minimum volume for wake word")
            print(f"  max_volume: {max_volume:.0f}         # Maximum volume before clipping")
            
        except Exception as e:
            print(f"\nError in microphone calibration: {e}")
            traceback.print_exc()

async def main():
    """Main test suite entry point"""
    print("\nInitializing audio test suite...")
    
    tester = AudioTester()
    if not await tester.initialize_audio():
        print("❌ Failed to initialize audio!")
        return
    
    try:
        while True:
            print("\nAudio Test Suite")
            print("----------------")
            print("1. Test Wake Word Detection")
            print("2. Test Audio Playback")
            print("3. Test Audio Recording")
            print("4. Calibrate Microphone")
            print("5. Set Test Duration")
            print("6. Exit")
            
            choice = input("\nSelect a test (1-6): ").strip()
            
            if choice == "1":
                await tester.test_wake_word()
            elif choice == "2":
                await tester.test_playback()
            elif choice == "3":
                await tester.test_recording()
            elif choice == "4":
                await tester.calibrate_microphone()
            elif choice == "5":
                try:
                    duration = float(input("Enter test duration in seconds: "))
                    tester.test_duration = max(1.0, duration)
                    print(f"Test duration set to {tester.test_duration} seconds")
                except ValueError:
                    print("Invalid duration! Using default.")
            elif choice == "6":
                break
            else:
                print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\nTest suite interrupted!")
    finally:
        await tester.cleanup()
        print("\nTest suite cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())