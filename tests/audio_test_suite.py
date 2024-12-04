import pyaudio
import wave
import numpy as np
import asyncio
import sys
from pathlib import Path
import time
import io
from datetime import datetime
import pvporcupine
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.audio import AudioInterface
from src.core.config import Settings

class AudioTester:
    def __init__(self):
        """Initialize audio testing environment"""
        print("\nInitializing audio test suite...")
        
        # Initialize PyAudio
        self.p = None
        
        # Load settings
        self.settings = Settings()
        self.audio = AudioInterface()
        self.test_file = None
        
        # Audio configuration
        self.input_rate = 48000  # Always use 48kHz
        self.chunk_size = self.settings.audio["input"]["chunk_size"]
        
        # Initialize wake word detector if needed
        self.porcupine = None
        self.wake_word_detected = False
        
    def __del__(self):
        """Clean up resources"""
        self.cleanup()
        
    def cleanup(self):
        """Clean up audio resources"""
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            
        if self.p:
            try:
                self.p.terminate()
            except:
                pass
            self.p = None
            
        if self.porcupine:
            try:
                self.porcupine.delete()
            except:
                pass
            self.porcupine = None

    async def run_tests(self):
        """Run comprehensive audio tests"""
        try:
            print("\n=== Audio System Test ===")
            
            # Initialize audio interface
            await self.audio.initialize()
            
            while True:
                print("\nTest Options:")
                print("1. Test wake word detection")
                print("2. Test audio playback")
                print("3. Test audio recording")
                print("4. Calibrate wake word")
                print("5. Calibrate microphone")
                print("6. Exit")
                
                choice = input("\nChoose test (1-6): ")
                
                if choice == '1':
                    await self.test_wake_word()
                elif choice == '2':
                    await self.test_playback()
                elif choice == '3':
                    await self.test_recording()
                elif choice == '4':
                    await self.calibrate_wake_word()
                elif choice == '5':
                    await self.calibrate_microphone()
                elif choice == '6':
                    break
                else:
                    print("Invalid choice")
            
        except Exception as e:
            print(f"Test error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()

    async def test_wake_word(self):
        """Test wake word detection"""
        print("\n=== Wake Word Test ===")
        
        try:
            # Clean up any existing resources
            self.cleanup()
            
            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            
            # Initialize wake word detector
            print("\nInitializing wake word detector...")
            self.porcupine = pvporcupine.create(
                access_key=self.settings.PICOVOICE_ACCESS_KEY,
                keyword_paths=[str(self.settings.WAKE_WORD_MODEL_PATH)],
                sensitivities=[0.5]  # Default sensitivity
            )
            
            # Calculate frame sizes
            self.frame_length = self.porcupine.frame_length
            self.samples_per_frame = int(self.frame_length * self.input_rate / 16000)
            
            print(f"\nAudio Configuration:")
            print(f"  Input rate: {self.input_rate}Hz")
            print(f"  Frame length: {self.frame_length} samples")
            print(f"  Samples per frame: {self.samples_per_frame}")
            print(f"  Chunk size: {self.chunk_size}")
            
            # Find TONOR TM20 device
            device_index = None
            print("\nLooking for TONOR TM20 device:")
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                print(f"\nDevice {i}:")
                print(f"    Name: {dev['name']}")
                print(f"    Max Input Channels: {dev['maxInputChannels']}")
                print(f"    Max Output Channels: {dev['maxOutputChannels']}")
                print(f"    Default Sample Rate: {dev['defaultSampleRate']}")
                if "TONOR TM20" in dev['name']:
                    device_index = i
                    print(f"✓ Found TONOR TM20 at index {i}")
            
            if device_index is None:
                print("❌ Could not find TONOR TM20 device!")
                return
            
            # Configure audio stream
            print(f"\nOpening audio stream with device index {device_index}")
            try:
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.input_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.samples_per_frame
                )
            except Exception as e:
                print(f"Error opening stream: {e}")
                print("Trying to terminate and reinitialize PyAudio...")
                self.cleanup()
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.input_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.samples_per_frame
                )
            
            print("\nListening for wake word 'Hey Messy'...")
            print("Press Ctrl+C to stop")
            
            # Get volume thresholds from config
            min_volume = 100  # Default minimum volume
            max_volume = 4000  # Default maximum volume
            
            while True:
                # Read audio frame
                data = self.stream.read(self.samples_per_frame, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                
                # Resample from 48kHz to 16kHz for Porcupine
                resampled = np.interp(
                    np.linspace(0, len(audio), self.frame_length),
                    np.arange(len(audio)),
                    audio
                ).astype(np.int16)
                
                # Process frame
                result = self.porcupine.process(resampled)
                
                # Calculate volume for feedback
                volume = np.abs(audio).mean()
                if volume > max_volume:
                    meter = "LOUD! "
                elif volume > min_volume:
                    meter = "#" * int((volume - min_volume) / (max_volume - min_volume) * 20)
                else:
                    meter = "soft "
                
                print(f"\rListening... Volume: {meter} ({volume:.0f})", end="", flush=True)
                
                if result >= 0:
                    print("\n\nWake word detected!")
                    break
                
                await asyncio.sleep(0.001)
            
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except Exception as e:
            print(f"\nError in wake word test: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if self.porcupine:
                self.porcupine.delete()

    async def test_playback(self):
        """Test audio playback"""
        print("\n=== Audio Playback Test ===")
        
        if not self.test_file:
            print("No test files found!")
            return
            
        try:
            with wave.open(str(self.test_file), 'rb') as wf:
                # Get WAV file properties
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                
                # Calculate actual frames from file size
                file_size = self.test_file.stat().st_size
                bytes_per_frame = channels * sample_width
                actual_frames = (file_size - 44) // bytes_per_frame  # 44 bytes for WAV header
                
                print("\nWAV File Properties:")
                print(f"Channels: {channels}")
                print(f"Sample width: {sample_width} bytes")
                print(f"Frame rate: {frame_rate} Hz")
                print(f"Frames: {actual_frames}")
                duration = actual_frames / float(frame_rate)
                print(f"Duration: {duration:.1f} seconds")
                print(f"File size: {file_size} bytes")
                print(f"Bytes per frame: {bytes_per_frame}")
                
                # Read the entire file
                print("\nReading audio file...")
                wf.setpos(0)  # Reset position to start of audio data
                audio_data = wf.readframes(actual_frames)
                print(f"Read {len(audio_data)} bytes of audio data")
                
                # Always use 48kHz for playback
                self.audio.settings.audio["output"]["rate"] = 48000
                
                print("\nPlaying audio...")
                await self.audio.play_audio_chunk(audio_data)
                print("Playback complete")
                
        except Exception as e:
            print(f"Playback error: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()

    async def test_recording(self):
        """Test audio recording"""
        print("\n=== Audio Recording Test ===")
        
        try:
            # Configure audio stream for recording
            input_rate = 48000  # Use 48kHz for hardware compatibility
            print(f"\nConfiguring recording at {input_rate}Hz")
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=input_rate,
                input=True,
                input_device_index=self.audio.settings.audio["input"]["device_index"],
                frames_per_buffer=self.audio.settings.audio["input"]["chunk_size"]
            )
            
            frames = []
            print("\nRecording for 5 seconds...")
            
            # Record for 5 seconds
            chunks_to_record = int(input_rate * 5 / self.audio.settings.audio["input"]["chunk_size"])
            print(f"Recording {chunks_to_record} chunks at {self.audio.settings.audio['input']['chunk_size']} samples per chunk")
            
            for _ in range(chunks_to_record):
                data = stream.read(self.audio.settings.audio["input"]["chunk_size"], exception_on_overflow=False)
                frames.append(data)
            
            print("Recording complete")
            stream.stop_stream()
            stream.close()
            
            # Save recording
            print("\nSaving recording...")
            self.test_file = Path("test_recording.wav")
            with wave.open(str(self.test_file), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(input_rate)  # Save at 48kHz
                wav_data = b''.join(frames)
                print(f"Total bytes to write: {len(wav_data)}")
                wf.writeframes(wav_data)
            
            # Verify the saved file
            with wave.open(str(self.test_file), 'rb') as wf:
                print(f"\nVerifying saved WAV file:")
                print(f"Channels: {wf.getnchannels()}")
                print(f"Sample width: {wf.getsampwidth()} bytes")
                print(f"Frame rate: {wf.getframerate()} Hz")
                print(f"Number of frames: {wf.getnframes()}")
                print(f"Expected duration: {wf.getnframes() / float(wf.getframerate()):.1f} seconds")
            
            print(f"\nSaved as {self.test_file}")
            
        except Exception as e:
            print(f"Recording error: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()

    async def calibrate_wake_word(self):
        """Calibrate wake word settings"""
        print("\n=== Wake Word Calibration ===")
        print("This will help find the optimal settings for wake word detection")
        
        try:
            # Test different sensitivity levels
            sensitivities = [0.3, 0.5, 0.7, 0.9]
            best_sensitivity = 0.5
            best_detections = 0
            
            for sensitivity in sensitivities:
                print(f"\nTesting sensitivity: {sensitivity}")
                
                # Initialize Porcupine with current sensitivity
                if self.audio.porcupine:
                    self.audio.porcupine.delete()
                
                self.audio.porcupine = pvporcupine.create(
                    access_key=self.settings.PICOVOICE_ACCESS_KEY,
                    keyword_paths=[str(self.settings.WAKE_WORD_MODEL_PATH)],
                    sensitivities=[sensitivity]
                )
                
                print("\nSay 'Hey Messy' 3 times...")
                detections = 0
                start_time = time.time()
                
                while time.time() - start_time < 10:  # 10 second test window
                    pcm = await self._get_next_audio_frame()
                    if pcm is None:
                        continue
                        
                    result = self.audio.porcupine.process(pcm)
                    level = np.abs(pcm).mean()
                    
                    print(f"\rLevel: {level:.0f}", end='', flush=True)
                    
                    if result >= 0:
                        detections += 1
                        print(f"\nDetection {detections}/3!")
                        if detections >= 3:
                            break
                            
                    await asyncio.sleep(0.001)
                
                if detections > best_detections:
                    best_detections = detections
                    best_sensitivity = sensitivity
            
            print(f"\n\nRecommended settings:")
            print(f"Wake word threshold: {best_sensitivity}")
            print(f"Min volume: {self.wake_word_min_volume}")
            print(f"Max volume: {self.wake_word_max_volume}")
            
        except Exception as e:
            print(f"Calibration error: {e}")

    async def calibrate_microphone(self):
        """Calibrate microphone levels"""
        print("\n=== Microphone Calibration ===")
        
        try:
            # Measure ambient noise
            print("\nMeasuring ambient noise...")
            print("Please remain quiet for 5 seconds...")
            
            ambient_levels = []
            start_time = time.time()
            
            while time.time() - start_time < 5:
                data = self.audio.stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio).mean()
                ambient_levels.append(level)
                meter = "#" * int(level/100)
                print(f"\rLevel: {meter} ({level:.0f})", end="", flush=True)
                await asyncio.sleep(0.001)
            
            ambient_mean = np.mean(ambient_levels)
            print(f"\n\nAmbient noise level: {ambient_mean:.0f}")
            
            # Measure speech levels
            print("\nNow testing speech levels...")
            print("Please speak normally for 5 seconds...")
            
            speech_levels = []
            start_time = time.time()
            
            while time.time() - start_time < 5:
                data = self.audio.stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio).mean()
                speech_levels.append(level)
                meter = "#" * int(level/100)
                print(f"\rLevel: {meter} ({level:.0f})", end="", flush=True)
                await asyncio.sleep(0.001)
            
            speech_mean = np.mean(speech_levels)
            print(f"\n\nAverage speech level: {speech_mean:.0f}")
            
            # Calculate recommended settings
            recommended_min = max(ambient_mean * 1.5, 100)
            recommended_max = min(speech_mean * 1.5, 4000)
            
            print("\nRecommended settings:")
            print(f"Minimum volume: {recommended_min:.0f}")
            print(f"Maximum volume: {recommended_max:.0f}")
            print(f"Silence threshold: {ambient_mean * 1.2:.0f}")
            
        except Exception as e:
            print(f"Calibration error: {e}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.audio.stream:
                self.audio.stream.stop_stream()
                self.audio.stream.close()
            if self.audio.porcupine:
                self.audio.porcupine.delete()
        except Exception as e:
            print(f"Cleanup error: {e}")

    async def _get_next_audio_frame(self):
        """Get the next audio frame for wake word detection"""
        try:
            # Calculate required input samples to get 512 samples after resampling
            input_samples = int(512 * self.input_rate / 16000)
            
            # Read enough samples
            data = self.audio.stream.read(input_samples, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            
            # Resample to 16kHz for Porcupine
            if self.input_rate != 16000:
                # Create time arrays for interpolation
                time_orig = np.linspace(0, len(audio), len(audio))
                time_new = np.linspace(0, len(audio), 512)
                
                # Resample using linear interpolation
                resampled = np.interp(time_new, time_orig, audio).astype(np.int16)
                
                # Ensure exactly 512 samples
                if len(resampled) > 512:
                    return resampled[:512]
                elif len(resampled) < 512:
                    return np.pad(resampled, (0, 512 - len(resampled)))
                return resampled
            
            # If already at 16kHz, ensure 512 samples
            if len(audio) > 512:
                return audio[:512]
            elif len(audio) < 512:
                return np.pad(audio, (0, 512 - len(audio)))
            return audio
            
        except Exception as e:
            print(f"Error getting audio frame: {e}")
            return None

if __name__ == "__main__":
    tester = AudioTester()
    asyncio.run(tester.run_tests())