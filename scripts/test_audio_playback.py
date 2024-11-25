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
        self.settings = Settings()
        self.audio = AudioInterface()
        
        # Load config settings for testing
        self.audio_config = self.settings._load_yaml_config().get('audio', {})
        self.wake_word_config = self.settings._load_yaml_config().get('wake_word', {})
        
        # Use configured values for calibration
        self.input_device_index = self.audio_config.get('input', {}).get('device_index', 1)
        self.input_rate = self.audio_config.get('input', {}).get('native_rate', 44100)
        self.chunk_size = self.audio_config.get('input', {}).get('chunk_size', 1024)
        self.silence_threshold = self.audio_config.get('input', {}).get('silence_threshold', 550)
        
        # Wake word settings
        self.wake_word_threshold = self.wake_word_config.get('threshold', 0.8)
        self.wake_word_min_volume = self.wake_word_config.get('min_volume', 686)
        self.wake_word_max_volume = self.wake_word_config.get('max_volume', 2000)
        
        # Look for test files in this order
        self.test_files = [
            Path("/home/pi/messi/cache/tts/last_response.wav"),  # Last TTS response
            Path("/home/pi/messi/assets/audio/test_audio.wav"),  # Test asset
            Path("/home/pi/messi/test_recording.wav")            # Test recording
        ]
        
        # Find first available test file
        self.test_file = None
        for file in self.test_files:
            if file.exists():
                self.test_file = file
                break

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
        """Test wake word detection using production code approach"""
        print("\n=== Wake Word Test ===")
        print("Say 'Hey Messy' to test wake word detection")
        print("Press Ctrl+C to stop")

        try:
            # Initialize audio stream if needed
            if not self.audio.stream:
                print("\nInitializing audio stream...")
                p = pyaudio.PyAudio()
                self.audio.stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.input_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.chunk_size
                )

            # Initialize Porcupine with production settings
            if not self.audio.porcupine:
                print("\nInitializing wake word detector...")
                self.audio.porcupine = pvporcupine.create(
                    access_key=self.settings.PICOVOICE_ACCESS_KEY,
                    keyword_paths=[str(self.settings.WAKE_WORD_MODEL_PATH)],
                    sensitivities=[self.wake_word_threshold]
                )

            # Create thread pool executor
            self.executor = ThreadPoolExecutor(max_workers=2)
            running = True

            print("\nListening for wake word...")
            while running:
                try:
                    # Get audio frame using production method
                    pcm = await self._get_next_audio_frame(
                        self.chunk_size, 
                        self.input_rate
                    )
                    if pcm is None:
                        continue

                    # Process in thread pool like production
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.audio.porcupine.process,
                        pcm
                    )

                    # Calculate level for feedback
                    level = np.abs(pcm).mean()
                    
                    # Visual feedback
                    if level > self.wake_word_max_volume:
                        print(f"\rLevel: TOO LOUD ({level:.0f})", end='', flush=True)
                    elif level > self.wake_word_min_volume:
                        print(f"\rLevel: GOOD ({level:.0f})", end='', flush=True)
                    else:
                        print(f"\rLevel: too soft ({level:.0f})", end='', flush=True)

                    # Handle detection
                    if result >= 0:
                        print(f"\n\n🎤 Wake word detected!")
                        print(f"Level: {level:.0f}")
                        print(f"Threshold: {self.wake_word_threshold}")

                    await asyncio.sleep(0.001)

                except KeyboardInterrupt:
                    running = False
                    print("\nStopping wake word test")
                    break

        except Exception as e:
            print(f"\nError in wake word test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if hasattr(self, 'executor'):
                self.executor.shutdown()

    async def test_playback(self):
        """Test audio playback with proper WAV handling"""
        print("\n=== Audio Playback Test ===")
        
        # Check for test files
        available_files = []
        for file in self.test_files:
            if file.exists():
                try:
                    with wave.open(str(file), 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                        available_files.append((file, duration))
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        if not available_files:
            print("No test files found. Available options:")
            print("1. Record a new test file")
            print("2. Back to main menu")
            
            choice = input("\nChoose option (1-2): ")
            if choice == '1':
                await self.test_recording()
                return
            else:
                return
        
        # Let user choose file
        print("\nAvailable test files:")
        for i, (file, duration) in enumerate(available_files, 1):
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"{i}. {file.name} ({minutes}m {seconds}s)")
        
        while True:
            try:
                choice = int(input(f"\nChoose file (1-{len(available_files)}): "))
                if 1 <= choice <= len(available_files):
                    self.test_file = available_files[choice-1][0]
                    break
                else:
                    print("Invalid choice")
            except ValueError:
                print("Please enter a number")
        
        # Play selected file
        try:
            with wave.open(str(self.test_file), 'rb') as wf:
                print("\nWAV File Properties:")
                print(f"Channels: {wf.getnchannels()}")
                print(f"Sample width: {wf.getsampwidth()} bytes")
                print(f"Frame rate: {wf.getframerate()} Hz")
                
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                print(f"Duration: {minutes}m {seconds}s")
                print(f"Total frames: {frames:,}")
                
                # Create WAV file in memory
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_out:
                    wav_out.setnchannels(wf.getnchannels())
                    wav_out.setsampwidth(wf.getsampwidth())
                    wav_out.setframerate(wf.getframerate())
                    
                    # Read and write in chunks
                    chunk_size = 8192
                    bytes_read = 0
                    
                    print("\nReading audio file...")
                    while True:
                        chunk = wf.readframes(chunk_size)
                        if not chunk:
                            break
                        wav_out.writeframes(chunk)
                        bytes_read += len(chunk)
                        # Show progress for large files
                        if bytes_read % (chunk_size * 100) == 0:
                            print(f"\rRead: {bytes_read/1024:.1f}KB", end='', flush=True)
                
                # Get the complete WAV data
                audio_data = wav_buffer.getvalue()
                print(f"\nRead {len(audio_data)/1024:.1f}KB")
        
                # Playback menu
                while True:
                    print("\nPlayback Options:")
                    print("1. Play original")
                    print("2. Play normalized")
                    print("3. Play with volume adjustment")
                    print("4. Back to main menu")
                    
                    choice = input("\nChoose option (1-4): ")
                    
                    try:
                        if choice == '1':
                            print(f"\nPlaying {minutes}m {seconds}s of audio...")
                            await self.audio.play_audio_chunk(audio_data)
                        elif choice == '2':
                            print(f"\nPlaying normalized audio ({minutes}m {seconds}s)...")
                            await self.play_normalized(audio_data)
                        elif choice == '3':
                            volume = float(input("Enter volume (0.1-2.0): "))
                            print(f"\nPlaying at {volume:.1f}x volume ({minutes}m {seconds}s)...")
                            await self.play_with_volume(audio_data, volume)
                        elif choice == '4':
                            break
                        else:
                            print("Invalid choice")
                    except Exception as e:
                        print(f"Playback error: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading WAV file: {e}")

    async def test_recording(self):
        """Test audio recording with proper WAV handling"""
        print("\n=== Audio Recording Test ===")
        
        try:
            if not self.audio.stream or not self.audio.stream.is_active():
                print("Error: No active audio stream")
                return
            
            print("Recording 5 seconds of audio...")
            print("Speak into the microphone...")
            
            # Record with progress indicator
            chunks = []
            frames_to_record = int(44100 * 5 / 1024)  # 5 seconds
            
            for i in range(frames_to_record):
                try:
                    data = self.audio.stream.read(1024, exception_on_overflow=False)
                    chunks.append(data)
                    # Show progress
                    progress = (i + 1) / frames_to_record * 100
                    print(f"\rProgress: {progress:.0f}%", end='', flush=True)
                    await asyncio.sleep(0.001)  # Allow other tasks to run
                except Exception as e:
                    print(f"\nError reading audio: {e}")
                    continue
            
            print("\nFinished recording")
            
            # Create WAV in memory first
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(b''.join(chunks))
            
            # Get the complete WAV data
            wav_data = wav_buffer.getvalue()
            
            # Save to file
            test_file = Path("test_recording.wav")
            with open(test_file, 'wb') as f:
                f.write(wav_data)
            
            print(f"\nSaved recording to {test_file}")
            
            # Get recording info
            with wave.open(str(test_file), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                print(f"\nRecording Properties:")
                print(f"Duration: {duration:.1f}s")
                print(f"Sample Rate: {rate}Hz")
                print(f"Channels: {wf.getnchannels()}")
                print(f"Sample Width: {wf.getsampwidth()} bytes")
                print(f"File Size: {test_file.stat().st_size/1024:.1f}KB")
            
            # Play back recording using the WAV data we already have
            print("\nPlaying back recording...")
            await self.audio.play_audio_chunk(wav_data)
            
        except Exception as e:
            print(f"Recording error: {e}")
            import traceback
            traceback.print_exc()

    async def play_normalized(self, audio_data: bytes):
        """Play normalized audio"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            peak = np.max(np.abs(audio_array))
            if peak > 0:
                normalized = (audio_array / peak * 32767).astype(np.int16)
                await self.audio.play_audio_chunk(normalized.tobytes())
        except Exception as e:
            print(f"Normalization error: {e}")

    async def play_with_volume(self, audio_data: bytes, volume: float):
        """Play audio with volume adjustment"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            adjusted = np.clip(audio_array * volume, -32768, 32767).astype(np.int16)
            await self.audio.play_audio_chunk(adjusted.tobytes())
        except Exception as e:
            print(f"Volume adjustment error: {e}")

    async def calibrate_wake_word(self):
        """Calibrate wake word settings with improved sensitivity"""
        try:
            # First list available devices
            p = pyaudio.PyAudio()
            print("\nAvailable Audio Devices:")
            
            # Track suitable input devices
            input_devices = []
            
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                print(f"\nIndex {i}: {dev['name']}")
                print(f"  Max Input Channels: {dev['maxInputChannels']}")
                print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
                
                # Check if device is suitable for input
                if dev['maxInputChannels'] > 0:
                    input_devices.append(i)
                    if "tonor" in dev['name'].lower():
                        print("  ✓ RECOMMENDED INPUT DEVICE")
                    else:
                        print("  ✓ Can use for input")
            
            if not input_devices:
                print("\nNo suitable input devices found!")
                return
                
            print("\nSuitable input devices:", input_devices)
            print("Choose the index number of your microphone (TONOR TM20 recommended)")
            
            # Let user choose input device
            device_index = int(input("\nChoose input device index: "))
            
            if device_index not in input_devices:
                print(f"Warning: Device {device_index} may not support input!")
                proceed = input("Continue anyway? (y/n): ").lower()
                if proceed != 'y':
                    return
            
            input_device_info = p.get_device_info_by_index(device_index)
            device_rate = int(input_device_info['defaultSampleRate'])
            print(f"\nUsing device: {input_device_info['name']}")
            print(f"Native sample rate: {device_rate}")
            
            # Initialize audio stream using config settings
            self.audio.stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.input_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            # Calculate frame sizes for Porcupine
            input_frames = int(self.input_rate * 512 / 16000)  # Convert Porcupine frames to input rate
            print(f"\nAudio Processing Details:")
            print(f"Input rate: {self.input_rate}Hz")
            print(f"Porcupine rate: 16000Hz")
            print(f"Input frames needed: {input_frames}")
            
            # Use configured sensitivities with wider range
            sensitivities = [
                0.3,   # Less sensitive
                0.5,   # Recommended default
                0.7,   # More sensitive
                0.9    # Most sensitive
            ]
            print("\nTesting sensitivity levels:")
            print("0.3 = Less sensitive (fewer false positives)")
            print("0.5 = Default balance")
            print("0.7 = More sensitive")
            print("0.9 = Most sensitive (more detections)")
            
            best_sensitivity = 0.7
            best_detections = 0
            
            print("\nTesting multiple sensitivity levels...")
            
            best_results = None  # Store best results
            
            for sensitivity in sensitivities:
                if self.audio.porcupine:
                    self.audio.porcupine.delete()
                
                print(f"\nTrying sensitivity: {sensitivity}")
                self.audio.porcupine = pvporcupine.create(
                    access_key=self.settings.PICOVOICE_ACCESS_KEY,
                    keyword_paths=[str(self.settings.WAKE_WORD_MODEL_PATH)],
                    sensitivities=[sensitivity]
                )
                print(f"Porcupine frames required: {self.audio.porcupine.frame_length}")
                
                # Initialize calibration data
                calibration_data = {
                    "ambient_levels": [],
                    "wake_word_levels": [],
                    "detections": [],
                    "missed_detections": 0,
                    "false_positives": 0,
                    "detection_levels": [],  # Track levels when detection occurs
                    "false_positive_levels": []  # Add this key
                }
                
                # Step 1: Measure ambient noise
                print("\n1. Measuring Ambient Noise")
                print("Please remain quiet for 5 seconds...")
                
                start_time = time.time()
                while time.time() - start_time < 5:
                    data = self.audio.stream.read(1024, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16)
                    level = np.abs(audio).mean()
                    calibration_data["ambient_levels"].append(level)
                    print(f"\rLevel: {'#' * int(level/100)}", end='', flush=True)
                    await asyncio.sleep(0.001)
                
                ambient_mean = np.mean(calibration_data["ambient_levels"])
                print(f"\nAmbient noise level: {ambient_mean:.0f}")
                
                # Step 2: Wake word calibration
                print("\n2. Wake Word Calibration")
                print("Please say 'Hey Messy' 5 times when prompted")
                print("Wait for the prompt between each attempt")
                
                for i in range(5):
                    print(f"\nAttempt {i+1}/5 - Say 'Hey Messy' now...")
                    
                    start_time = time.time()
                    detection_window = []
                    detected = False
                    
                    while time.time() - start_time < 3:  # 3 second window
                        # Read audio at native rate
                        data = self.audio.stream.read(input_frames, exception_on_overflow=False)
                        audio = np.frombuffer(data, dtype=np.int16)
                        
                        # Better resampling
                        resampled = self.resample_audio(audio)
                        
                        # Calculate level from original audio
                        level = np.abs(audio).mean()
                        detection_window.append(level)
                        
                        # Visual feedback with more detail
                        bar = "#" * int(level/100)
                        print(f"\rLevel: {bar} ({level:.0f})", end='', flush=True)
                        
                        # Process with Porcupine
                        result = self.audio.porcupine.process(resampled)
                        if result >= 0:
                            detected = True
                            print(f"\n✓ Wake word detected! (Level: {level:.0f})")
                            calibration_data["detections"].append({
                                "level": level,
                                "time": time.time() - start_time,
                                "resampled_peak": np.max(np.abs(resampled))
                            })
                            break
                        
                        await asyncio.sleep(0.001)
                    
                    if not detected:
                        print(f"\n✗ No detection (Avg Level: {np.mean(detection_window):.0f})")
                        calibration_data["missed_detections"] += 1
                    
                    await asyncio.sleep(1)  # Pause between attempts
                
                # Step 3: False Positive Testing
                print("\n3. Testing for False Positives")
                print("Please speak normally for 15 seconds")
                print("Say anything EXCEPT 'Hey Messy'")
                print("Examples: count numbers, say alphabet, tell a short story")
                
                start_time = time.time()
                test_duration = 15  # seconds
                frames_processed = 0
                
                print("\nStarting false positive test...")
                print("Speak normally now...")
                
                try:
                    while time.time() - start_time < test_duration:
                        # Read audio at native rate
                        data = self.audio.stream.read(input_frames, exception_on_overflow=False)
                        audio = np.frombuffer(data, dtype=np.int16)
                        
                        # Use same resampling as wake word test
                        resampled = self.resample_audio(audio)
                        
                        # Calculate level from original audio
                        level = np.abs(audio).mean()
                        calibration_data["false_positive_levels"].append(level)
                        
                        # Process with Porcupine
                        result = self.audio.porcupine.process(resampled)
                        if result >= 0:
                            calibration_data["false_positives"] += 1
                            print(f"\n! False detection at level: {level:.0f}")
                        
                        frames_processed += 1
                        remaining = test_duration - (time.time() - start_time)
                        if remaining > 0:  # Only print if time remaining
                            print(f"\rLevel: {'#' * int(level/100)} ({level:.0f}) [{remaining:.1f}s]", end='', flush=True)
                        
                        await asyncio.sleep(0.001)
                    
                    print("\nFalse positive testing complete")
                    
                except Exception as e:
                    print(f"\nError in false positive testing: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Calculate recommended settings based on detections
                detection_levels = [d["level"] for d in calibration_data["detections"]]
                if detection_levels:
                    min_detection = min(detection_levels)
                    max_detection = max(detection_levels)
                    
                    recommended_settings = {
                        "WAKE_WORD_MIN_VOLUME": max(min_detection * 0.8, ambient_mean * 1.5),
                        "WAKE_WORD_MAX_VOLUME": min(max_detection * 1.2, 5000),
                        "WAKE_WORD_THRESHOLD": sensitivity,
                        "WAKE_WORD_DETECTION_WINDOW": 2.0,
                        "WAKE_WORD_CONSECUTIVE_FRAMES": 1
                    }
                else:
                    recommended_settings = {
                        "WAKE_WORD_MIN_VOLUME": ambient_mean * 1.5,
                        "WAKE_WORD_MAX_VOLUME": 5000,
                        "WAKE_WORD_THRESHOLD": sensitivity,
                        "WAKE_WORD_DETECTION_WINDOW": 2.0,
                        "WAKE_WORD_CONSECUTIVE_FRAMES": 1
                    }
                
                # Store results if best so far
                if len(calibration_data["detections"]) > best_detections:
                    best_detections = len(calibration_data["detections"])
                    best_sensitivity = sensitivity
                    best_results = {
                        "ambient_mean": ambient_mean,
                        "calibration_data": calibration_data,
                        "recommended_settings": recommended_settings
                    }
            
            # After testing all sensitivities, print final results
            if best_results:
                print(f"\nBest sensitivity found: {best_sensitivity}")
                print(f"Successful detections at this sensitivity: {best_detections}/5")
                
                # Print results using best data
                print("\n=== Calibration Results ===")
                print(f"\nAmbient Noise:")
                print(f"  Mean Level: {best_results['ambient_mean']:.0f}")
                print(f"  Range: {min(best_results['calibration_data']['ambient_levels']):.0f} - {max(best_results['calibration_data']['ambient_levels']):.0f}")
                
                print(f"\nWake Word:")
                print(f"  Mean Level: {best_sensitivity:.2f}")
                print(f"  Successful Detections: {best_detections}/5")
                print(f"  Missed Detections: {best_results['calibration_data']['missed_detections']}")
                
                print(f"\nFalse Positive Test:")
                print(f"  False Detections: {best_results['calibration_data']['false_positives']}")
                print(f"  Speech Level: {np.mean(best_results['calibration_data']['false_positive_levels']):.0f}")
                
                print("\nRecommended Settings:")
                for key, value in best_results['recommended_settings'].items():
                    print(f"{key} = {value}")
                
                # Save results
                self._save_calibration_results(best_results)
            else:
                print("\nNo successful calibration found")
            
        except Exception as e:
            print(f"Error in wake word calibration: {e}")
            import traceback
            traceback.print_exc()

    async def calibrate_microphone(self):
        """Run microphone calibration with config settings"""
        try:
            print("\n=== Microphone Calibration ===")
            
            # Initialize audio stream if not already initialized
            if not self.audio.stream:
                print("\nInitializing audio stream...")
                p = pyaudio.PyAudio()
                self.audio.stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.input_rate,           # Use config rate
                    input=True,
                    input_device_index=self.input_device_index,  # Use config device
                    frames_per_buffer=self.chunk_size  # Use config chunk size
                )
            
            # Test 1: Ambient Noise
            print("\n1. Testing Ambient Noise")
            print(f"Please remain quiet for 5 seconds...")
            ambient_levels = []
            start_time = time.time()
            
            while time.time() - start_time < 5:
                data = self.audio.stream.read(1024, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio).mean()
                ambient_levels.append(level)
                
                # Visual feedback
                bar = "#" * int(level/100)
                print(f"\rLevel: {bar:<50} ({level:.0f})", end='', flush=True)
                await asyncio.sleep(0.001)
            
            ambient_mean = np.mean(ambient_levels)
            print(f"\nAmbient noise level: {ambient_mean:.0f}")
            
            # Test 2: Speech Levels
            print("\n2. Speech Level Test")
            print("Please speak normally for 5 seconds...")
            speech_levels = []
            start_time = time.time()
            
            while time.time() - start_time < 5:
                data = self.audio.stream.read(1024, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio).mean()
                speech_levels.append(level)
                
                # Visual feedback
                bar = "#" * int(level/100)
                print(f"\rLevel: {bar:<50} ({level:.0f})", end='', flush=True)
                await asyncio.sleep(0.001)
            
            speech_mean = np.mean(speech_levels)
            print(f"\nAverage speech level: {speech_mean:.0f}")
            
            # Calculate recommended settings
            print("\nRecommended Settings:")
            print(f"WAKE_WORD_MIN_VOLUME = {max(ambient_mean * 1.5, 100):.0f}")
            print(f"AUDIO_SILENCE_THRESHOLD = {max(ambient_mean * 1.2, 80):.0f}")
            print(f"WAKE_WORD_MAX_VOLUME = {min(speech_mean * 1.5, 4500):.0f}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path(f"calibration_results_{timestamp}.txt")
            
            with open(results_file, "w") as f:
                f.write("=== Microphone Calibration Results ===\n")
                f.write(f"Date: {timestamp}\n\n")
                f.write(f"Ambient Noise Level: {ambient_mean:.0f}\n")
                f.write(f"Average Speech Level: {speech_mean:.0f}\n\n")
                f.write("Recommended Settings:\n")
                f.write(f"WAKE_WORD_MIN_VOLUME = {max(ambient_mean * 1.5, 100):.0f}\n")
                f.write(f"AUDIO_SILENCE_THRESHOLD = {max(ambient_mean * 1.2, 80):.0f}\n")
                f.write(f"WAKE_WORD_MAX_VOLUME = {min(speech_mean * 1.5, 4500):.0f}\n")
            
            print(f"\nResults saved to: {results_file}")
            
        except Exception as e:
            print(f"Error in microphone calibration: {e}")
            import traceback
            traceback.print_exc()

    async def cleanup(self):
        """Clean up resources safely"""
        try:
            if hasattr(self.audio, 'stream') and self.audio.stream:
                if self.audio.stream.is_active():
                    self.audio.stream.stop_stream()
                self.audio.stream.close()
            if hasattr(self.audio, 'p'):
                self.audio.p.terminate()
        except Exception as e:
            print(f"Cleanup error: {e}")

    async def _get_next_audio_frame(self, input_frames: int, device_rate: int) -> np.ndarray:
        """Get audio frame with proper resampling"""
        try:
            # Read audio at native rate
            data = self.audio.stream.read(input_frames, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            
            # Resample to 16kHz for Porcupine
            if device_rate != 16000:
                ratio = 16000 / device_rate
                target_length = int(len(audio) * ratio)
                resampled = np.interp(
                    np.linspace(0, len(audio)-1, target_length),
                    np.arange(len(audio)),
                    audio
                ).astype(np.int16)
                
                if len(resampled) > 512:
                    return resampled[:512]
                return np.pad(resampled, (0, 512 - len(resampled)))
            
            return audio
            
        except Exception as e:
            print(f"Error getting audio frame: {e}")
            return None

    async def process_wake_word(self, audio: np.ndarray) -> bool:
        """Process audio frame for wake word with thread pool"""
        try:
            # Process in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                self.audio.porcupine.process,
                audio
            )
            return result >= 0
            
        except Exception as e:
            print(f"Error processing wake word: {e}")
            return False

    def resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """Higher quality resampling for wake word detection"""
        try:
            # Normalize audio first
            audio_float = audio.astype(np.float32) / 32768.0
            
            # Calculate resampling parameters
            time_orig = np.linspace(0, len(audio_float), len(audio_float))
            time_new = np.linspace(0, len(audio_float), 512)
            
            # Resample with cubic interpolation
            resampled = np.interp(time_new, time_orig, audio_float)
            
            # Scale back to int16
            return (resampled * 32768.0).astype(np.int16)
            
        except Exception as e:
            print(f"Error resampling audio: {e}")
            return np.zeros(512, dtype=np.int16)

    def _save_calibration_results(self, results: dict):
        """Save calibration results to file"""
        try:
            results_file = Path("wake_word_calibration.txt")
            with open(results_file, "w") as f:
                f.write("=== Wake Word Calibration Results ===\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Raw Metrics:\n")
                f.write(f"Ambient Noise Mean: {results['ambient_mean']:.0f}\n")
                f.write(f"Wake Word Mean: {results['calibration_data']['detection_levels'][-1] if results['calibration_data']['detection_levels'] else 0:.0f}\n")
                f.write(f"Successful Detections: {len(results['calibration_data']['detections'])}/5\n")
                f.write(f"Missed Detections: {results['calibration_data']['missed_detections']}\n")
                f.write(f"False Positives: {results['calibration_data']['false_positives']}\n\n")
                
                f.write("\nCurrent Config Settings:\n")
                f.write(f"WAKE_WORD_THRESHOLD = {self.wake_word_threshold}\n")
                f.write(f"WAKE_WORD_MIN_VOLUME = {self.wake_word_min_volume}\n")
                f.write(f"WAKE_WORD_MAX_VOLUME = {self.wake_word_max_volume}\n\n")
                
                f.write("Recommended Settings:\n")
                for key, value in results['recommended_settings'].items():
                    f.write(f"{key} = {value}\n")
                
                # Add comparison
                f.write("\nChanges Needed:\n")
                current = {
                    "WAKE_WORD_THRESHOLD": self.wake_word_threshold,
                    "WAKE_WORD_MIN_VOLUME": self.wake_word_min_volume,
                    "WAKE_WORD_MAX_VOLUME": self.wake_word_max_volume
                }
                for key, new_value in results['recommended_settings'].items():
                    if key in current and current[key] != new_value:
                        f.write(f"{key}: {current[key]} -> {new_value}\n")
            
            print(f"\nResults saved to {results_file}")
            
        except Exception as e:
            print(f"Error saving calibration results: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    tester = AudioTester()
    asyncio.run(tester.run_tests()) 