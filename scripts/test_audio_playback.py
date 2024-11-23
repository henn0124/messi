import pyaudio
import wave
import numpy as np
import asyncio
import sys
from pathlib import Path
import time
import io

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.audio import AudioInterface
from src.core.config import Settings

class AudioTester:
    def __init__(self):
        self.settings = Settings()
        self.audio = AudioInterface()
        self.test_file = self.settings.CACHE_DIR / "tts" / "last_response.wav"
        
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
                print("5. Exit")
                
                choice = input("\nChoose test (1-5): ")
                
                if choice == '1':
                    await self.test_wake_word()
                elif choice == '2':
                    await self.test_playback()
                elif choice == '3':
                    await self.test_recording()
                elif choice == '4':
                    await self.calibrate_wake_word()
                elif choice == '5':
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
        """Test wake word detection using existing audio stream"""
        print("\n=== Wake Word Test ===")
        print("Say 'Hey Messy' to test wake word detection")
        print("Press Ctrl+C to stop")
        
        # Track detections and performance
        detections = 0
        frames_processed = 0
        start_time = time.time()
        last_stats = time.time()
        
        try:
            # Use existing stream from AudioInterface
            if not self.audio.stream or not self.audio.stream.is_active():
                print("Error: No active audio stream")
                return
                
            print("\nStarting wake word detection...")
            print("Required FPS: 31.25 (512 samples @ 16kHz)")
            print("\nStatus:")
            print("- Green bars (#): Good audio level")
            print("- Red bars (!): Audio level too high")
            print("- Blue bars (_): Audio level too low")
            
            # Calculate buffer sizes
            input_frames = int(44100 * 512 / 16000)  # Convert Porcupine frames to input rate
            print(f"Input buffer size: {input_frames} samples")
            
            while True:
                try:
                    # Read audio at native rate
                    data = self.audio.stream.read(input_frames, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16)
                    
                    # Resample to 16kHz
                    resampled = np.interp(
                        np.linspace(0, len(audio), 512),
                        np.arange(len(audio)),
                        audio
                    ).astype(np.int16)
                    
                    # Process frame
                    level = np.abs(resampled).mean()
                    frames_processed += 1
                    
                    # Visual feedback
                    if level > 4500:
                        print(f"\rLevel: {'!' * int(level/100)} (Too loud: {level:.0f})", end='', flush=True)
                    elif level > 150:
                        print(f"\rLevel: {'#' * int(level/100)} (Good: {level:.0f})", end='', flush=True)
                    else:
                        print(f"\rLevel: {'_' * int(level/100)} (Too soft: {level:.0f})", end='', flush=True)
                    
                    # Check for wake word
                    if 150 <= level <= 4500:
                        result = self.audio.porcupine.process(resampled)
                        if result >= 0:
                            detections += 1
                            print(f"\n\n🎤 Wake Word Detected! (#{detections})")
                            print(f"Level: {level:.0f}")
                            print(f"Time: {time.time() - start_time:.1f}s")
                    
                    # Show stats every 5 seconds
                    now = time.time()
                    if now - last_stats >= 5:
                        fps = frames_processed / (now - last_stats)
                        print(f"\nProcessed {frames_processed} frames ({fps:.1f} fps)")
                        frames_processed = 0
                        last_stats = now
                    
                    await asyncio.sleep(0.001)
                    
                except IOError as e:
                    print(f"\nStream error: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nWake Word Test Summary:")
            print(f"Total Detections: {detections}")
            print(f"Test Duration: {time.time() - start_time:.1f}s")
            if detections > 0:
                print(f"Average: {detections / (time.time() - start_time):.2f} detections/second")
            print("\nStopping wake word test")

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
        """Calibrate wake word settings with false positive testing"""
        print("\n=== Wake Word Calibration ===")
        
        try:
            if not self.audio.stream or not self.audio.stream.is_active():
                print("Error: No active audio stream")
                return
            
            # Initialize calibration data
            calibration_data = {
                "ambient_levels": [],
                "wake_word_levels": [],
                "detections": [],
                "missed_detections": 0,
                "false_positives": 0,
                "false_positive_levels": []
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
                
                while time.time() - start_time < 3:
                    data = self.audio.stream.read(1024, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16)
                    
                    resampled = np.interp(
                        np.linspace(0, len(audio), 512),
                        np.arange(len(audio)),
                        audio
                    ).astype(np.int16)
                    
                    level = np.abs(resampled).mean()
                    detection_window.append(level)
                    
                    result = self.audio.porcupine.process(resampled)
                    if result >= 0:
                        detected = True
                        calibration_data["detections"].append({
                            "level": level,
                            "time": time.time() - start_time
                        })
                    
                    print(f"\rLevel: {'#' * int(level/100)}", end='', flush=True)
                    await asyncio.sleep(0.001)
                
                if detected:
                    print("\n✓ Wake word detected!")
                    calibration_data["wake_word_levels"].extend(detection_window)
                else:
                    print("\n✗ No detection")
                    calibration_data["missed_detections"] += 1
                
                await asyncio.sleep(1)
            
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
            
            while time.time() - start_time < test_duration:
                data = self.audio.stream.read(1024, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                
                resampled = np.interp(
                    np.linspace(0, len(audio), 512),
                    np.arange(len(audio)),
                    audio
                ).astype(np.int16)
                
                level = np.abs(resampled).mean()
                calibration_data["false_positive_levels"].append(level)
                
                result = self.audio.porcupine.process(resampled)
                if result >= 0:
                    calibration_data["false_positives"] += 1
                    print("\n! False detection at level:", level)
                
                frames_processed += 1
                remaining = test_duration - (time.time() - start_time)
                print(f"\rLevel: {'#' * int(level/100)} ({remaining:.1f}s remaining)", end='', flush=True)
                await asyncio.sleep(0.001)
            
            # Calculate recommended settings
            wake_word_mean = np.mean(calibration_data["wake_word_levels"]) if calibration_data["wake_word_levels"] else 0
            wake_word_std = np.std(calibration_data["wake_word_levels"]) if calibration_data["wake_word_levels"] else 0
            false_positive_mean = np.mean(calibration_data["false_positive_levels"]) if calibration_data["false_positive_levels"] else 0
            
            # Adjust settings based on false positives
            threshold_adjust = 0.05 * calibration_data["false_positives"]  # Increase threshold if false positives
            min_volume_adjust = 20 * calibration_data["false_positives"]   # Increase minimum if false positives
            
            recommended_settings = {
                "WAKE_WORD_MIN_VOLUME": max(ambient_mean * 1.2 + min_volume_adjust, 100),
                "WAKE_WORD_MAX_VOLUME": min(wake_word_mean * 3, 5000),
                "WAKE_WORD_THRESHOLD": min(0.75 + threshold_adjust, 0.95),  # Adjust based on false positives
                "WAKE_WORD_DETECTION_WINDOW": 2.0,
                "WAKE_WORD_CONSECUTIVE_FRAMES": 1 + calibration_data["false_positives"]  # Require more frames if false positives
            }
            
            # Print results
            print("\n=== Calibration Results ===")
            print(f"\nAmbient Noise:")
            print(f"  Mean Level: {ambient_mean:.0f}")
            print(f"  Range: {min(calibration_data['ambient_levels']):.0f} - {max(calibration_data['ambient_levels']):.0f}")
            
            print(f"\nWake Word:")
            print(f"  Mean Level: {wake_word_mean:.0f}")
            print(f"  Std Dev: {wake_word_std:.0f}")
            print(f"  Successful Detections: {len(calibration_data['detections'])}/5")
            print(f"  Missed Detections: {calibration_data['missed_detections']}")
            
            print(f"\nFalse Positive Test:")
            print(f"  False Detections: {calibration_data['false_positives']}")
            print(f"  Speech Level: {false_positive_mean:.0f}")
            print(f"  Frames Processed: {frames_processed}")
            
            print("\nRecommended Settings:")
            for key, value in recommended_settings.items():
                print(f"{key} = {value}")
            
            # Save results
            results_file = Path("wake_word_calibration.txt")
            with open(results_file, "w") as f:
                f.write("=== Wake Word Calibration Results ===\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Raw Metrics:\n")
                f.write(f"Ambient Noise Mean: {ambient_mean:.0f}\n")
                f.write(f"Wake Word Mean: {wake_word_mean:.0f}\n")
                f.write(f"Wake Word Std Dev: {wake_word_std:.0f}\n")
                f.write(f"Successful Detections: {len(calibration_data['detections'])}/5\n")
                f.write(f"Missed Detections: {calibration_data['missed_detections']}\n")
                f.write(f"False Positives: {calibration_data['false_positives']}\n\n")
                
                f.write("Recommended Settings:\n")
                for key, value in recommended_settings.items():
                    f.write(f"{key} = {value}\n")
            
            print(f"\nResults saved to {results_file}")
            
        except Exception as e:
            print(f"Calibration error: {e}")
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

if __name__ == "__main__":
    tester = AudioTester()
    asyncio.run(tester.run_tests()) 