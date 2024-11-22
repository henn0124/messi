import pyaudio
import numpy as np
from typing import Callable, Optional
import pvporcupine
import struct
from .config import Settings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import wave
import io
import time
from .audio_processor import AudioProcessor
import traceback

class AudioInterface:
    def __init__(self):
        # Get settings
        self.settings = Settings()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.output_stream = None
        
        # Build audio settings from environment
        self.AUDIO_SETTINGS = {
            "sample_rates": {
                "input_native": self.settings.AUDIO_NATIVE_RATE,
                "processing": self.settings.AUDIO_PROCESSING_RATE,
                "output": self.settings.AUDIO_OUTPUT_RATE
            },
            "formats": {
                "width": 2,             # 16-bit
                "channels": 1,          # Mono
                "format": pyaudio.paInt16
            },
            "wake_word": {
                "sensitivity": self.settings.WAKE_WORD_THRESHOLD,
                "min_volume": self.settings.WAKE_WORD_MIN_VOLUME,
                "max_volume": self.settings.WAKE_WORD_MAX_VOLUME,
                "detection_window": 0.5,
                "consecutive_frames": 1
            },
            "audio_processing": {
                "pre_emphasis": 0.97,
                "silence_threshold": 100,
                "chunk_size": 1024,
                "buffer_size": 8192
            },
            "devices": {
                "input_device_index": self.settings.AUDIO_INPUT_DEVICE_INDEX,
                "output_device_index": self.settings.AUDIO_OUTPUT_DEVICE_INDEX
            }
        }
        
        # Initialize Porcupine
        try:
            print("\nInitializing Porcupine wake word detector...")
            self.porcupine = pvporcupine.create(
                access_key=self.settings.PICOVOICE_ACCESS_KEY,
                keyword_paths=[str(self.settings.WAKE_WORD_MODEL_PATH)],
                sensitivities=[self.settings.WAKE_WORD_THRESHOLD]
            )
            
            # Update settings with Porcupine requirements
            self.AUDIO_SETTINGS.update({
                "porcupine": {
                    "frame_length": self.porcupine.frame_length,
                    "sample_rate": self.porcupine.sample_rate
                }
            })
            
            print(f"\n=== Audio Configuration ===")
            print(f"Input Device: Maono Elf USB Microphone")
            print(f"Native Sample Rate: {self.AUDIO_SETTINGS['sample_rates']['input_native']}Hz")
            print(f"Processing Rate: {self.AUDIO_SETTINGS['sample_rates']['processing']}Hz")
            print(f"Frame Length: {self.AUDIO_SETTINGS['porcupine']['frame_length']} samples")
            print(f"Wake Word Sensitivity: {self.AUDIO_SETTINGS['wake_word']['sensitivity']}")
            print(f"Volume Range: {self.AUDIO_SETTINGS['wake_word']['min_volume']} - {self.AUDIO_SETTINGS['wake_word']['max_volume']}")
            
        except Exception as e:
            print(f"Error initializing audio interface: {e}")
            raise
        
        # State tracking
        self.running = False
        self.is_playing = False
        self.wake_word_callback = None
        self.interrupt_event = asyncio.Event()
        
        # Audio processing state
        self.audio_buffer = []
        self.last_audio_level = 0
        self.peak_audio_level = 0
        self.detection_window = []
        self.consecutive_detections = 0
    
    async def initialize(self):
        """Initialize audio stream with proper settings"""
        try:
            # List available devices
            print("\nAvailable Audio Devices:")
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                print(f"Index {i}: {dev['name']}")
                print(f"  Max Input Channels: {dev['maxInputChannels']}")
                print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
            
            # Open input stream at native rate
            self.stream = self.p.open(
                rate=self.AUDIO_SETTINGS["sample_rates"]["input_native"],
                channels=self.AUDIO_SETTINGS["formats"]["channels"],
                format=self.AUDIO_SETTINGS["formats"]["format"],
                input=True,
                input_device_index=self.AUDIO_SETTINGS["devices"]["input_device_index"],
                frames_per_buffer=self._calculate_buffer_size()
            )
            
            print("\n✓ Audio stream initialized")
            return True
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self.stream = None
            return False
    
    def _calculate_buffer_size(self) -> int:
        """Calculate appropriate buffer size based on rates"""
        native_rate = self.AUDIO_SETTINGS["sample_rates"]["input_native"]
        target_rate = self.AUDIO_SETTINGS["sample_rates"]["processing"]
        base_size = self.AUDIO_SETTINGS["porcupine"]["frame_length"]
        
        return int(base_size * native_rate / target_rate)

    async def start_wake_word_detection(self, callback: Callable):
        """Start wake word detection"""
        try:
            self.running = True
            self.wake_word_callback = callback
            
            # Start monitoring task
            self.monitor_task = asyncio.create_task(self._monitor_wake_word())
            print("\nWake word detection started")
            print("Listening for 'Hey Messy'...")
            return True
            
        except Exception as e:
            print(f"Error starting wake word detection: {e}")
            return False

    async def stop(self):
        """Stop all audio processing"""
        try:
            print("\nStopping audio interface...")
            self.running = False
            
            # Cancel monitoring task
            if hasattr(self, 'monitor_task'):
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Close streams
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            
            # Clean up PyAudio
            if self.p:
                self.p.terminate()
            
            # Clean up Porcupine
            if self.porcupine:
                self.porcupine.delete()
            
            print("✓ Audio interface stopped")
            
        except Exception as e:
            print(f"Error stopping audio interface: {e}")

    async def _monitor_wake_word(self):
        """Monitor audio for wake word detection"""
        if not self.stream:
            print("Error: Audio stream not initialized")
            return
            
        print("\n=== Starting Wake Word Detection ===")
        print("Listening for 'Hey Messy'...")
        
        detection_window = []
        window_size = int(self.AUDIO_SETTINGS["wake_word"]["detection_window"] * 
                         self.AUDIO_SETTINGS["sample_rates"]["processing"] / 
                         self.porcupine.frame_length)
        
        while self.running and self.stream:
            try:
                # Get audio frame
                audio_frame = await self._get_next_audio_frame()
                if audio_frame is None:
                    continue
                
                # Convert to required format
                pcm = np.frombuffer(audio_frame, dtype=np.int16)
                
                # Calculate audio level
                audio_level = np.abs(pcm).mean()
                
                # Show audio activity
                if audio_level > self.AUDIO_SETTINGS["wake_word"]["min_volume"]:
                    level_indicator = "#" * int(audio_level / 100)
                    print(f"\rLevel: {level_indicator}", end="", flush=True)
                
                # Process for wake word if frame size is correct
                if len(pcm) == self.porcupine.frame_length:
                    result = self.porcupine.process(pcm)
                    
                    # Add result to detection window
                    detection_window.append((result, audio_level))
                    if len(detection_window) > window_size:
                        detection_window.pop(0)
                    
                    # Check for wake word in window
                    detections = sum(1 for r, _ in detection_window if r >= 0)
                    avg_level = np.mean([l for _, l in detection_window])
                    
                    if detections > 0:
                        print(f"\n\n🎯 Wake word detected!")
                        print(f"Audio level: {avg_level:.0f}")
                        
                        if avg_level > self.AUDIO_SETTINGS["wake_word"]["min_volume"]:
                            print("\n=== Wake Word Confirmed! ===")
                            print("Starting command collection...")
                            
                            # Clear detection window
                            detection_window.clear()
                            
                            # Collect command
                            command_audio = await self._collect_command_audio()
                            if command_audio and self.wake_word_callback:
                                await self.wake_word_callback(command_audio)
                            
                            print("\nListening for wake word...")
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"\nError in wake word monitoring: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)

    async def _get_next_audio_frame(self) -> Optional[bytes]:
        """Get next audio frame with proper resampling"""
        try:
            if not self.stream:
                return None
            
            # Calculate required samples at input rate
            input_rate = int(self.stream._rate)
            input_samples = int(input_rate/16000 * self.porcupine.frame_length)
            
            # Read at native rate
            data = self.stream.read(input_samples, exception_on_overflow=False)
            
            # Convert to numpy array
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Apply pre-emphasis filter
            pre_emphasis = self.AUDIO_SETTINGS["audio_processing"]["pre_emphasis"]
            emphasized = np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1])
            
            # Normalize audio
            normalized = emphasized / (np.max(np.abs(emphasized)) + 1e-10)
            normalized = (normalized * 32767).astype(np.int16)
            
            # Resample to 16kHz for Porcupine
            resampled = np.interp(
                np.linspace(0, len(normalized)-1, self.porcupine.frame_length),
                np.arange(len(normalized)),
                normalized
            ).astype(np.int16)
            
            return resampled.tobytes()
            
        except Exception as e:
            print(f"Error getting audio frame: {e}")
            return None

    async def _collect_command_audio(self) -> Optional[bytes]:
        """Collect command audio with improved silence detection"""
        try:
            audio_chunks = []
            silence_count = 0
            max_silence = 8  # Number of silent chunks before stopping
            speech_levels = []
            
            print("\n=== Waiting for Command ===")
            print("▶ Listening for your request...")
            
            while silence_count < max_silence:
                chunk = await self._get_next_audio_frame()
                if not chunk:
                    continue
                
                # Monitor levels
                audio_array = np.frombuffer(chunk, dtype=np.int16)
                audio_level = np.abs(audio_array).mean()
                
                # Update speech levels if we detect speech
                if audio_level > self.AUDIO_SETTINGS["wake_word"]["min_volume"]:
                    speech_levels.append(audio_level)
                
                # Calculate adaptive silence threshold
                silence_threshold = (
                    np.mean(speech_levels) * 0.2 if speech_levels 
                    else self.AUDIO_SETTINGS["wake_word"]["min_volume"] * 0.3
                )
                
                # Visual feedback
                level_indicator = "#" * int(audio_level / 100)
                silence_indicator = "-" * int(silence_threshold / 100)
                print(f"Level: {level_indicator}")
                print(f"Silence: {silence_indicator}")
                
                if audio_level > silence_threshold:
                    if not audio_chunks:  # First speech detected
                        print("✓ Speech detected, listening...")
                    audio_chunks.append(chunk)
                    silence_count = 0
                elif audio_chunks:  # Already collecting and detected silence
                    silence_count += 1
                    print(f"Silence detected ({silence_count}/{max_silence})")
                    audio_chunks.append(chunk)  # Keep some silence
                
                # Break if we've collected too much audio
                if len(audio_chunks) > 100:  # About 10 seconds
                    print("\n⚠️  Command too long, processing...")
                    break
            
            if audio_chunks:
                # Check if ending was clear
                ending_type = "clear ending" if silence_count >= max_silence else "length limit"
                print(f"\n✓ Speech ended ({ending_type})")
                
                # Join chunks and return
                command_audio = b''.join(audio_chunks)
                print(f"\n▶ Processing command ({len(command_audio)/1024:.1f}KB)...")
                return command_audio
            
            print("\n❌ No command detected")
            return None
            
        except Exception as e:
            print(f"Error collecting command: {e}")
            traceback.print_exc()
            return None

    async def play_audio_chunk(self, audio_data: bytes):
        """Play audio with proper initialization and error handling"""
        try:
            # Initialize output stream if needed
            if not self.output_stream:
                self.output_stream = self.p.open(
                    format=self.AUDIO_SETTINGS["formats"]["format"],
                    channels=1,
                    rate=self.AUDIO_SETTINGS["sample_rates"]["output"],
                    output=True,
                    output_device_index=self.AUDIO_SETTINGS["devices"]["output_device_index"]
                )
            
            print("\n▶ Playing audio...")
            self.is_playing = True
            
            # Play in smaller chunks to allow interruption
            chunk_size = self.AUDIO_SETTINGS["audio_processing"]["chunk_size"]
            for i in range(0, len(audio_data), chunk_size):
                if self.interrupt_event.is_set():
                    print("\n⚠️  Playback interrupted")
                    break
                    
                chunk = audio_data[i:i + chunk_size]
                self.output_stream.write(chunk)
                await asyncio.sleep(0.001)  # Allow other tasks to run
            
            self.is_playing = False
            self.interrupt_event.clear()
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_playing = False
        finally:
            # Reset interrupt flag
            self.interrupt_event.clear()

    async def _handle_wake_word(self, audio_data: bytes):
        """Handle wake word detection with debug info"""
        try:
            print("\n=== Audio Processing Debug ===")
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            duration = len(audio_array) / self.SAMPLE_RATE
            rms_level = np.sqrt(np.mean(np.square(audio_array)))
            
            print(f"Audio size: {len(audio_data)/1024:.1f}KB")
            print(f"Duration: {duration:.2f}s")
            print(f"RMS level: {rms_level:.0f}")
            
            if self.wake_word_callback:
                await self.wake_word_callback(audio_data)
            else:
                print("No wake word callback registered")
                
        except Exception as e:
            print(f"Error handling wake word: {e}")

    # ... rest of the methods using AUDIO_SETTINGS ...