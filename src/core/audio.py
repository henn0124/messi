"""
Audio Interface for Messi Assistant
---------------------------------
Handles audio I/O operations using ALSA directly for low latency.
"""

import alsaaudio
import numpy as np
from typing import Callable, Optional
import wave
import io
import asyncio
import traceback
from .config import Settings
import time
from concurrent.futures import ThreadPoolExecutor

class AudioInterface:
    def __init__(self):
        # Core initialization
        self.settings = Settings()
        self.input_stream = None
        self.output_stream = None
        
        # Audio configuration from settings
        self.input_rate = self.settings.audio["input"]["rate"]
        self.output_rate = self.settings.audio["output"]["rate"]
        self.processing_rate = self.settings.audio["input"]["processing_rate"]
        self.period_size = 4096  # Increased for better conversation handling
        
        # Thread and state management
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self.is_playing = False
        self.in_conversation = False
        self.interrupt_event = asyncio.Event()
        self.wake_word_callback = None
        
        # Initialize frame sizes after Porcupine is created
        self.frame_length = None
        self.samples_per_frame = None
        
        # Pre-allocate resampling arrays
        self.resample_x = None
        self.resample_y = None
        
        # Ring buffer for command collection
        self.ring_buffer = np.array([], dtype=np.int16)
        self.ring_buffer_size = int(self.input_rate * 10)  # Increased to 10 seconds for longer sentences
        
        # Debug counters
        self.frames_processed = 0
        self.last_debug_time = time.time()
        self.debug_interval = 5.0  # Print debug every 5 seconds

    def __del__(self):
        """Clean up resources"""
        self.cleanup()

    def cleanup(self):
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

    async def initialize(self):
        """Initialize audio interface"""
        try:
            # Clean up any existing resources
            self.cleanup()
            
            print("\nInitializing audio...")
            
            # List available devices
            print("\nAvailable PCM capture devices:")
            for device in alsaaudio.pcms(alsaaudio.PCM_CAPTURE):
                print(f"  {device}")
            
            print("\nAvailable PCM playback devices:")
            for device in alsaaudio.pcms(alsaaudio.PCM_PLAYBACK):
                print(f"  {device}")
            
            # Get device names from settings
            capture_device = self.settings.audio["input"]["device"]
            playback_device = self.settings.audio["output"]["device"]
            
            print(f"\nUsing devices:")
            print(f"  Capture: {capture_device}")
            print(f"  Playback: {playback_device}")
            
            # Initialize wake word detector first to get frame requirements
            from core.wake_word import WakeWordDetector
            self.wake_word_detector = WakeWordDetector(self.settings)
            self.frame_length = self.wake_word_detector.porcupine.frame_length
            self.samples_per_frame = int(self.frame_length * self.input_rate / self.processing_rate)
            
            print(f"\nAudio Configuration:")
            print(f"Frame length (16kHz): {self.frame_length}")
            print(f"Samples per frame (48kHz): {self.samples_per_frame}")
            print(f"Sample rates:")
            print(f"  Input: {self.input_rate}Hz")
            print(f"  Processing: {self.processing_rate}Hz")
            print(f"  Output: {self.output_rate}Hz")
            print(f"Period size: {self.period_size} frames")
            
            # Open input stream
            print(f"\nInitializing input stream...")
            
            self.input_stream = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,  # Blocking mode
                device=capture_device,
                format=alsaaudio.PCM_FORMAT_S16_LE,  # 16-bit signed little-endian
                channels=self.settings.audio["input"]["channels"],
                rate=self.input_rate,
                periodsize=self.period_size
            )
            
            print("✓ Audio input initialized")
            return True
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            traceback.print_exc()
            return False

    async def start_wake_word_detection(self, callback):
        """Start wake word detection"""
        print("\nStarting wake word detection...")
        self.wake_word_callback = callback
        self.running = True
        
        # Create monitoring task
        monitoring_task = asyncio.create_task(self._monitor_wake_word())
        
        # Start keep-alive task
        asyncio.create_task(self._keep_monitoring(monitoring_task))
        
        print("Wake word detection active")
        print("\nListening for wake word...")

    async def _keep_monitoring(self, monitoring_task):
        """Keep monitoring task alive"""
        try:
            while self.running:
                if monitoring_task.done():
                    print("\nRestarting wake word monitoring...")
                    monitoring_task = asyncio.create_task(self._monitor_wake_word())
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in monitoring loop: {e}")

    def _resample_audio(self, audio_48k):
        """Resample 48kHz audio to 16kHz using pre-allocated arrays"""
        try:
            # Initialize resampling arrays if needed
            if self.resample_x is None:
                self.resample_x = np.linspace(0, len(audio_48k) - 1, len(audio_48k))
                self.resample_y = np.linspace(0, len(audio_48k) - 1, self.frame_length)
            
            # Use pre-allocated arrays for faster resampling
            resampled = np.interp(self.resample_y, self.resample_x[:len(audio_48k)], audio_48k).astype(np.int16)
            
            # Ensure exact frame length
            if len(resampled) > self.frame_length:
                return resampled[:self.frame_length]
            elif len(resampled) < self.frame_length:
                return np.pad(resampled, (0, self.frame_length - len(resampled)))
            return resampled
            
        except Exception as e:
            print(f"Error in resampling: {e}")
            return np.zeros(self.frame_length, dtype=np.int16)

    async def _monitor_wake_word(self):
        """Monitor audio stream for wake word"""
        min_volume = self.settings.wake_word["volume_threshold"]
        max_volume = self.settings.wake_word["max_volume"]
        buffer = np.array([], dtype=np.int16)  # Buffer for incomplete frames
        
        print("\nListening for wake word 'Hey Messy'...")
        print("Audio levels will be shown below. Speak normally.")
        
        while self.running:
            try:
                # Read audio frame
                length, data = self.input_stream.read()
                if length > 0:
                    # Convert to numpy array and add to buffer
                    audio = np.frombuffer(data, dtype=np.int16)
                    buffer = np.concatenate([buffer, audio])
                    
                    # Calculate and display volume level
                    volume = np.abs(audio).mean()
                    
                    # Print debug info periodically
                    current_time = time.time()
                    if current_time - self.last_debug_time >= self.debug_interval:
                        print(f"\nDebug Info:")
                        print(f"  Frames processed: {self.frames_processed}")
                        print(f"  Current volume: {volume:.0f}")
                        print(f"  Buffer length: {len(buffer)}")
                        print(f"  Frame length needed: {self.samples_per_frame}")
                        print(f"  Min value: {np.min(audio)}")
                        print(f"  Max value: {np.max(audio)}")
                        self.last_debug_time = current_time
                    
                    # Visual volume meter
                    if volume > max_volume:
                        meter = "LOUD! 🔊"
                    elif volume > min_volume:
                        meter_length = int((volume - min_volume) / (max_volume - min_volume) * 30)
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
                        self.frames_processed += 1
                        
                        if result:
                            print("\n\n🎤 Wake word detected!")
                            print("Listening for command...")
                            
                            # Clear buffer
                            buffer = np.array([], dtype=np.int16)
                            
                            # Collect additional audio for command
                            command_audio = await self._collect_command_audio()
                            
                            if command_audio and self.wake_word_callback:
                                await self.wake_word_callback(command_audio)
                            else:
                                print("No command detected")
                                print("\nListening for wake word 'Hey Messy'...")
                                print("Audio levels will be shown below. Speak normally.")
                
            except Exception as e:
                print(f"\nError reading audio frame: {e}")
                if hasattr(e, '__traceback__'):
                    traceback.print_exc()
                await asyncio.sleep(0.1)
                continue

    async def _collect_command_audio(self):
        """Collect audio for command processing"""
        try:
            record_start = time.time()
            print("\nRecording command...")
            print("Speak your command. Recording will stop after silence.")
            
            # Initialize variables for silence detection
            silence_start = None
            # Default values if not in settings
            silence_threshold = self.settings.audio["input"].get("silence_threshold", 500)  # Default threshold
            silence_duration = self.settings.audio["input"].get("silence_duration", 0.5)    # Default 0.5 seconds
            
            print(f"\nSilence Detection Parameters:")
            print(f"  Threshold: {silence_threshold}")
            print(f"  Duration: {silence_duration}s")
            
            # Clear ring buffer at start of command
            self.ring_buffer = np.array([], dtype=np.int16)
            
            # Record until silence
            while True:
                length, data = self.input_stream.read()
                if length > 0:
                    # Convert to numpy array
                    audio = np.frombuffer(data, dtype=np.int16)
                    
                    # Calculate volume
                    volume = np.abs(audio).mean()
                    
                    # Visual volume meter
                    meter_length = int(volume / 100)
                    meter = "▮" * min(meter_length, 30)
                    print(f"\rLevel: {meter:<32} ({volume:>4.0f})", end="", flush=True)
                    
                    # Add to ring buffer
                    self.ring_buffer = np.concatenate([self.ring_buffer, audio])
                    
                    # Trim buffer if too long
                    if len(self.ring_buffer) > self.ring_buffer_size:
                        self.ring_buffer = self.ring_buffer[-self.ring_buffer_size:]
                    
                    # Check for silence
                    if volume < silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > silence_duration:
                            record_time = time.time() - record_start
                            print(f"\n⏱️  Recording time: {record_time*1000:.1f}ms")
                            audio_data = self.ring_buffer.tobytes()
                            # Clear ring buffer after command
                            self.ring_buffer = np.array([], dtype=np.int16)
                            return audio_data
                    else:
                        silence_start = None
                        
        except Exception as e:
            print(f"Error collecting command audio: {e}")
            traceback.print_exc()
            return None

    async def play_audio_chunk(self, audio_data: bytes) -> None:
        """Play audio chunk with fallback device handling"""
        try:
            # Inspect WAV file properties
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                wav_channels = wf.getnchannels()
                wav_rate = wf.getframerate()
                wav_width = wf.getsampwidth()
                print(f"\nWAV file properties:")
                print(f"  Sample rate: {wav_rate}Hz")
                print(f"  Channels: {wav_channels}")
                print(f"  Sample width: {wav_width} bytes")
            
            # Try to initialize output stream if needed
            if not self.output_stream:
                # Get device settings from config
                output_config = self.settings.audio["output"]
                device = output_config["device"]
                
                # Fallback devices if primary fails
                playback_devices = [
                    device,
                    "hw:3,0",
                    "default"
                ]
                
                # Try each device until one works
                for device in playback_devices:
                    try:
                        print(f"\nTrying audio device: {device}")
                        # Use WAV file's native rate instead of config
                        self.output_stream = alsaaudio.PCM(
                            type=alsaaudio.PCM_PLAYBACK,
                            mode=alsaaudio.PCM_NORMAL,
                            device=device,
                            format=alsaaudio.PCM_FORMAT_S16_LE,
                            channels=wav_channels,
                            rate=wav_rate,  # Use WAV file's rate
                            periodsize=output_config["period_size"]
                        )
                        print(f"✓ Using audio device: {device}")
                        print(f"  Sample rate: {wav_rate}Hz")
                        print(f"  Channels: {wav_channels}")
                        break
                    except alsaaudio.ALSAAudioError as e:
                        print(f"Could not open {device}: {e}")
                        continue
                
                if not self.output_stream:
                    raise RuntimeError("No available audio output devices found")
            
            # Play the audio directly
            self.output_stream.write(audio_data)
            
        except Exception as e:
            print(f"Error during playback: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
            
            # Reset output stream on error
            if self.output_stream:
                try:
                    self.output_stream.close()
                except:
                    pass
                self.output_stream = None

    async def stop(self):
        """Stop audio processing"""
        self.running = False
        self.cleanup()