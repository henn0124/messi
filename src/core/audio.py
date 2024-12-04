"""
Audio Interface for Messi Assistant
---------------------------------
Handles audio I/O operations using PyAudio.
"""

import pyaudio
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
        self.p = None  # Initialize PyAudio only when needed
        self.stream = None
        self.output_stream = None
        
        # Audio configuration
        self.input_sample_rate = 48000  # Always use 48kHz
        self.output_sample_rate = 48000  # Always use 48kHz
        self.processing_rate = 16000     # Required for Porcupine
        
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

    def __del__(self):
        """Clean up resources"""
        self.cleanup()

    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
            
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except:
                pass
            self.output_stream = None
            
        if self.p:
            try:
                self.p.terminate()
            except:
                pass
            self.p = None

    async def initialize(self):
        """Initialize audio interface"""
        try:
            # Clean up any existing resources
            self.cleanup()
            
            print("\nInitializing audio...")
            self.p = pyaudio.PyAudio()
            
            # List available devices and find TONOR TM20
            print("\nLooking for TONOR TM20 device:")
            device_index = None
            
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
                return False
            
            # Initialize wake word detector first to get frame requirements
            from core.wake_word import WakeWordDetector
            self.wake_word_detector = WakeWordDetector(self.settings)
            self.frame_length = self.wake_word_detector.porcupine.frame_length
            self.samples_per_frame = int(self.frame_length * self.input_sample_rate / self.processing_rate)
            
            print(f"\nAudio Configuration:")
            print(f"Frame length (16kHz): {self.frame_length}")
            print(f"Samples per frame (48kHz): {self.samples_per_frame}")
            print(f"Sample rates:")
            print(f"  Input: {self.input_sample_rate}Hz")
            print(f"  Processing: {self.processing_rate}Hz")
            print(f"  Output: {self.output_sample_rate}Hz")
            
            # Open input stream
            print(f"\nInitializing input stream...")
            print(f"Using device index: {device_index}")
            
            try:
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.input_sample_rate,
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
                    rate=self.input_sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.samples_per_frame
                )
            
            if self.stream.is_active():
                print(f"✓ Audio input initialized at {self.input_sample_rate}Hz")
                return True
            
            print("❌ Failed to initialize audio")
            return False
            
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

    async def _monitor_wake_word(self):
        """Monitor audio stream for wake word"""
        min_volume = 100   # Minimum volume to show activity
        max_volume = 4000  # Maximum volume before clipping
        
        print("\nListening for wake word 'Hey Messy'...")
        print("Audio levels will be shown below. Speak normally.")
        
        while self.running:
            try:
                # Read audio frame
                data = self.stream.read(self.samples_per_frame, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                
                # Calculate and display volume level
                volume = np.abs(audio).mean()
                if volume > max_volume:
                    meter = "LOUD! 🔊"
                elif volume > min_volume:
                    meter_length = int((volume - min_volume) / (max_volume - min_volume) * 30)
                    meter = "▮" * meter_length
                else:
                    meter = "quiet"
                
                print(f"\rLevel: {meter:<32} ({volume:>4.0f})", end="", flush=True)
                
                # Resample from 48kHz to 16kHz for Porcupine
                resampled = np.interp(
                    np.linspace(0, len(audio), self.frame_length),
                    np.arange(len(audio)),
                    audio
                ).astype(np.int16)
                
                # Process with wake word detector
                result = self.wake_word_detector.process_frame(resampled)
                
                if result:
                    print("\n\n🎤 Wake word detected!")
                    print("Listening for command...")
                    
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

    async def _get_next_audio_frame(self):
        """Get audio frame with efficient processing for 48kHz"""
        try:
            # Run blocking audio read in executor
            pcm = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.stream.read,
                self.samples_per_frame,
                False
            )
            
            # Process with numpy for efficiency
            audio = np.frombuffer(pcm, dtype=np.int16)
            
            # Resample 48kHz to 16kHz more efficiently
            ratio = 16000 / 48000  # Fixed ratio for known rates
            target_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio)-1, target_length).astype(int)
            resampled = audio[indices]
            
            if len(resampled) > self.frame_length:
                return resampled[:self.frame_length]
            return np.pad(resampled, (0, self.frame_length - len(resampled)))
            
        except Exception as e:
            print(f"Error reading audio frame: {e}")
            return None

    async def _collect_command_audio(self, duration=3.0):
        """Collect audio for command with better end detection"""
        try:
            chunks = []
            start_time = time.time()
            silence_duration = 0
            last_level = 0
            SILENCE_THRESHOLD = 500  # Adjust based on your mic
            MAX_SILENCE = 0.5  # Maximum silence before cutting off
            
            print("\nRecording command...")
            print("Speak your command. Recording will stop after silence.")
            
            while time.time() - start_time < duration:
                data = self.stream.read(self.samples_per_frame, exception_on_overflow=False)
                chunks.append(data)
                
                # Calculate audio level
                audio = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio).mean()
                
                # Detect sentence completion
                if level < SILENCE_THRESHOLD:
                    silence_duration += self.samples_per_frame / self.input_sample_rate
                    if silence_duration > MAX_SILENCE and last_level > SILENCE_THRESHOLD:
                        print("\nDetected end of command")
                        break
                else:
                    silence_duration = 0
                
                last_level = level
                
                # Visual feedback
                meter_length = int(level/100)
                if meter_length > 30:
                    meter = "LOUD! 🔊"
                else:
                    meter = "▮" * meter_length
                print(f"\rLevel: {meter:<32} ({level:>4.0f})", end="", flush=True)
                
                await asyncio.sleep(0.001)
            
            print("\nProcessing command...")
            
            # Combine chunks
            if chunks:
                return b''.join(chunks)
            else:
                print("No audio collected")
                return None
                
        except Exception as e:
            print(f"Error collecting command audio: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
            return None

    async def stop(self):
        """Stop audio processing"""
        self.running = False
        self.cleanup()

    async def play_audio_chunk(self, audio_data: bytes) -> None:
        """Play an audio chunk of raw PCM data"""
        try:
            # Configure output stream if not already configured
            if not hasattr(self, 'output_stream') or self.output_stream is None:
                output_rate = 48000  # Always use 48kHz
                chunk_size = self.settings.audio["output"]["chunk_size"]
                print(f"\nInitializing output stream:")
                print(f"  Rate: {output_rate}Hz")
                print(f"  Chunk size: {chunk_size} samples")
                print(f"  Format: 16-bit PCM")
                print(f"  Channels: 1 (mono)")
                print(f"  Device index: {self.settings.audio['output']['device_index']}")
                
                self.output_stream = self.p.open(
                    format=pyaudio.paInt16,  # Assuming 16-bit audio
                    channels=1,              # Mono audio
                    rate=output_rate,        # Always 48kHz
                    output=True,
                    output_device_index=self.settings.audio["output"]["device_index"],
                    frames_per_buffer=chunk_size
                )
            
            # Write the audio data in chunks
            if len(audio_data) > 0:
                print(f"Writing {len(audio_data)} bytes to output stream")
                chunk_size = 1024
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    self.output_stream.write(chunk)
            
        except Exception as e:
            print(f"Error during playback: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()