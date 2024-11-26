"""
Audio Interface for Messi Assistant on Raspberry Pi
------------------------------------------------

This module handles all audio I/O operations for the Messi Assistant,
specifically optimized for Raspberry Pi 4 hardware setup.

Hardware Configuration:
    - Device: Raspberry Pi 4
    - Input: Maono AU-PM461TR USB Microphone
        - Sample Rate: 44.1kHz
        - Bit Depth: 16-bit
        - Channels: Mono
        - Device Index: 1
    
    - Output: Raspberry Pi 3.5mm Audio Jack
        - Sample Rate: 24kHz
        - Bit Depth: 16-bit
        - Channels: Mono
        - Device Index: 0

Audio Format Standards:
    1. Input (Microphone):
        - Format: WAV (PCM)
        - Sample Rate: 44.1kHz (native)
        - Bit Depth: 16-bit
        - Channels: Mono
        - Chunk Size: 2048 samples
    
    2. Processing:
        - Format: WAV (PCM)
        - Sample Rate: 16kHz (required by Whisper/Porcupine)
        - Bit Depth: 16-bit
        - Channels: Mono
        - Frame Size: 512 samples (Porcupine requirement)
    
    3. Output (Playback):
        - Format: WAV (PCM)
        - Sample Rate: 24kHz (OpenAI TTS)
        - Bit Depth: 16-bit
        - Channels: Mono
        - Chunk Size: 2048 samples

Usage:
    audio = AudioInterface()
    await audio.initialize()
    await audio.start_wake_word_detection(callback)
"""

import pyaudio
import numpy as np
from typing import Callable, Optional
import pvporcupine
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
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.output_stream = None
        
        # Audio configuration - fixed for Raspberry Pi 4 USB setup
        self.input_sample_rate = 48000  # USB speaker native rate
        self.output_sample_rate = 48000  # Match input rate
        self.processing_rate = 16000    # Required for Porcupine/Whisper
        self.frame_length = 512         # Required Porcupine frame length
        
        # Calculate frame sizes for 48kHz
        self.samples_per_frame = int(self.input_sample_rate * self.frame_length / self.processing_rate)
        
        # Thread and state management
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self.is_playing = False
        self.in_conversation = False
        self.interrupt_event = asyncio.Event()
        self.wake_word_callback = None
        
        # Initialize Porcupine
        try:
            print("\nInitializing wake word detector...")
            self.porcupine = pvporcupine.create(
                access_key=self.settings.PICOVOICE_ACCESS_KEY,
                keyword_paths=[str(self.settings.WAKE_WORD_MODEL_PATH)],
                sensitivities=[self.settings.WAKE_WORD_THRESHOLD]
            )
            print("✓ Wake word detector initialized")
            print(f"Frame length required: {self.porcupine.frame_length}")
            print(f"Input samples per frame: {self.samples_per_frame}")
            print(f"Sample rates:")
            print(f"  Input: {self.input_sample_rate}Hz")
            print(f"  Processing: {self.processing_rate}Hz")
            print(f"  Output: {self.output_sample_rate}Hz")
        except Exception as e:
            print(f"Error initializing wake word detector: {e}")
            raise

    async def initialize(self):
        """Initialize audio for Raspberry Pi 4 USB setup"""
        try:
            print("\nInitializing audio...")
            self.p = pyaudio.PyAudio()
            
            # List available devices
            print("\nAvailable Audio Devices:")
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                print(f"\nDevice {i}:")
                print(f"    Name: {dev['name']}")
                print(f"    Max Input Channels: {dev['maxInputChannels']}")
                print(f"    Max Output Channels: {dev['maxOutputChannels']}")
                print(f"    Default Sample Rate: {dev['defaultSampleRate']}")
            
            # Initialize input stream (TONOR TM20)
            print(f"\nInitializing input stream...")
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=48000,  # TONOR TM20 native rate (48kHz)
                input=True,
                input_device_index=2,  # TONOR TM20 index
                frames_per_buffer=self.samples_per_frame
            )
            
            if self.stream.is_active():
                print(f"✓ Audio input initialized at 48000Hz")
                return True
                
            print("❌ Failed to initialize audio")
            return False
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def start_wake_word_detection(self, callback):
        """Start wake word detection with proper async handling"""
        print("\nStarting wake word detection...")
        self.wake_word_callback = callback
        self.running = True
        
        # Create monitoring task instead of awaiting directly
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
        """Monitor audio stream for wake word with optimized 48kHz handling"""
        while self.running:
            try:
                # Read audio frame
                data = self.stream.read(self.samples_per_frame, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                
                # Resample 48kHz to 16kHz efficiently
                ratio = 16000 / 48000  # Fixed ratio for known rates
                target_length = int(len(audio) * ratio)
                resampled = np.interp(
                    np.linspace(0, len(audio)-1, target_length),
                    np.arange(len(audio)),
                    audio
                ).astype(np.int16)
                
                # Ensure correct frame length
                if len(resampled) > self.frame_length:
                    resampled = resampled[:self.frame_length]
                elif len(resampled) < self.frame_length:
                    resampled = np.pad(resampled, (0, self.frame_length - len(resampled)))
                
                # Process wake word
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.porcupine.process,
                    resampled
                )
                
                if result >= 0:
                    print("\n🎤 Wake word detected!")
                    print("Listening for command...")
                    
                    # Collect additional audio for command
                    command_audio = await self._collect_command_audio()
                    
                    if command_audio and self.wake_word_callback:
                        await self.wake_word_callback(command_audio)
                    else:
                        print("No command detected")
                
            except Exception as e:
                print(f"Error reading audio frame: {e}")
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
            
            print("Recording command...")
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
                        print("\nDetected sentence completion")
                        break
                else:
                    silence_duration = 0
                
                last_level = level
                
                # Visual feedback
                bar = "#" * int(level/100)
                print(f"\rLevel: {bar} ({level:.0f})", end='', flush=True)
                
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
            return None

    async def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.executor:
            self.executor.shutdown()

    async def play_audio_chunk(self, audio_data):
        """Play WAV audio data with detailed error handling"""
        try:
            if not audio_data:
                print("No audio data received")
                return
                
            print("\n▶ Playing response...")
            print(f"Received audio type: {type(audio_data)}")
            
            # Convert response to bytes
            if hasattr(audio_data, 'aread'):
                print("Using async read")
                audio_bytes = await audio_data.aread()
            elif hasattr(audio_data, 'read'):
                print("Using sync read")
                audio_bytes = audio_data.read()
            elif isinstance(audio_data, bytes):
                print("Already bytes")
                audio_bytes = audio_data
            else:
                print(f"Converting from {type(audio_data)}")
                audio_bytes = bytes(audio_data)
            
            print(f"Converted to bytes, size: {len(audio_bytes)}")
            
            # Read WAV data first to get format
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'rb') as wav:
                print(f"\nAudio format:")
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frame_rate = wav.getframerate()
                print(f"Channels: {channels}")
                print(f"Sample Width: {sample_width} bytes")
                print(f"Frame Rate: {frame_rate} Hz")
            
            # Initialize output stream
            if not self.output_stream:
                print("Initializing output stream...")
                self.output_stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=48000,  # Use 48kHz for USB audio
                    output=True,
                    output_device_index=1,  # USB2.0 Device Audio
                    frames_per_buffer=2048
                )
                print(f"Output stream initialized at {self.output_stream._rate}Hz")
            
            # Reopen WAV and play with proper resampling
            wav_buffer.seek(0)
            with wave.open(wav_buffer, 'rb') as wav:
                print("\nPlaying audio...")
                chunk_size = 2048
                data = wav.readframes(chunk_size)
                
                # Resample if input and output rates don't match
                if frame_rate != self.output_stream._rate:
                    print(f"Resampling from {frame_rate}Hz to {self.output_stream._rate}Hz")
                    ratio = self.output_stream._rate / frame_rate
                    while data and not self.interrupt_event.is_set():
                        # Convert to numpy array
                        audio = np.frombuffer(data, dtype=np.int16)
                        
                        # Calculate target length
                        target_length = int(len(audio) * ratio)
                        
                        # High quality resampling
                        resampled = np.interp(
                            np.linspace(0, len(audio)-1, target_length, endpoint=False),
                            np.arange(len(audio)),
                            audio
                        ).astype(np.int16)
                        
                        # Play resampled audio
                        self.output_stream.write(resampled.tobytes())
                        data = wav.readframes(chunk_size)
                else:
                    # Play at original rate if rates match
                    while data and not self.interrupt_event.is_set():
                        self.output_stream.write(data)
                        data = wav.readframes(chunk_size)
                
                print("✓ Playback complete")
                
        except Exception as e:
            print(f"Error during playback: {e}")
            traceback.print_exc()
            
            # Try to recover stream
            try:
                if self.output_stream:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                self.output_stream = None
            except:
                pass