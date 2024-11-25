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
        
        # Audio configuration
        self.input_sample_rate = 44100  # Native input rate
        self.output_sample_rate = 24000  # TTS output rate
        self.processing_rate = 16000    # Porcupine/Whisper rate
        self.frame_length = 512         # Porcupine frame requirement
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
        """Initialize audio with better error handling"""
        try:
            print("\nInitializing PyAudio...")
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
            
            # Initialize input stream
            print("\nInitializing input stream...")
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.input_sample_rate,
                input=True,
                input_device_index=1,  # TONOR TM20
                frames_per_buffer=self.samples_per_frame
            )
            
            if not self.stream.is_active():
                print("❌ Failed to activate audio stream")
                return False
                
            print("✓ Audio initialized successfully")
            return True
            
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
        """Monitor audio stream for wake word"""
        while self.running:
            try:
                # Read audio frame
                data = self.stream.read(self.samples_per_frame, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                
                # Resample to 16kHz for Porcupine
                resampled = np.interp(
                    np.linspace(0, len(audio), self.frame_length),
                    np.arange(len(audio)),
                    audio
                ).astype(np.int16)
                
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
        """Get audio frame with efficient processing"""
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
            
            # Resample if needed
            if self.input_sample_rate != 16000:
                ratio = 16000 / self.input_sample_rate
                target_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio)-1, target_length).astype(int)
                resampled = audio[indices]
                
                if len(resampled) > self.frame_length:
                    return resampled[:self.frame_length]
                return np.pad(resampled, (0, self.frame_length - len(resampled)))
            
            return audio
            
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
            
            # Initialize output stream with Raspberry Pi safe rate
            if not self.output_stream:
                print("Initializing output stream...")
                # Try 48000Hz first, then 44100Hz if that fails
                try:
                    print("Trying 48000Hz output...")
                    self.output_stream = self.p.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=48000,
                        output=True,
                        output_device_index=self.settings.AUDIO_OUTPUT_DEVICE_INDEX,
                        frames_per_buffer=2048
                    )
                except:
                    print("48000Hz failed, trying 44100Hz...")
                    self.output_stream = self.p.open(
                        format=pyaudio.paInt16,
                        channels=channels,
                        rate=44100,
                        output=True,
                        output_device_index=self.settings.AUDIO_OUTPUT_DEVICE_INDEX,
                        frames_per_buffer=2048
                    )
                print(f"Output stream initialized at {self.output_stream._rate}Hz")
            
            # Reopen WAV and resample for playback
            wav_buffer.seek(0)
            with wave.open(wav_buffer, 'rb') as wav:
                print("\nPlaying audio...")
                chunk_size = 2048
                data = wav.readframes(chunk_size)
                
                # Resample data if needed
                if frame_rate != self.output_stream._rate:
                    ratio = self.output_stream._rate / frame_rate
                    while data:
                        # Convert to numpy array
                        audio = np.frombuffer(data, dtype=np.int16)
                        # Resample
                        resampled = np.interp(
                            np.linspace(0, len(audio)-1, int(len(audio) * ratio)),
                            np.arange(len(audio)),
                            audio
                        ).astype(np.int16)
                        # Play
                        self.output_stream.write(resampled.tobytes())
                        data = wav.readframes(chunk_size)
                else:
                    # Play at original rate
                    while data and not self.interrupt_event.is_set():
                        self.output_stream.write(data)
                        data = wav.readframes(chunk_size)
                
                print("✓ Playback complete")
                
        except Exception as e:
            print(f"Error during playback: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to recover stream
            try:
                if self.output_stream:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                self.output_stream = None
            except:
                pass