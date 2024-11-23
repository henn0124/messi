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
        except Exception as e:
            print(f"Error initializing wake word detector: {e}")
            raise

    async def initialize(self):
        """Initialize audio with proper configuration"""
        try:
            # Get device info
            input_device_info = self.p.get_device_info_by_index(
                self.settings.AUDIO_INPUT_DEVICE_INDEX
            )
            
            # Calculate frame parameters
            self.input_sample_rate = int(input_device_info['defaultSampleRate'])
            self.frame_length = self.porcupine.frame_length
            self.samples_per_frame = int(self.input_sample_rate / 16000 * self.frame_length)
            
            print(f"\nAudio Configuration:")
            print(f"Sample Rate: {self.input_sample_rate}Hz")
            print(f"Frame Length: {self.frame_length}")
            print(f"Samples per Frame: {self.samples_per_frame}")
            
            # Initialize input stream
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.input_sample_rate,
                input=True,
                frames_per_buffer=self.samples_per_frame,
                input_device_index=self.settings.AUDIO_INPUT_DEVICE_INDEX
            )
            
            return True
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
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
        """Monitor audio stream with enhanced logging"""
        while self.running:
            try:
                # Get audio frame non-blocking
                pcm = await self._get_next_audio_frame()
                if pcm is None:
                    continue
                
                # Process wake word in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.porcupine.process,
                    pcm
                )
                
                if result >= 0:
                    print("\n\n🎤 Wake word detected!")
                    print("Listening for command...")
                    
                    # Handle any ongoing playback
                    if self.is_playing:
                        self.interrupt_event.set()
                        await asyncio.sleep(0.1)
                    
                    # Get command audio
                    command_audio = await self._collect_command_audio()
                    if command_audio and self.wake_word_callback:
                        await self.wake_word_callback(command_audio)
                    else:
                        print("\n⏳ No valid command detected")
                    
                    print("\nListening for wake word...")
                
                # Minimal sleep to prevent CPU overuse
                await asyncio.sleep(0.001)
                    
            except Exception as e:
                print(f"Error in wake word monitoring: {e}")
                await asyncio.sleep(0.1)

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

    async def _collect_command_audio(self):
        """Collect command audio with smart end detection"""
        try:
            print("\n=== Recording Command ===")
            chunks = []
            silence_count = 0
            speech_started = False
            max_silence_frames = 25  # Longer silence tolerance (~0.5s)
            max_wait_for_speech = 2.0  # Seconds to wait for speech to start
            max_recording_duration = 10.0  # Maximum total duration
            natural_pause_threshold = 10  # Allow shorter pauses during speech
            
            start_time = time.time()
            last_speech_time = start_time
            
            # Pre-allocate buffer
            audio_buffer = np.zeros(1024, dtype=np.int16)
            
            print("Listening...")
            
            while True:
                # Non-blocking read
                chunk = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.stream.read,
                    1024,
                    False
                )
                
                # Process audio level
                np.copyto(audio_buffer, np.frombuffer(chunk, dtype=np.int16))
                level = float(np.abs(audio_buffer).mean())
                
                # Create equalizer visualization
                height = min(10, int((level / self.settings.WAKE_WORD_MAX_VOLUME) * 10))
                bars = [("█" if i < height else "░") for i in range(10)]
                viz = f"\r│{''.join(bars)}│ {level:4.0f}"
                print(viz, end='', flush=True)
                
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check for speech activity
                if level > self.settings.COMMAND_MIN_VOLUME:
                    if not speech_started:
                        speech_started = True
                        print("\n✓ Speech detected")
                    
                    chunks.append(chunk)
                    silence_count = 0
                    last_speech_time = current_time
                else:
                    silence_count += 1
                    
                    # Only add silence after speech has started
                    if speech_started:
                        chunks.append(chunk)
                
                # Check various end conditions
                if speech_started:
                    # End on long silence after speech
                    silence_duration = current_time - last_speech_time
                    if silence_duration > 1.0:  # 1 second of silence
                        print("\n✓ Command complete (silence)")
                        break
                        
                    # Natural pause handling
                    if silence_count > natural_pause_threshold:
                        # Peek at next chunk for upcoming speech
                        peek_chunk = self.stream.read(1024, exception_on_overflow=False)
                        peek_level = float(np.abs(np.frombuffer(peek_chunk, dtype=np.int16)).mean())
                        if peek_level > self.settings.COMMAND_MIN_VOLUME:
                            # Speech continues, add silence and continue
                            chunks.append(chunk)
                            silence_count = 0
                        elif silence_count > max_silence_frames:
                            print("\n✓ Command complete (end of speech)")
                            break
                
                else:
                    # No speech detected within timeout
                    if elapsed > max_wait_for_speech:
                        print("\n✗ No speech detected")
                        return None
                
                # Check maximum duration
                if elapsed > max_recording_duration:
                    print("\n✓ Command complete (max duration)")
                    break
                
                await asyncio.sleep(0.001)
            
            # Validate recording
            if not speech_started or len(chunks) < 3:
                print("\n✗ Command too short")
                return None
            
            # Create WAV data
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.input_sample_rate)
                wf.writeframes(b''.join(chunks))
            
            wav_data = wav_buffer.getvalue()
            duration = len(chunks) * 1024 / self.input_sample_rate
            
            print(f"\n✓ Command recorded:")
            print(f"Duration: {duration:.1f}s")
            print(f"Sample Rate: {self.input_sample_rate}Hz")
            print(f"Size: {len(wav_data)/1024:.1f}KB")
            print("\n▶ Sending to Whisper for processing...")
            
            return wav_data
            
        except Exception as e:
            print(f"\n✗ Error collecting command: {e}")
            return None

    async def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.executor:
            self.executor.shutdown()