import pyaudio
import numpy as np
from typing import Callable
import pvporcupine
import struct
from .config import Settings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import wave
import io
import time

class AudioInterface:
    # Audio Configuration Parameters
    AUDIO_CONFIG = {
        # Wake word detection settings
        "wake_word": {
            "sensitivity": 0.5,        # How sensitive wake word detection is (0.0-1.0). Higher = more sensitive
            "buffer_size": 4096,       # Audio buffer size for wake word processing
            "sample_rate": 16000,      # Required sample rate for wake word detection (Porcupine requirement)
        },
        
        # Speech detection thresholds
        "speech_detection": {
            "noise_floor": 100,        # Base level for background noise
            "silence_multiplier": 1.1,  # More sensitive silence detection
            "speech_multiplier": 1.2,   # More sensitive speech detection
            "end_multiplier": 1.15,     # More sensitive end detection
        },
        
        # Timing parameters for speech detection
        "timing": {
            "command_duration": 10.0,    # Maximum recording length
            "initial_silence": 1.0,      # Shorter wait for speech to start (was 3.0)
            "max_silence": 0.8,          # Shorter silence to end recording (was 2.0)
            "min_speech": 0.1,           # Keep short for quick commands
            "trailing_silence": 0.5,      # Shorter trailing silence (was 1.0)
            "inter_phrase_pause": 0.3,    # Shorter pause between phrases
        },
        
        # Frame-based detection parameters
        "frames": {
            "calibration": 30,           # Keep this value
            "silence_threshold": 5,       # Fewer silent frames needed (was 10)
            "trailing_frames": 3,         # Fewer trailing frames (was 5)
            "check_count": 4,            # Fewer check frames (was 8)
        },
        
        # Conversation management
        "conversation": {
            "timeout": 30,              # How long to maintain conversation context (seconds)
            "interrupt_delay": 0.2,      # Delay after interruption before new command
        },
        
        # Audio playback settings
        "playback": {
            "buffer_size": 4096,        # Size of audio playback buffer (larger = smoother)
            "channels": 1,              # Number of audio channels (1 = mono)
            "sample_width": 2,          # Bytes per sample (2 = 16-bit audio)
            "rate": 24000,              # Playback sample rate in Hz
        }
    }

    def __init__(self):
        """Initialize audio interface with configuration"""
        # Load config values into instance variables for easy access
        self.config = self.AUDIO_CONFIG
        
        # Core configuration
        self.settings = Settings()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.output_stream = None
        self.running = False
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Wake word detection
        self.porcupine = None
        
        # Audio processing parameters
        self.noise_floor = self.config["speech_detection"]["noise_floor"]
        self.silence_threshold = self.noise_floor * self.config["speech_detection"]["silence_multiplier"]
        self.speech_threshold = self.noise_floor * self.config["speech_detection"]["speech_multiplier"]
        
        # State tracking
        self.is_playing = False
        self.in_conversation = False
        self.last_interaction = 0
        
        # Playback control
        self.current_playback = None
        self.interrupt_event = asyncio.Event()
        self.wake_word_detected = asyncio.Event()
    
    async def initialize(self):
        """Initialize audio streams and wake word detection"""
        try:
            # Initialize Porcupine wake word detector first
            self.porcupine = pvporcupine.create(
                access_key=self.settings.PICOVOICE_ACCESS_KEY,
                keyword_paths=[str(self.settings.WAKE_WORD_MODEL_PATH)],
                sensitivities=[self.settings.WAKE_WORD_THRESHOLD]
            )
            
            # Display available audio devices for debugging
            self._list_audio_devices()
            
            # Get input/output device information
            input_device_info = self.p.get_device_info_by_index(self.settings.AUDIO_INPUT_DEVICE_INDEX)
            output_device_info = self.p.get_device_info_by_index(self.settings.AUDIO_OUTPUT_DEVICE_INDEX)
            
            print(f"\nInput device: {input_device_info['name']}")
            print(f"Output device: {output_device_info['name']}")
            
            # Calculate audio parameters
            input_sample_rate = int(input_device_info['defaultSampleRate'])
            print(f"Device sample rate: {input_sample_rate}")
            
            # Calculate frame sizes based on Porcupine requirements
            self.frame_length = self.porcupine.frame_length
            samples_per_frame = int(input_sample_rate / 16000 * self.frame_length)
            
            print(f"Frame length: {self.frame_length}")
            print(f"Samples per frame: {samples_per_frame}")
            
            # Initialize input stream for wake word detection and recording
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=input_sample_rate,
                input=True,
                frames_per_buffer=samples_per_frame,
                input_device_index=self.settings.AUDIO_INPUT_DEVICE_INDEX
            )
            
            # Initialize output stream for audio playback
            self.output_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=int(output_device_info['defaultSampleRate']),
                output=True,
                output_device_index=self.settings.AUDIO_OUTPUT_DEVICE_INDEX
            )
            
            # Store configuration for later use
            self.input_sample_rate = input_sample_rate
            self.samples_per_frame = samples_per_frame
            
            print(f"Using fixed noise floor value: {self.noise_floor}")
            print("Audio interface initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing audio interface: {e}")
            self.cleanup()
            return False
    
    async def _record_command(self, duration=None):
        """Record audio command with dynamic duration based on speech patterns"""
        if duration is None:
            duration = self.config["timing"]["command_duration"]
            
        print("\n=== Waiting for Command ===")
        print("▶ Listening for your request...")
        frames = []
        
        # More lenient thresholds
        silence_threshold = self.noise_floor * 1.1
        speech_threshold = self.noise_floor * 1.3
        end_of_speech_threshold = self.noise_floor * 1.2
        
        # Calculate frame counts
        max_silence_frames = int(self.input_sample_rate * self.config["timing"]["max_silence"] / self.frame_length)
        min_speech_frames = int(self.input_sample_rate * self.config["timing"]["min_speech"] / self.frame_length)
        max_frames = int(self.input_sample_rate * duration / self.frame_length)
        initial_silence_frames = int(self.input_sample_rate * self.config["timing"]["initial_silence"] / self.frame_length)
        
        # State tracking
        silence_frames = 0
        speech_frames = 0
        speech_detected = False
        consecutive_silence = 0
        initial_silence_count = 0
        
        try:
            last_speech_time = time.time()
            command_start_time = None
            
            for frame_count in range(max_frames):
                data = self.stream.read(self.frame_length, exception_on_overflow=False)
                frames.append(data)
                
                pcm_data = struct.unpack_from("h" * self.frame_length, data)
                audio_level = sum(abs(x) for x in pcm_data) / len(pcm_data)
                
                if audio_level > speech_threshold:
                    if not speech_detected:
                        print("✓ Speech detected, listening...")
                        command_start_time = time.time()
                    speech_detected = True
                    speech_frames += 1
                    silence_frames = 0
                    consecutive_silence = 0
                    last_speech_time = time.time()
                    print(f"Level: {'#' * int(audio_level / 75)}")
                else:
                    silence_frames += 1
                    consecutive_silence += 1
                    print(f"Level: {'#' * int(audio_level / 75)}")
                
                current_time = time.time()
                
                # Enhanced end detection
                if speech_detected and speech_frames > min_speech_frames:
                    # 1. Quick end after clear speech
                    if consecutive_silence > int(max_silence_frames * 0.5):  # End after half max silence
                        print("\n✓ Speech ended (clear ending)")
                        break
                    
                    # 2. End on falling volume
                    if (audio_level < end_of_speech_threshold and 
                        current_time - last_speech_time > self.config["timing"]["trailing_silence"]):
                        print("\n✓ Speech ended (volume drop)")
                        break
                    
                    # 3. End on long silence
                    if current_time - last_speech_time > self.config["timing"]["max_silence"]:
                        print("\n✓ Speech ended (silence)")
                        break
                
                # No initial speech detected - fail faster
                if not speech_detected and frame_count > initial_silence_frames:
                    print("\n✗ No speech detected")
                    return None
                
                await asyncio.sleep(0.001)
            
            if speech_detected and speech_frames > min_speech_frames:
                print("\n▶ Processing your request...")
                return b''.join(frames)
            else:
                print("\n✗ Insufficient speech detected")
                return None
            
        except Exception as e:
            print(f"\n✗ Error recording command: {e}")
            return None
    
    def _list_audio_devices(self):
        """Log available audio devices for debugging"""
        print("\nAvailable Audio Devices:")
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            print(f"Index {i}: {dev['name']}")
            print(f"  Max Input Channels: {dev['maxInputChannels']}")
            print(f"  Max Output Channels: {dev['maxOutputChannels']}")
            print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
    
    async def play_audio(self, audio_data: bytes):
        """Play audio with interrupt support"""
        try:
            self.is_playing = True
            # Create new playback task
            self.current_playback = asyncio.create_task(self._play_audio_interruptible(audio_data))
            await self.current_playback
            
        except asyncio.CancelledError:
            print("Audio playback cancelled")
        except Exception as e:
            print(f"Error in audio playback: {e}")
        finally:
            self.is_playing = False
    
    async def _play_audio_interruptible(self, audio_data: bytes):
        """Play audio with support for interruption"""
        try:
            # Initialize output stream if needed
            if not self.output_stream:
                self.output_stream = self._create_output_stream()
            
            # Convert audio data to wave format
            wav = wave.open(io.BytesIO(audio_data))
            
            # Get audio properties
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            print(f"WAV format: {channels} channels, {width} bytes per sample, {rate} Hz")
            
            # Create a new output stream matching the WAV format
            if self.output_stream:
                self.output_stream.close()
            
            # Use a larger buffer size for smoother playback
            buffer_size = 4096  # Increased from 1024
            
            self.output_stream = self.p.open(
                format=self.p.get_format_from_width(width),
                channels=channels,
                rate=rate,
                output=True,
                output_device_index=self.settings.AUDIO_OUTPUT_DEVICE_INDEX,
                frames_per_buffer=buffer_size,
                stream_callback=None  # Use blocking mode for better control
            )
            
            # Read the entire audio file into memory
            audio_data = wav.readframes(wav.getnframes())
            
            # Split into chunks
            chunks = [audio_data[i:i + buffer_size] for i in range(0, len(audio_data), buffer_size)]
            
            # Play chunks with proper timing
            for chunk in chunks:
                # Check for interruption
                if self.interrupt_event.is_set():
                    print("Playback interrupted")
                    break
                
                self.output_stream.write(chunk)
                
                # No sleep needed as write is blocking
            
            # Ensure all data is played before closing
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            
        except Exception as e:
            print(f"Error in audio playback: {e}")
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
    
    async def start_monitoring(self, wake_word_callback: Callable):
        """Start monitoring for wake word with interrupt support"""
        try:
            self.running = True
            self.wake_word_callback = wake_word_callback
            
            # Create and store the monitoring task
            self.monitor_task = asyncio.create_task(self._monitor_wake_word())
            
            # Keep track of the task but don't await it
            asyncio.create_task(self._keep_monitoring())
            
            return True
            
        except Exception as e:
            print(f"Error in monitoring: {e}")
            return False
    
    async def _monitor_wake_word(self):
        """Monitor for wake word with CPU optimization"""
        while self.running:
            try:
                # Get next audio frame with proper await
                pcm = await self._get_next_audio_frame()  # Add await here
                
                if pcm is None:
                    continue
                
                # Process wake word detection in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.porcupine.process,
                    pcm
                )
                
                if result >= 0:
                    print("\nWake word detected!")
                    
                    # If we're currently playing audio, interrupt it
                    if self.is_playing:
                        self.interrupt_event.set()
                        # Wait a moment for playback to stop
                        await asyncio.sleep(0.2)
                    
                    command_audio = await self._collect_command_audio()
                    if command_audio:
                        await self.wake_word_callback(command_audio)
                        print("\nCommand processing complete")
                    
                # Add small sleep to prevent CPU hogging
                await asyncio.sleep(0.01)
                    
            except Exception as e:
                print(f"Error in wake word monitoring: {e}")
                await asyncio.sleep(0.1)
    
    async def _get_next_audio_frame(self):
        """Get the next audio frame from the input stream"""
        try:
            # Run the blocking audio read in the executor
            pcm = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.stream.read,
                self.samples_per_frame,
                False  # exception_on_overflow=False
            )
            
            # Convert bytes to numpy array for resampling
            audio_array = np.frombuffer(pcm, dtype=np.int16)
            
            # Resample to 16kHz if needed (Porcupine requirement)
            if self.input_sample_rate != 16000:
                # Calculate resampling ratio
                ratio = 16000 / self.input_sample_rate
                target_length = int(len(audio_array) * ratio)
                
                # Use numpy for efficient resampling
                indices = np.linspace(0, len(audio_array)-1, target_length).astype(int)
                resampled = audio_array[indices]
                
                # Ensure we have exactly the frame length Porcupine expects
                if len(resampled) > self.frame_length:
                    resampled = resampled[:self.frame_length]
                elif len(resampled) < self.frame_length:
                    resampled = np.pad(resampled, (0, self.frame_length - len(resampled)))
                
                return resampled.tolist()
            else:
                # If already at 16kHz, just ensure correct frame length
                if len(audio_array) > self.frame_length:
                    return audio_array[:self.frame_length].tolist()
                elif len(audio_array) < self.frame_length:
                    return np.pad(audio_array, (0, self.frame_length - len(audio_array))).tolist()
                return audio_array.tolist()
            
        except Exception as e:
            print(f"Error reading audio frame: {e}")
            return None
    
    async def _collect_command_audio(self):
        """Collect audio for command recording"""
        try:
            command_audio = await self._record_command()
            if command_audio:
                return command_audio
            else:
                return None
        except Exception as e:
            print(f"Error collecting command audio: {e}")
            return None
    
    def _resample_audio(self, audio_data):
        """Resample audio to Porcupine's required format"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Simple downsampling
        target_length = self.porcupine.frame_length
        current_length = len(audio_array)
        
        if current_length == 0:
            return [0] * target_length
        
        # Use numpy for efficient resampling
        indices = np.linspace(0, current_length-1, target_length).astype(int)
        return audio_array[indices].tolist()
    
    def cleanup(self):
        """Clean up audio resources"""
        self.running = False
        
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error closing input stream: {e}")
            
        if hasattr(self, 'output_stream') and self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception as e:
                print(f"Error closing output stream: {e}")
            
        if hasattr(self, 'p') and self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            
        if hasattr(self, 'porcupine') and self.porcupine:
            try:
                self.porcupine.delete()
            except Exception as e:
                print(f"Error cleaning up Porcupine: {e}")
            
        if hasattr(self, 'executor') and self.executor:
            try:
                self.executor.shutdown()
            except Exception as e:
                print(f"Error shutting down executor: {e}")
    
    def _create_output_stream(self):
        """Create audio output stream"""
        return self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,  # TTS output rate
            output=True,
            output_device_index=self.settings.AUDIO_OUTPUT_DEVICE_INDEX,
            frames_per_buffer=4096  # Larger buffer for smoother playback
        )
    
    async def _keep_monitoring(self):
        """Keep monitoring task alive"""
        try:
            while self.running:
                if self.monitor_task.done():
                    # Restart monitoring if it stopped
                    print("Restarting wake word monitoring...")
                    self.monitor_task = asyncio.create_task(self._monitor_wake_word())
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in monitoring loop: {e}")