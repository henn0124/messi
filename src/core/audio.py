"""
Audio Interface for Messi Assistant
---------------------------------
Handles audio I/O operations using ALSA.
"""

import alsaaudio
import numpy as np
import wave
import io
import asyncio
from typing import Optional, Dict, Callable
import struct
from pydub import AudioSegment
import pyaudio
import time
from pvporcupine import create as create_porcupine
import yaml

class AudioManager:
    def __init__(self, config: Dict):
        """Initialize audio interface with configuration"""
        self.config = config
        self.pyaudio = None
        self.input_stream = None
        self.output_stream = None
        self.running = False
        self.wake_word_detected = False
        self.porcupine = None
        
        # Audio settings
        self.input_config = config["audio"]["input"]
        self.output_config = config["audio"]["output"]
        self.stream = None
        self.wake_word_config = config["wake_word"]

    async def initialize(self) -> bool:
        """Initialize PyAudio and open streams."""
        try:
            print("\n🎤 Initializing audio system...")
            self.pyaudio = pyaudio.PyAudio()
            
            # List all devices with detailed information
            print("\n📋 Available audio devices:")
            info = {}
            for i in range(self.pyaudio.get_device_count()):
                try:
                    device_info = self.pyaudio.get_device_info_by_index(i)
                    info[i] = device_info
                    print(f"\nDevice {i}: {device_info['name']}")
                    print(f"  Host API: {self.pyaudio.get_host_api_info_by_index(device_info['hostApi'])['name']}")
                    print(f"  Input Channels: {device_info['maxInputChannels']}")
                    print(f"  Output Channels: {device_info['maxOutputChannels']}")
                    print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
                    print(f"  Is Default Input: {self.pyaudio.get_default_input_device_info()['index'] == i}")
                    print(f"  Is Default Output: {self.pyaudio.get_default_output_device_info()['index'] == i}")
                except Exception as e:
                    print(f"  Error getting device info: {str(e)}")
            
            # Initialize Porcupine wake word engine
            try:
                print("\n🎯 Initializing wake word detector...")
                self.porcupine = create_porcupine(
                    access_key=self.wake_word_config["access_key"],
                    keyword_paths=[self.wake_word_config["model_path"]],
                    sensitivities=[self.wake_word_config["sensitivity"]]
                )
                print("✓ Wake word detector initialized")
                print(f"  Sample Rate: {self.porcupine.sample_rate} Hz")
                print(f"  Frame Length: {self.porcupine.frame_length} samples")
            except Exception as e:
                print(f"❌ Failed to initialize wake word detector: {str(e)}")
                raise e

            # Parse input device specification
            print("\n🎙️ Setting up input device...")
            try:
                if self.input_config["device"] == "default":
                    input_device_index = self.pyaudio.get_default_input_device_info()["index"]
                    print(f"Using default input device (index {input_device_index})")
                else:
                    # Parse hw:X,Y format
                    parts = self.input_config["device"].split(":")
                    if len(parts) == 2 and parts[0] == "hw":
                        card = parts[1].split(",")[0]
                        # Find the device index by name
                        input_device_index = None
                        for i, device_info in info.items():
                            if "TONOR TM20" in device_info["name"]:
                                input_device_index = i
                                break
                        if input_device_index is None:
                            raise ValueError(f"TONOR TM20 microphone not found")
                        print(f"Using TONOR TM20 microphone (index {input_device_index})")
                    else:
                        raise ValueError(f"Invalid device format: {self.input_config['device']}")
                
                if input_device_index not in info:
                    raise ValueError(f"Input device index {input_device_index} not found")
                    
                print("\nSelected input device details:")
                print(f"  Name: {info[input_device_index]['name']}")
                print(f"  Channels: {info[input_device_index]['maxInputChannels']}")
                print(f"  Sample Rate: {info[input_device_index]['defaultSampleRate']}")
            except Exception as e:
                print(f"❌ Error setting up input device: {str(e)}")
                raise e

            # Open input stream
            print("\n🔊 Opening audio streams...")
            try:
                self.input_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.porcupine.sample_rate,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=self.porcupine.frame_length,
                    stream_callback=None,
                    start=False
                )
                
                # Configure buffer size
                if hasattr(self.input_stream, '_frames_per_buffer'):
                    self.input_stream._frames_per_buffer = self.input_config["period_size"]
                
                # Start the stream
                self.input_stream.start_stream()
                print("✓ Input stream opened successfully")
                
            except Exception as e:
                print(f"❌ Failed to open input stream: {str(e)}")
                raise e

            # Open output stream
            try:
                print("\n🔊 Setting up output device...")
                output_device_index = None
                
                # First try to find USB audio device
                for i, device_info in info.items():
                    if device_info["maxOutputChannels"] > 0:  # Device has output capability
                        print(f"Found output device: {device_info['name']}")
                        if "USB" in device_info["name"]:
                            output_device_index = i
                            print(f"Selected USB output device: {device_info['name']}")
                            break
                
                # If no USB device found, try default
                if output_device_index is None:
                    output_device_index = self.pyaudio.get_default_output_device_info()["index"]
                    print(f"Using default output device (index {output_device_index})")
                
                print(f"\nSelected output device details:")
                print(f"  Name: {info[output_device_index]['name']}")
                print(f"  Channels: {info[output_device_index]['maxOutputChannels']}")
                print(f"  Sample Rate: {info[output_device_index]['defaultSampleRate']}")
                
                # Use device's native sample rate
                output_sample_rate = int(info[output_device_index]['defaultSampleRate'])
                print(f"Using output sample rate: {output_sample_rate} Hz")
                
                self.output_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=2,  # Use stereo output
                    rate=output_sample_rate,  # Use device's native sample rate
                    output=True,
                    output_device_index=output_device_index,
                    frames_per_buffer=self.output_config["period_size"]
                )
                print("✓ Output stream opened successfully")
                
            except Exception as e:
                print(f"❌ Failed to open output stream: {str(e)}")
                raise e

            print("\n✨ Audio system initialized successfully")
            return True
            
        except Exception as e:
            print(f"\n❌ Audio initialization failed: {str(e)}")
            if self.pyaudio:
                self.pyaudio.terminate()
            return False

    async def cleanup(self):
        """Clean up audio resources."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
        if self.porcupine:
            self.porcupine.delete()

    async def start_processing(self):
        """Start audio processing"""
        self.running = True
        print("\n👂 Listening for 'Hey Messy'...")

    async def wait_for_wake_word(self):
        """Wait for wake word to be detected."""
        print("\nListening for wake word 'Hey Messy'...")
        print("Audio settings:")
        print(f"  Sample rate: {self.porcupine.sample_rate} Hz")
        print(f"  Frame length: {self.porcupine.frame_length} samples")
        print(f"  Input device: {self.input_config['device']}")
        
        silence_count = 0
        last_level_print = time.time()
        levels_history = []
        
        while True:
            try:
                pcm = self.input_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm_np = np.frombuffer(pcm, dtype=np.int16)
                
                # Calculate RMS and peak levels
                audio_level_rms = np.sqrt(np.mean(pcm_np**2))
                peak_level = np.max(np.abs(pcm_np))
                levels_history.append(audio_level_rms)
                
                # Keep only last 10 seconds of history
                if len(levels_history) > 160:  # 16 frames per second * 10 seconds
                    levels_history.pop(0)
                
                current_time = time.time()
                if current_time - last_level_print >= 1.0:
                    avg_level = np.mean(levels_history)
                    max_level = np.max(levels_history)
                    
                    print(f"\n📊 Audio Levels:")
                    print(f"  Current RMS: {audio_level_rms:.0f}")
                    print(f"  Peak: {peak_level:.0f}")
                    print(f"  Average (10s): {avg_level:.0f}")
                    print(f"  Max (10s): {max_level:.0f}")
                    
                    if avg_level < 50:  # Very quiet
                        silence_count += 1
                        if silence_count >= 3:
                            print("\n⚠️ Audio levels are very low. Troubleshooting tips:")
                            print("1. Try speaking louder or moving closer to the mic")
                            print("2. Check system volume settings:")
                            print("   - Run 'alsamixer' to check/adjust capture levels")
                            print("   - Verify 'Input Device' volume in system settings")
                            print("3. Try unplugging and reconnecting the microphone")
                            silence_count = 0
                    else:
                        silence_count = 0
                    
                    last_level_print = current_time
                
                # Process for wake word
                keyword_index = self.porcupine.process(pcm_np)
                if keyword_index >= 0:
                    print(f"\n🎯 Wake word detected! (Level: {audio_level_rms:.0f})")
                    return
                    
            except Exception as e:
                print(f"❌ Error processing audio: {str(e)}")
                continue

    async def record(self, duration: float) -> Optional[bytes]:
        """Record audio for specified duration in seconds."""
        if not self.input_stream:
            return None
            
        try:
            frames = []
            num_frames = int(16000 * duration)
            
            for _ in range(0, num_frames, 1024):
                data = self.input_stream.read(1024, exception_on_overflow=False)
                frames.append(data)
                
            return b''.join(frames)
            
        except Exception as e:
            print(f"❌ Error recording audio: {str(e)}")
            return None

    async def record_command(self, max_duration: float = 5.0) -> Optional[bytes]:
        """Record audio command after wake word, stopping on silence."""
        if not self.input_stream:
            return None
            
        try:
            frames = []
            silence_threshold = self.input_config["silence_threshold"]
            silence_duration = self.input_config["silence_duration"]
            chunk_size = self.porcupine.frame_length
            
            print("\nRecording... (speak your command)")
            
            # Record until silence or max duration
            start_time = time.time()
            last_sound = start_time
            
            while (time.time() - start_time) < max_duration:
                data = self.input_stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Check for silence
                audio_array = np.frombuffer(data, dtype=np.int16)
                if np.abs(audio_array).mean() < silence_threshold:
                    if time.time() - last_sound > silence_duration:
                        break
                else:
                    last_sound = time.time()
                    
            print("Recording complete")
                    
            # Convert frames to WAV format
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self.porcupine.sample_rate)
                wav.writeframes(b''.join(frames))
                
            return wav_buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Error recording command: {str(e)}")
            return None

    def play(self, audio_data: bytes) -> None:
        """Play audio data through the output device."""
        try:
            # Convert audio data to AudioSegment for processing
            audio = AudioSegment.from_wav(io.BytesIO(audio_data))
            
            # Convert mono to stereo if needed
            if audio.channels == 1:
                audio = audio.set_channels(2)
            
            # Convert sample rate if needed
            if audio.frame_rate != self.output_stream.get_sample_rate():
                audio = audio.set_frame_rate(self.output_stream.get_sample_rate())
            
            # Export back to WAV format
            wav_data = io.BytesIO()
            audio.export(wav_data, format="wav")
            wav_data.seek(0)
            
            # Read the WAV data
            wav = wave.open(wav_data, 'rb')
            audio_data = wav.readframes(wav.getnframes())
            wav.close()
            
            # Play the audio
            self.output_stream.write(audio_data)
            
        except Exception as e:
            print(f"❌ Error playing audio: {str(e)}")
            raise

    async def stop(self):
        """Stop audio processing"""
        self.running = False
        if self.input_stream:
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.close()

    async def test_audio_loop(self, duration: float = 3.0) -> bool:
        """Test audio system by creating a feedback loop between speaker and microphone."""
        try:
            print("\n🔊 Starting audio system test...")
            
            # Generate test tone
            print("\n🎵 Generating test tone...")
            sample_rate = self.porcupine.sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 1000 * t)
            tone = (tone * 32767 * 0.8).astype(np.int16)  # Increased volume to 80% of max
            tone_bytes = tone.tobytes()
            
            print(f"  Sample Rate: {sample_rate} Hz")
            print(f"  Duration: {duration} seconds")
            print(f"  Buffer Size: {len(tone_bytes)} bytes")
            
            # First record some silence as baseline
            print("\n📊 Recording silence baseline...")
            silence_frames = await self._record_chunks(self.porcupine.frame_length, 10)
            silence_audio = np.frombuffer(b''.join(silence_frames), dtype=np.int16)
            silence_rms = np.sqrt(np.mean(silence_audio**2))
            print(f"  Silence RMS: {silence_rms:.0f}")
            
            # Play tone and record simultaneously
            print("\n🎵 Playing test tone and recording...")
            print("Please position the microphone near the speaker...")
            
            # Start recording
            frames = []
            chunk_size = self.porcupine.frame_length
            num_chunks = int((sample_rate * duration) / chunk_size)
            
            # Play tone and record in chunks
            for _ in range(num_chunks):
                # Record chunk
                data = self.input_stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
                # Play corresponding part of tone
                start_idx = len(frames) * chunk_size * 2  # 2 bytes per sample
                chunk = tone_bytes[start_idx:start_idx + chunk_size * 2]
                if chunk:
                    self.output_stream.write(chunk)
            
            # Analyze recorded audio
            recorded_audio = np.frombuffer(b''.join(frames), dtype=np.int16)
            rms_level = np.sqrt(np.mean(recorded_audio**2))
            peak_level = np.max(np.abs(recorded_audio))
            
            print("\n📊 Recording Analysis:")
            print(f"  RMS Level: {rms_level:.0f}")
            print(f"  Peak Level: {peak_level:.0f}")
            print(f"  Signal-to-Noise Ratio: {20 * np.log10(rms_level/silence_rms):.1f} dB")
            
            # Check if we recorded the tone
            if rms_level > silence_rms * 2:  # Signal should be at least 6dB above noise
                print("\n✅ Microphone test passed!")
                print("  - Successfully recorded the test tone")
                print("  - Good signal-to-noise ratio")
                return True
            else:
                print("\n❌ Microphone test failed")
                print("\nTroubleshooting tips:")
                print("1. Check if microphone is properly connected")
                print("2. Verify microphone isn't muted in system settings")
                print("3. Run 'alsamixer' and check capture levels")
                print("4. Try adjusting microphone position relative to speaker")
                print("5. Check if input device is correct in config.yml")
                return False
            
        except Exception as e:
            print(f"\n❌ Audio test failed: {str(e)}")
            return False
            
    async def _record_chunks(self, chunk_size: int, num_chunks: int) -> list:
        """Helper method to record a specific number of chunks."""
        frames = []
        try:
            for _ in range(num_chunks):
                data = self.input_stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
            return frames
        except Exception as e:
            print(f"❌ Error during recording: {str(e)}")
            return []