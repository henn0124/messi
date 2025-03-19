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
                    channels=1,
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

    async def play(self, audio_data: bytes) -> bool:
        """Play audio from bytes."""
        if not self.output_stream:
            return False
            
        try:
            # Read WAV data
            with wave.open(io.BytesIO(audio_data), 'rb') as wav:
                # Get audio parameters
                input_rate = wav.getframerate()
                input_channels = wav.getnchannels()
                output_rate = self.output_stream.get_sample_rate()
                
                print(f"\n🔊 Playing audio:")
                print(f"  Input Rate: {input_rate} Hz")
                print(f"  Output Rate: {output_rate} Hz")
                print(f"  Channels: {input_channels}")
                
                # Convert sample rate if needed
                if input_rate != output_rate:
                    print(f"Converting sample rate from {input_rate}Hz to {output_rate}Hz")
                    # Convert to AudioSegment for resampling
                    audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
                    audio_segment = audio_segment.set_frame_rate(output_rate)
                    # Convert back to WAV
                    wav_buffer = io.BytesIO()
                    audio_segment.export(wav_buffer, format="wav")
                    audio_data = wav_buffer.getvalue()
                
                # Read and play audio data
                chunk_size = self.porcupine.frame_length
                data = wav.readframes(chunk_size)
                while data:
                    self.output_stream.write(data)
                    data = wav.readframes(chunk_size)
                    
            return True
            
        except Exception as e:
            print(f"❌ Error playing audio: {str(e)}")
            return False

    async def stop(self):
        """Stop audio processing"""
        self.running = False
        if self.input_stream:
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.close()

    async def test_audio_loop(self, duration: float = 3.0) -> bool:
        """Test audio system by playing a tone and recording user speech."""
        try:
            print("\n🔊 Starting audio system test...")
            
            # First test output
            print("\n🎵 Testing audio output...")
            sample_rate = self.porcupine.sample_rate
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 1000 * t)
            tone = (tone * 32767).astype(np.int16)
            tone_bytes = tone.tobytes()
            
            print(f"  Sample Rate: {sample_rate} Hz")
            print(f"  Duration: {duration} seconds")
            
            try:
                print("\nPlaying test tone...")
                self.output_stream.write(tone_bytes)
                print("✓ Test tone played")
                print("\nDid you hear the test tone? (y/n)")
                response = input().lower()
                if response != 'y':
                    print("\n❌ Audio output test failed")
                    print("\nTroubleshooting output:")
                    print("1. Check if speakers/headphones are connected")
                    print("2. Verify system volume is not muted")
                    print("3. Try running 'alsamixer' to check output levels")
                    print("4. Check if output device is correct in config.yml")
                    return False
                print("✓ Audio output test passed")
                
            except Exception as e:
                print(f"❌ Failed to play test tone: {str(e)}")
                return False
            
            # Now test input with feedback loop
            print("\n🎤 Testing audio input with feedback loop...")
            print("Playing a test tone and recording it through the microphone...")
            chunk_size = self.porcupine.frame_length
            num_chunks = int((sample_rate * duration) / chunk_size)
            
            try:
                # Start recording
                print("\nStarting recording...")
                frames = []
                
                # Play tone and record simultaneously
                self.output_stream.write(tone_bytes)
                
                # Record for the duration
                for _ in range(num_chunks):
                    try:
                        data = self.input_stream.read(chunk_size, exception_on_overflow=False)
                        if not data:
                            print("❌ Empty audio chunk received")
                            return False
                        frames.append(data)
                    except Exception as e:
                        print(f"❌ Error reading audio chunk: {str(e)}")
                        return False
                
                if not frames:
                    print("❌ No audio recorded")
                    print("\nTroubleshooting input:")
                    print("1. Check if microphone is properly connected")
                    print("2. Verify microphone isn't muted in system settings")
                    print("3. Run 'alsamixer' and check capture levels")
                    print("4. Try a different USB port")
                    return False
                
                # Print recording details
                total_bytes = sum(len(frame) for frame in frames)
                print(f"\nRecording Details:")
                print(f"  Total Frames: {len(frames)}")
                print(f"  Total Bytes: {total_bytes}")
                
                # Analyze recorded audio
                recorded_audio = np.frombuffer(b''.join(frames), dtype=np.int16)
                
                # Add data validation
                if len(recorded_audio) == 0:
                    print("❌ No audio data recorded")
                    return False
                    
                # Check for invalid values
                if np.any(np.isnan(recorded_audio)):
                    print("❌ Invalid audio data detected (NaN values)")
                    return False
                
                # Calculate RMS and peak levels with error handling
                try:
                    rms_level = np.sqrt(np.mean(recorded_audio.astype(np.float64)**2))
                    peak_level = np.max(np.abs(recorded_audio))
                except Exception as e:
                    print(f"❌ Error calculating audio levels: {str(e)}")
                    return False
                
                print("\n📊 Recording Analysis:")
                print(f"  RMS Level: {rms_level:.0f}")
                print(f"  Peak Level: {peak_level:.0f}")
                
                if rms_level < 100:
                    print("\n⚠️ Warning: Very low audio levels detected")
                    print("Troubleshooting tips:")
                    print("1. Check if microphone is properly connected")
                    print("2. Verify microphone isn't muted in system settings")
                    print("3. Run 'alsamixer' and check capture levels")
                    print("4. Try a different USB port")
                    return False
                
                print("\n✅ Audio input test completed successfully")
                print("✓ Microphone is detecting audio levels")
                
            except Exception as e:
                print(f"❌ Failed to record audio: {str(e)}")
                return False
            
            print("\n✨ Audio system test completed")
            return True
            
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