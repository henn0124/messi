"""
Audio Interface for Messi Assistant
---------------------------------
Handles audio I/O operations using ALSA.
"""

import os
import pyaudio
import numpy as np
import wave
import io
import asyncio
from typing import Optional, Dict, Callable
import struct
from pydub import AudioSegment
import time
from pvporcupine import create as create_porcupine
import yaml
import sounddevice as sd
import queue
import threading
import logging
from pathlib import Path
from core.wake_word import WakeWordDetector

class AudioManager:
    def __init__(self, config: Dict):
        """Initialize audio manager with configuration"""
        self.config = config
        self.pyaudio = None
        self.input_device = None
        self.output_device = None
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.recording_thread = None
        self.wake_word_detector = None
        self.silence_threshold = config.get('silence_threshold', 152)
        self.sample_rate = config.get('sample_rate', 16000)
        self.period_size = config.get('period_size', 512)
        self.input_device_name = config.get('input_device', 'hw:4,0')
        self.output_device_name = config.get('output_device', 'hw:0,0')
        self.running = False
        self.wake_word_detected = False
        self.porcupine = None
        
        # Audio settings
        self.input_config = config["audio"]["input"]
        self.output_config = config["audio"]["output"]
        self.stream = None
        self.wake_word_config = config["wake_word"]

    def _find_device_index(self, device_name: str, is_input: bool = True) -> int:
        """Find device index by name"""
        for i in range(self.pyaudio.get_device_count()):
            try:
                device_info = self.pyaudio.get_device_info_by_index(i)
                if device_name in device_info["name"]:
                    if is_input and device_info["maxInputChannels"] > 0:
                        return i
                    elif not is_input and device_info["maxOutputChannels"] > 0:
                        return i
            except Exception:
                continue
        return -1

    async def initialize(self) -> bool:
        """Initialize audio system"""
        try:
            print("\n🎤 Initializing audio system...")
            
            # Force PyAudio to reinitialize
            if self.pyaudio:
                self.pyaudio.terminate()
            self.pyaudio = pyaudio.PyAudio()
            
            # List available devices
            print("\n📋 Available devices:")
            for i in range(self.pyaudio.get_device_count()):
                try:
                    device_info = self.pyaudio.get_device_info_by_index(i)
                    print(f"\nDevice {i}: {device_info['name']}")
                    print(f"  Host API: {self.pyaudio.get_host_api_info_by_index(device_info['hostApi'])['name']}")
                    print(f"  Input Channels: {device_info['maxInputChannels']}")
                    print(f"  Output Channels: {device_info['maxOutputChannels']}")
                    print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
                except Exception as e:
                    print(f"  Error getting device info: {str(e)}")

            # Find input device index
            print("\n🎙️ Finding input device...")
            input_device_index = self._find_device_index("TONOR TM20", is_input=True)
            if input_device_index == -1:
                print("❌ TONOR TM20 microphone not found")
                print("\nTroubleshooting tips:")
                print("1. Check if microphone is properly connected")
                print("2. Try unplugging and reconnecting the microphone")
                print("3. Run 'arecord -l' to verify device is detected")
                print("4. Check system logs with 'dmesg | grep -i audio'")
                return False

            # Find output device index
            print("\n🔊 Finding output device...")
            output_device_index = self._find_device_index("USB2.0 Device", is_input=False)
            if output_device_index == -1:
                print("❌ USB2.0 Device speaker not found")
                print("\nTroubleshooting tips:")
                print("1. Check if speaker is properly connected")
                print("2. Try unplugging and reconnecting the speaker")
                print("3. Run 'aplay -l' to verify device is detected")
                print("4. Check system logs with 'dmesg | grep -i audio'")
                return False

            # Open input stream
            try:
                print("\n🎙️ Opening input stream...")
                self.input_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.input_config["sample_rate"],
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=self.input_config["period_size"]
                )
                print("✓ Input stream opened successfully")
            except Exception as e:
                print(f"❌ Failed to open input stream: {str(e)}")
                return False

            # Open output stream
            try:
                print("\n🔊 Opening output stream...")
                self.output_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.output_config["sample_rate"],
                    output=True,
                    output_device_index=output_device_index,
                    frames_per_buffer=self.output_config["period_size"]
                )
                print("✓ Output stream opened successfully")
            except Exception as e:
                print(f"❌ Failed to open output stream: {str(e)}")
                return False

            print("\n✨ Audio system initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Audio initialization failed: {str(e)}")
            if self.pyaudio:
                self.pyaudio.terminate()
            return False

    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """Start recording audio"""
        try:
            self.is_running = True
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                args=(callback,)
            )
            self.recording_thread.start()
            return True
        except Exception as e:
            print(f"❌ Failed to start recording: {str(e)}")
            return False

    def stop_recording(self) -> None:
        """Stop recording audio"""
        self.is_running = False
        if self.recording_thread:
            self.recording_thread.join()

    def _recording_loop(self, callback: Optional[Callable] = None) -> None:
        """Main recording loop"""
        try:
            while self.is_running:
                try:
                    data = self.input_stream.read(self.period_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    if callback:
                        callback(audio_data)
                except Exception as e:
                    print(f"❌ Error reading audio data: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"❌ Error in recording loop: {str(e)}")

    def play_audio(self, audio_data: np.ndarray) -> bool:
        """Play audio data"""
        try:
            self.output_stream.write(audio_data.tobytes())
            return True
        except Exception as e:
            print(f"❌ Failed to play audio: {str(e)}")
            return False

    async def cleanup(self) -> None:
        """Clean up audio resources"""
        self.stop_recording()
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
        if self.wake_word_detector:
            self.wake_word_detector.stop()

    async def wait_for_wake_word(self) -> None:
        """Wait for wake word to be detected"""
        if not self.wake_word_detector:
            print("❌ Wake word detector not initialized")
            return

        def on_wake_word():
            print("🎯 Wake word detected!")

        self.wake_word_detector.start(on_wake_word)
        while True:
            await asyncio.sleep(0.1)
            if not self.is_running:
                break

    def test_audio_loop(self) -> bool:
        """Test audio input and output"""
        try:
            print("\n🔍 Running audio loopback test...")
            
            # Record test audio
            print("Recording test audio...")
            audio_data = []
            self.start_recording(lambda data: audio_data.append(data))
            time.sleep(2)  # Record for 2 seconds
            self.stop_recording()
            
            if not audio_data:
                print("❌ No audio data recorded")
                return False
                
            # Concatenate audio data
            audio_data = np.concatenate(audio_data)
            
            # Calculate audio levels
            rms_level = np.sqrt(np.mean(audio_data**2))
            peak_level = np.max(np.abs(audio_data))
            
            print(f"\n📊 Audio Analysis:")
            print(f"  RMS Level: {rms_level:.2f}")
            print(f"  Peak Level: {peak_level:.2f}")
            
            if rms_level < 100 or peak_level < 100:
                print("❌ Audio levels too low")
                return False
                
            print("\n🎙️ Audio test passed!")
            return True
                
            except Exception as e:
            print(f"❌ Audio test failed: {str(e)}")
            return False

    async def start_processing(self):
        """Start audio processing"""
        self.running = True
        print("\n👂 Listening for 'Hey Messy'...")

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
            chunk_size = self.period_size
            
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
                wav.setframerate(int(self.input_device['defaultSampleRate']))
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
                output_rate = int(self.output_device['defaultSampleRate'])
                
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
                chunk_size = self.period_size
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
            sample_rate = int(self.input_device['defaultSampleRate'])
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
            chunk_size = self.period_size
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