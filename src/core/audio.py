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

class AudioInterface:
    def __init__(self, config: Dict):
        """Initialize audio interface with configuration"""
        self.config = config
        self.input_stream = None
        self.output_stream = None
        self.running = False
        self.wake_word_detected = False
        
        # Audio settings
        self.input_config = config["input"]
        self.output_config = config["output"]

    async def initialize(self) -> bool:
        """Initialize audio devices"""
        try:
            print("\nInitializing audio...")
            
            # Initialize input stream
            self.input_stream = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NORMAL,
                device=self.input_config["device"],
                format=alsaaudio.PCM_FORMAT_S16_LE,
                channels=self.input_config["channels"],
                rate=self.input_config["rate"],
                periodsize=self.input_config["period_size"]
            )
            
            # Initialize output stream
            self.output_stream = alsaaudio.PCM(
                type=alsaaudio.PCM_PLAYBACK,
                mode=alsaaudio.PCM_NORMAL,
                device=self.output_config["device"],
                format=alsaaudio.PCM_FORMAT_S16_LE,
                channels=self.output_config["channels"],
                rate=self.output_config["rate"],
                periodsize=self.output_config["period_size"]
            )
            
            # Set buffer size using setchannels
            self.output_stream.setchannels(self.output_config["channels"])
            self.output_stream.setperiodsize(self.output_config["buffer_size"])
            
            print("✓ Audio initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            return False

    async def start_processing(self):
        """Start audio processing"""
        self.running = True
        print("\n👂 Listening for 'Hey Messy'...")

    async def wait_for_wake_word(self, wake_word_detector) -> bool:
        """Wait for wake word detection"""
        try:
            # Get required frame length from wake word detector
            frame_length = wake_word_detector.frame_length
            bytes_per_frame = frame_length * 2  # 16-bit audio = 2 bytes per sample
            
            # Calculate how many ALSA periods we need to read
            periods_per_frame = bytes_per_frame // (self.input_config["period_size"] * 2)
            if periods_per_frame < 1:
                periods_per_frame = 1
            
            # Read and process audio
            buffer = b""
            while self.running and not self.wake_word_detected:
                # Read from input stream
                length, data = self.input_stream.read()
                if length > 0:
                    buffer += data
                    
                    # Process complete frames
                    while len(buffer) >= bytes_per_frame:
                        # Extract frame
                        frame = buffer[:bytes_per_frame]
                        buffer = buffer[bytes_per_frame:]
                        
                        # Process with wake word detector
                        if wake_word_detector.process_frame(frame):
                            print("\n🎤 Wake word detected!")
                            self.wake_word_detected = True
                            return True
            
            return False
            
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False

    async def record_command(self) -> Optional[bytes]:
        """Record audio command until silence"""
        try:
            print("\nRecording... (speak your command)")
            buffer = []
            silence_count = 0
            silence_threshold = self.input_config["silence_threshold"]
            silence_duration = self.input_config["silence_duration"]
            frames_for_silence = int(silence_duration * self.input_config["rate"] / self.input_config["period_size"])
            
            while silence_count < frames_for_silence:
                length, data = self.input_stream.read()
                if length > 0:
                    audio = np.frombuffer(data, dtype=np.int16)
                    volume = np.abs(audio).mean()
                    
                    if volume < silence_threshold:
                        silence_count += 1
                    else:
                        silence_count = 0
                        
                    buffer.append(data)
            
            print("Recording complete")
            self.wake_word_detected = False  # Reset for next detection
            
            # Convert buffer to WAV format
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(self.input_config["channels"])
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.input_config["rate"])
                wf.writeframes(b"".join(buffer))
            
            return wav_buffer.getvalue()
            
        except Exception as e:
            print(f"Error recording command: {e}")
            return None

    async def play(self, audio_data: bytes) -> bool:
        """Play audio data"""
        try:
            print("\nPlaying audio...")
            
            # Load MP3 data
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            
            # Convert to WAV format with matching parameters
            audio = audio.set_frame_rate(self.output_config["rate"])
            audio = audio.set_channels(self.output_config["channels"])
            audio = audio.set_sample_width(2)  # 16-bit audio
            
            # Export to WAV in memory
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format='wav')
            wav_buffer.seek(0)
            
            # Play the WAV data
            with wave.open(wav_buffer, 'rb') as wf:
                # Get WAV file parameters
                channels = wf.getnchannels()
                rate = wf.getframerate()
                width = wf.getsampwidth()
                
                print(f"Audio parameters:")
                print(f"  Channels: {channels}")
                print(f"  Sample Rate: {rate}")
                print(f"  Sample Width: {width} bytes")
                
                # Configure output stream to match WAV file
                self.output_stream.setchannels(channels)
                self.output_stream.setrate(rate)
                
                # Read and play audio
                data = wf.readframes(self.output_config["period_size"])
                while data and len(data) > 0:
                    self.output_stream.write(data)
                    data = wf.readframes(self.output_config["period_size"])
            
            return True
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False

    async def stop(self):
        """Stop audio processing"""
        self.running = False
        if self.input_stream:
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.close()