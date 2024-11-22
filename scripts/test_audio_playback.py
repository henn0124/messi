import pyaudio
import wave
import numpy as np
import asyncio
import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.audio import AudioInterface
from src.core.config import Settings

class AudioPlaybackTest:
    def __init__(self):
        self.settings = Settings()
        self.audio = AudioInterface()
        self.test_file = Path("/home/pi/messi/cache/tts/last_response.wav")
        
    async def run_tests(self):
        """Run audio playback tests"""
        try:
            print("\n=== Audio Playback Test ===")
            
            # Check file exists
            if not self.test_file.exists():
                print(f"Error: Test file not found at {self.test_file}")
                return
                
            # Get file info safely
            with wave.open(str(self.test_file), 'rb') as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / float(framerate)
                
                print("\nWAV File Properties:")
                print(f"Channels: {channels}")
                print(f"Sample width: {sampwidth}")
                print(f"Frame rate: {framerate}")
                print(f"File size: {self.test_file.stat().st_size/1024/1024:.2f} MB")
                print(f"Duration: {duration:.2f} seconds")
                
                # Read audio data in chunks
                print("\nReading audio file...")
                chunk_size = 1024 * 1024  # 1MB chunks
                audio_chunks = []
                while True:
                    chunk = wf.readframes(chunk_size)
                    if not chunk:
                        break
                    audio_chunks.append(chunk)
                
                audio_data = b''.join(audio_chunks)
                print(f"Successfully read {len(audio_data)/1024/1024:.2f} MB of audio data")
            
            # Initialize audio interface
            await self.audio.initialize()
            
            # Test menu
            while True:
                print("\nPlayback Options:")
                print("1. Play original audio")
                print("2. Play with soft normalization")
                print("3. Play with smooth volume adjustment")
                print("4. Play with anti-clipping")
                print("5. Exit")
                
                choice = input("\nChoose test (1-5): ")
                
                if choice == '1':
                    await self.play_original(audio_data)
                elif choice == '2':
                    await self.play_soft_normalized(audio_data)
                elif choice == '3':
                    volume = float(input("Enter volume multiplier (0.1-2.0): "))
                    await self.play_smooth_volume(audio_data, volume)
                elif choice == '4':
                    await self.play_anti_clipping(audio_data)
                elif choice == '5':
                    break
                else:
                    print("Invalid choice")
            
        except Exception as e:
            print(f"Test error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()
    
    async def play_original(self, audio_data: bytes):
        """Play original audio"""
        print("\nPlaying original audio...")
        await self.audio.play_audio_chunk(audio_data)
    
    async def play_soft_normalized(self, audio_data: bytes):
        """Play with gentle normalization"""
        print("\nPlaying with soft normalization...")
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Convert to float [-1, 1]
        audio_array = audio_array / 32768.0
        
        # Calculate RMS and peak
        rms = np.sqrt(np.mean(np.square(audio_array)))
        peak = np.max(np.abs(audio_array))
        
        print(f"Original - RMS: {rms:.3f}, Peak: {peak:.3f}")
        
        # Soft normalize to -12dB RMS
        target_rms = 0.25  # -12dB
        gain = min(target_rms / rms, 1.0)  # Never amplify, only reduce
        audio_array = audio_array * gain
        
        # Convert back to int16
        audio_array = (audio_array * 32767).astype(np.int16)
        await self.audio.play_audio_chunk(audio_array.tobytes())
    
    async def play_smooth_volume(self, audio_data: bytes, volume: float):
        """Play with smooth volume adjustment"""
        print(f"\nPlaying with smooth volume {volume}x...")
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_array = audio_array / 32768.0
        
        # Apply smooth volume curve
        if volume != 1.0:
            sign = np.sign(audio_array)
            curve = 1.0 - np.exp(-np.abs(audio_array) * volume)
            audio_array = sign * curve
        
        # Prevent clipping
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        audio_array = (audio_array * 32767).astype(np.int16)
        await self.audio.play_audio_chunk(audio_array.tobytes())
    
    async def play_anti_clipping(self, audio_data: bytes):
        """Play with enhanced anti-clipping"""
        try:
            print("\nPlaying with enhanced anti-clipping...")
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0
            
            # Calculate initial levels
            rms = np.sqrt(np.mean(np.square(audio_array)))
            peak = np.max(np.abs(audio_array))
            print(f"Original - RMS: {rms:.3f}, Peak: {peak:.3f}")
            
            # Anti-clipping parameters
            threshold = 0.4
            ratio = 0.25
            knee_width = 0.2
            makeup_gain = 1.2
            
            print("Applying compression...")
            
            # Basic compression first
            audio_array = np.clip(audio_array, -threshold, threshold) * makeup_gain
            
            # Soft limiting
            audio_array = np.tanh(audio_array)
            
            # Final safety clip
            audio_array = np.clip(audio_array, -0.95, 0.95)
            
            # Calculate final levels
            final_rms = np.sqrt(np.mean(np.square(audio_array)))
            final_peak = np.max(np.abs(audio_array))
            print(f"After processing - RMS: {final_rms:.3f}, Peak: {final_peak:.3f}")
            
            # Convert back to int16
            print("Converting to audio...")
            audio_array = (audio_array * 32767).astype(np.int16)
            
            print("Playing audio...")
            await self.audio.play_audio_chunk(audio_array.tobytes())
            
        except Exception as e:
            print(f"Error in anti-clipping: {e}")
            import traceback
            traceback.print_exc()
    
    async def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'audio'):
            await self.audio.stop()

if __name__ == "__main__":
    tester = AudioPlaybackTest()
    asyncio.run(tester.run_tests()) 