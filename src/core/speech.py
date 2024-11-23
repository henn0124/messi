from typing import Optional
from openai import AsyncOpenAI
from .config import Settings
from pathlib import Path
import wave
import io
import traceback
import time
import numpy as np

class SpeechManager:
    def __init__(self):
        self.settings = Settings()
        print(f"SpeechManager initializing with API key starting with: {self.settings.OPENAI_API_KEY[:10]}...")
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
    async def process_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio through Whisper with latest debug file"""
        try:
            print("\n=== Speech Processing Chain ===")
            
            # First, check the incoming audio format
            try:
                with wave.open(io.BytesIO(audio_data), 'rb') as check_wav:
                    original_rate = check_wav.getframerate()
                    original_channels = check_wav.getnchannels()
                    original_width = check_wav.getsampwidth()
                    print("\n1. Original Audio Format:")
                    print(f"   Sample Rate: {original_rate} Hz")
                    print(f"   Channels: {original_channels}")
                    print(f"   Bit Depth: {original_width * 8} bits")
            except Exception as e:
                print("\n1. Received Raw PCM Audio:")
                print(f"   Size: {len(audio_data)/1024:.1f}KB")
                print(f"   Expected Rate: {self.settings.AUDIO_NATIVE_RATE} Hz")
            
            print("\n2. Converting for Whisper...")
            
            # Save to fixed debug file path
            debug_path = self.settings.CACHE_DIR / "debug" / "last_whisper_input.wav"
            debug_path.parent.mkdir(exist_ok=True)
            
            # If input is raw PCM, convert from native rate
            input_rate = getattr(self.settings, 'AUDIO_NATIVE_RATE', 44100)
            print(f"   Input Rate: {input_rate} Hz")
            print(f"   Target Rate: 16000 Hz (Whisper requirement)")
            
            # Convert sample rate if needed
            if input_rate != 16000:
                print("   Performing sample rate conversion...")
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Calculate resampling parameters
                ratio = 16000 / input_rate
                output_length = int(len(audio_np) * ratio)
                
                # Resample
                time_original = np.linspace(0, len(audio_np), len(audio_np))
                time_new = np.linspace(0, len(audio_np), output_length)
                resampled = np.interp(time_new, time_original, audio_np).astype(np.int16)
                
                # Save resampled audio
                with wave.open(str(debug_path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(resampled.tobytes())
                
                print(f"   Conversion complete: {len(audio_np)} -> {len(resampled)} samples")
            else:
                # Save directly if already at correct rate
                with wave.open(str(debug_path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
            
            # Verify output format
            with wave.open(str(debug_path), 'rb') as wf:
                print("\n3. Output WAV Details:")
                print(f"   Channels: {wf.getnchannels()} (Mono)")
                print(f"   Sample Width: {wf.getsampwidth() * 8} bits")
                print(f"   Sample Rate: {wf.getframerate()} Hz")
                print(f"   Duration: {wf.getnframes() / float(wf.getframerate()):.2f}s")
                print(f"   File size: {debug_path.stat().st_size/1024:.1f}KB")
                print(f"   Saved to: {debug_path}")
            
            print("\n4. Sending to Whisper API...")
            print(f"   Model: {self.settings.OPENAI_WHISPER_MODEL}")
            print(f"   Language: English (enforced)")
            
            # Process with Whisper
            start_time = time.time()
            with open(debug_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=self.settings.OPENAI_WHISPER_MODEL,
                    file=audio_file,
                    response_format="text",
                    language="en"
                )
            
            process_time = time.time() - start_time
            
            if response:
                text = str(response).strip()
                print(f"\n=== Whisper API Result ===")
                print(f"Processing time: {process_time:.1f}s")
                print(f"Transcribed text:")
                print(f"'{text}'")
                print(f"\nDebug WAV file: {debug_path}")
                print("=== End Whisper Result ===")
                return text
            else:
                print("\n✗ Whisper API returned no text")
                return None
                
        except Exception as e:
            print(f"\n✗ Error in Whisper processing: {e}")
            import traceback
            traceback.print_exc()
            return None