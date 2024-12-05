from typing import Optional
from openai import AsyncOpenAI
from .config import Settings
import wave
import io
import traceback
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import json

class SpeechManager:
    def __init__(self):
        self.settings = Settings()
        print(f"SpeechManager initializing with API key starting with: {self.settings.OPENAI_API_KEY[:10]}...")
        self.client = AsyncOpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            timeout=10.0  # 10 second timeout
        )
        
        # Create cache directory for STT WAV files
        self.cache_dir = Path(self.settings.CACHE_DIR) / "stt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Fixed filenames for latest files
        self.latest_wav = self.cache_dir / "latest.wav"
        self.latest_meta = self.cache_dir / "latest.meta"
        
    async def process_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio through Whisper with optimized processing"""
        try:
            start_time = time.time()
            print("\nProcessing command...")
            
            # Convert raw PCM to numpy array
            t0 = time.time()
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Trim silence from start and end
            threshold = self.settings.audio["input"]["silence_threshold"]
            is_sound = np.abs(audio_np) > threshold
            sound_indices = np.where(is_sound)[0]
            if len(sound_indices) > 0:
                start_idx = max(0, sound_indices[0] - int(0.1 * self.settings.audio["input"]["rate"]))  # 100ms buffer
                end_idx = min(len(audio_np), sound_indices[-1] + int(0.1 * self.settings.audio["input"]["rate"]))
                audio_np = audio_np[start_idx:end_idx]
            
            t1 = time.time()
            print(f"⏱️  PCM conversion and trimming: {(t1-t0)*1000:.1f}ms")
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.settings.audio["input"]["rate"])  # Use original sample rate
                wf.writeframes(audio_np.tobytes())
            
            t2 = time.time()
            print(f"⏱️  WAV creation: {(t2-t1)*1000:.1f}ms")
            
            # Cache the WAV file (overwrite previous)
            wav_buffer.seek(0)
            with open(self.latest_wav, 'wb') as f:
                f.write(wav_buffer.getvalue())
            print(f"📝 Cached audio: {self.latest_wav}")
            
            # Process with Whisper
            wav_buffer.seek(0)
            print("\n🎤 Sending to Whisper API...")
            print(f"API Request:")
            print(f"  Model: {self.settings.OPENAI_WHISPER_MODEL}")
            print(f"  Language: en")
            print(f"  Audio duration: {len(audio_np) / self.settings.audio['input']['rate']:.1f}s")
            print(f"  Audio size: {len(wav_buffer.getvalue()) / 1024:.1f}KB")
            
            response = await self.client.audio.transcriptions.create(
                model=self.settings.OPENAI_WHISPER_MODEL,
                file=('audio.wav', wav_buffer),
                response_format="text",
                language="en"
            )
            t3 = time.time()
            print(f"⏱️  Whisper API: {(t3-t2)*1000:.1f}ms")
            
            if response:
                text = str(response).strip()
                total_time = time.time() - start_time
                print(f"\n⏱️  Total STT processing time: {total_time*1000:.1f}ms")
                
                # Save metadata (overwrite previous)
                meta = {
                    'text': text,
                    'timestamp': datetime.now().isoformat(),
                    'audio': {
                        'sample_rate': self.settings.audio["input"]["rate"],
                        'channels': 1,
                        'sample_width': 2,
                        'original_samples': len(audio_data) // 2,  # 2 bytes per sample
                        'trimmed_samples': len(audio_np),
                        'duration': len(audio_np) / self.settings.audio["input"]["rate"]
                    },
                    'timing': {
                        'pcm_conversion_ms': (t1-t0)*1000,
                        'wav_creation_ms': (t2-t1)*1000,
                        'whisper_api_ms': (t3-t2)*1000,
                        'total_ms': total_time*1000
                    },
                    'api': {
                        'model': self.settings.OPENAI_WHISPER_MODEL,
                        'language': 'en',
                        'audio_duration': len(audio_np) / self.settings.audio["input"]["rate"],
                        'audio_size': len(wav_buffer.getvalue())
                    }
                }
                with open(self.latest_meta, 'w') as f:
                    json.dump(meta, f, indent=2)
                
                return text
            else:
                print("\n✗ Whisper API returned no text")
                return None
                
        except Exception as e:
            print(f"\n✗ Error in Whisper processing: {e}")
            traceback.print_exc()
            return None