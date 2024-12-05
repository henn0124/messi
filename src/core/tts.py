from typing import Optional, List, Union, Dict
from openai import AsyncOpenAI
import numpy as np
import wave
import io
from .config import Settings
from .audio import AudioInterface
import asyncio
from pathlib import Path
import aiofiles
import time
import json

class TextToSpeech:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.audio = AudioInterface()
        
        # Initialize cache directory
        self.cache_dir = Path("cache/tts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_response_path = self.cache_dir / "last_response.wav"

    async def _save_last_response(self, audio_data: bytes) -> None:
        """Save the last TTS response"""
        try:
            async with aiofiles.open(self.last_response_path, "wb") as f:
                await f.write(audio_data)
        except Exception as e:
            print(f"Error saving last response: {e}")

    async def _get_last_response(self) -> Optional[bytes]:
        """Get the last TTS response if available"""
        try:
            if self.last_response_path.exists():
                async with aiofiles.open(self.last_response_path, "rb") as f:
                    return await f.read()
            return None
        except Exception as e:
            print(f"Error reading last response: {e}")
            return None

    async def speak(self, response: Union[str, Dict]) -> None:
        """Convert text to speech and play it"""
        try:
            # Extract text from response if it's a dict
            if isinstance(response, dict):
                text = response.get('text', '')
            else:
                text = str(response)

            print("\n▶ Generating speech...")
            print(f"API Request:")
            print(f"  Model: {self.settings.OPENAI_TTS_MODEL}")
            print(f"  Voice: {self.settings.OPENAI_TTS_VOICE}")
            print(f"  Speed: {self.settings.OPENAI_TTS_SPEED}")
            print(f"  Text length: {len(text)} chars")
            print(f"  Text content: \"{text}\"")
            
            start_time = time.time()
            
            # Generate new audio
            tts_response = await self.client.audio.speech.create(
                model=self.settings.OPENAI_TTS_MODEL,
                voice=self.settings.OPENAI_TTS_VOICE,
                input=text,
                response_format="wav",
                speed=self.settings.OPENAI_TTS_SPEED
            )
            
            api_time = time.time() - start_time
            print(f"⏱️  TTS API: {api_time*1000:.1f}ms")
            
            # Stream the response directly to audio output
            if tts_response:
                print("\n▶ Playing response...")
                # Get the binary content
                audio_data = tts_response.read()
                audio_size = len(audio_data) / 1024  # KB
                print(f"  Response size: {audio_size:.1f}KB")
                
                # Save as last response
                await self._save_last_response(audio_data)
                
                # Save metadata
                meta = {
                    'timestamp': time.time(),
                    'api': {
                        'model': self.settings.OPENAI_TTS_MODEL,
                        'voice': self.settings.OPENAI_TTS_VOICE,
                        'speed': self.settings.OPENAI_TTS_SPEED,
                        'text_length': len(text),
                        'text_content': text,
                        'response_size': audio_size,
                        'api_time_ms': api_time * 1000
                    }
                }
                
                meta_path = self.cache_dir / "last_response.json"
                async with aiofiles.open(meta_path, 'w') as f:
                    await f.write(json.dumps(meta, indent=2))
                
                await self.audio.play_audio_chunk(audio_data)
            else:
                print("No audio generated")
                
        except Exception as e:
            print(f"Error in TTS: {e}")
            import traceback
            traceback.print_exc()

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to speech with optimized chunking"""
        try:
            # Quick path for short text
            if len(text) < 1000:
                print(f"\nSynthesizing text: \"{text}\"")
                response = await self.client.audio.speech.create(
                    model=self.settings.OPENAI_TTS_MODEL,
                    voice=self.settings.OPENAI_TTS_VOICE,
                    input=text,
                    response_format="wav",
                    speed=self.settings.OPENAI_TTS_SPEED
                )
                # Get the raw response data
                audio_data = response.read()
                
                # Save as last response
                await self._save_last_response(audio_data)
                
                return audio_data
            
            # Standard chunking for longer text
            chunks = self._chunk_text(text)
            chunk_count = len(chunks)
            print(f"\nProcessing {chunk_count} chunk{'s' if chunk_count > 1 else ''}")
            
            # Process chunks concurrently
            async def process_chunk(chunk: str, index: int):
                try:
                    print(f"Processing chunk {index + 1}/{chunk_count}: \"{chunk}\"")
                    response = await self.client.audio.speech.create(
                        model=self.settings.OPENAI_TTS_MODEL,
                        voice=self.settings.OPENAI_TTS_VOICE,
                        input=chunk,
                        response_format="wav",
                        speed=self.settings.OPENAI_TTS_SPEED
                    )
                    # Get the raw response data
                    return response.read()
                except Exception as e:
                    print(f"Error in chunk {index + 1}: {e}")
                    return None
            
            # Process chunks in parallel
            tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
            responses = await asyncio.gather(*tasks)
            
            # Filter out failed chunks
            all_audio = [r for r in responses if r]
            
            if not all_audio:
                print("No audio generated")
                return None
            
            # If only one chunk, return directly
            if len(all_audio) == 1:
                audio_data = all_audio[0]
                await self._save_last_response(audio_data)
                return audio_data
            
            # Combine chunks if needed
            print("Combining audio...")
            combined_audio = await self._combine_audio(all_audio)
            
            # Save combined audio as last response
            await self._save_last_response(combined_audio)
            
            return combined_audio
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            return None

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into optimal chunks for TTS"""
        # Don't chunk short text
        if len(text) < 1000:
            return [text]
        
        # For longer text, split on sentence boundaries
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) < 1000:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = len(sentence)
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks

    async def _combine_audio(self, audio_chunks: List[bytes]) -> bytes:
        """Combine WAV audio chunks"""
        try:
            # For single chunk, return directly
            if len(audio_chunks) == 1:
                return audio_chunks[0]
            
            # Create output buffer
            output = io.BytesIO()
            
            # Get WAV parameters from first chunk
            with wave.open(io.BytesIO(audio_chunks[0]), 'rb') as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                
                # Create output WAV
                with wave.open(output, 'wb') as wav_out:
                    wav_out.setnchannels(channels)
                    wav_out.setsampwidth(sampwidth)
                    wav_out.setframerate(framerate)
                    
                    # Write first chunk frames
                    wav_out.writeframes(wf.readframes(wf.getnframes()))
                    
                    # Write remaining chunks
                    for chunk in audio_chunks[1:]:
                        with wave.open(io.BytesIO(chunk), 'rb') as wav_in:
                            wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Error combining audio: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            # Return first chunk if combination fails
            if audio_chunks:
                return audio_chunks[0]
            return None