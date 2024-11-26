from typing import Optional, List, Union, Dict
from openai import AsyncOpenAI
import numpy as np
import wave
import io
from .config import Settings
from .audio import AudioInterface
import asyncio

class TextToSpeech:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.audio = AudioInterface()

    async def speak(self, response: Union[str, Dict]) -> None:
        """Convert text to speech and play it"""
        try:
            # Extract text from response if it's a dict
            if isinstance(response, dict):
                text = response.get('text', '')
            else:
                text = str(response)

            print("\n▶ Generating speech...")
            audio_data = await self.synthesize(text)
            
            if audio_data:
                print("\n▶ Playing response...")
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
                response = await self.client.audio.speech.create(
                    model=self.settings.OPENAI_TTS_MODEL,
                    voice=self.settings.OPENAI_TTS_VOICE,
                    input=text,
                    response_format="wav"
                )
                return response
            
            # Standard chunking for longer text
            chunks = self._chunk_text(text)
            chunk_count = len(chunks)
            print(f"\nProcessing {chunk_count} chunk{'s' if chunk_count > 1 else ''}")
            
            # Process chunks concurrently
            async def process_chunk(chunk: str, index: int):
                try:
                    print(f"Processing chunk {index + 1}/{chunk_count}...")
                    return await self.client.audio.speech.create(
                        model=self.settings.OPENAI_TTS_MODEL,
                        voice=self.settings.OPENAI_TTS_VOICE,
                        input=chunk,
                        response_format="wav"
                    )
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
                return all_audio[0]
            
            # Combine chunks if needed
            print("Combining audio...")
            return await self._combine_audio(all_audio)
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into optimal chunks for TTS with smart thresholds"""
        # Don't chunk short text (increased threshold)
        if len(text) < 1000:  # Increased from 500
            return [text]
        
        # For medium text, use larger chunks
        if len(text) < 2000:
            sentences = text.split('. ')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) < 1500:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
                else:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentence]
                    current_length = len(sentence)
            
            if current_chunk:
                chunks.append('. '.join(current_chunk))
            return chunks

        # For long text, use more aggressive chunking
        return self._chunk_long_text(text)

    async def _combine_audio(self, audio_chunks: List[bytes]) -> bytes:
        """Combine WAV audio chunks"""
        try:
            # For single chunk, return directly
            if len(audio_chunks) == 1:
                chunk = audio_chunks[0]
                if hasattr(chunk, 'aread'):
                    return await chunk.aread()
                elif hasattr(chunk, 'read'):
                    return chunk.read()
                return chunk
            
            # Create output buffer
            output = io.BytesIO()
            
            # Get first chunk data
            first_chunk = audio_chunks[0]
            if hasattr(first_chunk, 'aread'):
                first_data = await first_chunk.aread()
            elif hasattr(first_chunk, 'read'):
                first_data = first_chunk.read()
            else:
                first_data = first_chunk
            
            # Get WAV parameters from first chunk
            with wave.open(io.BytesIO(first_data), 'rb') as wf:
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
                        # Get chunk data
                        if hasattr(chunk, 'aread'):
                            chunk_data = await chunk.aread()
                        elif hasattr(chunk, 'read'):
                            chunk_data = chunk.read()
                        else:
                            chunk_data = chunk
                        
                        # Extract and write frames
                        with wave.open(io.BytesIO(chunk_data), 'rb') as wav_in:
                            wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Error combining audio: {e}")
            # Return first chunk if combination fails
            if audio_chunks:
                chunk = audio_chunks[0]
                if hasattr(chunk, 'aread'):
                    return await chunk.aread()
                elif hasattr(chunk, 'read'):
                    return chunk.read()
                return chunk
            return None