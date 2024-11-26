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
            # Split text into chunks
            chunks = self._chunk_text(text)
            chunk_count = len(chunks)
            print(f"\nProcessing {chunk_count} chunk{'s' if chunk_count > 1 else ''}")
            
            # For single short chunk, process directly
            if chunk_count == 1 and len(chunks[0]) < 500:
                response = await self.client.audio.speech.create(
                    model=self.settings.OPENAI_TTS_MODEL,
                    voice=self.settings.OPENAI_TTS_VOICE,
                    input=chunks[0],
                    response_format="wav"
                )
                return response
            
            # For multiple chunks, process in parallel
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
            
            # Process chunks concurrently
            tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
            responses = await asyncio.gather(*tasks)
            
            # Filter out failed chunks
            all_audio = [r for r in responses if r]
            
            if not all_audio:
                print("No audio generated")
                return None
            
            # If only one chunk succeeded, return it
            if len(all_audio) == 1:
                return all_audio[0]
            
            # Combine chunks
            print("Combining audio...")
            return await self._combine_audio(all_audio)
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into optimal chunks for TTS"""
        # Don't chunk short text
        if len(text) < 500:  # Increased from 250
            return [text]
            
        # For longer text, split on natural breaks
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split on sentence endings and other natural breaks
        sentences = [s.strip() for s in text.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|').split('|')]
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If sentence alone is too long, split on commas
            if sentence_length > 500:
                comma_parts = sentence.split(', ')
                for part in comma_parts:
                    if current_length + len(part) < 500:
                        current_chunk.append(part)
                        current_length += len(part)
                    else:
                        chunks.append(', '.join(current_chunk))
                        current_chunk = [part]
                        current_length = len(part)
            
            # Normal sentence handling
            elif current_length + sentence_length < 500:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks

    async def _combine_audio(self, audio_chunks: List[bytes]) -> bytes:
        """Combine WAV audio chunks"""
        # Create WAV in memory
        output = io.BytesIO()
        
        try:
            # Get parameters from first chunk
            first_chunk = audio_chunks[0]
            if hasattr(first_chunk, 'aread'):
                first_chunk = await first_chunk.aread()
            elif hasattr(first_chunk, 'read'):
                first_chunk = first_chunk.read()
                
            with wave.open(io.BytesIO(first_chunk), 'rb') as first_wav:
                params = first_wav.getparams()
                
                # Create output WAV
                with wave.open(output, 'wb') as wav_out:
                    wav_out.setparams(params)
                    
                    # Write all chunks
                    for chunk in audio_chunks:
                        if hasattr(chunk, 'aread'):
                            chunk = await chunk.aread()
                        elif hasattr(chunk, 'read'):
                            chunk = chunk.read()
                            
                        with wave.open(io.BytesIO(chunk), 'rb') as wav_in:
                            wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Error combining audio: {e}")
            # Return first chunk if combination fails
            return audio_chunks[0] if audio_chunks else None