from typing import Optional, List
from openai import AsyncOpenAI
from .config import Settings
import wave
import io
import aiofiles
import asyncio

class TextToSpeech:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.cache_dir = self.settings.CACHE_DIR / "tts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to speech with WAV format"""
        try:
            chunks = self._chunk_text(text)
            print(f"\nSplitting text into {len(chunks)} chunks")
            
            all_audio = []
            
            for i, chunk in enumerate(chunks, 1):
                print(f"\nProcessing chunk {i}/{len(chunks)}...")
                
                try:
                    # Generate speech as WAV
                    response = await self.client.audio.speech.create(
                        model=self.settings.OPENAI_TTS_MODEL,
                        voice=self.settings.OPENAI_TTS_VOICE,
                        input=chunk,
                        response_format="wav"  # Explicitly request WAV
                    )
                    
                    # Get audio content
                    print("Getting audio content...")
                    audio_content = await response.read()
                    print(f"Received {len(audio_content)} bytes of WAV data")
                    
                    if audio_content:
                        all_audio.append(audio_content)
                        print("Added WAV chunk to audio list")
                    
                except Exception as e:
                    print(f"Error in chunk {i}: {e}")
                    continue
            
            if not all_audio:
                print("No audio generated")
                return None
            
            # Return single chunk directly
            if len(all_audio) == 1:
                return all_audio[0]
            
            # Combine WAV chunks if needed
            return self._combine_wav_chunks(all_audio)
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None

    def _chunk_text(self, text: str, max_length: int = 4000) -> List[str]:
        """Split text into chunks"""
        chunks = []
        sentences = text.replace("...", "…").split(".")
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if sentence:
                sentence = sentence.strip() + "."
                if current_length + len(sentence) > max_length:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _combine_wav_chunks(self, wav_chunks: List[bytes]) -> bytes:
        """Combine WAV chunks properly handling headers"""
        if len(wav_chunks) == 1:
            return wav_chunks[0]
            
        # Create output buffer
        output = io.BytesIO()
        
        try:
            # Read first chunk to get WAV parameters
            with wave.open(io.BytesIO(wav_chunks[0]), 'rb') as first_wav:
                params = first_wav.getparams()
                
                # Create output WAV with same parameters
                with wave.open(output, 'wb') as wav_out:
                    wav_out.setparams(params)
                    
                    # Write all chunks' data (skipping headers after first)
                    for i, chunk in enumerate(wav_chunks):
                        with wave.open(io.BytesIO(chunk), 'rb') as wav_in:
                            if i == 0:
                                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
                            else:
                                # Skip WAV header (first 44 bytes) for subsequent chunks
                                wav_in.readframes(1)  # Advance past header
                                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Error combining WAV chunks: {e}")
            # Return first chunk if combination fails
            return wav_chunks[0]

    async def speak(self, text: str):
        """Convert text to speech and play it"""
        try:
            if isinstance(text, dict):
                text = text.get('text', '')
                
            print("\n▶ Generating speech...")
            audio_data = await self.synthesize(text)
            
            if audio_data:
                print("\n▶ Playing audio...")
                await self.play(audio_data)
            else:
                print("No audio generated")
                
        except Exception as e:
            print(f"Error in TTS: {e}")
            import traceback
            traceback.print_exc()