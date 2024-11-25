from typing import Optional, List, Union, Dict
from openai import AsyncOpenAI
import numpy as np
import wave
import io
from .config import Settings
from .audio import AudioInterface

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
        """Synthesize text to speech with proper response handling"""
        try:
            # Split text into chunks
            chunks = self._chunk_text(text)
            print(f"\nSplitting text into {len(chunks)} chunks")
            
            all_audio = []
            
            for i, chunk in enumerate(chunks, 1):
                try:
                    print(f"\nProcessing chunk {i}/{len(chunks)}...")
                    print("Calling OpenAI TTS API...")
                    
                    # Get audio content directly
                    response = await self.client.audio.speech.create(
                        model=self.settings.OPENAI_TTS_MODEL,
                        voice=self.settings.OPENAI_TTS_VOICE,
                        input=chunk,
                        response_format="wav"
                    )
                    
                    print(f"Response type: {type(response)}")
                    
                    if response:
                        all_audio.append(response)
                        print(f"✓ Audio chunk {i} added (type: {type(response)})")
                    else:
                        print(f"✗ No audio for chunk {i}")
                    
                except Exception as e:
                    print(f"Error in chunk {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not all_audio:
                print("No audio generated")
                return None
            
            # If only one chunk, return it directly
            if len(all_audio) == 1:
                print("✓ Audio ready")
                return all_audio[0]
            
            # Combine multiple chunks
            print("Combining audio chunks...")
            combined = self._combine_audio(all_audio)
            print("✓ Audio chunks combined")
            return combined
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        # Simple splitting by sentences
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            if sum(len(s) for s in current_chunk) + len(sentence) < 250:
                current_chunk.append(sentence)
            else:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks

    def _combine_audio(self, audio_chunks: List[bytes]) -> bytes:
        """Combine WAV audio chunks"""
        # Create WAV in memory
        output = io.BytesIO()
        
        try:
            # Get parameters from first chunk
            with wave.open(io.BytesIO(audio_chunks[0]), 'rb') as first_wav:
                params = first_wav.getparams()
                
                # Create output WAV
                with wave.open(output, 'wb') as wav_out:
                    wav_out.setparams(params)
                    
                    # Write all chunks
                    for chunk in audio_chunks:
                        with wave.open(io.BytesIO(chunk), 'rb') as wav_in:
                            wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Error combining audio: {e}")
            # Return first chunk if combination fails
            return audio_chunks[0] if audio_chunks else None