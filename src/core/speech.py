from typing import Optional
from openai import AsyncOpenAI
from .config import Settings
from pathlib import Path
import wave
import io
import traceback

class SpeechManager:
    def __init__(self):
        self.settings = Settings()
        print(f"SpeechManager initializing with API key starting with: {self.settings.OPENAI_API_KEY[:10]}...")
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
    async def process_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio using Whisper"""
        try:
            # Create temporary WAV file
            temp_path = Path("temp_audio.wav")
            with wave.open(str(temp_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)  # Whisper expects 16kHz
                wav_file.writeframes(audio_data)
            
            print("Sending to Whisper API...")
            with open(temp_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                    language="en"
                )
            
            temp_path.unlink()
            return response
            
        except Exception as e:
            print(f"Speech processing error: {e}")
            traceback.print_exc()
            return None