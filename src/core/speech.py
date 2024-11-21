from typing import Optional
from openai import AsyncOpenAI
from .config import Settings
from pathlib import Path
import wave
import io

class SpeechManager:
    def __init__(self):
        self.settings = Settings()
        print(f"SpeechManager initializing with API key starting with: {self.settings.OPENAI_API_KEY[:10]}...")
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.temp_dir = self.settings.TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)
    
    async def process_audio(self, audio_data: bytes) -> str:
        """Process audio using OpenAI's Whisper API"""
        try:
            # Create a proper WAV file
            temp_path = self.temp_dir / "temp_audio.wav"
            
            # Convert raw PCM to WAV
            with wave.open(str(temp_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wav_file.setframerate(44100)  # Sample rate
                wav_file.writeframes(audio_data)
            
            print("Sending audio to Whisper API...")
            print(f"Using API key starting with: {self.settings.OPENAI_API_KEY[:10]}...")
            
            with open(temp_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                    temperature=0.3,
                    language="en",
                    prompt="This is a child speaking to a bedtime story assistant."
                )
            
            temp_path.unlink()
            print(f"Transcription received: {transcript}")
            return transcript
            
        except Exception as e:
            print(f"Detailed error in speech processing: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response details: {e.response}")
            return ""