from typing import Optional, Union
from pathlib import Path
import aiofiles
from openai import AsyncOpenAI
from .config import Settings

class TextToSpeech:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.cache_dir = self.settings.CACHE_DIR / "tts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Define system prompts for different contexts
        self.prompts = {
            "storytelling": {
                "voice": "fable",  # Warm and gentle storytelling voice
                "speed": 0.9,      # Slightly slower for clarity
                "model": "tts-1-hd",
                "system_prompt": "You are a warm and engaging storyteller, speaking to young children at bedtime. Your voice should be gentle and soothing, perfect for bedtime stories."
            },
            "story_creation": {
                "voice": "nova",   # Encouraging and supportive voice
                "speed": 1.0,      # Normal speed for interaction
                "model": "tts-1-hd",
                "system_prompt": "You are an encouraging guide helping children create their own stories. Your voice should be supportive and enthusiastic about their creative ideas."
            }
        }
    
    async def synthesize(self, text: str, context: str = "storytelling") -> Union[bytes, None]:
        """Convert text to speech using OpenAI's TTS API"""
        try:
            print("Sending text to TTS API...")
            
            # Get context-specific settings
            prompt_settings = self.prompts.get(context, self.prompts["storytelling"])
            
            # Add narrative markers for better prosody
            marked_text = self._add_narrative_markers(text, context)
            
            print(f"Using voice: {prompt_settings['voice']}")
            print(f"Context: {context}")
            
            response = await self.client.audio.speech.create(
                model=prompt_settings["model"],
                voice=prompt_settings["voice"],
                input=marked_text,
                speed=prompt_settings["speed"],
                response_format="wav"
            )
            
            print("TTS response received")
            print(f"Audio content length: {len(response.content)} bytes")
            
            # For debugging, save the last response
            debug_path = self.cache_dir / "last_response.wav"
            with open(debug_path, "wb") as f:
                f.write(response.content)
            print(f"Debug: Saved audio to {debug_path}")
            
            return response.content
            
        except Exception as e:
            print(f"Detailed error in TTS: {str(e)}")
            return None
    
    def _add_narrative_markers(self, text: str, context: str) -> str:
        """Add SSML-like markers for better prosody"""
        if context == "storytelling":
            # Add subtle pauses and emphasis for storytelling
            text = text.replace(". ", "... ")
            text = text.replace("! ", "!... ")
            text = text.replace("? ", "?... ")
            
            # Mark dialogue if present
            if '"' in text:
                text = text.replace('"', '... "')
                
        elif context == "story_creation":
            # Add enthusiasm markers for encouragement
            text = text.replace("!", "!...")
            
        return text
    
    async def _cache_audio(self, path: Path, audio_data: bytes):
        """Cache audio data to file"""
        try:
            async with aiofiles.open(path, 'wb') as f:
                await f.write(audio_data)
        except Exception as e:
            print(f"Error caching audio: {e}")