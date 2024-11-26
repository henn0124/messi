from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings

class Story:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
    async def handle(self, text: str, context_info: Dict) -> Dict:
        """Handle story requests with context awareness"""
        try:
            # Check if this is actually an educational query
            if context_info.get("current") == "education":
                return {
                    "text": "I see you're asking about learning something. Let me help with that!",
                    "context": "education",
                    "auto_continue": True,
                    "conversation_active": True,
                    "redirect": "education"
                }
            
            # Generate story response
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a friendly storyteller."},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            story = response.choices[0].message.content.strip()
            
            return {
                "text": story,
                "context": "story",
                "auto_continue": False
            }
            
        except Exception as e:
            print(f"Error in story handler: {e}")
            # Return educational context for error handling
            return {
                "text": "I notice you're asking about something interesting! Let me help you learn about it.",
                "context": "education",
                "auto_continue": True,
                "conversation_active": True,
                "redirect": "education"
            }

skill_manifest = {
    "name": "story",
    "description": "Tell engaging stories",
    "intents": ["story", "tell_story"]
} 