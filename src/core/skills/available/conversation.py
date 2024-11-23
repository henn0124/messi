from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings

class Conversation:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        self.personality = {
            "tone": "friendly and warm",
            "style": "child-appropriate",
            "complexity": "simple",
            "engagement": "interactive"
        }
    
    async def handle(self, intent: Dict) -> Dict:
        """Handle general conversation"""
        try:
            text = intent.get("text", "")
            context = intent.get("context", {})
            
            system_prompt = f"""
            You are a friendly AI assistant talking to a child.
            Tone: {self.personality['tone']}
            Style: {self.personality['style']}
            
            Previous context: {context.get('previous_topic', 'none')}
            Recent mentions: {', '.join(context.get('mentioned_entities', []))}
            
            Keep responses:
            1. Short and simple
            2. Engaging and warm
            3. Age-appropriate
            4. Educational when possible
            """
            
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.7
            )
            
            return {
                "text": response.choices[0].message.content,
                "context": "conversation",
                "auto_continue": False
            }
            
        except Exception as e:
            print(f"Error in conversation: {e}")
            return {
                "text": "I didn't quite catch that. Could you say it again?",
                "context": "error"
            }

skill_manifest = {
    "name": "conversation",
    "intents": ["chat", "greet", "farewell", "thank"],
    "description": "General conversation handling"
} 