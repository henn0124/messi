from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings
import random

class Education:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
    
    async def handle(self, route: Dict) -> Dict:
        """Handle educational questions"""
        try:
            question = route["parameters"]["question"]
            context = route["parameters"].get("context", {})
            
            # If response is from cache, add context-aware prefix
            if route.get("from_cache"):
                prefix = self._get_cache_prefix(context)
                return {
                    "text": f"{prefix} {route['text']}",
                    "context": "education",
                    "auto_continue": False
                }
            
            # Create prompt with context
            prompt = f"""Question: {question}
Previous topic: {context.get('previous_topic', 'none')}
Related entities: {', '.join(context.get('mentioned_entities', []))}
Previous question: {context.get('last_question', 'none')}

Provide a child-friendly answer that:
1. Builds on previous knowledge if related
2. Makes connections to familiar concepts
3. Encourages further curiosity
"""

            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            print(f"Answer: {answer}")
            
            return {
                "text": answer,
                "context": "education",
                "auto_continue": False
            }
            
        except Exception as e:
            print(f"Error in education handler: {e}")
            return {
                "text": "I'm not sure about that. Would you like to hear a story instead?",
                "context": "error"
            }

    def _get_cache_prefix(self, context: Dict) -> str:
        """Get context-aware prefix for cached responses"""
        prefixes = [
            "I remember this one!",
            "As we discussed before,",
            "Let me tell you again about this.",
            "This is interesting to revisit:"
        ]
        return random.choice(prefixes)

skill_manifest = {
    "name": "education",
    "intents": ["answer_question", "explain_topic"],
    "description": "Educational answers and explanations for children"
} 