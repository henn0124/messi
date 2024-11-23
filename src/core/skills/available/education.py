from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings
import random

class Education:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
    
    async def handle(self, text: str) -> Dict:
        """Handle educational questions"""
        try:
            print("\n=== Education Skill ===")
            print(f"Question: '{text}'")
            
            # Create educational prompt
            system_prompt = """
            You are a friendly educational assistant for children.
            Provide simple, clear, and engaging answers.
            Include interesting facts but keep explanations brief.
            Use child-friendly language and examples.
            Keep responses under 3 sentences when possible.
            """
            
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                max_tokens=150  # Keep responses concise
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"Answer: '{answer}'")
            
            return {
                "text": answer,
                "context": "education",
                "auto_continue": False
            }
            
        except Exception as e:
            print(f"Error in education handler: {e}")
            return {
                "text": "I'm not sure about that. Would you like to try asking another way?",
                "context": "error"
            }

skill_manifest = {
    "name": "education",
    "description": "Educational answers and explanations for children",
    "intents": ["education", "answer_question", "explain_topic"]
} 