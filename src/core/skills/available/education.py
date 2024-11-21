from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings

class Education:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
    
    async def handle(self, intent: Dict) -> Dict:
        """Handle educational questions"""
        try:
            question = intent.get("parameters", {}).get("question", "")
            print(f"\n=== Education Handler ===")
            print(f"Question: '{question}'")
            
            # Get child-friendly answer using GPT-4o
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": """
                        You are a friendly teacher for children aged 4-12.
                        Provide simple, clear, and engaging answers.
                        Use age-appropriate language and examples.
                        Keep responses brief but informative.
                        Add a fun fact when relevant.
                    """},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=200  # Keep answers concise
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

skill_manifest = {
    "name": "education",
    "intents": ["answer_question", "explain_topic"],
    "description": "Educational answers and explanations for children"
} 