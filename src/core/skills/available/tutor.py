from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings

class Tutor:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        self.subjects = {
            "math": ["algebra", "geometry", "arithmetic"],
            "science": ["biology", "chemistry", "physics"],
            "language": ["grammar", "vocabulary", "writing"],
            "programming": ["python", "scratch", "logic"]
        }
    
    async def handle(self, intent: Dict) -> Dict:
        """Handle tutoring requests"""
        try:
            subject = intent.get("parameters", {}).get("subject", "")
            topic = intent.get("parameters", {}).get("topic", "")
            
            system_prompt = f"""
            You are a patient tutor helping a child understand {subject} - {topic}.
            
            Provide help that:
            1. Breaks down concepts simply
            2. Uses examples and analogies
            3. Encourages understanding
            4. Provides practice opportunities
            """
            
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": intent.get("text", "")}
                ],
                temperature=0.7
            )
            
            return {
                "text": response.choices[0].message.content,
                "context": "tutoring",
                "subject": subject,
                "topic": topic
            }
            
        except Exception as e:
            print(f"Error in tutor: {e}")
            return {
                "text": "I'm having trouble with that lesson. Should we try something else?",
                "context": "error"
            }

skill_manifest = {
    "name": "tutor",
    "intents": ["teach", "explain", "practice", "solve"],
    "description": "Specific subject tutoring and practice"
} 