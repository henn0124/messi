from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings
import random

class Tutor:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        # Define vocabulary difficulty levels
        self.difficulty_levels = {
            "beginner": {
                "age_range": "5-7",
                "word_length": "3-5 letters",
                "topics": ["colors", "animals", "food", "family", "numbers"]
            },
            "intermediate": {
                "age_range": "8-10",
                "word_length": "6-8 letters",
                "topics": ["science", "geography", "sports", "weather", "time"]
            },
            "advanced": {
                "age_range": "11+",
                "word_length": "9+ letters",
                "topics": ["technology", "literature", "history", "nature", "space"]
            }
        }
        
        # Track learning progress
        self.learning_progress = {
            "words_practiced": set(),
            "success_rate": {},
            "difficulty_level": "beginner",
            "current_topic": None
        }
    
    async def handle(self, text: str, context_info: Dict) -> Dict:
        """Handle spelling and vocabulary tutoring requests"""
        try:
            # Check for specific request types
            if "spell" in text.lower():
                return await self._handle_spelling_request(text, context_info)
            elif "mean" in text.lower() or "definition" in text.lower():
                return await self._handle_definition_request(text, context_info)
            elif "practice" in text.lower():
                return await self._handle_practice_request(text, context_info)
            
            # Default to vocabulary help
            return await self._generate_vocabulary_exercise(context_info)
            
        except Exception as e:
            print(f"Error in tutor: {e}")
            return {
                "text": "I'm having trouble with that lesson. Should we try something simpler?",
                "context": "tutor",
                "auto_continue": True
            }

    async def _handle_spelling_request(self, text: str, context_info: Dict) -> Dict:
        """Handle spelling practice"""
        try:
            system_prompt = f"""
            You are a friendly spelling tutor for children.
            Current difficulty level: {self.learning_progress['difficulty_level']}
            
            Provide:
            1. Clear pronunciation guidance
            2. Break the word into syllables
            3. A simple memory trick if possible
            4. A friendly example sentence
            Keep responses encouraging and child-friendly.
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
                "context": "tutor",
                "subject": "spelling",
                "auto_continue": True
            }
            
        except Exception as e:
            print(f"Error in spelling tutor: {e}")
            return self._generate_fallback_response()

    async def _handle_definition_request(self, text: str, context_info: Dict) -> Dict:
        """Handle vocabulary definition requests"""
        try:
            system_prompt = f"""
            You are a vocabulary tutor for children.
            Current level: {self.learning_progress['difficulty_level']}
            
            Provide:
            1. A child-friendly definition
            2. A simple example sentence
            3. A related word they might know
            4. A fun fact about the word if possible
            Keep explanations simple and engaging.
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
                "context": "tutor",
                "subject": "vocabulary",
                "auto_continue": True
            }
            
        except Exception as e:
            print(f"Error in vocabulary tutor: {e}")
            return self._generate_fallback_response()

    async def _handle_practice_request(self, text: str, context_info: Dict) -> Dict:
        """Generate a practice exercise"""
        try:
            level = self.learning_progress["difficulty_level"]
            topic = random.choice(self.difficulty_levels[level]["topics"])
            
            system_prompt = f"""
            Create a fun vocabulary practice exercise.
            Level: {level}
            Topic: {topic}
            
            Include:
            1. A word to spell or define
            2. A hint or clue
            3. An encouraging prompt
            Make it fun and interactive!
            """
            
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate a practice exercise"}
                ],
                temperature=0.8
            )
            
            return {
                "text": response.choices[0].message.content,
                "context": "tutor",
                "subject": "practice",
                "auto_continue": True
            }
            
        except Exception as e:
            print(f"Error generating practice: {e}")
            return self._generate_fallback_response()

    async def _generate_vocabulary_exercise(self, context_info: Dict) -> Dict:
        """Generate a vocabulary exercise based on current level"""
        level = self.learning_progress["difficulty_level"]
        topic = random.choice(self.difficulty_levels[level]["topics"])
        
        try:
            system_prompt = f"""
            Create an engaging vocabulary exercise.
            Level: {level}
            Topic: {topic}
            Age Range: {self.difficulty_levels[level]['age_range']}
            Word Length: {self.difficulty_levels[level]['word_length']}
            
            Format:
            1. Introduce a new word
            2. Provide a child-friendly definition
            3. Use it in a fun sentence
            4. Ask them to try using it
            Make it interactive and encouraging!
            """
            
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a {topic} vocabulary exercise"}
                ],
                temperature=0.8
            )
            
            return {
                "text": response.choices[0].message.content,
                "context": "tutor",
                "subject": "vocabulary",
                "topic": topic,
                "auto_continue": True
            }
            
        except Exception as e:
            print(f"Error generating exercise: {e}")
            return self._generate_fallback_response()

    def _generate_fallback_response(self) -> Dict:
        """Generate a safe fallback response"""
        return {
            "text": "Let's try something a bit simpler. Would you like to practice spelling or learn a new word?",
            "context": "tutor",
            "auto_continue": True
        }

skill_manifest = {
    "name": "tutor",
    "description": "Spelling and vocabulary tutoring for children",
    "intents": [
        "spell_word",
        "define_word", 
        "practice_vocabulary",
        "word_help"
    ],
    "topics": [
        "spelling",
        "vocabulary",
        "word_meanings",
        "pronunciation"
    ]
} 