from openai import AsyncOpenAI
from typing import Dict
from .config import Settings
import json
import time

class AssistantRouter:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        # Add conversation context tracking
        self.conversation_context = {
            "active": False,
            "topic": None,
            "last_question": None,
            "last_answer": None,
            "follow_up_window": 10.0,  # Seconds to wait for follow-up
            "last_interaction": 0
        }
        
        # Define available skills
        self.available_skills = {
            "interactive_story": {
                "description": "Interactive storytelling including reading, creating, and modifying stories"
            },
            "education": {
                "description": "Educational content and learning activities"
            }
        }
    
    async def route_request(self, user_input: str, time_of_day: str = "evening") -> Dict:
        """Route user request using chat completion"""
        try:
            print("\n=== Assistant Router ===")
            print(f"Processing request: '{user_input}'")
            print(f"Time of day: {time_of_day}")
            
            input_lower = user_input.lower()
            
            # Story requests - handle first
            if any(phrase in input_lower for phrase in ["tell me a story", "story about"]):
                print("✓ Story request detected")
                theme = input_lower.split("about", 1)[1].strip().rstrip('?') if "about" in input_lower else ""
                print(f"✓ Story theme: {theme}")
                
                # Return story intent without ending conversation
                return {
                    "skill": "interactive_story",
                    "intent": "tell_story",
                    "mode": "reading",
                    "text": user_input,
                    "parameters": {"theme": theme}
                }
            
            # Educational questions
            if any(input_lower.startswith(word) for word in ["what", "where", "when", "why", "how", "is", "are", "can", "tell"]):
                print("✓ Educational question detected")
                # Start new conversation context
                self.conversation_context.update({
                    "active": True,
                    "topic": self._extract_topic(input_lower),
                    "last_question": user_input,
                    "last_interaction": time.time()
                })
                return {
                    "skill": "education",
                    "intent": "answer_question",
                    "mode": "informative",
                    "parameters": {"question": user_input}
                }
            
            # If no direct match, treat as educational query
            print("✓ Treating as educational query")
            self.conversation_context.update({
                "active": True,
                "topic": self._extract_topic(input_lower),
                "last_question": user_input,
                "last_interaction": time.time()
            })
            return {
                "skill": "education",
                "intent": "answer_question",
                "mode": "informative",
                "parameters": {"question": user_input}
            }
            
        except Exception as e:
            print(f"✗ Error in router: {e}")
            return {
                "skill": "education",
                "intent": "answer_question",
                "mode": "informative",
                "parameters": {"question": user_input}
            }
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from question"""
        # Simple topic extraction - can be made more sophisticated
        common_words = {"what", "where", "when", "why", "how", "is", "are", "the", "a", "an"}
        words = text.lower().split()
        # Get first noun-like word not in common words
        for word in words:
            if word not in common_words:
                return word
        return words[-1]  # Fallback to last word
    
    def _reset_conversation_context(self):
        """Reset conversation context"""
        self.conversation_context.update({
            "active": False,
            "topic": None,
            "last_question": None,
            "last_answer": None,
            "last_interaction": 0
        })
    
    def update_context_with_answer(self, answer: str):
        """Update context with latest answer"""
        self.conversation_context["last_answer"] = answer
        self.conversation_context["last_interaction"] = time.time()