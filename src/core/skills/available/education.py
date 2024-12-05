from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings
import random
import time

"""
Education Skill
--------------
Handles educational queries with context awareness and conversation flow.

Key Features:
1. Context Awareness:
   - Maintains topic context
   - Builds on previous answers
   - Tracks learning progress

2. Conversation Management:
   - Handles follow-up questions
   - Maintains educational focus
   - Encourages exploration

3. Content Generation:
   - Uses OpenAI for responses
   - Adapts to user level
   - Provides relevant examples

4. Topic Transitions:
   - Handles related topics
   - Smooth context switches
   - Maintains learning flow

Usage:
    education = EducationSkill()
    response = await education.handle(text, context)
"""

class EducationSkill:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.conversation_context = []
        self.learning_manager = None  # Will be set by router
        
    async def handle(self, text: str, context_info: Dict) -> Dict:
        """Handle educational queries with better context awareness"""
        try:
            print("\nEducation Skill Processing:")
            print(f"Question: {text}")
            print(f"Context: {context_info}")
            
            # Generate response
            print("\nGenerating response...")
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": """
                    You are a friendly educational assistant.
                    Provide clear, engaging answers that encourage learning.
                    Use examples and analogies when helpful.
                    End with a gentle prompt for follow-up questions.
                    Keep responses concise but informative.
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=self.settings.MODEL_TEMPERATURE,
                max_tokens=self.settings.MODEL_MAX_TOKENS
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"\nGenerated answer: {answer}")
            
            # Store in context
            self.conversation_context.append({
                "text": answer,
                "timestamp": time.time(),
                "type": "response"
            })
            
            return {
                "text": answer,
                "context": "education",
                "auto_continue": True,
                "conversation_active": True,
                "context_info": {
                    "history": self.conversation_context,
                    "current_topic": context_info.get("current"),
                    "entities": context_info.get("entities", {})
                }
            }
            
        except Exception as e:
            print(f"\n❌ Error in education handler: {e}")
            return {
                "text": "I'm having trouble understanding. Could you rephrase your question?",
                "context": "education",
                "auto_continue": True,
                "conversation_active": True
            }
    
    def _is_followup_question(self, text: str) -> bool:
        """Detect if text is a follow-up question"""
        followup_indicators = [
            "what about",
            "how about",
            "and",
            "what else",
            "tell me more",
            "can you",
            "what are",
            "is there",
            "are there",
            "do they",
            "does it",
            "why does",
            "when did",
            "where is",
            "i don't"
        ]
        
        # Check for pronouns without context
        pronouns = ["it", "they", "them", "those", "that", "these", "this"]
        
        return (
            any(text.lower().startswith(ind) for ind in followup_indicators) or
            any(f" {p} " in f" {text.lower()} " for p in pronouns) or
            len(text.split()) < 5  # Short questions are often follow-ups
        )

skill_manifest = {
    "name": "education",
    "description": "Educational answers with conversation context",
    "intents": ["education", "answer_question", "explain_topic"]
} 