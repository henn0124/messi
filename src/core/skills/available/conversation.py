from typing import Dict, List
from openai import AsyncOpenAI
from ...config import Settings
import time

class ConversationSkill:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        self.personality = {
            "tone": "friendly and warm",
            "style": "child-appropriate",
            "complexity": "simple",
            "engagement": "interactive"
        }
        
        # Initialize conversation history
        self.conversation_history = []
        self.MAX_HISTORY = 10  # Maximum number of exchanges to keep
        
    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.MAX_HISTORY:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY:]
    
    async def handle(self, intent: Dict) -> Dict:
        """Handle general conversation"""
        try:
            text = intent.get("text", "")
            context = intent.get("context", {})
            
            system_prompt = f"""
            You are Messi, a knowledgeable and friendly AI assistant for children.
            
            Core Traits:
            - Tone: {self.personality['tone']}
            - Style: {self.personality['style']}
            - Complexity: {self.personality['complexity']}
            
            Key Instructions:
            1. Always provide direct, confident answers to questions
            2. For factual questions (like capitals, dates, or facts), answer immediately and clearly
            3. Keep responses short and child-friendly (1-2 sentences)
            4. If asked about a fact you know (like "What's the capital of Germany?"), answer directly ("The capital of Germany is Berlin.")
            5. Only ask for clarification if the question is genuinely unclear
            6. Be engaging and educational
            7. Never say you can't help or don't know - if unsure, ask a specific clarifying question
            
            Remember: You have extensive knowledge about geography, science, history, and other topics. Use it confidently to help and teach.
            """
            
            # Build messages list with history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for msg in self.conversation_history[-5:]:  # Last 5 exchanges
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current user message
            messages.append({"role": "user", "content": text})
            
            # Get response from OpenAI
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            # Get the response text
            response_text = response.choices[0].message.content.strip()
            
            # Add exchange to history
            self._add_to_history("user", text)
            self._add_to_history("assistant", response_text)
            
            return {
                "text": response_text,
                "context": "conversation",
                "auto_continue": False,
                "conversation_active": True,
                "context_info": {
                    "history": self.conversation_history,
                    "current_topic": context.get("current"),
                    "mentioned_entities": context.get("mentioned_entities", [])
                }
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