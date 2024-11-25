from typing import Dict
from openai import AsyncOpenAI
from ...config import Settings
import random
import time

class Education:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        # Conversation state
        self.conversation_active = False
        self.last_interaction_time = 0
        self.conversation_context = []
        self.CONVERSATION_TIMEOUT = 10.0  # Seconds of silence before conversation ends
        
    async def handle(self, text: str) -> Dict:
        """Handle educational questions with context"""
        try:
            current_time = time.time()
            
            # Check if this is a new conversation
            if not self.conversation_active or (current_time - self.last_interaction_time) > self.CONVERSATION_TIMEOUT:
                self.conversation_active = True
                self.conversation_context = []
                print("\n=== Starting New Educational Conversation ===")
            else:
                print("\n=== Continuing Educational Conversation ===")
                print(f"Context length: {len(self.conversation_context)} exchanges")
            
            # Update interaction time
            self.last_interaction_time = current_time
            
            # Check for incomplete question
            if text.endswith('...') or len(text.split()) < 3:
                return {
                    "text": "I didn't catch your full question. Could you please repeat it?",
                    "context": "education",
                    "auto_continue": True,
                    "conversation_active": True
                }
            
            # Create educational prompt with context
            system_prompt = """
            You are a friendly educational assistant for children.
            Provide engaging, clear answers that encourage further questions.
            Keep responses concise but informative.
            If the question relates to previous context, use that information.
            End responses with a gentle prompt for follow-up questions when appropriate.
            """
            
            # Build messages with context
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation context
            for exchange in self.conversation_context[-3:]:  # Last 3 exchanges
                messages.extend([
                    {"role": "user", "content": exchange["question"]},
                    {"role": "assistant", "content": exchange["answer"]}
                ])
            
            # Add current question
            messages.append({"role": "user", "content": text})
            
            # Get response
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Store in context
            self.conversation_context.append({
                "question": text,
                "answer": answer,
                "timestamp": current_time
            })
            
            # Check if conversation should continue
            should_continue = len(answer.split()) > 3 and "?" in answer
            
            return {
                "text": answer,
                "context": "education",
                "auto_continue": should_continue,  # Allow follow-ups if answer ends with question
                "conversation_active": True
            }
            
        except Exception as e:
            print(f"Error in education handler: {e}")
            self.conversation_active = False  # Reset on error
            return {
                "text": "I'm not sure about that. Would you like to try asking another way?",
                "context": "error",
                "auto_continue": False
            }
    
    def check_conversation_timeout(self) -> bool:
        """Check if conversation has timed out"""
        if not self.conversation_active:
            return False
        
        time_since_last = time.time() - self.last_interaction_time
        if time_since_last > self.CONVERSATION_TIMEOUT:
            print(f"\nEducation conversation timed out after {time_since_last:.1f}s")
            self.conversation_active = False
            return True
        return False

skill_manifest = {
    "name": "education",
    "description": "Educational answers with conversation context",
    "intents": ["education", "answer_question", "explain_topic"]
} 