from typing import Dict
import re

class IntentProcessor:
    """Simplified intent processor for MVP"""
    
    def __init__(self):
        self.intent_patterns = {
            "tell_story": r"tell .* story|read .* story|story time",
            "continue_story": r"continue|next|go on",
            "pause_story": r"pause|wait|hold on",
            "stop_story": r"stop|end|finish|good night"
        }
    
    async def process(self, text: str) -> Dict:
        """Simple pattern matching for MVP intents"""
        text = text.lower()
        
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text):
                return {
                    "skill": "bedtime_story",
                    "intent": intent,
                    "action": intent.split("_")[0],
                    "confidence": 1.0
                }
        
        return {
            "skill": "bedtime_story",
            "intent": "unknown",
            "action": "none",
            "confidence": 0.0
        }