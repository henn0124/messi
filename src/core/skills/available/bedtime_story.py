from typing import Dict, Optional
import json
from pathlib import Path
import random
import time
from openai import AsyncOpenAI
from ...config import Settings
import asyncio
import logging
from datetime import datetime
import os

class BedtimeStory:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.story_context = {
            "current_story": None,
            "characters": [],
            "plot_points": [],
            "story_stage": "beginning"  # beginning, middle, end
        }
        
    async def handle(self, text: str, context_info: Dict) -> Dict:
        """Handle story requests with context awareness"""
        try:
            # Check if continuing existing story
            if self.story_context["current_story"] and self._is_continuation_request(text):
                system_prompt = """
                Continue the bedtime story using the existing characters and plot.
                Maintain a gentle, soothing tone suitable for bedtime.
                Include descriptive elements but keep the story moving forward.
                End this part in a way that allows for more if desired.
                """
                
                # Build context for continuation
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": f"Previous story context: {self.story_context['current_story']}"},
                    {"role": "assistant", "content": f"Characters involved: {', '.join(self.story_context['characters'])}"},
                    {"role": "user", "content": text}
                ]
                
            else:
                # Start new story
                system_prompt = """
                Create a soothing bedtime story appropriate for children.
                Use gentle language and positive themes.
                Include vivid but calming descriptions.
                End the first part in a way that invites continuation.
                """
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            
            # Generate story content
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=messages,
                temperature=0.8,  # More creative
                max_tokens=250
            )
            
            story_part = response.choices[0].message.content.strip()
            
            # Update story context
            self._update_story_context(story_part)
            
            return {
                "text": story_part,
                "context": "bedtime_story",
                "auto_continue": True,
                "conversation_active": True,
                "context_info": {
                    "story_context": self.story_context,
                    "allow_continuation": True
                }
            }
            
        except Exception as e:
            print(f"Error in bedtime story handler: {e}")
            return self._generate_fallback_response()
    
    def _is_continuation_request(self, text: str) -> bool:
        """Detect if user wants story to continue"""
        continuation_phrases = [
            "what happened next",
            "and then",
            "continue",
            "what next",
            "go on",
            "tell me more",
            "what did they do",
            "what happens to"
        ]
        return any(phrase in text.lower() for phrase in continuation_phrases)
    
    def _update_story_context(self, story_part: str):
        """Update story context with new content"""
        # Update current story
        if not self.story_context["current_story"]:
            self.story_context["current_story"] = story_part
        else:
            self.story_context["current_story"] += f"\n{story_part}"
        
        # Extract characters (could use NLP here)
        # Update plot points
        # Track story progression
        
    def _generate_fallback_response(self) -> Dict:
        """Generate a safe fallback response"""
        return {
            "text": "Let me think of another part to the story. Would you like to hear what happens next?",
            "context": "bedtime_story",
            "auto_continue": True,
            "conversation_active": True
        }

skill_manifest = {
    "name": "bedtime_story",
    "description": "Interactive bedtime storytelling",
    "intents": ["story", "bedtime", "tell_story"]
} 