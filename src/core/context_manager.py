"""
Context Management System for Messi Assistant
------------------------------------------

This module manages conversation context and state across different skills,
providing persistent memory and adaptive behavior based on user interactions.

Key Features:
    1. Context Management:
        - Skill-specific context tracking
        - Global context across skills
        - Persistent storage between sessions
        - User preference learning
    
    2. Context Types:
        - Story Context: Track story progress and preferences
        - Education Context: Monitor learning progress
        - Tutor Context: Track subject mastery
        - Conversation Context: Maintain chat history
        - Timer Context: Manage active timers
    
    3. Persistence:
        - Automatic context saving
        - State recovery between sessions
        - Periodic updates
        - Error recovery
    
    4. User Tracking:
        - Interaction history
        - Preference learning
        - Entity tracking
        - Session management

Usage:
    context_manager = ContextManager()
    
    # Update context
    await context_manager.update_context("story", {
        "theme": "adventure",
        "characters": ["dragon", "princess"]
    })
    
    # Get context
    context = await context_manager.get_context("story")

File Structure:
    /cache/
        context.json    # Persistent storage file

Context Structure:
    - Global Context:
        - Session information
        - Interaction counts
        - Recent intents
        - User preferences
    
    - Skill Contexts:
        - Story progress
        - Learning history
        - Conversation state
        - Timer status

Integration:
    Works with:
    - AssistantRouter for intent routing
    - SkillManager for skill execution
    - Individual skills for state management

Author: Your Name
Created: 2024-01-24
"""

from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import json
from pathlib import Path

class ContextManager:
    def __init__(self):
        self.contexts = {
            "story": StoryContext(),
            "education": EducationContext(),
            "tutor": TutorContext(),
            "conversation": ConversationContext(),
            "timer": TimerContext()
        }
        
        # Global context tracking
        self.global_context = {
            "session_start": datetime.now(),
            "interaction_count": 0,
            "recent_intents": [],
            "mentioned_entities": set(),
            "user_preferences": {}
        }
        
        # Context persistence
        self.context_file = Path("cache/context.json")
        self._load_persistent_context()

    async def update_context(self, skill: str, data: Dict):
        """Update context for a specific skill"""
        if skill in self.contexts:
            await self.contexts[skill].update(data)
        
        # Update global context
        self.global_context["interaction_count"] += 1
        self.global_context["recent_intents"].append(skill)
        if len(self.global_context["recent_intents"]) > 10:
            self.global_context["recent_intents"].pop(0)
            
        # Extract entities from interaction
        if "entities" in data:
            self.global_context["mentioned_entities"].update(data["entities"])
        
        # Save context periodically
        if self.global_context["interaction_count"] % 5 == 0:
            self._save_persistent_context()

    async def get_context(self, skill: str) -> Dict:
        """Get context for a specific skill"""
        skill_context = await self.contexts[skill].get() if skill in self.contexts else {}
        
        return {
            "skill_specific": skill_context,
            "global": {
                "interaction_count": self.global_context["interaction_count"],
                "session_duration": (datetime.now() - self.global_context["session_start"]).seconds,
                "recent_intents": self.global_context["recent_intents"][-5:],
                "mentioned_entities": list(self.global_context["mentioned_entities"]),
                "user_preferences": self.global_context["user_preferences"]
            }
        }

    def _save_persistent_context(self):
        """Save context to file"""
        try:
            context_data = {
                "global": {
                    k: list(v) if isinstance(v, set) else v 
                    for k, v in self.global_context.items() 
                    if k != "session_start"
                },
                "skills": {
                    name: context.get_persistent_data()
                    for name, context in self.contexts.items()
                }
            }
            
            self.context_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.context_file, 'w') as f:
                json.dump(context_data, f)
                
        except Exception as e:
            print(f"Error saving context: {e}")

    def _load_persistent_context(self):
        """Load context from file"""
        try:
            if self.context_file.exists():
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    
                    # Restore global context
                    self.global_context.update(data.get("global", {}))
                    self.global_context["mentioned_entities"] = set(
                        self.global_context.get("mentioned_entities", [])
                    )
                    
                    # Restore skill contexts
                    for skill, context_data in data.get("skills", {}).items():
                        if skill in self.contexts:
                            self.contexts[skill].restore_persistent_data(context_data)
                            
        except Exception as e:
            print(f"Error loading context: {e}")

class BaseContext:
    def __init__(self):
        self.last_update = datetime.now()
        self.data = {}
    
    async def update(self, new_data: Dict):
        self.data.update(new_data)
        self.last_update = datetime.now()
    
    async def get(self) -> Dict:
        return self.data
    
    def get_persistent_data(self) -> Dict:
        """Get data that should be saved between sessions"""
        return self.data
    
    def restore_persistent_data(self, data: Dict):
        """Restore saved context data"""
        self.data = data

class StoryContext(BaseContext):
    def __init__(self):
        super().__init__()
        self.data = {
            "current_story": None,
            "current_position": 0,
            "story_history": [],
            "favorite_themes": set(),
            "favorite_characters": set(),
            "engagement_levels": []
        }
    
    async def update(self, new_data: Dict):
        await super().update(new_data)
        
        # Track favorites
        if "theme" in new_data:
            self.data["favorite_themes"].add(new_data["theme"])
        if "characters" in new_data:
            self.data["favorite_characters"].update(new_data["characters"])
        
        # Track engagement
        if "engagement_level" in new_data:
            self.data["engagement_levels"].append({
                "level": new_data["engagement_level"],
                "timestamp": datetime.now().isoformat()
            })

class EducationContext(BaseContext):
    def __init__(self):
        super().__init__()
        self.data = {
            "topics_covered": set(),
            "difficulty_level": "beginner",
            "learning_style": "visual",
            "interest_areas": set(),
            "question_history": []
        }

class TutorContext(BaseContext):
    def __init__(self):
        super().__init__()
        self.data = {
            "current_subject": None,
            "skill_levels": {},
            "practice_history": [],
            "learning_pace": "medium",
            "problem_areas": set()
        }

class ConversationContext(BaseContext):
    def __init__(self):
        super().__init__()
        self.data = {
            "conversation_history": [],
            "mood_tracking": [],
            "favorite_topics": set(),
            "interaction_style": "friendly",
            "response_length_preference": "medium"
        }

class TimerContext(BaseContext):
    def __init__(self):
        super().__init__()
        self.data = {
            "active_timers": {},
            "common_durations": {},
            "last_timer": None,
            "timer_history": []
        } 