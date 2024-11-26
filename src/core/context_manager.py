"""
Enhanced Context Management System
--------------------------------
Manages conversation context, persistence, and transitions between topics.

Key Features:
1. Context Tracking:
   - Maintains current context and history
   - Tracks entities and topics
   - Handles context timeouts and transitions

2. Persistence:
   - Saves context to disk
   - Auto-saves periodically
   - Maintains global entity memory

3. Context Relationships:
   - Defines related contexts (e.g., france -> french_food)
   - Maps entities to contexts
   - Tracks topic relationships

4. Conversation Flow:
   - Manages conversation state
   - Handles timeouts
   - Provides context history

Usage:
    context_manager = ContextManager()
    context = await context_manager.update_context(text, entities, intent)
    context_info = await context_manager.get_context(skill)
"""

from typing import Dict, List, Optional, Set, TYPE_CHECKING
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio

if TYPE_CHECKING:
    from .learning_manager import LearningManager

class ContextManager:
    def __init__(self, learning_manager: 'LearningManager'):
        self.learning_manager = learning_manager
        self.learning_enabled = learning_manager.learning_enabled if learning_manager else False
        
        # Core context tracking
        self.current_context = None
        self.context_history = []
        self.conversation_start = None
        self.last_interaction = None
        
        # Timeouts and limits
        self.CONTEXT_TIMEOUT = 60  # seconds
        self.MAX_ENTITIES = 20
        self.MAX_HISTORY = 10
        
        # Entity and topic tracking
        self.current_entities = set()
        self.current_topics = set()
        self.global_entities = set()
        
        # Persistent storage
        self.context_file = Path("cache/context.json")
        self.auto_save = True
        
        # Context relationships
        self.context_transitions = {
            "france": {
                "related": ["french_food", "french_culture", "paris"],
                "entities": ["france", "paris", "french"],
                "topics": ["cuisine", "culture", "travel"]
            },
            "food": {
                "related": ["cuisine", "restaurants", "chefs"],
                "entities": ["restaurant", "food", "meal"],
                "topics": ["cooking", "dining", "recipes"]
            },
            # Add more context definitions
        }
        
        # Load persistent data
        self._load_persistent_context()
        
        # Core context types
        self.context_types = {
            "education": {
                "related": ["learning", "questions", "facts"],
                "entities": ["topics", "concepts", "facts"],
                "transitions": ["story", "conversation"]
            },
            "story": {
                "related": ["narrative", "characters", "plot"],
                "entities": ["characters", "settings", "events"],
                "transitions": ["education", "conversation"]
            },
            "conversation": {
                "related": ["chat", "discussion", "dialogue"],
                "entities": ["topics", "participants", "mood"],
                "transitions": ["education", "story"]
            }
        }
        
        # Skill-specific context handlers
        self.context_handlers = {
            "education": self._handle_education_context,
            "story": self._handle_story_context,
            "conversation": self._handle_conversation_context
        }
        
        # Add default weights
        self.default_weights = {
            "previous_context": 0.4,
            "current_entities": 0.3,
            "user_engagement": 0.3
        }
    
    async def initialize(self):
        """Initialize async components of the context manager"""
        if self.auto_save:
            asyncio.create_task(self._auto_save_context())
    
    async def update_context(self, text: str, entities: List[str], skill: str) -> Dict:
        """Update context with skill-specific handling"""
        try:
            # Initialize config with defaults
            config = {
                "thresholds": {
                    "context_switch_threshold": 0.7
                }
            }
            
            if self.learning_enabled and self.learning_manager:
                # Get learned patterns
                patterns = await self.learning_manager.get_learned_patterns(skill)
                config = await self.learning_manager.get_current_config()
                
                # Use learned weights
                weights = config["optimizations"]["context_weights"]
                context_score = self._calculate_context_score(text, entities, weights)
            else:
                # Use default weights
                context_score = self._calculate_context_score(text, entities, self.default_weights)
            
            # Use threshold for transitions
            if context_score > config["thresholds"]["context_switch_threshold"]:
                # Update context
                self.current_context = self._determine_new_context(
                    text, entities, patterns["transitions"] if "patterns" in locals() else {}
                )
            
            return self._get_context_info()
            
        except Exception as e:
            print(f"Error updating context: {e}")
            return self._get_default_context()
    
    async def _handle_education_context(self, text: str, entities: List[str]) -> Dict:
        """Handle educational context updates"""
        return {
            "type": "education",
            "topics": self._extract_topics(entities),
            "learning_stage": self._determine_learning_stage(text),
            "allow_transitions": ["story", "conversation"]
        }
    
    async def _handle_story_context(self, text: str, entities: List[str]) -> Dict:
        """Handle story context updates"""
        return {
            "type": "story",
            "narrative_elements": self._extract_narrative(entities),
            "story_stage": self._determine_story_stage(text),
            "allow_transitions": ["education", "conversation"]
        }
    
    async def _handle_conversation_context(self, text: str, entities: List[str]) -> Dict:
        """Handle general conversation context updates"""
        return {
            "type": "conversation",
            "topics": self._extract_topics(entities),
            "mood": self._detect_conversation_mood(text),
            "allow_transitions": ["education", "story"]
        }
    
    def _detect_conversation_mood(self, text: str) -> str:
        """Detect conversation mood/tone"""
        # Simple mood detection
        positive_words = {"happy", "good", "great", "love", "like", "fun"}
        negative_words = {"sad", "bad", "angry", "upset", "hate"}
        
        words = set(text.lower().split())
        
        if words & positive_words:
            return "positive"
        elif words & negative_words:
            return "negative"
        return "neutral"
    
    def _extract_topics(self, entities: List[str]) -> List[str]:
        """Extract conversation topics from entities"""
        # Simple topic extraction
        return [e for e in entities if len(e) > 2]  # Filter very short entities
    
    async def get_context(self, skill: str) -> Dict:
        """Get rich context information"""
        try:
            context_info = {
                "current": self.current_context,
                "history": self.context_history[-self.MAX_HISTORY:],
                "entities": {
                    "current": list(self.current_entities),
                    "global": list(self.global_entities)
                },
                "topics": list(self.current_topics),
                "conversation": {
                    "active": not self._is_context_timed_out(),
                    "duration": (datetime.now() - self.conversation_start).seconds if self.conversation_start else 0,
                    "turns": len(self.context_history)
                },
                "skill_specific": self._get_skill_context(skill)
            }
            
            return context_info
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return self._get_default_context()
    
    def _get_skill_context(self, skill: str) -> Dict:
        """Get skill-specific context"""
        if skill in self.context_transitions:
            info = self.context_transitions[skill]
            return {
                "related_contexts": info["related"],
                "relevant_entities": [
                    e for e in self.current_entities 
                    if e in info["entities"]
                ],
                "active_topics": [
                    t for t in self.current_topics 
                    if t in info["topics"]
                ]
            }
        return {}
    
    async def _auto_save_context(self):
        """Periodically save context"""
        while True:
            await asyncio.sleep(60)  # Save every minute
            if self.auto_save:
                await self._save_persistent_context()

    def start_conversation(self, initial_context: str):
        """Start a new conversation with initial context"""
        self.current_context = initial_context
        self.conversation_start = datetime.now()
        self.last_interaction = datetime.now()
        self.context_history = [(initial_context, datetime.now())]
        
    def _is_context_timed_out(self) -> bool:
        """Check if current context has timed out"""
        if not self.last_interaction:
            return True
        
        elapsed = datetime.now() - self.last_interaction
        return elapsed > timedelta(seconds=self.CONTEXT_TIMEOUT)
    
    def _handle_timeout(self) -> Dict:
        """Handle context timeout"""
        self.current_context = None
        self.current_entities.clear()
        self.current_topics.clear()
        
        return {
            "context": None,
            "history": self.context_history,
            "entities": [],
            "topics": [],
            "maintain_conversation": False
        }
    
    def _update_entities(self, entities: List[str]):
        """Update tracked entities"""
        self.current_entities.update(entities)
        # Prune old entities if needed
        if len(self.current_entities) > self.MAX_ENTITIES:
            self.current_entities = set(list(self.current_entities)[-self.MAX_ENTITIES:])
    
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

    def _calculate_context_score(self, text: str, entities: List[str], weights: Dict[str, float]) -> float:
        """Calculate context score using weights"""
        try:
            previous_score = self._previous_context_score()
            entity_score = self._entity_match_score(entities)
            engagement_score = self._engagement_score()
            
            return (
                weights["previous_context"] * previous_score +
                weights["current_entities"] * entity_score +
                weights["user_engagement"] * engagement_score
            )
        except Exception as e:
            print(f"Error calculating context score: {e}")
            return 0.0
            
    def _previous_context_score(self) -> float:
        """Calculate score based on previous context"""
        return 1.0 if self.current_context else 0.0
        
    def _entity_match_score(self, entities: List[str]) -> float:
        """Calculate score based on entity matches"""
        if not entities or not self.current_entities:
            return 0.0
        matches = len(set(entities) & self.current_entities)
        return matches / len(entities) if entities else 0.0
        
    def _engagement_score(self) -> float:
        """Calculate user engagement score"""
        if not self.conversation_start:
            return 0.0
        duration = (datetime.now() - self.conversation_start).seconds
        return min(1.0, duration / 60.0)  # Max score after 1 minute

    def _determine_new_context(self, text: str, entities: List[str], transitions: Dict[str, float]) -> str:
        """Determine new context based on transitions"""
        try:
            # Use current context if available
            if self.current_context and self.current_context in transitions:
                return self.current_context
                
            # Find best matching context
            best_score = 0.0
            best_context = None
            
            for context, score in transitions.items():
                if score > best_score:
                    best_score = score
                    best_context = context
                    
            return best_context or "general"
            
        except Exception as e:
            print(f"Error determining context: {e}")
            return "general"

    def _get_default_context(self) -> Dict:
        """Get default context when error occurs"""
        return {
            "current": None,
            "history": [],
            "entities": {
                "current": [],
                "global": []
            },
            "topics": [],
            "conversation": {
                "active": False,
                "duration": 0,
                "turns": 0
            },
            "skill_specific": {}
        }

    def _get_context_info(self) -> Dict:
        """Get current context information"""
        return {
            "current": self.current_context,
            "history": self.context_history[-self.MAX_HISTORY:],
            "entities": {
                "current": list(self.current_entities),
                "global": list(self.global_entities)
            },
            "topics": list(self.current_topics),
            "conversation": {
                "active": not self._is_context_timed_out(),
                "duration": (datetime.now() - self.conversation_start).seconds if self.conversation_start else 0,
                "turns": len(self.context_history)
            },
            "skill_specific": self._get_skill_context(self.current_context or "general")
        }

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