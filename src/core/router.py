"""
Messi Assistant Router
--------------------
Consolidated router implementation that handles:
1. Intent detection and routing
2. Context management
3. User preferences
4. Skill handling
5. Learning integration
"""

from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
from .config import Settings
from .logger import ConversationLogger
from .context_manager import ContextManager
from .learning_manager import LearningManager
import json
import uuid
from pathlib import Path
from datetime import datetime
import traceback
import yaml

class Router:
    def __init__(self, settings: Settings, learning_manager=None, user_manager=None):
        """Initialize router with all dependencies"""
        self.settings = settings
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.learning_manager = learning_manager
        self.user_manager = user_manager
        self.logger = ConversationLogger()
        
        # Initialize state
        self.skills = {}
        self.current_conversation_id = None
        self.context = "general"
        self.history = []
        
        # Load configurations
        self.skills_config = self._load_skills_config()
        
        # Set up paths
        self.context_file = Path(settings.CACHE_DIR) / "context" / "context.json"
        self.context_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.context_manager = ContextManager(learning_manager=self.learning_manager)
        self._initialize_skills()
        self._load_context()
        
    def _load_skills_config(self) -> Dict:
        """Load skills configuration"""
        try:
            config_path = Path(self.settings.BASE_DIR) / "config" / "skills_config.yaml"
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading skills config: {e}")
            return {}

    def _initialize_skills(self):
        """Initialize available skills"""
        try:
            from .skills.available.education import EducationSkill
            from .skills.available.conversation import ConversationSkill
            
            self.skills["education"] = EducationSkill(self.settings)
            self.skills["conversation"] = ConversationSkill(self.settings)
            
        except Exception as e:
            self.logger.error(f"Error initializing skills: {e}")
            traceback.print_exc()

    def _load_context(self):
        """Load conversation context"""
        try:
            if self.context_file.exists():
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    self.context = data.get('context', self.context)
                    self.history = data.get('history', [])
                    while len(self.history) > self.skills_config["intents"]["max_history"]:
                        self.history.pop(0)
        except Exception as e:
            self.logger.error(f"Error loading context: {e}")

    def _save_context(self):
        """Save conversation context"""
        try:
            data = {
                'context': self.context,
                'history': self.history,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.context_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving context: {e}")

    async def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.models.chat,
                messages=[
                    {"role": "system", "content": """
                    Extract key entities from the text.
                    Focus on:
                    - Topics (food, culture, etc.)
                    - Places (countries, cities)
                    - Concepts (history, science)
                    - Objects (specific items)
                    Return as comma-separated list.
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=self.settings.models.temperature,
                max_tokens=self.settings.models.max_tokens
            )
            
            entities = response.choices[0].message.content.strip().split(',')
            return [e.strip().lower() for e in entities if e.strip()]
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []

    def _score_intents(self, text: str, context_info: Dict = None) -> Dict[str, float]:
        """Score intents using configuration"""
        scores = self.skills_config.get("intents", {}).get("base_scores", {}).copy()
        if not scores:
            scores = {
                "education": 0.5,
                "conversation": 0.3
            }
            
        text_lower = text.lower()
        
        # Apply pattern matching
        patterns = self.skills_config.get("intents", {}).get("patterns", {})
        weights = self.skills_config.get("intents", {}).get("weights", {})
        
        for intent, config in patterns.items():
            if any(pattern in text_lower for pattern in config.get("keywords", [])):
                scores[intent] += weights.get(intent, 1.0)
        
        # Apply context bonus
        if context_info and context_info.get("current"):
            context_bonus = self.skills_config.get("intents", {}).get("thresholds", {}).get("context_bonus", 1.0)
            scores[context_info["current"]] += context_bonus
        
        # Log scoring
        self.logger.debug(f"\nIntent Scores for '{text}':")
        for intent, score in scores.items():
            self.logger.debug(f"  {intent:12s}: {score:.2f}")
            
        return scores

    async def route_request(self, text: str, context: Dict) -> Dict:
        """Route request to appropriate skill"""
        try:
            # Get user context
            user_id = context.get("user_id", "default")
            user_preferences = self.user_manager.get_user_preferences(user_id) if self.user_manager else {}
            
            # Extract entities
            entities = await self._extract_entities(text)
            
            # Score intents
            intent_scores = self._score_intents(text, {"current": self.context})
            intent_name = max(intent_scores.items(), key=lambda x: x[1])[0]
            
            # Update context
            context_info = {
                "current": self.context,
                "history": self.history[-3:],
                "entities": entities,
                "user": {
                    "preferences": user_preferences,
                    "restrictions": self.user_manager.get_active_restrictions(user_id) if self.user_manager else []
                }
            }
            
            # Route to skill
            if intent_name in self.skills:
                try:
                    response = await self.skills[intent_name].handle(text, context_info)
                    
                    # Record interaction
                    if self.learning_manager:
                        await self.learning_manager.record_exchange({
                            "text": text,
                            "intent": intent_name,
                            "context": context_info,
                            "success": True
                        })
                    
                    # Update context if response changes it
                    if response and "context" in response:
                        self.context = response["context"]
                        self._save_context()
                    
                    return response
                except Exception as e:
                    self.logger.error(f"Error in skill {intent_name}: {e}")
                    return {
                        "text": "I'm having trouble processing that. Could you try asking in a different way?",
                        "context": self.context
                    }
            else:
                # Fallback to conversation skill
                try:
                    response = await self.skills["conversation"].handle(text, context_info)
                    return response
                except Exception as e:
                    self.logger.error(f"Error in fallback conversation: {e}")
                    return {
                        "text": "I'd be happy to help you learn about that. Could you rephrase your question?",
                        "context": "conversation"
                    }
                
        except Exception as e:
            self.logger.error(f"Error routing request: {e}")
            return {
                "text": "I'm having trouble understanding. Could you try asking in a different way?",
                "context": self.context or "general"
            }

    async def get_response(self, text: str, is_follow_up: bool = False) -> str:
        """Get response for user input"""
        try:
            # Format input for skills
            intent = {
                "text": text,
                "context": {
                    "is_follow_up": is_follow_up,
                    "current": self.context
                }
            }
            
            # Default to conversation skill for general context
            if self.context == "general":
                response = await self.skills["conversation"].handle(intent)
                return response["text"]
                
            # Use context-specific skill if available
            if self.context in self.skills:
                response = await self.skills[self.context].handle(intent)
                return response["text"]
                
            # Fallback to conversation
            response = await self.skills["conversation"].handle(intent)
            return response["text"]
            
        except Exception as e:
            self.logger.error(f"Error routing request: {self.context}")
            return "I'm having trouble understanding. Could you try asking in a different way?"