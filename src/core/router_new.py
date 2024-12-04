from typing import Dict, List
from openai import AsyncOpenAI
from .config import Settings
from .logger import ConversationLogger
from .context_manager import ContextManager
from .learning_manager import LearningManager
from .intent_learner import IntentLearner
import yaml
from pathlib import Path

class AssistantRouter:
    def __init__(self, user_manager=None):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.user_manager = user_manager
        
        # Load skills config
        self.skills_config = self._load_skills_config()
        
        # Initialize learning components
        self.learning_manager = LearningManager() if self.settings.learning.enabled else None
        self.intent_learner = IntentLearner() if self.settings.learning.enabled else None
        self.context_manager = ContextManager(self.learning_manager)
        
        # Initialize skills dictionary
        self.skills = {}
        
        # Load available skills
        self._load_skills()
        self.logger = ConversationLogger()

    def _load_skills_config(self) -> Dict:
        """Load skills configuration"""
        try:
            config_path = Path("config/skills_config.yaml")
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading skills config: {e}")
            return {}

    def _load_skills(self):
        """Load available skills from skills directory"""
        try:
            from .skills.available.education import Education, skill_manifest as education_manifest
            from .skills.available.story import Story, skill_manifest as story_manifest
            # Add more skills imports here
            
            # Initialize skills with manifests
            self.skills["education"] = Education()
            self.skills["story"] = Story()
            # Add more skills here
            
        except Exception as e:
            print(f"Error loading skills: {e}")

    def _score_intents(self, text: str, context_info: Dict = None) -> Dict[str, float]:
        """Score intents using skills config"""
        # Get base scores
        scores = self.skills_config.get("intents", {}).get("base_scores", {}).copy()
        if not scores:
            scores = {
                "story": 0.0,
                "education": 0.0,
                "tutor": 0.0,
                "timer": 0.0,
                "conversation": 0.0
            }
        text_lower = text.lower()
        
        # Apply pattern matching with config weights
        patterns = self.skills_config.get("intents", {}).get("patterns", {})
        weights = self.skills_config.get("intents", {}).get("weights", {})
        for intent, config in patterns.items():
            if any(pattern in text_lower for pattern in config.get("keywords", [])):
                scores[intent] += weights.get(intent, 1.0)
        
        # Apply context bonus if matching current context
        if context_info and context_info.get("current"):
            context_bonus = self.skills_config.get("intents", {}).get("thresholds", {}).get("context_bonus", 1.0)
            scores[context_info["current"]] += context_bonus
        
        # Log scoring details
        print("\n=== Intent Detection ===")
        print(f"Input: '{text}'")
        print("Intent Scores:")
        for intent, score in scores.items():
            print(f"  {intent:12s}: {score:.2f}")
        
        # Record for learning
        if self.intent_learner:
            self.intent_learner.learn_from_interaction(text, max(scores.items(), key=lambda x: x[1])[0], True)
        
        return scores

    async def route_request(self, text: str, context: Dict) -> Dict:
        """Route request with learning integration"""
        try:
            user_id = context.get("user_id", "default")
            user_preferences = self.user_manager.get_user_preferences(user_id) if self.user_manager else {}
            
            # Add user context to request
            context["user"] = {
                "preferences": user_preferences,
                "restrictions": self.user_manager.get_active_restrictions(user_id) if self.user_manager else []
            }
            
            # Get current context
            current_context = await self.context_manager.get_context("current")
            
            # Extract entities with context
            entities = await self._extract_entities(text)
            
            # Score intents with learning
            intent_scores = self._score_intents(text, current_context)
            intent_name = max(intent_scores.items(), key=lambda x: x[1])[0]
            
            # Update context
            context_info = await self.context_manager.update_context(text, entities, intent_name)
            
            # Use context to determine skill
            skill_name = context_info.get("current") or intent_name
            
            # Log the flow
            self.logger.log_intent_detection(text, intent_scores, skill_name)
            
            # Get response from skill
            if skill_name in self.skills:
                response = await self.skills[skill_name].handle(text, context_info)
                
                # Record successful interaction for learning
                if self.learning_manager:
                    await self.learning_manager.record_exchange({
                        "text": text,
                        "intent": intent_name,
                        "skill": skill_name,
                        "context": context_info,
                        "success": True,
                        "entities": entities
                    })
                    
                return response
            else:
                # Generate contextual response
                response = await self._generate_contextual_response(text)
                return {
                    "text": response,
                    "context": "education",
                    "auto_continue": True,
                    "conversation_active": True
                }
                
        except Exception as e:
            print(f"Error in router: {e}")
            return {
                "text": "I'd be happy to help you learn about that. Could you rephrase your question?",
                "context": "education",
                "auto_continue": True,
                "conversation_active": True
            }

    async def _generate_contextual_response(self, text: str) -> str:
        """Generate helpful contextual response using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": """
                    You are a friendly educational assistant.
                    The user has asked a question that needs clarification.
                    Generate a response that:
                    1. Acknowledges their interest in the topic
                    2. Asks for specific details about what they want to learn
                    3. Encourages them to ask follow-up questions
                    Keep the response educational and engaging.
                    """},
                    {"role": "user", "content": f"User asked: {text}"}
                ],
                temperature=self.settings.MODEL_TEMPERATURE,
                max_tokens=self.settings.MODEL_MAX_TOKENS
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating contextual response: {e}")
            return "I'd love to help you learn about that. Could you give me more details about what you'd like to know?"

    async def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text with context awareness"""
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
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
                temperature=self.settings.MODEL_TEMPERATURE,
                max_tokens=self.settings.MODEL_MAX_TOKENS
            )
            
            entities = response.choices[0].message.content.strip().split(',')
            return [e.strip().lower() for e in entities if e.strip()]
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return [] 