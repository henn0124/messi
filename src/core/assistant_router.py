"""
Assistant Router for Messi Assistant
----------------------------------

This module handles the routing and processing of user requests based on detected intents.
It serves as the central decision-making component for understanding and responding to
user interactions.

Key Features:
    1. Intent Detection:
        - Pattern-based intent recognition
        - Multi-category intent scoring
        - Confidence level calculation
        - Learning from interactions
    
    2. Request Routing:
        - Story generation
        - Educational responses
        - Tutoring assistance
        - Task handling
        - Timer management
        - Emotional support
    
    3. Context Management:
        - Conversation state tracking
        - Topic relationships
        - Entity tracking
        - Follow-up detection
    
    4. Response Generation:
        - Context-aware prompts
        - Cache-aware responses
        - Adaptive conversation flow
        - Age-appropriate content

Components:
    - Intent Patterns: Defined patterns for each type of interaction
    - Topic Relationships: Connected concepts for context
    - Conversation Context: Active state of interaction
    - Response Cache: Previously generated responses
    - Intent Learner: System for improving intent detection

Usage:
    router = AssistantRouter()
    
    # Detect intent
    intent = await router.detect_intent("tell me a story about a dragon")
    
    # Route request
    response = await router.route_request(user_text)

Integration:
    Works with:
    - IntentLearner for continuous improvement
    - ResponseCache for efficient responses
    - ConversationManager for context
    - Various skill-specific handlers

Author: Your Name
Created: 2024-01-24
"""

from typing import Dict, TYPE_CHECKING, List
from openai import AsyncOpenAI
from .config import Settings
from .logger import ConversationLogger
from .context_manager import ContextManager
from .skills import SkillManager

if TYPE_CHECKING:
    from .context_manager import ContextManager
    from .learning_manager import LearningManager

class AssistantRouter:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        # Initialize managers
        self.context_manager = ContextManager(None)  # Initialize without learning manager for now
        self.logger = ConversationLogger()
        self.skill_manager = SkillManager()
        self.skills = self.skill_manager.skills
        
        # Start with education context
        self.context_manager.start_conversation("education")
        
        # Initialize conversation state
        self.current_context = {
            "current": "education",
            "history": [],
            "entities": {
                "current": [],
                "global": []
            },
            "topics": [],
            "conversation": {
                "active": True,
                "duration": 0,
                "turns": 0
            },
            "skill_specific": {}
        }
        
    async def route_request(self, text: str) -> Dict:
        """Route request to appropriate skill with better error handling"""
        try:
            # Get current context
            current_context = await self.context_manager.get_context("current")
            
            # Extract entities with context
            entities = await self._extract_entities(text)
            
            # Score intents with context
            intent_scores = self._score_intents(text, current_context)
            intent_name = max(intent_scores.items(), key=lambda x: x[1])[0]
            
            # Update context
            context_info = await self.context_manager.update_context(text, entities, intent_name)
            
            # Use context to determine skill
            skill_name = context_info.get("current") or intent_name
            
            # Log the flow
            self.logger.log_intent_detection(text, intent_scores, skill_name)
            
            if skill_name in self.skills:
                return await self.skills[skill_name].handle(text, context_info)
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
            
    def _score_intents(self, text: str, context_info: Dict = None) -> Dict[str, float]:
        """Score intents with context awareness"""
        scores = {
            "story": 0.0,
            "education": 0.0,
            "tutor": 0.0,
            "timer": 0.0,
            "conversation": 0.0
        }
        
        # Context bonus
        if context_info and context_info.get("current"):
            scores[context_info["current"]] += 1.0
        
        # Education keywords
        if any(word in text.lower() for word in ["what", "why", "how", "when", "where", "explain", "tell me about", "learn"]):
            scores["education"] += 1.5
            
        # Story keywords
        if any(word in text.lower() for word in ["story", "tell", "once upon", "happened"]):
            scores["story"] += 1.5
            
        return scores
            
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