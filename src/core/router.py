from typing import Dict, List
from openai import AsyncOpenAI
from .config import Settings
from .logger import ConversationLogger
from .context_manager import ContextManager

class AssistantRouter:
    """
    Assistant Router System
    ----------------------
    Routes user requests to appropriate skills using context and intent.

    Key Features:
    1. Intent Detection:
       - Scores text against available intents
       - Uses keyword matching and context
       - Handles ambiguous cases

    2. Skill Management:
       - Loads available skills dynamically
       - Manages skill initialization
       - Handles skill transitions

    3. Context Integration:
       - Uses ContextManager for state
       - Maintains conversation flow
       - Handles context transitions

    4. Error Handling:
       - Graceful fallbacks
       - Error logging
       - Recovery strategies

    Usage:
        router = AssistantRouter()
        response = await router.route_request(text)
    """
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.skills = {}  # Will be populated with available skills
        self.context_manager = ContextManager()
        
        # Load available skills
        self._load_skills()

        # Set up conversation logging
        self.logger = ConversationLogger()

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

    async def _generate_contextual_response(self, text: str) -> str:
        """Generate helpful contextual response using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
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
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating contextual response: {e}")
            return "I'd love to help you learn about that. Could you give me more details about what you'd like to know?"

    async def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text with context awareness"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
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
                temperature=0.3,
                max_tokens=50
            )
            
            entities = response.choices[0].message.content.strip().split(',')
            return [e.strip().lower() for e in entities if e.strip()]
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []