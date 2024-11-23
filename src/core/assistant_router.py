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

from openai import AsyncOpenAI
from typing import Dict, Optional, Tuple
from .config import Settings
import json
import time
from .cache_manager import ResponseCache
import asyncio
from .intent_learner import IntentLearner
from .skills.available.bedtime_story import BedtimeStory
from .skills import SkillManager
from .context_manager import ContextManager
from datetime import datetime

class AssistantRouter:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        # Enhanced conversation context
        self.conversation_context = {
            "active": False,
            "current_topic": None,
            "subtopics": [],
            "last_question": None,
            "last_answer": None,
            "mentioned_entities": [],
            "follow_up_window": 10.0,
            "last_interaction": 0,
            "conversation_history": []    # Keep last N interactions
        }
        
        # Topic relationships for better context
        self.topic_relationships = {
            "location": {
                "related_aspects": ["culture", "history", "landmarks", "food"],
                "follow_up_patterns": [
                    "what else is in {location}",
                    "tell me more about {location}",
                    "what's famous in {location}"
                ]
            },
            "history": {
                "related_aspects": ["events", "people", "places", "dates"],
                "follow_up_patterns": [
                    "what happened next",
                    "who else was involved",
                    "when did this happen"
                ]
            }
        }
        
        self.response_cache = ResponseCache()
        
        # Enhanced intent detection patterns
        self.intent_patterns = {
            "story": {
                "keywords": [
                    "story", "tale", "tell me", "read", "once upon",
                    "bedtime", "fairy tale"
                ],
                "story_types": [
                    "adventure", "animal", "fairy tale", "bedtime",
                    "funny", "magical", "educational"
                ],
                "characters": [
                    "dragon", "princess", "cat", "dog", "robot",
                    "wizard", "fairy", "unicorn"
                ]
            },
            "education": {
                "keywords": [
                    "what is", "how does", "why does", "explain",
                    "teach me about", "tell me about", "learn about"
                ],
                "subjects": [
                    "science", "nature", "animals", "space",
                    "history", "world", "weather"
                ]
            },
            "tutor": {
                "keywords": [
                    "help me with", "practice", "solve", "how to",
                    "show me how", "teach me to"
                ],
                "subjects": {
                    "math": ["add", "subtract", "multiply", "divide", "equation"],
                    "language": ["spell", "write", "read", "grammar"],
                    "science": ["experiment", "chemistry", "physics"],
                    "programming": ["code", "python", "program"]
                }
            },
            "timer": {
                "keywords": [
                    "timer", "set timer", "remind", "wait", "alarm",
                    "countdown"
                ],
                "units": ["seconds", "minutes", "hours"]
            },
            "conversation": {
                "greetings": [
                    "hello", "hi", "hey", "good morning", "good evening"
                ],
                "farewells": [
                    "goodbye", "bye", "see you", "good night"
                ],
                "gratitude": [
                    "thank you", "thanks", "appreciate"
                ]
            }
        }
        
        # Context tracking
        self.session_context = {
            "last_intent": None,
            "story_mode": False,
            "education_mode": False,
            "consecutive_story_requests": 0,
            "consecutive_education_requests": 0
        }
        
        # Initialize timer storage
        self.active_timers = {}
        self.timer_id = 0
        
        self.intent_learner = IntentLearner()
        self.skill_manager = SkillManager()
        self.story_skill = BedtimeStory()
        self.context_manager = ContextManager()
        
        # Add system prompt
        self.system_prompt = """
        You are a helpful assistant for children. Keep responses:
        1. Simple and clear
        2. Age-appropriate
        3. Educational when possible
        4. Engaging and friendly
        """
    
    async def _create_contextual_prompt(self, user_input: str, intent: str) -> list:
        """Create a context-aware system prompt using cache history"""
        messages = []
        
        # Check cache for relevant context
        cached_context = await self.response_cache.get_relevant_context(
            user_input,
            self.conversation_context["current_topic"],
            self.conversation_context["mentioned_entities"]
        )
        
        # Build system prompt with cache-aware context
        system_content = f"""You are a friendly AI assistant for children.
Current conversation state: {self.conversation_context['current_topic']}
Recent topics discussed: {', '.join(list(self.conversation_context['mentioned_entities'])[-3:])}
Last question asked: {self.conversation_context['last_question']}

Previous relevant information:
{self._format_cached_context(cached_context)}

Maintain conversation flow and use previous context appropriately.
If referring to cached information, acknowledge it naturally.
"""
        
        messages.append({"role": "system", "content": system_content})
        
        # Add recent conversation history
        for interaction in self.conversation_context["conversation_history"][-3:]:
            messages.append({
                "role": "user",
                "content": interaction["input"]
            })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
    
    def _format_cached_context(self, cached_context: list) -> str:
        """Format cached context for prompt inclusion"""
        if not cached_context:
            return "No relevant previous information."
            
        formatted = []
        for entry in cached_context:
            formatted.append(f"- Previously discussed {entry['topic']}: {entry['summary']}")
        
        return "\n".join(formatted)
    
    async def route_request(self, text: str) -> Optional[Dict]:
        """Route the request to appropriate handler"""
        try:
            print("\n▶ Processing request...")
            
            # Get intent scores
            intent_scores = await self.intent_learner.get_intent_scores(text)
            
            print("\nIntent Scores:")
            for intent, score in intent_scores.items():
                print(f"- {intent}: {score:.2f}")
            
            # Get highest scoring intent
            intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            score = intent_scores[intent]
            
            print(f"\nIntent Detection:")
            print(f"Text: '{text}'")
            print(f"Intent: {intent} ({score:.2f})")
            
            # Route to appropriate skill
            skill = self.skill_manager.get_skill(intent)
            if skill:
                return await skill.handle(text)
            else:
                print(f"Skill not found: {intent}")
                print(f"Available skills: {self.skill_manager.list_skills()}")
                return {
                    "text": "I'm not sure how to help with that.",
                    "context": "error"
                }
                
        except Exception as e:
            print(f"Error routing request: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "I'm having trouble processing that request.",
                "context": "error"
            }
    
    async def _route_story_request(self, text: str, intent: Dict) -> Dict:
        """Handle story-specific routing"""
        # Check if we're in the middle of a story
        if hasattr(self, 'current_story') and self.current_story:
            return await self.skill_manager.execute({
                "skill": "story",
                "intent": "continue_story",
                "text": text,
                "parameters": {
                    "current_story": self.current_story,
                    "current_position": self.current_story_position
                }
            })
        
        # Check if we need a theme
        if not intent["parameters"].get("theme"):
            return await self.skill_manager.execute({
                "skill": "story",
                "intent": "ask_theme",
                "text": "What kind of story would you like to hear?"
            })
        
        # Start new story
        return await self.skill_manager.execute({
            "skill": "story",
            "intent": "tell_story",
            "text": text,
            "parameters": intent["parameters"]
        })

    async def _route_timer_request(self, text: str, intent: Dict) -> Dict:
        """Handle timer-specific routing"""
        # Check if we need duration
        if not intent["parameters"].get("duration"):
            return await self.skill_manager.execute({
                "skill": "timer",
                "intent": "ask_duration",
                "text": "How long would you like the timer for?"
            })
        
        return await self.skill_manager.execute({
            "skill": "timer",
            "intent": "set_timer",
            "text": text,
            "parameters": intent["parameters"]
        })

    async def _route_education_request(self, text: str, intent: Dict) -> Dict:
        """Handle education-specific routing"""
        # Add context from previous interactions
        context = {
            "previous_topic": getattr(self, 'last_education_topic', None),
            "mentioned_entities": getattr(self, 'education_entities', set()),
            "last_question": getattr(self, 'last_question', None)
        }
        
        return await self.skill_manager.execute({
            "skill": "education",
            "intent": "answer_question",
            "text": text,
            "parameters": {
                **intent["parameters"],
                "context": context
            }
        })

    async def _route_tutor_request(self, text: str, intent: Dict) -> Dict:
        """Handle tutor-specific routing"""
        # Check if we need subject specification
        if not intent["parameters"].get("subject"):
            return await self.skill_manager.execute({
                "skill": "tutor",
                "intent": "ask_subject",
                "text": "What subject would you like help with?"
            })
        
        return await self.skill_manager.execute({
            "skill": "tutor",
            "intent": "teach",
            "text": text,
            "parameters": intent["parameters"]
        })

    async def _route_conversation_request(self, text: str, intent: Dict) -> Dict:
        """Handle conversation-specific routing"""
        # Add conversation history context
        context = {
            "previous_topic": getattr(self, 'last_conversation_topic', None),
            "conversation_history": getattr(self, 'conversation_history', [])[-5:],
            "user_preferences": getattr(self, 'user_preferences', {})
        }
        
        return await self.skill_manager.execute({
            "skill": "conversation",
            "intent": "chat",
            "text": text,
            "parameters": {
                **intent["parameters"],
                "context": context
            }
        })
    
    async def handle_story_request(self, text: str, intent: Dict) -> Dict:
        """Handle story-related requests"""
        try:
            # If theme is present, start new story
            if intent.get("theme"):
                print(f"▶ Creating story with theme: {intent['theme']}")
                return await self.story_skill.handle({
                    "text": text,
                    "intent": "tell_story",
                    "parameters": {
                        "theme": intent["theme"]
                    }
                })
            
            # If no theme but story intent, ask for theme
            else:
                print("▶ Requesting story theme")
                return await self.story_skill.handle({
                    "text": text,
                    "intent": "ask_theme",
                    "prompt": "What kind of story would you like to hear?"
                })
            
        except Exception as e:
            print(f"Error handling story request: {e}")
            return {
                "text": "I had trouble with that story. Would you like to try a different one?",
                "context": "error"
            }
    
    def _extract_story_elements(self, text: str) -> Dict:
        """Extract story elements from request"""
        elements = {
            "characters": [],
            "theme": "general",
            "style": "simple"
        }
        
        # Extract characters
        text_lower = text.lower()
        for animal in ["cat", "dog", "fox", "hen", "rabbit", "bear", "mouse"]:
            if animal in text_lower:
                elements["characters"].append(animal)
        
        # Determine theme
        if any(word in text_lower for word in ["adventure", "journey", "quest"]):
            elements["theme"] = "adventure"
        elif any(word in text_lower for word in ["magic", "wizard", "fairy"]):
            elements["theme"] = "fantasy"
        elif any(word in text_lower for word in ["learn", "lesson", "moral"]):
            elements["theme"] = "educational"
        
        # Determine style
        if "funny" in text_lower or "silly" in text_lower:
            elements["style"] = "humorous"
        elif "scary" in text_lower:
            elements["style"] = "spooky"
        elif "bedtime" in text_lower:
            elements["style"] = "gentle"
            
        return elements
    
    def _adapt_cached_response(self, cached_response: Dict) -> Dict:
        """Adapt cached response to current context"""
        # Add variety to cached responses
        variations = {
            "Also, ": "Additionally, ",
            "Moreover, ": "Furthermore, ",
            "You see, ": "As you might know, "
        }
        
        text = cached_response["text"]
        for old, new in variations.items():
            if text.startswith(old):
                text = new + text[len(old):]
                break
                
        return {
            **cached_response,
            "text": text,
            "from_cache": True
        }
    
    def _check_followup(self, text: str) -> tuple[bool, dict]:
        """Check if question is a follow-up and get context"""
        if not self.conversation_context["current_topic"]:
            return False, {}
            
        # Check for follow-up indicators
        followup_indicators = [
            "what about", "and", "also", "then", "so",
            "is it", "are they", "does it", "do they"
        ]
        
        # Check for pronouns referring to previous context
        context_pronouns = [
            "it", "they", "there", "that", "those", "these",
            "this", "he", "she", "their", "its"
        ]
        
        is_followup = (
            any(text.lower().startswith(i) for i in followup_indicators) or
            any(p in text.lower().split() for p in context_pronouns)
        )
        
        if is_followup:
            return True, {
                "previous_topic": self.conversation_context["current_topic"],
                "mentioned_entities": list(self.conversation_context["mentioned_entities"]),
                "last_question": self.conversation_context["last_question"],
                "last_answer": self.conversation_context["last_answer"]
            }
            
        return False, {}
    
    def _extract_entities(self, text: str) -> set:
        """Extract important entities from text"""
        # Simple entity extraction - could be enhanced with NLP
        common_words = {"what", "where", "when", "why", "how", "is", "are", "the", "a", "an"}
        words = text.lower().split()
        return {word for word in words if word not in common_words}
    
    def _handle_followup(self, text: str, context: dict) -> Dict:
        """Handle follow-up questions with context"""
        return {
            "skill": "education",
            "intent": "answer_question",
            "mode": "follow_up",
            "parameters": {
                "question": text,
                "context": context
            }
        }
    
    def _route_educational_request(self, text: str) -> Dict:
        """Route educational questions with context"""
        # Update conversation context
        self.conversation_context.update({
            "active": True,
            "current_topic": self._extract_topic(text),
            "last_question": text,
            "last_interaction": time.time()
        })
        
        return {
            "skill": "education",
            "intent": "answer_question",
            "mode": "informative",
            "parameters": {
                "question": text,
                "context": {
                    "topic": self.conversation_context["current_topic"],
                    "mentioned_entities": list(self.conversation_context["mentioned_entities"]),
                    "conversation_history": self.conversation_context["conversation_history"][-3:]  # Last 3 interactions
                }
            }
        }
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from question"""
        # Simple topic extraction - can be made more sophisticated
        common_words = {"what", "where", "when", "why", "how", "is", "are", "the", "a", "an"}
        words = text.lower().split()
        # Get first noun-like word not in common words
        for word in words:
            if word not in common_words:
                return word
        return words[-1]  # Fallback to last word
    
    def _reset_conversation_context(self):
        """Reset conversation context"""
        self.conversation_context.update({
            "active": False,
            "topic": None,
            "last_question": None,
            "last_answer": None,
            "last_interaction": 0
        })
    
    def update_context_with_answer(self, answer: str):
        """Update context with latest answer"""
        self.conversation_context["last_answer"] = answer
        self.conversation_context["last_interaction"] = time.time()
    
    def _extract_story_theme(self, text: str) -> str:
        """Extract the main theme from a story request"""
        # Common themes to look for
        themes = {
            "dragon": "fantasy dragon",
            "princess": "fairy tale",
            "magic": "magical adventure",
            "space": "space adventure",
            "animal": "animal friends",
            "dinosaur": "dinosaur adventure",
            "robot": "robot story",
            "pirate": "pirate adventure"
        }
        
        text_lower = text.lower()
        for key, theme in themes.items():
            if key in text_lower:
                return theme
        
        return "general story"  # Default theme
    
    async def detect_intent(self, text: str) -> dict:
        """Enhanced intent detection"""
        text_lower = text.lower()
        scores = {
            "story": 0.0,
            "education": 0.0,
            "tutor": 0.0,
            "timer": 0.0,
            "conversation": 0.0
        }
        
        # Education patterns
        if any(word in text_lower for word in ["what", "where", "when", "why", "how"]):
            scores["education"] += 1.5
            
        if any(word in text_lower for word in ["capital", "country", "city", "explain", "tell me about"]):
            scores["education"] += 1.0
            
        # Story patterns
        if any(word in text_lower for word in ["story", "tale", "once upon", "adventure"]):
            scores["story"] += 1.5
            
        # Get highest scoring intent
        max_score = max(scores.values())
        primary_intent = max(scores.items(), key=lambda x: x[1])[0]
        confidence = max_score / 3.0  # Normalize to 0-1
        
        print(f"\nIntent Scores:")
        for intent, score in scores.items():
            print(f"- {intent}: {score:.2f}")
        
        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "scores": scores,
            "parameters": {
                "question": text if primary_intent == "education" else None,
                "theme": text if primary_intent == "story" else None
            }
        }
    
    def _extract_parameters(self, text: str, intent: str) -> Dict:
        """Extract relevant parameters based on intent"""
        params = {}
        
        if intent == "story":
            # Extract story theme
            for story_type in self.intent_patterns["story"]["story_types"]:
                if story_type in text:
                    params["theme"] = story_type
                    break
            # Extract characters
            for character in self.intent_patterns["story"]["characters"]:
                if character in text:
                    params["character"] = character
                    break
                    
        elif intent == "timer":
            # Extract time values
            import re
            time_match = re.search(r'(\d+)\s*(second|minute|hour|sec|min|hr)s?', text)
            if time_match:
                value, unit = time_match.groups()
                params["duration"] = int(value)
                params["unit"] = unit
                
        elif intent in ["education", "tutor"]:
            # Extract subject and topic
            for subject in self.intent_patterns[intent].get("subjects", []):
                if subject in text:
                    params["subject"] = subject
                    # Extract surrounding context as topic
                    words = text.split()
                    subject_idx = words.index(subject)
                    params["topic"] = " ".join(words[max(0, subject_idx-2):subject_idx+3])
                    break
        
        return params
    
    def _create_error_response(self, user_input: str, error: str) -> Dict:
        """Create an error response"""
        return {
            "skill": "education",
            "intent": "answer_question",
            "mode": "error",
            "parameters": {
                "question": user_input,
                "error": error
            }
        }
    
    def _extract_educational_subject(self, text: str) -> str:
        """Extract educational subject from query"""
        subjects = {
            "history": ["history", "past", "ancient", "when did", "who was"],
            "science": ["science", "scientific", "physics", "chemistry", "biology"],
            "geography": ["geography", "country", "city", "where is", "capital"],
            "nature": ["nature", "animals", "plants", "environment", "wildlife"],
            "general": ["facts", "information", "tell me about", "what is"]
        }
        
        text_lower = text.lower()
        for subject, patterns in subjects.items():
            if any(pattern in text_lower for pattern in patterns):
                return subject
        
        return "general"
    
    async def _handle_timer_request(self, text: str) -> Dict:
        """Handle timer requests"""
        try:
            # Extract time from request
            time_value, time_unit = self._parse_time(text)
            
            if time_value is None:
                return {
                    "text": "I couldn't understand the time duration. Please specify like '5 minutes' or '30 seconds'.",
                    "type": "timer_error"
                }
            
            # Create timer ID
            self.timer_id += 1
            timer_id = f"timer_{self.timer_id}"
            
            # Convert to seconds
            seconds = self._convert_to_seconds(time_value, time_unit)
            
            # Start timer
            print(f"\n⏰ Starting timer for {time_value} {time_unit}")
            asyncio.create_task(self._run_timer(timer_id, seconds))
            
            return {
                "text": f"I've set a timer for {time_value} {time_unit}. I'll notify you when it's done!",
                "type": "timer_start",
                "timer_id": timer_id,
                "duration": seconds
            }
            
        except Exception as e:
            print(f"Error setting timer: {e}")
            return {
                "text": "Sorry, I had trouble setting the timer. Please try again.",
                "type": "timer_error"
            }

    def _parse_time(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        """Extract time value and unit from text"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Find numbers in text
        numbers = []
        for word in words:
            try:
                numbers.append(int(word))
            except ValueError:
                continue
        
        if not numbers:
            return None, None
        
        time_value = numbers[0]
        
        # Find time unit
        for unit, unit_words in self.intent_patterns["timer"]["time_units"].items():
            if any(word in text_lower for word in unit_words):
                return time_value, unit
        
        return None, None

    def _convert_to_seconds(self, value: int, unit: str) -> int:
        """Convert time to seconds"""
        conversions = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600
        }
        return value * conversions[unit]

    async def _run_timer(self, timer_id: str, seconds: int):
        """Run timer and notify when done"""
        try:
            self.active_timers[timer_id] = True
            
            # Show countdown every minute for long timers, every 10 seconds for short ones
            update_interval = 60 if seconds > 300 else 10
            
            while seconds > 0 and self.active_timers[timer_id]:
                await asyncio.sleep(update_interval)
                seconds -= update_interval
                if seconds > 0:
                    print(f"\n⏰ Timer update: {seconds//60}m {seconds%60}s remaining")
            
            if self.active_timers[timer_id]:
                print(f"\n🔔 Timer {timer_id} finished!")
                # Here you could add sound playback or other notification methods
                
            del self.active_timers[timer_id]
            
        except Exception as e:
            print(f"Error in timer: {e}")

    async def _generate_story_response(self, text: str, elements: Dict) -> str:
        """Generate a story based on request and elements"""
        try:
            # Create story prompt
            prompt = self._create_story_prompt(text, elements)
            
            # Generate story using OpenAI
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a creative storyteller who creates engaging, age-appropriate stories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating story response: {e}")
            return "I'm sorry, I had trouble creating that story. Would you like to try another one?"

    def _create_story_prompt(self, text: str, elements: Dict) -> str:
        """Create a detailed story prompt"""
        prompt_parts = [
            "Please create an engaging story with the following elements:",
            f"\nRequest: {text}",
            "\nGuidelines:",
            "- Make it engaging and descriptive",
            "- Include dialogue between characters",
            "- Add sensory details and emotions",
            "- Create a clear beginning, middle, and end",
            "- Include a subtle moral or lesson",
            f"\nCharacters: {', '.join(elements['characters']) if elements['characters'] else 'Create appropriate characters'}",
            f"Theme: {elements['theme']}",
            f"Style: {elements['style']}",
            "\nFormat the story with proper paragraphs and include a title.",
            "\nEnd with a gentle question to engage the listener."
        ]
        
        return "\n".join(prompt_parts)