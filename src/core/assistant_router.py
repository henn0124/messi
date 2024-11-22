from openai import AsyncOpenAI
from typing import Dict
from .config import Settings
import json
import time
from .cache_manager import ResponseCache

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
                    "dragon", "princess", "adventure", "fairy tale"
                ],
                "phrases": [
                    "can you tell", "tell me about", "read me",
                    "how about a story", "i want to hear"
                ],
                "characters": [
                    "dragon", "princess", "knight", "fairy",
                    "wizard", "unicorn", "robot", "pirate"
                ],
                "themes": [
                    "magic", "adventure", "fantasy", "space",
                    "animals", "friendship", "bedtime"
                ]
            },
            "education": {
                "keywords": [
                    "what", "why", "how", "when", "where",
                    "explain", "teach", "learn", "help me understand"
                ],
                "subjects": [
                    "math", "science", "history", "nature",
                    "animals", "space", "weather"
                ]
            },
            "conversation": {
                "keywords": [
                    "chat", "talk", "discuss", "let's",
                    "can we", "what do you think"
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
    
    async def route_request(self, user_input: str, time_of_day: str = "evening") -> Dict:
        try:
            print("\n=== Assistant Router ===")
            print(f"Processing request: '{user_input}'")
            
            # Detect intent with enhanced system
            intent = await self.detect_intent(user_input)
            print(f"Detected intent: {intent['primary_intent']} (confidence: {intent['confidence']:.2f})")
            
            # Create base result
            result = {
                "text": "",
                "skill": intent["primary_intent"],
                "intent": "tell_story" if intent["primary_intent"] == "story" else "answer_question",
                "mode": "storytelling" if intent["primary_intent"] == "story" else "informative",
                "parameters": {
                    "question": user_input,
                    "context": {
                        "topic": self.conversation_context["current_topic"],
                        "mentioned_entities": list(self.conversation_context["mentioned_entities"]),
                        "conversation_history": self.conversation_context["conversation_history"][-3:],
                        "intent_confidence": intent["confidence"],
                        "time_of_day": time_of_day
                    }
                }
            }
            
            # Add specific parameters based on intent
            if intent["primary_intent"] == "story":
                result["parameters"]["theme"] = self._extract_story_theme(user_input)
            else:  # education
                result["parameters"]["subject"] = self._extract_educational_subject(user_input)
            
            # Get response from OpenAI with appropriate prompt
            messages = await self._create_contextual_prompt(user_input, intent["primary_intent"])
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=messages,
                temperature=0.7
            )
            result["text"] = response.choices[0].message.content
            
            return result
            
        except Exception as e:
            print(f"✗ Error in router: {e}")
            return self._create_error_response(user_input, str(e))
    
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
            "conversation": 0.0
        }
        
        # Story detection
        story_keywords = ["story", "tell me", "once upon", "tale", "adventure"]
        story_subjects = ["cat", "dog", "dragon", "princess", "robot", "animal"]
        
        for keyword in story_keywords:
            if keyword in text_lower:
                scores["story"] += 1.5
                
        for subject in story_subjects:
            if subject in text_lower:
                scores["story"] += 1.0
        
        # Education detection
        education_keywords = ["what", "why", "how", "when", "where", "explain"]
        for keyword in education_keywords:
            if keyword in text_lower:
                scores["education"] += 1.0
        
        # Conversation detection
        conversation_keywords = ["hello", "hi", "hey", "thanks", "thank you", "bye"]
        for keyword in conversation_keywords:
            if keyword in text_lower:
                scores["conversation"] += 1.0
        
        # Get highest scoring intent
        max_score = max(scores.values())
        primary_intent = max(scores.items(), key=lambda x: x[1])[0]
        confidence = max_score / 5.0  # Normalize to 0-1
        
        print(f"\nIntent Detection:")
        print(f"Text: '{text}'")
        print(f"Scores: {scores}")
        print(f"Selected: {primary_intent} ({confidence:.2f})")
        
        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "scores": scores
        }
    
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