from typing import Optional, Dict, Any
from .config import Settings
import logging
import json
from pathlib import Path
from datetime import datetime

class Router:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.context = self.settings.router["default_context"]
        self.history = []
        
        # Setup persistent storage
        self.storage_dir = Path(self.settings.CACHE_DIR) / "context"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.context_file = self.storage_dir / "conversation_history.json"
        
        # Load previous context if exists
        self._load_context()
        
    def _load_context(self) -> None:
        """Load context from persistent storage"""
        try:
            if self.context_file.exists():
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    self.context = data.get('context', self.context)
                    self.history = data.get('history', [])
                    # Trim history if needed
                    while len(self.history) > self.settings.router["max_history"]:
                        self.history.pop(0)
        except Exception as e:
            logging.error(f"Error loading context: {e}")
            
    def _save_context(self) -> None:
        """Save context to persistent storage"""
        try:
            data = {
                'context': self.context,
                'history': self.history,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.context_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving context: {e}")
        
    def get_context(self) -> str:
        """Get current conversation context"""
        return self.context
        
    def set_context(self, context: str) -> None:
        """Set conversation context"""
        self.context = context
        self._save_context()
        
    def add_to_history(self, intent: Dict[str, Any]) -> None:
        """Add intent to history"""
        # Add timestamp to intent
        intent['timestamp'] = datetime.now().isoformat()
        self.history.append(intent)
        if len(self.history) > self.settings.router["max_history"]:
            self.history.pop(0)
        self._save_context()
            
    def get_history(self) -> list:
        """Get conversation history"""
        return self.history
        
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.history = []
        self.context = self.settings.router["default_context"]
        self._save_context()
        
    def route_intent(self, intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route intent to appropriate handler"""
        try:
            # Add to history
            self.add_to_history(intent)
            
            # Get current context
            context = self.get_context()
            
            # Basic routing based on context
            if context == "general":
                return self._handle_general(intent)
            elif context == "help":
                return self._handle_help(intent)
            else:
                logging.warning(f"Unknown context: {context}")
                return self._handle_general(intent)
                
        except Exception as e:
            logging.error(f"Error routing intent: {e}")
            return {
                "response": "I'm having trouble processing that request. Could you try rephrasing it?",
                "context": self.settings.router["fallback_context"]
            }
            
    def _handle_general(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general conversation with context awareness"""
        # Extract question from intent
        question = intent.get("text", "").strip()
        
        if not question:
            return {
                "response": "I didn't catch that. Could you repeat your question?",
                "context": "general"
            }
            
        # Get recent context from history
        recent_context = []
        for past_intent in self.history[-3:]:  # Look at last 3 interactions
            if "text" in past_intent:
                recent_context.append(past_intent["text"])
        
        # Detect factual/educational questions
        educational_indicators = [
            "what is", "what's", "where is", "who is", "when did",
            "why does", "how does", "tell me about", "explain",
            "capital", "country", "city", "history", "science",
            "math", "calculate", "solve"
        ]
        
        # Check if this is an educational question
        if any(indicator in question.lower() for indicator in educational_indicators):
            return {
                "response": question,
                "context": "education",
                "subject": "factual_question",
                "auto_continue": True
            }
        
        # Handle follow-up questions
        if question.lower().startswith(("what else", "tell me more", "and")):
            if self.context == "education":
                return {
                    "response": question,
                    "context": "education",
                    "subject": "follow_up",
                    "auto_continue": True
                }
        
        # If no special handling, treat as general conversation
        return {
            "response": question,
            "context": "conversation"
        }
        
    def _handle_help(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help requests"""
        return {
            "response": "I can help answer questions and have conversations. Try asking me about facts, " +
                       "or just chat with me about any topic you're interested in.",
            "context": "general"
        }
        
    def _get_conversation_context(self) -> Dict[str, Any]:
        """Get relevant context from conversation history"""
        context = {
            "current_context": self.context,
            "recent_questions": [],
            "subjects_discussed": set(),
            "last_response": None
        }
        
        # Analyze recent history
        for intent in self.history[-5:]:  # Look at last 5 interactions
            if "text" in intent:
                context["recent_questions"].append(intent["text"])
            if "subject" in intent:
                context["subjects_discussed"].add(intent["subject"])
            if "response" in intent:
                context["last_response"] = intent["response"]
                
        return context