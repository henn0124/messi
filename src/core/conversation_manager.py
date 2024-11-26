"""
Conversation Manager for Messi Assistant
--------------------------------------

This module manages the conversation state and context for the assistant,
ensuring coherent and contextually appropriate interactions over time.

Key Features:
    1. Context Management:
        - Tracks current conversation topic
        - Maintains conversation history
        - Manages entity references
        - Handles topic transitions
    
    2. State Tracking:
        - Active conversation status
        - Time-based context expiry
        - Topic relationships
        - User engagement levels
    
    3. Conversation Flow:
        - Natural topic progression
        - Context-aware responses
        - Follow-up handling
        - Conversation recovery
    
    4. Memory Management:
        - Short-term conversation memory
        - Long-term topic memory
        - Entity relationship tracking
        - Context pruning

Configuration:
    - Memory window size
    - Context expiry time
    - Topic transition thresholds
    - Entity reference limits

Usage:
    manager = ConversationManager()
    
    # Start conversation
    await manager.start_conversation()
    
    # Update context
    await manager.update_context(user_input, response)
    
    # End conversation
    await manager.end_conversation()

Integration:
    Works with:
    - AssistantRouter for intent context
    - ResponseCache for historical context
    - Various skill handlers for specialized context

Author: Your Name
Created: 2024-01-24
"""

from enum import Enum
from typing import Optional, Set, Dict
import time
import asyncio

class ConversationState(Enum):
    ACTIVE = "active"
    WINDING_DOWN = "winding_down"
    ENDED = "ended"
    PAUSED = "paused"

class ConversationManager:
    # Timing thresholds (in seconds)
    INACTIVITY_THRESHOLD = 5.0      # Short pause - might indicate thinking or brief pause
    EXTENDED_PAUSE_THRESHOLD = 10.0  # Long pause - likely end of conversation
    RESPONSE_WINDOW = 3.0           # Expected time window for a response
    
    # Expanded exit phrases with variations
    EXIT_PHRASES = {
        # Direct endings
        "goodbye", "bye", "bye bye", "see you", 
        "that's all", "we're done", "i'm done",
        
        # Polite endings
        "thank you", "thanks", "thank you very much",
        
        # Bedtime endings
        "good night", "sleep well", "time for bed",
        
        # Command endings
        "stop", "end", "finish", "quit", "exit"
    }
    
    # Engagement thresholds for detecting user interest
    MIN_RESPONSE_LENGTH = 3         # Minimum words to consider a full response
    SHORT_RESPONSE_THRESHOLD = 2    # Number of consecutive short responses before considering disengagement
    
    def __init__(self):
        # Initialize conversation state
        self.state = ConversationState.ENDED  # Start in ENDED state until first interaction
        self.last_interaction_time = 0        # Track time of last interaction
        self.conversation_start_time = 0      # When did this conversation begin
        self.interaction_count = 0            # How many back-and-forth exchanges
        self.consecutive_short_responses = 0  # Track potential disengagement
        self.last_response_length = 0         # Length of last user response
        self.current_topic = None             # Current conversation topic
        
        # Metrics for analyzing conversation patterns
        self.metrics = {
            "total_duration": 0,              # How long conversation lasted
            "response_times": [],             # Time between responses
            "interaction_gaps": [],           # Gaps in conversation
            "topics": set(),                  # Topics discussed
            "exit_type": None                 # How did conversation end
        }
        
        # Conversation type handlers
        self.conversation_types = {
            "education": EducationalConversation,
            "story": StoryConversation,
            "general": GeneralConversation
        }
        
        self.active_conversations = {}
    
    async def start_conversation(self):
        """Initialize a new conversation"""
        self.state = ConversationState.ACTIVE
        self.conversation_start_time = time.time()
        self.last_interaction_time = time.time()
        self.interaction_count = 0
        self.consecutive_short_responses = 0
        print("Starting new conversation")
    
    async def process_interaction(self, text: str) -> bool:
        """
        Process user interaction and determine if conversation should continue.
        Returns True if conversation should continue, False if it should end.
        """
        current_time = time.time()
        
        # Update interaction metrics
        if self.last_interaction_time > 0:
            gap = current_time - self.last_interaction_time
            self.metrics["interaction_gaps"].append(gap)
        
        self.last_interaction_time = current_time
        self.interaction_count += 1
        
        # Check for explicit endings - more thorough check
        text_lower = text.lower().strip()
        
        # Check for exact matches
        if text_lower in self.EXIT_PHRASES:
            print(f"Explicit exit phrase detected: '{text_lower}'")
            await self.end_conversation("explicit_exit")
            return False
        
        # Check for phrases within text
        for phrase in self.EXIT_PHRASES:
            if phrase in text_lower:
                print(f"Exit phrase detected within text: '{phrase}'")
                await self.end_conversation("explicit_exit")
                return False
        
        # Story requests should continue conversation
        if any(phrase in text_lower for phrase in ["tell me a story", "story about"]):
            self.state = ConversationState.ACTIVE
            self.consecutive_short_responses = 0
            print("Starting story conversation")
            return True
        
        # Analyze response length for engagement
        words = text.split()
        if len(words) < self.MIN_RESPONSE_LENGTH:
            self.consecutive_short_responses += 1
        else:
            self.consecutive_short_responses = 0
        
        # Check if user seems to be disengaging
        if self.consecutive_short_responses >= self.SHORT_RESPONSE_THRESHOLD:
            self.state = ConversationState.WINDING_DOWN
            print("Conversation appears to be winding down")
        
        return True
    
    async def check_continuation(self) -> bool:
        """
        Check if conversation should continue based on timing and state.
        Returns True if we should continue listening, False if we should end.
        """
        if self.state == ConversationState.ENDED:
            return False
            
        current_time = time.time()
        time_since_last = current_time - self.last_interaction_time
        
        # Check for timeout conditions
        if time_since_last > self.EXTENDED_PAUSE_THRESHOLD:
            # Long pause - end conversation
            await self.end_conversation("timeout")
            return False
        elif time_since_last > self.INACTIVITY_THRESHOLD:
            # Short pause - mark as winding down but continue
            self.state = ConversationState.WINDING_DOWN
            print("Extended pause detected")
            return True
            
        return True
    
    async def end_conversation(self, reason: str):
        """
        End conversation and record final metrics.
        reason: Why the conversation ended (timeout, explicit_exit, etc.)
        """
        self.state = ConversationState.ENDED
        self.metrics["total_duration"] = time.time() - self.conversation_start_time
        self.metrics["exit_type"] = reason
        print(f"Conversation ended: {reason}")
        print(f"Duration: {self.metrics['total_duration']:.1f}s")
        print(f"Interactions: {self.interaction_count}")
    
    def get_state(self) -> ConversationState:
        """Get current conversation state"""
        return self.state
    
    def get_metrics(self) -> Dict:
        """Get conversation metrics"""
        return self.metrics 

class BaseConversation:
    """Base class for all conversation types"""
    def __init__(self):
        self.history = []
        self.state = "active"
        self.last_interaction = time.time()
    
    async def add_exchange(self, text: str, role: str):
        """Add an exchange to conversation history"""
        self.history.append({
            "text": text,
            "role": role,
            "timestamp": time.time()
        })
        self.last_interaction = time.time()

class EducationalConversation(BaseConversation):
    """Educational conversation handling"""
    def __init__(self):
        super().__init__()
        self.learning_topics = set()
        self.knowledge_level = "beginner"
        
    async def add_exchange(self, text: str, role: str):
        await super().add_exchange(text, role)
        if role == "user":
            self._update_learning_progress(text)

class StoryConversation(BaseConversation):
    """Story conversation handling"""
    def __init__(self):
        super().__init__()
        self.narrative_state = "beginning"
        self.characters = set()
        
    async def add_exchange(self, text: str, role: str):
        await super().add_exchange(text, role)
        if role == "assistant":
            self._update_narrative_state(text)