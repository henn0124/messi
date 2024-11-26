"""
Conversation Logging System
--------------------------
Provides detailed logging for debugging and analysis.

Key Features:
1. Multi-level Logging:
   - Console output
   - File logging
   - Debug information

2. Context Tracking:
   - Conversation flow
   - Intent detection
   - Error states

3. Performance Monitoring:
   - Response times
   - API calls
   - Error rates

4. Debug Support:
   - Detailed error logs
   - State transitions
   - Context changes

Usage:
    logger = ConversationLogger()
    logger.log_intent_detection(text, scores, intent)
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class ConversationLogger:
    def __init__(self):
        # Set up file logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Main log file
        self.logger = logging.getLogger("messi")
        self.logger.setLevel(logging.DEBUG)
        
        # Create handlers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(f"logs/messi_{timestamp}.log")
        ch = logging.StreamHandler()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)15s | %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(message)s'
        )
        
        # Set formatters
        fh.setFormatter(file_formatter)
        ch.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Conversation tracking
        self.conversation_id = timestamp
        self.turn_count = 0
        
    def log_conversation_start(self):
        """Log start of new conversation"""
        self.logger.info("=== Starting New Conversation ===")
        self.logger.info(f"Conversation ID: {self.conversation_id}")
        
    def log_intent_detection(self, text: str, scores: Dict[str, float], intent: str):
        """Log intent detection details"""
        self.logger.info("\n=== Intent Detection ===")
        self.logger.info(f"Input: '{text}'")
        self.logger.info("Intent Scores:")
        for name, score in scores.items():
            self.logger.info(f"  {name:12s}: {score:.2f}")
        self.logger.info(f"Selected Intent: {intent}")
        
    def log_skill_execution(self, skill: str, input_data: Any, output_data: Any):
        """Log skill execution details"""
        self.turn_count += 1
        self.logger.info(f"\n=== Turn {self.turn_count} ===")
        self.logger.info(f"Skill: {skill}")
        self.logger.info(f"Input: {json.dumps(input_data, indent=2)}")
        self.logger.info(f"Output: {json.dumps(output_data, indent=2)}")
        
    def log_error(self, context: str, error: Exception, details: Dict = None):
        """Log error with context"""
        self.logger.error(f"\n=== Error in {context} ===")
        self.logger.error(f"Error Type: {type(error).__name__}")
        self.logger.error(f"Error Message: {str(error)}")
        if details:
            self.logger.error("Additional Details:")
            self.logger.error(json.dumps(details, indent=2))
        
    def log_audio_processing(self, stage: str, details: Dict):
        """Log audio processing details"""
        self.logger.info(f"\n=== Audio Processing: {stage} ===")
        self.logger.info(json.dumps(details, indent=2))
        
    def log_learning_status(self, enabled: bool):
        """Log learning system status"""
        status = "enabled" if enabled else "disabled"
        self.logger.info(f"Learning system is {status}")
        
    def log_learning_event(self, event_type: str, details: Dict):
        """Log learning events when enabled"""
        if self.learning_manager and self.learning_manager.learning_enabled:
            self.logger.info(f"Learning event: {event_type}")
            self.logger.info(json.dumps(details, indent=2)) 