"""
Conversation Logging System
--------------------------
Provides detailed logging for debugging and analysis.
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
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def critical(self, msg: str):
        self.logger.critical(msg)
    
    def log_conversation_start(self):
        """Log start of new conversation"""
        self.info("=== Starting New Conversation ===")
        self.info(f"Conversation ID: {self.conversation_id}")
    
    def log_intent_detection(self, text: str, scores: Dict[str, float], intent: str):
        """Log intent detection details"""
        self.debug(f"Intent Detection:\nText: {text}\nScores: {json.dumps(scores, indent=2)}\nSelected: {intent}")
    
    def log_conversation_state(self, state: Dict[str, Any]):
        """Log conversation state changes"""
        self.debug(f"Conversation State:\n{json.dumps(state, indent=2)}")
    
    def log_api_call(self, api: str, params: Dict[str, Any], duration: float):
        """Log API call details"""
        self.debug(f"API Call - {api}:\nParams: {json.dumps(params, indent=2)}\nDuration: {duration:.2f}s")
    
    def log_skill_execution(self, skill: str, input_data: Any, output_data: Any):
        """Log skill execution details"""
        self.turn_count += 1
        self.info(f"\n=== Turn {self.turn_count} ===")
        self.info(f"Skill: {skill}")
        self.info(f"Input: {json.dumps(input_data, indent=2)}")
        self.info(f"Output: {json.dumps(output_data, indent=2)}")
    
    def log_error(self, context: str, error: Exception, details: Dict = None):
        """Log error with context"""
        self.error(f"\n=== Error in {context} ===")
        self.error(f"Error Type: {type(error).__name__}")
        self.error(f"Error Message: {str(error)}")
        if details:
            self.error("Additional Details:")
            self.error(json.dumps(details, indent=2))
    
    def log_audio_processing(self, stage: str, details: Dict):
        """Log audio processing details"""
        self.info(f"\n=== Audio Processing: {stage} ===")
        self.info(json.dumps(details, indent=2))
    
    def log_learning_status(self, enabled: bool):
        """Log learning system status"""
        status = "enabled" if enabled else "disabled"
        self.info(f"Learning system is {status}")
    
    def log_learning_event(self, event_type: str, details: Dict):
        """Log learning events when enabled"""
        if self.learning_manager and self.learning_manager.learning_enabled:
            self.info(f"Learning event: {event_type}")
            self.info(json.dumps(details, indent=2)) 