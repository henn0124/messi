"""
Intent Learning System for Messi Assistant
----------------------------------------

This module implements a learning system that continuously analyzes user interactions
to improve intent detection and suggest new features. It serves as both a monitoring
and improvement suggestion system.

Key Features:
    1. Interaction Analysis:
        - Monitors confidence levels of intent detection
        - Tracks usage patterns and common phrases
        - Identifies areas where intent detection fails
    
    2. Automated Learning:
        - Generates suggestions for new intent patterns
        - Identifies potential new features/skills
        - Tracks user interaction patterns
    
    3. Reporting System:
        - Generates daily reports of system performance
        - Maintains rolling window of latest reports
        - Provides both JSON data and human-readable summaries
    
    4. Improvement Tracking:
        - Logs unmatched or low-confidence interactions
        - Generates specific improvement suggestions
        - Prioritizes areas needing attention

Configuration:
    The system is configurable through a dictionary of settings including:
    - Report retention period
    - Learning thresholds
    - File paths and locations
    - Reporting schedule

Usage:
    learner = IntentLearner()
    
    # Analyze an interaction
    await learner.analyze_interaction(
        text="tell me a story",
        detected_intent="story",
        confidence=0.85
    )
    
    # Generate a report
    await learner.generate_daily_report()

File Structure:
    /learning/
        - intent_suggestions.json  # Improvement suggestions
        - unmatched_intents.csv   # Low confidence interactions
        - intent_stats.json       # Usage statistics
        /reports/
            - intent_report_YYYY-MM-DD.json  # Daily data reports
            - summary_YYYY-MM-DD.txt         # Human-readable summaries

Author: Your Name
Created: 2024-01-24
"""

from pathlib import Path
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import csv
from core.config import Settings

class IntentLearner:
    # Default configuration
    DEFAULT_CONFIG = {
        "reports": {
            "keep_latest": 10,          # Number of reports to keep
            "generate_time": "00:00",   # Time to generate daily report (24hr format)
            "formats": ["json", "txt"]  # Report formats to generate
        },
        "learning": {
            "low_confidence_threshold": 0.6,   # Threshold for logging unmatched intents
            "min_word_length": 3,             # Minimum word length for pattern analysis
            "improvement_threshold": 0.7,      # Threshold for suggesting improvements
            "high_priority_threshold": 0.5     # Threshold for high priority suggestions
        },
        "paths": {
            "base_dir": "/home/pi/messi",
            "learning_dir": "learning",
            "reports_dir": "reports"
        }
    }

    def __init__(self, learning_manager: 'LearningManager' = None, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.learning_manager = learning_manager  # Reference to learning manager
        
        # Load skills config
        self.skills_config = self._load_skills_config()
        
        # Intent learning state
        self.learned_patterns = {
            intent: {
                "patterns": set(),
                "weights": {},
                "success_rate": 0.0
            }
            for intent in self.skills_config["intents"]["patterns"].keys()
        }
        
        # Add recent interactions tracking
        self.recent_interactions = []
        self.max_interaction_history = 100  # Keep last 100 interactions
        
        # Initialize maintenance task as None
        self.maintenance_task = None
    
    async def initialize(self):
        """Initialize async components of the IntentLearner"""
        if self.maintenance_task is None:
            self.maintenance_task = asyncio.create_task(self._periodic_pattern_maintenance())
    
    async def learn_from_interaction(self, text: str, selected_intent: str, success: bool):
        """Learn from user interaction"""
        try:
            # Store interaction in history
            self.recent_interactions.append({
                "text": text,
                "intent": selected_intent,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            
            # Trim history if needed
            if len(self.recent_interactions) > self.max_interaction_history:
                self.recent_interactions = self.recent_interactions[-self.max_interaction_history:]
            
            # Update local learning state
            self._update_patterns(text, selected_intent, success)
            
            # Report learning to manager
            if self.learning_manager:
                await self.learning_manager.record_intent_learning({
                    "text": text,
                    "intent": selected_intent,
                    "success": success,
                    "patterns": self.learned_patterns[selected_intent],
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            print(f"Error learning from interaction: {e}")
    
    async def _periodic_pattern_maintenance(self):
        """Automatically maintain patterns periodically"""
        while True:
            try:
                # Check if conversation is active
                if self.learning_manager and self.learning_manager.is_conversation_active():
                    await asyncio.sleep(5)  # Short sleep and check again
                    continue
                    
                print("\nPerforming automated pattern maintenance...")
                
                # Analyze pattern performance
                for intent in self.learned_patterns:
                    # Check conversation state before each intent analysis
                    if self.learning_manager and self.learning_manager.is_conversation_active():
                        print("Conversation started - pausing maintenance")
                        break
                        
                    patterns = self.learned_patterns[intent]["weights"]
                    total_patterns = len(patterns)
                    
                    # Check pattern utilization
                    unused_patterns = self._find_unused_patterns(intent)
                    if unused_patterns:
                        print(f"Found {len(unused_patterns)} unused patterns in {intent}")
                        for pattern in unused_patterns:
                            del patterns[pattern]
                    
                    # Look for new significant patterns
                    new_patterns = self._discover_new_patterns(intent)
                    if new_patterns:
                        print(f"Discovered {len(new_patterns)} new patterns for {intent}")
                        for pattern, weight in new_patterns.items():
                            patterns[pattern] = weight
                    
                    # Report changes
                    if total_patterns != len(patterns):
                        print(f"Pattern count for {intent}: {total_patterns} -> {len(patterns)}")
                    
                    await asyncio.sleep(0.1)  # Allow other tasks to run
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                print(f"Error in pattern maintenance: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    def _find_unused_patterns(self, intent: str) -> List[str]:
        """Find patterns that haven't been used recently"""
        unused = []
        cutoff_time = datetime.now() - timedelta(days=7)  # One week
        
        for pattern in self.learned_patterns[intent]["weights"].keys():
            # Skip core patterns from config
            if pattern in self.skills_config["intents"]["patterns"][intent]["keywords"]:
                continue
                
            # Check if pattern has been used recently
            pattern_used = False
            for interaction in self.recent_interactions:
                if (interaction["intent"] == intent and 
                    pattern in interaction["text"].lower() and
                    datetime.fromisoformat(interaction["timestamp"]) > cutoff_time):
                    pattern_used = True
                    break
            
            if not pattern_used:
                unused.append(pattern)
        
        return unused
    
    def _discover_new_patterns(self, intent: str) -> Dict[str, float]:
        """Discover new significant patterns from recent interactions"""
        # Count pattern occurrences in successful interactions
        pattern_counts = {}
        success_counts = {}
        
        for interaction in self.recent_interactions:
            if interaction["intent"] == intent:
                text = interaction["text"].lower()
                # Extract words and phrases
                patterns = set(text.split()) | set(self._extract_phrases(text))
                
                for pattern in patterns:
                    if pattern not in self.learned_patterns[intent]["weights"]:
                        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                        if interaction["success"]:
                            success_counts[pattern] = success_counts.get(pattern, 0) + 1
        
        # Find patterns with high success rate
        new_patterns = {}
        for pattern, count in pattern_counts.items():
            if count >= 3:  # Require minimum occurrences
                success_rate = success_counts.get(pattern, 0) / count
                if success_rate > 0.7:  # Require high success rate
                    new_patterns[pattern] = 0.5 + (success_rate * 0.3)  # Initial weight
        
        return new_patterns
    
    def get_intent_scores(self, text: str) -> Dict[str, float]:
        """Get learned intent scores"""
        scores = {}
        text_lower = text.lower()
        
        for intent, data in self.learned_patterns.items():
            score = 0.0
            
            # Check learned patterns
            for pattern, weight in data["weights"].items():
                if pattern in text_lower:
                    score += weight
            
            # Apply success rate modifier
            score *= (0.5 + data["success_rate"] / 2)  # Scale from 0.5 to 1.0
            scores[intent] = score
            
        return scores
    
    def _update_patterns(self, text: str, selected_intent: str, success: bool):
        """Update learned patterns with dynamic pattern discovery"""
        try:
            text_lower = text.lower()
            words = text_lower.split()
            phrases = self._extract_phrases(text_lower)
            
            # Get learning parameters from config
            learning_rate = self.skills_config["intents"]["thresholds"]["learning_rate"]
            max_patterns = self.skills_config["intents"]["thresholds"]["max_patterns"]
            
            # Update existing patterns
            for word in words:
                if len(word) >= self.DEFAULT_CONFIG["learning"]["min_word_length"]:
                    if word not in self.learned_patterns[selected_intent]["weights"]:
                        # New pattern discovered
                        if len(self.learned_patterns[selected_intent]["weights"]) < max_patterns:
                            self.learned_patterns[selected_intent]["weights"][word] = 0.5
                            print(f"New pattern discovered for {selected_intent}: '{word}'")
                    
                    # Update weight
                    self._adjust_pattern_weight(selected_intent, word, success, learning_rate)
            
            # Update phrase patterns
            for phrase in phrases:
                if phrase not in self.learned_patterns[selected_intent]["weights"]:
                    # Check if phrase appears in multiple successful interactions
                    if self._is_significant_phrase(phrase, selected_intent):
                        if len(self.learned_patterns[selected_intent]["weights"]) < max_patterns:
                            self.learned_patterns[selected_intent]["weights"][phrase] = 0.6  # Higher initial weight for phrases
                            print(f"New phrase pattern discovered for {selected_intent}: '{phrase}'")
                else:
                    self._adjust_pattern_weight(selected_intent, phrase, success, learning_rate)
            
            # Prune low-performing patterns
            self._prune_patterns(selected_intent)
            
            # Update success rate
            self._update_success_rate(selected_intent, success)
            
        except Exception as e:
            print(f"Error updating patterns: {e}")

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract potential meaningful phrases"""
        phrases = []
        words = text.split()
        
        # Look for 2-3 word combinations
        for i in range(len(words)-1):
            phrases.append(f"{words[i]} {words[i+1]}")
            if i < len(words)-2:
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return phrases

    def _is_significant_phrase(self, phrase: str, intent: str) -> bool:
        """Determine if a phrase is significant based on history"""
        # Check if phrase appears in recent successful interactions
        success_count = 0
        total_count = 0
        
        for interaction in self.recent_interactions:
            if phrase in interaction["text"].lower():
                total_count += 1
                if interaction["intent"] == intent and interaction["success"]:
                    success_count += 1
        
        # Require multiple successful uses
        return success_count >= 2 and (success_count / total_count if total_count > 0 else 0) > 0.7

    def _adjust_pattern_weight(self, intent: str, pattern: str, success: bool, learning_rate: float):
        """Adjust pattern weight based on success"""
        current_weight = self.learned_patterns[intent]["weights"][pattern]
        
        if success:
            # Increase weight for successful matches
            new_weight = current_weight + (learning_rate * (1 - current_weight))
        else:
            # Decrease weight for failures
            new_weight = current_weight - (learning_rate * current_weight)
        
        # Keep weights between 0.1 and 1.0
        self.learned_patterns[intent]["weights"][pattern] = max(0.1, min(1.0, new_weight))

    def _prune_patterns(self, intent: str):
        """Remove low-performing patterns"""
        weights = self.learned_patterns[intent]["weights"]
        
        # Remove patterns with consistently low weights
        low_performers = [
            pattern for pattern, weight in weights.items()
            if weight < 0.2 and pattern not in self.skills_config["intents"]["patterns"][intent]["keywords"]
        ]
        
        for pattern in low_performers:
            del weights[pattern]
            print(f"Removed low-performing pattern from {intent}: '{pattern}'")

    def _update_success_rate(self, intent: str, success: bool):
        """Update success rate statistics"""
        patterns = self.learned_patterns[intent]
        if "total_attempts" not in patterns:
            patterns["total_attempts"] = 0
            patterns["successful_attempts"] = 0
        
        patterns["total_attempts"] += 1
        if success:
            patterns["successful_attempts"] += 1
        
        patterns["success_rate"] = patterns["successful_attempts"] / patterns["total_attempts"]

    def _load_skills_config(self) -> dict:
        """Load skills configuration from settings"""
        try:
            # Get skills config from settings
            if hasattr(self.settings, 'skills'):
                return self.settings.skills
            
            # If no skills config in settings, return default empty structure
            return {
                "intents": {
                    "patterns": {},
                    "thresholds": {
                        "learning_rate": 0.1,
                        "max_patterns": 20
                    }
                }
            }
        except Exception as e:
            print(f"Error loading skills config: {e}")
            # Return default config on error
            return {
                "intents": {
                    "patterns": {},
                    "thresholds": {
                        "learning_rate": 0.1,
                        "max_patterns": 20
                    }
                }
            }