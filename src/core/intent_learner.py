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
from datetime import datetime
from typing import Dict, List, Optional
import csv

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

    def __init__(self, config: dict = None):
        # Merge provided config with defaults
        self.config = self._merge_config(self.DEFAULT_CONFIG, config or {})
        
        # Initialize paths using config
        self.base_dir = Path(self.config["paths"]["base_dir"])
        self.learning_dir = self.base_dir / self.config["paths"]["learning_dir"]
        self.reports_dir = self.learning_dir / self.config["paths"]["reports_dir"]
        
        # File paths
        self.suggestions_file = self.learning_dir / "intent_suggestions.json"
        self.unmatched_file = self.learning_dir / "unmatched_intents.csv"
        self.stats_file = self.learning_dir / "intent_stats.json"
        
        # Create directories
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize files
        self._initialize_files()
        
        # Load existing data
        self.suggestions = self._load_suggestions()
        self.stats = self._load_stats()

    def _merge_config(self, default: dict, custom: dict) -> dict:
        """Deep merge custom config with defaults"""
        merged = default.copy()
        
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_config(merged[key], value)
            else:
                merged[key] = value
                
        return merged

    def _initialize_files(self):
        """Initialize logging files with proper structure"""
        if not self.suggestions_file.exists():
            initial_data = {
                "pending": [],
                "approved": [],
                "rejected": []
            }
            with open(self.suggestions_file, 'w') as f:
                json.dump(initial_data, f, indent=2)

        if not self.unmatched_file.exists():
            with open(self.unmatched_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'text', 'detected_intent', 'confidence', 'context'])

        if not self.stats_file.exists():
            initial_stats = {
                "intent_counts": {},
                "confidence_levels": {},
                "common_patterns": {},
                "improvement_areas": []
            }
            with open(self.stats_file, 'w') as f:
                json.dump(initial_stats, f, indent=2)

    async def analyze_interaction(self, text: str, detected_intent: str, 
                                confidence: float, context: Optional[Dict] = None):
        """Analyze user interaction for learning"""
        try:
            # Log low confidence interactions using config threshold
            if confidence < self.config["learning"]["low_confidence_threshold"]:
                await self._log_unmatched_intent(text, detected_intent, confidence, context)
            
            # Update statistics
            await self._update_stats(detected_intent, confidence, text)
            
            # Generate improvement suggestions
            await self._generate_suggestions(text, detected_intent, confidence, context)
            
        except Exception as e:
            print(f"Error analyzing interaction: {e}")

    async def _log_unmatched_intent(self, text: str, detected_intent: str, 
                                   confidence: float, context: Optional[Dict]):
        """Log interactions that weren't well matched"""
        try:
            with open(self.unmatched_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    text,
                    detected_intent,
                    confidence,
                    json.dumps(context) if context else ''
                ])
        except Exception as e:
            print(f"Error logging unmatched intent: {e}")

    async def _update_stats(self, intent: str, confidence: float, text: str):
        """Update usage statistics"""
        try:
            # Load current stats
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            
            # Update intent counts
            stats["intent_counts"][intent] = stats["intent_counts"].get(intent, 0) + 1
            
            # Update confidence levels
            conf_key = f"{int(confidence * 10)}/10"
            stats["confidence_levels"][conf_key] = stats["confidence_levels"].get(conf_key, 0) + 1
            
            # Update common patterns
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Ignore short words
                    stats["common_patterns"][word] = stats["common_patterns"].get(word, 0) + 1
            
            # Save updated stats
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"Error updating stats: {e}")

    async def _generate_suggestions(self, text: str, detected_intent: str, 
                                  confidence: float, context: Optional[Dict]):
        """Generate improvement suggestions based on interaction"""
        try:
            suggestion = {
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "detected_intent": detected_intent,
                "confidence": confidence,
                "context": context,
                "type": "intent_improvement",
                "status": "pending"
            }
            
            # Add specific improvement suggestions
            if confidence < 0.4:
                suggestion["suggestion"] = "New intent category needed"
                suggestion["priority"] = "high"
            elif confidence < 0.6:
                suggestion["suggestion"] = "Additional patterns needed"
                suggestion["priority"] = "medium"
            
            # Load current suggestions
            with open(self.suggestions_file, 'r') as f:
                suggestions = json.load(f)
            
            # Add new suggestion
            suggestions["pending"].append(suggestion)
            
            # Save updated suggestions
            with open(self.suggestions_file, 'w') as f:
                json.dump(suggestions, f, indent=2)
                
        except Exception as e:
            print(f"Error generating suggestions: {e}")

    async def get_improvement_report(self) -> Dict:
        """Generate a comprehensive improvement report"""
        try:
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            
            with open(self.suggestions_file, 'r') as f:
                suggestions = json.load(f)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_interactions": sum(stats["intent_counts"].values()),
                    "intent_distribution": stats["intent_counts"],
                    "low_confidence_count": sum(1 for s in suggestions["pending"] 
                                             if s.get("confidence", 1.0) < 0.6),
                    "pending_suggestions": len(suggestions["pending"])
                },
                "improvement_areas": [],
                "suggested_actions": []
            }
            
            # Analyze areas needing improvement
            for intent, count in stats["intent_counts"].items():
                avg_confidence = self._calculate_avg_confidence(intent)
                if avg_confidence < 0.7:
                    report["improvement_areas"].append({
                        "intent": intent,
                        "avg_confidence": avg_confidence,
                        "usage_count": count
                    })
            
            # Generate action items
            report["suggested_actions"] = self._generate_action_items(report["improvement_areas"])
            
            return report
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return {}

    def _calculate_avg_confidence(self, intent: str) -> float:
        """Calculate average confidence for an intent"""
        try:
            with open(self.unmatched_file, 'r') as f:
                reader = csv.DictReader(f)
                confidences = [float(row['confidence']) for row in reader 
                             if row['detected_intent'] == intent]
                return sum(confidences) / len(confidences) if confidences else 1.0
        except Exception:
            return 1.0

    def _generate_action_items(self, improvement_areas: List[Dict]) -> List[Dict]:
        """Generate specific action items from improvement areas"""
        actions = []
        for area in improvement_areas:
            actions.append({
                "intent": area["intent"],
                "action": "Add more patterns",
                "priority": "high" if area["avg_confidence"] < 0.5 else "medium",
                "suggestion": f"Improve {area['intent']} intent recognition patterns"
            })
        return actions 

    async def generate_daily_report(self):
        """Generate and save daily report, keeping only latest 10"""
        try:
            report = await self.get_improvement_report()
            
            # Create dated report file
            date_str = datetime.now().strftime("%Y-%m-%d")
            report_file = self.reports_dir / f"intent_report_{date_str}.json"
            summary_file = self.reports_dir / f"summary_{date_str}.txt"
            
            # Save current report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save current summary
            with open(summary_file, 'w') as f:
                f.write(f"Intent Learning Report - {date_str}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Summary:\n")
                f.write(f"- Total Interactions: {report['summary']['total_interactions']}\n")
                f.write(f"- Low Confidence Count: {report['summary']['low_confidence_count']}\n")
                f.write(f"- Pending Suggestions: {report['summary']['pending_suggestions']}\n\n")
                
                f.write("Intent Distribution:\n")
                for intent, count in report['summary']['intent_distribution'].items():
                    f.write(f"- {intent}: {count}\n")
                
                f.write("\nImprovement Areas:\n")
                for area in report['improvement_areas']:
                    f.write(f"- {area['intent']} (conf: {area['avg_confidence']:.2f})\n")
                
                f.write("\nSuggested Actions:\n")
                for action in report['suggested_actions']:
                    f.write(f"- [{action['priority']}] {action['suggestion']}\n")
            
            # Cleanup old reports
            await self._cleanup_old_reports()
            
            print(f"\nDaily report generated: {report_file}")
            print(f"Summary created: {summary_file}")
            
        except Exception as e:
            print(f"Error generating daily report: {e}")

    async def _cleanup_old_reports(self):
        """Maintain only the configured number of latest reports"""
        try:
            keep_latest = self.config["reports"]["keep_latest"]
            
            # Get all report files
            json_reports = sorted(self.reports_dir.glob("intent_report_*.json"))
            text_summaries = sorted(self.reports_dir.glob("summary_*.txt"))
            
            # Remove old JSON reports
            if len(json_reports) > keep_latest:
                for old_report in json_reports[:-keep_latest]:
                    old_report.unlink()
                    print(f"Removed old report: {old_report.name}")
            
            # Remove old text summaries
            if len(text_summaries) > keep_latest:
                for old_summary in text_summaries[:-keep_latest]:
                    old_summary.unlink()
                    print(f"Removed old summary: {old_summary.name}")
                    
        except Exception as e:
            print(f"Error cleaning up old reports: {e}")

    async def start_reporting_schedule(self):
        """Start scheduled report generation"""
        while True:
            now = datetime.now()
            report_time = datetime.strptime(
                self.config["reports"]["generate_time"], 
                "%H:%M"
            ).time()
            
            # Generate report at configured time
            if now.hour == report_time.hour and now.minute == report_time.minute:
                await self.generate_daily_report()
            await asyncio.sleep(60)  # Check every minute

    def _load_suggestions(self) -> dict:
        """Load existing suggestions from file"""
        try:
            if self.suggestions_file.exists():
                with open(self.suggestions_file, 'r') as f:
                    return json.load(f)
            else:
                # Return default structure if file doesn't exist
                return {
                    "pending": [],
                    "approved": [],
                    "rejected": []
                }
        except Exception as e:
            print(f"Error loading suggestions: {e}")
            return {
                "pending": [],
                "approved": [],
                "rejected": []
            }

    def _load_stats(self) -> dict:
        """Load existing stats from file"""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            else:
                # Return default structure if file doesn't exist
                return {
                    "intent_counts": {},
                    "confidence_levels": {},
                    "common_patterns": {},
                    "improvement_areas": []
                }
        except Exception as e:
            print(f"Error loading stats: {e}")
            return {
                "intent_counts": {},
                "confidence_levels": {},
                "common_patterns": {},
                "improvement_areas": []
            }

    async def get_intent_scores(self, text: str) -> dict:
        """Get scores for each possible intent"""
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
            
        # Timer patterns
        if any(word in text_lower for word in ["timer", "remind", "minutes", "seconds"]):
            scores["timer"] += 1.5
            
        # Conversation patterns
        if any(word in text_lower for word in ["hello", "hi", "hey", "thanks", "thank you"]):
            scores["conversation"] += 1.0
            
        # Log the scores for debugging
        print("\nIntent Scores:")
        for intent, score in scores.items():
            print(f"  {intent}: {score:.2f}")
            
        return scores

    async def learn_from_interaction(self, text: str, intent: str, success: bool):
        """Learn from user interactions"""
        try:
            # Load existing suggestions
            suggestions = []
            if self.suggestions_file.exists():
                with open(self.suggestions_file, 'r') as f:
                    suggestions = json.load(f)
            
            # Add new interaction
            suggestions.append({
                "text": text,
                "intent": intent,
                "success": success,
                "timestamp": time.time()
            })
            
            # Save updated suggestions
            with open(self.suggestions_file, 'w') as f:
                json.dump(suggestions, f, indent=2)
                
        except Exception as e:
            print(f"Error learning from interaction: {e}")