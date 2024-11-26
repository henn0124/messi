"""
Continuous Learning System
-------------------------
Manages continuous learning and improvement of context and conversation handling.

Key Features:
1. Pattern Learning:
   - Tracks successful conversation patterns
   - Learns from user engagement
   - Updates context transitions

2. Automated Updates:
   - Self-updating configuration
   - Persistent learning storage
   - Periodic optimization

3. Analytics:
   - Conversation success metrics
   - Context transition analysis
   - User engagement tracking

4. Adaptive Behavior:
   - Dynamic context switching
   - Improved topic detection
   - Better conversation flow
"""

from typing import Dict, List, Set, TYPE_CHECKING
import json
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import numpy as np
import yaml
from .config import Settings

if TYPE_CHECKING:
    from .context_manager import ContextManager
    from .conversation_manager import ConversationManager

class LearningManager:
    def __init__(self):
        self.settings = Settings()
        self.learning_enabled = self.settings.learning.enabled
        
        # Initialize patterns dict
        self.patterns = {
            "context_transitions": {},
            "topic_relationships": {},
            "engagement_patterns": {},
            "conversation_flows": {}
        }
        
        if self.learning_enabled:
            # Get paths from settings
            self.learning_file = Path(self.settings.learning.storage["data_path"])
            self.config_file = Path(self.settings.learning.storage["config_path"])
            
            # Create directories
            self.learning_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize data
            self._initialize_learning_data()
            print(f"Learning system initialized with data at: {self.learning_file}")
        else:
            print("Learning system disabled via config")
    
    async def initialize(self):
        """Initialize async components"""
        if self.learning_enabled:
            # Start learning loop
            asyncio.create_task(self._continuous_learning_loop())
            print("Learning system async components initialized")
    
    async def record_exchange(self, exchange_data: Dict):
        """Record exchange with toggle check"""
        if not self.learning_enabled:
            return
            
        try:
            # Extract key information
            context = exchange_data.get("context")
            previous_context = exchange_data.get("previous_context")
            user_text = exchange_data.get("user_text")
            response = exchange_data.get("response")
            engagement = exchange_data.get("engagement_score", 0)
            
            # Update transition patterns
            if previous_context and context != previous_context:
                self._update_transition_pattern(previous_context, context, engagement)
            
            # Update topic relationships
            topics = self._extract_topics(user_text)
            self._update_topic_relationships(topics, engagement)
            
            # Update engagement patterns
            self._update_engagement_patterns(exchange_data)
            
            # Save if significant learning occurred
            if self._should_save_learning():
                await self._save_learning_data()
                
        except Exception as e:
            print(f"Error recording exchange: {e}")
    
    async def get_learned_patterns(self, context: str) -> Dict:
        """Get learned patterns for a specific context"""
        return {
            "transitions": self.patterns["context_transitions"].get(context, {}),
            "topics": self.patterns["topic_relationships"].get(context, {}),
            "engagement": self.patterns["engagement_patterns"].get(context, {}),
            "flows": self.patterns["conversation_flows"].get(context, {})
        }
    
    async def optimize_context_manager(self, context_manager) -> None:
        """Update context manager with learned patterns"""
        try:
            # Update transition weights
            for context, transitions in self.patterns["context_transitions"].items():
                if context in context_manager.context_types:
                    context_manager.context_types[context]["transitions"] = [
                        t for t, w in sorted(transitions.items(), key=lambda x: x[1], reverse=True)
                    ]
            
            # Update related topics
            for context, topics in self.patterns["topic_relationships"].items():
                if context in context_manager.context_types:
                    context_manager.context_types[context]["related"] = [
                        t for t, w in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]
                    ]
                    
        except Exception as e:
            print(f"Error optimizing context manager: {e}")
    
    async def optimize_conversation_manager(self, conversation_manager) -> None:
        """Update conversation manager with learned patterns"""
        try:
            # Update engagement thresholds
            engagement_patterns = self.patterns["engagement_patterns"]
            if engagement_patterns:
                avg_engagement = np.mean(list(engagement_patterns.values()))
                conversation_manager.MIN_RESPONSE_LENGTH = max(2, int(avg_engagement * 0.5))
                
            # Update timeout thresholds
            flow_patterns = self.patterns["conversation_flows"]
            if flow_patterns:
                avg_pause = np.mean([p["pause_duration"] for p in flow_patterns.values()])
                conversation_manager.INACTIVITY_THRESHOLD = max(3.0, avg_pause * 0.8)
                
        except Exception as e:
            print(f"Error optimizing conversation manager: {e}")
    
    async def _continuous_learning_loop(self):
        """Learning loop with toggle check"""
        while True:
            if not self.learning_enabled:
                await asyncio.sleep(60)  # Check toggle every minute
                continue
                
            try:
                # Analyze patterns
                self._analyze_patterns()
                
                # Update configuration
                await self._update_dynamic_config()
                
                # Save learning data
                await self._save_learning_data()
                
                # Wait for next learning cycle
                await asyncio.sleep(3600)  # Hourly updates
                
            except Exception as e:
                print(f"Error in learning loop: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    def _analyze_patterns(self):
        """Analyze recorded patterns for insights"""
        # Analyze context transitions
        for context, transitions in self.patterns["context_transitions"].items():
            successful = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            self.metrics[f"successful_transitions_{context}"] = successful[:5]
        
        # Analyze topic relationships
        for context, topics in self.patterns["topic_relationships"].items():
            related = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            self.metrics[f"related_topics_{context}"] = related[:10]
    
    async def _update_dynamic_config(self):
        """Update dynamic configuration based on learning"""
        config = {
            "learning": {
                "patterns": self.patterns,
                "metrics": self.metrics,
                "last_updated": datetime.now().isoformat()
            }
        }
        
        # Save to YAML
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f) 
    
    async def get_learning_status(self) -> Dict:
        """Get current learning status and file locations"""
        return {
            "files": {
                "learning_data": str(self.learning_file),
                "dynamic_config": str(self.config_file),
                "last_updated": datetime.now().isoformat()
            },
            "metrics": self.metrics,
            "active_patterns": {
                k: len(v) for k, v in self.patterns.items()
            },
            "optimizations": {
                "context_manager": {
                    "transitions_updated": len(self.patterns["context_transitions"]),
                    "topics_updated": len(self.patterns["topic_relationships"])
                },
                "conversation_manager": {
                    "engagement_patterns": len(self.patterns["engagement_patterns"]),
                    "flow_patterns": len(self.patterns["conversation_flows"])
                }
            }
        }
    
    def _log_updates(self, update_type: str, details: Dict):
        """Log what's being updated"""
        log_file = Path("logs/learning_updates.log")
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "type": update_type,
            "details": details
        }
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
    
    async def _save_learning_data(self):
        """Save learning data to both JSON and YAML"""
        try:
            # Save detailed learning data to JSON
            self.learning_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_file, 'w') as f:
                json.dump({
                    "patterns": self.patterns,
                    "metrics": self.metrics,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            # Update dynamic config YAML
            config = {
                "learning": {
                    "patterns": {
                        "context_transitions": self.patterns["context_transitions"],
                        "topic_relationships": self.patterns["topic_relationships"],
                        "engagement_patterns": self.patterns["engagement_patterns"]
                    },
                    "metrics": self.metrics,
                    "thresholds": self._calculate_thresholds(),
                    "optimizations": self._get_optimizations(),
                    "parameters": self._get_learning_parameters()
                }
            }
            
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Log the update
            self._log_updates("config_update", {
                "patterns_updated": len(self.patterns),
                "metrics_updated": len(self.metrics),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error saving learning data: {e}")
            
    def _calculate_thresholds(self) -> Dict:
        """Calculate adaptive thresholds based on learning"""
        return {
            "min_response_length": max(2, int(self.metrics["avg_conversation_length"] * 0.3)),
            "max_silence": min(10.0, self.metrics["avg_conversation_length"] * 0.2),
            "context_switch_threshold": 0.7,  # Adjusted based on success rate
            "engagement_threshold": 0.6
        }
        
    def _get_optimizations(self) -> Dict:
        """Get current optimization settings"""
        return {
            "context_weights": {
                "previous_context": 0.4,
                "current_entities": 0.3,
                "user_engagement": 0.3
            },
            "transition_rules": {
                "min_confidence": 0.6,
                "max_transitions": 3,
                "cooldown_period": 60
            }
        }
        
    def _get_learning_parameters(self) -> Dict:
        """Get current learning parameters"""
        return {
            "learning_rate": 0.1,
            "decay_factor": 0.95,
            "update_frequency": 3600,
            "min_samples": 10
        }
    
    async def get_learning_metrics(self) -> Dict:
        """Get current learning progress and effectiveness"""
        try:
            # Calculate improvement metrics
            improvements = {
                "context_accuracy": self._calculate_context_accuracy(),
                "engagement_improvement": self._calculate_engagement_improvement(),
                "response_quality": self._calculate_response_quality()
            }
            
            # Get learning status
            status = await self.get_learning_status()
            
            return {
                "improvements": improvements,
                "status": status,
                "config": await self.get_current_config(),
                "recommendations": self._get_optimization_recommendations()
            }
        except Exception as e:
            print(f"Error getting metrics: {e}")
    
    def _initialize_learning_data(self):
        """Initialize or load learning data"""
        try:
            # Initialize metrics
            self.metrics = {
                "successful_exchanges": 0,
                "failed_exchanges": 0,
                "avg_conversation_length": 0,
                "topic_success_rates": {}
            }
            
            # Try to load existing data
            if self.learning_file.exists():
                try:
                    with open(self.learning_file, 'r') as f:
                        data = json.load(f)
                        self.learning_data = data
                        # Update metrics from loaded data
                        self.metrics.update(data.get("metrics", {}))
                except json.JSONDecodeError:
                    print("Warning: Learning data file corrupted, creating new")
                    self._create_initial_learning_data()
            else:
                print("Creating new learning data file")
                self._create_initial_learning_data()
                
        except Exception as e:
            print(f"Error initializing learning data: {e}")
            self._create_initial_learning_data()
    
    def _create_initial_learning_data(self):
        """Create initial learning data structure"""
        self.learning_data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "exchanges": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "by_context": {}
            },
            "patterns": {
                "context_transitions": {},
                "topic_relationships": {},
                "engagement_patterns": {},
                "conversation_flows": {}
            },
            "learning_progress": {
                "context_accuracy": 0.0,
                "engagement_improvement": 0.0,
                "response_quality": 0.0
            },
            "metrics": self.metrics
        }
        
        # Ensure directory exists
        self.learning_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save initial data
        try:
            with open(self.learning_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
            print(f"Created new learning data file: {self.learning_file}")
        except Exception as e:
            print(f"Error creating learning data file: {e}")
    
    async def update_learning_data(self, exchange_data: Dict):
        """Update learning data with new exchange"""
        try:
            # Update exchange counts
            self.learning_data["exchanges"]["total"] += 1
            if exchange_data.get("success", False):
                self.learning_data["exchanges"]["successful"] += 1
            else:
                self.learning_data["exchanges"]["failed"] += 1

            # Update context-specific data
            context = exchange_data.get("context", "unknown")
            if context not in self.learning_data["exchanges"]["by_context"]:
                self.learning_data["exchanges"]["by_context"][context] = {
                    "total": 0,
                    "successful": 0,
                    "patterns": []
                }
            
            context_data = self.learning_data["exchanges"]["by_context"][context]
            context_data["total"] += 1
            if exchange_data.get("success", False):
                context_data["successful"] += 1
            
            # Store pattern data
            context_data["patterns"].append({
                "timestamp": datetime.now().isoformat(),
                "text": exchange_data.get("text", ""),
                "entities": exchange_data.get("entities", []),
                "engagement": exchange_data.get("engagement_score", 0)
            })

            # Update learning progress
            self._update_learning_progress()

            # Save if needed
            if self._should_save():
                await self._save_learning_data()

        except Exception as e:
            print(f"Error updating learning data: {e}")

    def _update_learning_progress(self):
        """Update learning progress metrics"""
        try:
            # Calculate context accuracy
            total = self.learning_data["exchanges"]["total"]
            if total > 0:
                success_rate = (
                    self.learning_data["exchanges"]["successful"] / total
                )
                self.learning_data["learning_progress"]["context_accuracy"] = success_rate

            # Calculate engagement improvement
            engagement_scores = []
            for context in self.learning_data["exchanges"]["by_context"].values():
                for pattern in context["patterns"]:
                    engagement_scores.append(pattern.get("engagement", 0))
            
            if engagement_scores:
                avg_engagement = sum(engagement_scores) / len(engagement_scores)
                self.learning_data["learning_progress"]["engagement_improvement"] = avg_engagement

        except Exception as e:
            print(f"Error updating learning progress: {e}")
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        try:
            if not text:  # Handle None or empty text
                return []
                
            # Simple topic extraction based on keywords
            topics = []
            text_lower = text.lower()
            
            # Common educational topics
            educational_topics = {
                "math": ["mathematics", "algebra", "geometry"],
                "science": ["physics", "chemistry", "biology"],
                "history": ["historical", "ancient", "period"],
                "geography": ["countries", "capitals", "cities"],
                "language": ["grammar", "vocabulary", "writing"]
            }
            
            for topic, keywords in educational_topics.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
    
    def _update_topic_relationships(self, topics: List[str], engagement: float):
        """Update topic relationship patterns"""
        try:
            if not topics:
                return
                
            # Update each topic's relationships
            for topic in topics:
                if topic not in self.patterns["topic_relationships"]:
                    self.patterns["topic_relationships"][topic] = {}
                    
                # Update relationships with other topics
                for other_topic in topics:
                    if other_topic != topic:
                        if other_topic not in self.patterns["topic_relationships"][topic]:
                            self.patterns["topic_relationships"][topic][other_topic] = 0.0
                        
                        # Increase relationship strength based on engagement
                        self.patterns["topic_relationships"][topic][other_topic] += (
                            engagement * self.settings.learning.parameters["learning_rate"]
                        )
                        
        except Exception as e:
            print(f"Error updating topic relationships: {e}")
    
    def _update_engagement_patterns(self, exchange_data: Dict):
        """Update engagement patterns"""
        try:
            context = exchange_data.get("context", "unknown")
            engagement = exchange_data.get("engagement_score", 0)
            
            if context not in self.patterns["engagement_patterns"]:
                self.patterns["engagement_patterns"][context] = {
                    "total_engagement": 0,
                    "interactions": 0,
                    "avg_engagement": 0
                }
            
            pattern = self.patterns["engagement_patterns"][context]
            pattern["total_engagement"] += engagement
            pattern["interactions"] += 1
            pattern["avg_engagement"] = pattern["total_engagement"] / pattern["interactions"]
            
        except Exception as e:
            print(f"Error updating engagement patterns: {e}")