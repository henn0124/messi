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
import aiofiles

if TYPE_CHECKING:
    from .context_manager import ContextManager
    from .conversation_manager import ConversationManager

class LearningManager:
    def __init__(self):
        self.settings = Settings()
        self.learning_enabled = self.settings.learning.enabled
        
        # Initialize components
        self.context_manager = None  # Will be set by MessiAssistant
        
        # Initialize patterns dict
        self.patterns = {
            "context_transitions": {},
            "topic_relationships": {},
            "engagement_patterns": {},
            "conversation_flows": {}
        }
        
        # Initialize metrics
        self.metrics = {
            "successful_exchanges": 0,
            "failed_exchanges": 0,
            "avg_conversation_length": 0,
            "topic_success_rates": {}
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
            
            # Update metrics from learning data
            if hasattr(self, 'learning_data'):
                self.metrics.update(self.learning_data.get("metrics", {}))
            
            print(f"Learning system initialized with data at: {self.learning_file}")
        else:
            print("Learning system disabled via config")
        
        # Add queue for deferred learning tasks
        self.learning_queue = asyncio.Queue()
        self.is_processing = False
        self.in_conversation = False  # Track conversation state
        
        # Track intent learning
        self.intent_learning = {
            "patterns": {},
            "success_rates": {},
            "transitions": {}
        }
    
    async def initialize(self):
        """Initialize async components"""
        if self.learning_enabled:
            # Start learning loop
            asyncio.create_task(self._continuous_learning_loop())
            print("Learning system async components initialized")
    
    async def record_exchange(self, exchange_data: Dict):
        """Queue exchange data for processing during downtime"""
        if not self.learning_enabled:
            return
            
        # Just queue the data, don't process now
        await self.learning_queue.put({
            "type": "exchange",
            "data": exchange_data,
            "timestamp": datetime.now().isoformat()
        })
        
    async def process_learning_queue(self):
        """Process queued learning tasks during system downtime"""
        # Don't process if conversation is active
        if self.in_conversation:
            return
            
        self.is_processing = True
        try:
            while not self.learning_queue.empty():
                # Check conversation state before each task
                if self.in_conversation:
                    print("Conversation started - pausing learning queue")
                    break
                    
                task = await self.learning_queue.get()
                
                if task["type"] == "exchange":
                    await self._process_exchange(task["data"])
                
                self.learning_queue.task_done()
                await asyncio.sleep(0.1)  # Allow other tasks to run
                
        except Exception as e:
            print(f"Error processing learning queue: {e}")
        finally:
            self.is_processing = False
            
    async def _process_exchange(self, exchange_data: Dict):
        """Process a single exchange (during downtime)"""
        try:
            # Extract key information
            context = exchange_data.get("context")
            previous_context = exchange_data.get("previous_context")
            user_text = exchange_data.get("user_text")
            response = exchange_data.get("response")
            engagement = exchange_data.get("engagement_score", 0)
            
            # Update patterns
            if previous_context and context != previous_context:
                self._update_transition_pattern(previous_context, context, engagement)
            
            topics = self._extract_topics(user_text)
            self._update_topic_relationships(topics, engagement)
            self._update_engagement_patterns(exchange_data)
            
            # Save if needed
            if self._should_save_learning():
                await self._save_learning_data()
                
        except Exception as e:
            print(f"Error processing exchange: {e}")
    
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
        """Save learning data including weights to disk"""
        try:
            # Update learning data with current weights and patterns
            self.learning_data.update({
                "last_updated": datetime.now().isoformat(),
                "intent_learning": {
                    "patterns": self.intent_learning["patterns"],
                    "success_rates": self.intent_learning["success_rates"],
                    "transitions": self.intent_learning["transitions"]
                },
                "weights": {
                    "patterns": self.get_pattern_weights(),
                    "context": self.context_manager.get_weights() if self.context_manager else {},
                    "last_updated": datetime.now().isoformat()
                }
            })
            
            # Save to JSON file
            async with aiofiles.open(self.learning_file, 'w') as f:
                await f.write(json.dumps(self.learning_data, indent=2))
            
            # Also update dynamic config
            config_data = {
                "learning": {
                    "weights": self.learning_data["weights"],
                    "patterns": self.intent_learning["patterns"],
                    "success_rates": self.intent_learning["success_rates"]
                }
            }
            
            async with aiofiles.open(self.config_file, 'w') as f:
                await f.write(yaml.dump(config_data, default_flow_style=False))
                
            print(f"Learning data saved - patterns: {len(self.intent_learning['patterns'])} success_rates: {len(self.intent_learning['success_rates'])}")
            
        except Exception as e:
            print(f"Error saving learning data: {e}")
    
    def _calculate_thresholds(self) -> Dict:
        """Calculate adaptive thresholds based on learning"""
        try:
            # Get metrics from learning data
            exchanges = self.learning_data["exchanges"]
            avg_length = self.learning_data.get("learning_progress", {}).get("avg_conversation_length", 0)
            
            return {
                "min_response_length": max(2, int(avg_length * 0.3)),
                "max_silence": min(10.0, avg_length * 0.2),
                "context_switch_threshold": 0.7,  # Adjusted based on success rate
                "engagement_threshold": 0.6
            }
            
        except Exception as e:
            print(f"Error calculating thresholds: {e}")
            # Return defaults
            return {
                "min_response_length": 2,
                "max_silence": 5.0,
                "context_switch_threshold": 0.7,
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
            # Try to load existing data
            if self.learning_file.exists():
                with open(self.learning_file, 'r') as f:
                    saved_data = json.load(f)
                    
                    # Restore intent learning data
                    if "intent_learning" in saved_data:
                        self.intent_learning = saved_data["intent_learning"]
                    
                    # Restore weights
                    if "weights" in saved_data:
                        self.weights = saved_data["weights"]
                        
                    # Update learning data
                    self.learning_data = saved_data
            else:
                # Create new learning data structure
                self._create_initial_learning_data()
                
            print(f"Loaded learning data with {len(self.intent_learning['patterns'])} patterns")
            
        except Exception as e:
            print(f"Error loading learning data: {e}")
            self._create_initial_learning_data()
    
    def _create_initial_learning_data(self):
        """Create initial learning data structure with all skills"""
        try:
            # Load skills config to get all patterns
            with open("config/skills_config.yaml") as f:
                skills_config = yaml.safe_load(f)
            
            # Initialize patterns for each skill
            initial_patterns = {}
            for intent, config in skills_config["intents"]["patterns"].items():
                initial_patterns[intent] = {
                    keyword: 0.5  # Start with neutral weight
                    for keyword in config["keywords"]
                }
            
            self.learning_data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "intent_learning": {
                    "patterns": initial_patterns,
                    "success_rates": {
                        intent: {
                            "successes": 0,
                            "total": 0,
                            "recent_success": 0.0
                        }
                        for intent in skills_config["intents"]["patterns"].keys()
                    },
                    "transitions": {
                        source: {
                            target: 0.5
                            for target in transitions
                        }
                        for source, transitions in skills_config["intents"]["transitions"].items()
                    }
                },
                "weights": {
                    "patterns": {
                        intent: skills_config["intents"]["weights"].get(intent, 0.5)
                        for intent in skills_config["intents"]["patterns"].keys()
                    },
                    "context": {
                        "previous_context": 0.4,
                        "current_entities": 0.3,
                        "user_engagement": 0.3
                    },
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            # Save initial data
            self.learning_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
                
            print(f"Created new learning data with {len(initial_patterns)} skills")
            
        except Exception as e:
            print(f"Error creating initial learning data: {e}")
            # Create minimal structure if config load fails
            self._create_minimal_learning_data()
    
    def _create_minimal_learning_data(self):
        """Create minimal learning data structure"""
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
    
    def _should_save_learning(self) -> bool:
        """Determine if learning data should be saved"""
        try:
            # Save if we have enough new data
            min_samples = self.settings.learning.parameters["min_samples"]
            total_exchanges = self.learning_data["exchanges"]["total"]
            
            # Save every min_samples exchanges
            return total_exchanges % min_samples == 0
            
        except Exception as e:
            print(f"Error checking save condition: {e}")
            return False
    
    async def record_intent_learning(self, learning_data: Dict):
        """Record intent learning data with dynamic weight adjustment"""
        try:
            intent = learning_data["intent"]
            success = learning_data["success"]
            
            # Update patterns
            if intent not in self.intent_learning["patterns"]:
                self.intent_learning["patterns"][intent] = {}
            
            # Get learning parameters
            learning_rate = self.settings.learning.parameters["learning_rate"]
            decay_factor = self.settings.learning.parameters["decay_factor"]
            
            # Update pattern weights
            patterns = learning_data["patterns"]
            for pattern, weight in patterns["weights"].items():
                if pattern not in self.intent_learning["patterns"][intent]:
                    self.intent_learning["patterns"][intent][pattern] = weight
                else:
                    current_weight = self.intent_learning["patterns"][intent][pattern]
                    
                    # Apply decay to old weight
                    current_weight *= decay_factor
                    
                    # Blend with new weight
                    if success:
                        # Increase influence of successful patterns
                        new_weight = current_weight + (learning_rate * (weight - current_weight))
                    else:
                        # Decrease influence of failed patterns
                        new_weight = current_weight - (learning_rate * current_weight)
                    
                    # Keep weights in valid range
                    self.intent_learning["patterns"][intent][pattern] = max(0.1, min(1.0, new_weight))
            
            # Update success rates
            if intent not in self.intent_learning["success_rates"]:
                self.intent_learning["success_rates"][intent] = {
                    "successes": 0,
                    "total": 0,
                    "recent_success": 0.0  # Track recent performance
                }
            
            rates = self.intent_learning["success_rates"][intent]
            rates["total"] += 1
            if success:
                rates["successes"] += 1
                rates["recent_success"] = rates["recent_success"] * decay_factor + learning_rate
            else:
                rates["recent_success"] = rates["recent_success"] * decay_factor
            
            # Save if needed
            if self._should_save_learning():
                await self._save_learning_data()
                
        except Exception as e:
            print(f"Error recording intent learning: {e}")
    
    def get_pattern_weights(self) -> Dict[str, float]:
        """Get learned pattern weights"""
        weights = {}
        
        for intent, data in self.intent_learning["patterns"].items():
            if intent in self.intent_learning["success_rates"]:
                rates = self.intent_learning["success_rates"][intent]
                success_rate = rates["successes"] / rates["total"] if rates["total"] > 0 else 0.5
                weights[intent] = 0.5 + (success_rate * 0.5)  # Scale from 0.5 to 1.0
                
        return weights
    
    def is_conversation_active(self) -> bool:
        """Check if conversation is active"""
        return self.in_conversation
    
    async def validate_learning_system(self) -> Dict:
        """Validate learning system components and autonomy"""
        try:
            validation = {
                "components": {
                    "intent_learning": False,
                    "pattern_learning": False,
                    "context_learning": False,
                    "queue_processing": False
                },
                "files": {
                    "learning_data": False,
                    "dynamic_config": False
                },
                "metrics": {
                    "patterns_count": 0,
                    "success_rates_count": 0,
                    "queue_size": 0
                },
                "autonomous_features": {
                    "pattern_discovery": False,
                    "weight_adjustment": False,
                    "context_optimization": False
                }
            }
            
            # Check components
            validation["components"]["intent_learning"] = hasattr(self, 'intent_learning')
            validation["components"]["pattern_learning"] = hasattr(self, 'patterns')
            validation["components"]["context_learning"] = self.context_manager is not None
            validation["components"]["queue_processing"] = hasattr(self, 'learning_queue')
            
            # Check files
            validation["files"]["learning_data"] = self.learning_file.exists()
            validation["files"]["dynamic_config"] = self.config_file.exists()
            
            # Check metrics
            validation["metrics"]["patterns_count"] = len(self.intent_learning.get("patterns", {}))
            validation["metrics"]["success_rates_count"] = len(self.intent_learning.get("success_rates", {}))
            validation["metrics"]["queue_size"] = self.learning_queue.qsize()
            
            # Check autonomous features
            validation["autonomous_features"]["pattern_discovery"] = (
                hasattr(self, '_discover_new_patterns') and 
                callable(getattr(self, '_discover_new_patterns'))
            )
            validation["autonomous_features"]["weight_adjustment"] = (
                hasattr(self, '_adjust_pattern_weight') and 
                callable(getattr(self, '_adjust_pattern_weight'))
            )
            validation["autonomous_features"]["context_optimization"] = (
                hasattr(self, 'optimize_context_manager') and 
                callable(getattr(self, 'optimize_context_manager'))
            )
            
            return validation
            
        except Exception as e:
            print(f"Error validating learning system: {e}")
            return {}