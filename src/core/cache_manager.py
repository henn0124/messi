from typing import Optional, Dict, Any
import json
from pathlib import Path
import time
import hashlib

class ResponseCache:
    def __init__(self):
        self.cache_dir = Path("src/cache/responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.config = {
            "max_age": 24 * 60 * 60,  # 24 hours in seconds
            "max_size": 1000,         # Maximum number of cached responses
            "min_similarity": 0.85,    # Threshold for similar questions
        }
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "similar_matches": 0
        }
        
        # Context tracking for better matches
        self.context_history = []
    
    def _generate_cache_key(self, query: str, context: Dict = None) -> str:
        """Generate a unique cache key from query and context"""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Include relevant context in key generation
        if context:
            context_str = json.dumps({
                k: v for k, v in context.items() 
                if k in ['topic', 'previous_question', 'mentioned_entities']
            }, sort_keys=True)
        else:
            context_str = ""
        
        # Generate hash
        key_string = f"{normalized_query}:{context_str}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get_cached_response(self, query: str, context: Dict = None) -> Optional[Dict]:
        """Get cached response if available and valid"""
        cache_key = self._generate_cache_key(query, context)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid
                if time.time() - cached_data['timestamp'] < self.config['max_age']:
                    self.stats['hits'] += 1
                    print("✓ Cache hit!")
                    return cached_data['response']
                
                # Check for similar questions in context
                if context and self._check_similar_context(query, context, cached_data):
                    self.stats['similar_matches'] += 1
                    print("✓ Similar context match!")
                    return self._adapt_response(cached_data['response'], context)
            
            except Exception as e:
                print(f"Error reading cache: {e}")
        
        self.stats['misses'] += 1
        return None
    
    async def cache_response(self, query: str, response: Dict, context: Dict = None):
        """Cache a new response"""
        try:
            cache_key = self._generate_cache_key(query, context)
            cache_data = {
                'query': query,
                'response': response,
                'context': context,
                'timestamp': time.time()
            }
            
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            # Maintain cache size limit
            await self._cleanup_old_cache()
            
        except Exception as e:
            print(f"Error caching response: {e}")
    
    def _check_similar_context(self, query: str, current_context: Dict, cached_data: Dict) -> bool:
        """Check if cached response is from similar context"""
        if not cached_data.get('context'):
            return False
            
        cached_context = cached_data['context']
        
        # Check topic similarity
        if current_context.get('topic') == cached_context.get('topic'):
            return True
            
        # Check entity overlap
        current_entities = set(current_context.get('mentioned_entities', []))
        cached_entities = set(cached_context.get('mentioned_entities', []))
        entity_overlap = len(current_entities & cached_entities) / len(current_entities | cached_entities)
        
        return entity_overlap > self.config['min_similarity']
    
    def _adapt_response(self, cached_response: Dict, new_context: Dict) -> Dict:
        """Adapt cached response to new context"""
        # Clone response to avoid modifying cache
        adapted = dict(cached_response)
        
        # Update context-specific elements
        if 'context' in adapted:
            adapted['context'].update(new_context)
        
        return adapted
    
    async def _cleanup_old_cache(self):
        """Remove old cache entries"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            if len(cache_files) > self.config['max_size']:
                # Sort by modification time
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                
                # Remove oldest files
                for file in cache_files[:-self.config['max_size']]:
                    file.unlink()
                    
        except Exception as e:
            print(f"Error cleaning cache: {e}") 
    
    async def get_relevant_context(self, query: str, current_topic: str, entities: set) -> list:
        """Get relevant cached context for the current query"""
        relevant_context = []
        
        try:
            # Scan cache directory
            cache_files = list(self.cache_dir.glob("*.json"))
            
            for cache_file in cache_files:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check relevance
                if self._is_relevant(query, cached_data, current_topic, entities):
                    relevant_context.append({
                        "topic": cached_data["context"].get("topic", "unknown"),
                        "summary": self._create_context_summary(cached_data),
                        "timestamp": cached_data["timestamp"]
                    })
            
            # Sort by relevance and recency
            relevant_context.sort(
                key=lambda x: (
                    self._calculate_relevance(x, current_topic, entities),
                    x["timestamp"]
                ),
                reverse=True
            )
            
            # Return top 3 most relevant contexts
            return relevant_context[:3]
            
        except Exception as e:
            print(f"Error getting relevant context: {e}")
            return []
    
    def _is_relevant(self, query: str, cached_data: Dict, current_topic: str, entities: set) -> bool:
        """Check if cached data is relevant to current query"""
        cached_topic = cached_data["context"].get("topic")
        cached_entities = set(cached_data["context"].get("mentioned_entities", []))
        
        # Check topic match
        if cached_topic and cached_topic == current_topic:
            return True
            
        # Check entity overlap
        if cached_entities & entities:
            return True
            
        # Check text similarity (simple word overlap)
        query_words = set(query.lower().split())
        cached_words = set(cached_data["query"].lower().split())
        word_overlap = len(query_words & cached_words) / len(query_words | cached_words)
        
        return word_overlap > self.config["min_similarity"]
    
    def _create_context_summary(self, cached_data: Dict) -> str:
        """Create a brief summary of cached response"""
        response = cached_data["response"]["text"]
        # Take first sentence or first 100 characters
        summary = response.split('.')[0] if '.' in response else response[:100]
        return f"{summary}..."
    
    def _calculate_relevance(self, context: Dict, current_topic: str, entities: set) -> float:
        """Calculate relevance score for sorting"""
        score = 0.0
        
        # Topic match is highest priority
        if context["topic"] == current_topic:
            score += 1.0
            
        # Entity overlap
        context_entities = set(context.get("entities", []))
        entity_overlap = len(entities & context_entities) / len(entities | context_entities) if entities else 0
        score += entity_overlap * 0.5
        
        # Recency bonus (within last hour)
        time_diff = time.time() - context["timestamp"]
        if time_diff < 3600:  # 1 hour
            score += 0.3
            
        return score