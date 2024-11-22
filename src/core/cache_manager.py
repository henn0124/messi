from typing import Optional, Dict, Any
import json
from pathlib import Path
import time
import hashlib
import aiofiles

class ResponseCache:
    def __init__(self, max_size: int = 1024 * 1024):
        self.cache_dir = Path("src/cache/responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.config = {
            "max_age": 24 * 60 * 60,  # 24 hours in seconds
            "max_size": max_size,      # Maximum cache size in bytes
            "min_similarity": 0.85     # Threshold for similar questions
        }
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "similar_matches": 0
        }
    
    async def get(self, key: str) -> Optional[Dict]:
        """Get cached response"""
        try:
            cache_key = self._generate_cache_key(key)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                async with aiofiles.open(cache_file, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    # Check if cache is still valid
                    if time.time() - data['timestamp'] < self.config['max_age']:
                        self.stats['hits'] += 1
                        return data['response']
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Dict):
        """Set cache value"""
        try:
            cache_key = self._generate_cache_key(key)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            data = {
                'key': key,
                'response': value,
                'timestamp': time.time()
            }
            
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(data))
                
            await self._cleanup_old_cache()
            
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate cache key from input"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def _cleanup_old_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.json"):
                if current_time - cache_file.stat().st_mtime > self.config['max_age']:
                    cache_file.unlink()
        except Exception as e:
            print(f"Cache cleanup error: {e}")
    
    async def get_relevant_context(self, query: str, current_topic: str = None, entities: set = None) -> list:
        """Get relevant cached context for the current query"""
        try:
            relevant_context = []
            
            # Scan cache directory
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    async with aiofiles.open(cache_file, 'r') as f:
                        content = await f.read()
                        cached_data = json.loads(content)
                        
                        # Check relevance
                        if self._is_relevant(query, cached_data, current_topic, entities):
                            relevant_context.append({
                                "topic": cached_data.get("topic", "unknown"),
                                "text": cached_data.get("response", {}).get("text", ""),
                                "timestamp": cached_data.get("timestamp", 0)
                            })
                except Exception as e:
                    print(f"Error reading cache file {cache_file}: {e}")
                    continue
            
            # Sort by timestamp (most recent first)
            relevant_context.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Return top 3 most relevant contexts
            return relevant_context[:3]
            
        except Exception as e:
            print(f"Error getting relevant context: {e}")
            return []
    
    def _is_relevant(self, query: str, cached_data: Dict, current_topic: str = None, entities: set = None) -> bool:
        """Check if cached data is relevant to current query"""
        try:
            # Check topic match
            if current_topic and cached_data.get("topic") == current_topic:
                return True
            
            # Check text similarity
            cached_text = cached_data.get("response", {}).get("text", "").lower()
            query = query.lower()
            
            words_query = set(query.split())
            words_cached = set(cached_text.split())
            
            # Calculate word overlap
            if words_query and words_cached:
                overlap = len(words_query & words_cached) / len(words_query | words_cached)
                if overlap > self.config["min_similarity"]:
                    return True
            
            # Check entity overlap
            if entities and cached_data.get("entities"):
                cached_entities = set(cached_data["entities"])
                if entities & cached_entities:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking relevance: {e}")
            return False