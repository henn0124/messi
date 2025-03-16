"""
Content Manager for Messi Assistant
---------------------------------
Manages Fun Facts and Kids Jokes content delivery.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

class ContentManager:
    def __init__(self, content_dir: str = "data/content"):
        self.content_dir = Path(content_dir)
        self.fun_facts: Dict[str, List[str]] = {}
        self.jokes: List[Dict[str, str]] = []
        self.load_content()

    def load_content(self):
        """Load fun facts and jokes from JSON files"""
        try:
            # Load fun facts
            facts_file = self.content_dir / "fun_facts.json"
            if facts_file.exists():
                with open(facts_file, 'r') as f:
                    self.fun_facts = json.load(f)
            
            # Load jokes
            jokes_file = self.content_dir / "kids_jokes.json"
            if jokes_file.exists():
                with open(jokes_file, 'r') as f:
                    self.jokes = json.load(f)
        except Exception as e:
            print(f"Error loading content: {e}")
            # Initialize with empty data if files don't exist
            self.fun_facts = {}
            self.jokes = []

    def get_random_fact(self, category: Optional[str] = None) -> str:
        """Get a random fun fact, optionally from a specific category"""
        try:
            if category and category in self.fun_facts:
                facts = self.fun_facts[category]
            else:
                # Get a fact from any category
                facts = [fact for facts in self.fun_facts.values() for fact in facts]
            
            return random.choice(facts) if facts else "I don't have any fun facts right now!"
        except Exception as e:
            print(f"Error getting fun fact: {e}")
            return "Oops! I had trouble finding a fun fact!"

    def get_random_joke(self) -> Dict[str, str]:
        """Get a random joke with setup and punchline"""
        try:
            if self.jokes:
                joke = random.choice(self.jokes)
                return {
                    "setup": joke["setup"],
                    "punchline": joke["punchline"]
                }
            return {
                "setup": "Why did the assistant look sad?",
                "punchline": "Because it couldn't find any jokes to tell!"
            }
        except Exception as e:
            print(f"Error getting joke: {e}")
            return {
                "setup": "What did the assistant say when it broke?",
                "punchline": "Oops! Something went wrong!"
            }

    def get_categories(self) -> List[str]:
        """Get available fun fact categories"""
        return list(self.fun_facts.keys())

    def add_fact(self, category: str, fact: str) -> bool:
        """Add a new fun fact to a category"""
        try:
            if category not in self.fun_facts:
                self.fun_facts[category] = []
            self.fun_facts[category].append(fact)
            self._save_facts()
            return True
        except Exception as e:
            print(f"Error adding fact: {e}")
            return False

    def add_joke(self, setup: str, punchline: str) -> bool:
        """Add a new joke"""
        try:
            self.jokes.append({
                "setup": setup,
                "punchline": punchline
            })
            self._save_jokes()
            return True
        except Exception as e:
            print(f"Error adding joke: {e}")
            return False

    def _save_facts(self):
        """Save fun facts to file"""
        try:
            self.content_dir.mkdir(parents=True, exist_ok=True)
            with open(self.content_dir / "fun_facts.json", 'w') as f:
                json.dump(self.fun_facts, f, indent=2)
        except Exception as e:
            print(f"Error saving facts: {e}")

    def _save_jokes(self):
        """Save jokes to file"""
        try:
            self.content_dir.mkdir(parents=True, exist_ok=True)
            with open(self.content_dir / "kids_jokes.json", 'w') as f:
                json.dump(self.jokes, f, indent=2)
        except Exception as e:
            print(f"Error saving jokes: {e}") 