from pathlib import Path
from typing import Dict
import importlib
import json

class SkillManager:
    def __init__(self):
        self.skills = {}
        self._load_skills()
    
    def _load_skills(self):
        """Dynamically load and instantiate all skills"""
        skills_dir = Path(__file__).parent / "available"
        print("\nLoading skills:")
        
        for skill_file in skills_dir.glob("*.py"):
            if skill_file.stem != "__init__":
                try:
                    # Import the module
                    module = importlib.import_module(f"core.skills.available.{skill_file.stem}")
                    
                    # Get the manifest
                    if hasattr(module, "skill_manifest"):
                        skill_name = module.skill_manifest["name"]
                        skill_class = getattr(module, skill_file.stem.title().replace('_', ''))
                        self.skills[skill_name] = skill_class()
                        print(f"✓ Loaded skill: {skill_name}")
                    else:
                        print(f"✗ No manifest for: {skill_file.stem}")
                        
                except Exception as e:
                    print(f"✗ Error loading {skill_file.stem}: {e}")

    def get_skill(self, intent: str):
        """Get the appropriate skill for an intent"""
        # Map intents to skills
        intent_to_skill = {
            "education": "education",
            "story": "bedtime_story",
            "timer": "timer",
            "conversation": "conversation"
        }
        
        # Get skill name for intent
        skill_name = intent_to_skill.get(intent)
        if skill_name:
            return self.skills.get(skill_name)
            
        print(f"No skill found for intent: {intent}")
        print(f"Available skills: {list(self.skills.keys())}")
        return None

    def list_skills(self):
        """Get list of available skills"""
        return list(self.skills.keys())

    async def execute(self, intent: Dict):
        """Execute the appropriate skill based on intent"""
        try:
            skill_name = intent.get("skill")
            if skill_name in self.skills:
                print(f"\nExecuting skill: {skill_name}")
                return await self.skills[skill_name].handle(intent)
            else:
                print(f"Skill not found: {skill_name}")
                available = list(self.skills.keys())
                print(f"Available skills: {available}")
                return {
                    "text": "I'm not sure how to help with that.",
                    "context": "error"
                }
        except Exception as e:
            print(f"Error executing skill: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "I had trouble processing that request.",
                "context": "error"
            } 