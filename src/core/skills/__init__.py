from pathlib import Path
from typing import Dict
import importlib
import json

class SkillManager:
    def __init__(self):
        self.skills = {}
        self._load_skills()
    
    def _load_skills(self):
        """Dynamically load all skills from skills directory"""
        skills_dir = Path(__file__).parent / "available"
        for skill_file in skills_dir.glob("*.py"):
            if skill_file.stem != "__init__":
                module = importlib.import_module(f"core.skills.available.{skill_file.stem}")
                if hasattr(module, "skill_manifest"):
                    self.skills[module.skill_manifest["name"]] = module
    
    async def execute(self, intent: Dict):
        """Execute the appropriate skill based on intent"""
        skill_name = intent.get("skill")
        if skill_name in self.skills:
            return await self.skills[skill_name].handle(intent)
        return {"error": "Skill not found"} 