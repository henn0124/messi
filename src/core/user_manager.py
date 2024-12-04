from typing import Dict, Optional, List
import yaml
import json
from pathlib import Path
from datetime import datetime
import asyncio
from .config import Settings
from .logger import ConversationLogger

class UserManager:
    def __init__(self, settings: Settings, logger: ConversationLogger):
        self.settings = settings
        self.logger = logger
        self.config_file = Path("config/user_config.yaml")
        self.users_file = Path("data/users.json")
        self.current_user: Optional[Dict] = None
        self.active_sessions: Dict = {}
        
        # Load configurations
        self.load_config()
        self.load_users()
        
    def load_config(self):
        """Load user configuration settings"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.log_error("user_manager", e, {"context": "loading_config"})
            self.config = {"users": {"default": {}}}
            
    def load_users(self):
        """Load user data"""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            else:
                self.users = {"users": {}}
                self.save_users()
        except Exception as e:
            self.logger.log_error("user_manager", e, {"context": "loading_users"})
            self.users = {"users": {}}
            
    def save_users(self):
        """Save user data"""
        try:
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            self.logger.log_error("user_manager", e, {"context": "saving_users"})
    
    async def identify_user(self, voice_data: bytes) -> Dict:
        """Identify user from voice input"""
        try:
            # TODO: Implement voice identification
            # For now, return default user
            return self.get_default_user()
        except Exception as e:
            self.logger.log_error("user_manager", e, {"context": "voice_identification"})
            return self.get_default_user()
    
    def get_default_user(self) -> Dict:
        """Get default user profile"""
        return {
            "id": "default",
            "profile": self.config["users"]["default"],
            "type": "guest",
            "restrictions": [],
            "preferences": {}
        }
    
    def create_user(self, user_data: Dict) -> bool:
        """Create new user profile"""
        try:
            user_id = user_data.get("id")
            if not user_id:
                return False
                
            if user_id in self.users["users"]:
                return False
                
            self.users["users"][user_id] = {
                "profile": user_data.get("profile", {}),
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "type": user_data.get("type", "child"),
                "preferences": user_data.get("preferences", {}),
                "restrictions": user_data.get("restrictions", []),
                "voice_samples": []
            }
            
            self.save_users()
            return True
            
        except Exception as e:
            self.logger.log_error("user_manager", e, {"context": "create_user"})
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences and settings"""
        try:
            user = self.users["users"].get(user_id)
            if not user:
                return {}
                
            profile_type = user.get("type", "child")
            profile_settings = self.config["users"]["profiles"].get(profile_type, {})
            
            return {
                "preferences": user.get("preferences", {}),
                "restrictions": user.get("restrictions", []),
                "features": profile_settings.get("features", []),
                "content_filters": profile_settings.get("content_filters", [])
            }
            
        except Exception as e:
            self.logger.log_error("user_manager", e, {"context": "get_preferences"})
            return {}
    
    def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp"""
        try:
            if user_id in self.users["users"]:
                self.users["users"][user_id]["last_active"] = datetime.now().isoformat()
                self.save_users()
        except Exception as e:
            self.logger.log_error("user_manager", e, {"context": "update_activity"})
    
    def get_active_restrictions(self, user_id: str) -> List[str]:
        """Get active restrictions for user"""
        try:
            user = self.users["users"].get(user_id)
            if not user:
                return []
                
            profile_type = user.get("type", "child")
            profile_settings = self.config["users"]["profiles"].get(profile_type, {})
            
            restrictions = user.get("restrictions", []).copy()
            restrictions.extend(profile_settings.get("content_filters", []))
            
            return list(set(restrictions))  # Remove duplicates
            
        except Exception as e:
            self.logger.log_error("user_manager", e, {"context": "get_restrictions"})
            return [] 