from pydantic_settings import BaseSettings
from pathlib import Path
import yaml
import os
from pydantic import BaseModel, Field
from typing import Dict, Optional
import traceback

class LearningConfig(BaseModel):
    """Learning system configuration"""
    enabled: bool = True
    parameters: dict = {
        "learning_rate": 0.1,
        "decay_factor": 0.95,
        "update_frequency": 3600,
        "min_samples": 10
    }
    storage: dict = {
        "data_path": "cache/learning/learning.json",
        "config_path": "config/dynamic_config.yaml",
        "logs_path": "logs/learning_updates.log"
    }

class LoggingHandlerConfig(BaseModel):
    """Logging handler configuration"""
    enabled: bool = True
    path: str = "messi.log"
    level: str = "INFO"

class LoggingHandlers(BaseModel):
    """Logging handlers configuration"""
    file: LoggingHandlerConfig = Field(default_factory=LoggingHandlerConfig)
    console: LoggingHandlerConfig = Field(default_factory=lambda: LoggingHandlerConfig(path=""))

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_rotation: str = "1 MB"
    backup_count: int = 5
    handlers: LoggingHandlers = Field(default_factory=LoggingHandlers)

class ModelConfig(BaseModel):
    """Model configuration"""
    chat: str = "gpt-4-1106-preview"
    whisper: str = "whisper-1"
    tts: str = "tts-1"
    tts_voice: str = "alloy"
    tts_fallback: str = "nova"
    tts_speed: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 150

class VoiceConfig(BaseModel):
    """Voice configuration"""
    api_key: str = Field(..., env="ELEVENLABS_API_KEY")
    agent_id: str = Field(..., env="ELEVENLABS_AGENT_ID")

class Settings(BaseSettings):
    """Application settings loaded from .env and config.yaml"""
    
    # API Keys (from .env)
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    PICOVOICE_ACCESS_KEY: str = Field(..., env="PICOVOICE_ACCESS_KEY")
    BASE_DIR: str = Field(..., env="BASE_DIR")
    
    # Model configuration
    models: ModelConfig = Field(default_factory=ModelConfig)
    
    # Voice configuration
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    
    # Learning configuration  
    learning: LearningConfig = Field(default_factory=LearningConfig)
    
    # Logging configuration
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Cache directory
    CACHE_DIR: str = Field(default="cache")
    
    # Audio configuration (will be loaded from config.yaml)
    audio: Dict = Field(default_factory=dict)
    
    # Wake word configuration (will be loaded from config.yaml)
    wake_word: Dict = Field(default_factory=dict)
    
    # Model settings (derived from models config)
    OPENAI_CHAT_MODEL: str = ""
    OPENAI_WHISPER_MODEL: str = ""
    OPENAI_TTS_MODEL: str = ""
    OPENAI_TTS_VOICE: str = ""
    OPENAI_TTS_FALLBACK_VOICE: str = ""
    OPENAI_TTS_SPEED: float = 1.0
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 150
    
    # Wake word settings (will be set from wake_word config)
    WAKE_WORD_MODEL_PATH: str = ""
    WAKE_WORD_SENSITIVITY: Optional[float] = None
    WAKE_WORD_VOLUME_THRESHOLD: Optional[int] = None
    WAKE_WORD_MAX_VOLUME: Optional[int] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_yaml_config()
    
    def _load_yaml_config(self):
        """Load configuration from YAML file"""
        try:
            config_path = Path(self.BASE_DIR) / "config" / "config.yaml"
            if not config_path.exists():
                print(f"Warning: Config file not found at {config_path}")
                return
                
            with open(config_path) as f:
                config = yaml.safe_load(f)
                
            # Update configurations from YAML
            if "models" in config:
                self.models = ModelConfig(**config["models"])
                # Set model settings
                self.OPENAI_CHAT_MODEL = self.models.chat
                self.OPENAI_WHISPER_MODEL = self.models.whisper
                self.OPENAI_TTS_MODEL = self.models.tts
                self.OPENAI_TTS_VOICE = self.models.tts_voice
                self.OPENAI_TTS_FALLBACK_VOICE = self.models.tts_fallback
                self.OPENAI_TTS_SPEED = self.models.tts_speed
                self.OPENAI_TEMPERATURE = self.models.temperature
                self.OPENAI_MAX_TOKENS = self.models.max_tokens
                
            if "audio" in config:
                self.audio = config["audio"]
                
            if "wake_word" in config:
                # Don't override API key from .env
                wake_word_config = config["wake_word"].copy()
                if "access_key" in wake_word_config:
                    del wake_word_config["access_key"]
                self.wake_word = wake_word_config
                
                # Set wake word settings
                if "model_path" in wake_word_config:
                    self.WAKE_WORD_MODEL_PATH = str(Path(self.BASE_DIR) / wake_word_config["model_path"])
                self.WAKE_WORD_SENSITIVITY = wake_word_config.get("sensitivity")
                self.WAKE_WORD_VOLUME_THRESHOLD = wake_word_config.get("volume_threshold")
                self.WAKE_WORD_MAX_VOLUME = wake_word_config.get("max_volume")
                
            if "voice" in config:
                # Don't override API key and agent ID from .env
                voice_config = config["voice"].copy()
                if "api_key" in voice_config:
                    del voice_config["api_key"]
                if "agent_id" in voice_config:
                    del voice_config["agent_id"]
                self.voice = VoiceConfig(**voice_config)
                
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            traceback.print_exc()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # Allow case-insensitive env var names
        extra = "allow"  # Allow extra fields from environment