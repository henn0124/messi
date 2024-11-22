from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    PICOVOICE_ACCESS_KEY: str
    
    # OpenAI Model Settings
    OPENAI_CHAT_MODEL: str
    OPENAI_WHISPER_MODEL: str
    OPENAI_TTS_MODEL: str
    OPENAI_TTS_VOICE: str
    OPENAI_TTS_SPEED: float
    
    # Model Parameters
    MODEL_TEMPERATURE: float
    MODEL_MAX_TOKENS: int
    
    # Audio Device Settings
    AUDIO_INPUT_DEVICE_INDEX: int
    AUDIO_OUTPUT_DEVICE_INDEX: int
    AUDIO_NATIVE_RATE: int
    AUDIO_PROCESSING_RATE: int
    AUDIO_OUTPUT_RATE: int
    
    # Audio Processing Settings - with defaults
    AUDIO_CHANNELS: int = 1
    AUDIO_CHUNK_SIZE: int = 1024
    AUDIO_BUFFER_SIZE: int = 8192
    AUDIO_PRE_EMPHASIS: float = 0.97
    AUDIO_SILENCE_THRESHOLD: int = 100
    
    # Wake Word Settings
    WAKE_WORD: str
    WAKE_WORD_MODEL_PATH: Path
    WAKE_WORD_THRESHOLD: float
    WAKE_WORD_MIN_VOLUME: int
    WAKE_WORD_MAX_VOLUME: int
    WAKE_WORD_DETECTION_WINDOW: float = 0.5
    WAKE_WORD_CONSECUTIVE_FRAMES: int = 1
    
    # Directory Settings
    BASE_DIR: Path = Path("/home/pi/messi")
    CACHE_DIR: Path = BASE_DIR / "cache"
    TEMP_DIR: Path = BASE_DIR / "temp"
    MODEL_DIR: Path = BASE_DIR / "models"
    CONTENT_DIR: Path = BASE_DIR / "content"
    LOG_DIR: Path = BASE_DIR / "logs"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "allow"

    def validate_paths(self):
        """Validate that required files exist and create directories"""
        # Check wake word model
        if not self.WAKE_WORD_MODEL_PATH.exists():
            raise FileNotFoundError(f"Wake word model not found at: {self.WAKE_WORD_MODEL_PATH}")
            
        # Create required directories
        for directory in [self.CACHE_DIR, self.TEMP_DIR, self.MODEL_DIR, 
                         self.CONTENT_DIR, self.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories
        (self.CACHE_DIR / "tts").mkdir(exist_ok=True)
        (self.CACHE_DIR / "responses").mkdir(exist_ok=True)