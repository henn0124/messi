from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    SKILLS_DIR: Path = BASE_DIR / "skills"
    CACHE_DIR: Path = BASE_DIR / "cache"
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    # API Keys (loaded directly from .env)
    OPENAI_API_KEY: str
    PICOVOICE_ACCESS_KEY: str
    
    # Audio Device Settings (loaded from .env)
    AUDIO_INPUT_DEVICE_INDEX: int = 1  # USB PnP Sound Device
    AUDIO_OUTPUT_DEVICE_INDEX: int = 0  # bcm2835 Headphones
    
    # Audio Processing Settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    COMMAND_DURATION: int = 3
    
    # Wake Word Settings
    WAKE_WORD: str = "hey messy"
    WAKE_WORD_MODEL_PATH: Path = MODELS_DIR / "hey-messy_en_raspberry-pi_v3_0_0.ppn"
    WAKE_WORD_THRESHOLD: float = 0.5
    
    # ASR Settings
    ASR_PROVIDER: str = "speech"
    ASR_MODEL: str = "speech"
    
    # TTS Settings
    TTS_MODEL: str = "tts-1-hd"
    TTS_VOICE: str = "nova"
    TTS_SPEED: float = 0.9
    
    # OpenAI Model Settings
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    OPENAI_ASSISTANT_MODEL: str = "gpt-4o"
    OPENAI_WHISPER_MODEL: str = "whisper-1"
    OPENAI_TTS_MODEL: str = "tts-1-hd"
    OPENAI_TTS_VOICE: str = "fable"
    OPENAI_TTS_SPEED: float = 0.9
    
    # Model Parameters
    MODEL_TEMPERATURE: float = 0.7
    MODEL_MAX_TOKENS: int = 4096
    
    class Config:
        env_file = ".env"
        case_sensitive = True