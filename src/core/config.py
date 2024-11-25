from pydantic_settings import BaseSettings
from pathlib import Path
import yaml
import os

class Settings(BaseSettings):
    """Application settings loaded from .env and config.yaml"""
    
    # Directory Settings (from .env)
    BASE_DIR: Path = Path("/home/pi/messi")
    CACHE_DIR: Path = BASE_DIR / "cache"
    MODELS_DIR: Path = BASE_DIR / "models"
    TEMP_DIR: Path = BASE_DIR / "temp"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # Add default for wake word model path
    WAKE_WORD_MODEL_PATH: Path = MODELS_DIR / "hey-messy_en_raspberry-pi_v3_0_0.ppn"
    
    # Sensitive Settings (from .env)
    OPENAI_API_KEY: str
    PICOVOICE_ACCESS_KEY: str
    
    # Model Settings (with defaults)
    OPENAI_CHAT_MODEL: str = "gpt-4-1106-preview"
    OPENAI_WHISPER_MODEL: str = "whisper-1"
    OPENAI_TTS_MODEL: str = "tts-1"
    OPENAI_TTS_VOICE: str = "fable"
    OPENAI_TTS_SPEED: float = 1.0
    MODEL_TEMPERATURE: float = 0.7
    MODEL_MAX_TOKENS: int = 4096
    
    # Audio Settings (with defaults)
    AUDIO_INPUT_DEVICE_INDEX: int = 1
    AUDIO_OUTPUT_DEVICE_INDEX: int = 0
    AUDIO_NATIVE_RATE: int = 44100
    AUDIO_PROCESSING_RATE: int = 16000
    AUDIO_OUTPUT_RATE: int = 24000
    AUDIO_CHANNELS: int = 1
    AUDIO_CHUNK_SIZE: int = 1024
    AUDIO_BUFFER_SIZE: int = 8192
    AUDIO_PRE_EMPHASIS: float = 0.97
    AUDIO_SILENCE_THRESHOLD: int = 90
    
    # Wake Word Settings (with defaults)
    WAKE_WORD: str = "hey messy"
    WAKE_WORD_THRESHOLD: float = 0.75
    WAKE_WORD_MIN_VOLUME: int = 100
    WAKE_WORD_MAX_VOLUME: int = 385
    WAKE_WORD_DETECTION_WINDOW: float = 2.0
    WAKE_WORD_CONSECUTIVE_FRAMES: int = 1
    
    # Command Settings (with defaults)
    COMMAND_MIN_DURATION: float = 0.5
    COMMAND_MAX_DURATION: float = 10.0
    COMMAND_MIN_VOLUME: int = 90
    COMMAND_SILENCE_TIMEOUT: float = 0.5
    COMMAND_PRE_BUFFER: float = 0.1
    COMMAND_MAX_SILENCE_CHUNKS: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "allow"

    def __init__(self, **kwargs):
        # Add debug prints before super init
        print(f"\nDebug - Config Paths:")
        print(f"BASE_DIR: {Path('/home/pi/messi')}")
        print(f"MODELS_DIR: {Path('/home/pi/messi/models')}")
        print(f"Expected Wake Word Model: {Path('/home/pi/messi/models/hey-messy_en_raspberry-pi_v3_0_0.ppn')}")
        
        # Now do super init
        super().__init__(**kwargs)
        
        # Load YAML config once
        config = self._load_yaml_config()
        
        # Update settings from YAML
        self._update_from_yaml(config)
        
        # Create directories
        self._create_directories()
        
        # Set derived paths
        self.WAKE_WORD_MODEL_PATH = self.MODELS_DIR / config.get('wake_word', {}).get('model_name', 'hey-messy_en_raspberry-pi_v3_0_0.ppn')
        
        # Add post-init debug prints
        print(f"Final Wake Word Model Path: {self.WAKE_WORD_MODEL_PATH}")
        print(f"Model exists: {self.WAKE_WORD_MODEL_PATH.exists()}")

    def _load_yaml_config(self) -> dict:
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def _update_from_yaml(self, config: dict):
        """Update settings from YAML config"""
        if 'models' in config:
            self.OPENAI_CHAT_MODEL = config['models'].get('chat', self.OPENAI_CHAT_MODEL)
            self.OPENAI_WHISPER_MODEL = config['models'].get('whisper', self.OPENAI_WHISPER_MODEL)
            self.OPENAI_TTS_MODEL = config['models'].get('tts', self.OPENAI_TTS_MODEL)
            self.OPENAI_TTS_VOICE = config['models'].get('tts_voice', self.OPENAI_TTS_VOICE)
            self.OPENAI_TTS_SPEED = float(config['models'].get('tts_speed', self.OPENAI_TTS_SPEED))
            self.MODEL_TEMPERATURE = float(config['models'].get('temperature', self.MODEL_TEMPERATURE))
            self.MODEL_MAX_TOKENS = int(config['models'].get('max_tokens', self.MODEL_MAX_TOKENS))
        
        if 'audio' in config:
            audio_input = config['audio'].get('input', {})
            audio_output = config['audio'].get('output', {})
            
            self.AUDIO_INPUT_DEVICE_INDEX = int(audio_input.get('device_index', self.AUDIO_INPUT_DEVICE_INDEX))
            self.AUDIO_OUTPUT_DEVICE_INDEX = int(audio_output.get('device_index', self.AUDIO_OUTPUT_DEVICE_INDEX))
            self.AUDIO_NATIVE_RATE = int(audio_input.get('native_rate', self.AUDIO_NATIVE_RATE))
            self.AUDIO_PROCESSING_RATE = int(audio_input.get('processing_rate', self.AUDIO_PROCESSING_RATE))
            self.AUDIO_OUTPUT_RATE = int(audio_output.get('rate', self.AUDIO_OUTPUT_RATE))
            self.AUDIO_CHANNELS = int(audio_input.get('channels', self.AUDIO_CHANNELS))
            self.AUDIO_CHUNK_SIZE = int(audio_input.get('chunk_size', self.AUDIO_CHUNK_SIZE))
            self.AUDIO_BUFFER_SIZE = int(audio_input.get('buffer_size', self.AUDIO_BUFFER_SIZE))
            self.AUDIO_PRE_EMPHASIS = float(audio_input.get('pre_emphasis', self.AUDIO_PRE_EMPHASIS))
            self.AUDIO_SILENCE_THRESHOLD = int(audio_input.get('silence_threshold', self.AUDIO_SILENCE_THRESHOLD))

        if 'wake_word' in config:
            wake_word = config['wake_word']
            self.WAKE_WORD = wake_word.get('name', self.WAKE_WORD)
            self.WAKE_WORD_THRESHOLD = float(wake_word.get('threshold', self.WAKE_WORD_THRESHOLD))
            self.WAKE_WORD_MIN_VOLUME = int(wake_word.get('min_volume', self.WAKE_WORD_MIN_VOLUME))
            self.WAKE_WORD_MAX_VOLUME = int(wake_word.get('max_volume', self.WAKE_WORD_MAX_VOLUME))
            self.WAKE_WORD_DETECTION_WINDOW = float(wake_word.get('detection_window', self.WAKE_WORD_DETECTION_WINDOW))
            self.WAKE_WORD_CONSECUTIVE_FRAMES = int(wake_word.get('consecutive_frames', self.WAKE_WORD_CONSECUTIVE_FRAMES))

        if 'command' in config:
            command = config['command']
            self.COMMAND_MIN_DURATION = float(command.get('min_duration', self.COMMAND_MIN_DURATION))
            self.COMMAND_MAX_DURATION = float(command.get('max_duration', self.COMMAND_MAX_DURATION))
            self.COMMAND_MIN_VOLUME = int(command.get('min_volume', self.COMMAND_MIN_VOLUME))
            self.COMMAND_SILENCE_TIMEOUT = float(command.get('silence_timeout', self.COMMAND_SILENCE_TIMEOUT))
            self.COMMAND_PRE_BUFFER = float(command.get('pre_buffer', self.COMMAND_PRE_BUFFER))
            self.COMMAND_MAX_SILENCE_CHUNKS = int(command.get('max_silence_chunks', self.COMMAND_MAX_SILENCE_CHUNKS))

    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.CACHE_DIR, self.MODELS_DIR, self.TEMP_DIR, self.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.CACHE_DIR / "tts").mkdir(exist_ok=True)
        (self.CACHE_DIR / "responses").mkdir(exist_ok=True)

    def validate_paths(self):
        """Validate that required files exist"""
        if not self.WAKE_WORD_MODEL_PATH.exists():
            raise FileNotFoundError(f"Wake word model not found at: {self.WAKE_WORD_MODEL_PATH}")