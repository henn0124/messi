"""
Audio Assets Manager for Messi Assistant
-------------------------------------

This module manages pre-recorded audio phrases and transitions used by the assistant
for natural and varied responses. It works in conjunction with the audio asset
generator script (scripts/generate_audio_assets.py) for maintaining and updating
the audio library.

Key Features:
    1. Phrase Management:
        - Categorized audio phrases
        - Random variation selection
        - Missing asset detection
        - Directory structure maintenance
    
    2. Audio Categories:
        - Conversation Flow (greetings, goodbyes)
        - Transitions (thinking, continue)
        - Questions (clarification, repetition)
        - Feedback (success, error)
        - Story Elements (start, end, transitions)
    
    3. Asset Organization:
        - Category-based directories
        - Consistent naming conventions
        - Version tracking
        - Asset verification
    
    4. Integration:
        Works with generate_audio_assets.py for:
        - Creating new assets
        - Updating existing phrases
        - Adding new categories
        - Maintaining consistency

File Structure:
    /assets/audio/phrases/
        /greeting/
            - hello.wav
            - hi_there.wav
            - welcome_back.wav
        /goodbye/
            - goodbye.wav
            - bye_for_now.wav
        ... (other categories)

Asset Generation:
    New assets can be generated using:
    python scripts/generate_audio_assets.py
    
    The generator script:
    - Uses OpenAI's TTS API
    - Maintains consistent voice
    - Handles missing assets
    - Verifies generations

Usage:
    manager = AudioAssetManager()
    
    # Get random greeting
    greeting_path = manager.get_phrase(AudioPhraseType.GREETING)
    
    # Check for missing assets
    missing = manager.list_missing_phrases()

Adding New Phrases:
    1. Add new type to AudioPhraseType enum
    2. Add phrases to PHRASES dictionary
    3. Update generate_audio_assets.py
    4. Run generator script
    5. Verify new assets

Author: Your Name
Created: 2024-01-24
"""

from enum import Enum
from pathlib import Path
from typing import Optional
import random

class AudioPhraseType(Enum):
    # Conversation Flow
    GREETING = "greeting"
    GOODBYE = "goodbye"
    ACKNOWLEDGMENT = "acknowledgment"
    CONFIRMATION = "confirmation"
    
    # Transitions
    THINKING = "thinking"
    CONTINUE = "continue"
    PAUSE = "pause"
    
    # Questions
    ASK_REPEAT = "ask_repeat"
    ASK_CLARIFY = "ask_clarify"
    ASK_CONTINUE = "ask_continue"
    
    # Feedback
    ERROR = "error"
    SUCCESS = "success"
    
    # Story-specific
    STORY_START = "story_start"
    STORY_END = "story_end"
    CHAPTER_TRANSITION = "chapter_transition"

class AudioAssetManager:
    # Define common phrases and their variations
    PHRASES = {
        AudioPhraseType.GREETING: [
            "hello.wav",
            "hi_there.wav",
            "welcome_back.wav"
        ],
        AudioPhraseType.GOODBYE: [
            "goodbye.wav",
            "bye_for_now.wav",
            "talk_to_you_later.wav"
        ],
        AudioPhraseType.ACKNOWLEDGMENT: [
            "i_see.wav",
            "interesting.wav",
            "got_it.wav",
            "okay.wav"
        ],
        AudioPhraseType.THINKING: [
            "let_me_think.wav",
            "thinking.wav",
            "hmm.wav"
        ],
        AudioPhraseType.CONTINUE: [
            "shall_we_continue.wav",
            "moving_on.wav",
            "next_part.wav"
        ],
        AudioPhraseType.ERROR: [
            "sorry_error.wav",
            "something_went_wrong.wav",
            "lets_try_again.wav"
        ],
        AudioPhraseType.STORY_START: [
            "once_upon_a_time.wav",
            "let_me_tell_you.wav",
            "are_you_ready.wav"
        ],
        AudioPhraseType.STORY_END: [
            "the_end.wav",
            "and_they_lived.wav",
            "wasnt_that_fun.wav"
        ]
    }
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent / "assets" / "audio"
        self.phrases_path = self.base_path / "phrases"
        self._verify_paths()
    
    def _verify_paths(self):
        """Verify all required directories exist"""
        for phrase_type in AudioPhraseType:
            path = self.phrases_path / phrase_type.value
            path.mkdir(parents=True, exist_ok=True)
    
    def get_phrase(self, phrase_type: AudioPhraseType) -> Optional[Path]:
        """Get a random variation of a phrase"""
        if phrase_type not in AudioPhraseType:
            return None
            
        phrase_dir = self.phrases_path / phrase_type.value
        available_phrases = list(phrase_dir.glob("*.wav"))
        
        if not available_phrases:
            print(f"No phrases available for {phrase_type.value}")
            return None
            
        return random.choice(available_phrases)
    
    def list_missing_phrases(self) -> dict:
        """List all missing phrase files"""
        missing = {}
        for phrase_type, phrases in self.PHRASES.items():
            phrase_dir = self.phrases_path / phrase_type.value
            missing_files = [
                phrase for phrase in phrases 
                if not (phrase_dir / phrase).exists()
            ]
            if missing_files:
                missing[phrase_type.value] = missing_files
        return missing 