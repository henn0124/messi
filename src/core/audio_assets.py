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