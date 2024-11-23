"""
Available Skills for Messi Assistant
----------------------------------

Core Skills:
1. BedtimeStory - Interactive storytelling
2. Education - Educational responses
3. Timer - Time management
4. Conversation - General chat
5. Tutor - Specific learning assistance

Each skill should provide:
- skill_manifest
- handle() method
- Proper error handling
"""

from .bedtime_story import BedtimeStory
from .education import Education
from .timer import Timer
from .conversation import Conversation
from .tutor import Tutor

__all__ = [
    'BedtimeStory',
    'Education',
    'Timer',
    'Conversation',
    'Tutor'
]
