"""
Available Skills for Messi Assistant
----------------------------------

Core Skills:
1. BedtimeStory - Interactive storytelling
2. EducationSkill - Educational responses
3. Timer - Time management
4. ConversationSkill - General chat
5. TutorSkill - Specific learning assistance

Each skill should provide:
- skill_manifest
- handle() method
- Proper error handling
"""

from .bedtime_story import BedtimeStory
from .education import EducationSkill
from .timer import Timer
from .conversation import ConversationSkill
from .tutor import TutorSkill

__all__ = [
    'BedtimeStory',
    'EducationSkill',
    'Timer',
    'ConversationSkill',
    'TutorSkill'
]
