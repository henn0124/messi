from typing import Dict, Optional
import json
from pathlib import Path
import random
import time
from openai import AsyncOpenAI
from ...config import Settings
import asyncio
import logging
from datetime import datetime
import os

class BedtimeStory:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        # Create absolute paths
        self.base_path = Path(__file__).parent.parent.parent.parent.resolve()
        self.story_path = self.base_path / "content" / "stories"
        self.generated_path = self.story_path / "generated"
        self.arc_path = self.story_path / "arcs"
        self.logs_path = self.story_path / "logs"
        
        # Create directories with debug output
        for path in [self.story_path, self.generated_path, self.arc_path, self.logs_path]:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created/verified directory: {path}")
                print(f"Directory exists: {path.exists()}")
                print(f"Directory is writable: {os.access(path, os.W_OK)}")
            except Exception as e:
                print(f"Error creating directory {path}: {e}")
        
        # Initialize story-specific logger
        self.setup_logger()
        
        # Load existing stories and arcs
        self.stories = {}
        self._load_stories()
        print(f"Loaded {len(self.stories)} stories")
        
        self.current_story = None
        self.current_position = 0
        self.story_context = {}
        self.story_arc = None
        self.current_chapter = 0
        self.story_start_time = 0
        self.STORY_DURATION = 1800  # 30 minutes
        
        # Add paths for audio assets
        self.audio_assets_path = self.base_path / "assets" / "audio"
        self.transitions_path = self.audio_assets_path / "transitions"
        
        # Create audio asset directories
        for path in [self.audio_assets_path, self.transitions_path]:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created/verified audio directory: {path}")
            except Exception as e:
                print(f"Error creating audio directory {path}: {e}")
        
        # Define transition sounds with multiple variations
        self.transition_sounds = {
            "thinking": self._get_transition_files("thinking"),
            "continue": self._get_transition_files("continue"),
            "question": self._get_transition_files("question"),
            "waiting": self._get_transition_files("waiting"),
            "complete": self._get_transition_files("complete"),
            "error": self._get_transition_files("error")
        }
        
        self.future_chapters = {}  # Store pre-generated chapters
        self.chapter_generation_task = None  # Track background generation task
        
        # Define target audience parameters
        self.audience_settings = {
            "age_range": {
                "min": 7,
                "max": 12,
                "default": 8
            },
            "reading_level": "early reader",  # early reader, beginner, intermediate
            "language_complexity": "moderate",   # simple, moderate, advanced
            "themes": [
                "friendship",
                "kindness",
                "family",
                "adventure",
                "nature",
                "learning",
                "bedtime"
            ],
            "content_filters": {
                "avoid_scary": True,
                "avoid_conflict": True,
                "educational_focus": True,
                "bedtime_appropriate": True
            },
            "interaction_style": {
                "frequency": "moderate",      # low, moderate, high
                "complexity": "simple",       # simple, moderate, complex
                "encouragement_level": "high" # low, moderate, high
            },
            "voice_preferences": {
                "speed": "slow",             # slow, medium, fast
                "tone": "gentle",            # gentle, energetic, neutral
                "style": "soothing"          # soothing, playful, neutral
            }
        }
        
        # Add resource management
        self.max_cached_chapters = 2
        self.generation_semaphore = asyncio.Semaphore(1)  # Limit concurrent generations
    
    def setup_logger(self):
        """Setup logging for story interactions"""
        self.logger = logging.getLogger('story_interactions')
        self.logger.setLevel(logging.INFO)
        self.current_log_file = None
        
    async def start_story_logging(self, theme: str):
        """Start logging for a new story session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = self.logs_path / f"story_log_{timestamp}_{theme.replace(' ', '_')}.log"
        
        # Create new file handler for this story
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Remove old handlers if they exist
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self.logger.addHandler(file_handler)
        self.current_log_file = log_filename
        
        # Log story start
        self.logger.info(f"Started new story about: {theme}")
        
    async def log_interaction(self, interaction_type: str, content: str, metadata: Dict = None):
        """Log a story interaction"""
        if metadata is None:
            metadata = {}
            
        log_entry = {
            "type": interaction_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **metadata
        }
        
        self.logger.info(json.dumps(log_entry))
    
    async def generate_story_arc(self, theme: str) -> Dict:
        """Generate a complete story arc in the background"""
        try:
            system_prompt = f"""
            You are a master storyteller creating engaging children's stories. 
            Target audience specifications:
            - Age range: {self.audience_settings['age_range']['min']}-{self.audience_settings['age_range']['max']} years
            - Reading level: {self.audience_settings['reading_level']}
            - Language complexity: {self.audience_settings['language_complexity']}
            - Approved themes: {', '.join(self.audience_settings['themes'])}
            
            Content guidelines:
            - Avoid scary content: {self.audience_settings['content_filters']['avoid_scary']}
            - Avoid conflict: {self.audience_settings['content_filters']['avoid_conflict']}
            - Educational focus: {self.audience_settings['content_filters']['educational_focus']}
            - Bedtime appropriate: {self.audience_settings['content_filters']['bedtime_appropriate']}
            
            Create a story arc with exactly this JSON structure:
            {{
                "title": "The story title",
                "theme": "Main theme",
                "age_range": "7-12",
                "characters": [
                    {{
                        "name": "Character name",
                        "description": "Brief description",
                        "key_traits": ["trait1", "trait2"]
                    }}
                ],
                "chapters": [
                    {{
                        "title": "Chapter title",
                        "summary": "Brief summary",
                        "key_events": ["event 1", "event 2"],
                        "interaction_points": [
                            {{
                                "type": "question",
                                "content": "What do you think happens next?",
                                "choices": ["choice 1", "choice 2", "choice 3"]
                            }}
                        ]
                    }}
                ],
                "moral": "Story's gentle moral or lesson",
                "educational_themes": ["theme1", "theme2"]
            }}
            """
            
            story_prompt = f"Create a bedtime story about {theme}. Make it engaging, gentle, and perfect for bedtime reading."
            
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": story_prompt}
                ],
                temperature=self.settings.MODEL_TEMPERATURE,
                max_tokens=self.settings.MODEL_MAX_TOKENS,
                response_format={"type": "json_object"}
            )
            
            # Debug print
            print("Raw response content:", response.choices[0].message.content)
            
            try:
                story_arc = json.loads(response.choices[0].message.content)
                print(f"Generated story arc for: {story_arc['title']}")
                return story_arc
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print("Failed to parse response content")
                return None
            
        except Exception as e:
            print(f"Error generating story arc: {e}")
            print(f"Full error details: {str(e)}")
            return None

    async def generate_chapter(self, chapter: Dict, previous_events: list) -> str:
        """Generate a specific chapter based on the story arc"""
        try:
            # Create context-aware prompt
            prompt = f"""Chapter theme: {chapter['theme']}
Previous events: {', '.join(previous_events)}
Child's engagement level: {self.state.engagement_level}
Preferred elements: {self.memory.get_favorite_elements()}
Current time: {time.strftime('%H:%M')}

Generate a chapter that:
1. Maintains continuity with previous events
2. Includes elements the child has enjoyed
3. Adjusts pacing based on engagement
4. Is appropriate for bedtime
"""

            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Generate chapter: {chapter['title']}"}
                ],
                temperature=0.8
            )
            
            chapter_content = response.choices[0].message.content
            
            # Process the content to handle interaction points
            interaction_points = chapter.get("interaction_points", [])
            for i, point in enumerate(interaction_points):
                if f"{{CHOICE_{i+1}}}" in chapter_content:
                    # Replace placeholder with actual interaction
                    chapter_content = chapter_content.replace(
                        f"{{CHOICE_{i+1}}}",
                        f"\n[{point['content']}]\n"
                    )
            
            return chapter_content
            
        except Exception as e:
            print(f"Error generating chapter: {e}")
            return None

    async def save_story_arc(self, story_arc: Dict):
        """Save the story arc to a file"""
        try:
            timestamp = int(time.time())
            title = story_arc.get("title", "untitled").lower().replace(" ", "_")
            filename = f"{timestamp}_{title}_arc.json"
            filepath = self.arc_path / filename
            
            print(f"\nAttempting to save story arc:")
            print(f"Arc path: {self.arc_path}")
            print(f"Arc path exists: {self.arc_path.exists()}")
            print(f"Full filepath: {filepath}")
            
            # Ensure directory exists
            self.arc_path.mkdir(parents=True, exist_ok=True)
            
            # Save with error handling
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(story_arc, f, indent=4, ensure_ascii=False)
                print(f"Successfully wrote file: {filepath}")
            except IOError as e:
                print(f"IOError while writing file: {e}")
                return None
            
            # Verify file was created
            if filepath.exists():
                print(f"Verified file exists: {filepath}")
                file_size = filepath.stat().st_size
                print(f"File size: {file_size} bytes")
                
                # Add to loaded stories
                arc_id = filepath.stem.replace("_arc", "")
                self.stories[arc_id] = story_arc
                
                print(f"Total stories now: {len(self.stories)}")
                return filepath
            else:
                print(f"File was not created: {filepath}")
                return None
            
        except Exception as e:
            print(f"Error saving story arc: {e}")
            print(f"Current working directory: {Path.cwd()}")
            print(f"Arc path exists: {self.arc_path.exists()}")
            return None

    async def load_story_arc(self, title: str) -> Dict:
        """Load a story arc by title"""
        try:
            for arc_file in self.arc_path.glob("*_arc.json"):
                with open(arc_file) as f:
                    arc_data = json.load(f)
                    if title.lower() in arc_data.get("title", "").lower():
                        print(f"Loaded story arc: {arc_data['title']}")
                        return arc_data
            return None
            
        except Exception as e:
            print(f"Error loading story arc: {e}")
            return None

    async def _prepare_story_arc(self, theme: str):
        """Prepare and save the complete story arc in the background"""
        self.story_arc = await self.generate_story_arc(theme)
        if self.story_arc:
            await self.save_story_arc(self.story_arc)
            print("Story arc prepared, saved, and ready for continuation")

    async def handle_story_interaction(self, interaction_point: Dict, user_response: str = None) -> Dict:
        """Handle interactive story points and modify story arc based on responses"""
        try:
            if not user_response:
                # Present the interaction point to the user
                return {
                    "text": interaction_point["content"],
                    "choices": interaction_point.get("choices", []),
                    "context": "story_interaction",
                    "waiting_for_response": True,
                    "interaction_type": interaction_point["type"],
                    "response_timeout": 10  # Wait 10 seconds for response
                }
            
            # If user_response is empty string (timeout/no response), generate our own choice
            if user_response.strip() == "":
                system_prompt = """
                You are helping continue a children's story. 
                When the child doesn't respond to a question, choose an appropriate answer
                and explain it gently to the child.
                Make it encouraging and maintain the story's flow.
                """
                
                choice_response = await self.client.chat.completions.create(
                    model=self.settings.OPENAI_CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"""
                            Question was: {interaction_point['content']}
                            Possible choices were: {', '.join(interaction_point.get('choices', []))}
                            Choose one and explain why in a child-friendly way.
                        """}
                    ],
                    temperature=0.7
                )
                
                explanation = choice_response.choices[0].message.content
                user_response = explanation  # Use the AI's choice and explanation
                
                # Return the explanation first
                return {
                    "text": f"Since you're quiet, I'll choose! {explanation}",
                    "transition_sound": self.get_random_transition("thinking"),
                    "context": "story_interaction",
                    "continue_story": True,
                    "auto_continue": True,
                    "delay_before_next": 2
                }
            
            # Process user's response (or AI's choice) and modify story arc
            system_prompt = """
            You are adapting a children's story based on input.
            Current interaction point: {interaction_point}
            Response received: {user_response}
            
            Modify the upcoming story elements to naturally incorporate this input while maintaining:
            1. Overall story coherence
            2. Age-appropriate content
            3. Bedtime-suitable tone
            4. Educational value
            
            Return only valid JSON with modified story elements.
            """
            
            # Get remaining chapters
            remaining_chapters = self.story_arc["chapters"][self.current_chapter + 1:]
            
            # Request story modification
            response = await self.client.chat.completions.create(
                model=self.settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
                        Modify these upcoming chapters based on the input:
                        Input: {user_response}
                        Remaining chapters: {json.dumps(remaining_chapters)}
                        
                        Keep the same structure but adapt events and details to incorporate the choice.
                    """}
                ],
                temperature=0.7,
                max_tokens=self.settings.MODEL_MAX_TOKENS
            )
            
            # Update story arc with modified chapters
            modified_chapters = json.loads(response.choices[0].message.content)
            self.story_arc["chapters"][self.current_chapter + 1:] = modified_chapters
            
            # Log the interaction and modification
            await self.log_interaction(
                "story_modification",
                f"Story modified based on {'AI choice' if user_response.strip() == '' else 'user input'}: {user_response}",
                {
                    "original_interaction": interaction_point,
                    "user_response": user_response,
                    "chapter_number": self.current_chapter,
                    "auto_generated": user_response.strip() == ""
                }
            )
            
            return {
                "text": "Let's see what happens next...",
                "continue_story": True,
                "context": "storytelling",
                "auto_continue": True,
                "delay_before_next": 1
            }
            
        except Exception as e:
            print(f"Error handling story interaction: {e}")
            return {
                "text": "Let's continue with our story...",
                "continue_story": True,
                "context": "storytelling"
            }

    def _get_transition_files(self, category: str) -> list:
        """Get all available transition files for a category"""
        category_path = self.transitions_path / category
        if not category_path.exists():
            # Return empty list silently instead of warning
            return []
        
        files = list(category_path.glob(f"{category}_*.wav"))
        return files
    
    def get_random_transition(self, category: str) -> Optional[Path]:
        """Get a random transition file from a category"""
        files = self.transition_sounds.get(category, [])
        if files:
            return random.choice(files)
        return None  # Return None silently if no transitions available

    async def handle(self, intent: Dict) -> Dict:
        """Handle story-related intents"""
        try:
            # Extract data from flattened structure
            command = intent.get("text", "")
            theme = intent.get("parameters", {}).get("theme", "")
            
            print("\n=== Story Handler ===")
            print(f"Command: '{command}'")
            print(f"Theme: '{theme}'")
            print(f"Intent: {intent.get('intent')}")
            print(f"Mode: {intent.get('mode')}")
            
            if theme:
                print(f"▶ Starting story creation for theme: '{theme}'")
                
                # Generate story arc
                print("▶ Generating story arc...")
                self.story_arc = await self.generate_story_arc(theme)
                
                if self.story_arc:
                    print("✓ Story arc generated successfully")
                    print(f"✓ Title: {self.story_arc.get('title')}")
                    print(f"✓ Chapters: {len(self.story_arc.get('chapters', []))}")
                    
                    # Save the story arc
                    await self.save_story_arc(self.story_arc)
                    self.current_chapter = 0
                    self.story_start_time = time.time()
                    
                    # Generate first chapter
                    print("▶ Generating first chapter")
                    chapter = self.story_arc["chapters"][0]
                    chapter_content = await self.generate_chapter(chapter, [])
                    
                    if chapter_content:
                        print("✓ First chapter generated")
                        # Save progress after first chapter
                        await self.save_story_progress(0, chapter_content)
                        
                        # Start background generation of future chapters
                        print("▶ Starting background generation of future chapters")
                        asyncio.create_task(self.generate_future_chapters(0))
                        
                        return {
                            "text": f"Alright! {chapter_content}",
                            "transition_sound": self.get_random_transition("continue"),
                            "context": "storytelling",
                            "auto_continue": True,
                            "delay_before_next": 2,
                            "chapter": 0
                        }
                    else:
                        print("✗ Failed to generate first chapter")
                        return {
                            "text": "I started creating a story but had trouble with the first chapter. Would you like to try a different theme?",
                            "context": "error"
                        }
                else:
                    print("✗ Failed to generate story arc")
                    return {
                        "text": "I'm having trouble creating that story. Could you try asking for a different kind of story?",
                        "context": "error"
                    }
            
            # Handle theme request only if no theme was provided
            elif intent.get("intent") == "ask_theme":
                return {
                    "text": intent.get("prompt", "What kind of story would you like to hear?"),
                    "context": "story_theme_request",
                    "waiting_for_input": True
                }
            
            # Continue with existing story
            elif self.story_arc and hasattr(self, 'current_chapter'):
                print(f"▶ Continuing story from chapter {self.current_chapter + 1}")
                return await self.handle_story_continuation()
            
            # If we get here without a theme or story, something's wrong
            print("✗ No valid story context or theme found")
            print(f"Command was: '{command}'")
            print(f"Intent was: {intent.get('intent')}")
            print(f"Mode was: {intent.get('mode')}")
            return {
                "text": "I'd love to tell you a story! What kind of story would you like to hear about?",
                "context": "error"
            }
            
        except Exception as e:
            print(f"✗ Error in story handling: {e}")
            print(f"✗ Full error details: {str(e)}")
            return {
                "text": "I'm having trouble with the story. Let's start fresh - what kind of story would you like to hear?",
                "context": "error"
            }

    def _load_stories(self):
        """Load all available stories and arcs"""
        try:
            # Load pre-written stories
            for story_file in self.story_path.glob("*.json"):
                if story_file.stem not in ["generated", "arcs", "logs"]:
                    with open(story_file) as f:
                        story_data = json.load(f)
                        self.stories[story_file.stem] = story_data
            
            # Load generated stories
            for story_file in self.generated_path.glob("*.json"):
                with open(story_file) as f:
                    story_data = json.load(f)
                    self.stories[story_file.stem] = story_data
            
            # Load story arcs
            for arc_file in self.arc_path.glob("*_arc.json"):
                with open(arc_file) as f:
                    arc_data = json.load(f)
                    arc_id = arc_file.stem.replace("_arc", "")
                    self.stories[arc_id] = arc_data
                    
            print(f"Found stories in:")
            print(f"  Main: {list(self.story_path.glob('*.json'))}")
            print(f"  Generated: {list(self.generated_path.glob('*.json'))}")
            print(f"  Arcs: {list(self.arc_path.glob('*_arc.json'))}")
            
        except Exception as e:
            print(f"Error loading stories: {e}")
            raise

    def _create_default_transitions(self):
        """Create default transition sounds if they don't exist"""
        default_messages = {
            "thinking": "Hmm, let me think about what happens next...",
            "continue": "And the story continues...",
            "question": "What do you think about that?",
            "waiting": "I'm working on the next part of our story...",
            "complete": "And that's the end of our story for now..."
        }
        
        # Create default WAV files using TTS if they don't exist
        for sound_type, message in default_messages.items():
            if not self.transition_sounds[sound_type].exists():
                print(f"Creating default transition sound: {sound_type}")
                # We'll implement this later with TTS
                # For now, we'll use placeholder WAV files
    
    async def get_transition_response(self, transition_type: str) -> Dict:
        """Get a transition response with appropriate audio"""
        messages = {
            "thinking": [
                "I'm thinking about what happens next...",
                "Let me ponder the next part of our story...",
                "Imagining what happens next..."
            ],
            "continue": [
                "The story continues...",
                "Let's see what happens next...",
                "Our story goes on..."
            ],
            "question": [
                "What do you think about that?",
                "Isn't that interesting?",
                "I wonder what you think..."
            ],
            "waiting": [
                "I'm crafting the next part of our story...",
                "Working on what happens next...",
                "Getting the next part ready..."
            ],
            "complete": [
                "We've reached a good stopping point...",
                "That's all for this part of our story...",
                "Let's take a little break here..."
            ]
        }
        
        return {
            "text": random.choice(messages[transition_type]),
            "transition_sound": self.transition_sounds[transition_type],
            "context": "transition",
            "continue_story": True
        }

    async def generate_future_chapters(self, start_from: int):
        """Generate future chapters with resource limits"""
        try:
            async with self.generation_semaphore:  # Ensure only one generation at a time
                for chapter_index in range(start_from + 1, min(start_from + self.max_cached_chapters + 1, len(self.story_arc["chapters"]))):
                    if chapter_index not in self.future_chapters:
                        chapter = self.story_arc["chapters"][chapter_index]
                        previous_events = [
                            c["summary"] for c in self.story_arc["chapters"][:chapter_index]
                        ]
                        
                        chapter_content = await self.generate_chapter(chapter, previous_events)
                        if chapter_content:
                            self.future_chapters[chapter_index] = chapter_content
                            
                            # Clean up old chapters
                            keys = sorted(self.future_chapters.keys())
                            while len(keys) > self.max_cached_chapters:
                                del self.future_chapters[keys.pop(0)]
                                
                        # Add small delay between generations
                        await asyncio.sleep(0.1)
                        
        except Exception as e:
            print(f"Error generating future chapters: {e}")

    async def save_completed_story(self):
        """Save the complete story text and metadata"""
        try:
            if not self.story_arc:
                print("No story to save")
                return None
            
            timestamp = int(time.time())
            title = self.story_arc.get("title", "untitled").lower().replace(" ", "_")
            
            # Prepare story content with full text and metadata
            story_data = {
                "title": self.story_arc["title"],
                "theme": self.story_arc.get("theme", ""),
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "age_range": self.story_arc.get("age_range", ""),
                    "moral": self.story_arc.get("moral", ""),
                    "educational_themes": self.story_arc.get("educational_themes", []),
                    "characters": self.story_arc.get("characters", []),
                    "completion_status": "complete" if getattr(self, 'story_complete', False) else "partial",
                    "duration": time.time() - self.story_start_time if hasattr(self, 'story_start_time') else None,
                    "interaction_count": len(self.state.user_inputs) if hasattr(self, 'state') else 0
                },
                "story_arc": self.story_arc,  # Save original story arc
                "chapters": [],
                "user_interactions": getattr(self, 'state', {}).get('user_inputs', []),
                "analytics": {
                    "engagement_levels": getattr(self, 'analytics', {}).get('engagement_levels', []),
                    "completion_rate": getattr(self, 'analytics', {}).get('completion_rate', 0.0),
                    "modifications": getattr(self, 'analytics', {}).get('story_modifications', [])
                }
            }
            
            # Add full chapter content
            for chapter_index, chapter in enumerate(self.story_arc["chapters"]):
                chapter_content = self.future_chapters.get(chapter_index, "Chapter content not generated")
                story_data["chapters"].append({
                    "title": chapter["title"],
                    "content": chapter_content,
                    "summary": chapter["summary"],
                    "key_events": chapter["key_events"],
                    "interaction_points": chapter.get("interaction_points", []),
                    "user_responses": [
                        response for response in story_data["user_interactions"]
                        if response.get("chapter") == chapter_index
                    ]
                })
            
            # Save to generated stories directory
            story_filename = f"{timestamp}_{title}_complete.json"
            story_filepath = self.generated_path / story_filename
            
            print(f"\nSaving complete story:")
            print(f"Title: {story_data['title']}")
            print(f"Chapters: {len(story_data['chapters'])}")
            print(f"Total content length: {sum(len(ch['content']) for ch in story_data['chapters'])} characters")
            
            with open(story_filepath, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=4, ensure_ascii=False)
            
            print(f"✓ Saved complete story to: {story_filepath}")
            return story_filepath
            
        except Exception as e:
            print(f"✗ Error saving complete story: {e}")
            return None

    async def create_interactive_story_arc(self, theme: str) -> Dict:
        """Create story arc with interactive input from child"""
        try:
            # Initialize story state
            self.state = StoryState()
            self.analytics = StoryAnalytics()
            
            # Initial questions to engage the child
            initial_questions = [
                {
                    "question": f"What kind of {theme} story would you like? A funny one, an exciting one, or a magical one?",
                    "suggestions": ["funny", "exciting", "magical"]
                },
                {
                    "question": "Who should be the main character in our story?",
                    "suggestions": ["a brave child", "a friendly animal", "a magical creature"]
                },
                {
                    "question": "Where should our story take place?",
                    "suggestions": ["in a magical forest", "in a cozy town", "in a cloud castle"]
                }
            ]
            
            # Ask first question
            first_question = initial_questions[0]
            self.state.questions_asked.append(first_question)
            
            return {
                "text": first_question["question"],
                "suggestions": first_question["suggestions"],
                "waiting_for_input": True,
                "context": "story_creation",
                "next_action": "gather_story_input"
            }
            
        except Exception as e:
            print(f"Error in interactive story creation: {e}")
            return None

    async def handle_story_creation_input(self, user_input: str) -> Dict:
        """Handle user input during story creation"""
        try:
            # Store user's response
            self.state.responses_received.append(user_input)
            self.state.user_inputs.append({
                "question": self.state.questions_asked[-1]["question"],
                "response": user_input,
                "timestamp": time.time()
            })
            
            # Update analytics
            self.analytics.update_metrics({
                "engagement": "active",
                "response": user_input,
                "question_index": len(self.state.responses_received) - 1
            })
            
            # If we have all needed responses, generate story
            if len(self.state.responses_received) >= 3:
                print("Generating story arc with user inputs...")
                self.story_arc = await self.generate_story_arc(
                    theme=self.state.user_inputs[0]["response"],
                    character=self.state.user_inputs[1]["response"],
                    setting=self.state.user_inputs[2]["response"]
                )
                
                if self.story_arc:
                    await self.save_story_arc(self.story_arc)
                    self.current_chapter = 0
                    
                    # Generate first chapter
                    chapter = self.story_arc["chapters"][0]
                    chapter_content = await self.generate_chapter(chapter, [])
                    
                    # Start background generation of future chapters
                    asyncio.create_task(self.generate_future_chapters(0))
                    
                    return {
                        "text": chapter_content,
                        "transition_sound": self.get_random_transition("continue"),
                        "context": "storytelling",
                        "auto_continue": True,
                        "delay_before_next": 2,
                        "chapter": 0
                    }
            
            # Otherwise, ask next question
            next_question = self.initial_questions[len(self.state.responses_received)]
            self.state.questions_asked.append(next_question)
            
            return {
                "text": next_question["question"],
                "suggestions": next_question["suggestions"],
                "waiting_for_input": True,
                "context": "story_creation",
                "next_action": "gather_story_input",
                "transition_sound": self.get_random_transition("question")
            }
            
        except Exception as e:
            print(f"Error handling story creation input: {e}")
            return {
                "text": "I'm having trouble with the story. Let's try again!",
                "context": "error"
            }

    async def handle_story_continuation(self) -> Dict:
        """Handle continuation of the story to next chapter"""
        try:
            print(f"▶ Continuing from chapter {self.current_chapter}")
            next_chapter = self.current_chapter + 1
            
            if next_chapter < len(self.story_arc["chapters"]):
                print(f"▶ Moving to chapter {next_chapter}")
                self.current_chapter = next_chapter
                
                # Use pre-generated chapter if available
                if next_chapter in self.future_chapters:
                    print("✓ Using pre-generated chapter")
                    chapter_content = self.future_chapters[next_chapter]
                    del self.future_chapters[next_chapter]  # Clean up used chapter
                else:
                    print("▶ Generating chapter on demand")
                    chapter = self.story_arc["chapters"][next_chapter]
                    previous_events = [
                        c["summary"] for c in self.story_arc["chapters"][:next_chapter]
                    ]
                    chapter_content = await self.generate_chapter(chapter, previous_events)
                
                if chapter_content:
                    print("✓ Chapter content ready")
                    # Save progress after generating chapter
                    await self.save_story_progress(next_chapter, chapter_content)
                    
                    return {
                        "text": chapter_content,
                        "transition_sound": self.get_random_transition("continue"),
                        "context": "storytelling",
                        "auto_continue": True,
                        "delay_before_next": 2,
                        "chapter": next_chapter
                    }
                else:
                    print("✗ Failed to get chapter content")
                    return {
                        "text": "I seem to have lost my place in the story. Would you like me to start over?",
                        "context": "error"
                    }
            else:
                print("✓ Story complete")
                # Story is complete
                self.story_complete = True
                await self.save_completed_story()
                
                return {
                    "text": f"And that's the end of our story about {self.story_arc['title']}. Would you like to hear another one?",
                    "transition_sound": self.get_random_transition("complete"),
                    "context": "story_complete",
                    "story_complete": True
                }
                
        except Exception as e:
            print(f"✗ Error in story continuation: {e}")
            return {
                "text": "I lost my place in the story. Would you like to start over?",
                "context": "error"
            }

    async def save_story_progress(self, chapter_index: int, chapter_content: str):
        """Save story progress after each chapter generation"""
        try:
            if not self.story_arc:
                print("No story to save")
                return None
            
            timestamp = int(time.time())
            title = self.story_arc.get("title", "untitled").lower().replace(" ", "_")
            
            # Prepare story content with current progress
            story_data = {
                "title": self.story_arc["title"],
                "theme": self.story_arc.get("theme", ""),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "metadata": {
                    "age_range": self.story_arc.get("age_range", ""),
                    "moral": self.story_arc.get("moral", ""),
                    "educational_themes": self.story_arc.get("educational_themes", []),
                    "characters": self.story_arc.get("characters", []),
                    "completion_status": "in_progress",
                    "current_chapter": chapter_index,
                    "total_chapters": len(self.story_arc["chapters"]),
                    "duration": time.time() - self.story_start_time if hasattr(self, 'story_start_time') else None
                },
                "story_arc": self.story_arc,
                "chapters": [],
                "user_interactions": getattr(self, 'state', {}).get('user_inputs', [])
            }
            
            # Add chapters up to current point
            for idx, chapter in enumerate(self.story_arc["chapters"]):
                if idx <= chapter_index:
                    content = chapter_content if idx == chapter_index else self.future_chapters.get(idx, "Chapter not generated")
                    story_data["chapters"].append({
                        "title": chapter["title"],
                        "content": content,
                        "summary": chapter["summary"],
                        "key_events": chapter["key_events"],
                        "interaction_points": chapter.get("interaction_points", []),
                        "user_responses": [
                            response for response in story_data["user_interactions"]
                            if response.get("chapter") == idx
                        ]
                    })
            
            # Save to generated stories directory with progress indicator
            story_filename = f"{timestamp}_{title}_progress.json"
            story_filepath = self.generated_path / story_filename
            
            print(f"\nSaving story progress:")
            print(f"Title: {story_data['title']}")
            print(f"Current chapter: {chapter_index + 1} of {len(self.story_arc['chapters'])}")
            print(f"Generated chapters: {len(story_data['chapters'])}")
            
            with open(story_filepath, 'w', encoding='utf-8') as f:
                json.dump(story_data, f, indent=4, ensure_ascii=False)
            
            print(f"✓ Saved story progress to: {story_filepath}")
            return story_filepath
            
        except Exception as e:
            print(f"✗ Error saving story progress: {e}")
            return None

class StoryState:
    def __init__(self):
        self.current_chapter = 0
        self.story_arc = None
        self.user_inputs = []
        self.last_interaction = time.time()
        self.questions_asked = []
        self.responses_received = []
        self.story_metrics = {
            "start_time": time.time(),
            "interaction_count": 0,
            "completion_percentage": 0.0
        }

class StoryAnalytics:
    def __init__(self):
        self.start_time = time.time()
        self.interaction_times = []
        self.completion_rate = 0.0
        
    def update_metrics(self, interaction_data: Dict):
        self.interaction_times.append(time.time())
        self.completion_rate = interaction_data["chapter"] / interaction_data["total_chapters"]

class StoryMemory:
    def __init__(self):
        self.character_history = {}  # Track recurring characters
        self.plot_threads = {}       # Track ongoing storylines
        self.child_preferences = {}  # Remember child's choices
        self.themes_used = set()     # Track used themes
        self.favorite_elements = {}  # Track elements child responds well to
        
    def add_character(self, character: Dict):
        """Add or update a character in memory"""
        char_id = character["name"].lower()
        if char_id not in self.character_history:
            self.character_history[char_id] = {
                "appearances": 0,
                "traits": set(),
                "relationships": {},
                "key_events": []
            }
        
        char_record = self.character_history[char_id]
        char_record["appearances"] += 1
        char_record["traits"].update(character.get("key_traits", []))
        
    def add_plot_thread(self, thread_name: str, details: Dict):
        """Track an ongoing plot thread"""
        if thread_name not in self.plot_threads:
            self.plot_threads[thread_name] = {
                "status": "active",
                "mentions": 0,
                "related_characters": set(),
                "key_events": [],
                "child_reactions": []
            }
        
        thread = self.plot_threads[thread_name]
        thread["mentions"] += 1
        thread["related_characters"].update(details.get("characters", []))
        thread["key_events"].extend(details.get("events", []))
        
    def update_preferences(self, interaction_data: Dict):
        """Update child's preferences based on interactions"""
        response_type = interaction_data.get("type")
        content = interaction_data.get("content")
        engagement = interaction_data.get("engagement_level", "neutral")
        
        if response_type and content:
            if response_type not in self.child_preferences:
                self.child_preferences[response_type] = {
                    "liked": set(),
                    "disliked": set(),
                    "neutral": set()
                }
            
            category = "liked" if engagement == "high" else "disliked" if engagement == "low" else "neutral"
            self.child_preferences[response_type][category].add(content)
            
    async def create_connected_story(self, theme: str, recent_elements: Dict) -> Dict:
        """Create a story that references past elements"""
        story_elements = {
            "recurring_characters": self._select_relevant_characters(theme),
            "ongoing_threads": self._get_active_plot_threads(),
            "preferred_elements": self._get_favorite_elements(),
            "recent_references": recent_elements
        }
        
        # Generate story prompt incorporating memory elements
        memory_prompt = f"""
        Create a story about {theme} that naturally incorporates these elements:
        
        Recurring Characters: {story_elements['recurring_characters']}
        Ongoing Plot Threads: {story_elements['ongoing_threads']}
        Child's Preferred Elements: {story_elements['preferred_elements']}
        Recent Story Elements: {story_elements['recent_references']}
        
        Maintain continuity with previous stories while introducing new elements.
        """
        
        return await self.generate_story_arc(theme, memory_prompt)
    
    def _select_relevant_characters(self, theme: str) -> list:
        """Select characters relevant to the current theme"""
        relevant_chars = []
        for char_name, char_data in self.character_history.items():
            relevance_score = 0
            # Check theme relevance
            if any(trait in theme.lower() for trait in char_data["traits"]):
                relevance_score += 2
            # Consider recency and frequency
            if char_data["appearances"] > 0:
                relevance_score += min(char_data["appearances"], 3)
            if relevance_score > 2:
                relevant_chars.append({
                    "name": char_name,
                    "traits": list(char_data["traits"]),
                    "history": char_data["key_events"][-3:]  # Last 3 events
                })
        return relevant_chars[:2]  # Return top 2 most relevant characters
    
    def _get_active_plot_threads(self) -> list:
        """Get active plot threads that could be continued"""
        active_threads = []
        for thread_name, thread_data in self.plot_threads.items():
            if thread_data["status"] == "active":
                active_threads.append({
                    "name": thread_name,
                    "latest_events": thread_data["key_events"][-2:],
                    "characters": list(thread_data["related_characters"])
                })
        return active_threads[:1]  # Return most recent active thread
    
    def _get_favorite_elements(self) -> Dict:
        """Get child's favorite story elements"""
        favorites = {}
        for element_type, preferences in self.child_preferences.items():
            if preferences["liked"]:
                favorites[element_type] = list(preferences["liked"])[:3]
        return favorites

skill_manifest = {
    "name": "bedtime_story",
    "intents": ["tell_story", "continue_story", "pause_story", "stop_story"],
    "description": "A simple bedtime story skill"
} 