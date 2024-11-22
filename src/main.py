import asyncio
from core.audio import AudioInterface
from core.speech import SpeechManager
from core.tts import TextToSpeech
from core.skills.available.bedtime_story import BedtimeStory
from core.assistant_router import AssistantRouter
from pathlib import Path
from core.led_manager import LEDManager, LEDState
from typing import Dict
from core.skills.available.education import Education
import time
from datetime import datetime
import json
import traceback
import psutil
from core.conversation_manager import ConversationManager, ConversationState

class SmartSpeaker:
    def __init__(self):
        print("Initializing Smart Speaker...")
        self.audio = AudioInterface()
        self.speech = SpeechManager()
        self.tts = TextToSpeech()
        self.router = AssistantRouter()
        self.story = BedtimeStory()
        self.led = LEDManager()
        self.education = Education()
        self.conversation_manager = ConversationManager()
        
        # Setup logging
        self.conversation_log = []
        self.log_file = Path("logs") / f"conversation_{int(time.time())}.log"
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Give router access to story skill
        self.router.story = self.story
        
        # Track current state
        self.current_task = None
        self.is_speaking = False
        
        # Add resource monitoring
        self.monitor_resources = True
        self.resource_check_interval = 60  # Check every minute
        
        # Add conversation state tracking
        self.in_conversation = False
        self.conversation_timeout = 10.0  # Seconds to wait for follow-up
        self.last_response_time = 0
    
    async def log_interaction(self, interaction_type: str, content: str, metadata: dict = None):
        """Log an interaction with timestamp"""
        if metadata is None:
            metadata = {}
            
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": interaction_type,
            "content": content,
            **metadata
        }
        
        # Print to console
        print(f"\n[{timestamp}] {interaction_type}:")
        print(f"Content: {content}")
        if metadata:
            print("Metadata:", json.dumps(metadata, indent=2))
        
        # Save to log file
        self.conversation_log.append(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    async def initialize(self):
        """Initialize all components"""
        success = await self.audio.initialize()
        if not success:
            raise RuntimeError("Failed to initialize audio interface!")
            
        # Remove assistant initialization
        # await self.router._init_assistant()
        
        print("Smart Speaker initialized successfully")
    
    async def handle_wake_word(self, audio_data: bytes):
        try:
            # Process speech to text
            text = await self.speech.process_audio(audio_data)
            if not text:
                return
            
            await self.handle_command(text)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            await self.handle_error()
    
    async def handle_command(self, text: str):
        """Handle command with conversation management"""
        try:
            if self.conversation_manager.get_state() == ConversationState.ENDED:
                await self.conversation_manager.start_conversation()
            
            # Process the interaction
            should_continue = await self.conversation_manager.process_interaction(text)
            if not should_continue:
                print("Conversation ended by user")
                await self.led.set_state(LEDState.READY)
                # Play goodbye response
                goodbye_response = await self.tts.synthesize(
                    "Goodbye! Let me know if you need anything else.",
                    "farewell"
                )
                if goodbye_response:
                    await self.audio.play_audio(goodbye_response)
                return
            
            # Start routing and response generation in parallel
            route_task = asyncio.create_task(self.router.route_request(text))
            await self.led.set_state(LEDState.PROCESSING)
            
            # Get route result
            route = await route_task
            
            if route["skill"] == "interactive_story":
                await self.led.set_state(LEDState.STORY)
                await self.handle_story(route)
                # Don't end conversation after story starts
                await self.listen_for_followup()
                
            elif route["skill"] == "education":
                response = await self.education.handle(route)
                if response:
                    await self.led.set_state(LEDState.SPEAKING)
                    audio_response = await self.tts.synthesize(
                        response['text'],
                        response.get('context', 'education')
                    )
                    if audio_response:
                        await self.audio.play_audio(audio_response)
                        
                    # Start listening for follow-up
                    await self.listen_for_followup()
            
        except Exception as e:
            print(f"✗ Error in command handling: {e}")
            await self.handle_error()
    
    async def listen_for_followup(self):
        """Listen for follow-up without wake word"""
        try:
            start_time = time.time()
            
            while True:
                current_time = time.time()
                if current_time - start_time > self.conversation_timeout:
                    print("\n✓ Conversation timeout - returning to wake word mode")
                    await self.conversation_manager.end_conversation("timeout")
                    await self.led.set_state(LEDState.READY)
                    break
                
                print("\n=== Listening for follow-up ===")
                await self.led.set_state(LEDState.LISTENING)
                
                command_audio = await self.audio._record_command()
                if command_audio:
                    text = await self.speech.process_audio(command_audio)
                    if text:
                        await self.handle_command(text)
                        start_time = time.time()  # Reset timeout
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"✗ Error in follow-up listening: {e}")
            await self.conversation_manager.end_conversation("error")
            await self.led.set_state(LEDState.READY)
    
    async def handle_error(self):
        """Handle errors consistently"""
        await self.led.set_state(LEDState.ERROR)
        error_response = await self.tts.synthesize(
            "I'm sorry, I had trouble with that. Could you try again?",
            "error"
        )
        if error_response:
            await self.audio.play_audio(error_response)
        await self.led.set_state(LEDState.READY)
    
    async def handle_story(self, route: Dict):
        """Handle story interaction with interrupt support"""
        try:
            # Get initial response
            response = await self.story.handle(route)
            await self.log_interaction("story_response", response['text'], {
                "chapter": getattr(self.story, 'current_chapter', 0),
                "context": response.get('context'),
                "auto_continue": True
            })
            
            # Play initial chapter
            if not self.audio.interrupt_event.is_set():
                self.is_speaking = True
                audio_response = await self.tts.synthesize(
                    response['text'], 
                    response.get('context', 'storytelling')
                )
                if audio_response:
                    await self.audio.play_audio(audio_response)
                self.is_speaking = False
                
                # Keep conversation active during story
                self.conversation_manager.state = ConversationState.ACTIVE
                
                # Continue with story
                while not self.audio.interrupt_event.is_set():
                    await asyncio.sleep(2)
                    
                    if self.audio.interrupt_event.is_set():
                        break
                    
                    response = await self.story.handle({"text": "", "auto_continue": True})
                    
                    if response.get('story_complete', False):
                        break
                    
                    await self.log_interaction("story_continuation", response['text'])
                    
                    self.is_speaking = True
                    audio_response = await self.tts.synthesize(
                        response['text'], 
                        response.get('context', 'storytelling')
                    )
                    if audio_response:
                        await self.audio.play_audio(audio_response)
                    self.is_speaking = False
            
            await self.led.set_state(LEDState.READY)
            
        except asyncio.CancelledError:
            print("Story handling cancelled")
            await self.log_interaction("story_cancelled", "Story handling cancelled")
            await self.led.set_state(LEDState.READY)
        except Exception as e:
            print(f"Error in story handling: {e}")
            await self.log_interaction("error", str(e), {
                "location": "story_handler",
                "traceback": traceback.format_exc()
            })
            await self.led.set_state(LEDState.ERROR)
    
    async def monitor_system_resources(self):
        """Monitor system resources"""
        while self.monitor_resources:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                # Get memory usage
                memory = psutil.virtual_memory()
                
                print(f"\nSystem Resources:")
                print(f"CPU Usage: {cpu_percent}%")
                print(f"Memory Usage: {memory.percent}%")
                
                if cpu_percent > 80 or memory.percent > 80:
                    print("⚠️ High resource usage detected!")
                    
                await asyncio.sleep(self.resource_check_interval)
                
            except Exception as e:
                print(f"Error monitoring resources: {e}")
                await asyncio.sleep(self.resource_check_interval)
    
    async def run(self):
        """Main run loop"""
        try:
            print("\nStarting bedtime story assistant...")
            print("Say 'hey messy' to wake me up, then ask for a story!")
            
            await self.led.set_state(LEDState.READY)
            
            # Start monitoring but don't await it
            await self.audio.start_monitoring(self.handle_wake_word)
            
            # Start resource monitoring
            monitor_task = asyncio.create_task(self.monitor_system_resources())
            
            try:
                # Keep the main loop running
                while True:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping bedtime story assistant...")
            finally:
                # Clean up tasks
                self.monitor_resources = False
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.audio.cleanup()
        self.led.cleanup()

async def main():
    """Application entrypoint"""
    try:
        speaker = SmartSpeaker()
        await speaker.initialize()
        await speaker.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 