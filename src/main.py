"""
Messi Assistant Main Application
------------------------------
A fun voice assistant for kids featuring fun facts and jokes.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
import numpy as np
import wave
import io

from core.audio import AudioInterface
from core.voice_synthesis import VoiceSynthesis
from core.wake_word import WakeWordDetector

class MessiAssistant:
    def __init__(self):
        """Initialize Messi Assistant"""
        # Load environment variables first
        load_dotenv()
        
        # Load configuration
        self.config = self._load_config()
        
        # Update config with environment variables
        self.config["wake_word"]["access_key"] = os.getenv("PICOVOICE_ACCESS_KEY")
        self.config["voice"]["api_key"] = os.getenv("ELEVENLABS_API_KEY")
        
        # Get agent ID and strip any comments
        agent_id = os.getenv("ELEVENLABS_AGENT_ID", "").split("#")[0].strip()
        self.config["voice"]["agent_id"] = agent_id
        
        # Initialize components
        self.voice_synthesis = VoiceSynthesis(
            api_key=self.config["voice"]["api_key"],
            agent_id=self.config["voice"]["agent_id"]
        )
        self.audio = AudioInterface(self.config["audio"])
        self.wake_word = WakeWordDetector(self.config["wake_word"])
        
        # State management
        self.running = False
        
    def _load_config(self) -> dict:
        """Load configuration from YAML"""
        # Get the project root directory (parent of src)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def start(self):
        """Start the assistant"""
        print("\n🚀 Starting Messi Assistant...")
        
        try:
            # Initialize audio
            if not await self.audio.initialize():
                print("❌ Failed to initialize audio!")
                return
            
            print("✅ Audio initialized successfully!")
            
            # Start conversation with Elevenlabs agent
            if not await self.voice_synthesis.start_conversation():
                print("❌ Failed to start conversation!")
                return
            
            # Start audio processing
            self.running = True
            await self.audio.start_processing()
            
            # Keep the assistant running
            while self.running:
                # Wait for wake word
                if await self.audio.wait_for_wake_word(self.wake_word):
                    # Record command
                    audio_data = await self.audio.record_command()
                    if not audio_data:
                        continue
                    
                    # Process command with Elevenlabs agent
                    response_audio = await self.voice_synthesis.process_audio(audio_data)
                    if response_audio:
                        await self.audio.play(response_audio)
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        finally:
            await self.stop()

    async def stop(self):
        """Stop the assistant"""
        print("\n👋 Stopping Messi Assistant...")
        self.running = False
        await self.audio.stop()
        await self.voice_synthesis.end_conversation()
        self.wake_word.cleanup()
        self.voice_synthesis.cleanup_cache()

async def main():
    """Main entry point"""
    assistant = MessiAssistant()
    await assistant.start()

if __name__ == "__main__":
    asyncio.run(main()) 