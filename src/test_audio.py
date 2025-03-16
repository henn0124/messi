"""
Audio Output Test Script
----------------------
Tests the audio output functionality using Elevenlabs voice synthesis.
"""

import asyncio
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from core.audio import AudioInterface
from core.voice_synthesis import VoiceSynthesis

async def main():
    """Main test function"""
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    voice = VoiceSynthesis(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        agent_id=os.getenv("ELEVENLABS_AGENT_ID")
    )
    audio = AudioInterface(config["audio"])
    
    try:
        # Initialize audio
        print("\n🔊 Testing audio output...")
        if not await audio.initialize():
            print("❌ Failed to initialize audio!")
            return
            
        print("✅ Audio initialized successfully!")
        
        # Start conversation with test message
        print("\n💬 Starting test conversation...")
        if await voice.start_conversation():
            # Send a test message
            print("\n🗣️ Generating test message...")
            test_message = "Hi! I'm testing the audio output. Can you hear me clearly?"
            response = await voice.process_audio(test_message.encode())
            
            if response:
                print("\n🔈 Playing test message...")
                await audio.play(response)
                print("✅ Test complete!")
            else:
                print("❌ Failed to generate test message!")
        else:
            print("❌ Failed to start conversation!")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        
    finally:
        # Cleanup
        await voice.end_conversation()
        await audio.stop()

if __name__ == "__main__":
    asyncio.run(main()) 