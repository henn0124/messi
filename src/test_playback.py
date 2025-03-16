"""
Audio Playback Test Script
------------------------
Tests the audio output functionality using a WAV file.
"""

import asyncio
import os
from pathlib import Path
import yaml
from core.audio import AudioInterface

async def main():
    """Main test function"""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize audio interface
    audio = AudioInterface(config["audio"])
    
    try:
        # Initialize audio
        print("\n🔊 Testing audio output...")
        if not await audio.initialize():
            print("❌ Failed to initialize audio!")
            return
            
        print("✅ Audio initialized successfully!")
        
        # Load test WAV file
        test_file = Path(__file__).parent.parent / "test_recording.wav"
        if not test_file.exists():
            print(f"\n❌ Test file not found: {test_file}")
            return
            
        print(f"\n📂 Loading test file: {test_file}")
        with open(test_file, 'rb') as f:
            audio_data = f.read()
        
        # Play audio
        print("\n🔈 Playing test audio...")
        if await audio.play(audio_data):
            print("✅ Test complete!")
        else:
            print("❌ Failed to play audio!")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        
    finally:
        # Cleanup
        await audio.stop()

if __name__ == "__main__":
    asyncio.run(main()) 