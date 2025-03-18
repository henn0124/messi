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

from core.audio import AudioManager
from core.voice_synthesis import VoiceSynthesisManager
from core.wake_word import WakeWordDetector

async def main():
    """Main entry point for Messi Assistant"""
    try:
        # Load configuration
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)
            
        print("\n🎯 Wake Word Configuration:")
        print(f"  Frame Length: {config['wake_word']['frame_length']}")
        print(f"  Sample Rate: {config['wake_word']['sample_rate']}")
        print(f"  Sensitivity: {config['wake_word']['sensitivity']}")
        print(f"  Model Path: {config['wake_word']['model_path']}")
        
        # Initialize audio and voice synthesis
        audio_manager = AudioManager(config)
        if not await audio_manager.initialize():
            print("❌ Failed to initialize audio system")
            return
            
        print("\n🔍 Running audio loopback test...")
        if not await audio_manager.test_audio_loop(duration=2.0):
            print("\n⚠️ Audio loopback test failed. Please check your audio setup.")
            await audio_manager.cleanup()
            return
            
        print("\n🎙️ Audio test passed! Starting main loop...")
        
        # Start audio processing
        await audio_manager.start_processing()
        
        try:
            while True:
                # Wait for wake word
                await audio_manager.wait_for_wake_word()
                
                # Record command
                print("\n🎤 Recording command...")
                audio_data = await audio_manager.record_command()
                if not audio_data:
                    print("❌ Failed to record command")
                    continue
                    
                # Play back the recorded command for testing
                print("\n🔊 Playing back recorded command...")
                await audio_manager.play(audio_data)
                
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        
    finally:
        if 'audio_manager' in locals():
            await audio_manager.cleanup()
            
if __name__ == "__main__":
    asyncio.run(main()) 