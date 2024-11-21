import asyncio
from src.core.audio import AudioInterface

async def on_wake_word(audio_data: bytes):
    print("Wake word detected! Captured audio command.")
    # In a real implementation, this would go to speech recognition

async def main():
    audio = AudioInterface()
    
    try:
        await audio.initialize()
        print("\nStarting audio monitoring. Say 'hey pi' to test wake word...")
        await audio.start_monitoring(on_wake_word)
    except KeyboardInterrupt:
        print("\nStopping audio monitoring...")
    finally:
        audio.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 