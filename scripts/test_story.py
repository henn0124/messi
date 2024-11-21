import asyncio
from src.core.skills.available.bedtime_story import BedtimeStory
from src.core.intent import IntentProcessor
from src.core.hardware.led import LEDController

async def main():
    # Initialize components
    story = BedtimeStory()
    intent_processor = IntentProcessor()
    led = LEDController()
    
    try:
        # Simulate story interaction
        print("Saying: 'tell me a story'")
        intent = await intent_processor.process("tell me a story")
        response = await story.handle(intent)
        print(f"Pi: {response['response']['text']}")
        led.set_mode(response['response']['led'])
        
        await asyncio.sleep(2)
        
        # Simulate continue
        print("\nSaying: 'continue'")
        intent = await intent_processor.process("continue")
        response = await story.handle(intent)
        print(f"Pi: {response['response']['text']}")
        led.set_mode(response['response']['led'])
        
    finally:
        led.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 