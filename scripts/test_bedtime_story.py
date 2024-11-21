import asyncio
from src.core.audio import AudioInterface
from src.core.speech import SpeechManager
from src.core.tts import TextToSpeech
from src.core.skills.available.bedtime_story import BedtimeStory

async def main():
    # Initialize components
    print("Initializing components...")
    audio = AudioInterface()
    speech = SpeechManager()
    tts = TextToSpeech()
    story = BedtimeStory()
    
    try:
        # Initialize audio interface
        success = await audio.initialize()
        if not success:
            print("Failed to initialize audio interface!")
            return
        
        print("\nStarting bedtime story assistant...")
        print("Say 'hey messy' to wake me up, then ask for a story!")
        
        async def handle_wake_word(audio_data: bytes):
            # Process speech to text
            text = await speech.process_audio(audio_data)
            print(f"\nYou said: {text}")
            
            # Handle with story skill
            response = await story.handle({"text": text})
            print(f"\nA: {response['text']}")
            
            # Handle transition sounds if present
            if response.get('transition_sound'):
                transition_path = response['transition_sound']
                if transition_path.exists():
                    with open(transition_path, 'rb') as f:
                        transition_audio = f.read()
                        await audio.play_audio(transition_audio)
                    # Brief pause after transition
                    await asyncio.sleep(0.5)
            
            # Play main response
            audio_response = await tts.synthesize(response['text'], response.get('context', 'storytelling'))
            if audio_response:
                await audio.play_audio(audio_response)
                
            # Handle automatic continuation
            while response.get('auto_continue', False):
                # Wait specified delay
                await asyncio.sleep(response.get('delay_before_next', 2))
                
                # Get next part of story
                response = await story.handle({"text": ""})
                print(f"\nContinuing story...\n{response['text']}")
                
                # Handle transition sounds if present
                if response.get('transition_sound'):
                    transition_path = response['transition_sound']
                    if transition_path.exists():
                        with open(transition_path, 'rb') as f:
                            transition_audio = f.read()
                            await audio.play_audio(transition_audio)
                        # Brief pause after transition
                        await asyncio.sleep(0.5)
                
                # Play main response
                audio_response = await tts.synthesize(response['text'], response.get('context', 'storytelling'))
                if audio_response:
                    await audio.play_audio(audio_response)
                
                # Break if story is complete
                if response.get('story_complete', False):
                    break
        
        # Start monitoring for wake word
        await audio.start_monitoring(handle_wake_word)
        
    except KeyboardInterrupt:
        print("\nStopping bedtime story assistant...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        audio.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 