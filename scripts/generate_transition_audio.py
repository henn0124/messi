import asyncio
from src.core.tts import TextToSpeech
from pathlib import Path
from scripts.generate_transition_texts import transitions

async def generate_transition_audio():
    tts = TextToSpeech()
    output_dir = Path("assets/audio/transitions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating transition audio files...")
    
    for category, messages in transitions.items():
        print(f"\nProcessing {category} transitions:")
        
        # Create a subdirectory for each category
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Generate audio for each message variant
        for i, message in enumerate(messages, 1):
            filename = f"{category}_{i}.wav"
            output_path = category_dir / filename
            
            print(f"Generating: {filename}")
            print(f"Text: {message}")
            
            # Generate audio with appropriate voice settings
            audio_data = await tts.synthesize(
                text=message,
                context="transition"
            )
            
            if audio_data:
                # Save audio file
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                print(f"Saved: {output_path}")
            else:
                print(f"Failed to generate audio for: {message}")
        
        print(f"Generated {len(messages)} variations for {category}")

if __name__ == "__main__":
    asyncio.run(generate_transition_audio()) 