import asyncio
from pathlib import Path
from openai import AsyncOpenAI
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class AudioAssetGenerator:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.base_path = Path(__file__).parent.parent / "src" / "assets" / "audio" / "phrases"
        
        # Define only missing phrases to generate
        self.phrases = {
            "greeting": [
                ("hello.wav", "Hello! I'm ready to chat."),
                ("hi_there.wav", "Hi there! How can I help?"),
                ("welcome_back.wav", "Welcome back! Ready for more?")
            ],
            "goodbye": [
                ("goodbye.wav", "Goodbye! Come back soon!"),
                ("bye_for_now.wav", "Bye for now! Have a great time!"),
                ("talk_to_you_later.wav", "Talk to you later! Take care!")
            ],
            "acknowledgment": [
                ("i_see.wav", "I see!"),
                ("interesting.wav", "That's interesting!"),
                ("got_it.wav", "Got it!"),
                ("okay.wav", "Okay!")
            ],
            "story_start": [
                ("once_upon_a_time.wav", "Once upon a time..."),
                ("let_me_tell_you.wav", "Let me tell you a story..."),
                ("are_you_ready.wav", "Are you ready for a story?")
            ],
            "story_end": [
                ("the_end.wav", "The end."),
                ("and_they_lived.wav", "And they all lived happily ever after."),
                ("wasnt_that_fun.wav", "Wasn't that a fun story?")
            ]
        }
    
    async def generate_audio(self, text: str, output_path: Path) -> bool:
        """Generate audio file using OpenAI TTS"""
        try:
            response = await self.client.audio.speech.create(
                model="tts-1-hd",
                voice="fable",
                input=text,
                speed=0.9
            )
            
            # Save the audio file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                response.write_to(f)
            
            print(f"✓ Generated: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"✗ Error generating {output_path.name}: {e}")
            return False
    
    async def generate_all_assets(self):
        """Generate all audio assets"""
        print("\nGenerating audio assets...")
        
        for category, phrases in self.phrases.items():
            print(f"\nCategory: {category}")
            category_path = self.base_path / category
            
            for filename, text in phrases:
                output_path = category_path / filename
                if not output_path.exists():
                    await self.generate_audio(text, output_path)
                else:
                    print(f"• Skipping existing: {filename}")
    
    def verify_assets(self) -> dict:
        """Verify all required audio assets exist"""
        missing = {}
        for category, phrases in self.phrases.items():
            category_path = self.base_path / category
            missing_files = [
                filename for filename, _ in phrases 
                if not (category_path / filename).exists()
            ]
            if missing_files:
                missing[category] = missing_files
        return missing

async def main():
    generator = AudioAssetGenerator()
    
    # Generate all assets
    await generator.generate_all_assets()
    
    # Verify assets
    print("\nVerifying assets...")
    missing = generator.verify_assets()
    if missing:
        print("\nMissing files:")
        for category, files in missing.items():
            print(f"{category}: {', '.join(files)}")
    else:
        print("\n✓ All audio assets generated successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 