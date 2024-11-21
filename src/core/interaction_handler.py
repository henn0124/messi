class InteractionHandler:
    def __init__(self):
        self.router = AssistantRouter()
        self.skills = {
            InteractionMode.STORY: StorySkill(),
            InteractionMode.EDUCATION: EducationSkill(),
            # ... other skills
        }
    
    async def handle_interaction(self, audio_input: bytes, context: Dict) -> bytes:
        """Main interaction handling loop"""
        # Convert audio to text
        text_input = await self.speech_manager.process_audio(audio_input)
        
        # Route the interaction
        new_mode, instructions = await self.router.route_interaction(text_input, context)
        
        # Handle mode transition if needed
        if new_mode != self.router.current_mode:
            transition_message = await self.router.transition_to_mode(new_mode)
            # Convert transition message to audio and play it
            
        # Hand off to specific skill
        skill = self.skills[new_mode]
        response = await skill.process(text_input, instructions)
        
        # Convert response to audio and return
        return await self.tts.synthesize(response) 