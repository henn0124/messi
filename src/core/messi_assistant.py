class MessiAssistant:
    def __init__(self):
        self.settings = Settings()
        self.audio = AudioInterface()
        self.tts = TextToSpeech()
        self.speech = SpeechManager()
        self.router = AssistantRouter()
        self.running = False
        self.resource_monitor = ResourceMonitor() 