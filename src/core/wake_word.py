def initialize(self):
    wake_word_config = self.settings.config.get("wake_word", {})
    self.threshold = wake_word_config.get("threshold", 0.85)
    self.min_volume = wake_word_config.get("min_volume", 600) 