import time
from rpi_ws281x import PixelStrip, Color
import threading

class LEDController:
    def __init__(self):
        # Configure for Raspberry Pi
        LED_COUNT = 12        # Number of LED pixels
        LED_PIN = 18         # GPIO pin
        LED_FREQ_HZ = 800000 # LED signal frequency
        LED_DMA = 10         # DMA channel
        LED_BRIGHTNESS = 255 # Set to 0 for darkest and 255 for brightest
        LED_CHANNEL = 0
        
        self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, False, LED_BRIGHTNESS, LED_CHANNEL)
        self.strip.begin()
        self.current_effect = None
        self.running = True
    
    def set_mode(self, mode: str):
        """Set LED mode/effect"""
        self.current_effect = mode
        if mode == "soft_blue":
            self._solid_color(Color(0, 0, 50))  # Dim blue
        elif mode == "dim":
            self._solid_color(Color(0, 0, 20))  # Very dim blue
        elif mode == "off":
            self._solid_color(Color(0, 0, 0))
        elif mode == "soft_pulse":
            threading.Thread(target=self._pulse_effect).start()
    
    def _solid_color(self, color):
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, color)
        self.strip.show()
    
    def _pulse_effect(self):
        while self.current_effect == "soft_pulse" and self.running:
            for brightness in range(0, 30, 2):
                if self.current_effect != "soft_pulse":
                    break
                self._solid_color(Color(0, 0, brightness))
                time.sleep(0.05)
            for brightness in range(30, 0, -2):
                if self.current_effect != "soft_pulse":
                    break
                self._solid_color(Color(0, 0, brightness))
                time.sleep(0.05)
    
    def cleanup(self):
        """Clean up LED resources"""
        self.running = False
        self._solid_color(Color(0, 0, 0)) 