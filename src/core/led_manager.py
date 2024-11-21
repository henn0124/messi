import RPi.GPIO as GPIO
import asyncio
from enum import Enum

class LEDState(Enum):
    READY = "ready"                 # Solid green - ready for wake word
    LISTENING = "listening"         # Pulsing green - listening for command
    PROCESSING = "processing"       # Blinking blue - processing request
    SPEAKING = "speaking"          # Solid blue - speaking response
    STORY = "story"                # Gentle pulsing blue - telling story
    ERROR = "error"                # Red flash - error occurred
    OFF = "off"                    # LED off

class LEDManager:
    def __init__(self):
        # Disable GPIO warnings
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        
        # Define GPIO pins for RGB LED
        self.RED_PIN = 17    # GPIO17
        self.GREEN_PIN = 27  # GPIO27
        self.BLUE_PIN = 22   # GPIO22
        
        # Setup GPIO pins
        GPIO.setup(self.RED_PIN, GPIO.OUT)
        GPIO.setup(self.GREEN_PIN, GPIO.OUT)
        GPIO.setup(self.BLUE_PIN, GPIO.OUT)
        
        # Setup PWM for smooth transitions
        self.red_pwm = GPIO.PWM(self.RED_PIN, 100)
        self.green_pwm = GPIO.PWM(self.GREEN_PIN, 100)
        self.blue_pwm = GPIO.PWM(self.BLUE_PIN, 100)
        
        # Start PWM at 0 duty cycle
        self.red_pwm.start(0)
        self.green_pwm.start(0)
        self.blue_pwm.start(0)
        
        self.current_state = LEDState.OFF
        self._running = False
        self._task = None
    
    async def set_state(self, state: LEDState):
        """Set LED state with appropriate color and pattern"""
        self.current_state = state
        
        # Cancel any running pattern
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if state == LEDState.READY:
            # Solid green
            self._set_color(0, 100, 0)
            
        elif state == LEDState.LISTENING:
            # Pulsing green
            self._task = asyncio.create_task(self._pulse_pattern(0, 100, 0))
            
        elif state == LEDState.PROCESSING:
            # Blinking blue
            self._task = asyncio.create_task(self._blink_pattern(0, 0, 100))
            
        elif state == LEDState.SPEAKING:
            # Solid blue
            self._set_color(0, 0, 100)
            
        elif state == LEDState.STORY:
            # Gentle pulsing blue
            self._task = asyncio.create_task(self._pulse_pattern(0, 0, 100, speed=0.5))
            
        elif state == LEDState.ERROR:
            # Red flash
            self._task = asyncio.create_task(self._flash_pattern(100, 0, 0))
            
        elif state == LEDState.OFF:
            self._set_color(0, 0, 0)
    
    def _set_color(self, red: int, green: int, blue: int):
        """Set solid RGB color"""
        self.red_pwm.ChangeDutyCycle(red)
        self.green_pwm.ChangeDutyCycle(green)
        self.blue_pwm.ChangeDutyCycle(blue)
    
    async def _pulse_pattern(self, red: int, green: int, blue: int, speed: float = 1.0):
        """Create smooth pulsing pattern"""
        try:
            while True:
                # Fade in
                for i in range(0, 101, 2):
                    self._set_color(
                        red * i / 100,
                        green * i / 100,
                        blue * i / 100
                    )
                    await asyncio.sleep(0.02 / speed)
                
                # Fade out
                for i in range(100, -1, -2):
                    self._set_color(
                        red * i / 100,
                        green * i / 100,
                        blue * i / 100
                    )
                    await asyncio.sleep(0.02 / speed)
                
        except asyncio.CancelledError:
            pass
    
    async def _blink_pattern(self, red: int, green: int, blue: int):
        """Create blinking pattern"""
        try:
            while True:
                self._set_color(red, green, blue)
                await asyncio.sleep(0.5)
                self._set_color(0, 0, 0)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
    
    async def _flash_pattern(self, red: int, green: int, blue: int):
        """Create quick flash pattern"""
        try:
            for _ in range(3):
                self._set_color(red, green, blue)
                await asyncio.sleep(0.1)
                self._set_color(0, 0, 0)
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.set_state(LEDState.READY)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.red_pwm.stop()
        self.green_pwm.stop()
        self.blue_pwm.stop()
        GPIO.cleanup() 