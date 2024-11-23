"""
LED Manager for Messi Assistant
-----------------------------

This module manages the RGB LED indicator system for the Messi Assistant,
providing visual feedback about the assistant's current state and activities
through color patterns and animations.

Key Features:
    1. State Management:
        - Ready state (solid green)
        - Listening mode (pulsing green)
        - Processing state (blinking blue)
        - Speaking state (solid blue)
        - Story mode (gentle pulsing blue)
        - Error indication (red flash)
    
    2. LED Patterns:
        - Solid colors
        - Smooth pulsing
        - Blinking patterns
        - Flash sequences
        - Fade transitions
    
    3. Hardware Control:
        - RGB LED control via GPIO
        - PWM for smooth transitions
        - Multiple animation patterns
        - Resource cleanup
    
    4. Async Operation:
        - Non-blocking patterns
        - Pattern interruption
        - State transitions
        - Task management

Hardware Setup:
    GPIO Pins:
        - Red:   GPIO17
        - Green: GPIO27
        - Blue:  GPIO22
    
    Connection:
        - Common anode/cathode RGB LED
        - Current-limiting resistors required
        - PWM capability on all channels

Usage:
    led = LEDManager()
    
    # Set different states
    await led.set_state(LEDState.READY)      # Ready for wake word
    await led.set_state(LEDState.LISTENING)  # Listening for command
    await led.set_state(LEDState.SPEAKING)   # Speaking response
    
    # Cleanup when done
    led.cleanup()

States:
    READY      - Solid green:       Ready for wake word
    LISTENING  - Pulsing green:     Actively listening
    PROCESSING - Blinking blue:     Processing request
    SPEAKING   - Solid blue:        Speaking response
    STORY      - Gentle blue pulse: Telling story
    ERROR      - Red flash:         Error occurred
    OFF        - No light:          System inactive

Safety:
    - Includes GPIO cleanup
    - Handles pattern interruption
    - Manages PWM resources
    - Graceful state transitions

Author: Your Name
Created: 2024-01-24
"""

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