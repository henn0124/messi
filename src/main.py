from typing import Optional, Dict
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
import os
from pathlib import Path
import time
import wave
import csv
from datetime import datetime
import traceback
import yaml

from core.audio import AudioInterface
from core.assistant_router import AssistantRouter
from core.config import Settings
from core.cache_manager import ResponseCache
from core.tts import TextToSpeech
from core.conversation_manager import ConversationManager
from openai import AsyncOpenAI
from core.speech import SpeechManager
from core.audio_processor import AudioProcessor

class ResourceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory_mb = 0
        self.peak_cpu = 0
        self.last_print_time = time.time()
        self.print_interval = 5  # Only update display every 5 seconds
    
    @staticmethod
    def get_minimal_usage():
        process = psutil.Process()
        return {
            'cpu': psutil.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / (1024 * 1024),  # Convert to MB
            'thread_count': process.num_threads()
        }
    
    def update_peaks(self, usage):
        """Update peak values"""
        self.peak_cpu = max(self.peak_cpu, usage['cpu'])
        self.peak_memory_mb = max(self.peak_memory_mb, usage['memory_mb'])
    
    def print_usage(self, label: str = ""):
        """Print current and peak resource usage"""
        usage = self.get_minimal_usage()
        self.update_peaks(usage)
        
        current_time = time.time()
        # Only print if it's been more than print_interval seconds
        if current_time - self.last_print_time >= self.print_interval:
            print(f"\nSystem Resources {label}:")
            print(f"Current: CPU: {usage['cpu']:>3.1f}% | Memory: {usage['memory_mb']:>5.1f}MB | Threads: {usage['thread_count']}")
            print(f"Peak:    CPU: {self.peak_cpu:>3.1f}% | Memory: {self.peak_memory_mb:>5.1f}MB")
            print(f"Uptime:  {(current_time - self.start_time):.0f}s")
            self.last_print_time = current_time

    async def monitor_critical_thresholds(self):
        """Monitor critical thresholds with peak tracking"""
        while True:
            usage = self.get_minimal_usage()
            self.update_peaks(usage)
            
            # Alert on critical conditions
            if usage['cpu'] > 90 or usage['memory_mb'] > 400:
                print("\n⚠️  High Resource Usage:")
                print(f"Current: CPU: {usage['cpu']:>3.1f}% | Memory: {usage['memory_mb']:>5.1f}MB")
                print(f"Peak:    CPU: {self.peak_cpu:>3.1f}% | Memory: {self.peak_memory_mb:>5.1f}MB")
            
            await asyncio.sleep(5)

class MessiAssistant:
    def __init__(self):
        self.settings = Settings()
        self.audio = AudioInterface()
        self.tts = TextToSpeech()
        self.speech = SpeechManager()
        self.router = AssistantRouter()
        self.running = False
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def start(self):
        """Start the assistant"""
        try:
            print("\nStarting Messi Assistant...")
            self.running = True
            
            # Initialize audio with detailed logging
            print("\nInitializing audio interface...")
            print("1. Loading audio config...")
            audio_config = self.settings._load_yaml_config().get('audio', {})
            print(f"Audio config: {audio_config}")
            
            print("\n2. Setting up audio stream...")
            await self.audio.initialize()
            
            if not self.audio.stream:
                print("❌ Failed to initialize audio stream!")
                return
            
            print("✓ Audio stream initialized")
            print(f"Device: {self.audio.stream._device_info['name']}")
            print(f"Rate: {self.audio.stream._rate}Hz")
            print(f"Channels: {self.audio.stream._channels}")
            
            # Start wake word detection with callback
            print("\n3. Starting wake word detection...")
            await self.audio.start_wake_word_detection(self.on_wake_word)
            
            print("\nListening for wake word...")
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"Error starting assistant: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()

    async def stop(self):
        """Stop the assistant"""
        print("\nStopping Messi Assistant...")
        self.running = False
        await self.audio.stop()
        self.executor.shutdown(wait=True)

    async def on_wake_word(self, audio_data: bytes):
        """Handle wake word detection"""
        try:
            # Process speech to text
            print("\n▶ Processing speech...")
            text = await self.speech.process_audio(audio_data)
            
            if not text:
                print("No speech detected")
                return
                
            print(f"\n✓ Recognized Text ({len(text)} chars):")
            print(f"'{text}'")
            
            # Process request
            print("\n▶ Processing request...")
            response = await self.router.route_request(text)
            
            # Generate and play response
            print("\n▶ Playing response...")
            print(f"\nA: {response}")
            
            await self.tts.speak(response)
            
        except Exception as e:
            print(f"Error processing command: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Create and start assistant
    assistant = MessiAssistant()
    
    try:
        # Run the assistant
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc() 