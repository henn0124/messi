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
import uuid

from core.config import Settings
from core.logger import ConversationLogger
from core.learning_manager import LearningManager
from core.context_manager import ContextManager
from core.router import AssistantRouter
from core.cache_manager import ResponseCache
from openai import AsyncOpenAI
from core.speech import SpeechManager
from core.audio_processor import AudioProcessor
from core.audio import AudioInterface
from core.tts import TextToSpeech
from core.user_manager import UserManager

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
        # Initialize settings
        self.settings = Settings()
        
        # Initialize logger first
        self.logger = ConversationLogger()
        
        # Initialize user management
        self.user_manager = UserManager(settings=self.settings, logger=self.logger)
        
        # Initialize router with user manager
        self.router = AssistantRouter(user_manager=self.user_manager)
        
        # Initialize speech components
        self.speech = SpeechManager()
        self.tts = TextToSpeech()
        
        # Initialize learning system
        self.learning_manager = LearningManager() if self.settings.learning.enabled else None
        
        # Initialize audio and wake word detector last
        self.audio = None
        self.wake_word_detector = None
        
        # Initialize conversation state
        self.in_conversation = False
        self.current_user = None
        self.running = False

    async def initialize(self):
        """Initialize async components"""
        try:
            # Initialize router first
            await self.router.initialize()
            
            # Initialize audio interface
            from core.audio import AudioInterface
            self.audio = AudioInterface()
            if not await self.audio.initialize():
                raise RuntimeError("Failed to initialize audio interface")
            
            # Initialize wake word detector
            from core.wake_word import WakeWordDetector
            self.wake_word_detector = WakeWordDetector(settings=self.settings)
            
            return True
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            traceback.print_exc()
            return False

    async def start(self):
        """Start the assistant"""
        try:
            print("\nStarting Messi Assistant...")
            
            # Initialize components
            if not await self.initialize():
                print("❌ Failed to initialize components!")
                return
            
            self.running = True
            
            # Initialize and validate learning system
            if self.learning_manager:
                await self.learning_manager.initialize()
                
                # Validate learning system
                validation = await self.learning_manager.validate_learning_system()
                
                print("\nLearning System Validation:")
                print("Components:")
                for component, status in validation["components"].items():
                    print(f"  {component}: {'✓' if status else '✗'}")
                    
                print("\nFiles:")
                for file, status in validation["files"].items():
                    print(f"  {file}: {'✓' if status else '✗'}")
                    
                print("\nMetrics:")
                for metric, value in validation["metrics"].items():
                    print(f"  {metric}: {value}")
                    
                print("\nAutonomous Features:")
                for feature, status in validation["autonomous_features"].items():
                    print(f"  {feature}: {'✓' if status else '✗'}")
                    
                # Check if system is fully autonomous
                is_autonomous = all(validation["autonomous_features"].values())
                print(f"\nSystem is{' ' if is_autonomous else ' not '}fully autonomous")
            else:
                print("Learning system disabled - using static configurations")
            
            # Start wake word detection if everything is initialized
            if self.audio and self.wake_word_detector:
                await self.audio.start_wake_word_detection(self.on_wake_word)
                print("\nListening for wake word...")
            else:
                print("❌ Cannot start wake word detection - components not initialized!")
                return
            
            # Keep running until stopped
            while self.running:
                # Process learning queue during idle time
                if (self.learning_manager and 
                    not self.in_conversation and 
                    not self.learning_manager.is_processing):
                    await self.learning_manager.process_learning_queue()
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"Error starting assistant: {e}")
            traceback.print_exc()
        finally:
            await self.stop()

    async def on_wake_word(self, audio_data: bytes):
        """Handle wake word detection with user context"""
        try:
            # Identify user from voice
            self.current_user = await self.user_manager.identify_user(audio_data)
            user_id = self.current_user["id"]
            
            # Set conversation state
            self.in_conversation = True
            if self.learning_manager:
                self.learning_manager.in_conversation = True
            
            # Process speech to text
            text = await self.speech.process_audio(audio_data)
            
            if not text:
                print("No speech detected")
                self.in_conversation = False
                return
                
            print(f"\n✓ Recognized Text ({len(text)} chars):")
            print(f"'{text}'")
            
            # Create request context with user info
            request_context = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "conversation_id": str(uuid.uuid4())
            }
            
            # Process request with user context
            response = await self.router.route_request(text, request_context)
            
            # Update user activity
            self.user_manager.update_user_activity(user_id)
            
            # Record learning data if enabled
            if self.learning_manager and response:
                await self.learning_manager.record_exchange({
                    "text": text,
                    "context": response.get("context", "unknown"),
                    "success": True,
                    "user_id": user_id
                })
            
            # Generate and play response
            await self.tts.speak(response or {
                "text": "I'm having trouble understanding. Could you try again?",
                "context": "error",
                "auto_continue": True
            })
            
            # Handle continued conversation
            if response and response.get("conversation_active") and response.get("auto_continue"):
                print("\nConversation active, listening for follow-up...")
                await self.listen_for_followup()
            else:
                self.in_conversation = False
            
        except Exception as e:
            print(f"Error processing command: {e}")
            traceback.print_exc()
            # Provide fallback response
            await self.tts.speak({
                "text": "I'm having trouble processing that. Could you try again?",
                "context": "error",
                "auto_continue": False
            })
            self.in_conversation = False
        finally:
            # Reset conversation state
            if self.learning_manager:
                self.learning_manager.in_conversation = False

    async def listen_for_followup(self):
        """Listen for follow-up with user context"""
        try:
            await asyncio.sleep(0.5)  # Brief pause
            
            audio_data = await self.audio._collect_command_audio(duration=5.0)
            if audio_data:
                # Re-identify user for security
                if self.settings.user_config["session"]["require_reidentification"]:
                    self.current_user = await self.user_manager.identify_user(audio_data)
                
                text = await self.speech.process_audio(audio_data)
                
                if text and not text.endswith('...'):
                    await self.on_wake_word(audio_data)
                else:
                    print("\nIncomplete follow-up detected")
                    await self.tts.speak({
                        "text": "I didn't catch that. Could you repeat your question?",
                        "context": "clarification",
                        "auto_continue": True
                    })
                    await self.listen_for_followup()
            
        except Exception as e:
            print(f"Error in follow-up: {e}")
            traceback.print_exc()

    async def stop(self):
        """Clean shutdown"""
        print("\nStopping Messi Assistant...")
        self.running = False
        await self.audio.stop()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

if __name__ == "__main__":
    assistant = MessiAssistant()
    
    async def main():
        await assistant.initialize()
        await assistant.start()
    
    asyncio.run(main()) 