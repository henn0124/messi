from typing import Optional
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
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, psutil.cpu_count(logical=False) - 1),
            thread_name_prefix="messi_worker"
        )
        
        # Initialize components with resource limits
        self.chunk_size = 32 * 1024  # 32KB audio chunks
        self.max_buffer_size = 256 * 1024  # 256KB max buffer
        self.cache_size_limit = 1024 * 1024  # 1MB cache limit
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Component initialization
        self.resource_monitor.print_usage("Init")
        self.audio = AudioInterface()
        self.router = AssistantRouter()
        self.cache = ResponseCache(max_size=self.cache_size_limit)
        self.tts = TextToSpeech()
        self.conversation = ConversationManager()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        
        # Audio processing parameters
        self.SAMPLE_RATE = 16000
        self.MIN_AUDIO_LENGTH = 0.5
        self.MIN_AUDIO_LEVEL = 50  # Lowered from 100
        self.MAX_AUDIO_LENGTH = 30.0
        self.TARGET_RMS = 2000  # Target RMS level for normalization
        
        self.speech_manager = SpeechManager()
        self.audio_processor = AudioProcessor()

    async def _process_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio using centralized processor"""
        try:
            # Process audio
            processed_audio, metrics = await self.audio_processor.process_input(audio_data)
            if processed_audio is None:
                return None

            # Save for Whisper
            wav_path = await self.audio_processor.save_wav(
                processed_audio,
                AudioProcessor.SETTINGS["sample_rates"]["processing"],
                "whisper_input.wav"
            )
            
            if wav_path:
                # Process with Whisper
                with open(wav_path, "rb") as audio_file:
                    response = await self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        language="en",
                        temperature=0.0
                    )
                
                # Cleanup
                wav_path.unlink()
                return response if response else None
            
            return None
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None

    async def _handle_wake_word(self, audio_data: bytes):
        """Handle wake word with speech processing"""
        try:
            print("\n▶ Processing speech...")
            
            # Process through Whisper
            text = await self.speech_manager.process_audio(audio_data)
            
            if text:
                print(f"\n✓ Recognized Text ({len(text)} chars):")
                print(f"'{text}'")
                
                print("\n▶ Processing request...")
                response = await self._handle_request(text)
                if response:
                    print("\n▶ Generating response...")
                    await self._play_response(response)
                    
        except Exception as e:
            print(f"Error handling wake word: {e}")
            traceback.print_exc()

    async def _cleanup(self):
        """Clean up resources with monitoring"""
        try:
            self.resource_monitor.print_usage("Starting Cleanup")
            
            # Stop audio processing
            await self.audio.stop()
            
            # Clean up executor
            self.executor.shutdown(wait=True)
            
            # Clear caches
            await self.cache.clear()
            
            # End conversation
            await self.conversation.end_conversation()
            
            # Force garbage collection
            gc.collect()
            
            self.resource_monitor.print_usage("Cleanup Complete")
            
        except Exception as e:
            print(f"Cleanup error: {e}")

    async def run(self):
        """Main loop with peak tracking"""
        try:
            print("\nStarting Messi Assistant...")
            self.resource_monitor.print_usage("Startup")
            
            # Start monitoring task here, inside the event loop
            monitor_task = asyncio.create_task(
                self.resource_monitor.monitor_critical_thresholds()
            )
            
            # Initialize audio
            await self.audio.initialize()
            
            # Start wake word detection
            await self.audio.start_wake_word_detection(self._handle_wake_word)
            
            self.resource_monitor.print_usage("Initialization Complete")
            print("\nListening for wake word...")
            
            while True:
                await asyncio.sleep(5)
                self.resource_monitor.print_usage("Running")
                
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"Fatal error in main loop: {e}")
        finally:
            self.resource_monitor.print_usage("Cleanup")
            await self._cleanup()

    async def speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert speech to text with resource monitoring"""
        try:
            self.resource_monitor.print_usage("Speech-to-Text Start")
            
            # Create temporary WAV file
            temp_path = Path("temp_audio.wav")
            with wave.open(str(temp_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            
            # Send to Whisper API
            with open(temp_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                    language="en"
                )
            
            # Clean up
            temp_path.unlink()
            
            self.resource_monitor.print_usage("Speech-to-Text Complete")
            return transcript
            
        except Exception as e:
            print(f"Speech-to-text error: {e}")
            return None

    async def _handle_request(self, text: str) -> Optional[dict]:
        """Handle user request with resource monitoring"""
        try:
            self.resource_monitor.print_usage("Request Processing Start")
            
            # Check cache first
            cached = await self.cache.get(text)
            if cached:
                print("\nUsing cached response")
                return cached
            
            # Process request through router
            print("\n▶ Routing request...")
            response = await self.router.route_request(text)
            
            # Cache result if not too large
            if response and len(str(response)) < self.cache_size_limit:
                await self.cache.set(text, response)
            
            self.resource_monitor.print_usage("Request Processing Complete")
            return response
            
        except Exception as e:
            print(f"Request handling error: {e}")
            self.resource_monitor.print_usage("Request Processing Error")
            return None

    async def _play_response(self, response: dict):
        """Play response with resource monitoring"""
        try:
            self.resource_monitor.print_usage("Response Generation")
            
            if not response or not response.get("text"):
                print("No response to play")
                return
                
            print(f"\nAssistant: {response['text']}")
            
            # Generate speech
            audio_data = await self.tts.synthesize(response["text"])
            if not audio_data:
                print("Failed to generate speech")
                return
            
            # Play audio
            print("\n▶ Playing response...")
            await self.audio.play_audio_chunk(audio_data)
            
            self.resource_monitor.print_usage("Response Complete")
            
        except Exception as e:
            print(f"Error playing response: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    assistant = MessiAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {e}") 