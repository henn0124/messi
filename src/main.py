"""
Messi Assistant Main Application
------------------------------
A fun voice assistant for kids featuring fun facts and jokes.
"""

import os
import asyncio
import json
import base64
import websockets
import struct
import pvporcupine
from dotenv import load_dotenv
import pyaudio
import numpy as np
import sys
import aiohttp

# Audio configuration
CHANNELS = 1
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
INPUT_DEVICE = "hw:3,0"  # TONOR TM20
OUTPUT_DEVICE = "hw:4,0"  # USB2.0 Device

def get_audio_level_meter(level, width=40):
    """Creates a visual audio level meter"""
    meter_chars = "▁▂▃▄▅▆▇█"
    normalized = min(1.0, level)
    meter_width = int(normalized * width)
    return "█" * meter_width + "▁" * (width - meter_width) + f" [{level:.2f}]"

class MessiAssistant:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        self.agent_id = os.getenv('ELEVENLABS_AGENT_ID', "p5HeBE5gJwzXkNuqBFR4")
        self.porcupine_key = os.getenv('PORCUPINE_API_KEY')
        
        if not all([self.elevenlabs_key, self.porcupine_key]):
            raise ValueError("Missing required API keys")
            
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize Porcupine
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hey-messy_en_raspberry-pi_v3_0_0.ppn")
        self.porcupine = pvporcupine.create(
            access_key=self.porcupine_key,
            keyword_paths=[model_path]
        )
        
        print("Messi Assistant initialized!")
        print("Listening for 'Hey Messy'...")

    async def start(self):
        """Main entry point - starts wake word detection loop"""
        while True:
            try:
                await self.run_wake_word_detection()
            except Exception as e:
                print(f"Error in main loop: {e}")
                await asyncio.sleep(1)  # Prevent rapid retries

    async def run_wake_word_detection(self):
        """Runs wake word detection until 'Hey Messi' is detected"""
        
        # Open audio stream for wake word detection
        stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        
        print("Listening for wake word...")
        print("Audio levels (Ctrl+C to exit):")
        
        try:
            while True:
                # Read audio frame
                pcm = stream.read(self.porcupine.frame_length)
                pcm_unpacked = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                # Calculate audio level
                audio_data = np.array(pcm_unpacked, dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
                audio_level = np.abs(audio_data).mean()
                
                # Display level meter
                sys.stdout.write("\r" + get_audio_level_meter(audio_level))
                sys.stdout.flush()
                
                # Process with Porcupine
                keyword_index = self.porcupine.process(pcm_unpacked)
                
                # If wake word detected
                if keyword_index >= 0:
                    print("\nWake word detected!")
                    stream.stop_stream()
                    stream.close()
                    
                    # Start conversation
                    await self.handle_conversation()
                    
                    # Reopen stream for next detection
                    stream = self.audio.open(
                        rate=self.porcupine.sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.porcupine.frame_length
                    )
                    print("\nListening for wake word...")
                    
        except Exception as e:
            print(f"\nError in wake word detection: {e}")
            if stream.is_active():
                stream.stop_stream()
                stream.close()
            raise

    async def handle_conversation(self):
        """Handles conversation with ElevenLabs after wake word detection"""
        
        print("Starting conversation...")
        
        try:
            # First get the signed WebSocket URL via HTTP
            url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={self.agent_id}"
            headers = {"xi-api-key": self.elevenlabs_key}
            
            print(f"Getting WebSocket URL from: {url}")
            print(f"Using API key: {self.elevenlabs_key[:5]}...{self.elevenlabs_key[-5:]}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        print(f"Error getting WebSocket URL: {response.status}")
                        response_text = await response.text()
                        print(f"Response: {response_text}")
                        return
                    
                    data = await response.json()
                    print(f"Response data: {data}")
                    ws_url = data.get('signed_url')
                    if not ws_url:
                        print("No WebSocket URL received")
                        return
                    
                    print("Got WebSocket URL, connecting...")
            
            # Now connect to the WebSocket
            async with websockets.connect(ws_url) as websocket:
                print("Connected to ElevenLabs")
                
                # Send initialization message
                init_message = {
                    "type": "conversation_initiation_client_data",
                    "conversation_config_override": {
                        "audio_format": {
                            "type": "wav",
                            "sample_rate": SAMPLE_RATE,
                            "channels": CHANNELS,
                            "sample_width": SAMPLE_WIDTH
                        }
                    }
                }
                await websocket.send(json.dumps(init_message))
                
                # Open audio stream for conversation
                stream = self.audio.open(
                    rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=1024
                )
                
                # Start audio streaming task
                audio_task = asyncio.create_task(self.stream_audio(websocket, stream))
                
                # Start message handling task
                message_task = asyncio.create_task(self.handle_messages(websocket))
                
                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [audio_task, message_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending task
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Clean up
                stream.stop_stream()
                stream.close()
                
        except Exception as e:
            print(f"Error in conversation: {e}")
        finally:
            print("Conversation ended")

    async def stream_audio(self, websocket, stream):
        """Streams audio to ElevenLabs"""
        try:
            while True:
                # Read audio chunk
                chunk = stream.read(1024)
                if not chunk:
                    break
                    
                # Create audio message
                message = {
                    "type": "user_audio_chunk",
                    "user_audio_chunk": base64.b64encode(chunk).decode()
                }
                await websocket.send(json.dumps(message))
                await asyncio.sleep(0.01)  # Prevent flooding
                
        except Exception as e:
            print(f"Error streaming audio: {e}")

    async def handle_messages(self, websocket):
        """Handles incoming messages from ElevenLabs"""
        try:
            while True:
                message = await websocket.recv()
                
                # Handle binary audio data
                if isinstance(message, bytes):
                    await self.play_audio(message)
                    continue
                    
                # Handle JSON messages
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "user_transcript":
                        transcript = data.get("user_transcription_event", {}).get("user_transcript")
                        print(f"\nTranscript: {transcript}")
                    elif msg_type == "agent_response":
                        response = data.get("agent_response_event", {}).get("agent_response")
                        print(f"\n🤖 Messi: {response}")
                    elif msg_type == "error":
                        print(f"\nError from server: {data}")
                        return  # End conversation on error
                        
                except json.JSONDecodeError:
                    print(f"Received invalid JSON: {message}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"WebSocket closed: {e.code}")
        except Exception as e:
            print(f"Error handling messages: {e}")

    async def play_audio(self, audio_data):
        """Plays audio data through output device"""
        try:
            # Open playback stream
            stream = self.audio.open(
                rate=SAMPLE_RATE,
                channels=CHANNELS,
                format=pyaudio.paInt16,
                output=True
            )
            
            # Play audio
            stream.write(audio_data)
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Error playing audio: {e}")

    def cleanup(self):
        """Cleanup resources"""
        if self.porcupine is not None:
            self.porcupine.delete()
        if self.audio is not None:
            self.audio.terminate()

async def main():
    """Main entry point"""
    assistant = MessiAssistant()
    try:
        await assistant.start()
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 