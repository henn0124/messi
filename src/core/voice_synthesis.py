"""
Voice Synthesis Manager using Elevenlabs
--------------------------------------
Handles conversation with Elevenlabs AI agent through websockets.
"""

import os
from pathlib import Path
from typing import Optional
import asyncio
from elevenlabs.client import ElevenLabs
import websockets
import json
import io
import wave
import requests

class VoiceSynthesis:
    def __init__(self, api_key: str, agent_id: str, voice_id: str = None):
        """Initialize voice synthesis with Elevenlabs"""
        self.api_key = api_key
        self.agent_id = agent_id
        self.voice_id = voice_id  # Not needed when using agent
        self.cache_dir = Path("cache/voice")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Websocket connection
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.conversation_id = None
        
        print(f"\nInitialized ElevenLabs client with API key: {api_key[:8]}...")
        print(f"Using agent ID: {agent_id}")

    async def get_signed_url(self) -> str:
        """Get a signed websocket URL from Elevenlabs."""
        headers = {"xi-api-key": self.api_key}
        response = requests.get(
            f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={self.agent_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()["signed_url"]

    async def start_conversation(self) -> bool:
        """Start a conversation with the Elevenlabs agent."""
        try:
            signed_url = await self.get_signed_url()
            print("✓ Got signed websocket URL")
            
            self.websocket = await websockets.connect(signed_url)
            print("✓ Connected to Elevenlabs agent")
            
            # Send initial message to start conversation with CD-quality audio settings
            await self.websocket.send(json.dumps({
                "type": "conversation_start",
                "data": {
                    "sampling_rate": 44100,  # CD-quality audio (44.1kHz)
                    "audio_format": "pcm_44100",  # CD-quality PCM
                    "chunk_size": 4096  # Larger chunks for better streaming
                }
            }))
            print("✓ Sent conversation start message")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start conversation: {str(e)}")
            return False

    async def process_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Process audio through the Elevenlabs agent and get response."""
        if not self.websocket:
            print("❌ No active websocket connection")
            return None
            
        try:
            # Convert input WAV to 44.1kHz PCM
            with wave.open(io.BytesIO(audio_data), 'rb') as wav:
                # Read original WAV parameters
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                
                # Read all frames
                pcm_data = wav.readframes(wav.getnframes())
                
                # If input is not 44.1kHz, we should resample
                # For now, just warn about mismatched rates
                if framerate != 44100:
                    print(f"⚠️ Input audio sample rate ({framerate}Hz) doesn't match output (44100Hz)")
            
            # Send audio data
            await self.websocket.send(json.dumps({
                "type": "audio_data",
                "data": {
                    "audio": pcm_data.hex()
                }
            }))
            print("✓ Sent audio data")
            
            # Wait for responses until we get audio data
            while True:
                print("Waiting for response...")
                response = await self.websocket.recv()
                print(f"Got response: {response[:200]}...")  # Print first 200 chars
                
                response_data = json.loads(response)
                message_type = response_data.get("type")
                
                print(f"Message type: {message_type}")
                
                if message_type == "audio_data":
                    audio_hex = response_data["data"]["audio"]
                    pcm_bytes = bytes.fromhex(audio_hex)
                    
                    # Convert PCM to WAV at 44.1kHz
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wav:
                        wav.setnchannels(1)  # Mono
                        wav.setsampwidth(2)  # 16-bit
                        wav.setframerate(44100)  # 44.1kHz
                        wav.writeframes(pcm_bytes)
                    
                    wav_bytes = wav_buffer.getvalue()
                    
                    # Save response audio for debugging
                    with open("debug_response.wav", "wb") as f:
                        f.write(wav_bytes)
                    print("✓ Saved response audio")
                    
                    return wav_bytes
                    
                elif message_type == "error":
                    print(f"❌ Error from server: {response_data.get('data', {}).get('message', 'Unknown error')}")
                    return None
                    
                elif message_type == "status":
                    print(f"Status update: {response_data.get('data', {}).get('message', 'No message')}")
                    continue
                    
                elif message_type == "conversation_initiation_metadata":
                    # Store conversation ID and continue
                    metadata = response_data.get("conversation_initiation_metadata_event", {})
                    self.conversation_id = metadata.get("conversation_id")
                    print(f"✓ Got conversation ID: {self.conversation_id}")
                    continue
                    
                else:
                    print(f"Unknown message type: {message_type}")
                    continue
            
        except Exception as e:
            print(f"❌ Error processing audio: {str(e)}")
            return None

    async def end_conversation(self):
        """End the conversation and close the websocket connection."""
        if self.websocket:
            try:
                await self.websocket.close()
                print("✓ Closed websocket connection")
            except Exception as e:
                print(f"❌ Error closing websocket: {str(e)}")
            finally:
                self.websocket = None

    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached audio files"""
        try:
            current_time = asyncio.get_event_loop().time()
            for file in self.cache_dir.glob("*.wav"):
                file_age = current_time - file.stat().st_mtime
                if file_age > max_age_hours * 3600:
                    file.unlink()
        except Exception as e:
            print(f"Error cleaning cache: {e}") 