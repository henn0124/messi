"""
Voice Synthesis Manager using Elevenlabs
--------------------------------------
Handles conversation with Elevenlabs AI agent through websockets.
"""

import os
from pathlib import Path
from typing import Optional
import asyncio
import websockets
import json
import io
import wave
import requests
import traceback
import aiohttp
import base64

class VoiceSynthesisManager:
    def __init__(self, api_key: str, agent_id: str):
        """Initialize the voice synthesis manager."""
        self.api_key = api_key
        self.agent_id = agent_id
        self.websocket = None
        self.conversation_id = None
        
    async def start_conversation(self):
        """Start a new conversation with the Elevenlabs agent."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
        try:
            # Get signed websocket URL
            url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={self.agent_id}"
            headers = {"xi-api-key": self.api_key}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        print(f"❌ Failed to get websocket URL: {response.status}")
                        return False
                    data = await response.json()
                    websocket_url = data["signed_url"]
                    print("✓ Got signed websocket URL")
            
            # Connect to websocket
            self.websocket = await websockets.connect(websocket_url)
            print("✓ Connected to Elevenlabs agent")
            
            # Send initial configuration
            config_message = {
                "type": "conversation_initiation_client_data",
                "conversation_config_override": {
                    "agent": {
                        "prompt": {
                            "prompt": "You are Messi, a friendly AI assistant for kids."
                        },
                        "first_message": "Hi! I'm Messi, your friendly AI assistant. How can I help you today?",
                        "language": "en"
                    }
                }
            }
            await self.websocket.send(json.dumps(config_message))
            print("✓ Sent conversation configuration")
            
            # Wait for conversation metadata
            while True:
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    message_type = response_data.get("type")
                    
                    if message_type == "conversation_initiation_metadata":
                        metadata = response_data.get("conversation_initiation_metadata_event", {})
                        self.conversation_id = metadata.get("conversation_id")
                        print(f"✓ Got conversation ID: {self.conversation_id}")
                        print(f"✓ Agent output format: {metadata.get('agent_output_audio_format')}")
                        print(f"✓ User input format: {metadata.get('user_input_audio_format')}")
                        return True
                    elif message_type == "ping":
                        # Handle ping
                        ping_event = response_data.get("ping_event", {})
                        event_id = ping_event.get("event_id")
                        pong_message = {
                            "type": "pong",
                            "event_id": event_id
                        }
                        await self.websocket.send(json.dumps(pong_message))
                        continue
                        
                except asyncio.TimeoutError:
                    print("❌ Timeout waiting for conversation metadata")
                    return False
                    
        except Exception as e:
            print(f"❌ Failed to start conversation: {str(e)}")
            return False
            
    async def end_conversation(self):
        """End the current conversation."""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"⚠️ Error closing websocket: {str(e)}")
            finally:
                self.websocket = None
                self.conversation_id = None
                
    async def process_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Process audio through the Elevenlabs agent and get response."""
        if not self.websocket:
            print("❌ No active websocket connection")
            if not await self.start_conversation():
                return None
            
        try:
            # Convert input WAV to PCM
            with wave.open(io.BytesIO(audio_data), 'rb') as wav:
                # Read original WAV parameters
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                
                # Read all frames
                pcm_data = wav.readframes(wav.getnframes())
                
                # Verify sample rate
                if framerate != 16000:
                    print(f"⚠️ Input audio sample rate ({framerate}Hz) doesn't match expected rate (16000Hz)")
            
            # Send audio data
            message = {
                "type": "user_audio_chunk",
                "user_audio_chunk": base64.b64encode(pcm_data).decode()
            }
            print(f"📤 Sending audio chunk ({len(pcm_data)} bytes)")
            await self.websocket.send(json.dumps(message))
            print("✓ Sent audio data")
            
            # Wait for responses until we get audio data
            while True:
                print("\n⏳ Waiting for response...")
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    print(f"📥 Got response length: {len(response)} bytes")
                    print(f"📥 First 200 chars: {response[:200]}...")
                    
                    response_data = json.loads(response)
                    message_type = response_data.get("type")
                    
                    print(f"📥 Message type: {message_type}")
                    
                    if message_type == "audio":
                        print("🎵 Processing audio data...")
                        audio_event = response_data.get("audio_event", {})
                        audio_base64 = audio_event.get("audio_base_64")
                        if not audio_base64:
                            print("❌ No audio data in response")
                            continue
                            
                        pcm_bytes = base64.b64decode(audio_base64)
                        print(f"🎵 Received {len(pcm_bytes)} bytes of audio")
                        
                        # Convert PCM to WAV at 16kHz
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, 'wb') as wav:
                            wav.setnchannels(1)  # Mono
                            wav.setsampwidth(2)  # 16-bit
                            wav.setframerate(16000)  # 16kHz
                            wav.writeframes(pcm_bytes)
                        
                        wav_bytes = wav_buffer.getvalue()
                        print(f"🎵 Converted to {len(wav_bytes)} bytes WAV")
                        
                        # Save response audio for debugging
                        with open("debug_response.wav", "wb") as f:
                            f.write(wav_bytes)
                        print("✓ Saved response audio to debug_response.wav")
                        
                        return wav_bytes
                        
                    elif message_type == "error":
                        error_msg = response_data.get("data", {}).get("message", "Unknown error")
                        print(f"❌ Error from server: {error_msg}")
                        return None
                        
                    elif message_type == "user_transcript":
                        transcript = response_data.get("user_transcription_event", {}).get("user_transcript")
                        print(f"📝 Transcript: {transcript}")
                        continue
                        
                    elif message_type == "agent_response":
                        response = response_data.get("agent_response_event", {}).get("agent_response")
                        print(f"🤖 Agent: {response}")
                        continue
                        
                    elif message_type == "ping":
                        # Respond to ping with pong
                        ping_event = response_data.get("ping_event", {})
                        event_id = ping_event.get("event_id")
                        pong_message = {
                            "type": "pong",
                            "event_id": event_id
                        }
                        print("📤 Sending pong...")
                        await self.websocket.send(json.dumps(pong_message))
                        continue
                        
                    else:
                        print(f"ℹ️ Ignoring message type: {message_type}")
                        continue
                        
                except asyncio.TimeoutError:
                    print("❌ Timeout waiting for response from ElevenLabs")
                    return None
                    
                except Exception as e:
                    print(f"❌ Error processing response: {str(e)}")
                    return None
            
        except Exception as e:
            print(f"❌ Error processing audio: {str(e)}")
            print(f"❌ Full error: {traceback.format_exc()}")
            return None

    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached audio files"""
        try:
            current_time = asyncio.get_event_loop().time()
            for file in Path("cache/voice").glob("*.wav"):
                file_age = current_time - file.stat().st_mtime
                if file_age > max_age_hours * 3600:
                    file.unlink()
        except Exception as e:
            print(f"Error cleaning cache: {e}") 