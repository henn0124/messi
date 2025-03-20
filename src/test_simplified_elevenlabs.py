"""
Simplified ElevenLabs Integration Test
------------------------------------
Tests the simplified websocket-based conversation with ElevenLabs AI agent.
"""

import os
import asyncio
import websockets
import json
import aiohttp
from dotenv import load_dotenv
import wave
import subprocess
import time
import base64

# Audio device configuration
INPUT_DEVICE = "hw:3,0"  # TONOR TM20
OUTPUT_DEVICE = "hw:4,0"  # USB2.0 Device
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio
RECORD_SECONDS = 3

async def record_audio(output_file):
    print("Recording audio...")
    cmd = f"arecord -D {INPUT_DEVICE} -f S16_LE -c {CHANNELS} -r {SAMPLE_RATE} -d {RECORD_SECONDS} {output_file}"
    subprocess.run(cmd, shell=True)
    
    # Read and return audio properties
    with wave.open(output_file, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        print(f"Recorded audio: {channels} channel(s), {width} bytes sample width, {rate} Hz frame rate, {frames} frames")
    return True

async def play_audio(audio_file):
    print("Playing audio response...")
    cmd = f"aplay -D {OUTPUT_DEVICE} {audio_file}"
    subprocess.run(cmd, shell=True)

async def save_audio_response(audio_data, output_file):
    # Decode base64 audio data and save to file
    audio_bytes = base64.b64decode(audio_data)
    with open(output_file, 'wb') as f:
        f.write(audio_bytes)
    return True

async def test_simplified_elevenlabs():
    """Test simplified ElevenLabs websocket integration"""
    try:
        print("\n🚀 Testing simplified ElevenLabs integration...")
        
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('ELEVENLABS_API_KEY')
        agent_id = os.getenv('ELEVENLABS_AGENT_ID', "p5HeBE5gJwzXkNuqBFR4")
        
        if not api_key:
            print("\n❌ Missing required environment variables")
            return
        
        print("\n📝 Configuration:")
        print(f"API Key: {'*' * len(api_key)}")
        print(f"Agent ID: {agent_id}")
        
        # List audio devices
        print("\nAvailable audio devices:")
        subprocess.run(["arecord", "-l"], capture_output=False)
        print("\nPlayback devices:")
        subprocess.run(["aplay", "-l"], capture_output=False)
        
        conversation_active = True
        turn_count = 0
        
        while conversation_active:
            turn_count += 1
            print(f"\n=== Conversation Turn {turn_count} ===")
            
            # Record audio input
            input_file = "test_input.wav"
            await record_audio(input_file)
            
            # Get signed URL for WebSocket connection
            url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={agent_id}"
            headers = {"xi-api-key": api_key}
            
            # Get the signed WebSocket URL
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        print(f"\n❌ Failed to get signed URL: {await response.text()}")
                        return
                    signed_url = (await response.json())["signed_url"]
                    print("✓ Got signed URL")

            try:
                async with websockets.connect(signed_url) as websocket:
                    print("\nConnected to WebSocket")

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
                    print("✓ Sent initialization")

                    # Read recorded audio file
                    with open(input_file, 'rb') as f:
                        # Skip WAV header (44 bytes)
                        f.seek(44)
                        audio_data = f.read()

                    # Calculate chunk size for 1-second chunks
                    chunk_size = SAMPLE_RATE * SAMPLE_WIDTH
                    chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]

                    print(f"Sending audio in {len(chunks)} chunks...")
                    
                    # Send each chunk as a JSON message
                    for i, chunk in enumerate(chunks):
                        message = {
                            "type": "user_audio_chunk",
                            "user_audio_chunk": base64.b64encode(chunk).decode()
                        }
                        await websocket.send(json.dumps(message))
                        print(f"Sent chunk {i + 1}/{len(chunks)}")
                        await asyncio.sleep(0.1)  # Small delay between chunks

                    # Send end-of-stream message
                    end_message = {
                        "type": "user_audio_chunk",
                        "user_audio_chunk": ""
                    }
                    await websocket.send(json.dumps(end_message))
                    print("✓ Sent end-of-stream")

                    print("\nReceiving response...")
                    output_file = f"response_{turn_count}.wav"
                    current_audio = bytearray()
                    
                    while True:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            
                            # Handle binary audio data
                            if isinstance(message, bytes):
                                print(f"\nReceived audio response: {len(message)} bytes")
                                current_audio.extend(message)
                                continue
                                
                            # Handle JSON messages
                            try:
                                data = json.loads(message)
                                msg_type = data.get("type", "unknown")
                                print(f"\nReceived message type: {msg_type}")
                                print(f"Full message: {message}")

                                if msg_type == "user_transcript":
                                    transcript = data.get("user_transcription_event", {}).get("user_transcript")
                                    print(f"\n📝 Transcript: {transcript}")
                                elif msg_type == "agent_response":
                                    response = data.get("agent_response_event", {}).get("agent_response")
                                    print(f"\n🤖 Agent: {response}")
                                elif msg_type == "error":
                                    print(f"\n❌ Error from server: {data}")
                                    break
                            except json.JSONDecodeError:
                                print(f"\nReceived non-JSON message: {message}")

                        except asyncio.TimeoutError:
                            print("\nTimeout waiting for response, closing connection...")
                            break
                        except websockets.exceptions.ConnectionClosed:
                            print("\n✓ WebSocket connection closed")
                            break
                        except Exception as e:
                            print(f"\nError processing message: {e}")
                            break

                    # Save and play the complete audio response
                    if current_audio:
                        with open(output_file, 'wb') as f:
                            f.write(current_audio)
                        await play_audio(output_file)

                    # Ask if user wants to continue
                    user_input = input("\nContinue conversation? (y/n): ").lower()
                    conversation_active = user_input.startswith('y')

            except Exception as e:
                print(f"Error: {e}")
                break

        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simplified_elevenlabs()) 