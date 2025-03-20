"""
ElevenLabs Integration Test
--------------------------
Tests the connection and interaction with ElevenLabs websocket API.
"""

import os, asyncio, websockets, json, requests
from dotenv import load_dotenv

async def test_elevenlabs():
    """Test ElevenLabs websocket connection and agent interaction."""
    try:
        print("\n🚀 Starting ElevenLabs integration test...")
        
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('ELEVENLABS_API_KEY')
        agent_id = os.getenv('ELEVENLABS_AGENT_ID')
        
        print("\n📝 Configuration:")
        print(f"API Key: {'*' * len(api_key) if api_key else 'Not found'}")
        print(f"Agent ID: {agent_id if agent_id else 'Not found'}")
        
        if not api_key or not agent_id:
            print("\n❌ Missing required environment variables")
            return
        
        print(f"\n📝 Using Agent ID: {agent_id}")
        
        print("\n🔑 Getting signed WebSocket URL...")
        
        # Get signed URL
        url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={agent_id}"
        headers = {"xi-api-key": api_key}
        
        print(f"\nMaking request to: {url}")
        response = requests.get(url, headers=headers)
        
        print(f"\nResponse status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        if not response.ok:
            print(f"\n❌ Failed to get signed URL: {response.text}")
            return
        
        signed_url = response.json()["signed_url"]
        print(f"\n✓ Got signed URL: {signed_url}")
        
        print("\n🔌 Connecting to WebSocket...")
        async with websockets.connect(signed_url) as ws:
            print("✓ Connected successfully")
            
            # Send conversation initialization
            init_message = {
                "type": "conversation_initiation_client_data",
                "conversation_config_override": {}
            }
            
            print("\n📤 Sending initialization message:")
            print(json.dumps(init_message, indent=2))
            await ws.send(json.dumps(init_message))
            print("✓ Sent initialization")
            
            # Wait for responses
            print("\n📥 Waiting for responses...")
            while True:
                try:
                    msg = await ws.recv()
                    if isinstance(msg, str):
                        try:
                            data = json.loads(msg)
                            print("\nReceived message:")
                            print(json.dumps(data, indent=2))
                            
                            # If we get the initialization metadata, send our test message
                            if data.get("type") == "conversation_initiation_metadata":
                                test_message = "Hello! Can you tell me about yourself?"
                                print(f"\n📤 Sending message: {test_message}")
                                await ws.send(json.dumps({"text": test_message}))
                                print("✓ Sent message")
                                
                        except json.JSONDecodeError:
                            print("\nReceived non-JSON message:", msg)
                    else:
                        print(f"\nReceived audio data: {len(msg)} bytes")
                except Exception as e:
                    print(f"\n❌ Error during WebSocket communication: {e}")
                    break
            
            print("\n✅ Test completed!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_elevenlabs()) 