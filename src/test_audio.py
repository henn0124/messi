"""
Audio Output Test Script
----------------------
Tests the audio output functionality using Elevenlabs voice synthesis.
"""

import asyncio
import os
from dotenv import load_dotenv
from core.audio import AudioManager
import pyaudio

async def test_audio():
    print("\n🎵 Starting audio test...")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Print available devices
    print("\n📋 Available audio devices:")
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            print(f"\nDevice {i}: {info['name']}")
            print(f"  Input channels: {info['maxInputChannels']}")
            print(f"  Output channels: {info['maxOutputChannels']}")
            print(f"  Default sample rate: {info['defaultSampleRate']}")
        except Exception as e:
            print(f"Error getting device info: {e}")
    
    # Find input device (TONOR TM20)
    input_index = -1
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if "TONOR TM20" in info["name"] and info["maxInputChannels"] > 0:
                input_index = i
                break
        except:
            continue
    
    if input_index == -1:
        print("\n❌ TONOR TM20 microphone not found!")
        p.terminate()
        return
    
    print(f"\n✓ Found TONOR TM20 at index {input_index}")
    
    # Find output device (USB2.0 Device)
    output_index = -1
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if "USB2.0 Device" in info["name"] and info["maxOutputChannels"] > 0:
                output_index = i
                break
        except:
            continue
    
    if output_index == -1:
        print("\n❌ USB2.0 Device speaker not found!")
        p.terminate()
        return
    
    print(f"\n✓ Found USB2.0 Device at index {output_index}")
    
    try:
        # Open input stream
        print("\n🎙️ Opening input stream...")
        input_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            input=True,
            input_device_index=input_index,
            frames_per_buffer=1024
        )
        print("✓ Input stream opened successfully")
        
        # Open output stream
        print("\n🔊 Opening output stream...")
        output_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            output=True,
            output_device_index=output_index,
            frames_per_buffer=1024
        )
        print("✓ Output stream opened successfully")
        
        # Record and play back a short sample
        print("\n🎤 Recording 3 seconds of audio...")
        frames = []
        for _ in range(int(48000 / 1024 * 3)):  # 3 seconds
            data = input_stream.read(1024)
            frames.append(data)
        
        print("\n🔊 Playing back recorded audio...")
        for frame in frames:
            output_stream.write(frame)
        
        print("\n✨ Audio test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during audio test: {str(e)}")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if 'input_stream' in locals():
            input_stream.stop_stream()
            input_stream.close()
        if 'output_stream' in locals():
            output_stream.stop_stream()
            output_stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(test_audio()) 