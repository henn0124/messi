#!/usr/bin/env python3
import os
import pyaudio
import numpy as np
import subprocess
import wave
import tempfile
import time

print("USB Speaker Test Script")
print("======================")

# List all audio devices with PyAudio
p = pyaudio.PyAudio()
print("\n1. Available Audio Devices:")
print("--------------------------")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    name = dev['name']
    channels = dev['maxOutputChannels']
    print(f"Device {i}: {name} (Output channels: {channels})")

# Test 1: Generate and play a sine wave tone with PyAudio
def test_pyaudio_output(device_index):
    print(f"\n2. Testing PyAudio output on device index {device_index}")
    print("-------------------------------------------------")
    try:
        # Generate a sine wave
        sample_rate = 44100
        duration = 2  # seconds
        frequency = 440  # A4 note
        
        # Generate data
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
        audio_data = (tone * 32767).astype(np.int16)
        
        # Open stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        output=True,
                        output_device_index=device_index,
                        frames_per_buffer=1024)
                        
        # Play tone
        print("Playing tone... (you should hear a 2-second beep)")
        
        # Convert to bytes and play in chunks
        audio_bytes = audio_data.tobytes()
        chunk_size = 1024 * 2  # 1024 frames, 2 bytes per frame
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            stream.write(chunk)
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        
        print("Tone playback complete with PyAudio")
        return True
    except Exception as e:
        print(f"Error with PyAudio: {e}")
        return False

# Test 2: Create a WAV file and play it with aplay
def test_aplay_output(device_name):
    print(f"\n3. Testing aplay output to device '{device_name}'")
    print("-------------------------------------------------")
    try:
        # Create WAV file
        sample_rate = 44100
        duration = 2  # seconds
        frequency = 523  # C5 note (different from PyAudio test)
        
        # Generate data
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
        audio_data = (tone * 32767).astype(np.int16)
        
        # Save as WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_file = f.name
            
        with wave.open(wav_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
            
        print(f"Created test WAV file: {wav_file}")
        
        # Play with aplay
        print("Playing tone with aplay... (you should hear a 2-second beep)")
        cmd = ["aplay", "-D", device_name, wav_file]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("aplay playback successful")
        else:
            print(f"aplay error: {result.stderr}")

        # Try alternate device format
        if result.returncode != 0:
            print("\nTrying alternate device format...")
            cmd = ["aplay", "-D", f"plughw:{device_name.split(':')[1]}", wav_file]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("aplay playback successful with alternate device format")
            else:
                print(f"aplay error: {result.stderr}")
        
        # Cleanup
        os.unlink(wav_file)
        print(f"Removed test file: {wav_file}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error with aplay: {e}")
        return False

# Find USB device
usb_device_index = None
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if "USB2.0 Device" in dev['name'] and dev['maxOutputChannels'] > 0:
        usb_device_index = i
        print(f"\nFound USB speaker at index {i}: {dev['name']}")
        break

if usb_device_index is None:
    print("\nUSB speaker not found. Using default output device (index 0)")
    usb_device_index = 0

# Test with PyAudio
pyaudio_success = test_pyaudio_output(usb_device_index)

# Test with aplay
# Common device names for USB audio on Raspberry Pi
device_names = [
    f"hw:4,0",
    f"plughw:4,0",
    "hw:2,0",
    "plughw:2,0",
    "default"
]

print("\nTrying different device names with aplay:")
aplay_success = False
for device in device_names:
    print(f"\nTesting device: {device}")
    if test_aplay_output(device):
        print(f"✓ Success with device: {device}")
        aplay_success = True
        break
    else:
        print(f"✗ Failed with device: {device}")
        time.sleep(1)  # Give the audio system some time to recover

# Summary
print("\n======================")
print("Test Results Summary:")
print("======================")
print(f"PyAudio test: {'SUCCESS' if pyaudio_success else 'FAILED'}")
print(f"aplay test: {'SUCCESS' if aplay_success else 'FAILED'}")

if not (pyaudio_success or aplay_success):
    print("\nNo audio playback methods succeeded. Check your audio configuration.")
    print("You might need to run 'sudo raspi-config' and ensure USB audio is properly configured.")
else:
    print("\nAt least one audio playback method worked!")
    
p.terminate() 