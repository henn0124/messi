#!/usr/bin/env python3
import pyaudio
import numpy as np
import time
import sys

print("Microphone Test Script")
print("=====================")

# Initialize PyAudio
p = pyaudio.PyAudio()

# List all audio devices
print("\nAvailable Audio Devices:")
print("-----------------------")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    name = dev['name']
    inputs = dev['maxInputChannels']
    outputs = dev['maxOutputChannels']
    print(f"Device {i}: {name} (Inputs: {inputs}, Outputs: {outputs})")

# Find TONOR TM20 microphone
tonor_index = None
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if "TONOR TM20" in dev['name'] and dev['maxInputChannels'] > 0:
        tonor_index = i
        print(f"\nFound TONOR TM20 microphone at index {i}")
        break

if tonor_index is None:
    print("\nTONOR TM20 not found. Testing default input device.")
    tonor_index = p.get_default_input_device_info()['index']
    print(f"Using default device at index {tonor_index}")

# Open stream for mic test
try:
    print("\nOpening stream for TONOR TM20 microphone...")
    stream = p.open(
        rate=16000,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        input_device_index=tonor_index,
        frames_per_buffer=1024
    )
    
    print("Stream opened successfully!")
    print("\nAudio Levels (Ctrl+C to exit):")
    print("-----------------------------")
    
    # Display audio levels for 10 seconds
    for _ in range(100):  # 10 seconds at 0.1s intervals
        # Read audio
        data = stream.read(1024)
        
        # Convert to numpy array and normalize
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Calculate level
        level = np.abs(audio_data).mean()
        
        # Create meter display
        meter_width = 50
        bars = int(level * meter_width)
        meter = "█" * bars + "▁" * (meter_width - bars)
        
        # Print level meter
        sys.stdout.write(f"\r{meter} [{level:.4f}]")
        sys.stdout.flush()
        
        time.sleep(0.1)
    
    # Close stream
    stream.stop_stream()
    stream.close()
    
except Exception as e:
    print(f"\nError testing microphone: {e}")

# Try alternate method if the first one fails
if 'stream' not in locals() or not stream.is_active():
    print("\n\nTrying alternate method with ALSA hw device...")
    try:
        # Try with ALSA hw device string
        cmd = f"arecord -D hw:3,0 -d 1 -f S16_LE -r 16000 -c 1 /dev/null"
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("ALSA direct access to microphone works!")
        else:
            print(f"ALSA error: {result.stderr}")
    except Exception as e:
        print(f"Error with ALSA test: {e}")

# Clean up
p.terminate()
print("\n\nMicrophone test complete.") 