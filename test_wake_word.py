#!/usr/bin/env python3
import os
import struct
import pyaudio
import numpy as np
import sys
import time
import pvporcupine
from dotenv import load_dotenv

# Configuration
INPUT_DEVICE_INDEX = 1  # TONOR TM20 at index 1
DISPLAY_AUDIO_LEVEL = True

def get_audio_level_meter(level, width=40):
    """Creates a visual audio level meter"""
    normalized = min(1.0, level)
    meter_width = int(normalized * width)
    return "█" * meter_width + "▁" * (width - meter_width) + f" [{level:.2f}]"

def main():
    # Load environment variables
    print("Wake Word Detection Test")
    print("=======================\n")
    
    load_dotenv()
    porcupine_key = os.getenv('PORCUPINE_API_KEY')
    
    if not porcupine_key:
        print("Error: PORCUPINE_API_KEY not found in .env file")
        return
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # List audio devices
    print("Available Audio Devices:")
    print("-----------------------")
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        name = dev['name']
        inputs = dev['maxInputChannels']
        print(f"Device {i}: {name} (Input channels: {inputs})")
    
    # Initialize Porcupine
    print("\nInitializing wake word detection...")
    model_path = os.path.join("models", "hey-messy_en_raspberry-pi_v3_0_0.ppn")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "hey-messy_en_raspberry-pi_v3_0_0.ppn")
    
    if not os.path.exists(model_path):
        print(f"Error: Wake word model not found at {model_path}")
        return
    
    print(f"Using wake word model: {model_path}")
    
    porcupine = pvporcupine.create(
        access_key=porcupine_key,
        keyword_paths=[model_path]
    )
    
    # Open audio stream
    print(f"Opening audio stream with device index {INPUT_DEVICE_INDEX}...")
    try:
        stream = audio.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=INPUT_DEVICE_INDEX,
            frames_per_buffer=porcupine.frame_length
        )
        
        print("Successfully opened microphone stream!")
        print("\nListening for 'Hey Messy' wake word...")
        print("Press Ctrl+C to exit\n")
        
        if DISPLAY_AUDIO_LEVEL:
            print("Audio levels:")
            print("-------------")
        
        # Main detection loop
        detections = 0
        overflow_count = 0
        while True:
            try:
                # Read audio frame with overflow handling
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
                
                # Calculate and display audio level
                if DISPLAY_AUDIO_LEVEL:
                    audio_data = np.array(pcm_unpacked, dtype=np.float32) / 32768.0
                    audio_level = np.abs(audio_data).mean()
                    sys.stdout.write("\r" + get_audio_level_meter(audio_level))
                    sys.stdout.flush()
                
                # Process with Porcupine
                keyword_index = porcupine.process(pcm_unpacked)
                
                # If wake word detected
                if keyword_index >= 0:
                    detections += 1
                    detection_time = time.strftime("%H:%M:%S")
                    print(f"\n🎤 Wake word detected at {detection_time}! (#{detections})")
                    time.sleep(1)  # Brief pause after detection
                    
                    if DISPLAY_AUDIO_LEVEL:
                        print("\nAudio levels:")
                        print("-------------")
                
            except OSError as e:
                if "Input overflowed" in str(e):
                    overflow_count += 1
                    print(f"\nInput overflow detected ({overflow_count})")
                    time.sleep(0.1)
                else:
                    print(f"\nError reading from microphone: {e}")
                    break
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
    
    except Exception as e:
        print(f"Error initializing audio stream: {e}")
    
    finally:
        # Clean up
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        porcupine.delete()
        audio.terminate()
        
        print("\nWake word test completed.")
        print(f"Total detections: {detections}")

if __name__ == "__main__":
    main() 