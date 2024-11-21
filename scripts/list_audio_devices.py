import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    
    print("\nAvailable Audio Devices:")
    print("------------------------")
    
    # Get device count
    device_count = p.get_device_count()
    
    # List all devices
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        print(f"\nDevice {i}:")
        print(f"    Name: {device_info['name']}")
        print(f"    Max Input Channels: {device_info['maxInputChannels']}")
        print(f"    Max Output Channels: {device_info['maxOutputChannels']}")
        print(f"    Default Sample Rate: {device_info['defaultSampleRate']}")
        
    p.terminate()

if __name__ == "__main__":
    list_audio_devices() 