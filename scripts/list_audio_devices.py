import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    
    print("\nAvailable Audio Devices:")
    print("------------------------")
    
    # Get device count
    device_count = p.get_device_count()
    
    # List all devices with more detail
    for i in range(device_count):
        try:
            device_info = p.get_device_info_by_index(i)
            print(f"\nDevice {i}:")
            print(f"    Name: {device_info['name']}")
            print(f"    Max Input Channels: {device_info['maxInputChannels']}")
            print(f"    Max Output Channels: {device_info['maxOutputChannels']}")
            print(f"    Default Sample Rate: {device_info['defaultSampleRate']}")
            print(f"    Host API: {p.get_host_api_info_by_index(device_info['hostApi'])['name']}")
            
            # Check if this is a default device
            if i == p.get_default_input_device_info()['index']:
                print("    *** Default Input Device ***")
            if i == p.get_default_output_device_info()['index']:
                print("    *** Default Output Device ***")
                
        except Exception as e:
            print(f"Error getting info for device {i}: {e}")
            
    p.terminate()

if __name__ == "__main__":
    list_audio_devices() 