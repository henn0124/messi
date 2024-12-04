import pyaudio
import numpy as np
import time
import wave
import subprocess
import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_alsa_devices():
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error getting ALSA devices: {e}")
        return ""

def find_device_by_name(p, target_name):
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if target_name.lower() in dev_info['name'].lower():
            return i, dev_info
    return None, None

def test_microphone():
    # Load configuration
    config = load_config()
    audio_config = config['audio']
    input_config = audio_config['input']
    
    print("\nALSA Devices:")
    print(get_alsa_devices())
    
    # Audio settings
    CHUNK = input_config['chunk_size']
    FORMAT = pyaudio.paInt16
    CHANNELS = input_config['channels']
    RATE = 48000  # Fixed at 48kHz
    RECORD_SECONDS = 5  # Test duration
    WAVE_OUTPUT_FILENAME = "test_recording.wav"
    BUFFER_SIZE = input_config['buffer_size']
    SILENCE_THRESHOLD = input_config['silence_threshold']
    DEVICE_INDEX = 2  # TONOR TM20 Audio Device

    print("\nUsing audio configuration:")
    print(f"Device Index: {DEVICE_INDEX}")
    print(f"Sample Rate: {RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Chunk Size: {CHUNK}")
    print(f"Buffer Size: {BUFFER_SIZE}")
    print(f"Silence Threshold: {SILENCE_THRESHOLD}\n")

    p = pyaudio.PyAudio()

    # List all audio devices
    print("PyAudio Devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        print(f"Device {i}: {dev_info['name']}")
        print(f"  Max Input Channels: {dev_info['maxInputChannels']}")
        print(f"  Max Output Channels: {dev_info['maxOutputChannels']}")
        print(f"  Default Sample Rate: {dev_info['defaultSampleRate']}")
        if i == DEVICE_INDEX:
            print("  *** Selected Device ***")
        print()

    try:
        # Open stream using TONOR device
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK,
                       input_device_index=DEVICE_INDEX,
                       )

        print("* Recording for 5 seconds...")
        print("* Speak into the microphone to test...")
        print("\nVolume meter:")

        frames = []
        volumes = []
        last_line_length = 0

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                # Calculate volume
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume_norm = np.linalg.norm(audio_data) * 10
                volumes.append(volume_norm)
                # Print volume meter (clear previous line first)
                meter = "#" * min(40, int(volume_norm/100))
                output = f"\rVolume: [{meter:<40}] {volume_norm:>6.0f}"
                print(output + " " * (last_line_length - len(output)), end="", flush=True)
                last_line_length = len(output)
            except Exception as e:
                print(f"\nError reading audio chunk: {e}")
                break

        print("\n\n* Done recording")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        if frames:
            # Save the recorded data as a WAV file
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Print statistics
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            max_volume = max(volumes) if volumes else 0
            print(f"\nAudio Statistics:")
            print(f"  Average volume: {avg_volume:.2f}")
            print(f"  Maximum volume: {max_volume:.2f}")
            print(f"  Samples below threshold: {sum(1 for v in volumes if v < SILENCE_THRESHOLD)}")
            print(f"  Total samples: {len(volumes)}")
            print(f"\nRecording saved as {WAVE_OUTPUT_FILENAME}")
        else:
            print("No audio data was recorded")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    test_microphone()

class TestAudio:
    def setup_method(self):
        self.settings = Settings()
        self.audio = AudioInterface()
  