import pyaudio
import wave
import json
from vosk import Model, KaldiRecognizer
import sys
import os
import time
import numpy as np
from datetime import datetime
import asyncio
import openai
from pydantic_settings import BaseSettings
import concurrent.futures
from typing import List, Dict
import csv
from concurrent.futures import ThreadPoolExecutor
import aiofiles

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    
    class Config:
        env_file = ".env"
        extra = "allow"

class LatencyBenchmark:
    def __init__(self):
        self.measurements = {
            'vosk': [],
            'whisper': []
        }
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_measurement(self, service: str, latency: float, text: str, success: bool):
        self.measurements[service].append({
            'timestamp': time.time(),
            'latency': latency,
            'text': text,
            'success': success
        })
        
    def save_results(self):
        """Save benchmark results to CSV"""
        filename = f"benchmark_results_{self.current_session}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Service', 'Timestamp', 'Latency', 'Text', 'Success'])
            for service, measurements in self.measurements.items():
                for m in measurements:
                    writer.writerow([
                        service,
                        m['timestamp'],
                        m['latency'],
                        m['text'],
                        m['success']
                    ])
        print(f"\nResults saved to {filename}")
    
    def print_summary(self):
        """Print statistical summary of benchmarks"""
        print("\n=== Benchmark Summary ===")
        for service in ['vosk', 'whisper']:
            latencies = [m['latency'] for m in self.measurements[service]]
            if latencies:
                print(f"\n{service.upper()} Statistics:")
                print(f"Samples: {len(latencies)}")
                print(f"Mean latency: {np.mean(latencies):.3f}s")
                print(f"Median latency: {np.median(latencies):.3f}s")
                print(f"Min latency: {np.min(latencies):.3f}s")
                print(f"Max latency: {np.max(latencies):.3f}s")
                print(f"Std dev: {np.std(latencies):.3f}s")
                
                success_rate = sum(1 for m in self.measurements[service] if m['success']) / len(self.measurements[service])
                print(f"Success rate: {success_rate*100:.1f}%")

class VoskTest:
    def __init__(self):
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paInt16
        self.model_path = "models/vosk"
        self.benchmark = LatencyBenchmark()
        self._model_instance = None  # Initialize model reference
        
        # Speech detection parameters
        self.silence_threshold = 200
        self.min_speech_duration = 0.2
        self.speech_pad_duration = 0.3
        self.speech_energy_threshold = 1000

    def setup_vosk(self):
        """Initialize Vosk model with proper error handling"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Error: Model path {self.model_path} does not exist")
                return None
            
            if self._model_instance is None:
                print("Loading Vosk model...")
                load_start = time.time()
                self._model_instance = Model(self.model_path)
                load_time = time.time() - load_start
                print(f"Vosk model loaded successfully in {load_time:.2f}s")
            
            return self._model_instance
            
        except Exception as e:
            print(f"Error initializing Vosk model: {e}")
            return None

    async def test_microphone(self, duration=30):
        """Test real-time speech recognition with latency measurements"""
        try:
            model = self.setup_vosk()
            rec = KaldiRecognizer(model, self.RATE)
            
            p = pyaudio.PyAudio()
            print("\nAvailable audio devices:")
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                print(f"Index {i}: {dev['name']}")
                print(f"  Max Input Channels: {dev['maxInputChannels']}")
                print(f"  Max Output Channels: {dev['maxOutputChannels']}")
                print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
            
            input_device_info = p.get_device_info_by_index(1)
            device_rate = int(input_device_info['defaultSampleRate'])
            print(f"\nUsing device: {input_device_info['name']}")
            print(f"Device sample rate: {device_rate}")
            
            stream = p.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=device_rate,
                          input=True,
                          input_device_index=1,
                          frames_per_buffer=self.CHUNK)
            
            print(f"\nListening for {duration} seconds... Speak into the microphone")
            print("Recording latency metrics...")
            
            start_time = time.time()
            processing_start = None
            audio_buffer = []
            
            while time.time() - start_time < duration:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                if device_rate != 16000:
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    ratio = 16000 / device_rate
                    target_length = int(len(audio_array) * ratio)
                    indices = np.linspace(0, len(audio_array)-1, target_length).astype(int)
                    resampled = audio_array[indices]
                    data = resampled.tobytes()
                
                if not processing_start:
                    processing_start = time.time()
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result["text"]:
                        processing_time = time.time() - processing_start
                        print(f"\nRecognized: {result['text']}")
                        print(f"Processing time: {processing_time:.3f}s")
                        
                        self.benchmark.add_measurement(
                            'vosk',
                            processing_time,
                            result['text'],
                            bool(result['text'])
                        )
                        
                        processing_start = None
                
                await asyncio.sleep(0.001)
            
            # Get final result
            final_result = json.loads(rec.FinalResult())
            if final_result["text"]:
                print(f"\nFinal recognition: {final_result['text']}")
            
            return True
            
        except Exception as e:
            print(f"Error in microphone test: {e}")
            return False
            
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()
            
            self.benchmark.print_summary()
            self.benchmark.save_results()

    def test_wav_file(self, wav_path):
        """Test speech recognition from WAV file with latency measurements"""
        model = self.setup_vosk()
        
        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            sys.exit(1)
            
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        
        print(f"\nProcessing file: {wav_path}")
        
        start_time = time.time()
        total_audio_length = wf.getnframes() / wf.getframerate()
        
        while True:
            data = wf.readframes(self.CHUNK)
            if len(data) == 0:
                break
            
            process_start = time.time()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result["text"]:
                    process_time = time.time() - process_start
                    print(f"\nRecognized: {result['text']}")
                    print(f"Processing time: {process_time:.3f}s")
                    
                    self.benchmark.add_measurement(
                        'vosk',
                        process_time,
                        result['text'],
                        bool(result['text'])
                    )
        
        # Get final bits of audio
        result = json.loads(rec.FinalResult())
        if result["text"]:
            total_time = time.time() - start_time
            print(f"\nFinal recognition: {result['text']}")
            print(f"Total processing time: {total_time:.3f}s")
            print(f"Real-time factor: {total_time/total_audio_length:.2f}x")
            
            self.benchmark.add_measurement(
                'vosk',
                total_time,
                result['text'],
                bool(result['text'])
            )
        
        self.benchmark.print_summary()
        self.benchmark.save_results()

    async def test_vosk_wav(self, audio_data: bytes) -> dict:
        """Test Vosk with improved error handling"""
        try:
            # Initialize model
            model = self.setup_vosk()
            if model is None:
                print("Failed to initialize Vosk model")
                return {'text': "", 'latency': 0, 'success': False}
            
            # Create recognizer
            rec = KaldiRecognizer(model, 16000)
            if rec is None:
                print("Failed to create KaldiRecognizer")
                return {'text': "", 'latency': 0, 'success': False}
            
            # Enable better speech detection
            rec.SetWords(True)
            
            # Convert audio to numpy array for processing
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Resample to 16kHz if needed
            if self.RATE != 16000:
                print(f"Resampling from {self.RATE} to 16000 Hz")
                samples = len(audio_array)
                new_samples = int(samples * 16000 / self.RATE)
                audio_array = np.interp(
                    np.linspace(0, samples, new_samples, endpoint=False),
                    np.arange(samples),
                    audio_array
                ).astype(np.int16)
            
            # Normalize audio
            audio_normalized = audio_array / np.max(np.abs(audio_array))
            processed_audio = (audio_normalized * 32767).astype(np.int16)
            
            # Process in chunks
            chunk_size = 8192
            start_time = time.time()
            
            for i in range(0, len(processed_audio), chunk_size):
                chunk = processed_audio[i:i + chunk_size]
                
                # Skip silent chunks
                if np.abs(chunk).mean() < self.silence_threshold:
                    continue
                
                if rec.AcceptWaveform(chunk.tobytes()):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        latency = time.time() - start_time
                        return {
                            'text': result["text"],
                            'latency': latency,
                            'success': True
                        }
            
            # Get final result
            final_result = json.loads(rec.FinalResult())
            latency = time.time() - start_time
            
            return {
                'text': final_result.get("text", ""),
                'latency': latency,
                'success': bool(final_result.get("text"))
            }
            
        except Exception as e:
            print(f"Error in Vosk WAV test: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            return {'text': "", 'latency': 0, 'success': False}

    def _has_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if chunk contains speech"""
        # Calculate energy
        energy = np.sum(np.abs(audio_chunk)) / len(audio_chunk)
        
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_chunk)))) / 2
        zero_crossing_rate = zero_crossings / len(audio_chunk)
        
        # Check if chunk likely contains speech
        return (energy > self.speech_energy_threshold and 
                zero_crossing_rate > 0.01 and 
                zero_crossing_rate < 0.2)

    def _check_result_confidence(self, result: dict) -> bool:
        """Check if result seems reliable"""
        if not result.get("text"):
            return False
            
        # Check word count (very short results are often errors)
        words = result["text"].split()
        if len(words) < 2:
            return False
            
        # Check if result contains common error indicators
        error_indicators = ["um", "uh", "eh", "ah"]
        if any(word in error_indicators for word in words):
            return False
            
        return True

class LatencyComparison:
    def __init__(self):
        self.settings = Settings()
        self.client = openai.AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.benchmark = LatencyBenchmark()
        self.vosk_test = VoskTest()
        self.silence_threshold = 500  # Adjust based on testing
        self.silence_duration = 0.5   # Half second of silence to mark end of speech
        self.executor = ThreadPoolExecutor(max_workers=2)  # For parallel processing

    async def detect_speech_end(self, audio_data: bytes) -> bool:
        """Detect if audio segment is silence"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_level = np.abs(audio_array).mean()
        return audio_level < self.silence_threshold

    async def _run_whisper_test(self, duration):
        """Helper method for Whisper testing with true latency measurement"""
        try:
            p = pyaudio.PyAudio()
            input_device_info = p.get_device_info_by_index(1)
            device_rate = int(input_device_info['defaultSampleRate'])
            
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=device_rate,
                          input=True,
                          input_device_index=1,
                          frames_per_buffer=4096)
            
            print("\nWhisper Test: Speak a phrase and pause for response...")
            print("(Recording will automatically stop after silence is detected)")
            
            audio_chunks = []
            is_speaking = False
            speech_start = None
            silence_start = None
            
            start_time = time.time()
            while time.time() - start_time < duration:
                chunk = stream.read(4096, exception_on_overflow=False)
                
                # Detect speech/silence
                is_silence = await self.detect_speech_end(chunk)
                
                if not is_speaking and not is_silence:
                    # Speech started
                    is_speaking = True
                    speech_start = time.time()
                    silence_start = None
                    audio_chunks = [chunk]
                
                elif is_speaking:
                    audio_chunks.append(chunk)
                    
                    if is_silence:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= self.silence_duration:
                            # Speech ended, process the audio
                            speech_end = time.time()
                            print("\nProcessing speech...")
                            
                            # Resample to 16kHz
                            audio_data = b''.join(audio_chunks)
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            if device_rate != 16000:
                                ratio = 16000 / device_rate
                                target_length = int(len(audio_array) * ratio)
                                indices = np.linspace(0, len(audio_array)-1, target_length).astype(int)
                                resampled = audio_array[indices]
                                audio_data = resampled.tobytes()
                            
                            # Process with Whisper
                            result = await self.test_whisper(audio_data, speech_end)
                            if result['success']:
                                true_latency = result['latency']
                                print(f"Whisper recognized: {result['text']}")
                                print(f"True latency (speech end to response): {true_latency:.3f}s")
                            
                            # Reset for next phrase
                            is_speaking = False
                            speech_start = None
                            silence_start = None
                            audio_chunks = []
                    else:
                        silence_start = None
                
                await asyncio.sleep(0.001)
            
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()

    async def test_whisper(self, audio_data: bytes, speech_end_time: float) -> dict:
        """Test Whisper API with true latency measurement"""
        try:
            temp_file = "temp_audio.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            
            with open(temp_file, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Calculate true latency from end of speech
            true_latency = time.time() - speech_end_time
            
            self.benchmark.add_measurement(
                'whisper',
                true_latency,
                response,
                bool(response)
            )
            
            return {
                'text': response,
                'latency': true_latency,
                'success': bool(response)
            }
            
        except Exception as e:
            print(f"Whisper API error: {e}")
            return {'text': "", 'latency': 0, 'success': False}
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Similar modifications needed for Vosk test
    async def run_vosk_test(self, duration=30):
        """Run Vosk test with true latency measurement"""
        model = self.vosk_test.setup_vosk()
        rec = KaldiRecognizer(model, 16000)
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=44100,  # Native rate
                       input=True,
                       input_device_index=1,
                       frames_per_buffer=4096)
        
        print("\nVosk Test: Speak a phrase and pause for response...")
        
        is_speaking = False
        speech_start = None
        silence_start = None
        audio_chunks = []
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                chunk = stream.read(4096, exception_on_overflow=False)
                is_silence = await self.detect_speech_end(chunk)
                
                if not is_speaking and not is_silence:
                    is_speaking = True
                    speech_start = time.time()
                    silence_start = None
                    audio_chunks = [chunk]
                
                elif is_speaking:
                    audio_chunks.append(chunk)
                    
                    if is_silence:
                        if silence_start is None:
                            silence_start = time.time()
                            speech_end = time.time()
                        elif time.time() - silence_start >= self.silence_duration:
                            # Process with Vosk
                            audio_data = b''.join(audio_chunks)
                            if rec.AcceptWaveform(audio_data):
                                result = json.loads(rec.Result())
                                if result["text"]:
                                    true_latency = time.time() - speech_end
                                    print(f"\nVosk recognized: {result['text']}")
                                    print(f"True latency: {true_latency:.3f}s")
                                    
                                    self.benchmark.add_measurement(
                                        'vosk',
                                        true_latency,
                                        result['text'],
                                        True
                                    )
                            
                            # Reset for next phrase
                            is_speaking = False
                            speech_start = None
                            silence_start = None
                            audio_chunks = []
                    else:
                        silence_start = None
                
                await asyncio.sleep(0.001)
                
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def run_parallel_test(self, duration=30):
        """Run Vosk and Whisper tests in parallel"""
        print("\n=== Running Parallel Latency Test ===")
        
        try:
            # Setup audio capture
            p = pyaudio.PyAudio()
            
            # Get device info
            print("\nAvailable audio devices:")
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                print(f"Index {i}: {dev['name']}")
                print(f"  Max Input Channels: {dev['maxInputChannels']}")
                print(f"  Max Output Channels: {dev['maxOutputChannels']}")
                print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
            
            # Get Maono Elf device info
            input_device_info = p.get_device_info_by_index(1)  # Maono Elf index
            device_rate = int(input_device_info['defaultSampleRate'])
            print(f"\nUsing device: {input_device_info['name']}")
            print(f"Device sample rate: {device_rate}")
            
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=device_rate,  # Use device's native rate
                          input=True,
                          input_device_index=1,
                          frames_per_buffer=4096)
            
            print(f"Recording for {duration} seconds...")
            
            audio_chunks = []
            start_time = time.time()
            
            # Initialize Vosk
            model = self.vosk_test.setup_vosk()
            rec = KaldiRecognizer(model, 16000)  # Vosk expects 16kHz
            
            while time.time() - start_time < duration:
                chunk = stream.read(4096, exception_on_overflow=False)
                
                # Resample for Vosk (16kHz)
                if device_rate != 16000:
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    ratio = 16000 / device_rate
                    target_length = int(len(audio_array) * ratio)
                    indices = np.linspace(0, len(audio_array)-1, target_length).astype(int)
                    resampled = audio_array[indices]
                    vosk_chunk = resampled.tobytes()
                else:
                    vosk_chunk = chunk
                
                # Process with Vosk
                if rec.AcceptWaveform(vosk_chunk):
                    result = json.loads(rec.Result())
                    if result["text"]:
                        print(f"\nVosk recognized: {result['text']}")
                        self.benchmark.add_measurement(
                            'vosk',
                            time.time() - start_time,
                            result['text'],
                            True
                        )
                
                # Collect audio for Whisper
                audio_chunks.append(chunk)
                if len(audio_chunks) >= 10:  # Process every ~2.5s
                    # Process with Whisper in parallel
                    audio_data = b''.join(audio_chunks)
                    asyncio.create_task(self.test_whisper(audio_data))
                    audio_chunks = []
                
                await asyncio.sleep(0.001)
            
            # Process any remaining audio
            if audio_chunks:
                audio_data = b''.join(audio_chunks)
                await self.test_whisper(audio_data)
            
        except Exception as e:
            print(f"Error in parallel test: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()
            
            self.benchmark.print_summary()
            self.benchmark.save_results()
    
    async def run_serial_test(self, duration=30):
        """Run Vosk and Whisper tests serially"""
        print("\n=== Running Serial Latency Test ===")
        
        try:
            # First run Vosk test
            print("\nRunning Vosk test...")
            vosk_success = await self.vosk_test.test_microphone(duration=duration)
            
            if vosk_success:
                # Then run Whisper test
                print("\nRunning Whisper test...")
                await self._run_whisper_test(duration)
            
        except Exception as e:
            print(f"Error in serial test: {e}")
        finally:
            self.benchmark.print_summary()
            self.benchmark.save_results()

    async def run_wav_benchmark(self, wav_path="/home/pi/messi/cache/tts/last_response.wav"):
        """Run optimized benchmark tests with chunked processing"""
        print("\n=== WAV File Benchmark Test ===")
        print(f"Testing file: {wav_path}")
        
        try:
            # Get file info without loading entire file
            with wave.open(wav_path, 'rb') as wf:
                print(f"\nWAV File Properties:")
                print(f"Channels: {wf.getnchannels()}")
                print(f"Sample width: {wf.getsampwidth()}")
                print(f"Frame rate: {wf.getframerate()}")
                
                file_size = os.path.getsize(wav_path)
                print(f"File size: {file_size/1024/1024:.2f} MB")
                
                # Calculate actual frames from file size
                bytes_per_frame = wf.getnchannels() * wf.getsampwidth()
                actual_frames = (file_size - 44) // bytes_per_frame  # 44 bytes for WAV header
                duration = actual_frames / float(wf.getframerate())
                print(f"Duration: {duration:.2f} seconds")
                
                # Read in smaller chunks
                chunk_size = 1024 * 1024  # 1MB chunks
                audio_chunks = []
                total_size = 0
                
                print("\nReading audio in chunks...")
                while True:
                    chunk = wf.readframes(chunk_size)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    audio_chunks.append(chunk)
                    if total_size >= 5 * 1024 * 1024:  # Limit to 5MB
                        break
                
                audio_data = b''.join(audio_chunks)
                print(f"Successfully read {len(audio_data)/1024/1024:.2f} MB of audio data")
            
            # Run tests in parallel
            print("\nRunning parallel tests...")
            vosk_task = asyncio.create_task(self.vosk_test.test_vosk_wav(audio_data))
            whisper_task = asyncio.create_task(self.test_whisper_wav(audio_data))
            
            # Wait for both tests to complete
            vosk_result, whisper_result = await asyncio.gather(vosk_task, whisper_task)
            
            # Print results
            print("\n=== Results ===")
            print("\nVosk Results:")
            print(f"Text: {vosk_result['text']}")
            print(f"Latency: {vosk_result['latency']:.3f}s")
            print(f"Success: {vosk_result['success']}")
            
            print("\nWhisper Results:")
            print(f"Text: {whisper_result['text']}")
            print(f"Latency: {whisper_result['latency']:.3f}s")
            print(f"Success: {whisper_result['success']}")
            
            # Save results
            self.benchmark.save_results()
            
        except MemoryError:
            print("Memory error: File too large to process")
            print("Try using a smaller audio file for testing")
        except Exception as e:
            print(f"Error in WAV benchmark: {e}")
            print(f"Error type: {type(e)}")
            if hasattr(e, 'args'):
                print(f"Error details: {e.args}")

    async def test_whisper_wav(self, audio_data: bytes) -> dict:
        """Test Whisper with chunked processing"""
        start_time = time.time()
        try:
            # Create temporary file with optimized settings
            temp_file = "temp_benchmark.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                
                # Process audio in smaller chunks
                chunk_size = 1024 * 1024  # 1MB chunks
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    wf.writeframes(chunk)
            
            # Send to Whisper API
            async with aiofiles.open(temp_file, 'rb') as audio_file:
                audio_data = await audio_file.read()
                
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", audio_data),
                    response_format="text",
                    temperature=0.0,
                    language="en"
                )
            
            latency = time.time() - start_time
            success = bool(response)
            
            return {
                'text': response if success else "",
                'latency': latency,
                'success': success
            }
            
        except Exception as e:
            print(f"Whisper API error: {e}")
            return {'text': "", 'latency': 0, 'success': False}
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def main():
    comparison = LatencyComparison()
    
    print("STT Latency Comparison")
    print("1. Run parallel test")
    print("2. Run serial test")
    print("3. Run WAV file benchmark")
    choice = input("Choose test (1/2/3): ")
    
    if choice == "1":
        asyncio.run(comparison.run_parallel_test())
    elif choice == "2":
        asyncio.run(comparison.run_serial_test())
    elif choice == "3":
        asyncio.run(comparison.run_wav_benchmark())
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 