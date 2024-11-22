import os
import wave
import time
import asyncio
import numpy as np
from typing import Dict, List
import aiofiles
import openai
from pydantic_settings import BaseSettings
import csv
from datetime import datetime

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    
    class Config:
        env_file = ".env"
        extra = "allow"

class WhisperLite:
    def __init__(self):
        self.settings = Settings()
        self.client = openai.AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.chunk_size = 32 * 1024  # 32KB chunks
        self.max_audio_size = 256 * 1024  # 256KB max audio size
        self.sample_rate = 16000
        self.results_file = f"whisper_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    async def process_audio_file(self, file_path: str) -> Dict:
        """Process audio file with minimal memory usage"""
        try:
            print(f"\nProcessing file: {file_path}")
            
            # Get file info without loading entire file
            with wave.open(file_path, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                file_size = os.path.getsize(file_path)
                
                print(f"File properties:")
                print(f"Channels: {channels}")
                print(f"Sample width: {sample_width}")
                print(f"Frame rate: {frame_rate}")
                print(f"File size: {file_size/1024:.1f}KB")

                # Process in small chunks
                chunks = []
                total_size = 0
                
                while True:
                    chunk = wf.readframes(self.chunk_size)
                    if not chunk:
                        break
                    
                    # Convert to mono and resample if needed
                    audio_chunk = np.frombuffer(chunk, dtype=np.int16)
                    if channels == 2:
                        audio_chunk = audio_chunk[::2]  # Take only left channel
                    
                    if frame_rate != self.sample_rate:
                        # Simple resampling
                        samples = len(audio_chunk)
                        new_samples = int(samples * self.sample_rate / frame_rate)
                        resampled = np.interp(
                            np.linspace(0, samples, new_samples, endpoint=False),
                            np.arange(samples),
                            audio_chunk
                        ).astype(np.int16)
                        chunks.append(resampled)
                    else:
                        chunks.append(audio_chunk)
                    
                    total_size += len(chunk)
                    if total_size >= self.max_audio_size:
                        print(f"Reached size limit of {self.max_audio_size/1024:.1f}KB")
                        break

                # Combine chunks and normalize
                audio_data = np.concatenate(chunks)
                audio_data = audio_data / np.max(np.abs(audio_data))
                audio_data = (audio_data * 32767).astype(np.int16)

            # Save processed audio to temporary file
            temp_file = "temp_whisper.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())

            # Process with Whisper
            start_time = time.time()
            async with aiofiles.open(temp_file, 'rb') as audio_file:
                audio_bytes = await audio_file.read()
                
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", audio_bytes),
                    response_format="text",
                    temperature=0.0,
                    language="en"
                )

            latency = time.time() - start_time
            
            # Save results
            result = {
                'text': response,
                'latency': latency,
                'file_size': total_size/1024,
                'success': True
            }
            
            await self.save_result(result)
            return result

        except Exception as e:
            print(f"Error processing audio: {e}")
            return {'text': "", 'latency': 0, 'file_size': 0, 'success': False}
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    async def save_result(self, result: Dict):
        """Save results with minimal memory usage"""
        try:
            with open(self.results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:  # File is empty, write header
                    writer.writerow(['Timestamp', 'Text', 'Latency', 'File Size (KB)', 'Success'])
                writer.writerow([
                    datetime.now().isoformat(),
                    result['text'],
                    f"{result['latency']:.3f}",
                    f"{result['file_size']:.1f}",
                    result['success']
                ])
        except Exception as e:
            print(f"Error saving results: {e}")

async def main():
    processor = WhisperLite()
    test_file = "/home/pi/messi/cache/tts/last_response.wav"
    
    print("Whisper Lite Test")
    print("Memory-optimized for Raspberry Pi")
    print(f"Max audio size: {processor.max_audio_size/1024:.1f}KB")
    print(f"Chunk size: {processor.chunk_size/1024:.1f}KB")
    
    result = await processor.process_audio_file(test_file)
    
    print("\nResults:")
    print(f"Text: {result['text']}")
    print(f"Latency: {result['latency']:.3f}s")
    print(f"File size: {result['file_size']:.1f}KB")
    print(f"Success: {result['success']}")
    print(f"\nResults saved to: {processor.results_file}")

if __name__ == "__main__":
    asyncio.run(main()) 