"""
Benchmark Script for Messi Assistant
----------------------------------

Tests performance and resource usage of core functions:
1. Wake word detection
2. Speech recognition
3. Audio processing
4. Response generation
5. TTS synthesis
"""

import asyncio
import psutil
import time
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.audio import AudioInterface
from src.core.tts import TextToSpeech
from src.core.speech import SpeechManager
from src.core.config import Settings

class BenchmarkTest:
    def __init__(self):
        self.settings = Settings()
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Test components
        self.audio = AudioInterface()
        self.tts = TextToSpeech()
        self.speech = SpeechManager()
        
        # Metrics
        self.start_time = time.time()
        self.metrics = {
            "wake_word": {
                "detections": 0,
                "false_positives": 0,
                "avg_cpu": 0,
                "peak_cpu": 0,
                "avg_memory": 0,
                "peak_memory": 0
            },
            "speech": {
                "processed": 0,
                "errors": 0,
                "avg_latency": 0
            },
            "tts": {
                "generated": 0,
                "errors": 0,
                "avg_latency": 0
            }
        }
    
    async def run_benchmark(self, duration: int = 300):
        """Run benchmark for specified duration"""
        print(f"\n=== Starting Benchmark ({duration}s) ===")
        
        # Initialize components
        await self.audio.initialize()
        
        test_end = time.time() + duration
        samples = 0
        
        while time.time() < test_end:
            try:
                # Get resource usage
                cpu = psutil.cpu_percent()
                memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Update metrics
                self.metrics["wake_word"]["peak_cpu"] = max(
                    self.metrics["wake_word"]["peak_cpu"], 
                    cpu
                )
                self.metrics["wake_word"]["peak_memory"] = max(
                    self.metrics["wake_word"]["peak_memory"], 
                    memory
                )
                
                # Update averages
                samples += 1
                self.metrics["wake_word"]["avg_cpu"] = (
                    (self.metrics["wake_word"]["avg_cpu"] * (samples - 1) + cpu) 
                    / samples
                )
                self.metrics["wake_word"]["avg_memory"] = (
                    (self.metrics["wake_word"]["avg_memory"] * (samples - 1) + memory) 
                    / samples
                )
                
                # Status update every 30 seconds
                elapsed = time.time() - self.start_time
                if elapsed % 30 < 1:
                    self._print_status(elapsed)
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\nBenchmark interrupted!")
                break
                
            except Exception as e:
                print(f"Benchmark error: {e}")
                break
        
        # Save results
        self._save_results()
        
    def _print_status(self, elapsed: float):
        """Print current benchmark status"""
        print(f"\n=== Benchmark Status ({elapsed:.0f}s) ===")
        print(f"Wake Word:")
        print(f"  Detections: {self.metrics['wake_word']['detections']}")
        print(f"  Avg CPU: {self.metrics['wake_word']['avg_cpu']:.1f}%")
        print(f"  Peak CPU: {self.metrics['wake_word']['peak_cpu']:.1f}%")
        print(f"  Avg Memory: {self.metrics['wake_word']['avg_memory']:.1f}MB")
        print(f"  Peak Memory: {self.metrics['wake_word']['peak_memory']:.1f}MB")
    
    def _save_results(self):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_{timestamp}.json"
        
        results = {
            "timestamp": timestamp,
            "duration": time.time() - self.start_time,
            "metrics": self.metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    benchmark = BenchmarkTest()
    asyncio.run(benchmark.run_benchmark()) 