from typing import Dict
import asyncio
import time

class Timer:
    async def handle(self, intent: Dict) -> Dict:
        """Handle timer requests"""
        try:
            duration = intent.get("parameters", {}).get("duration", 0)
            unit = intent.get("parameters", {}).get("unit", "seconds")
            
            if not duration:
                return {
                    "text": "How long would you like the timer for?",
                    "context": "timer_request",
                    "waiting_for_input": True
                }
            
            # Convert to seconds
            seconds = self._convert_to_seconds(duration, unit)
            
            # Start timer in background
            asyncio.create_task(self._run_timer(seconds))
            
            return {
                "text": f"Timer set for {duration} {unit}",
                "context": "timer_set"
            }
            
        except Exception as e:
            print(f"Error in timer: {e}")
            return {"text": "I couldn't set that timer", "context": "error"}

    async def _run_timer(self, seconds: int):
        await asyncio.sleep(seconds)
        # Timer complete notification would go here
        print(f"Timer complete after {seconds} seconds")

    def _convert_to_seconds(self, duration: int, unit: str) -> int:
        conversions = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600
        }
        return duration * conversions.get(unit, 1)

skill_manifest = {
    "name": "timer",
    "intents": ["set_timer", "check_timer", "cancel_timer"],
    "description": "Timer management for activities"
} 