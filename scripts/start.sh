#!/bin/bash

# Suppress ALSA warnings
export ALSA_CARD=1
export ALSA_PCM_CARD=1
export ALSA_CTL_CARD=1

# Additional ALSA configuration
export AUDIODEV=hw:1,0
export AUDIODRIVER=alsa

# Activate virtual environment
source .venv/bin/activate

# Set Python path
export PYTHONPATH="/home/pi/messi:$PYTHONPATH"

# Run the main application with ALSA warning suppression
python src/main.py 2>/dev/null  # Redirect stderr to suppress ALSA warnings 