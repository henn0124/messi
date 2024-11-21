#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    redis-server \
    git

# Install Porcupine dependencies
sudo apt-get install -y python3-dev

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Setup systemd service
sudo cp deployment/smart-speaker.service /etc/systemd/system/
sudo systemctl enable smart-speaker
sudo systemctl start smart-speaker 