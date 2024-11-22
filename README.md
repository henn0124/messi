# Messi - Interactive AI Storytelling Assistant

An AI-powered storytelling assistant designed for children, running on Raspberry Pi with LED feedback and natural conversation flow.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Troubleshooting](#troubleshooting)

## Features

### Voice Interaction
- Wake word detection ("hey messy")
- Natural conversation flow with context awareness
- Follow-up questions without wake word
- RGB LED feedback for system states
- Interrupt handling for fluid interaction

### Storytelling
- Interactive story generation with GPT-4
- Dynamic story adaptation based on child's preferences
- Character and plot thread management
- Story continuation and pause support
- Educational content integration

### Audio System
- High-quality TTS using OpenAI's TTS-1-HD
- Pre-recorded audio assets for common phrases
- Adaptive speech detection thresholds
- Multi-threaded audio processing
- Conversation state management

### LED Feedback System
- Ready state (solid green)
- Listening state (pulsing green)
- Processing state (blinking blue)
- Speaking state (solid blue)
- Story mode (gentle pulsing blue)
- Error state (red flash)

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git
- OpenAI API key
- Picovoice Access key

## Hardware Requirements

1. Raspberry Pi 4 (recommended)
2. USB Microphone (tested with Maono Elf)
   - Input device index: 1
3. Speakers/Headphones (bcm2835)
   - Output device index: 0
4. RGB LED for visual feedback
   - Red: GPIO17
   - Green: GPIO27
   - Blue: GPIO22

## Installation

1. Clone the repository:
