#!/bin/bash

echo "🚀 START.SH RUNNING"

# Make sure the Linux FFmpeg binary is executable
echo "Making FFmpeg executable..."
chmod +x ./system/ffmpeg/ffmpeg

# Check FFmpeg version to verify it works
echo "Checking FFmpeg binary..."
./system/ffmpeg/ffmpeg -version

echo "Moving to app directory..."
cd BoostMatch/system

# Set port
export PORT=${PORT:-8080}

echo "Starting Gunicorn..."
gunicorn app:app --bind 0.0.0.0:$PORT --log-level debug