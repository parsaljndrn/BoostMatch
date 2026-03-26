#!/bin/bash

echo "🚀 START.SH RUNNING"

# Make sure the Linux FFmpeg and FFprobe binaries are executable
echo "Making FFmpeg and FFprobe executable..."
chmod +x ./system/ffmpeg/ffmpeg
chmod +x ./system/ffmpeg/ffprobe

# Check FFmpeg and FFprobe version
echo "Checking FFmpeg binary..."
./system/ffmpeg/ffmpeg -version

echo "Checking FFprobe binary..."
./system/ffmpeg/ffprobe -version

echo "Moving to app directory..."
cd BoostMatch/system

# Set port
export PORT=${PORT:-8080}

echo "Starting Gunicorn..."
gunicorn app:app --bind 0.0.0.0:$PORT --log-level debug