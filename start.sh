#!/bin/bash

echo "🚀 START.SH RUNNING"

# Absolute path to this script's folder
DIR="$(cd "$(dirname "$0")" && pwd)"

# Make FFmpeg and FFprobe executable
echo "Making FFmpeg and FFprobe executable..."
chmod +x "$DIR/system/ffmpeg/ffmpeg"
chmod +x "$DIR/system/ffmpeg/ffprobe"

# Check binaries
echo "Checking FFmpeg version..."
"$DIR/system/ffmpeg/ffmpeg" -version

echo "Checking FFprobe version..."
"$DIR/system/ffmpeg/ffprobe" -version

# Move to app directory
cd "$DIR/system"

# Set port
export PORT=${PORT:-8080}

# Start Gunicorn
echo "Starting Gunicorn..."
gunicorn app:app --bind 0.0.0.0:$PORT --log-level debug