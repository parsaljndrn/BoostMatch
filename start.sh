#!/bin/bash

echo "🚀 START.SH RUNNING"

# Absolute path to this script's folder
DIR="$(cd "$(dirname "$0")" && pwd)"

# Make FFmpeg and FFprobe executable
chmod +x "$DIR/BoostMatch/system/ffmpeg/ffmpeg"
chmod +x "$DIR/BoostMatch/system/ffmpeg/ffprobe"

# Check binaries
"$DIR/BoostMatch/system/ffmpeg/ffmpeg" -version
"$DIR/BoostMatch/system/ffmpeg/ffprobe" -version

# Move to app directory
cd "$DIR/BoostMatch/system" || { echo "System directory not found"; exit 1; }

# Set port
export PORT=${PORT:-8080}

# Start Gunicorn
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 2 --log-level debug