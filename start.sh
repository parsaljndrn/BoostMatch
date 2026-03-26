#!/bin/bash

echo "🚀 START.SH RUNNING"

# Absolute path to this script's folder
DIR="$(cd "$(dirname "$0")" && pwd)"

# Make FFmpeg and FFprobe executable
if [ -f "$DIR/BoostMatch/system/ffmpeg/ffmpeg" ]; then
    chmod +x "$DIR/BoostMatch/system/ffmpeg/ffmpeg"
    export WHISPER_FFMPEG="$DIR/BoostMatch/system/ffmpeg/ffmpeg"
    echo "Using local FFmpeg for Faster-Whisper: $WHISPER_FFMPEG"
else
    echo "⚠️ FFmpeg binary not found at $DIR/BoostMatch/system/ffmpeg/ffmpeg"
fi

if [ -f "$DIR/BoostMatch/system/ffmpeg/ffprobe" ]; then
    chmod +x "$DIR/BoostMatch/system/ffmpeg/ffprobe"
fi

# Check FFmpeg binaries (optional)
"$DIR/BoostMatch/system/ffmpeg/ffmpeg" -version
"$DIR/BoostMatch/system/ffmpeg/ffprobe" -version

# Move to app directory
cd "$DIR/BoostMatch/system" || { echo "System directory not found"; exit 1; }

# Debug: Python environment and package
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
python -m pip show faster-whisper || echo "⚠️ Faster-Whisper not installed!"

# Set port
export PORT=${PORT:-8080}

# Start Gunicorn
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 2 --log-level debug