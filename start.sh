#!/bin/bash
# Install ffmpeg (includes ffprobe)
apt-get update && apt-get install -y ffmpeg

# Navigate to the folder where app.py is
cd BoostMatch/system
# Set PORT (Railway provides this automatically)
export PORT=${PORT:-8080}

# Run the app with gunicorn 
gunicorn app:app --bind 0.0.0.0:$PORT