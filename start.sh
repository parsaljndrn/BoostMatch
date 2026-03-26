#!/bin/bash

echo "🚀 START.SH RUNNING"

echo "Updating packages..."
apt-get update

echo "Installing ffmpeg..."
apt-get install -y ffmpeg

echo "Checking ffmpeg..."
which ffmpeg
which ffprobe

echo "Moving to app directory..."
cd BoostMatch/system

export PORT=${PORT:-8080}

echo "Starting Gunicorn..."
gunicorn app:app --bind 0.0.0.0:$PORT --log-level debug