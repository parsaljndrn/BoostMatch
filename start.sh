#!/bin/bash
# Navigate to the folder where app.py is
cd BoostMatch/system
# Set PORT (Railway provides this automatically)
export PORT=${PORT:-8080}

# Run the app with uvicorn
uvicorn app:app --host 0.0.0.0 --port $PORT