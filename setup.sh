#!/bin/bash
# One-click setup for CPR AI Scorer
set -e

echo "=== CPR AI Scorer Setup ==="

# Check Python
python3 --version || python --version

# Install deps
echo ""
echo "Installing dependencies..."
pip install mediapipe opencv-python numpy

# Download model
MODEL_DIR="$HOME/.hermes/cache"
mkdir -p "$MODEL_DIR"
MODEL_PATH="$MODEL_DIR/pose_landmarker_lite.task"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading MediaPipe model..."
    curl -L -o "$MODEL_PATH" \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    echo "Model downloaded: $MODEL_PATH"
else
    echo "Model already exists: $MODEL_PATH"
fi

echo ""
echo "=== Setup Complete ==="
echo "Run: python mvp.py"
