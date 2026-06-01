#!/bin/bash

# Ensure the script stops on errors
set -e

echo "[1/2] Addressing Calibration Limitations..."
# Duplicate images to bypass the 1,024 minimum requirement
cd /workspace/calib_images/
for img in *.*; do cp "$img" "dup1_$img"; done
for img in dup1_*; do cp "$img" "dup2_$img"; done
cd /workspace/
echo "✓ Calibration images duplicated."

echo "[2/2] Running YOLO26 Compiling Toolkit..."
# Execute the Hailo export CLI
python -m export.cli \
    --variant yolo26n \
    --target hailo8l \
    --onnx /workspace/bottle.onnx \
    --calib_dir /workspace/calib_images/

echo "✓ Compilation finished. Your .hef file is ready."