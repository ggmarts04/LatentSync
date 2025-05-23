#!/bin/bash
set -e

# Create a new conda environment
conda create -y -n latentsync python=3.10.13
# conda activate latentsync

# Install ffmpeg
conda run -n latentsync --no-capture-output conda install -y -c conda-forge ffmpeg

# Python dependencies
conda run -n latentsync --no-capture-output pip install -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
conda run -n latentsync --no-capture-output huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints
conda run -n latentsync --no-capture-output huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints