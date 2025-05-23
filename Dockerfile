# 1. Base Image  
# Using a PyTorch image that supports CUDA 12.1 and Python 3.10.x
# Example: pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime (Python 3.10.12)
# Check Docker Hub for the latest appropriate pytorch/pytorch image compatible with Python 3.10 and CUDA 12.1
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 2. Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_HTTP_TIMEOUT=3600
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PATH="/usr/local/bin:${PATH}"


# 3. System Packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    curl \
    xz-utils \
    build-essential \
    python3-dev \
    cmake \
    libglib2.0-0 \
    # other system dependencies from cog.yaml if any were missed (libgl1 is listed)
 && rm -rf /var/lib/apt/lists/*

# Install static ffmpeg
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg-release-amd64-static.tar.xz && \
    mkdir -p /opt/ffmpeg && \
    tar -xf ffmpeg-release-amd64-static.tar.xz -C /opt/ffmpeg --strip-components=1 && \
    rm ffmpeg-release-amd64-static.tar.xz && \
    ln -s /opt/ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    ln -s /opt/ffmpeg/ffprobe /usr/local/bin/ffprobe

# 4. Install pget
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.10.2/pget_linux_x86_64" \
 && chmod +x /usr/local/bin/pget

# 5. Set up Working Directory
WORKDIR /app

# 6. Copy Requirements
COPY requirements.txt .

# 7. Install Python Dependencies
# The --extra-index-url is part of the requirements.txt content itself, so pip should pick it up.
# If not, it needs to be specified here. Assuming it's handled if present in the file.
RUN pip install --no-cache-dir -r requirements.txt hf_transfer
RUN pip install huggingface_hub --quiet

# 8. Copy Project Files
COPY . .

# 9. Expose Port (Good practice, RunPod might manage ports differently)
EXPOSE 8080

# Make setup_env.sh executable and run it
RUN chmod +x setup_env.sh
RUN ./setup_env.sh

# 10. Command to start the RunPod worker
# This assumes 'runpod_handler.py' contains a function 'handler'
# and that the 'runpod' Python package provides the serverless entry point.
# The user should verify this CMD with RunPod's documentation.
CMD ["python", "runpod_handler.py"]
