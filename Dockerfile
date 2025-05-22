# 1. Base Image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 2. Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_HTTP_TIMEOUT=3600
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 3. System Packages (include OpenCV and Glib dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    build-essential \
    python3-dev \
    cmake && \
    rm -rf /var/lib/apt/lists/*

# 4. Install static ffmpeg
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    | tar -xJ && \
    cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg && \
    cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    ffmpeg -version

# 5. Install pget
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.10.2/pget_linux_x86_64" \
 && chmod +x /usr/local/bin/pget

# 6. Set up Working Directory
WORKDIR /app

# 7. Copy Requirements
COPY requirements.txt .

# 8. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt hf_transfer

# 9. Copy Project Files
COPY . .

# 10. Make setup_env.sh executable and run it
RUN chmod +x setup_env.sh && ./setup_env.sh

# 11. Expose Port
EXPOSE 8080

# 11. Start the RunPod handler
CMD ["python", "runpod_handler.py"]
