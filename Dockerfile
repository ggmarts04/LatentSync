# 1. Base Image
# Using a PyTorch image that supports CUDA 12.1 and Python 3.10.x
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 2. Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_HTTP_TIMEOUT=3600
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 3. System Packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libgl1-mesa-glx \
    curl \
    build-essential \
    python3-dev \
    cmake \
 && add-apt-repository ppa:savoury1/ffmpeg4 -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && ffmpeg -version \
 && rm -rf /var/lib/apt/lists/*

# 4. Install pget
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.10.2/pget_linux_x86_64" \
 && chmod +x /usr/local/bin/pget

# 5. Set up Working Directory
WORKDIR /app

# 6. Copy Requirements
COPY requirements.txt .

# 7. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt hf_transfer

# 8. Copy Project Files
COPY . .

# 9. Expose Port (optional for serverless, but good practice)
EXPOSE 8080

# 10. Make setup_env.sh executable and run it
RUN chmod +x setup_env.sh
RUN ./setup_env.sh

# 11. Start the RunPod handler
CMD ["python", "runpod_handler.py"]
