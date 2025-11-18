# TinyRecursiveModels training container targeting NVIDIA L4 GPUs
# Uses PyTorch official CUDA-enabled runtime as the base
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# System deps commonly needed for ML training
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        python3-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/TinyRecursiveModels

# Install Python dependencies first to leverage build cache
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 && \
    python -m pip install --no-build-isolation --no-cache-dir adam-atan2 && \
    python -m pip install -r requirements.txt

# Copy the rest of the source tree
COPY . .

# Ensure the project is importable without tweaking PYTHONPATH manually
ENV PYTHONPATH=/workspace/TinyRecursiveModels:$PYTHONPATH

# Default to an interactive shell; override CMD when launching jobs
CMD ["bash"]
