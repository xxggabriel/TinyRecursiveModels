# TinyRecursiveModels training container targeting NVIDIA L4 GPUs
# Uses PyTorch official CUDA-enabled runtime as the base
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

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
    python -m pip install -r requirements.txt

# Copy the rest of the source tree
COPY . .

# Ensure the project is importable without tweaking PYTHONPATH manually
ENV PYTHONPATH=/workspace/TinyRecursiveModels:$PYTHONPATH

# Default to an interactive shell; override CMD when launching jobs
CMD ["bash"]
