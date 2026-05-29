FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Override per-GPU at build time, e.g.:
#   docker compose build --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX"
# Default covers RTX 30xx (8.6), RTX 40xx (8.9), H100 (9.0), RTX 50xx (12.0)
ARG TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0 12.0+PTX"
ARG MAX_JOBS=4
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    colmap \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    ninja-build \
    git \
    curl \
    wget \
    vim \
    libgl1 \
    libglib2.0-0 \
    libxi6 \
    libxcursor1 \
    libxinerama1 \
    libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default and install pip for it
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# PyTorch 2.8 + CUDA 12.8 (separate layer — expensive, cache it)
RUN pip install --no-cache-dir \
    torch==2.8.0+cu128 \
    torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Python dependencies
RUN pip install --no-cache-dir \
    "plyfile>=0.8.1" \
    tqdm \
    matplotlib \
    "hdbscan>=0.8.33" \
    opencv-python-headless \
    open-clip-torch \
    "joblib>=1.1.0" \
    safetensors \
    scipy \
    dearpygui

# Install segment-anything from upstream (third_party/ is excluded from build context)
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/segment-anything.git"

# Copy submodule CUDA extensions and build them — separate layer so changes to
# the main Python source don't invalidate this expensive compile step
COPY submodules/ ./submodules/

RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    FORCE_CUDA=1 \
    MAX_JOBS=${MAX_JOBS} \
    pip install --no-build-isolation --no-cache-dir \
        submodules/diff-gaussian-rasterization \
        submodules/diff-gaussian-rasterization_contrastive_f \
        submodules/diff-gaussian-rasterization-depth \
        submodules/simple-knn

# YOLO inference for the GUI detection overlay (saga_gui.py)
RUN pip install --no-cache-dir ultralytics

# Copy the rest of the project
COPY . .
