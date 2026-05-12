# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
# Compile CUDA extensions for Volta/Turing/Ampere natively.
# The "+PTX" on 8.6 embeds PTX bytecode so the NVIDIA driver can JIT-compile
# for newer architectures (e.g. RTX 5090 / sm_120 / Blackwell) at first run.
# First-run JIT on sm_120 can take 20-30 min; cache it with the cuda_jit_cache
# Docker volume (see docker-compose.yml) so subsequent runs are instant.
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# Persist compiled PTX kernels across container restarts via the cuda_jit_cache volume
ENV CUDA_CACHE_PATH=/root/.nv/ComputeCache
# All sm_120 (Blackwell) JIT-compiled cubins total ~1 GB across the four CUDA extensions.
# The default limit is 256 MB, which evicts ~75% of entries and forces PTX recompilation
# on every run. Set to 2 GB so the full kernel set fits in the index.
ENV CUDA_CACHE_MAXSIZE=2147483648
# Unbuffered Python output so log files update immediately
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git wget curl unzip \
    build-essential cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    libxi6 libxcursor1 libxinerama1 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ─────────────────────────────────────────────────────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /workspace

# ── Third-party repos (no source code needed — runs before any COPY) ──────────
RUN git clone --depth 1 https://github.com/facebookresearch/segment-anything.git \
        third_party/segment-anything \
    && git clone --depth 1 https://github.com/subhadarship/kmeans_pytorch.git \
        third_party/kmeans_pytorch \
    && mkdir -p third_party/segment-anything/sam_ckpt

# ── Conda environment (conda packages only) ───────────────────────────────────
# Accept Anaconda ToS for non-commercial use so conda can use the default channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -y -n gaussian_splatting \
        -c pytorch -c conda-forge -c defaults \
        "python=3.7.13" \
        "cudatoolkit=11.6" \
        "plyfile=0.8.1" \
        "pytorch=1.12.1" \
        "torchaudio=0.12.1" \
        "torchvision=0.13.1" \
        "tqdm" \
        "hdbscan" \
        "matplotlib" \
        "pip=22.3.1" \
    && conda clean -afy

ENV PATH="/opt/conda/envs/gaussian_splatting/bin:$PATH"

# ── CUDA extension submodules ─────────────────────────────────────────────────
# Copy only the C++/CUDA extension sources needed to compile the wheels.
# Isolated here so that changes to Python source files don't invalidate these
# slow compilation layers.
COPY submodules/ submodules/

# ── Pip packages: local CUDA extensions and third-party libs ──────────────────
# BuildKit cache mounts keep the pip HTTP/wheel cache across builds so pure-Python
# packages don't re-download; CUDA extension builds still recompile when sources change.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation /workspace/submodules/diff-gaussian-rasterization
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation /workspace/submodules/diff-gaussian-rasterization_contrastive_f
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation /workspace/submodules/diff-gaussian-rasterization-depth
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation /workspace/submodules/simple-knn
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /workspace/third_party/segment-anything
# open-clip-torch >=2.20 pulls in puccinialin which requires Python >=3.9.
# Pin to 2.0.2 (2022 release) which has no such dependency.
# dearpygui 1.9.x is the last series with Python 3.7 wheels; it bundles GLFW
# and uses OpenGL which is provided via Mesa D3D12 on WSL2 (see docker-compose.yml).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install opencv-python "open-clip-torch==2.0.2" "joblib==1.1.0" "dearpygui==1.9.0"

# ── Fix missing libtiff.so.5 (conda ships libtiff.so.6) ──────────────────────
RUN ln -sf /opt/conda/envs/gaussian_splatting/lib/libtiff.so.6 \
           /opt/conda/envs/gaussian_splatting/lib/libtiff.so.5

# ── pytorch3d (KNN ops needed by train_contrastive_feature.py) ───────────────
# Build from source for CUDA 11.6 / PyTorch 1.12.1 (no pre-built wheel exists)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "fvcore==0.1.5.post20221221" "iopath==0.1.10" && \
    git clone --depth 1 --branch v0.7.2 \
        https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d && \
    pip install --no-build-isolation /tmp/pytorch3d && \
    rm -rf /tmp/pytorch3d

# ── Source code ───────────────────────────────────────────────────────────────
# Comes last so that editing Python scripts, notebooks, or shell scripts does
# not invalidate any of the slow layers above.
# sam_vit_h_4b8939.pth and data/ are excluded via .dockerignore and must be
# bind-mounted at runtime (see docker-compose.yml).
# third_party/ is also excluded — it was populated by the git clones above.
COPY . .

WORKDIR /workspace
CMD ["/bin/bash"]
