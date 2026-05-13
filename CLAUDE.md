# SegAnyGaussians — Claude Code Notes

## Environment & Setup

### Docker-based workflow
- **Container**: `docker compose run --rm saga` (or `docker compose build` first)
- **GPU**: NVIDIA RTX 5080 (Blackwell, sm_120), 16 GB VRAM
- **CUDA inside container**: CUDA 12.8 (`nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` base image)
- **PyTorch**: 2.7.0+cu128
- **Python**: 3.11 (system Python in Ubuntu 22.04)
- **Host OS**: WSL2 on Windows (kernel `6.6.x-microsoft-standard-WSL2`)
- **Host CUDA driver**: 591.86 (supports CUDA 13.1 runtime, compat with 12.8)

### CUDA extension compilation
The four CUDA extensions (`diff-gaussian-rasterization`, `diff-gaussian-rasterization-depth`,
`diff-gaussian-rasterization_contrastive_f`, `simple-knn`) are compiled **at image build time**
with `TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0 12.0+PTX"`. The `12.0` generates native SASS for
sm_120 (Blackwell); `+PTX` embeds forward-compatible PTX bytecode as well.

Key build flags:
- `FORCE_CUDA=1` — force CUDA builds even without a GPU at build time
- `--no-build-isolation` — required so setup.py can import torch from the system install
- `MAX_JOBS=4` — parallel compile jobs

### JIT cache
- `ENV CUDA_CACHE_PATH=/root/.nv/ComputeCache`
- Named Docker volume `cuda_jit_cache` mounted at `/root/.nv`
- Persistent across container restarts

### segment-anything
Installed at build time from upstream git:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
SAM checkpoint (`sam_vit_h_4b8939.pth`) is bind-mounted read-only from the host.

## Full Pipeline (flowers dataset example)

```bash
# 1. Train scene (fast test: 500 iters, images_8 = 8× downsampled)
python train_scene.py -s data/360_extra_scenes/flowers \
    --images images_8 --iterations 500 \
    -m output/flowers_test

# 2. Extract SAM masks (uses images_8 pre-downsampled)
python extract_segment_everything_masks.py \
    --image_root data/360_extra_scenes/flowers \
    --sam_checkpoint_path /workspace/sam_vit_h_4b8939.pth \
    --downsample 8

# 3. Get CLIP features (reads from images/, masks auto-resampled to match)
python get_clip_features.py \
    --image_root data/360_extra_scenes/flowers

# 4. Get mask scales (requires trained scene model)
python get_scale.py \
    --model_path output/flowers_test \
    --image_root data/360_extra_scenes/flowers

# 5. Train contrastive features
python train_contrastive_feature.py \
    --model_path output/flowers_test \
    --iterations 200 \
    --num_sampled_rays 1000
```

## Known fixes applied (Update-cuda branch)

- `Dockerfile`: PyTorch 2.6→2.7, `--no-build-isolation`, segment-anything from git
- `docker-compose.yml`: `cuda_jit_cache` volume, `CUDA_CACHE_PATH` env var
- `submodules/*/rasterize_points.cu`, `simple-knn/spatial.cu`: `.data<T>()` → `.data_ptr<T>()`
- `extract_segment_everything_masks.py`: filter `Zone.Identifier` files, fix `int == "1"` bug
- `get_clip_features.py`: filter `Zone.Identifier` files
- `clip_utils/__init__.py`: remove headless `plt.imshow()`, add missing `import os`
- `clip_utils/clip_utils.py`: hard-coded local CLIP path → `"laion2b_s34b_b88k"` registry ID
