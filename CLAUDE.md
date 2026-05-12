# SegAnyGaussians — Claude Code Notes

## Environment & Setup

### Docker-based workflow
- **Container**: `docker compose run --rm saga` (or `docker compose build` first)
- **GPU**: NVIDIA RTX 5090 (Blackwell, sm_120)
- **CUDA inside container**: CUDA 11.6 (`nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04` base image)
- **Python**: 3.7.13 (conda env `gaussian_splatting`)
- **Host OS**: WSL2 on Windows (kernel `6.6.x-microsoft-standard-WSL2`)

### CUDA extension compilation
The four CUDA extensions (`diff-gaussian-rasterization`, `diff-gaussian-rasterization-depth`,
`diff-gaussian-rasterization_contrastive_f`, `simple-knn`) are compiled **at image build time**
with `TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"`. The `+PTX` on 8.6 embeds forward-compatible
PTX bytecode into the `.so` files so the NVIDIA driver can JIT-compile them for sm_120 (Blackwell)
at first run.

### JIT compilation on first run (~20 min)
Because CUDA 11.6 predates Blackwell, no native sm_120 SASS is embedded. On every **new image**
or **cold-cache container**, the NVIDIA driver must JIT-compile the PTX from all 14+ `.cu`
compilation units → ~20 min on first execution.

**JIT cache setup (docker-compose.yml + Dockerfile)**:
- `ENV CUDA_CACHE_PATH=/root/.nv/ComputeCache`
- Named Docker volume `cuda_jit_cache` mounted at `/root/.nv`
- Intent: cache compiled sm_120 cubins so subsequent `docker compose run` calls are instant

