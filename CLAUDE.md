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

# 3. Get CLIP features (--downsample must match SAM step to avoid OOM)
python get_clip_features.py \
    --image_root data/360_extra_scenes/flowers \
    --downsample 8

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

## Dataset formats & COLMAP

The scene loader auto-detects the dataset type by what's present in the source dir
(`scene/__init__.py:97-107`), in priority order:

| Detected file/dir | Loader | Poses come from | Initial point cloud |
|-------------------|--------|-----------------|---------------------|
| `sparse/` | `Colmap` | COLMAP SfM | COLMAP sparse cloud |
| `transforms_train.json` | `Blender` | JSON | random (100k) |
| `transforms.json` | `Lerf` | JSON | random (100k) |

### COLMAP — only needed for raw, un-posed image sets
SAGA is built on 3D Gaussian Splatting, which needs **camera poses + a sparse point
cloud** before training. COLMAP (Structure-from-Motion) produces both; it is *not*
the training step. COLMAP ships in the container (`Dockerfile:15`) and is driven by
`convert.py` (standard Inria converter).

Run it only on your **own captured images** that have no poses yet:
```bash
# Raw images must live in <scene>/input/
mkdir -p data/my_scene/input && cp /path/to/photos/*.jpg data/my_scene/input/
docker compose run --rm saga python convert.py -s data/my_scene --no_gpu
# → creates <scene>/images/ and <scene>/sparse/0/  (Colmap loader then kicks in)
```
- **`--no_gpu`** is the safe default — COLMAP's GPU SIFT needs an OpenGL context that
  often fails headless. CPU SIFT always works and is fine for a few hundred images.
- **Avoid `--resize`** — `magick_command` is hard-coded empty (`convert.py:29`) and
  ImageMagick isn't installed, so it would fail. Generate `images_2/4/8` separately.

### Bundled scenes — which already have what
- **`flowers`, `treehill`** — ship `sparse/0/` (COLMAP done). Use the COLMAP recipe
  above with downsampled `images_8` (see flowers pipeline). **No COLMAP rerun needed.**
- **`waldo_kitchen`** — a **LERF-format** scene: ships `transforms.json` (190 frames,
  poses included), **no `sparse/`**. Loads via the `Lerf` path automatically. See below.

### waldo_kitchen (LERF scene) — full-res, NOT images_8
Because it has `transforms.json` and no `sparse/`, the `Lerf` loader is used. Key
differences from the flowers/COLMAP recipe:
- **`--images images_8` is ignored.** The Lerf reader reads each image path straight
  from `transforms.json` (`./images/frame_00001.jpg`, `dataset_readers.py:267`), so it
  always uses full-res `images/` (994×738). SAM/CLIP must run at full res too
  (`--downsample 1`), the *opposite* of the flowers `--downsample 8`.
- Init point cloud is **random 100k** (`dataset_readers.py:310`), not a COLMAP cloud —
  a weaker init, but poses are already good so it trains fine.

```bash
# 1. Train scene (Lerf loader, full-res images/)
python train_scene.py -s data/360_extra_scenes/waldo_kitchen \
    --iterations 500 -m output/waldo_kitchen_test

# 2-3. SAM + CLIP at FULL RES (downsample 1, not 8)
python extract_segment_everything_masks.py \
    --image_root data/360_extra_scenes/waldo_kitchen \
    --sam_checkpoint_path /workspace/sam_vit_h_4b8939.pth --downsample 1
python get_clip_features.py \
    --image_root data/360_extra_scenes/waldo_kitchen --downsample 1

# 4-5. Scale + contrastive (same as flowers)
python get_scale.py --model_path output/waldo_kitchen_test \
    --image_root data/360_extra_scenes/waldo_kitchen
python train_contrastive_feature.py --model_path output/waldo_kitchen_test \
    --iterations 200 --num_sampled_rays 1000
```

> The stray `waldo_kitchen/distorted/` dir is a half-finished COLMAP attempt (no
> `database.db`, empty `sparse/`). Harmless — the loader keys off `sparse/`, which
> doesn't exist — but can be deleted. Running `convert.py` on waldo_kitchen would flip
> it to the Colmap loader (COLMAP poses replace `transforms.json`) **and overwrite the
> existing `images/`** during undistortion; only do it on a fresh copy of the scene.

## Known fixes applied (Update-cuda branch)

- `Dockerfile`: PyTorch 2.6→2.7, `--no-build-isolation`, segment-anything from git
- `docker-compose.yml`: `cuda_jit_cache` volume, `CUDA_CACHE_PATH` env var, `.:/workspace` source bind-mount
- `submodules/*/rasterize_points.cu`, `simple-knn/spatial.cu`: `.data<T>()` → `.data_ptr<T>()`
- `submodules/simple-knn/simple_knn.cu`: `#include <float.h>` (CUDA 12 no longer auto-includes it)
- `extract_segment_everything_masks.py`: filter `Zone.Identifier` files, fix `int == "1"` bug
- `get_clip_features.py`: `--downsample` flag (without it, full-res masked_images tensor OOMs); filter `Zone.Identifier` files
- `clip_utils/__init__.py`: remove headless `plt.imshow()`, add missing `import os`
- `clip_utils/clip_utils.py`: hard-coded local CLIP path → `"laion2b_s34b_b88k"` registry ID
- `get_scale.py`: `torch.meshgrid(..., indexing='ij')`
- `utils/loss_utils.py`: removed deprecated `torch.autograd.Variable`

## Pipeline status (2026-05-13)

All five steps verified end-to-end on RTX 5080 (sm_120):

| Step | Command | Status | Speed |
|------|---------|--------|-------|
| Scene training | `train_scene.py` 500 iters, images_8 | ✅ | ~4 sec @ 130 it/s |
| SAM masks | `extract_segment_everything_masks.py --downsample 8` | ✅ | 173 images |
| Scale | `get_scale.py` | ✅ | 173 images |
| CLIP features | `get_clip_features.py --downsample 8` | ✅ | 53 sec, 173 images |
| Contrastive features | `train_contrastive_feature.py` 200 iters | ✅ | ~12 sec @ 20 it/s |

## YOLO detection overlay (GUI) — experimental

> ⚠️ **WIP / initial testing.** Lives on the `yolo-experiments` branch; not merged
> to `Update-cuda`.

Optional Ultralytics YOLO object-detection overlay for the SAGA GUI. When enabled,
the rendered viewpoint frame is run through YOLO and the detected boxes/labels are
drawn on top of the render.

- **`yolo_inference.py`** — `YoloDetector` wraps Ultralytics `YOLO`; loads the
  checkpoint once on `cuda`, flips RGB→BGR (Ultralytics expects BGR for numpy
  inputs), and returns `(labels, scores, boxes)`.
- **`saga_gui.py`** — adds a "YOLO detection" checkbox and a `YOLO conf` slider
  (`_YoloConf`, default 0.25). The detector is lazy-loaded on first enable
  (~1–3 s pause); boxes are drawn via `cv2` in `draw_yolo_overlay()`. The latest
  results are cached on the GUI as `self.yolo_labels`, `self.yolo_scores`, and
  `self.yolo_boxes` (refreshed each render while enabled).
- **Dependency** — `pip install ultralytics` (added to the `Dockerfile`).
- **Weights** — `yolo26l.pt` (~53 MB) is committed at the repo root and used by
  default. Test artifacts `bus.jpg` and `runs/detect/predict/bus.jpg` are also
  committed.

### YOLO-gated paintbrush prompts
When the **"reject brush prompts outside boxes"** checkbox is on *and* YOLO detection
*and* 2D paintbrush mode are all enabled, **commit brush prompts** keeps only painted
pixels that fall inside an *allowed* YOLO box; the rest are rejected (positives only —
the negative/penalty-zone samples are untouched).
- **Blocklist** — the `block labels (comma-sep)` field (`_YoloBlockLabels`, e.g.
  `person, knife`) marks boxes whose label is rejected too. Parsed live,
  case-insensitive, matched against the model's class names (COCO uses `knife`,
  not `knives`).
- **Allow wins** — a pixel inside *any* allowed box is kept even if a blocked box
  also overlaps it.
- **Live highlight** (`apply_paintbrush_overlay`) — kept = orange, outside all
  boxes = purple, inside a blocked box = red. Blocked-label detection boxes are
  outlined **red** (vs green) in `draw_yolo_overlay`. Gating helpers:
  `brush_gate_active()`, `blocked_labels()`, `yolo_prompt_masks()`.
- **Empty case** — if no painted pixel is inside an allowed box (nothing allowed /
  YOLO found nothing), nothing is committed and the stroke is cleared
  (`[YOLO gate]` message printed).

### YOLO-guided auto ScoreThres
The **`auto ScoreThres (YOLO)`** button (under the `ScoreThres` slider) picks a
`ScoreThres` for the *current view* using YOLO boxes as weak supervision, so you don't
have to hand-tune the slider. **Non-destructive** — it only moves the slider; you still
click `segment3d` to carve. Run it *before* `segment3d` (which prunes the cloud); after a
prune it still works but only over the already-pruned subset. Needs a click/brush prompt
active (else `[auto ScoreThres] add a click or brush prompt first`) and YOLO enabled with
≥1 visible box (else a hint is printed and the slider is left unchanged). It reads
`self.yolo_boxes` from the previous frame, which is fine since the camera is static while
clicking.

Per Gaussian (`auto_tune_scorethres`, `saga_gui.py`):
- **Score** — reuses the same similarity metric as `segment3d` via the shared
  `compute_point_scores()` helper (refactored out of the `segment3d` block).
- **Project** — Gaussian centers → current-view pixels using
  `view_camera.world_view_transform` / `full_proj_transform` and the rasterizer's
  `ndc2Pix`, so projected pixels line up with the YOLO boxes. Only Gaussians that are
  **in front** (`depth > 0.2`) **and** inside the image are judged.
- **+reward** if the pixel lands in an *allowed* box (reuses `yolo_prompt_masks()`, so the
  `block labels` blocklist is respected — blocked boxes count as "outside"), weighted by
  `exp(-angle²/2σ²)` where `angle` is the offset from the camera's optical axis (dead-center
  = full reward). The falloff width `σ` grows with the **`Scale`** slider
  (`AUTO_SIGMA_MIN`=0.12 → `AUTO_SIGMA_MAX`=0.60 rad; constants on `GaussianSplattingGUI`).
- **−penalty** (the **`auto penalty weight`** slider, `_AutoPenalty`, default 1.0) if the
  Gaussian is in view but outside every allowed box. Off-screen Gaussians get weight `0`.
- **Threshold** — `ScoreThres = argmax_t Σ_{score>t} weight`, found in one sorted-`cumsum`
  pass; set to the midpoint between the kept and first-dropped score. Penalties only count
  once a Gaussian is selected, so the low-scoring background is ignored and the penalty
  bites only on real false positives (robust to the in-box/outside count imbalance). If the
  best sum ≤ 0 the threshold is pushed above the max score (select ~nothing).
- **Output** — `dpg.set_value('_ScoreThres', t)` (so the 2D preview refreshes live) plus a
  `[auto ScoreThres] = <t> (reward J=…, in-box=…, kept k/N)` console line.

> **Tuning** — raise `auto penalty weight` for a tighter cut (higher threshold, fewer
> outside-box Gaussians), lower it for a looser one. The angular falloff `AUTO_SIGMA_*`
> are plain constants; promote them to a slider if per-scene control is needed.
