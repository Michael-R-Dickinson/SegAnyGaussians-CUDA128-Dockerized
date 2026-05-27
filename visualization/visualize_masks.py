"""
Visualize SAM masks (.pt files) produced by extract_segment_everything_masks.py,
optionally coloured by the 3-D scale values from get_scale.py.

Each sam_masks/<name>.pt  is a (N, H, W) bool tensor  — N masks for one image.
Each mask_scales/<name>.pt is a (N,)  float tensor — one scale per mask.

Minimal usage (single file, random colours):
    python visualization/visualize_masks.py \\
        data/nerf_llff_data/trex/sam_masks/DJI_20200223_163645_961.pt

With the original image blended in:
    python visualization/visualize_masks.py \\
        data/360_extra_scenes/flowers/sam_masks/DSC00001.pt \\
        --image-root data/360_extra_scenes/flowers

Colour masks by 3-D scale (requires matching mask_scales file):
    python visualization/visualize_masks.py \\
        data/360_extra_scenes/flowers/sam_masks/DSC00001.pt \\
        --image-root data/360_extra_scenes/flowers \\
        --scales data/360_extra_scenes/flowers/mask_scales/DSC00001.pt

Batch mode — process every .pt in a sam_masks directory:
    python visualization/visualize_masks.py \\
        data/360_extra_scenes/flowers/sam_masks/ \\
        --image-root data/360_extra_scenes/flowers \\
        --scales-dir data/360_extra_scenes/flowers/mask_scales/ \\
        --out-dir /tmp/flowers_mask_vis

Via Docker:
    docker compose run --rm saga python visualization/visualize_masks.py \\
        data/360_extra_scenes/flowers/sam_masks/ \\
        --image-root data/360_extra_scenes/flowers \\
        --scales-dir data/360_extra_scenes/flowers/mask_scales/ \\
        --out-dir /tmp/flowers_mask_vis
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def load_image(image_root: str, stem: str) -> np.ndarray | None:
    """Return HxWx3 uint8 array for *stem* found under image_root/images*/, or None."""
    for sub in ("images", "images_8", "images_4", "images_2"):
        d = os.path.join(image_root, sub)
        if not os.path.isdir(d):
            continue
        for ext in IMAGE_EXTS:
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return np.array(Image.open(p).convert("RGB"))
    return None


def colorize_by_random(masks: torch.Tensor) -> np.ndarray:
    """Assign each mask a distinct random colour; return HxWx3 uint8."""
    N, H, W = masks.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for i in range(N):
        color = rng.integers(60, 230, size=3).astype(np.uint8)
        canvas[masks[i].numpy()] = color
    return canvas


def colorize_by_scale(masks: torch.Tensor, scales: torch.Tensor) -> np.ndarray:
    """Colour each mask by its normalised 3-D scale using a jet-like colormap."""
    from matplotlib import colormaps

    N, H, W = masks.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    s = scales.detach().numpy().astype(float)
    lo, hi = s.min(), s.max()
    norm = (s - lo) / (hi - lo + 1e-8)

    cmap = colormaps["jet"]
    for i in range(N):
        r, g, b, _ = cmap(norm[i])
        color = np.array([r * 255, g * 255, b * 255], dtype=np.uint8)
        canvas[masks[i].numpy()] = color
    return canvas


def add_colorbar(canvas: np.ndarray, scales: torch.Tensor, width: int = 30) -> np.ndarray:
    """Append a vertical jet colorbar to the right of *canvas*."""
    from matplotlib import colormaps

    H, W, _ = canvas.shape
    bar = np.zeros((H, width, 3), dtype=np.uint8)
    cmap = colormaps["jet"]
    for row in range(H):
        t = 1.0 - row / (H - 1)
        r, g, b, _ = cmap(t)
        bar[row] = [int(r * 255), int(g * 255), int(b * 255)]

    s = scales.detach()
    lo, hi = float(s.min()), float(s.max())
    result = np.concatenate([canvas, bar], axis=1)

    try:
        from PIL import ImageDraw, ImageFont
        img = Image.fromarray(result)
        draw = ImageDraw.Draw(img)
        draw.text((W + 2, 2), f"{hi:.2f}", fill=(255, 255, 255))
        draw.text((W + 2, H - 12), f"{lo:.2f}", fill=(255, 255, 255))
        result = np.array(img)
    except Exception:
        pass

    return result


def visualize_single(
    pt_path: str,
    image_root: str | None,
    scales_path: str | None,
    out_path: str,
) -> None:
    masks = torch.load(pt_path, map_location="cpu")  # (N, H, W) bool
    if masks.dtype != torch.bool:
        masks = masks.bool()

    stem = os.path.splitext(os.path.basename(pt_path))[0]
    print(f"{stem}: {masks.shape[0]} masks  {masks.shape[1]}×{masks.shape[2]}")

    scales = None
    if scales_path and os.path.exists(scales_path):
        scales = torch.load(scales_path, map_location="cpu")  # (N,)

    if scales is not None and len(scales) == len(masks):
        overlay = colorize_by_scale(masks, scales)
        overlay = add_colorbar(overlay, scales)
    else:
        overlay = colorize_by_random(masks)

    if image_root:
        orig = load_image(image_root, stem)
        if orig is not None:
            H, W = overlay.shape[:2]
            orig = np.array(Image.fromarray(orig).resize((W, H), Image.BILINEAR))
            overlay = (orig * 0.45 + overlay * 0.55).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(overlay).save(out_path)
    print(f"  → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SAM mask .pt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help=".pt file or directory of .pt files (sam_masks/)")
    parser.add_argument("--image-root", default=None,
                        help="Dataset root containing images/ (or images_8/ etc.) for blending")
    parser.add_argument("--scales", default=None,
                        help="mask_scales .pt file matching the input (single-file mode)")
    parser.add_argument("--scales-dir", default=None,
                        help="Directory of mask_scales .pt files (batch mode)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: alongside each input .pt)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        pt_files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".pt") and "Zone.Identifier" not in f
        )
        if not pt_files:
            sys.exit(f"No .pt files found in {args.input}")
        for pt in pt_files:
            stem = os.path.splitext(os.path.basename(pt))[0]
            scales_path = (
                os.path.join(args.scales_dir, stem + ".pt") if args.scales_dir else None
            )
            out_dir = args.out_dir or os.path.dirname(pt)
            out_path = os.path.join(out_dir, stem + "_vis.png")
            visualize_single(pt, args.image_root, scales_path, out_path)
    else:
        out_dir = args.out_dir or os.path.dirname(args.input)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(out_dir, stem + "_vis.png")
        visualize_single(args.input, args.image_root, args.scales, out_path)


if __name__ == "__main__":
    main()
