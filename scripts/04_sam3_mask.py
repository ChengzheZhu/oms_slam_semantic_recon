#!/usr/bin/env python3
"""
04_sam3_mask.py — Pre-compute SAM3 raw mask cache for all extracted frames (L1 cache).

For each colour frame, runs SAM3 instance segmentation and saves raw binary
masks + per-mask confidence scores to:
  <frames_dir>/sam3_mask_cache/masks_NNNNNN.npz

Already-cached frames are skipped — safe to resume after interruption.
The L1 cache is stored beside the frames so it survives across multiple
meshing runs from the same bag.

Usage:
  python scripts/04_sam3_mask.py --frames_dir /path/to/frames
"""

import sys
sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib/python3.12')]

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def initialize_sam3(confidence_threshold=0.1):
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  ✓ Using CUDA with bfloat16")
    else:
        print("  ⚠ CUDA not available, using CPU")
    model     = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print("  ✓ SAM3 ready")
    return processor


def cache_masks_frame(image_path, processor, prompt, cache_path):
    if os.path.exists(cache_path):
        return
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    h, w = image.size[1], image.size[0]

    state = processor.set_image(image)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=prompt)

    masks  = state["masks"]
    scores = state["scores"]

    if masks.shape[0] == 0:
        masks_np  = np.zeros((0, h, w), dtype=np.uint8)
        scores_np = np.zeros(0, dtype=np.float32)
    else:
        masks_np  = masks.squeeze(1).cpu().numpy().astype(np.uint8)
        scores_np = scores.float().cpu().numpy()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, masks=masks_np, scores=scores_np)


def save_debug_previews(color_files, cache_dir, debug_dir, sam_prompt,
                        sam_confidence, n_samples=15):
    """
    Save side-by-side debug images (original | coloured mask overlay) for
    n_samples evenly-spaced frames.  Written to debug_dir/NNNNNN_masks.jpg.

    Each mask is drawn with:
      - semi-transparent colour fill
      - bounding box outline
      - mask ID and confidence score label
    Footer bar on the overlay side shows: prompt and confidence threshold.
    """
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = font_sm = ImageFont.load_default()

    os.makedirs(debug_dir, exist_ok=True)
    indices = np.linspace(0, len(color_files) - 1, n_samples, dtype=int)
    rng     = np.random.default_rng(42)

    for idx in indices:
        cache_path = os.path.join(cache_dir, f"masks_{idx:06d}.npz")
        if not os.path.exists(cache_path):
            continue

        data      = np.load(cache_path)
        masks_np  = data['masks']    # (N, H, W) uint8
        scores_np = data['scores']   # (N,) float32

        orig    = Image.open(color_files[idx]).convert('RGB')
        overlay = orig.copy().convert('RGBA')

        colours = [tuple(rng.integers(60, 220, size=3).tolist())
                   for _ in range(len(masks_np))]

        for mask_id, (mask, score, colour) in enumerate(
                zip(masks_np, scores_np, colours)):
            # Semi-transparent fill
            fill_arr         = np.zeros((*mask.shape, 4), dtype=np.uint8)
            fill_arr[mask == 1] = (*colour, 100)
            overlay = Image.alpha_composite(
                overlay, Image.fromarray(fill_arr, 'RGBA'))

            # Label at mask centroid
            ys, xs = np.where(mask)
            if not len(xs):
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            label  = f"#{mask_id}  {score:.2f}"
            lw     = len(label) * 7
            ann = Image.new('RGBA', orig.size, (0, 0, 0, 0))
            d   = ImageDraw.Draw(ann)
            d.rectangle([cx - lw // 2 - 2, cy - 8,
                         cx + lw // 2 + 2, cy + 8], fill=(0, 0, 0, 160))
            d.text((cx - lw // 2, cy - 7), label, fill=(*colour, 255), font=font_sm)
            overlay = Image.alpha_composite(overlay, ann)

        overlay_rgb = overlay.convert('RGB')

        # Footer bar: prompt + confidence threshold
        footer_h = 28
        footer   = Image.new('RGB', (orig.width, footer_h), (30, 30, 30))
        fd       = ImageDraw.Draw(footer)
        fd.text((6, 6),
                f'prompt: "{sam_prompt}"   conf: {sam_confidence}   '
                f'masks: {len(masks_np)}   frame: {idx}',
                fill=(220, 220, 220), font=font_sm)

        combined = Image.new('RGB', (orig.width * 2, orig.height + footer_h))
        combined.paste(orig,         (0, 0))
        combined.paste(overlay_rgb,  (orig.width, 0))
        combined.paste(footer,       (orig.width, orig.height))

        combined.save(os.path.join(debug_dir, f"{idx:06d}_masks.jpg"), quality=88)

    print(f"  ✓ {len(indices)} debug previews → {debug_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Step 04 — SAM3 mask cache (L1)")
    parser.add_argument('--frames_dir',      required=True,
                        help='Frames directory (from step 01)')
    parser.add_argument('--sam_prompt',      default="individual stone")
    parser.add_argument('--sam_confidence',  type=float, default=0.1)
    args = parser.parse_args()

    color_dir = os.path.join(args.frames_dir, 'color')
    cache_dir = os.path.join(args.frames_dir, f'sam3_mask_cache_conf_{args.sam_confidence}')

    color_files = sorted(os.path.join(color_dir, f)
                         for f in os.listdir(color_dir)
                         if f.endswith(('.jpg', '.png')))
    n_total  = len(color_files)
    n_cached = sum(1 for i in range(n_total)
                   if os.path.exists(os.path.join(cache_dir, f"masks_{i:06d}.npz")))

    print("=" * 60)
    print("Step 04 — SAM3 Mask Cache")
    print("=" * 60)
    print(f"  frames_dir  : {args.frames_dir}")
    print(f"  cache_dir   : {cache_dir}")
    print(f"  frames      : {n_total}  (cached: {n_cached})")
    print(f"  prompt      : {args.sam_prompt!r}")
    print(f"  confidence  : {args.sam_confidence}")

    debug_dir = cache_dir + "_debug"

    if n_cached == n_total:
        print("  All frames already cached — skipping inference.")
    else:
        print("\nInitializing SAM3…")
        processor = initialize_sam3(args.sam_confidence)
        os.makedirs(cache_dir, exist_ok=True)

        for i, img_path in enumerate(tqdm(color_files, desc="SAM3 masks")):
            cache_masks_frame(img_path, processor,
                              prompt=args.sam_prompt,
                              cache_path=os.path.join(cache_dir, f"masks_{i:06d}.npz"))

        print(f"\n✓ Mask cache complete: {cache_dir}")

    print("\nSaving debug previews…")
    save_debug_previews(color_files, cache_dir, debug_dir,
                        sam_prompt=args.sam_prompt,
                        sam_confidence=args.sam_confidence)


if __name__ == "__main__":
    main()
