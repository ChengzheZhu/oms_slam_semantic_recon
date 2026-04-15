#!/usr/bin/env python3
"""
Pre-compute SAM3 raw mask cache for all extracted frames.

For each colour frame, runs SAM3 instance segmentation and saves the raw
binary masks and per-mask confidence scores to
  <frames_dir>/sam3_mask_cache/masks_NNNNNN.npz

The alpha score (EDT-based stone-interior gradient) is derived on-the-fly
during meshing so that parameters like max_size_ratio and alpha_threshold
can be tuned without re-running SAM3 inference.

Cache format per frame:
  masks:  uint8  (N, H, W) — one binary mask per detected instance
  scores: float32 (N,)     — SAM3 confidence score per instance

Already-cached frames are skipped automatically, so this script is safe
to re-run and can resume after interruption.

Usage:
    python scripts/01b_run_sam3_alpha.py --frames_dir /path/to/frames/
"""

import sys
sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib/python3.12')]

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
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
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print("  ✓ SAM3 ready")
    return processor


def cache_masks_frame(image_path, processor, prompt, cache_path):
    """
    Run SAM3 on one frame and save raw masks + confidence scores to cache.

    Saves nothing if cache_path already exists (resumable).
    """
    if os.path.exists(cache_path):
        return

    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    h, w = image.size[1], image.size[0]

    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks  = inference_state["masks"]   # (N, 1, H, W) bool tensor
    scores = inference_state["scores"]  # (N,) float32 tensor

    if masks.shape[0] == 0:
        masks_np  = np.zeros((0, h, w), dtype=np.uint8)
        scores_np = np.zeros(0, dtype=np.float32)
    else:
        masks_np  = masks.squeeze(1).cpu().numpy().astype(np.uint8)
        scores_np = scores.float().cpu().numpy()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, masks=masks_np, scores=scores_np)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute SAM3 mask cache for all frames")
    parser.add_argument('--frames_dir', required=True,
                        help='Directory with color/ subdir (output of 00_extract_frames.py)')
    parser.add_argument('--sam_prompt', default="individual stone")
    parser.add_argument('--sam_confidence', type=float, default=0.1,
                        help='SAM3 confidence threshold (baked into cache — affects which '
                             'masks are stored; changing requires re-running this script)')
    args = parser.parse_args()

    color_dir  = os.path.join(args.frames_dir, 'color')
    cache_dir  = os.path.join(args.frames_dir, 'sam3_mask_cache')
    color_files = sorted([os.path.join(color_dir, f)
                          for f in os.listdir(color_dir)
                          if f.endswith(('.jpg', '.png'))])

    n_total  = len(color_files)
    n_cached = sum(1 for i in range(n_total)
                   if os.path.exists(os.path.join(cache_dir, f"masks_{i:06d}.npz")))

    print("=" * 60)
    print("SAM3 Mask Cache Generation")
    print("=" * 60)
    print(f"  Frames dir  : {args.frames_dir}")
    print(f"  Cache dir   : {cache_dir}")
    print(f"  Frames      : {n_total}  (already cached: {n_cached})")
    print(f"  Confidence  : {args.sam_confidence}")
    print(f"  Prompt      : {args.sam_prompt!r}")

    if n_cached == n_total:
        print("  All frames already cached — nothing to do.")
        return

    print("\nInitializing SAM3...")
    processor = initialize_sam3(args.sam_confidence)

    os.makedirs(cache_dir, exist_ok=True)
    for i, img_path in enumerate(tqdm(color_files, desc="SAM3 masks")):
        cache_path = os.path.join(cache_dir, f"masks_{i:06d}.npz")
        cache_masks_frame(img_path, processor, prompt=args.sam_prompt,
                          cache_path=cache_path)

    print(f"\n✓ Mask cache complete: {cache_dir}")


if __name__ == "__main__":
    main()
