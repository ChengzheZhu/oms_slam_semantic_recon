#!/usr/bin/env python3
"""
alpha_common.py — shared SAM3 EDT-alpha helpers for the L2 scoring stages
(05 / 05b). Loads the L1 mask cache and turns masks into per-pixel EDT alpha
maps. Extracted verbatim from the per-stage copies (bodies were identical).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


def load_mask_cache(cache_path):
    """Load masks from L1 cache. Raises FileNotFoundError if not found."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"L1 mask cache not found: {cache_path}\n"
            "Run step 04 first, or check --mask_cache_dir.")
    data = np.load(cache_path)
    return data['masks'].astype(bool), data['scores'].astype(np.float32)


def generate_alpha_frame(image_path, cache_path, max_size_ratio, edt_gamma):
    """Compute per-pixel stone score via EDT on SAM3 masks loaded from L1 cache."""
    masks_bool, _ = load_mask_cache(cache_path)
    if masks_bool.shape[0] == 0:
        img = Image.open(image_path)
        return np.zeros((img.size[1], img.size[0]), dtype=np.float32)
    h, w     = masks_bool.shape[1], masks_bool.shape[2]
    alpha    = np.zeros((h, w), dtype=np.float32)
    img_area = h * w
    for mask in masks_bool:
        if mask.sum() / img_area > max_size_ratio:
            continue
        dist  = ndimage.distance_transform_edt(mask).astype(np.float32)
        max_d = dist.max()
        score = (dist / max_d) ** edt_gamma if max_d > 0 else mask.astype(np.float32)
        alpha = np.maximum(alpha, score)
    return alpha


def precompute_alphas(cache_dir, alpha_dir, color_files, n_frames,
                      sam_max_size_ratio, edt_gamma):
    os.makedirs(alpha_dir, exist_ok=True)
    n_done = sum(1 for i in range(n_frames)
                 if os.path.exists(os.path.join(alpha_dir, f"alpha_{i:06d}.npz")))
    if n_done == n_frames:
        print(f"  Alpha maps already complete ({n_done}/{n_frames}) — skipping")
        return
    n_workers = max((os.cpu_count() or 4) - 4, 1)
    print(f"  Pre-computing alpha maps: {n_frames} frames  EDT ×{n_workers} thread(s)")

    def worker(idx):
        alpha_path = os.path.join(alpha_dir, f"alpha_{idx:06d}.npz")
        if os.path.exists(alpha_path):
            return
        cache_path = os.path.join(cache_dir, f"masks_{idx:06d}.npz")
        alpha = generate_alpha_frame(
            color_files[idx],
            cache_path=cache_path,
            max_size_ratio=sam_max_size_ratio,
            edt_gamma=edt_gamma)
        np.savez_compressed(alpha_path, alpha=alpha)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(tqdm(pool.map(worker, range(n_frames)),
                  total=n_frames, desc="EDT alpha maps"))
    print(f"  ✓ Alpha maps → {alpha_dir}")
