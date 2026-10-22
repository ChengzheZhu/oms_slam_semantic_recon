#!/usr/bin/env python3
"""
vote_common.py — per-frame seam voting for the L2 scoring stages (05 / 05b)
and its decoding for 06 / 06b.

Per frame, from the step 04 masks (masks covering > max_size_ratio of the
image are skipped):
  inside  1 where the pixel lies in a stone mask, else 0
  edt     pixel distance to the nearest mask edge (scipy EDT: 1 on the edge
          pixels; the image border is not an edge)
At integration the EDT becomes millimetres with the pixel's own depth (depth
is aligned to colour in step 01), d_mm = (edt - 1) * Z / fx, and a seam score

  s = 2 * clip(d_mm / seam_width_mm, 0, 1) ** seam_gamma - 1    (-1 edge … +1 interior)

The TSDF colour carries R = inside * (s + 1) / 2 and G = inside. Its weighted
average over every frame that saw a voxel gives

  vote  V = avg G                 share of those frames that put it inside a stone mask
  score S = 2 * avg R / avg G - 1 mean s over just the frames that put it inside

The weights are shared, so they cancel and S is exact (Open3D keeps the
average as float). Pixels without valid depth are not integrated at all.
Step 06 keeps vertices with V >= vote_min and S >= its score threshold.

EDT cache: <scoring dir>/edt_maps/edt_NNNNNN.npz (inside, edt). It holds no
depth or seam settings, so changing seam_width_mm / seam_gamma only re-runs
the TSDF. Its params.json records the mask cache and max_size_ratio; a
mismatch rebuilds the maps.
"""

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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


def edt_frame(cache_path, max_size_ratio):
    """(inside, edt) for one frame: union of the kept masks, and the largest
    per-mask EDT at each pixel (0 outside every mask)."""
    masks, _ = load_mask_cache(cache_path)
    h, w = masks.shape[1:]
    inside = np.zeros((h, w), dtype=bool)
    edt = np.zeros((h, w), dtype=np.float32)
    for mask in masks:
        if mask.sum() / (h * w) > max_size_ratio:
            continue
        ys, xs = np.nonzero(mask)
        if not len(ys):
            continue
        # The EDT inside the bounding box padded by 1 px (within the image)
        # equals the full-image EDT: every pixel beyond is farther than the pad.
        y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, h)
        x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, w)
        crop = mask[y0:y1, x0:x1]
        inside[y0:y1, x0:x1] |= crop
        np.maximum(edt[y0:y1, x0:x1],
                   ndimage.distance_transform_edt(crop).astype(np.float32),
                   out=edt[y0:y1, x0:x1])
    return inside, edt


def _edt_params(cache_dir, max_size_ratio):
    mask_params = None
    mask_params_path = os.path.join(cache_dir, 'params.json')
    if os.path.exists(mask_params_path):
        with open(mask_params_path) as f:
            mask_params = json.load(f)
    return dict(mask_cache_dir=os.path.abspath(cache_dir), mask_cache_params=mask_params,
                sam_max_size_ratio=max_size_ratio)


def precompute_edts(cache_dir, edt_dir, n_frames, max_size_ratio):
    os.makedirs(edt_dir, exist_ok=True)
    params = _edt_params(cache_dir, max_size_ratio)
    params_path = os.path.join(edt_dir, 'params.json')
    old_params = None
    if os.path.exists(params_path):
        with open(params_path) as f:
            old_params = json.load(f)
    stale = glob.glob(os.path.join(edt_dir, 'edt_*.npz'))
    if stale and old_params != params:
        print(f"  EDT maps were made from other masks/settings "
              f"({'no params.json' if old_params is None else old_params}); "
              f"rebuilding {len(stale)} maps")
        for path in stale:
            os.remove(path)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    n_done = sum(1 for i in range(n_frames)
                 if os.path.exists(os.path.join(edt_dir, f"edt_{i:06d}.npz")))
    if n_done == n_frames:
        print(f"  EDT maps already complete ({n_done}/{n_frames}) — skipping")
        return
    n_workers = max((os.cpu_count() or 4) - 4, 1)
    print(f"  Pre-computing EDT maps: {n_frames} frames  ×{n_workers} thread(s)")

    def worker(idx):
        edt_path = os.path.join(edt_dir, f"edt_{idx:06d}.npz")
        if os.path.exists(edt_path):
            return
        inside, edt = edt_frame(os.path.join(cache_dir, f"masks_{idx:06d}.npz"),
                                max_size_ratio)
        np.savez_compressed(edt_path, inside=inside, edt=edt.astype(np.float16))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(tqdm(pool.map(worker, range(n_frames)),
                  total=n_frames, desc="EDT maps"))
    print(f"  ✓ EDT maps → {edt_dir}")


def vote_rgb(edt_path, depth_np, depth_scale, fx, seam_width_mm, seam_gamma):
    """uint8 RGB vote image for one frame: R = inside * (s + 1) / 2, G = inside."""
    data = np.load(edt_path)
    inside, edt = data['inside'], data['edt'].astype(np.float32)
    d_mm = np.maximum(edt - 1.0, 0.0) * (depth_np.astype(np.float32) * (1000.0 / depth_scale)) / fx
    s = 2.0 * np.clip(d_mm / seam_width_mm, 0.0, 1.0) ** seam_gamma - 1.0
    rgb = np.zeros((*inside.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.where(inside, np.rint((s + 1.0) * 127.5), 0).astype(np.uint8)
    rgb[..., 1] = np.where(inside, 255, 0).astype(np.uint8)
    return rgb


def decode_votes(vertex_colors):
    """(vote V, score S) per vertex from a vote mesh's colours (0-1 floats).
    S is -1 where no frame put the vertex inside a mask."""
    c = np.asarray(vertex_colors)
    vote = c[:, 1].astype(np.float32)
    score = np.full(len(c), -1.0, dtype=np.float32)
    seen = vote > 1e-6
    score[seen] = np.clip(2.0 * c[seen, 0] / vote[seen] - 1.0, -1.0, 1.0)
    return vote, score
