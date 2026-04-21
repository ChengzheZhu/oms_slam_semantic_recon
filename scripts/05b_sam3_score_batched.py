#!/usr/bin/env python3
"""
05b_sam3_score_batched.py — Batched semantic TSDF; batches kept separate.

Same as 05_sam3_score.py but divides the frame sequence into overlapping
temporal batches with the same batch_size / batch_overlap as 03b so that
each alpha batch has a 1-to-1 correspondence with its RGB batch.

No merging step — alpha batches are consumed individually by 06b.

Writes to <output_dir>/alpha_batches/:
  alpha_batch_NNNN.ply  — per-batch alpha mesh (grey = EDT score)
  index.json            — maps batch id → frame range

precompute_alphas() is unchanged: threaded, cache-idempotent.

Extra arguments vs 05_sam3_score.py:
  --batch_size N     Frames per batch (default 100).  Must match 03b.
  --batch_overlap K  Frames shared with next batch (default 15).  Must match 03b.
"""

import sys
sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib/python3.12')]

import gc
import os
import json
import argparse

import numpy as np
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (identical to 05_sam3_score.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_trajectory_log(log_file):
    poses = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def load_intrinsic(intrinsic_file):
    with open(intrinsic_file) as f:
        d = json.load(f)
    m = d['intrinsic_matrix']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=d['width'], height=d['height'],
        fx=m[0], fy=m[4], cx=m[6], cy=m[7])
    return intrinsic, d.get('depth_scale', 1000.0)


def get_rgbd_file_lists(frames_dir):
    color_dir = os.path.join(frames_dir, 'color')
    depth_dir = os.path.join(frames_dir, 'depth')
    color_files = sorted(os.path.join(color_dir, f) for f in os.listdir(color_dir)
                         if f.endswith(('.jpg', '.png')))
    depth_files = sorted(os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                         if f.endswith('.png'))
    return color_files, depth_files


def apply_depth_filter(depth_np, depth_scale, min_depth_m=0.15):
    min_raw = int(min_depth_m * depth_scale)
    invalid = (depth_np == 0) | (depth_np < min_raw)
    if invalid.any():
        depth_np = depth_np.copy()
        depth_np[invalid] = 0
    return depth_np


# ─────────────────────────────────────────────────────────────────────────────
# L1 mask cache + EDT alpha computation
# ─────────────────────────────────────────────────────────────────────────────

def load_mask_cache(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"L1 mask cache not found: {cache_path}\n"
            "Run step 04 first, or check --mask_cache_dir.")
    data = np.load(cache_path)
    return data['masks'].astype(bool), data['scores'].astype(np.float32)


def generate_alpha_frame(image_path, cache_path, max_size_ratio, edt_gamma):
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


# ─────────────────────────────────────────────────────────────────────────────
# Batched semantic TSDF — save each batch, no merge
# ─────────────────────────────────────────────────────────────────────────────

def _make_alpha_volume(voxel_size):
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _integrate_alpha_range(volume, frame_range, alpha_dir, depth_files,
                            intrinsic, poses, depth_scale, depth_max,
                            depth_min_m, desc):
    for i in tqdm(frame_range, desc=desc):
        alpha_path = os.path.join(alpha_dir, f"alpha_{i:06d}.npz")
        alpha      = np.load(alpha_path)['alpha']
        a_uint8    = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        alpha_rgb  = np.stack([a_uint8, a_uint8, a_uint8], axis=-1)
        depth_np   = np.asarray(o3d.io.read_image(depth_files[i]))
        depth_np   = apply_depth_filter(depth_np, depth_scale, depth_min_m)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(alpha_rgb),
            o3d.geometry.Image(depth_np.astype(np.uint16)),
            depth_scale=depth_scale, depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[i]))


def integrate_semantic_tsdf_batched(frames_dir, intrinsic, poses, alpha_dir,
                                     output_dir,
                                     depth_scale=1000.0, depth_max=3.0,
                                     voxel_size=0.005, depth_min_m=0.15,
                                     batch_size=100, batch_overlap=15):
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))

    alpha_batches_dir = os.path.join(output_dir, 'alpha_batches')
    os.makedirs(alpha_batches_dir, exist_ok=True)

    starts    = list(range(0, n_frames, batch_size))
    n_batches = len(starts)

    print(f"\n  Semantic TSDF (batched): {n_frames} frames → {n_batches} batches "
          f"× ~{batch_size} (+{batch_overlap} overlap)  voxel={voxel_size}m")
    print(f"  Saving alpha batches → {alpha_batches_dir}/")

    index_entries = []

    for b, b_start in enumerate(starts):
        b_end    = min(b_start + batch_size + batch_overlap, n_frames)
        n_b      = b_end - b_start
        ply_name = f"alpha_batch_{b:04d}.ply"
        ply_path = os.path.join(alpha_batches_dir, ply_name)

        if os.path.exists(ply_path):
            print(f"\n── Batch {b+1}/{n_batches}  frames [{b_start}:{b_end}]  "
                  f"(already exists, skipping)")
            index_entries.append({
                "id": f"{b:04d}",
                "frame_start": b_start,
                "frame_end": b_end,
                "n_frames": n_b,
                "ply": ply_name,
            })
            continue

        print(f"\n── Batch {b+1}/{n_batches}  frames [{b_start}:{b_end}] ({n_b} frames) ──")

        volume = _make_alpha_volume(voxel_size)
        _integrate_alpha_range(
            volume, range(b_start, b_end),
            alpha_dir, depth_files,
            intrinsic, poses, depth_scale, depth_max, depth_min_m,
            desc=f"Alpha batch {b+1}/{n_batches}")

        mesh_b = volume.extract_triangle_mesh()
        del volume
        gc.collect()
        print(f"  extracted: {len(mesh_b.vertices):,} verts  "
              f"{len(mesh_b.triangles):,} tris")

        o3d.io.write_triangle_mesh(ply_path, mesh_b)
        del mesh_b
        gc.collect()
        print(f"  saved → {ply_path}")

        index_entries.append({
            "id": f"{b:04d}",
            "frame_start": b_start,
            "frame_end": b_end,
            "n_frames": n_b,
            "ply": ply_name,
        })

    index = {
        "voxel_size": voxel_size,
        "total_frames": n_frames,
        "batch_size": batch_size,
        "batch_overlap": batch_overlap,
        "n_batches": n_batches,
        "batches": index_entries,
    }
    index_path = os.path.join(alpha_batches_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"\n✓ index.json → {index_path}")
    return alpha_batches_dir, index


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 05b — Batched SAM3 EDT scoring + semantic TSDF (no merge)")
    parser.add_argument('--frames_dir',         required=True)
    parser.add_argument('--mask_cache_dir',      required=True,
                        help='L1 mask cache directory (step 04 output)')
    parser.add_argument('--trajectory',         required=True,
                        help='trajectory_open3d.log (step 02 output)')
    parser.add_argument('--output_dir',         required=True,
                        help='Directory to write alpha_maps/ and alpha_batches/')
    parser.add_argument('--intrinsic',          default=None)
    parser.add_argument('--sam_max_size_ratio', type=float, default=0.15)
    parser.add_argument('--edt_gamma',          type=float, default=0.5,
                        help='EDT gamma: <1 = sharp seams, >1 = conservative')
    parser.add_argument('--voxel_size',         type=float, default=0.005)
    parser.add_argument('--depth_max',          type=float, default=3.0)
    parser.add_argument('--depth_min',          type=float, default=0.15)
    parser.add_argument('--batch_size',         type=int,   default=100,
                        help='Frames per batch — must match 03b value.')
    parser.add_argument('--batch_overlap',      type=int,   default=15,
                        help='Overlap frames — must match 03b value.')
    args = parser.parse_args()

    if args.intrinsic is None:
        args.intrinsic = os.path.join(args.frames_dir, 'intrinsic.json')

    print("=" * 60)
    print("Step 05b — Batched SAM3 EDT Scoring (no merge)")
    print("=" * 60)
    print(f"  frames_dir     : {args.frames_dir}")
    print(f"  mask_cache_dir : {args.mask_cache_dir}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  edt_gamma      : {args.edt_gamma}")
    print(f"  batch_size     : {args.batch_size}  overlap: {args.batch_overlap}")

    if not os.path.isdir(args.mask_cache_dir):
        print(f"ERROR: mask_cache_dir not found: {args.mask_cache_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    poses = load_trajectory_log(args.trajectory)
    print(f"  Camera: {intrinsic.width}x{intrinsic.height}  Poses: {len(poses)}")

    color_files, _ = get_rgbd_file_lists(args.frames_dir)
    n_frames  = min(len(color_files), len(poses))
    alpha_dir = os.path.join(args.output_dir, 'alpha_maps')

    n_cached = sum(1 for i in range(n_frames)
                   if os.path.exists(os.path.join(args.mask_cache_dir, f"masks_{i:06d}.npz")))
    print(f"\n  L1 cache: {n_cached}/{n_frames} frames found in {args.mask_cache_dir}")
    if n_cached < n_frames:
        print(f"WARNING: {n_frames - n_cached} frames missing from L1 cache.")

    precompute_alphas(
        cache_dir=args.mask_cache_dir,
        alpha_dir=alpha_dir,
        color_files=color_files,
        n_frames=n_frames,
        sam_max_size_ratio=args.sam_max_size_ratio,
        edt_gamma=args.edt_gamma,
    )

    alpha_batches_dir, index = integrate_semantic_tsdf_batched(
        args.frames_dir, intrinsic, poses,
        alpha_dir=alpha_dir,
        output_dir=args.output_dir,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        depth_min_m=args.depth_min,
        batch_size=args.batch_size,
        batch_overlap=args.batch_overlap,
    )

    print("\n" + "=" * 60)
    print(f"  {index['n_batches']} alpha batch meshes in {alpha_batches_dir}/")
    print(f"\nNext: run 06b with --alpha_batches_dir {alpha_batches_dir}")


if __name__ == "__main__":
    main()
