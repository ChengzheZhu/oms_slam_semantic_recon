#!/usr/bin/env python3
"""
05_sam3_score.py — Compute per-frame EDT alpha maps and fuse them into a
semantic TSDF mesh (L2 scoring pass).

Reads:
  L1 mask cache : <mask_cache_dir>/masks_NNNNNN.npz  (step 04 output)
  trajectory    : trajectory_open3d.log  (step 02)

Writes:
  <output_dir>/alpha_maps/alpha_NNNNNN.npz  — per-frame EDT alpha maps (L2 cache)
  <output_dir>/alpha_mesh.ply               — semantic TSDF mesh encoding alpha scores

The alpha_mesh.ply is consumed by step 06 (cull_segment.py) which transfers
scores to the raw RGB mesh and performs culling + segmentation.

EDT alpha score per pixel:
  0  = seam / background (on mask border or outside all masks)
  1  = stone interior (maximum EDT distance)
  score = (dist / max_dist) ** edt_gamma  (adjustable without re-running step 04)

Usage:
  python scripts/05_sam3_score.py \\
      --frames_dir     /path/to/frames \\
      --mask_cache_dir /path/to/frames/sam3_mask_cache_conf_0.3 \\
      --trajectory     output/sparse/trajectory_open3d.log \\
      --output_dir     output/scoring
"""

import sys
sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib/python3.12')]

import open3d as o3d
import numpy as np
import argparse
import os
from tqdm import tqdm

from pipeline_common import (
    apply_depth_filter,
    get_rgbd_file_lists,
    load_intrinsic,
    load_trajectory_log,
)
from alpha_common import precompute_alphas


def _create_tsdf_volume(voxel_size, depth_max):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    return volume


# ---------------------------------------------------------------------------
# Semantic TSDF integration
# ---------------------------------------------------------------------------

def integrate_semantic_tsdf(frames_dir, intrinsic, poses, alpha_dir,
                             depth_scale=1000.0, depth_max=3.0,
                             voxel_size=0.005, depth_min_m=0.15):
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))
    print(f"\n  Semantic TSDF: {n_frames} frames")

    volume = _create_tsdf_volume(voxel_size, depth_max)
    for idx in tqdm(range(n_frames), desc="Semantic TSDF"):
        alpha_path = os.path.join(alpha_dir, f"alpha_{idx:06d}.npz")
        alpha    = np.load(alpha_path)['alpha']
        a_uint8  = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        alpha_rgb = np.stack([a_uint8, a_uint8, a_uint8], axis=-1)
        depth_np = np.asarray(o3d.io.read_image(depth_files[idx]))
        depth_np = apply_depth_filter(depth_np, depth_scale, depth_min_m)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(alpha_rgb),
            o3d.geometry.Image(depth_np.astype(np.uint16)),
            depth_scale=depth_scale, depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[idx]))

    print("  ✓ Semantic TSDF complete — extracting mesh…")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"  Alpha mesh vertices: {len(mesh.vertices):,}")
    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 05 — SAM3 EDT scoring + semantic TSDF")
    parser.add_argument('--frames_dir',         required=True)
    parser.add_argument('--mask_cache_dir',      required=True,
                        help='L1 mask cache directory (step 04 output)')
    parser.add_argument('--trajectory',         required=True,
                        help='trajectory_open3d.log (step 02 output)')
    parser.add_argument('--output_dir',         required=True,
                        help='Directory to write alpha_maps/ and alpha_mesh.ply')
    parser.add_argument('--intrinsic',          default=None)
    parser.add_argument('--sam_max_size_ratio', type=float, default=0.15)
    parser.add_argument('--edt_gamma',          type=float, default=0.5,
                        help='EDT gamma: <1 = sharp seams, >1 = conservative')
    parser.add_argument('--voxel_size',         type=float, default=0.005)
    parser.add_argument('--depth_max',          type=float, default=3.0)
    parser.add_argument('--depth_min',          type=float, default=0.15)
    args = parser.parse_args()

    if args.intrinsic is None:
        args.intrinsic = os.path.join(args.frames_dir, 'intrinsic.json')

    print("=" * 60)
    print("Step 05 — SAM3 EDT Scoring")
    print("=" * 60)
    print(f"  frames_dir     : {args.frames_dir}")
    print(f"  mask_cache_dir : {args.mask_cache_dir}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  edt_gamma      : {args.edt_gamma}")

    if not os.path.isdir(args.mask_cache_dir):
        print(f"ERROR: mask_cache_dir not found: {args.mask_cache_dir}")
        print("Run step 04 first.")
        import sys; sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    poses = load_trajectory_log(args.trajectory)
    print(f"  Camera: {intrinsic.width}x{intrinsic.height}  Poses: {len(poses)}")

    color_files, _ = get_rgbd_file_lists(args.frames_dir)
    n_frames  = min(len(color_files), len(poses))
    alpha_dir = os.path.join(args.output_dir, 'alpha_maps')

    # Verify L1 cache coverage
    n_cached = sum(1 for i in range(n_frames)
                   if os.path.exists(os.path.join(args.mask_cache_dir, f"masks_{i:06d}.npz")))
    print(f"\n  L1 cache: {n_cached}/{n_frames} frames found in {args.mask_cache_dir}")
    if n_cached < n_frames:
        print(f"WARNING: {n_frames - n_cached} frames missing from L1 cache — "
              "those frames will raise an error during EDT computation.")

    precompute_alphas(
        cache_dir=args.mask_cache_dir,
        alpha_dir=alpha_dir,
        color_files=color_files,
        n_frames=n_frames,
        sam_max_size_ratio=args.sam_max_size_ratio,
        edt_gamma=args.edt_gamma,
    )

    alpha_mesh = integrate_semantic_tsdf(
        args.frames_dir, intrinsic, poses,
        alpha_dir=alpha_dir,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        depth_min_m=args.depth_min,
    )

    alpha_mesh_path = os.path.join(args.output_dir, 'alpha_mesh.ply')
    o3d.io.write_triangle_mesh(alpha_mesh_path, alpha_mesh)
    size_mb = os.path.getsize(alpha_mesh_path) / 1024**2
    print(f"\n✓ alpha_mesh.ply saved: {alpha_mesh_path}  ({size_mb:.1f} MB)")
    print(f"  alpha_maps/ : {alpha_dir}/")
    print(f"\nNext: run step 06 with --raw_mesh <path> --alpha_mesh {alpha_mesh_path}")


if __name__ == "__main__":
    main()
