#!/usr/bin/env python3
"""
05_sam3_score.py — Per-frame seam votes from the SAM3 masks, fused into a
semantic TSDF mesh (L2 scoring pass). Encoding: see vote_common.py.

Reads:
  L1 mask cache : <mask_cache_dir>/masks_NNNNNN.npz  (step 04 output)
  trajectory    : trajectory_open3d.log  (step 02)
  depth         : <frames_dir>/depth/ (aligned to colour in step 01)

Writes:
  <output_dir>/edt_maps/edt_NNNNNN.npz  — per-frame inside mask + pixel EDT (L2 cache)
  <output_dir>/vote_mesh.ply            — semantic TSDF mesh; vertex colour R, G
                                          encode seam score S and vote V

Per pixel inside a stone mask: seam score s = -1 on the mask edge, rising to
+1 at --seam_width_mm inside (distance in mm from the pixel's depth). Per
voxel after fusion: V = share of the frames seeing it that put it inside a
mask, S = mean s over those frames. Step 06 transfers V and S to the raw RGB
mesh and culls.

Usage:
  python scripts/05_sam3_score.py \\
      --frames_dir     /path/to/frames \\
      --mask_cache_dir /path/to/frames/sam3_merged_individual_stone+paper_label_conf_0.5 \\
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
    load_frame_poses,
)
from vote_common import decode_votes, precompute_edts, vote_rgb


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

def integrate_vote_tsdf(frames_dir, intrinsic, poses, edt_dir,
                        seam_width_mm, seam_gamma,
                        depth_scale=1000.0, depth_max=3.0,
                        voxel_size=0.005, depth_min_m=0.15):
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))
    fx = intrinsic.get_focal_length()[0]
    print(f"\n  Semantic TSDF: {n_frames} frames  seam width {seam_width_mm} mm  "
          f"gamma {seam_gamma}")

    volume = _create_tsdf_volume(voxel_size, depth_max)
    for idx in tqdm(range(n_frames), desc="Semantic TSDF"):
        if poses[idx] is None:   # not tracked by ORB-SLAM3
            continue
        depth_np = np.asarray(o3d.io.read_image(depth_files[idx]))
        depth_np = apply_depth_filter(depth_np, depth_scale, depth_min_m)
        vote_np  = vote_rgb(os.path.join(edt_dir, f"edt_{idx:06d}.npz"), depth_np,
                            depth_scale, fx, seam_width_mm, seam_gamma)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(vote_np),
            o3d.geometry.Image(depth_np.astype(np.uint16)),
            depth_scale=depth_scale, depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[idx]))

    print("  ✓ Semantic TSDF complete — extracting mesh…")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"  Vote mesh vertices: {len(mesh.vertices):,}")
    return mesh


def print_vote_summary(mesh):
    vote, score = decode_votes(mesh.vertex_colors)
    pct = lambda v: '  '.join(f"p{p} {np.percentile(v, p):+.2f}" for p in (10, 50, 90))
    print(f"  vote V  : {pct(vote)}   (V = 0 on {np.mean(vote < 1e-6):.1%} of vertices)")
    print(f"  score S : {pct(score[vote > 1e-6])}   (vertices with V > 0)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 05 — SAM3 seam votes + semantic TSDF")
    parser.add_argument('--frames_dir',         required=True)
    parser.add_argument('--mask_cache_dir',      required=True,
                        help='L1 mask cache directory (step 04 output)')
    parser.add_argument('--trajectory',         required=True,
                        help='trajectory_open3d.log (step 02 output)')
    parser.add_argument('--output_dir',         required=True,
                        help='Directory to write edt_maps/ and vote_mesh.ply')
    parser.add_argument('--intrinsic',          default=None)
    parser.add_argument('--sam_max_size_ratio', type=float, default=0.15)
    parser.add_argument('--seam_width_mm',      type=float, default=12.0,
                        help='Seam score rises from -1 at a mask edge to +1 this far inside (mm)')
    parser.add_argument('--seam_gamma',         type=float, default=1.0,
                        help='Score falloff shape: <1 rises fast (narrow seam), 1 linear, >1 wide')
    parser.add_argument('--voxel_size',         type=float, default=0.005)
    parser.add_argument('--depth_max',          type=float, default=3.0)
    parser.add_argument('--depth_min',          type=float, default=0.15)
    args = parser.parse_args()

    if args.intrinsic is None:
        args.intrinsic = os.path.join(args.frames_dir, 'intrinsic.json')

    print("=" * 60)
    print("Step 05 — SAM3 Seam Votes")
    print("=" * 60)
    print(f"  frames_dir     : {args.frames_dir}")
    print(f"  mask_cache_dir : {args.mask_cache_dir}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  seam           : {args.seam_width_mm} mm, gamma {args.seam_gamma}")

    if not os.path.isdir(args.mask_cache_dir):
        print(f"ERROR: mask_cache_dir not found: {args.mask_cache_dir}")
        print("Run step 04 first.")
        import sys; sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    print(f"  Camera: {intrinsic.width}x{intrinsic.height}")
    poses = load_frame_poses(args.trajectory, args.frames_dir)

    color_files, _ = get_rgbd_file_lists(args.frames_dir)
    n_frames = min(len(color_files), len(poses))
    edt_dir  = os.path.join(args.output_dir, 'edt_maps')

    # Verify L1 cache coverage
    n_cached = sum(1 for i in range(n_frames)
                   if os.path.exists(os.path.join(args.mask_cache_dir, f"masks_{i:06d}.npz")))
    print(f"\n  L1 cache: {n_cached}/{n_frames} frames found in {args.mask_cache_dir}")
    if n_cached < n_frames:
        print(f"WARNING: {n_frames - n_cached} frames missing from L1 cache — "
              "those frames will raise an error during EDT computation.")

    precompute_edts(
        cache_dir=args.mask_cache_dir,
        edt_dir=edt_dir,
        n_frames=n_frames,
        max_size_ratio=args.sam_max_size_ratio,
    )

    vote_mesh = integrate_vote_tsdf(
        args.frames_dir, intrinsic, poses,
        edt_dir=edt_dir,
        seam_width_mm=args.seam_width_mm,
        seam_gamma=args.seam_gamma,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        depth_min_m=args.depth_min,
    )
    print_vote_summary(vote_mesh)

    vote_mesh_path = os.path.join(args.output_dir, 'vote_mesh.ply')
    o3d.io.write_triangle_mesh(vote_mesh_path, vote_mesh)
    size_mb = os.path.getsize(vote_mesh_path) / 1024**2
    print(f"\n✓ vote_mesh.ply saved: {vote_mesh_path}  ({size_mb:.1f} MB)")
    print(f"  edt_maps/ : {edt_dir}/")
    print(f"\nNext: run step 06 with --raw_mesh <path> --vote_mesh {vote_mesh_path}")


if __name__ == "__main__":
    main()
