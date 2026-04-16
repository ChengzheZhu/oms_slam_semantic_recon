#!/usr/bin/env python3
"""
03_tsdf_rgb.py — Fuse RGB-D frames into a dense mesh using TSDF integration.

Reads the ORB-SLAM3 trajectory (Open3D log format) and integrates all frames
into a ScalableTSDFVolume. Outputs a single coloured PLY mesh.

Inputs:
  --frames_dir   Directory with color/ and depth/ subdirs (step 01 output)
  --trajectory   trajectory_open3d.log (step 02 output)
  --output       Output PLY path, e.g. output/20260127.../raw_mesh_rgb.ply

Usage:
  python scripts/03_tsdf_rgb.py \\
      --frames_dir /path/to/frames \\
      --trajectory output/sparse/trajectory_open3d.log \\
      --output     output/raw_mesh_rgb.ply
"""

import open3d as o3d
import numpy as np
import json
import argparse
import os
from tqdm import tqdm
from PIL import Image


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


def apply_depth_filter(depth_np, depth_scale, min_depth_m=0.15,
                       confidence_np=None, confidence_threshold=0):
    min_raw = int(min_depth_m * depth_scale)
    invalid = (depth_np == 0) | (depth_np < min_raw)
    if confidence_np is not None and confidence_threshold > 0:
        invalid |= (confidence_np < confidence_threshold)
    if invalid.any():
        depth_np = depth_np.copy()
        depth_np[invalid] = 0
    return depth_np


def integrate_tsdf(frames_dir, intrinsic, poses, depth_scale=1000.0,
                   depth_max=3.0, voxel_size=0.005, depth_min_m=0.15,
                   confidence_threshold=0):
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))

    conf_dir  = os.path.join(frames_dir, 'confidence')
    use_conf  = (confidence_threshold > 0) and os.path.isdir(conf_dir)
    conf_files = (sorted(os.path.join(conf_dir, f) for f in os.listdir(conf_dir)
                         if f.endswith('.png')) if use_conf else [])

    print(f"  {n_frames} frames, voxel={voxel_size}m, "
          f"depth=[{depth_min_m},{depth_max}]m, confidence={confidence_threshold}"
          f"{' (active)' if use_conf else ''}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in tqdm(range(n_frames), desc="TSDF integration"):
        color_np = np.asarray(Image.open(color_files[i]).convert('RGB'))
        depth_np = np.asarray(o3d.io.read_image(depth_files[i]))
        conf_np  = (np.asarray(o3d.io.read_image(conf_files[i]))
                    if use_conf and i < len(conf_files) else None)
        depth_np = apply_depth_filter(depth_np, depth_scale, depth_min_m,
                                      conf_np, confidence_threshold)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_np),
            o3d.geometry.Image(depth_np.astype(np.uint16)),
            depth_scale=depth_scale, depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[i]))

    print("✓ Integration complete — extracting mesh…")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"  Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Step 03 — RGB TSDF meshing")
    parser.add_argument('--frames_dir',  required=True)
    parser.add_argument('--trajectory',  required=True,
                        help='trajectory_open3d.log (step 02 output)')
    parser.add_argument('--output',      required=True,
                        help='Output PLY path for raw RGB mesh')
    parser.add_argument('--intrinsic',   default=None,
                        help='intrinsic.json (default: <frames_dir>/intrinsic.json)')
    parser.add_argument('--voxel_size',  type=float, default=0.005)
    parser.add_argument('--depth_max',   type=float, default=3.0)
    parser.add_argument('--depth_min',   type=float, default=0.15)
    parser.add_argument('--confidence_threshold', type=int, default=0)
    args = parser.parse_args()

    if args.intrinsic is None:
        args.intrinsic = os.path.join(args.frames_dir, 'intrinsic.json')

    print("=" * 60)
    print("Step 03 — RGB TSDF Meshing")
    print("=" * 60)

    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    print(f"  Camera: {intrinsic.width}x{intrinsic.height}")

    poses = load_trajectory_log(args.trajectory)
    print(f"  Poses: {len(poses)}")

    mesh = integrate_tsdf(
        args.frames_dir, intrinsic, poses,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        depth_min_m=args.depth_min,
        confidence_threshold=args.confidence_threshold,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    o3d.io.write_triangle_mesh(args.output, mesh)
    size_mb = os.path.getsize(args.output) / 1024**2
    print(f"\n✓ Saved: {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
