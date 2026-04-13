#!/usr/bin/env python3
"""
Dense TSDF reconstruction using ORB_SLAM3 trajectory and Open3D.

This script:
1. Loads camera trajectory from ORB_SLAM3
2. Reads RGB-D frames from RealSense dataset
3. Integrates frames into TSDF volume using SLAM poses
4. Extracts final mesh
"""

import open3d as o3d
import numpy as np
import json
import argparse
from pathlib import Path
import os


def apply_depth_filter(depth_np, depth_scale, min_depth_m=0.15,
                       confidence_np=None, confidence_threshold=0):
    """
    Zero out unreliable depth pixels before TSDF integration.

    Always applied:
      - Zero-depth pixels (hardware holes / no return)
      - Pixels closer than min_depth_m (below D456 reliable stereo range)

    Optional (requires confidence/ frames from 00_extract_frames.py):
      - confidence_threshold > 0: drop pixels where confidence < threshold
        D456 values: 0=none, 1=low, 2=high
        Recommended: 1 to drop zero-confidence, 2 for high-confidence only
    """
    min_depth_raw = int(min_depth_m * depth_scale)
    invalid = (depth_np == 0) | (depth_np < min_depth_raw)

    if confidence_np is not None and confidence_threshold > 0:
        invalid |= (confidence_np < confidence_threshold)

    if invalid.any():
        depth_np = depth_np.copy()
        depth_np[invalid] = 0

    return depth_np


def load_trajectory_log(log_file):
    """Load Open3D trajectory log format."""
    poses = []

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            values = [float(x) for x in line.split()]
            if len(values) == 16:
                T = np.array(values).reshape(4, 4)
                poses.append(T)

    return poses


def load_intrinsic(intrinsic_file):
    """Load camera intrinsic parameters."""
    with open(intrinsic_file, 'r') as f:
        intr_data = json.load(f)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=intr_data['width'],
        height=intr_data['height'],
        fx=intr_data['intrinsic_matrix'][0],
        fy=intr_data['intrinsic_matrix'][4],
        cx=intr_data['intrinsic_matrix'][6],
        cy=intr_data['intrinsic_matrix'][7]
    )

    depth_scale = intr_data.get('depth_scale', 1000.0)

    return intrinsic, depth_scale


def get_rgbd_file_lists(frames_dir):
    """Get sorted lists of color and depth image files."""
    color_dir = os.path.join(frames_dir, 'color')
    depth_dir = os.path.join(frames_dir, 'depth')

    color_files = sorted([os.path.join(color_dir, f) for f in os.listdir(color_dir)
                         if f.endswith('.jpg') or f.endswith('.png')])
    depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                         if f.endswith('.png')])

    return color_files, depth_files


def integrate_rgbd_frames(frames_dir, intrinsic, poses, depth_scale=1000.0,
                         depth_max=3.0, voxel_size=0.02,
                         depth_min_m=0.15, confidence_threshold=0):
    """
    Integrate RGB-D frames into TSDF volume using ORB_SLAM3 poses.

    Args:
        frames_dir: Directory containing color/ and depth/ folders
        intrinsic: Open3D PinholeCameraIntrinsic
        poses: List of 4x4 camera pose matrices
        depth_scale: Depth scale factor (depth_value / depth_scale = meters)
        depth_max: Maximum depth in meters
        voxel_size: TSDF voxel size in meters
    """
    # Get frame files
    color_files, depth_files = get_rgbd_file_lists(frames_dir)

    print(f"\nDataset:")
    print(f"  Color frames: {len(color_files)}")
    print(f"  Depth frames: {len(depth_files)}")
    print(f"  Trajectory poses: {len(poses)}")

    # Match frames to poses
    n_frames = min(len(color_files), len(depth_files), len(poses))
    print(f"  Using {n_frames} frames for integration")

    # Confidence stream: auto-detect from extraction output
    conf_dir = os.path.join(frames_dir, 'confidence')
    use_confidence = (confidence_threshold > 0) and os.path.isdir(conf_dir)
    if confidence_threshold > 0 and not use_confidence:
        print(f"  ⚠ confidence_threshold={confidence_threshold} requested but no confidence/ dir found")
    conf_files = (sorted([os.path.join(conf_dir, f) for f in os.listdir(conf_dir)
                          if f.endswith('.png')]) if use_confidence else [])
    print(f"  Depth filter: min={depth_min_m}m, confidence_threshold={confidence_threshold}"
          f"{' (active)' if use_confidence else ' (no conf stream)'}")

    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    print(f"\nTSDF Volume:")
    print(f"  Voxel size: {voxel_size}m")
    print(f"  SDF truncation: {voxel_size * 4.0}m")
    print(f"  Depth max: {depth_max}m")
    print(f"  Depth scale: {depth_scale}")

    # Integrate frames
    print(f"\nIntegrating {n_frames} frames...")

    for i in range(n_frames):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Frame {i+1}/{n_frames} ({100*(i+1)/n_frames:.1f}%)")

        # Load RGB-D images and apply depth quality filter
        color = o3d.io.read_image(color_files[i])
        depth_np = np.asarray(o3d.io.read_image(depth_files[i]))
        conf_np = (np.asarray(o3d.io.read_image(conf_files[i]))
                   if use_confidence and i < len(conf_files) else None)
        depth_np = apply_depth_filter(depth_np, depth_scale,
                                      min_depth_m=depth_min_m,
                                      confidence_np=conf_np,
                                      confidence_threshold=confidence_threshold)
        depth = o3d.geometry.Image(depth_np.astype(np.uint16))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False
        )

        # Get camera pose (invert because ORB_SLAM3 gives camera-to-world)
        extrinsic = np.linalg.inv(poses[i])

        # Integrate into volume
        volume.integrate(rgbd, intrinsic, extrinsic)

    print("✓ Integration complete")

    return volume


def main():
    parser = argparse.ArgumentParser(description="Dense reconstruction with ORB_SLAM3 trajectory")
    parser.add_argument('--frames_dir', type=str,
                       
                       help='Directory containing frames (with color/ and depth/ subdirs)')
    parser.add_argument('--intrinsic', type=str,
                       
                       help='Camera intrinsic JSON file')
    parser.add_argument('--trajectory', type=str,
                       
                       help='ORB_SLAM3 trajectory in Open3D format')
    parser.add_argument('--output', type=str,
                       
                       help='Output mesh file')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                       help='TSDF voxel size in meters')
    parser.add_argument('--depth_max', type=float, default=3.0,
                       help='Maximum depth in meters')
    parser.add_argument('--depth_min', type=float, default=0.15,
                       help='Minimum valid depth in metres (default: 0.15)')
    parser.add_argument('--confidence_threshold', type=int, default=0,
                       help='Confidence filter threshold (default: 0 = disabled). '
                            'Requires confidence/ frames from 00_extract_frames.py. '
                            'D456 values: 1=drop zero-confidence, 2=keep high-confidence only.')

    args = parser.parse_args()

    print("="*80)
    print("Dense TSDF Reconstruction with ORB_SLAM3 Trajectory")
    print("="*80)

    # Load camera intrinsic
    print(f"\nLoading camera intrinsic: {args.intrinsic}")
    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    print(f"✓ Camera: {intrinsic.width}x{intrinsic.height}")
    print(f"  fx={intrinsic.get_focal_length()[0]:.2f}, fy={intrinsic.get_focal_length()[1]:.2f}")
    print(f"  cx={intrinsic.get_principal_point()[0]:.2f}, cy={intrinsic.get_principal_point()[1]:.2f}")

    # Load trajectory
    print(f"\nLoading trajectory: {args.trajectory}")
    poses = load_trajectory_log(args.trajectory)
    print(f"✓ Loaded {len(poses)} camera poses")

    # Integrate frames
    volume = integrate_rgbd_frames(
        args.frames_dir,
        intrinsic,
        poses,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        depth_min_m=args.depth_min,
        confidence_threshold=args.confidence_threshold,
    )

    # Extract mesh
    print("\nExtracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Triangles: {len(mesh.triangles):,}")

    # Save mesh
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving mesh: {args.output}")
    o3d.io.write_triangle_mesh(args.output, mesh)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"✓ Mesh saved ({file_size_mb:.1f} MB)")

    print("\n" + "="*80)
    print("Dense Reconstruction Complete!")
    print("="*80)
    print(f"\nOutput: {args.output}")
    print("\nVisualize with:")
    print(f"  python -c \"import open3d as o3d; mesh = o3d.io.read_triangle_mesh('{args.output}'); o3d.visualization.draw_geometries([mesh])\"")


if __name__ == "__main__":
    main()
