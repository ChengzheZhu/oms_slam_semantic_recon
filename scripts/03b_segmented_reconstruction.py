#!/usr/bin/env python3
"""
Segmented dense TSDF reconstruction for high-resolution meshes with limited GPU RAM.

This script:
1. Loads the full ORB_SLAM3 trajectory (globally consistent)
2. Splits trajectory and frames into N segments
3. Runs TSDF reconstruction on each segment independently
4. Merges all segment meshes (already aligned in same coordinate frame)

Advantages:
- Enables higher resolution (smaller voxel size) by processing segments
- No re-alignment needed (trajectory already global)
- Reduces peak GPU/RAM usage
"""

import open3d as o3d
import numpy as np
import json
import argparse
from pathlib import Path
import os
from tqdm import tqdm


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


def integrate_segment(color_files, depth_files, poses, intrinsic,
                      depth_scale, depth_max, voxel_size, segment_id):
    """
    Integrate a segment of frames into TSDF volume.

    Args:
        color_files: List of color image paths for this segment
        depth_files: List of depth image paths for this segment
        poses: List of 4x4 camera poses for this segment
        intrinsic: Open3D PinholeCameraIntrinsic
        depth_scale: Depth scale factor
        depth_max: Maximum depth in meters
        voxel_size: TSDF voxel size in meters
        segment_id: Segment identifier for logging

    Returns:
        mesh: Triangle mesh for this segment
    """
    n_frames = len(color_files)

    print(f"\n  Segment {segment_id}: Processing {n_frames} frames")

    # Create TSDF volume for this segment
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Integrate frames
    for i in tqdm(range(n_frames), desc=f"  Segment {segment_id}", leave=False):
        # Load RGB-D images
        color = o3d.io.read_image(color_files[i])
        depth = o3d.io.read_image(depth_files[i])

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

    # Extract mesh for this segment
    print(f"  Segment {segment_id}: Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"  Segment {segment_id}: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    return mesh


def merge_meshes(meshes, output_file):
    """
    Merge multiple meshes into one.

    Note: Meshes are already in the same coordinate frame (ORB_SLAM3 global coords),
    so we simply concatenate vertices and triangles.
    """
    print(f"\nMerging {len(meshes)} segment meshes...")

    if len(meshes) == 0:
        raise ValueError("No meshes to merge")

    if len(meshes) == 1:
        print("  Only one segment, no merging needed")
        return meshes[0]

    # Merge by concatenating
    merged = meshes[0]
    for i, mesh in enumerate(meshes[1:], start=2):
        merged += mesh
        print(f"  Merged {i}/{len(meshes)} meshes")

    print(f"\nMerged mesh:")
    print(f"  Total vertices: {len(merged.vertices):,}")
    print(f"  Total triangles: {len(merged.triangles):,}")

    # Optional: Remove duplicate vertices
    print("\n  Removing duplicate vertices...")
    merged = merged.remove_duplicated_vertices()
    print(f"  After deduplication: {len(merged.vertices):,} vertices")

    # Recompute normals
    print("  Recomputing normals...")
    merged.compute_vertex_normals()

    return merged


def main():
    parser = argparse.ArgumentParser(description="Segmented dense reconstruction")
    parser.add_argument('--frames_dir', type=str, required=True,
                       help='Directory containing frames (with color/ and depth/ subdirs)')
    parser.add_argument('--intrinsic', type=str, required=True,
                       help='Camera intrinsic JSON file')
    parser.add_argument('--trajectory', type=str, required=True,
                       help='ORB_SLAM3 trajectory in Open3D format')
    parser.add_argument('--output', type=str, required=True,
                       help='Output merged mesh file')
    parser.add_argument('--num_segments', type=int, default=4,
                       help='Number of segments to split sequence into')
    parser.add_argument('--voxel_size', type=float, default=0.005,
                       help='TSDF voxel size in meters (can use smaller values than non-segmented)')
    parser.add_argument('--depth_max', type=float, default=3.0,
                       help='Maximum depth in meters')
    parser.add_argument('--save_segments', action='store_true',
                       help='Save individual segment meshes')
    parser.add_argument('--overlap', type=int, default=10,
                       help='Number of overlapping frames between segments (for better merging)')

    args = parser.parse_args()

    print("="*80)
    print("Segmented Dense TSDF Reconstruction")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of segments: {args.num_segments}")
    print(f"  Voxel size: {args.voxel_size}m")
    print(f"  Depth max: {args.depth_max}m")
    print(f"  Frame overlap: {args.overlap} frames")

    # Load camera intrinsic
    print(f"\nLoading camera intrinsic: {args.intrinsic}")
    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    print(f"✓ Camera: {intrinsic.width}x{intrinsic.height}")

    # Load full trajectory
    print(f"\nLoading trajectory: {args.trajectory}")
    poses = load_trajectory_log(args.trajectory)
    print(f"✓ Loaded {len(poses)} camera poses")

    # Get frame lists
    color_files, depth_files = get_rgbd_file_lists(args.frames_dir)
    print(f"\nDataset:")
    print(f"  Color frames: {len(color_files)}")
    print(f"  Depth frames: {len(depth_files)}")

    # Match frames to poses
    n_frames = min(len(color_files), len(depth_files), len(poses))
    print(f"  Using {n_frames} frames total")

    color_files = color_files[:n_frames]
    depth_files = depth_files[:n_frames]
    poses = poses[:n_frames]

    # Calculate segment boundaries
    segment_size = n_frames // args.num_segments
    print(f"\n  Frames per segment: ~{segment_size}")

    # Process each segment
    segment_meshes = []
    segment_output_dir = os.path.join(os.path.dirname(args.output), 'segments')

    if args.save_segments:
        os.makedirs(segment_output_dir, exist_ok=True)

    print(f"\nProcessing {args.num_segments} segments:")
    print("="*80)

    for seg_id in range(args.num_segments):
        # Calculate segment range with overlap
        start_idx = max(0, seg_id * segment_size - args.overlap)

        if seg_id == args.num_segments - 1:
            # Last segment: go to end
            end_idx = n_frames
        else:
            end_idx = min(n_frames, (seg_id + 1) * segment_size + args.overlap)

        print(f"\nSegment {seg_id + 1}/{args.num_segments}: frames {start_idx} to {end_idx-1} ({end_idx - start_idx} frames)")

        # Extract segment data
        seg_color = color_files[start_idx:end_idx]
        seg_depth = depth_files[start_idx:end_idx]
        seg_poses = poses[start_idx:end_idx]

        # Integrate segment
        mesh = integrate_segment(
            seg_color, seg_depth, seg_poses,
            intrinsic, depth_scale, args.depth_max,
            args.voxel_size, seg_id + 1
        )

        # Save segment if requested
        if args.save_segments:
            segment_file = os.path.join(segment_output_dir, f'segment_{seg_id+1:02d}.ply')
            o3d.io.write_triangle_mesh(segment_file, mesh)
            print(f"  Segment {seg_id + 1}: Saved to {segment_file}")

        segment_meshes.append(mesh)

    # Merge all segments
    print("\n" + "="*80)
    merged_mesh = merge_meshes(segment_meshes, args.output)

    # Save merged mesh
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving merged mesh: {args.output}")
    o3d.io.write_triangle_mesh(args.output, merged_mesh)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"✓ Mesh saved ({file_size_mb:.1f} MB)")

    print("\n" + "="*80)
    print("Segmented Reconstruction Complete!")
    print("="*80)
    print(f"\nOutput: {args.output}")

    if args.save_segments:
        print(f"Segments: {segment_output_dir}/")

    print("\nVisualize with:")
    print(f"  python -c \"import open3d as o3d; mesh = o3d.io.read_triangle_mesh('{args.output}'); o3d.visualization.draw_geometries([mesh])\"")


if __name__ == "__main__":
    main()
