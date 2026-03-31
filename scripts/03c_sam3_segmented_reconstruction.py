#!/usr/bin/env python3
"""
SAM3-based Semantic Segmentation + 3D Reconstruction

Integrates SAM3 segmentation with ORB-SLAM3 dense reconstruction to create
semantically segmented 3D meshes (e.g., individual wall stones).

Pipeline:
1. Run SAM3 on each RGB frame to detect individual stones/segments
2. Project 2D segments to 3D using depth + camera poses
3. Build 3D mesh with per-vertex segment labels
4. Extract individual meshes for each detected segment
"""

import sys
# Remove system Python paths to avoid conflicts
sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib/python3.12')]

# Add SAM3 to path
sys.path.insert(0, '/home/chengzhe/projects/sam3')

import open3d as o3d
import numpy as np
import json
import argparse
from pathlib import Path
import os
from tqdm import tqdm
import torch
from PIL import Image

# SAM3 imports
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


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
                         if f.endswith(('.jpg', '.png'))])
    depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                         if f.endswith('.png')])

    return color_files, depth_files


def initialize_sam3(confidence_threshold=0.1):
    """Initialize SAM3 model."""
    print("\nInitializing SAM3...")
    print(f"  Confidence threshold: {confidence_threshold}")

    # Enable GPU acceleration if available
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


def segment_frame_with_sam3(image_path, processor, prompt="individual stone",
                             max_size_ratio=0.15, cache_path=None):
    """
    Segment a frame using SAM3.

    Args:
        image_path: Path to RGB image
        processor: SAM3 processor
        prompt: Text prompt for detection
        max_size_ratio: Max segment size as fraction of image
        cache_path: Path to cache segmentation mask

    Returns:
        segment_mask: HxW array with segment IDs (0=background, 1,2,3,... = segments)
        num_segments: Number of segments found
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        return data['mask'], data['num_segments']

    # Load image
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Run SAM3
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = inference_state["masks"]
    scores = inference_state["scores"]

    # Filter oversized masks
    img_area = image.size[0] * image.size[1]
    mask_areas = [mask.sum().item() / img_area for mask in masks]
    valid_indices = [i for i, area in enumerate(mask_areas) if area <= max_size_ratio]

    masks = [masks[i] for i in valid_indices]

    # Create segment mask image
    h, w = image.size[1], image.size[0]
    segment_mask = np.zeros((h, w), dtype=np.int32)

    for seg_id, mask in enumerate(masks, start=1):
        mask_np = mask.squeeze(0).cpu().numpy().astype(bool)
        segment_mask[mask_np] = seg_id

    num_segments = len(masks)

    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, {'mask': segment_mask, 'num_segments': num_segments})

    return segment_mask, num_segments


def generate_sam3_masks(color_files, processor, cache_dir, prompt="individual stone",
                        subsample=1, max_size_ratio=0.15):
    """Generate/load SAM3 segmentation masks for all frames."""
    os.makedirs(cache_dir, exist_ok=True)

    segment_masks = []
    total_segments = 0

    print(f"\nGenerating SAM3 segmentation masks...")
    print(f"  Prompt: '{prompt}'")
    print(f"  Subsample: every {subsample} frame(s)")
    print(f"  Max segment size: {max_size_ratio*100:.0f}% of image")

    for i, color_file in enumerate(tqdm(color_files, desc="Segmenting frames")):
        if i % subsample != 0:
            segment_masks.append(None)
            continue

        cache_path = os.path.join(cache_dir, f'mask_{i:06d}.npy')

        mask, num_segs = segment_frame_with_sam3(
            color_file, processor, prompt=prompt,
            max_size_ratio=max_size_ratio, cache_path=cache_path
        )

        segment_masks.append(mask)
        total_segments += num_segs

    processed_count = len([m for m in segment_masks if m is not None])
    print(f"\n  ✓ Processed {processed_count} frames")
    print(f"  ✓ Total segments across all frames: {total_segments}")

    return segment_masks


def integrate_with_sam3_segments(color_files, depth_files, segment_masks, poses,
                                  intrinsic, depth_scale, depth_max, voxel_size):
    """
    Integrate RGB-D frames with SAM3 segment labels.

    Returns:
        mesh: Reconstructed mesh
        vertex_segments: Per-vertex segment labels
    """
    n_frames = len(color_files)

    print(f"\nIntegrating {n_frames} frames with SAM3 segments...")

    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Collect labeled points
    all_points = []
    all_segments = []

    for i in tqdm(range(n_frames), desc="Integrating"):
        # Load RGB-D
        color = o3d.io.read_image(color_files[i])
        depth = o3d.io.read_image(depth_files[i])

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False
        )

        # Camera pose
        extrinsic = np.linalg.inv(poses[i])

        # Integrate into TSDF
        volume.integrate(rgbd, intrinsic, extrinsic)

        # Extract labeled points
        if segment_masks[i] is not None:
            # Create point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic, extrinsic
            )

            points = np.asarray(pcd.points)
            segment_mask = segment_masks[i]
            h, w = segment_mask.shape

            # Project 3D -> 2D to get segment labels
            points_cam = (extrinsic @ np.hstack([points, np.ones((len(points), 1))]).T).T[:, :3]

            u = (intrinsic.intrinsic_matrix[0, 0] * points_cam[:, 0] / points_cam[:, 2] +
                 intrinsic.intrinsic_matrix[0, 2])
            v = (intrinsic.intrinsic_matrix[1, 1] * points_cam[:, 1] / points_cam[:, 2] +
                 intrinsic.intrinsic_matrix[1, 2])

            u = np.round(u).astype(int)
            v = np.round(v).astype(int)

            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (points_cam[:, 2] > 0)
            segments = np.zeros(len(points), dtype=np.int32)
            segments[valid] = segment_mask[v[valid], u[valid]]

            all_points.append(points)
            all_segments.append(segments)

    # Extract mesh
    print("\n  Extracting mesh from TSDF...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"  ✓ Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # Assign segment labels to vertices
    print("\n  Assigning SAM3 segment labels to vertices...")

    if len(all_points) > 0:
        all_points = np.vstack(all_points)
        all_segments = np.hstack(all_segments)

        # KD-tree for nearest neighbor
        pcd_labeled = o3d.geometry.PointCloud()
        pcd_labeled.points = o3d.utility.Vector3dVector(all_points)
        kdtree = o3d.geometry.KDTreeFlann(pcd_labeled)

        vertices = np.asarray(mesh.vertices)
        vertex_segments = np.zeros(len(vertices), dtype=np.int32)

        for i, v in enumerate(tqdm(vertices, desc="  Labeling", leave=False)):
            [_, idx, _] = kdtree.search_knn_vector_3d(v, 1)
            vertex_segments[i] = all_segments[idx[0]]

        unique_segs = len(np.unique(vertex_segments[vertex_segments > 0]))
        print(f"  ✓ Labeled {len(vertex_segments)} vertices")
        print(f"  ✓ Unique segments: {unique_segs}")
    else:
        vertex_segments = None
        print("  ⚠ No segment labels available")

    return mesh, vertex_segments


def extract_segment_meshes(mesh, vertex_segments, min_vertices=100):
    """Extract individual meshes for each segment."""
    print(f"\nExtracting segment meshes (min {min_vertices} vertices)...")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    unique_segments = np.unique(vertex_segments[vertex_segments > 0])
    segment_meshes = []

    for seg_id in tqdm(unique_segments, desc="Extracting"):
        v_mask = vertex_segments == seg_id

        if np.sum(v_mask) < min_vertices:
            continue

        t_mask = np.all(v_mask[triangles], axis=1)
        if not np.any(t_mask):
            continue

        seg_mesh = o3d.geometry.TriangleMesh()
        v_idx = np.where(v_mask)[0]
        v_map = {old: new for new, old in enumerate(v_idx)}

        new_tri = np.array([[v_map[v] for v in t] for t in triangles[t_mask]])

        seg_mesh.vertices = o3d.utility.Vector3dVector(vertices[v_idx])
        seg_mesh.triangles = o3d.utility.Vector3iVector(new_tri)

        if colors is not None:
            seg_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[v_idx])

        seg_mesh.compute_vertex_normals()

        segment_meshes.append({
            'segment_id': int(seg_id),
            'mesh': seg_mesh,
            'num_vertices': len(v_idx)
        })

    print(f"  ✓ Extracted {len(segment_meshes)} segments")
    return segment_meshes


def main():
    parser = argparse.ArgumentParser(description="SAM3-based segmented 3D reconstruction")
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--intrinsic', type=str, required=True)
    parser.add_argument('--trajectory', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--segments_dir', type=str, required=True)
    parser.add_argument('--sam_cache_dir', type=str, default=None)
    parser.add_argument('--sam_prompt', type=str, default='individual stone')
    parser.add_argument('--sam_confidence', type=float, default=0.1)
    parser.add_argument('--sam_max_size', type=float, default=0.15)
    parser.add_argument('--frame_subsample', type=int, default=1)
    parser.add_argument('--voxel_size', type=float, default=0.005)
    parser.add_argument('--depth_max', type=float, default=3.0)
    parser.add_argument('--min_segment_vertices', type=int, default=100)

    args = parser.parse_args()

    if args.sam_cache_dir is None:
        args.sam_cache_dir = os.path.join(os.path.dirname(args.output), 'sam3_cache')

    print("="*80)
    print("SAM3-BASED SEGMENTED 3D RECONSTRUCTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  SAM prompt: '{args.sam_prompt}'")
    print(f"  SAM confidence: {args.sam_confidence}")
    print(f"  Voxel size: {args.voxel_size}m")
    print(f"  Frame subsample: {args.frame_subsample}")

    # Load intrinsic & trajectory
    print(f"\nLoading camera intrinsic: {args.intrinsic}")
    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    print(f"  ✓ {intrinsic.width}x{intrinsic.height}")

    print(f"\nLoading trajectory: {args.trajectory}")
    poses = load_trajectory_log(args.trajectory)
    print(f"  ✓ {len(poses)} poses")

    # Get frames
    color_files, depth_files = get_rgbd_file_lists(args.frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))
    print(f"\nDataset: {n_frames} frames")

    color_files = color_files[:n_frames]
    depth_files = depth_files[:n_frames]
    poses = poses[:n_frames]

    # Initialize SAM3
    processor = initialize_sam3(confidence_threshold=args.sam_confidence)

    # Generate SAM3 masks
    segment_masks = generate_sam3_masks(
        color_files, processor, args.sam_cache_dir,
        prompt=args.sam_prompt, subsample=args.frame_subsample,
        max_size_ratio=args.sam_max_size
    )

    # Integrate with segments
    mesh, vertex_segments = integrate_with_sam3_segments(
        color_files, depth_files, segment_masks, poses,
        intrinsic, depth_scale, args.depth_max, args.voxel_size
    )

    # Save full mesh
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nSaving full mesh: {args.output}")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"  ✓ {os.path.getsize(args.output) / (1024 * 1024):.1f} MB")

    # Extract & save segments
    if vertex_segments is not None:
        segment_meshes = extract_segment_meshes(
            mesh, vertex_segments, min_vertices=args.min_segment_vertices
        )

        os.makedirs(args.segments_dir, exist_ok=True)

        print(f"\nSaving {len(segment_meshes)} individual segments...")
        for seg in tqdm(segment_meshes, desc="Saving"):
            seg_file = os.path.join(args.segments_dir, f'segment_{seg["segment_id"]:04d}.ply')
            o3d.io.write_triangle_mesh(seg_file, seg['mesh'])

        print(f"  ✓ Saved to: {args.segments_dir}")

        # Summary
        print(f"\nTop 10 segments by size:")
        sorted_segs = sorted(segment_meshes, key=lambda x: x['num_vertices'], reverse=True)
        for i, seg in enumerate(sorted_segs[:10], 1):
            print(f"  {i}. segment_{seg['segment_id']:04d}: {seg['num_vertices']:,} vertices")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
