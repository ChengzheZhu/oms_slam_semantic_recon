#!/usr/bin/env python3
"""
SAM3 Boundary-Based Segmentation + 3D Reconstruction

Key difference from 03c: Instead of propagating inconsistent segment IDs,
we detect object BOUNDARIES in 2D and use them to guide 3D mesh segmentation.

Pipeline:
1. Run SAM3 on each frame to detect objects (local IDs, not tracked)
2. Extract boundary pixels between segments
3. Project boundary information to 3D
4. Build mesh and accumulate boundary confidence
5. Segment mesh using 3D clustering with boundary constraints
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
from scipy import ndimage
from sklearn.cluster import DBSCAN

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


def detect_segment_boundaries(segment_mask, dilation_size=2):
    """
    Detect boundaries between segments.

    Args:
        segment_mask: HxW array with segment IDs (0=background, 1,2,3...=segments)
        dilation_size: Size of boundary region (in pixels)

    Returns:
        boundary_mask: HxW boolean array (True = boundary pixel)
    """
    if segment_mask is None or segment_mask.max() == 0:
        return np.zeros_like(segment_mask, dtype=bool)

    # Detect edges using morphological gradient
    # (dilation - erosion) highlights boundaries
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    dilated = ndimage.grey_dilation(segment_mask, footprint=struct)
    eroded = ndimage.grey_erosion(segment_mask, footprint=struct)

    # Boundary = pixels where dilation != erosion (ID changes)
    boundaries = (dilated != eroded)

    # Dilate boundaries to create a thicker boundary region
    if dilation_size > 1:
        for _ in range(dilation_size - 1):
            boundaries = ndimage.binary_dilation(boundaries, structure=struct)

    return boundaries


def segment_frame_with_sam3(image_path, processor, prompt="individual stone",
                             max_size_ratio=0.15, cache_path=None):
    """
    Segment a frame using SAM3 and extract boundaries.

    Returns:
        boundary_mask: HxW boolean array (True = boundary between objects)
        num_segments: Number of segments detected
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        return data['boundary_mask'], data['num_segments']

    # Load image
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Run SAM3
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = inference_state["masks"]

    # Filter oversized masks (likely entire wall)
    img_area = image.size[0] * image.size[1]
    mask_areas = [mask.sum().item() / img_area for mask in masks]
    valid_indices = [i for i, area in enumerate(mask_areas) if area <= max_size_ratio]

    masks = [masks[i] for i in valid_indices]
    num_segments = len(masks)

    # Create temporary segment mask (IDs are frame-local, not consistent across frames)
    h, w = image.size[1], image.size[0]
    segment_mask = np.zeros((h, w), dtype=np.int32)

    for seg_id, mask in enumerate(masks, start=1):
        mask_np = mask.squeeze(0).cpu().numpy().astype(bool)
        segment_mask[mask_np] = seg_id

    # Extract boundaries (this is what we actually care about)
    boundary_mask = detect_segment_boundaries(segment_mask, dilation_size=3)

    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, {
            'boundary_mask': boundary_mask,
            'num_segments': num_segments
        })

    return boundary_mask, num_segments


def integrate_with_boundary_detection(frames_dir, intrinsic, poses, processor,
                                      depth_scale=1000.0, depth_max=3.0, voxel_size=0.005,
                                      frame_subsample=1, sam_prompt="individual stone",
                                      sam_confidence=0.1, sam_max_size_ratio=0.15,
                                      cache_dir=None):
    """
    Integrate RGB-D frames with boundary detection.

    Returns:
        mesh: 3D mesh
        boundary_points: Point cloud of boundary locations (for mesh segmentation)
    """
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))

    # Subsample frames
    frame_indices = list(range(0, n_frames, frame_subsample))
    print(f"\nDataset: {len(color_files)} color, {len(depth_files)} depth, {len(poses)} poses")
    print(f"Using {len(frame_indices)} frames (subsample={frame_subsample})")

    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    print(f"\nTSDF: voxel={voxel_size}m, trunc={voxel_size*4.0}m, depth_max={depth_max}m")

    # Accumulate boundary points across all frames
    all_boundary_points = []
    all_boundary_colors = []

    print(f"\nProcessing {len(frame_indices)} frames...")

    for idx in tqdm(frame_indices, desc="Integrating frames"):
        # Segment frame with SAM3
        cache_path = None
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"frame_{idx:06d}.npy")

        boundary_mask, num_segs = segment_frame_with_sam3(
            color_files[idx], processor,
            prompt=sam_prompt,
            max_size_ratio=sam_max_size_ratio,
            cache_path=cache_path
        )

        # Load RGB-D images
        color = o3d.io.read_image(color_files[idx])
        depth = o3d.io.read_image(depth_files[idx])

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False
        )

        # Get camera pose
        extrinsic = np.linalg.inv(poses[idx])

        # Integrate into TSDF (standard reconstruction)
        volume.integrate(rgbd, intrinsic, extrinsic)

        # Extract boundary points in 3D
        if boundary_mask is not None and boundary_mask.any():
            # Create point cloud from RGBD
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic, extrinsic
            )

            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            # Boundary mask is HxW, flatten to match point cloud
            h, w = boundary_mask.shape
            boundary_flat = boundary_mask.flatten()

            # Filter to boundary points only
            if len(boundary_flat) == len(points):
                boundary_pts = points[boundary_flat]
                boundary_cols = colors[boundary_flat]

                if len(boundary_pts) > 0:
                    all_boundary_points.append(boundary_pts)
                    all_boundary_colors.append(boundary_cols)

    print("✓ Integration complete")

    # Extract mesh
    print("\nExtracting mesh from TSDF...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"  Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")

    # Combine all boundary points
    if all_boundary_points:
        boundary_points = np.vstack(all_boundary_points)
        boundary_colors = np.vstack(all_boundary_colors)

        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_points)
        boundary_pcd.colors = o3d.utility.Vector3dVector(boundary_colors)

        print(f"  Boundary points: {len(boundary_points):,}")
    else:
        boundary_pcd = o3d.geometry.PointCloud()
        print("  ⚠ No boundary points detected")

    return mesh, boundary_pcd


def segment_mesh_with_boundaries(mesh, boundary_pcd, boundary_distance_threshold=0.01,
                                 min_cluster_size=500):
    """
    Segment mesh using boundary constraints.

    Strategy:
    1. For each mesh vertex, check if it's near a boundary point
    2. Use region growing / clustering that respects boundaries
    3. Extract connected components as separate segments

    Args:
        mesh: Triangle mesh to segment
        boundary_pcd: Point cloud marking boundary locations
        boundary_distance_threshold: Distance to boundary to consider "on boundary"
        min_cluster_size: Minimum triangles per segment

    Returns:
        segment_meshes: List of (segment_id, submesh) tuples
    """
    print("\nSegmenting mesh using boundary constraints...")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Build KD-tree for boundary points
    if len(boundary_pcd.points) == 0:
        print("  ⚠ No boundaries - returning full mesh as single segment")
        return [(1, mesh)]

    boundary_tree = o3d.geometry.KDTreeFlann(boundary_pcd)

    # Mark vertices as boundary or interior
    print("  Classifying vertices...")
    is_boundary = np.zeros(len(vertices), dtype=bool)

    for i, v in enumerate(vertices):
        [k, idx, dist] = boundary_tree.search_radius_vector_3d(v, boundary_distance_threshold)
        if k > 0:  # Close to boundary
            is_boundary[i] = True

    print(f"  Boundary vertices: {is_boundary.sum():,} / {len(vertices):,} ({100*is_boundary.sum()/len(vertices):.1f}%)")

    # Build adjacency graph (edges that DON'T cross boundaries)
    print("  Building adjacency graph...")
    adjacency = [set() for _ in range(len(vertices))]

    for tri in triangles:
        v0, v1, v2 = tri

        # Add edges only if neither vertex is on boundary
        # (boundaries act as barriers)
        if not is_boundary[v0] and not is_boundary[v1]:
            adjacency[v0].add(v1)
            adjacency[v1].add(v0)
        if not is_boundary[v1] and not is_boundary[v2]:
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
        if not is_boundary[v2] and not is_boundary[v0]:
            adjacency[v2].add(v0)
            adjacency[v0].add(v2)

    # Connected component analysis (region growing)
    print("  Finding connected components...")
    vertex_labels = -np.ones(len(vertices), dtype=np.int32)  # -1 = unlabeled
    current_label = 0

    for start_v in range(len(vertices)):
        if vertex_labels[start_v] >= 0:  # Already labeled
            continue
        if is_boundary[start_v]:  # Skip boundary vertices
            continue

        # BFS from this vertex
        queue = [start_v]
        vertex_labels[start_v] = current_label

        while queue:
            v = queue.pop(0)
            for neighbor in adjacency[v]:
                if vertex_labels[neighbor] < 0:  # Unlabeled
                    vertex_labels[neighbor] = current_label
                    queue.append(neighbor)

        current_label += 1

    num_segments = current_label
    print(f"  Found {num_segments} segments")

    # Extract submeshes for each segment
    segment_meshes = []

    for seg_id in range(num_segments):
        # Find triangles where ALL vertices belong to this segment
        mask = np.all(vertex_labels[triangles] == seg_id, axis=1)
        seg_triangles = triangles[mask]

        if len(seg_triangles) < min_cluster_size:
            continue  # Too small

        # Extract submesh
        submesh = o3d.geometry.TriangleMesh()
        submesh.vertices = mesh.vertices
        submesh.triangles = o3d.utility.Vector3iVector(seg_triangles)
        submesh.vertex_colors = mesh.vertex_colors
        submesh.vertex_normals = mesh.vertex_normals

        # Remove unreferenced vertices
        submesh.remove_unreferenced_vertices()
        submesh.compute_vertex_normals()

        segment_meshes.append((seg_id + 1, submesh))
        print(f"    Segment {seg_id+1}: {len(submesh.vertices):,} vertices, {len(submesh.triangles):,} triangles")

    return segment_meshes


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 boundary-based segmentation + 3D reconstruction"
    )
    parser.add_argument('--frames_dir', type=str, required=True,
                       help='Directory with color/ and depth/ subdirs')
    parser.add_argument('--intrinsic', type=str, required=True,
                       help='Camera intrinsic JSON')
    parser.add_argument('--trajectory', type=str, required=True,
                       help='ORB_SLAM3 trajectory (Open3D format)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output mesh path')
    parser.add_argument('--segments_dir', type=str, required=True,
                       help='Directory for individual segment meshes')

    # SAM3 parameters
    parser.add_argument('--sam_prompt', type=str, default="individual stone",
                       help='SAM3 text prompt')
    parser.add_argument('--sam_confidence', type=float, default=0.1,
                       help='SAM3 confidence threshold')
    parser.add_argument('--sam_max_size_ratio', type=float, default=0.15,
                       help='Max segment size as fraction of image')

    # Reconstruction parameters
    parser.add_argument('--voxel_size', type=float, default=0.005,
                       help='TSDF voxel size (meters)')
    parser.add_argument('--depth_max', type=float, default=3.0,
                       help='Max depth (meters)')
    parser.add_argument('--frame_subsample', type=int, default=1,
                       help='Process every Nth frame')

    # Segmentation parameters
    parser.add_argument('--boundary_threshold', type=float, default=0.01,
                       help='Distance threshold for boundary detection (meters)')
    parser.add_argument('--min_cluster_size', type=int, default=500,
                       help='Minimum triangles per segment')

    args = parser.parse_args()

    print("="*80)
    print("SAM3 Boundary-Based Segmented Reconstruction")
    print("="*80)

    # Initialize SAM3
    processor = initialize_sam3(args.sam_confidence)

    # Load camera data
    print(f"\nLoading intrinsic: {args.intrinsic}")
    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    print(f"✓ Camera: {intrinsic.width}x{intrinsic.height}")

    print(f"\nLoading trajectory: {args.trajectory}")
    poses = load_trajectory_log(args.trajectory)
    print(f"✓ Loaded {len(poses)} poses")

    # Setup cache directory
    cache_dir = os.path.join(args.segments_dir, 'boundary_masks')
    os.makedirs(cache_dir, exist_ok=True)

    # Integrate with boundary detection
    mesh, boundary_pcd = integrate_with_boundary_detection(
        args.frames_dir, intrinsic, poses, processor,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        frame_subsample=args.frame_subsample,
        sam_prompt=args.sam_prompt,
        sam_confidence=args.sam_confidence,
        sam_max_size_ratio=args.sam_max_size_ratio,
        cache_dir=cache_dir
    )

    # Save full mesh
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nSaving full mesh: {args.output}")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"✓ Saved ({os.path.getsize(args.output)/(1024**2):.1f} MB)")

    # Save boundary point cloud for debugging
    boundary_path = args.output.replace('.ply', '_boundaries.ply')
    o3d.io.write_point_cloud(boundary_path, boundary_pcd)
    print(f"✓ Boundary points: {boundary_path}")

    # Segment mesh using boundaries
    segment_meshes = segment_mesh_with_boundaries(
        mesh, boundary_pcd,
        boundary_distance_threshold=args.boundary_threshold,
        min_cluster_size=args.min_cluster_size
    )

    # Save individual segments
    os.makedirs(args.segments_dir, exist_ok=True)
    print(f"\nSaving {len(segment_meshes)} segments to {args.segments_dir}/")

    for seg_id, submesh in segment_meshes:
        seg_path = os.path.join(args.segments_dir, f"segment_{seg_id:03d}.ply")
        o3d.io.write_triangle_mesh(seg_path, submesh)

    print(f"✓ Saved {len(segment_meshes)} segment meshes")

    print("\n" + "="*80)
    print("Reconstruction Complete!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Full mesh: {args.output}")
    print(f"  Boundaries: {boundary_path}")
    print(f"  Segments: {args.segments_dir}/segment_*.ply")


if __name__ == "__main__":
    main()
