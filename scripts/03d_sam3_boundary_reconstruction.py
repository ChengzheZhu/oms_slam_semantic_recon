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
from collections import deque
# Remove system Python paths to avoid conflicts
sys.path = [p for p in sys.path if not p.startswith('/usr/local/lib/python3.12')]

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


def apply_depth_filter(depth_np, depth_scale, min_depth_m=0.15,
                       confidence_np=None, confidence_threshold=0):
    """
    Zero out unreliable depth pixels before TSDF integration.

    Always applied:
      - Zero-depth pixels (hardware holes / no return)
      - Pixels closer than min_depth_m (below D456 reliable stereo range)

    Optional (requires confidence stream extracted by 00_extract_frames.py):
      - confidence_threshold > 0: drop pixels where confidence < threshold
        D456 confidence values: 0=none, 1=low, 2=high
        Recommended: threshold=1 to drop only zero-confidence,
                     threshold=2 to keep high-confidence only
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


def integrate_tsdf(frames_dir, intrinsic, poses,
                   depth_scale=1000.0, depth_max=3.0, voxel_size=0.005,
                   depth_min_m=0.15, confidence_threshold=0):
    """
    Pass 1: pure TSDF integration over all frames. Returns the fused mesh.
    SAM3 is not involved here — boundary detection happens in pass 2 after
    the mesh is available.
    """
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))

    conf_dir = os.path.join(frames_dir, 'confidence')
    use_confidence = (confidence_threshold > 0) and os.path.isdir(conf_dir)
    if confidence_threshold > 0 and not use_confidence:
        print(f"  ⚠ confidence_threshold={confidence_threshold} requested but no confidence/ dir found — skipping")
    conf_files = (sorted([os.path.join(conf_dir, f) for f in os.listdir(conf_dir)
                          if f.endswith('.png')]) if use_confidence else [])

    print(f"\nTSDF pass: {n_frames} frames, voxel={voxel_size}m, depth=[{depth_min_m},{depth_max}]m")
    print(f"Depth filter: confidence_threshold={confidence_threshold}"
          f"{' (active)' if use_confidence else ' (no conf stream)'}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in tqdm(range(n_frames), desc="TSDF integration"):
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
        volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[i]))

    print("✓ TSDF integration complete")
    print("\nExtracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"  Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")
    return mesh


def compute_vertex_boundary_votes(mesh, frames_dir, poses, intrinsic, processor,
                                   frame_subsample=1, sam_prompt="individual stone",
                                   sam_max_size_ratio=0.15,
                                   boundary_vote_ratio=0.3,
                                   cache_dir=None):
    """
    Pass 2: project mesh vertices back into SAM3 frames and vote on seams.

    For each subsampled frame:
      - Run SAM3 → boundary mask (True where adjacent mask IDs differ)
      - Project all mesh vertices into this camera (vectorised)
      - stone_votes[v]++  for every visible vertex
      - boundary_votes[v]++ if that pixel is a seam pixel

    A vertex is classified as a seam if:
        boundary_votes[v] / stone_votes[v] > boundary_vote_ratio

    No cross-frame ID tracking is needed: the seam signal is purely local
    (two adjacent pixels in the same frame having different mask IDs).

    Returns:
        is_boundary: bool array [N_vertices]
    """
    color_files, _ = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(poses))
    frame_indices = list(range(0, n_frames, frame_subsample))

    vertices = np.asarray(mesh.vertices)  # [N, 3]
    N = len(vertices)
    verts_h = np.hstack([vertices, np.ones((N, 1))])  # [N, 4]

    stone_votes    = np.zeros(N, dtype=np.int32)
    boundary_votes = np.zeros(N, dtype=np.int32)

    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()
    W, H   = intrinsic.width, intrinsic.height

    print(f"\nBoundary voting pass: {len(frame_indices)} frames "
          f"(subsample={frame_subsample}), ratio_threshold={boundary_vote_ratio}")

    for idx in tqdm(frame_indices, desc="Boundary voting"):
        cache_path = (os.path.join(cache_dir, f"frame_{idx:06d}.npy")
                      if cache_dir else None)

        boundary_mask, _ = segment_frame_with_sam3(
            color_files[idx], processor,
            prompt=sam_prompt,
            max_size_ratio=sam_max_size_ratio,
            cache_path=cache_path
        )

        if boundary_mask is None:
            continue

        # Project vertices into this camera frame (vectorised)
        extrinsic = np.linalg.inv(poses[idx])        # world → camera
        verts_cam = (extrinsic @ verts_h.T).T        # [N, 4]
        z = verts_cam[:, 2]

        valid_z = z > 0.1
        x_v = verts_cam[valid_z, 0]
        y_v = verts_cam[valid_z, 1]
        z_v = z[valid_z]
        vidx = np.where(valid_z)[0]

        u = (fx * x_v / z_v + cx).astype(int)
        v = (fy * y_v / z_v + cy).astype(int)

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        vidx   = vidx[in_bounds]
        u_ok   = u[in_bounds]
        v_ok   = v[in_bounds]

        stone_votes[vidx] += 1

        is_bnd = boundary_mask[v_ok, u_ok]
        boundary_votes[vidx[is_bnd]] += 1

    ratio = boundary_votes / np.maximum(stone_votes, 1)
    is_boundary = ratio > boundary_vote_ratio

    n_seam = is_boundary.sum()
    print(f"✓ Voting complete: {n_seam:,} / {N:,} vertices classified as seam "
          f"({100*n_seam/N:.1f}%)")
    return is_boundary


def segment_mesh_with_boundaries(mesh, is_boundary, min_cluster_size=500):
    """
    Segment mesh using a pre-computed per-vertex boundary label.

    is_boundary[v] == True means vertex v sits on a stone seam and acts as
    a barrier in the BFS — edges that touch a boundary vertex are not
    traversed, so each connected stone interior becomes its own segment.

    Args:
        mesh: Triangle mesh to segment
        is_boundary: bool array [N_vertices] from compute_vertex_boundary_votes()
        min_cluster_size: Minimum triangles per segment

    Returns:
        segment_meshes: List of (segment_id, submesh) tuples
    """
    print("\nSegmenting mesh using boundary constraints...")

    vertices  = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if not is_boundary.any():
        print("  ⚠ No boundary vertices — returning full mesh as single segment")
        return [(1, mesh)]

    print(f"  Boundary vertices: {is_boundary.sum():,} / {len(vertices):,} "
          f"({100*is_boundary.sum()/len(vertices):.1f}%)")

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
        queue = deque([start_v])
        vertex_labels[start_v] = current_label

        while queue:
            v = queue.popleft()
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
                       help='SAM3 boundary voting runs on every Nth frame; TSDF uses all frames')

    # Depth quality filter
    parser.add_argument('--depth_min', type=float, default=0.15,
                       help='Minimum valid depth in metres (default: 0.15)')
    parser.add_argument('--confidence_threshold', type=int, default=0,
                       help='Confidence filter threshold (default: 0 = disabled). '
                            'Requires confidence/ frames from 00_extract_frames.py. '
                            'D456 values: 1=drop zero-confidence, 2=keep high-confidence only.')

    # Segmentation parameters
    parser.add_argument('--boundary_vote_ratio', type=float, default=0.3,
                       help='Fraction of observations that must be a seam pixel for a vertex '
                            'to be classified as boundary (default: 0.3)')
    parser.add_argument('--min_cluster_size', type=int, default=500,
                       help='Minimum triangles per segment')

    args = parser.parse_args()

    print("="*80)
    print("SAM3 Boundary-Based Segmented Reconstruction")
    print("="*80)

    # Load camera data
    print(f"\nLoading intrinsic: {args.intrinsic}")
    intrinsic, depth_scale = load_intrinsic(args.intrinsic)
    print(f"✓ Camera: {intrinsic.width}x{intrinsic.height}")

    print(f"\nLoading trajectory: {args.trajectory}")
    poses = load_trajectory_log(args.trajectory)
    print(f"✓ Loaded {len(poses)} poses")

    os.makedirs(args.segments_dir, exist_ok=True)
    cache_dir = os.path.join(args.segments_dir, 'boundary_masks')
    os.makedirs(cache_dir, exist_ok=True)

    # Pass 1: TSDF fusion (all frames, no SAM3)
    mesh = integrate_tsdf(
        args.frames_dir, intrinsic, poses,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        depth_min_m=args.depth_min,
        confidence_threshold=args.confidence_threshold,
    )

    # Save full mesh
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nSaving full mesh: {args.output}")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"✓ Saved ({os.path.getsize(args.output)/(1024**2):.1f} MB)")

    # Pass 2: SAM3 boundary voting (subsampled frames)
    processor = initialize_sam3(args.sam_confidence)

    is_boundary = compute_vertex_boundary_votes(
        mesh, args.frames_dir, poses, intrinsic, processor,
        frame_subsample=args.frame_subsample,
        sam_prompt=args.sam_prompt,
        sam_max_size_ratio=args.sam_max_size_ratio,
        boundary_vote_ratio=args.boundary_vote_ratio,
        cache_dir=cache_dir,
    )

    # Save boundary vertices as point cloud for inspection
    boundary_pts = np.asarray(mesh.vertices)[is_boundary]
    boundary_pcd = o3d.geometry.PointCloud()
    boundary_pcd.points = o3d.utility.Vector3dVector(boundary_pts)
    boundary_path = args.output.replace('.ply', '_boundaries.ply')
    o3d.io.write_point_cloud(boundary_path, boundary_pcd)
    print(f"✓ Boundary vertices saved: {boundary_path}")

    # Segment mesh using boundary vertex labels
    segment_meshes = segment_mesh_with_boundaries(
        mesh, is_boundary,
        min_cluster_size=args.min_cluster_size
    )

    print(f"\nSaving {len(segment_meshes)} segments to {args.segments_dir}/")
    for seg_id, submesh in segment_meshes:
        seg_path = os.path.join(args.segments_dir, f"segment_{seg_id:03d}.ply")
        o3d.io.write_triangle_mesh(seg_path, submesh)

    print(f"✓ Saved {len(segment_meshes)} segment meshes")

    print("\n" + "="*80)
    print("Reconstruction Complete!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Full mesh:  {args.output}")
    print(f"  Boundaries: {boundary_path}")
    print(f"  Segments:   {args.segments_dir}/segment_*.ply")


if __name__ == "__main__":
    main()
