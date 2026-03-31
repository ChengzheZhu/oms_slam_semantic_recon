#!/usr/bin/env python3
"""
SAM3 Voxel-Based Labeling + 3D Reconstruction

Key difference from 03d: Instead of detecting boundaries post-hoc,
we label each voxel during TSDF integration with a distance-based scoring strategy.

Scoring Strategy:
- Points OUTSIDE masks: Negative punishment score (-1.0)
- Points NEAR mask boundaries: Lower positive scores (0.0 - 0.5)
- Points WELL WITHIN mask boundaries: Highest scores (0.5 - 1.0)

Pipeline:
1. Run SAM3 on each frame, filter by min size, shrink masks for clearance
2. Compute distance-based scores for each pixel
3. During TSDF integration, accumulate scores per voxel
4. After integration, classify voxels as "rock" or "seam" based on average scores
5. Extract separate meshes
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
from collections import defaultdict
import cv2
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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


def create_scored_mask_image(image_path, processor, prompt="individual stone",
                             max_size_ratio=0.15, min_size_ratio=0.001,
                             shrink_pixels=3, boundary_distance=10,
                             cache_path=None):
    """
    Create scored mask image with distance-based scoring.

    Scoring:
    - Outside all masks: -1.0 (punishment)
    - At mask boundary: 0.0
    - Inside mask, near boundary: 0.0 to 0.8 (linear with distance)
    - Deep inside mask (>boundary_distance): 1.0

    Args:
        min_size_ratio: Minimum mask size (fraction of image area)
        shrink_pixels: Erode masks by N pixels for clearance
        boundary_distance: Distance from boundary to reach max score (pixels)

    Returns:
        score_image: HxW array with scores from -1.0 to 1.0
        num_segments: Number of valid segments
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        return data['score_image'], data['num_segments']

    # Load image
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Run SAM3
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = inference_state["masks"]

    # Filter by size (remove too large and too small masks)
    img_area = image.size[0] * image.size[1]
    mask_areas = [mask.sum().item() / img_area for mask in masks]
    valid_indices = [i for i, area in enumerate(mask_areas)
                    if min_size_ratio <= area <= max_size_ratio]

    if len(valid_indices) == 0:
        # No valid masks - entire image gets punishment score
        h, w = image.size[1], image.size[0]
        score_image = np.full((h, w), -1.0, dtype=np.float32)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, {'score_image': score_image, 'num_segments': 0})

        return score_image, 0

    masks = [masks[i] for i in valid_indices]
    num_segments = len(masks)

    # Initialize score image with punishment score
    h, w = image.size[1], image.size[0]
    score_image = np.full((h, w), -1.0, dtype=np.float32)

    # Process each mask
    for mask in masks:
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        # Shrink mask for clearance between adjacent masks
        if shrink_pixels > 0:
            kernel = np.ones((shrink_pixels*2+1, shrink_pixels*2+1), np.uint8)
            mask_np = cv2.erode(mask_np, kernel, iterations=1)

        # Skip if mask disappeared after erosion
        if mask_np.sum() == 0:
            continue

        # Compute distance transform (distance to nearest 0 pixel)
        # For pixels inside mask, this gives distance to boundary
        dist_inside = ndimage.distance_transform_edt(mask_np)

        # Compute score based on distance from boundary with non-linear boost
        # dist=0 (at boundary) -> score=0.0
        # dist=boundary_distance (deep inside) -> score=1.0
        # Use square root to boost scores for pixels well inside
        normalized_dist = np.clip(dist_inside / boundary_distance, 0.0, 1.0)
        mask_scores = np.sqrt(normalized_dist)  # Non-linear boost

        # Alternative: Use power function for even stronger boost
        # mask_scores = normalized_dist ** 0.5  # Same as sqrt
        # mask_scores = normalized_dist ** 0.4  # Stronger boost

        # Update score image (only where this mask is present)
        # Use max() to handle overlapping masks (take best score)
        mask_bool = mask_np > 0
        score_image[mask_bool] = np.maximum(score_image[mask_bool], mask_scores[mask_bool])

    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, {
            'score_image': score_image,
            'num_segments': num_segments
        })

    return score_image, num_segments


class VoxelMaskTracker:
    """
    Tracks mask scores per voxel during TSDF integration.

    Each voxel accumulates:
    - Total score (sum of -1.0 to 1.0 values)
    - Number of observations

    Average = total / count (-1.0 to 1.0)
    """

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.voxel_data = defaultdict(lambda: {'sum': 0.0, 'count': 0})

    def _point_to_voxel(self, point):
        """Convert 3D point to voxel grid coordinates."""
        return tuple((point / self.voxel_size).astype(np.int32))

    def add_observation(self, point, score):
        """Add a score observation for a 3D point."""
        voxel_idx = self._point_to_voxel(point)
        self.voxel_data[voxel_idx]['sum'] += score
        self.voxel_data[voxel_idx]['count'] += 1

    def get_average_score(self, point):
        """Get average mask score for a 3D point."""
        voxel_idx = self._point_to_voxel(point)
        data = self.voxel_data.get(voxel_idx)
        if data is None or data['count'] == 0:
            return -1.0  # Default to punishment score if no observations
        return data['sum'] / data['count']

    def build_kdtree(self):
        """
        Build KD-tree for nearest neighbor lookups.

        Returns:
            kdtree: scipy cKDTree
            voxel_centers: Nx3 array of voxel center positions
            scores: N array of average scores for each voxel
        """
        print(f"\n  Building KD-tree for {len(self.voxel_data):,} tracked voxels...")

        # Extract voxel indices and convert to 3D positions
        voxel_indices = list(self.voxel_data.keys())
        voxel_centers = np.array(voxel_indices, dtype=np.float32) * self.voxel_size

        # Compute average scores
        scores = np.array([
            data['sum'] / data['count']
            for data in self.voxel_data.values()
        ], dtype=np.float32)

        # Build KD-tree
        print(f"  Creating KD-tree...")
        kdtree = cKDTree(voxel_centers)
        print(f"  ✓ KD-tree ready")

        return kdtree, voxel_centers, scores

    def get_all_scores_knn(self, points, max_distance=None, grid_size=0.5):
        """
        Get scores using k-nearest neighbor lookup with spatial partitioning.

        Uses spatial partitioning to avoid loading all 58M voxels into memory at once.

        Args:
            points: Nx3 array of query points
            max_distance: Maximum distance to search (default: 3x voxel size)
            grid_size: Size of spatial partition grid (meters)

        Returns:
            scores: N array of scores
        """
        if max_distance is None:
            max_distance = self.voxel_size * 3  # Search within 3 voxels

        print(f"  Querying {len(points):,} vertices using KNN (max_dist={max_distance*1000:.1f}mm)...")
        print(f"  Using spatial partitioning (grid_size={grid_size}m) to reduce memory usage...")

        # Build spatial grid WITHOUT extracting all voxels at once
        print(f"  Building spatial partitions from {len(self.voxel_data):,} voxels...")
        grid_cells = defaultdict(lambda: {'voxel_indices': [], 'voxel_centers': [], 'scores': []})

        # Incrementally add voxels to grid cells (never load all at once)
        for voxel_idx, data in tqdm(self.voxel_data.items(), desc="  Partitioning voxels", mininterval=1.0):
            # Convert voxel index to 3D position
            voxel_center = np.array(voxel_idx, dtype=np.float32) * self.voxel_size

            # Determine which grid cell this voxel belongs to
            grid_coord = tuple((voxel_center / grid_size).astype(np.int32))

            # Add to grid cell
            grid_cells[grid_coord]['voxel_indices'].append(voxel_idx)
            grid_cells[grid_coord]['voxel_centers'].append(voxel_center)
            grid_cells[grid_coord]['scores'].append(data['sum'] / data['count'])

        print(f"  Created {len(grid_cells):,} spatial partitions")

        # Convert lists to numpy arrays for each grid cell
        print(f"  Converting to numpy arrays...")
        for grid_coord in tqdm(grid_cells.keys(), desc="  Converting", mininterval=1.0):
            grid_cells[grid_coord]['voxel_centers'] = np.array(grid_cells[grid_coord]['voxel_centers'], dtype=np.float32)
            grid_cells[grid_coord]['scores'] = np.array(grid_cells[grid_coord]['scores'], dtype=np.float32)

        # Query vertices
        all_scores = np.full(len(points), -1.0, dtype=np.float32)

        # Determine which grid cells each point belongs to
        point_grid_coords = (points / grid_size).astype(np.int32)

        print(f"  Searching nearest neighbors (vectorized batching)...")

        # Group vertices by their grid cell for batch processing
        vertex_by_grid = defaultdict(list)
        for i, grid_coord in enumerate(point_grid_coords):
            vertex_by_grid[tuple(grid_coord)].append(i)

        print(f"  Vertices grouped into {len(vertex_by_grid):,} grid cells")

        # Process each grid cell
        for grid_coord, vertex_indices in tqdm(vertex_by_grid.items(), desc="  KNN lookup"):
            # Get all vertices in this grid cell
            batch_points = points[vertex_indices]

            # Collect voxels from this cell and 26 neighbors
            nearby_centers_list = []
            nearby_scores_list = []

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_cell = (grid_coord[0] + dx, grid_coord[1] + dy, grid_coord[2] + dz)
                        if neighbor_cell in grid_cells:
                            cell_data = grid_cells[neighbor_cell]
                            nearby_centers_list.append(cell_data['voxel_centers'])
                            nearby_scores_list.append(cell_data['scores'])

            if not nearby_centers_list:
                continue  # No voxels nearby, keep default -1.0

            # Concatenate all nearby voxels
            nearby_centers = np.vstack(nearby_centers_list)  # Shape: (N_voxels, 3)
            nearby_scores = np.concatenate(nearby_scores_list)  # Shape: (N_voxels,)

            # Vectorized distance computation for all vertices in batch
            # batch_points: (N_vertices, 3), nearby_centers: (N_voxels, 3)
            # Result: (N_vertices, N_voxels) distance matrix
            diffs = batch_points[:, np.newaxis, :] - nearby_centers[np.newaxis, :, :]
            distances = np.linalg.norm(diffs, axis=2)

            # Find nearest voxel for each vertex
            min_indices = np.argmin(distances, axis=1)
            min_distances = distances[np.arange(len(batch_points)), min_indices]

            # Assign scores where distance < max_distance
            valid_mask = min_distances < max_distance
            valid_scores = nearby_scores[min_indices[valid_mask]]

            # Update scores for valid vertices
            valid_vertex_indices = np.array(vertex_indices)[valid_mask]
            all_scores[valid_vertex_indices] = valid_scores

        return all_scores

    def get_all_scores(self, points):
        """
        Get average mask scores for multiple points (vectorized for speed).

        Processes points in batches to avoid memory issues with large meshes.
        """
        print(f"  Querying scores for {len(points):,} vertices...")

        batch_size = 100000  # Process 100k vertices at a time
        all_scores = np.zeros(len(points), dtype=np.float32)

        for i in tqdm(range(0, len(points), batch_size), desc="  Computing vertex scores"):
            batch = points[i:i+batch_size]

            # Vectorized voxel coordinate computation
            voxel_coords = (batch / self.voxel_size).astype(np.int32)

            # Look up scores for each voxel
            for j, voxel_idx in enumerate(voxel_coords):
                voxel_tuple = tuple(voxel_idx)
                data = self.voxel_data.get(voxel_tuple)
                if data is None or data['count'] == 0:
                    all_scores[i + j] = -1.0
                else:
                    all_scores[i + j] = data['sum'] / data['count']

        return all_scores


def visualize_score_image_2d(image_path, score_image, output_path, frame_idx):
    """
    Create 2D visualization of score image for debugging.

    Saves a 4-panel figure:
    - Top left: Original RGB image
    - Top right: Score heatmap
    - Bottom left: Mask overlay (scores > 0)
    - Bottom right: Score histogram
    """
    # Load original image
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_np = np.array(image)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Frame {frame_idx:06d} Score Visualization', fontsize=14, fontweight='bold')

    # Top left: Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')

    # Top right: Score heatmap
    # Custom colormap: purple (-1) -> blue (0) -> green (0.5) -> red (1)
    colors_list = ['purple', 'blue', 'cyan', 'green', 'yellow', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('score_cmap', colors_list, N=n_bins)

    im = axes[0, 1].imshow(score_image, cmap=cmap, vmin=-1.0, vmax=1.0)
    axes[0, 1].set_title('Score Heatmap')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], label='Score', fraction=0.046)

    # Bottom left: Mask overlay (scores > 0)
    mask_binary = score_image > 0
    overlay = image_np.copy()
    overlay[mask_binary] = (overlay[mask_binary] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f'Mask Overlay (score > 0): {mask_binary.sum():,} pixels')
    axes[1, 0].axis('off')

    # Bottom right: Score histogram
    axes[1, 1].hist(score_image.flatten(), bins=50, range=(-1, 1), edgecolor='black')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=1.5, label='Boundary (0)', alpha=0.7)
    axes[1, 1].axvline(x=0.5, color='orange', linestyle='--', linewidth=1.5, label='Mid-score (0.5)', alpha=0.7)
    axes[1, 1].axvline(x=0.8, color='darkgreen', linestyle='--', linewidth=1.5, label='High-score (0.8)', alpha=0.7)
    axes[1, 1].legend(fontsize=8)

    # Add statistics text
    stats_text = f'Mean: {score_image.mean():.3f}\n'
    stats_text += f'Std: {score_image.std():.3f}\n'
    stats_text += f'Min: {score_image.min():.3f}\n'
    stats_text += f'Max: {score_image.max():.3f}\n'
    stats_text += f'Pixels > 0: {(score_image > 0).sum():,} ({100*(score_image > 0).sum()/score_image.size:.1f}%)\n'
    stats_text += f'Pixels > 0.5: {(score_image > 0.5).sum():,} ({100*(score_image > 0.5).sum()/score_image.size:.1f}%)\n'
    stats_text += f'Pixels > 0.8: {(score_image > 0.8).sum():,} ({100*(score_image > 0.8).sum()/score_image.size:.1f}%)'
    axes[1, 1].text(0.98, 0.98, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def precompute_sam3_scores(frames_dir, processor, frame_indices,
                           sam_prompt="individual stone", sam_confidence=0.1,
                           sam_max_size_ratio=0.15, sam_min_size_ratio=0.001,
                           sam_shrink_pixels=3, sam_boundary_distance=10,
                           cache_dir=None, force_recompute=False,
                           debug_vis=False, debug_vis_stride=10):
    """
    PASS 1: Precompute SAM3 score images for all frames.

    Args:
        force_recompute: If True, ignore existing cache and recompute all frames
        debug_vis: If True, save 2D visualizations of score images
        debug_vis_stride: Only visualize every Nth frame

    Returns:
        score_stats: Dictionary with SAM3 statistics
    """
    color_files, _ = get_rgbd_file_lists(frames_dir)

    print("\n" + "="*80)
    print("PASS 1: Running SAM3 on all frames")
    print("="*80)
    print(f"SAM3 filtering: size=[{sam_min_size_ratio:.4f}, {sam_max_size_ratio:.2f}], shrink={sam_shrink_pixels}px")
    print(f"Scoring: boundary_distance={sam_boundary_distance}px (outside=-1.0, boundary=0.0, inside=1.0)")
    if force_recompute:
        print("⚠ Force recompute enabled - ignoring existing cache")
    if debug_vis:
        print(f"🎨 Debug visualization enabled (every {debug_vis_stride} frames)")
        vis_dir = os.path.join(cache_dir, '../debug_vis')
        os.makedirs(vis_dir, exist_ok=True)

    stats = {
        'frames_processed': 0,
        'frames_cached': 0,
        'frames_recomputed': 0,
        'frames_with_masks': 0,
        'total_masks': 0,
        'frames_no_masks': 0
    }

    for idx in tqdm(frame_indices, desc="SAM3 segmentation"):
        cache_path = None
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"frame_{idx:06d}.npy")

        # Check if already cached (unless force recompute)
        if not force_recompute and cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True).item()
            num_segs = data['num_segments']
            stats['frames_cached'] += 1
        else:
            # Run SAM3
            if cache_path and os.path.exists(cache_path):
                stats['frames_recomputed'] += 1

            score_image, num_segs = create_scored_mask_image(
                color_files[idx], processor,
                prompt=sam_prompt,
                max_size_ratio=sam_max_size_ratio,
                min_size_ratio=sam_min_size_ratio,
                shrink_pixels=sam_shrink_pixels,
                boundary_distance=sam_boundary_distance,
                cache_path=cache_path
            )

        stats['frames_processed'] += 1
        if num_segs > 0:
            stats['frames_with_masks'] += 1
            stats['total_masks'] += num_segs
        else:
            stats['frames_no_masks'] += 1

        # Generate debug visualization
        if debug_vis and (idx % debug_vis_stride == 0):
            # Load score_image if it was cached
            if cache_path and os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True).item()
                score_image = data['score_image']

            vis_path = os.path.join(vis_dir, f'frame_{idx:06d}_scores.png')
            visualize_score_image_2d(color_files[idx], score_image, vis_path, idx)

    print(f"\n✓ SAM3 Pass Complete:")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Frames cached (reused): {stats['frames_cached']}")
    print(f"  Frames recomputed (overwritten): {stats['frames_recomputed']}")
    print(f"  Frames with masks: {stats['frames_with_masks']} ({100*stats['frames_with_masks']/stats['frames_processed']:.1f}%)")
    print(f"  Frames with NO masks: {stats['frames_no_masks']} ({100*stats['frames_no_masks']/stats['frames_processed']:.1f}%)")
    print(f"  Total masks detected: {stats['total_masks']}")
    print(f"  Average masks/frame: {stats['total_masks']/stats['frames_processed']:.1f}")

    return stats


def integrate_with_voxel_labeling(frames_dir, intrinsic, poses,
                                  depth_scale=1000.0, depth_max=3.0, voxel_size=0.005,
                                  frame_subsample=1, cache_dir=None):
    """
    PASS 2: Integrate RGB-D frames with pre-computed voxel scores using dual-TSDF.

    Uses Open3D's C++ TSDF integration for both geometry and scores (fast!).

    Returns:
        mesh: 3D mesh
        score_volume: TSDF volume containing scores
        vertex_scores: Per-vertex average scores
    """
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))

    # Subsample frames
    frame_indices = list(range(0, n_frames, frame_subsample))
    print(f"\n" + "="*80)
    print("PASS 2: TSDF Integration with Dual-TSDF Scoring")
    print("="*80)
    print(f"Dataset: {len(color_files)} color, {len(depth_files)} depth, {len(poses)} poses")
    print(f"Using {len(frame_indices)} frames (subsample={frame_subsample})")

    # Create TSDF volume for geometry
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Create TSDF volume for scores (use RGB8 to match geometry volume)
    score_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    print(f"TSDF: voxel={voxel_size}m, trunc={voxel_size*4.0}m, depth_max={depth_max}m")
    print(f"Using dual-TSDF: geometry + score volumes (C++ integration)")

    # Track statistics
    frames_with_scores = 0
    total_positive_pixels = 0
    total_pixels = 0

    for frame_count, idx in enumerate(tqdm(frame_indices, desc="Integrating frames")):
        # Load pre-computed SAM3 scores
        cache_path = os.path.join(cache_dir, f"frame_{idx:06d}.npy")

        if not os.path.exists(cache_path):
            print(f"\n⚠ Warning: No cached score for frame {idx}, skipping")
            continue

        data = np.load(cache_path, allow_pickle=True).item()
        score_image_raw = data['score_image']  # HxW array, values -1 to 1

        # Load RGB-D images
        color = o3d.io.read_image(color_files[idx])
        depth = o3d.io.read_image(depth_files[idx])

        # Integrate geometry (standard)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False
        )

        extrinsic = np.linalg.inv(poses[idx])
        volume.integrate(rgbd, intrinsic, extrinsic)

        # Integrate scores by using the same depth but different "color"
        # The color will be the score replicated across RGB channels
        # Convert score_image from [-1, 1] to [0, 255] for Open3D
        score_image_normalized = ((score_image_raw + 1.0) / 2.0 * 255).astype(np.uint8)

        # Replicate to 3 channels (RGB) - same format as color image
        h, w = score_image_normalized.shape
        score_image_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        score_image_rgb[:, :, 0] = score_image_normalized
        score_image_rgb[:, :, 1] = score_image_normalized
        score_image_rgb[:, :, 2] = score_image_normalized

        # Use PIL to create a proper RGB image, then convert to Open3D
        from PIL import Image as PILImage
        score_pil = PILImage.fromarray(score_image_rgb, mode='RGB')

        # Convert PIL to numpy in the exact format Open3D expects
        score_np = np.asarray(score_pil)

        # Create Open3D image (this should match the color image format exactly)
        score_o3d = o3d.geometry.Image(score_np)

        # Create RGBD image for score integration (same as geometry)
        score_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            score_o3d, depth,
            depth_scale=depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False
        )

        # Integrate scores using Open3D's C++ code!
        score_volume.integrate(score_rgbd, intrinsic, extrinsic)

        # Statistics
        frames_with_scores += 1
        total_positive_pixels += (score_image_raw > 0).sum()
        total_pixels += score_image_raw.size

        # Debug output for first few frames and every 50th frame
        if frame_count < 3 or frame_count % 50 == 0:
            pos_pct = 100 * (score_image_raw > 0).sum() / score_image_raw.size
            print(f"\n  Frame {idx} ({frame_count+1}/{len(frame_indices)}): {pos_pct:.1f}% positive pixels")

    print("✓ Integration complete")

    # Print statistics
    print(f"\n📊 Integration Statistics:")
    print(f"  Frames integrated: {frames_with_scores}/{len(frame_indices)}")
    print(f"  Average positive pixels per frame: {100*total_positive_pixels/max(total_pixels,1):.1f}%")

    # Extract mesh from geometry volume
    print("\nExtracting mesh from geometry TSDF...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"  ✓ Geometry mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # Extract mesh from score volume (should have identical geometry!)
    print("\nExtracting mesh from score TSDF...")
    score_mesh = score_volume.extract_triangle_mesh()
    print(f"  ✓ Score mesh: {len(score_mesh.vertices):,} vertices, {len(score_mesh.triangles):,} triangles")

    # Verify they match
    if len(mesh.vertices) == len(score_mesh.vertices):
        print(f"  ✓ Meshes have identical vertex counts - direct mapping possible!")

        # Extract scores from score mesh vertex colors (R channel, all RGB are same)
        score_colors = np.asarray(score_mesh.vertex_colors)
        vertex_scores = score_colors[:, 0] * 2.0 - 1.0  # Convert from [0,1] to [-1,1]

        print(f"  ✓ Extracted scores from {len(vertex_scores):,} vertices")
    else:
        print(f"  ⚠️ Warning: Mesh vertex counts don't match!")
        print(f"    Geometry: {len(mesh.vertices):,}, Score: {len(score_mesh.vertices):,}")
        print(f"  Falling back to nearest neighbor matching...")

        # Fallback: use KNN
        score_vertices = np.asarray(score_mesh.vertices)
        score_colors = np.asarray(score_mesh.vertex_colors)

        score_tree = cKDTree(score_vertices)
        distances, indices = score_tree.query(np.asarray(mesh.vertices), k=1)

        vertex_scores = score_colors[indices, 0] * 2.0 - 1.0

    # Print score distribution
    print(f"\n  Vertex score distribution:")
    print(f"    < -0.5 (seam): {(vertex_scores < -0.5).sum():,} ({100*(vertex_scores < -0.5).sum()/len(vertices):.1f}%)")
    print(f"    -0.5 to 0.0 (weak seam): {((vertex_scores >= -0.5) & (vertex_scores < 0.0)).sum():,} ({100*((vertex_scores >= -0.5) & (vertex_scores < 0.0)).sum()/len(vertices):.1f}%)")
    print(f"    0.0 to 0.5 (boundary): {((vertex_scores >= 0.0) & (vertex_scores < 0.5)).sum():,} ({100*((vertex_scores >= 0.0) & (vertex_scores < 0.5)).sum()/len(vertices):.1f}%)")
    print(f"    >= 0.5 (rock): {(vertex_scores >= 0.5).sum():,} ({100*(vertex_scores >= 0.5).sum()/len(vertices):.1f}%)")
    print(f"  Average score: {vertex_scores.mean():.3f}")

    return mesh, score_volume, vertex_scores


def segment_mesh_by_voxel_scores(mesh, vertex_scores, rock_threshold=0.3, min_cluster_size=500):
    """
    Segment mesh based on voxel mask scores.

    Args:
        mesh: Triangle mesh
        vertex_scores: Per-vertex mask scores (-1.0 to 1.0)
        rock_threshold: Score threshold (>= threshold = rock, < threshold = seam)
        min_cluster_size: Minimum triangles per segment

    Returns:
        rock_meshes: List of (segment_id, submesh) for rock segments
        seam_mesh: Mesh of seam regions
    """
    print(f"\nSegmenting mesh by voxel scores (threshold={rock_threshold})...")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Classify vertices
    is_rock = vertex_scores >= rock_threshold

    print(f"  Rock vertices: {is_rock.sum():,} / {len(vertices):,} ({100*is_rock.sum()/len(vertices):.1f}%)")

    # Build adjacency graph for rock vertices only
    print("  Finding rock components...")
    adjacency = [set() for _ in range(len(vertices))]

    for tri in triangles:
        v0, v1, v2 = tri

        # Add edges only between rock vertices
        if is_rock[v0] and is_rock[v1]:
            adjacency[v0].add(v1)
            adjacency[v1].add(v0)
        if is_rock[v1] and is_rock[v2]:
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
        if is_rock[v2] and is_rock[v0]:
            adjacency[v2].add(v0)
            adjacency[v0].add(v2)

    # Connected component analysis for rock regions
    vertex_labels = -np.ones(len(vertices), dtype=np.int32)  # -1 = seam/unlabeled
    current_label = 0

    for start_v in range(len(vertices)):
        if vertex_labels[start_v] >= 0:  # Already labeled
            continue
        if not is_rock[start_v]:  # Seam vertex
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

    num_rock_segments = current_label
    print(f"  Found {num_rock_segments} rock components")

    # Extract rock segments
    rock_meshes = []

    for seg_id in range(num_rock_segments):
        # Find triangles where ALL vertices belong to this rock segment
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

        rock_meshes.append((seg_id + 1, submesh))
        print(f"    Rock {seg_id+1}: {len(submesh.vertices):,} vertices, {len(submesh.triangles):,} triangles")

    # Extract seam mesh (vertices with label -1)
    print("  Extracting seam mesh...")
    seam_mask = np.any(vertex_labels[triangles] == -1, axis=1)
    seam_triangles = triangles[seam_mask]

    seam_mesh = o3d.geometry.TriangleMesh()
    seam_mesh.vertices = mesh.vertices
    seam_mesh.triangles = o3d.utility.Vector3iVector(seam_triangles)
    seam_mesh.vertex_colors = mesh.vertex_colors
    seam_mesh.vertex_normals = mesh.vertex_normals
    seam_mesh.remove_unreferenced_vertices()
    seam_mesh.compute_vertex_normals()

    print(f"    Seam: {len(seam_mesh.vertices):,} vertices, {len(seam_mesh.triangles):,} triangles")

    return rock_meshes, seam_mesh


def visualize_vertex_scores(mesh, vertex_scores, output_path):
    """
    Create a colored mesh showing voxel mask scores.

    Purple (-1.0) = seam (punishment)
    Blue (0.0) = boundary
    Green (0.5) = inside
    Red (1.0) = deep inside rock
    """
    print(f"\nCreating score visualization...")

    # Normalize scores from [-1, 1] to [0, 1] for color mapping
    norm_scores = (vertex_scores + 1.0) / 2.0

    # Vectorized color computation (much faster than for loop!)
    colors = np.zeros((len(norm_scores), 3))

    # Mask for each range
    mask1 = norm_scores < 0.25  # -1.0 to -0.5: Purple to blue
    mask2 = (norm_scores >= 0.25) & (norm_scores < 0.5)  # -0.5 to 0.0: Blue
    mask3 = (norm_scores >= 0.5) & (norm_scores < 0.75)  # 0.0 to 0.5: Blue to green
    mask4 = norm_scores >= 0.75  # 0.5 to 1.0: Green to red

    # Purple to blue
    t1 = norm_scores[mask1] * 4
    colors[mask1, 0] = 0.5 * (1 - t1)
    colors[mask1, 1] = 0
    colors[mask1, 2] = 1

    # Blue
    colors[mask2, 0] = 0
    colors[mask2, 1] = 0
    colors[mask2, 2] = 1

    # Blue to green
    t3 = (norm_scores[mask3] - 0.5) * 4
    colors[mask3, 0] = 0
    colors[mask3, 1] = t3
    colors[mask3, 2] = 1 - t3

    # Green to red
    t4 = (norm_scores[mask4] - 0.75) * 4
    colors[mask4, 0] = t4
    colors[mask4, 1] = 1 - t4
    colors[mask4, 2] = 0

    # Create colored mesh
    vis_mesh = o3d.geometry.TriangleMesh(mesh)
    vis_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    vis_mesh.compute_vertex_normals()

    print(f"  Writing mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, vis_mesh)
    print(f"  ✓ Saved ({os.path.getsize(output_path)/(1024**2):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 voxel-labeling segmentation + 3D reconstruction"
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
    parser.add_argument('--sam_confidence', type=float, default=0.15,
                       help='SAM3 confidence threshold')
    parser.add_argument('--sam_max_size_ratio', type=float, default=0.15,
                       help='Max segment size as fraction of image')
    parser.add_argument('--sam_min_size_ratio', type=float, default=0.001,
                       help='Min segment size as fraction of image')
    parser.add_argument('--sam_shrink_pixels', type=int, default=3,
                       help='Shrink masks by N pixels for clearance')
    parser.add_argument('--sam_boundary_distance', type=int, default=10,
                       help='Distance from boundary to max score (pixels)')

    # Reconstruction parameters
    parser.add_argument('--voxel_size', type=float, default=0.005,
                       help='TSDF voxel size (meters)')
    parser.add_argument('--depth_max', type=float, default=3.0,
                       help='Max depth (meters)')
    parser.add_argument('--frame_subsample', type=int, default=5,
                       help='Process every Nth frame')

    # Segmentation parameters
    parser.add_argument('--rock_threshold', type=float, default=0.3,
                       help='Score threshold for rock classification')
    parser.add_argument('--min_cluster_size', type=int, default=500,
                       help='Minimum triangles per segment')

    # Cache control
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recompute all SAM3 scores (ignore cache)')

    # Debug options
    parser.add_argument('--debug_vis', action='store_true',
                       help='Generate 2D visualizations of score images')
    parser.add_argument('--debug_vis_stride', type=int, default=10,
                       help='Visualize every Nth frame (default: 10)')
    parser.add_argument('--skip_tsdf', action='store_true',
                       help='Skip TSDF integration (only run SAM3 + visualization)')

    args = parser.parse_args()

    print("="*80)
    print("SAM3 Voxel-Labeling Segmented Reconstruction")
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
    cache_dir = os.path.join(args.segments_dir, 'scored_masks')
    os.makedirs(cache_dir, exist_ok=True)

    # Get frame indices
    color_files, depth_files = get_rgbd_file_lists(args.frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))
    frame_indices = list(range(0, n_frames, args.frame_subsample))

    # PASS 1: Precompute SAM3 scores
    sam3_stats = precompute_sam3_scores(
        args.frames_dir, processor, frame_indices,
        sam_prompt=args.sam_prompt,
        sam_confidence=args.sam_confidence,
        sam_max_size_ratio=args.sam_max_size_ratio,
        sam_min_size_ratio=args.sam_min_size_ratio,
        sam_shrink_pixels=args.sam_shrink_pixels,
        sam_boundary_distance=args.sam_boundary_distance,
        cache_dir=cache_dir,
        force_recompute=args.force_recompute,
        debug_vis=args.debug_vis,
        debug_vis_stride=args.debug_vis_stride
    )

    # Early exit if skipping TSDF
    if args.skip_tsdf:
        print("\n" + "="*80)
        print("⚠ TSDF integration skipped (--skip_tsdf enabled)")
        print("="*80)
        if args.debug_vis:
            vis_dir = os.path.join(cache_dir, '../debug_vis')
            print(f"\nDebug visualizations saved to: {vis_dir}/")
            print(f"Generated {len([f for f in os.listdir(vis_dir) if f.endswith('.png')])} visualization images")
        print("\nTo run full reconstruction, remove --skip_tsdf flag")
        return

    # PASS 2: TSDF integration with dual-TSDF scoring
    mesh, score_volume, vertex_scores = integrate_with_voxel_labeling(
        args.frames_dir, intrinsic, poses,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        frame_subsample=args.frame_subsample,
        cache_dir=cache_dir
    )

    # Save full mesh
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nSaving full mesh: {args.output}")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"✓ Saved ({os.path.getsize(args.output)/(1024**2):.1f} MB)")

    # Save score visualization
    score_vis_path = args.output.replace('.ply', '_scores.ply')
    visualize_vertex_scores(mesh, vertex_scores, score_vis_path)

    # DISABLED: Mesh segmentation (too memory-intensive for large meshes)
    # TODO: Implement more efficient segmentation using spatial partitioning
    print("\n⚠ Mesh segmentation disabled (too memory-intensive)")
    print("  Use score visualization to inspect rock/seam distribution")

    # # Segment mesh by voxel scores
    # rock_meshes, seam_mesh = segment_mesh_by_voxel_scores(
    #     mesh, vertex_scores,
    #     rock_threshold=args.rock_threshold,
    #     min_cluster_size=args.min_cluster_size
    # )

    # # Save seam mesh
    # seam_path = args.output.replace('.ply', '_seam.ply')
    # o3d.io.write_triangle_mesh(seam_path, seam_mesh)
    # print(f"\n✓ Seam mesh: {seam_path}")

    # # Save individual rock segments
    # os.makedirs(args.segments_dir, exist_ok=True)
    # print(f"\nSaving {len(rock_meshes)} rock segments to {args.segments_dir}/")

    # for seg_id, submesh in rock_meshes:
    #     seg_path = os.path.join(args.segments_dir, f"rock_{seg_id:03d}.ply")
    #     o3d.io.write_triangle_mesh(seg_path, submesh)

    print("\n" + "="*80)
    print("Reconstruction Complete!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Full mesh: {args.output}")
    print(f"  Score visualization: {score_vis_path}")
    print(f"\nScore color map:")
    print(f"    Purple: seam (score < -0.5)")
    print(f"    Blue: weak seam/boundary (score -0.5 to 0.5)")
    print(f"    Green: inside rock (score 0.5 to 0.8)")
    print(f"    Red: deep inside rock (score > 0.8)")
    print(f"\nTo visualize:")
    print(f"  open3d {score_vis_path}")
    print(f"\nNote: Mesh segmentation is disabled. To extract rock segments,")
    print(f"use CloudCompare or MeshLab to segment by vertex color.")


if __name__ == "__main__":
    main()
