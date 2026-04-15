#!/usr/bin/env python3
"""
SAM3 Dual-TSDF Segmented Reconstruction

Fuses an RGB-D sequence into a stone-segment mesh using two TSDF passes and
an EDT-based alpha scoring scheme.

Pass 1 — Raw TSDF  (integrate_tsdf)
    All frames fused into a dense mesh preserving original RGB colours.
    Output: raw_mesh_rgb.ply

Pass 2 — Semantic TSDF  (precompute_alphas → integrate_semantic_tsdf)
    2a. EDT alpha maps  (precompute_alphas, parallel CPU threads)
        Reads L1 mask cache (frames_dir/sam3_mask_cache/masks_NNNNNN.npz)
        produced by 01b_run_sam3_alpha.py.  For each mask runs Euclidean
        Distance Transform; normalises and applies gamma:
            score = (dist / max_dist) ** edt_gamma
        0 = seam / background, 1 = stone interior.
        Saved to L2 alpha cache (output_dir/alpha_maps/alpha_NNNNNN.npz).
        Re-used automatically when edt_gamma and sam_max_size_ratio are
        unchanged (same output dir).

    2b. Alpha TSDF  (integrate_semantic_tsdf)
        Reads L2 alpha maps, encodes them as grayscale RGB, and fuses into
        a second TSDF.  TSDF weighted-averaging accumulates per-voxel stone
        scores across all viewpoints, naturally handling occlusion.

Scoring & culling
    cKDTree transfers alpha scores from alpha_mesh vertices → raw_mesh
    vertices.  Vertices below alpha_threshold are marked as seams.
    culled_mesh_rgb.ply: raw RGB mesh with seam triangles removed.

Segmentation
    scipy connected_components (C-level, vectorised numpy edges) finds stone
    patches separated by the seam network.  Segments written one at a time
    to avoid OOM.  Pseudo-colour visualisations: score_map_mesh.ply,
    segments.ply.

Cache layers
    L1  frames_dir/sam3_mask_cache/masks_NNNNNN.npz   raw SAM3 output
    L2  output_dir/alpha_maps/alpha_NNNNNN.npz         EDT alpha (per run)
"""

import sys
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
from scipy.spatial import cKDTree

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


def _run_sam3_frame(image_path, processor, prompt, cache_path):
    """
    Run SAM3 on one frame and cache raw masks + confidence scores.

    Cache format (masks_NNNNNN.npz):
      masks:  uint8  (N, H, W) — binary instance masks
      scores: float32 (N,)     — per-mask SAM3 confidence scores

    Returns (masks_bool (N,H,W), scores_f32 (N,)).
    Loads from cache if available; runs inference otherwise.
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['masks'].astype(bool), data['scores'].astype(np.float32)

    if processor is None:
        raise RuntimeError(
            f"No SAM3 processor and no cache found at {cache_path}. "
            "Run 01b_run_sam3_alpha.py first to populate the mask cache."
        )

    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    h, w = image.size[1], image.size[0]

    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks  = inference_state["masks"]   # (N, 1, H, W) bool tensor
    scores = inference_state["scores"]  # (N,) float32 tensor

    if masks.shape[0] == 0:
        masks_np  = np.zeros((0, h, w), dtype=np.uint8)
        scores_np = np.zeros(0, dtype=np.float32)
    else:
        masks_np  = masks.squeeze(1).cpu().numpy().astype(np.uint8)
        scores_np = scores.float().cpu().numpy()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, masks=masks_np, scores=scores_np)

    return masks_np.astype(bool), scores_np


def generate_alpha_frame(image_path, processor, prompt="individual stone",
                          max_size_ratio=0.15, cache_path=None, edt_gamma=1.0):
    """
    Compute a per-pixel stone-score (alpha) image for one frame.

    Loads raw masks from the mask cache (masks_NNNNNN.npz) if available,
    then applies the size filter and EDT on-the-fly.  This means max_size_ratio,
    alpha_threshold, and edt_gamma can be tuned without re-running SAM3 inference.

    Score per pixel (after EDT normalisation and gamma):
      - Outside all masks              → 0.0  (background / not a stone)
      - On a mask perimeter            → 0.0  (seam)
      - Inside a mask, at max distance → 1.0  (stone interior)
      - Between perimeter and centre   → score = (dist / max_dist) ** edt_gamma

    edt_gamma < 1  (e.g. 0.5): interiors saturate quickly, seams stay sharp
    edt_gamma = 1  (default):  linear falloff
    edt_gamma > 1  (e.g. 2.0): slow rise from seam, more conservative

    When a pixel belongs to multiple masks the maximum score is kept.

    Returns:
        alpha: float32 [H, W], values in [0, 1]
    """
    masks_bool, _ = _run_sam3_frame(image_path, processor, prompt, cache_path)

    if masks_bool.shape[0] == 0:
        # No masks: read image size from file for zero array dimensions
        img = Image.open(image_path)
        h, w = img.size[1], img.size[0]
        return np.zeros((h, w), dtype=np.float32)

    h, w = masks_bool.shape[1], masks_bool.shape[2]
    alpha    = np.zeros((h, w), dtype=np.float32)
    img_area = h * w

    for mask in masks_bool:
        if mask.sum() / img_area > max_size_ratio:
            continue
        dist  = ndimage.distance_transform_edt(mask).astype(np.float32)
        max_d = dist.max()
        if max_d > 0:
            score = (dist / max_d) ** edt_gamma
        else:
            score = mask.astype(np.float32)
        alpha = np.maximum(alpha, score)

    return alpha


# ── TSDF helpers ──────────────────────────────────────────────────────────────

def _make_intrinsic_tensor(intrinsic):
    """Open3D PinholeCameraIntrinsic → 3×3 float64 Tensor (t-geometry API)."""
    return o3d.core.Tensor(np.asarray(intrinsic.intrinsic_matrix),
                           dtype=o3d.core.float64)


def _estimate_block_count(voxel_size, depth_max):
    """Conservative upper-bound on VoxelBlockGrid blocks for this scene."""
    vol      = (2 / 3) * np.pi * depth_max ** 3 * 0.20   # hemisphere at 20% fill
    n_blocks = int(vol / voxel_size ** 3 / 16 ** 3)
    return max(n_blocks * 3, 20_000)


def _gpu_bytes_for_blocks(block_count):
    return block_count * (16 ** 3) * (4 + 2 + 3)   # tsdf f32 + weight u16 + color u8×3


def _create_tsdf_volume(voxel_size, depth_max):
    """
    Create a CPU ScalableTSDFVolume.
    Returns (volume, device=None).

    Note: Open3D's VoxelBlockGrid GPU API has device-placement inconsistencies
    across versions (extrinsic must be CPU while depth must be CUDA). Using the
    proven CPU path until the API stabilises.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc   =voxel_size * 4.0,
        color_type  =o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    print("  CPU TSDF (ScalableTSDFVolume)")
    return volume, None


def _integrate_frame(volume, device, intrinsic, intrinsic_t,
                      color_np, depth_np, pose, depth_scale, depth_max):
    """Integrate one RGBD frame — GPU (VoxelBlockGrid) or CPU (ScalableTSDFVolume)."""
    if device is not None:
        depth_t = o3d.t.geometry.Image(
            np.ascontiguousarray(depth_np, dtype=np.uint16)).to(device)
        color_t = o3d.t.geometry.Image(
            np.ascontiguousarray(color_np, dtype=np.uint8)).to(device)
        # Tensors must be on the same device as the volume
        intr_d  = intrinsic_t.to(device)
        extr_d  = o3d.core.Tensor(
            np.linalg.inv(pose), dtype=o3d.core.float64).to(device)
        # VoxelBlockGrid requires block_coords computed from current frustum
        block_coords = volume.compute_unique_block_coordinates(
            depth_t, intr_d, extr_d, float(depth_scale), float(depth_max)
        )
        volume.integrate(block_coords, depth_t, color_t, intr_d, extr_d,
                         float(depth_scale), float(depth_max))
    else:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_np),
            o3d.geometry.Image(depth_np.astype(np.uint16)),
            depth_scale=depth_scale, depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))


def _extract_mesh(volume, device):
    """Extract triangle mesh from TSDF volume."""
    mesh = volume.extract_triangle_mesh()
    if device is not None:
        mesh = mesh.to_legacy()
    mesh.compute_vertex_normals()
    return mesh

# ─────────────────────────────────────────────────────────────────────────────


def precompute_alphas(cache_dir, alpha_dir, color_files, n_frames,
                      sam_prompt, sam_max_size_ratio, edt_gamma,
                      processor=None):
    """
    Phase 1 of semantic TSDF: pre-compute EDT alpha maps for all frames and
    save them to alpha_dir/alpha_NNNNNN.npz.

    Already-computed files are skipped (safe to re-run; acts as a cache when
    gamma and max_size_ratio are unchanged).

    EDT runs in a thread pool — scipy.ndimage releases the GIL so multiple
    CPU cores are used.  When processor is not None (SAM3 inference needed),
    execution is kept serial to avoid GPU contention.
    """
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(alpha_dir, exist_ok=True)

    n_done = sum(1 for i in range(n_frames)
                 if os.path.exists(os.path.join(alpha_dir, f"alpha_{i:06d}.npz")))
    if n_done == n_frames:
        print(f"  Alpha maps already complete ({n_done}/{n_frames}) — skipping EDT")
        return

    # Leave 1 core for the main TSDF thread; use the rest for EDT.
    # scipy EDT releases the GIL so threads give true CPU parallelism.
    n_workers = 1 if processor is not None else max((os.cpu_count() or 4) - 4, 1)
    print(f"\nPre-computing alpha maps: {n_frames} frames  "
          f"EDT ×{n_workers} thread{'s' if n_workers > 1 else ''}")

    def worker(idx):
        alpha_path = os.path.join(alpha_dir, f"alpha_{idx:06d}.npz")
        if os.path.exists(alpha_path):
            return
        cache_path = os.path.join(cache_dir, f"masks_{idx:06d}.npz")
        alpha = generate_alpha_frame(
            color_files[idx], processor,
            prompt=sam_prompt, max_size_ratio=sam_max_size_ratio,
            cache_path=cache_path, edt_gamma=edt_gamma,
        )
        np.savez_compressed(alpha_path, alpha=alpha)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(tqdm(pool.map(worker, range(n_frames)),
                  total=n_frames, desc="EDT alpha maps"))

    print(f"✓ Alpha maps written to {alpha_dir}")


def integrate_semantic_tsdf(frames_dir, intrinsic, poses, alpha_dir,
                             depth_scale=1000.0, depth_max=3.0, voxel_size=0.005,
                             depth_min_m=0.15):
    """
    Phase 2 of semantic TSDF: read pre-computed alpha maps from alpha_dir and
    fuse them into a TSDF volume (GPU or CPU).

    EDT is already done — this function is pure TSDF integration.

    Returns:
        alpha_mesh: mesh whose vertex_colors encode per-vertex alpha score
                    (R=G=B=score, range 0–1).
    """
    color_files, depth_files = get_rgbd_file_lists(frames_dir)
    n_frames = min(len(color_files), len(depth_files), len(poses))

    print(f"\nSemantic TSDF: {n_frames} frames (reading pre-computed alphas)")
    volume, device = _create_tsdf_volume(voxel_size, depth_max)
    intrinsic_t = _make_intrinsic_tensor(intrinsic) if device is not None else None

    for idx in tqdm(range(n_frames), desc="Semantic TSDF"):
        alpha_path = os.path.join(alpha_dir, f"alpha_{idx:06d}.npz")
        alpha = np.load(alpha_path)['alpha']

        a_uint8   = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        alpha_rgb = np.stack([a_uint8, a_uint8, a_uint8], axis=-1)

        depth_np = np.asarray(o3d.io.read_image(depth_files[idx]))
        depth_np = apply_depth_filter(depth_np, depth_scale, min_depth_m=depth_min_m)

        _integrate_frame(volume, device, intrinsic, intrinsic_t,
                         alpha_rgb, depth_np, poses[idx], depth_scale, depth_max)

    print("✓ Semantic TSDF complete")
    alpha_mesh = _extract_mesh(volume, device)
    print(f"  Alpha mesh vertices: {len(alpha_mesh.vertices):,}")
    return alpha_mesh


def transfer_alpha_scores(raw_mesh, alpha_mesh):
    """
    For each vertex in raw_mesh find the nearest vertex in alpha_mesh and
    return its alpha score (single float, extracted from R channel).

    Uses a scipy cKDTree for vectorised nearest-neighbour lookup.
    The raw mesh's vertex_colors are NOT modified.

    Returns:
        alpha_scores: float32 [N_raw_vertices], values in [0, 1]
    """
    raw_pts   = np.asarray(raw_mesh.vertices)
    alpha_pts = np.asarray(alpha_mesh.vertices)
    alpha_col = np.asarray(alpha_mesh.vertex_colors)  # [M, 3], R=G=B=score

    tree = cKDTree(alpha_pts)
    _, indices = tree.query(raw_pts, k=1, workers=-1)
    alpha_scores = alpha_col[indices, 0].astype(np.float32)  # take R channel

    print(f"  Alpha score transfer: min={alpha_scores.min():.3f}  "
          f"mean={alpha_scores.mean():.3f}  max={alpha_scores.max():.3f}")
    return alpha_scores


def integrate_tsdf(frames_dir, intrinsic, poses,
                   depth_scale=1000.0, depth_max=3.0, voxel_size=0.005,
                   depth_min_m=0.15, confidence_threshold=0):
    """
    Pass 1: pure TSDF integration over all frames. Returns the fused mesh
    with original RGB colours. SAM3 is not involved here — alpha scoring
    and segmentation happen in pass 2 (precompute_alphas → integrate_semantic_tsdf).
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

    volume, device = _create_tsdf_volume(voxel_size, depth_max)
    intrinsic_t = _make_intrinsic_tensor(intrinsic) if device is not None else None

    for i in tqdm(range(n_frames), desc="TSDF integration"):
        color_np = np.asarray(Image.open(color_files[i]).convert('RGB'))
        depth_np = np.asarray(o3d.io.read_image(depth_files[i]))
        conf_np = (np.asarray(o3d.io.read_image(conf_files[i]))
                   if use_confidence and i < len(conf_files) else None)
        depth_np = apply_depth_filter(depth_np, depth_scale,
                                      min_depth_m=depth_min_m,
                                      confidence_np=conf_np,
                                      confidence_threshold=confidence_threshold)
        _integrate_frame(volume, device, intrinsic, intrinsic_t,
                         color_np, depth_np, poses[i], depth_scale, depth_max)

    print("✓ TSDF integration complete\nExtracting mesh...")
    mesh = _extract_mesh(volume, device)
    print(f"  Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")
    return mesh


def clean_mesh_keep_largest(mesh, keep_n=1):
    """
    Remove small floating fragments by keeping only the N largest connected
    components (by triangle count). Runs before alpha scoring so scores
    aren't polluted by noise geometry.
    """
    print(f"\nCleaning mesh: keeping top {keep_n} connected components...")
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters  = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    # Rank clusters by size, pick top-N
    ranked = np.argsort(cluster_n_triangles)[::-1]
    keep_ids = set(ranked[:keep_n].tolist())

    mask = np.array([c in keep_ids for c in triangle_clusters])
    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    kept   = cluster_n_triangles[list(keep_ids)].sum()
    total  = len(np.asarray(mesh.triangles))
    print(f"  Components kept: {min(keep_n, len(cluster_n_triangles))} / {len(cluster_n_triangles)}")
    print(f"  Triangles after clean: {total:,}")
    return mesh



def propagate_boundary_scores(mesh, is_boundary_seed, max_hops=5):
    """
    Propagate boundary scores outward from seed seam vertices via BFS.

    Seed vertices (directly detected as seam) get score 1.0.
    Each hop away the score decays linearly: score = 1 - hop/max_hops.
    Vertices beyond max_hops stay at 0.

    This fattens and connects the seam network so that small gaps between
    directly-detected seam pixels don't create open passages for BFS flooding
    in the segmentation step.

    Args:
        mesh: Triangle mesh
        is_boundary_seed: bool array [N_vertices] — True where alpha < alpha_threshold
        max_hops: How many mesh edges to propagate from each seed vertex

    Returns:
        scores: float32 array [N_vertices], 0 = interior, 1 = seam seed
    """
    from collections import deque
    triangles = np.asarray(mesh.triangles)
    N = len(np.asarray(mesh.vertices))

    # Build vertex adjacency
    adjacency = [[] for _ in range(N)]
    for tri in triangles:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        adjacency[v0].append(v1); adjacency[v1].append(v0)
        adjacency[v1].append(v2); adjacency[v2].append(v1)
        adjacency[v2].append(v0); adjacency[v0].append(v2)

    scores   = np.zeros(N, dtype=np.float32)
    distance = np.full(N, -1, dtype=np.int32)

    seeds = np.where(is_boundary_seed)[0]
    scores[seeds]   = 1.0
    distance[seeds] = 0

    queue = deque(seeds.tolist())

    while queue:
        v = queue.popleft()
        d = distance[v]
        if d >= max_hops:
            continue
        for nb in adjacency[v]:
            if distance[nb] < 0:
                distance[nb] = d + 1
                scores[nb]   = 1.0 - (d + 1) / max_hops
                queue.append(nb)

    n_expanded = int((scores > 0).sum())
    print(f"  Boundary propagation: {len(seeds):,} seeds → {n_expanded:,} vertices "
          f"covered (max_hops={max_hops})")
    return scores


def make_boundary_score_mesh(mesh, boundary_scores):
    """
    Return a copy of mesh with vertices colored by boundary score.
    0 (interior) → blue [0.15, 0.35, 0.85]
    1 (seam seed) → red  [0.90, 0.15, 0.10]
    Linear interpolation in between.
    """
    interior = np.array([0.15, 0.35, 0.85], dtype=np.float64)
    seam     = np.array([0.90, 0.15, 0.10], dtype=np.float64)
    s = np.clip(boundary_scores, 0.0, 1.0)[:, None]  # [N, 1]
    colors = (1 - s) * interior + s * seam            # [N, 3]

    debug = o3d.geometry.TriangleMesh(mesh)
    debug.vertex_colors = o3d.utility.Vector3dVector(colors)
    return debug


def cull_mesh_by_alpha(mesh, alpha_scores, alpha_threshold):
    """
    Remove seam/background triangles from the raw RGB mesh.

    A triangle is dropped if ANY of its 3 vertices has alpha < alpha_threshold.
    RGB vertex colors are preserved on surviving vertices.

    Returns a new mesh (raw mesh is unchanged).
    """
    triangles = np.asarray(mesh.triangles)
    keep_tri  = np.all(alpha_scores[triangles] >= alpha_threshold, axis=1)

    culled = o3d.geometry.TriangleMesh(mesh)
    culled.remove_triangles_by_mask(~keep_tri)
    culled.remove_unreferenced_vertices()
    culled.compute_vertex_normals()

    n_kept = int(keep_tri.sum())
    print(f"  Vertex culling: {n_kept:,}/{len(triangles):,} triangles kept "
          f"({100*n_kept/len(triangles):.1f}%)")
    return culled


def make_segment_color_mesh(mesh, vertex_labels):
    """
    Return a copy of mesh with one pseudo-random color per segment.
    Boundary / unlabeled vertices (label == -1) are colored mid-gray.
    Uses golden-ratio hue rotation for visually distinct colors.
    """
    n_vertices = len(np.asarray(mesh.vertices))
    colors = np.full((n_vertices, 3), 0.55, dtype=np.float64)  # default gray

    unique_labels = np.unique(vertex_labels)
    unique_labels = unique_labels[unique_labels >= 0]

    golden = 0.6180339887
    for i, label in enumerate(unique_labels):
        hue = (i * golden) % 1.0
        # HSV → RGB (S=0.75, V=0.92)
        h6 = hue * 6.0
        ki = int(h6)
        f  = h6 - ki
        s, v = 0.75, 0.92
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        rgb_map = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]
        r, g, b = rgb_map[ki % 6]
        colors[vertex_labels == label] = [r, g, b]

    debug = o3d.geometry.TriangleMesh(mesh)
    debug.vertex_colors = o3d.utility.Vector3dVector(colors)
    return debug


def segment_mesh_with_boundaries(mesh, is_boundary):
    """
    Segment mesh using a pre-computed per-vertex boundary mask.

    Boundary vertices act as barriers — edges crossing them are excluded so
    each connected stone interior becomes its own component.

    Uses scipy connected_components (C-level) instead of Python BFS, and
    builds the adjacency with vectorised numpy operations to avoid the memory
    overhead of Python list-of-sets.

    Args:
        mesh:        Triangle mesh to segment
        is_boundary: bool array [N_vertices] — True = seam vertex (barrier)

    Returns:
        vertex_labels: int32 [N_vertices] — component id ≥0, or -1 for boundary
    """
    print("\nSegmenting mesh using boundary constraints...")
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    vertices  = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    N = len(vertices)

    if not is_boundary.any():
        print("  ⚠ No boundary vertices — single component")
        # All non-boundary vertices get label 0
        vertex_labels = np.zeros(N, dtype=np.int32)
        return vertex_labels

    print(f"  Boundary vertices: {is_boundary.sum():,} / {N:,} "
          f"({100*is_boundary.sum()/N:.1f}%)")

    # Map non-boundary vertices to a compressed index space
    non_boundary = ~is_boundary
    non_boundary_idx = np.where(non_boundary)[0]
    M = len(non_boundary_idx)
    mapping = np.full(N, -1, dtype=np.int32)
    mapping[non_boundary_idx] = np.arange(M, dtype=np.int32)

    # Build COO edge arrays with pure numpy (no Python loops)
    print("  Building adjacency (vectorised)...")
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    e01 = non_boundary[v0] & non_boundary[v1]
    e12 = non_boundary[v1] & non_boundary[v2]
    e20 = non_boundary[v2] & non_boundary[v0]

    row = np.concatenate([mapping[v0[e01]], mapping[v1[e01]],
                          mapping[v1[e12]], mapping[v2[e12]],
                          mapping[v2[e20]], mapping[v0[e20]]])
    col = np.concatenate([mapping[v1[e01]], mapping[v0[e01]],
                          mapping[v2[e12]], mapping[v1[e12]],
                          mapping[v0[e20]], mapping[v2[e20]]])

    data = np.ones(len(row), dtype=np.float32)
    adj  = csr_matrix((data, (row, col)), shape=(M, M))

    print("  Finding connected components (scipy)...")
    n_components, compressed_labels = connected_components(
        adj, directed=False, connection='weak'
    )
    print(f"  Found {n_components} raw components")

    # Map compressed labels back to full vertex array; boundary stays -1
    vertex_labels = np.full(N, -1, dtype=np.int32)
    vertex_labels[non_boundary_idx] = compressed_labels

    return vertex_labels


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
                       help='Every Nth frame used for SAM3 reconstruction (TSDF uses all frames)')

    # Depth quality filter
    parser.add_argument('--depth_min', type=float, default=0.15,
                       help='Minimum valid depth in metres (default: 0.15)')
    parser.add_argument('--confidence_threshold', type=int, default=0,
                       help='Confidence filter threshold (default: 0 = disabled). '
                            'Requires confidence/ frames from 00_extract_frames.py. '
                            'D456 values: 1=drop zero-confidence, 2=keep high-confidence only.')

    # Segmentation parameters
    parser.add_argument('--mesh_keep_components', type=int, default=1,
                       help='Keep only the N largest connected components after TSDF '
                            'to remove floating noise (default: 8, 0 = disabled)')
    parser.add_argument('--alpha_threshold', type=float, default=0.1,
                       help='Alpha score below which a vertex is classified as seam/background '
                            '(default: 0.1). Higher = more aggressive culling of seam vertices.')
    parser.add_argument('--edt_gamma', type=float, default=0.5,
                       help='Gamma for EDT score falloff: score = (dist/max_dist)**gamma. '
                            '<1 (e.g. 0.5): interiors saturate quickly, sharp seams. '
                            '=1: linear. >1: slow rise, more conservative. (default: 0.5)')
    parser.add_argument('--boundary_propagation_hops', type=int, default=5,
                       help='BFS hops to propagate boundary scores outward from seam seeds '
                            '(default: 5). Larger values fatten the seam network and reduce '
                            'small island fragments. Set to 0 to disable propagation.')
    parser.add_argument('--min_cluster_size', type=int, default=1000,
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
    # Cache SAM3 masks beside the extracted frames so they survive
    # across multiple meshing runs from the same bag.
    cache_dir = os.path.join(args.frames_dir, 'sam3_mask_cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"SAM3 mask cache: {cache_dir}")

    # Pass 1: TSDF fusion (all frames, no SAM3)
    mesh = integrate_tsdf(
        args.frames_dir, intrinsic, poses,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        depth_min_m=args.depth_min,
        confidence_threshold=args.confidence_threshold,
    )

    # Clean mesh: drop small floating fragments before voting
    if args.mesh_keep_components > 0:
        mesh = clean_mesh_keep_largest(mesh, keep_n=args.mesh_keep_components)

    # All outputs go into the same directory; --output names the raw RGB mesh
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    raw_mesh_path = os.path.join(out_dir, 'raw_mesh_rgb.ply')
    o3d.io.write_triangle_mesh(raw_mesh_path, mesh)
    print(f"\n✓ Raw RGB mesh: {raw_mesh_path} "
          f"({os.path.getsize(raw_mesh_path)/(1024**2):.1f} MB)")

    # Pass 2a: EDT — compute alpha maps in parallel, save to alpha_dir
    color_files, _ = get_rgbd_file_lists(args.frames_dir)
    n_total  = min(len(color_files), len(poses))
    n_cached = sum(1 for i in range(n_total)
                   if os.path.exists(os.path.join(cache_dir, f"masks_{i:06d}.npz")))
    if n_cached >= n_total:
        print(f"\nSAM3 mask cache complete ({n_cached}/{n_total}) — skipping SAM3 init")
        processor = None
    else:
        print(f"\nSAM3 mask cache: {n_cached}/{n_total} frames — initializing SAM3")
        processor = initialize_sam3(args.sam_confidence)

    alpha_dir = os.path.join(out_dir, 'alpha_maps')
    precompute_alphas(
        cache_dir=cache_dir,
        alpha_dir=alpha_dir,
        color_files=color_files,
        n_frames=n_total,
        sam_prompt=args.sam_prompt,
        sam_max_size_ratio=args.sam_max_size_ratio,
        edt_gamma=args.edt_gamma,
        processor=processor,
    )

    # Pass 2b: semantic TSDF — reads pre-computed alphas, no EDT
    alpha_mesh = integrate_semantic_tsdf(
        args.frames_dir, intrinsic, poses,
        alpha_dir=alpha_dir,
        depth_scale=depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        depth_min_m=args.depth_min,
    )

    # Transfer alpha scores to raw mesh vertices (raw mesh RGB untouched)
    print("\nTransferring alpha scores to raw mesh...")
    alpha_scores = transfer_alpha_scores(mesh, alpha_mesh)
    is_boundary = alpha_scores < args.alpha_threshold
    n_seam = is_boundary.sum()
    print(f"  is_boundary (alpha < {args.alpha_threshold}): "
          f"{n_seam:,} / {len(alpha_scores):,} vertices ({100*n_seam/len(alpha_scores):.1f}%)")

    # score_map_mesh.ply — alpha score visualization (blue=interior, red=seam)
    score_map_path = os.path.join(out_dir, 'score_map_mesh.ply')
    score_mesh = make_boundary_score_mesh(mesh, 1.0 - alpha_scores)
    o3d.io.write_triangle_mesh(score_map_path, score_mesh)
    print(f"✓ Score map mesh: {score_map_path}")

    # culled_mesh_rgb.ply — raw RGB mesh with seam/background vertices removed
    print("\nApplying vertex culling...")
    culled_mesh = cull_mesh_by_alpha(mesh, alpha_scores, args.alpha_threshold)
    culled_mesh_path = os.path.join(out_dir, 'culled_mesh_rgb.ply')
    o3d.io.write_triangle_mesh(culled_mesh_path, culled_mesh)
    print(f"✓ Culled RGB mesh: {culled_mesh_path} "
          f"({os.path.getsize(culled_mesh_path)/(1024**2):.1f} MB)")

    # Optional BFS propagation to fill gaps in the seam network
    if args.boundary_propagation_hops > 0:
        print(f"\nPropagating boundary scores (max_hops={args.boundary_propagation_hops})...")
        boundary_scores  = propagate_boundary_scores(
            mesh, is_boundary, max_hops=args.boundary_propagation_hops
        )
        is_boundary_final = boundary_scores > 0
    else:
        is_boundary_final = is_boundary

    # Segment mesh using boundary vertex labels
    vertex_labels = segment_mesh_with_boundaries(mesh, is_boundary_final)

    # segments.ply — pseudo-color segment visualization
    segments_vis_path = os.path.join(out_dir, 'segments.ply')
    seg_color_mesh = make_segment_color_mesh(mesh, vertex_labels)
    o3d.io.write_triangle_mesh(segments_vis_path, seg_color_mesh)
    print(f"✓ Segments vis: {segments_vis_path}")

    # Stream individual segment meshes to disk one at a time to avoid OOM.
    # Each submesh is extracted, written, then freed before the next one.
    print(f"\nSaving segments to {args.segments_dir}/")
    os.makedirs(args.segments_dir, exist_ok=True)
    triangles = np.asarray(mesh.triangles)
    unique_labels = np.unique(vertex_labels)
    unique_labels = unique_labels[unique_labels >= 0]
    saved_count = 0

    for label in tqdm(unique_labels, desc="Saving segments"):
        tri_mask = np.all(vertex_labels[triangles] == label, axis=1)
        if tri_mask.sum() < args.min_cluster_size:
            vertex_labels[vertex_labels == label] = -1   # mark as boundary in debug mesh
            continue

        submesh = o3d.geometry.TriangleMesh(mesh)        # deep copy
        submesh.remove_triangles_by_mask(~tri_mask)
        submesh.remove_unreferenced_vertices()
        submesh.compute_vertex_normals()

        seg_path = os.path.join(args.segments_dir, f"segment_{label:04d}.ply")
        o3d.io.write_triangle_mesh(seg_path, submesh)
        saved_count += 1
        del submesh                                       # free before next iteration

    print(f"✓ Saved {saved_count} segments")

    print("\n" + "="*80)
    print("Reconstruction Complete!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  raw_mesh_rgb.ply    : {raw_mesh_path}")
    print(f"  score_map_mesh.ply  : {score_map_path}")
    print(f"  culled_mesh_rgb.ply : {culled_mesh_path}")
    print(f"  segments.ply        : {segments_vis_path}")
    print(f"  segments/           : {args.segments_dir}/segment_*.ply")


if __name__ == "__main__":
    main()
