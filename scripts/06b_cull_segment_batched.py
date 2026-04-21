#!/usr/bin/env python3
"""
06b_cull_segment_batched.py — Per-batch score transfer + cull; no full-mesh load.

Consumes the separate batch directories produced by 03b and 05b.
For each batch pair (rgb_batch_NNNN.ply / alpha_batch_NNNN.ply):
  1. Load both small batch meshes
  2. Chunked KD-tree score transfer (alpha → rgb)
  3. Cull triangles below threshold
  4. Save culled_batch_NNNN.ply
  5. Delete both meshes, gc.collect()

Peak RAM = 1 RGB batch + 1 alpha batch + small score array.
No single large mesh is ever held in memory.

Reads:
  --rgb_batches_dir     <output_dir>/batches/       (from 03b)
  --alpha_batches_dir   <output_dir>/alpha_batches/ (from 05b)

Writes to <output_dir>/culled_batches/:
  culled_batch_NNNN.ply  — culled RGB mesh per batch
  index.json             — frame range + file list

score_map_batch_NNNN.ply (diagnostic) written if --save_score_maps is set.
"""

import open3d as o3d
import numpy as np
import argparse
import os
import gc
import json
from tqdm import tqdm
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────────────────────
# Score transfer (chunked)
# ─────────────────────────────────────────────────────────────────────────────

def transfer_alpha_scores_chunked(raw_mesh, alpha_mesh, chunk_size=500_000):
    raw_pts   = np.asarray(raw_mesh.vertices)
    alpha_pts = np.asarray(alpha_mesh.vertices)
    alpha_col = np.asarray(alpha_mesh.vertex_colors)   # R=G=B=score

    n_raw  = len(raw_pts)
    tree   = cKDTree(alpha_pts)
    scores = np.empty(n_raw, dtype=np.float32)

    starts = list(range(0, n_raw, chunk_size))
    if len(starts) > 1:
        for b_start in tqdm(starts, desc="  Score transfer"):
            b_end = min(b_start + chunk_size, n_raw)
            _, idx = tree.query(raw_pts[b_start:b_end], k=1, workers=-1)
            scores[b_start:b_end] = alpha_col[idx, 0].astype(np.float32)
    else:
        _, idx = tree.query(raw_pts, k=1, workers=-1)
        scores[:] = alpha_col[idx, 0].astype(np.float32)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Culling
# ─────────────────────────────────────────────────────────────────────────────

def cull_mesh_by_alpha(mesh, alpha_scores, alpha_threshold):
    triangles = np.asarray(mesh.triangles)
    keep      = np.all(alpha_scores[triangles] >= alpha_threshold, axis=1)
    culled    = o3d.geometry.TriangleMesh(mesh)
    culled.remove_triangles_by_mask(~keep)
    culled.remove_unreferenced_vertices()
    culled.compute_vertex_normals()
    n_kept = int(keep.sum())
    pct    = 100 * n_kept / max(len(triangles), 1)
    print(f"    culled: {n_kept:,}/{len(triangles):,} tris kept ({pct:.1f}%)")
    return culled


# ─────────────────────────────────────────────────────────────────────────────
# Score-map diagnostic mesh
# ─────────────────────────────────────────────────────────────────────────────

def make_score_map_mesh(mesh, scores):
    interior = np.array([0.15, 0.35, 0.85])
    seam     = np.array([0.90, 0.15, 0.10])
    s        = np.clip(1.0 - scores, 0.0, 1.0)[:, None]
    colors   = (1 - s) * interior + s * seam
    debug    = o3d.geometry.TriangleMesh(mesh)
    debug.vertex_colors = o3d.utility.Vector3dVector(colors)
    return debug


# ─────────────────────────────────────────────────────────────────────────────
# Per-batch pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_batches(rgb_batches_dir, alpha_batches_dir, output_dir,
                    alpha_threshold=0.5, chunk_size=500_000,
                    save_score_maps=False):

    # Load indices
    with open(os.path.join(rgb_batches_dir,   'index.json')) as f:
        rgb_index = json.load(f)
    with open(os.path.join(alpha_batches_dir, 'index.json')) as f:
        alpha_index = json.load(f)

    n_rgb   = rgb_index['n_batches']
    n_alpha = alpha_index['n_batches']
    if n_rgb != n_alpha:
        raise ValueError(
            f"Batch count mismatch: rgb={n_rgb}, alpha={n_alpha}. "
            "Run 03b and 05b with identical --batch_size and --batch_overlap.")

    culled_dir = os.path.join(output_dir, 'culled_batches')
    os.makedirs(culled_dir, exist_ok=True)
    print(f"  {n_rgb} batch pairs  threshold={alpha_threshold}")
    print(f"  Saving culled batches → {culled_dir}/")

    index_entries = []

    for b in range(n_rgb):
        re = rgb_index['batches'][b]
        ae = alpha_index['batches'][b]

        rgb_ply   = os.path.join(rgb_batches_dir,   re['ply'])
        alpha_ply = os.path.join(alpha_batches_dir, ae['ply'])
        culled_name = f"culled_batch_{b:04d}.ply"
        culled_ply  = os.path.join(culled_dir, culled_name)

        print(f"\n── Batch {b+1}/{n_rgb}  frames [{re['frame_start']}:{re['frame_end']}] ──")

        if os.path.exists(culled_ply):
            print(f"  already exists, skipping → {culled_ply}")
            index_entries.append({
                "id": f"{b:04d}",
                "frame_start": re['frame_start'],
                "frame_end":   re['frame_end'],
                "n_frames":    re['n_frames'],
                "rgb_ply":     re['ply'],
                "alpha_ply":   ae['ply'],
                "culled_ply":  culled_name,
            })
            continue

        rgb_mesh   = o3d.io.read_triangle_mesh(rgb_ply)
        alpha_mesh = o3d.io.read_triangle_mesh(alpha_ply)
        print(f"  rgb:   {len(rgb_mesh.vertices):,} verts")
        print(f"  alpha: {len(alpha_mesh.vertices):,} verts")

        scores = transfer_alpha_scores_chunked(rgb_mesh, alpha_mesh, chunk_size)
        del alpha_mesh
        gc.collect()

        if save_score_maps:
            score_map = make_score_map_mesh(rgb_mesh, scores)
            sm_path   = os.path.join(culled_dir, f"score_map_batch_{b:04d}.ply")
            o3d.io.write_triangle_mesh(sm_path, score_map)
            del score_map

        culled = cull_mesh_by_alpha(rgb_mesh, scores, alpha_threshold)
        del rgb_mesh, scores
        gc.collect()

        o3d.io.write_triangle_mesh(culled_ply, culled)
        print(f"  saved → {culled_ply}  "
              f"({os.path.getsize(culled_ply)/1024**2:.1f} MB)")
        del culled
        gc.collect()

        index_entries.append({
            "id": f"{b:04d}",
            "frame_start": re['frame_start'],
            "frame_end":   re['frame_end'],
            "n_frames":    re['n_frames'],
            "rgb_ply":     re['ply'],
            "alpha_ply":   ae['ply'],
            "culled_ply":  culled_name,
        })

    index = {
        "alpha_threshold": alpha_threshold,
        "voxel_size": rgb_index.get('voxel_size'),
        "total_frames": rgb_index.get('total_frames'),
        "batch_size": rgb_index.get('batch_size'),
        "batch_overlap": rgb_index.get('batch_overlap'),
        "n_batches": n_rgb,
        "batches": index_entries,
    }
    index_path = os.path.join(culled_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"\n✓ index.json → {index_path}")
    return culled_dir, index


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 06b — Per-batch score transfer + cull (no full-mesh load)")
    parser.add_argument('--rgb_batches_dir',   required=True,
                        help='batches/ directory from 03b (contains index.json)')
    parser.add_argument('--alpha_batches_dir', required=True,
                        help='alpha_batches/ directory from 05b (contains index.json)')
    parser.add_argument('--output_dir',        required=True,
                        help='Directory to write culled_batches/ into')
    parser.add_argument('--alpha_threshold',   type=float, default=0.5,
                        help='Cull triangles with score below this value (default 0.5)')
    parser.add_argument('--chunk_size',        type=int,   default=500_000,
                        help='Vertices per KD-tree query chunk (default 500000)')
    parser.add_argument('--save_score_maps',   action='store_true',
                        help='Also save score_map_batch_NNNN.ply for diagnostics')
    args = parser.parse_args()

    for label, d in [("rgb_batches_dir",   args.rgb_batches_dir),
                     ("alpha_batches_dir", args.alpha_batches_dir)]:
        if not os.path.isdir(d):
            print(f"ERROR: {label} not found: {d}")
            import sys; sys.exit(1)
        if not os.path.exists(os.path.join(d, 'index.json')):
            print(f"ERROR: index.json missing in {d}")
            import sys; sys.exit(1)

    print("=" * 60)
    print("Step 06b — Per-batch Cull (no full-mesh load)")
    print("=" * 60)
    print(f"  rgb_batches_dir   : {args.rgb_batches_dir}")
    print(f"  alpha_batches_dir : {args.alpha_batches_dir}")
    print(f"  output_dir        : {args.output_dir}")
    print(f"  alpha_threshold   : {args.alpha_threshold}")
    print(f"  chunk_size        : {args.chunk_size:,}")

    os.makedirs(args.output_dir, exist_ok=True)

    culled_dir, index = process_batches(
        rgb_batches_dir   = args.rgb_batches_dir,
        alpha_batches_dir = args.alpha_batches_dir,
        output_dir        = args.output_dir,
        alpha_threshold   = args.alpha_threshold,
        chunk_size        = args.chunk_size,
        save_score_maps   = args.save_score_maps,
    )

    print("\n" + "=" * 60)
    print("Step 06b complete")
    print(f"  {index['n_batches']} culled batch meshes in {culled_dir}/")
    print(f"  index.json records frame ranges for downstream QR queries")


if __name__ == "__main__":
    main()
