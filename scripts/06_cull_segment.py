#!/usr/bin/env python3
"""
06_cull_segment.py — Transfer alpha scores, cull seam triangles, and segment
the wall into individual stone submeshes.

Reads:
  --raw_mesh    raw_mesh_rgb.ply        (step 03 output)
  --alpha_mesh  alpha_mesh.ply          (step 05 output)

For each alpha threshold value:
  1. Transfer alpha scores from alpha_mesh → raw_mesh vertices (KD-tree)
  2. Optionally clean small floating fragments (keep N largest components)
  3. Optionally propagate boundary (BFS hop fattening)
  4. Cull triangles whose vertices fall below the threshold
  5. Segment remaining mesh into connected stone patches

Outputs per threshold in <output_dir>/thresh_<t>/:
  culled_mesh_rgb.ply   — culled RGB mesh
  segments.ply          — pseudo-colour segment visualisation
  sam3_segments/        — individual stone PLY submeshes

Shared outputs in <output_dir>/:
  score_map_mesh.ply    — raw mesh coloured by alpha score (diagnostic)

Usage:
  python scripts/06_cull_segment.py \\
      --raw_mesh   output/raw_mesh_rgb.ply \\
      --alpha_mesh output/scoring/alpha_mesh.ply \\
      --output_dir output/segments
"""

import open3d as o3d
import numpy as np
import argparse
import os
from tqdm import tqdm
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Score transfer
# ---------------------------------------------------------------------------

def transfer_alpha_scores(raw_mesh, alpha_mesh):
    """KD-tree nearest-neighbour: alpha_mesh vertex scores → raw_mesh vertices."""
    raw_pts   = np.asarray(raw_mesh.vertices)
    alpha_pts = np.asarray(alpha_mesh.vertices)
    alpha_col = np.asarray(alpha_mesh.vertex_colors)   # R=G=B=score

    tree = cKDTree(alpha_pts)
    _, idx = tree.query(raw_pts, k=1, workers=-1)
    scores = alpha_col[idx, 0].astype(np.float32)
    print(f"  Score transfer: min={scores.min():.3f}  "
          f"mean={scores.mean():.3f}  max={scores.max():.3f}")
    return scores


# ---------------------------------------------------------------------------
# Mesh cleaning
# ---------------------------------------------------------------------------

def clean_mesh_keep_largest(mesh, keep_n=1):
    print(f"\n  Keeping top {keep_n} connected components…")
    tri_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
    tri_clusters    = np.asarray(tri_clusters)
    cluster_n_tri   = np.asarray(cluster_n_tri)
    ranked   = np.argsort(cluster_n_tri)[::-1]
    keep_ids = set(ranked[:keep_n].tolist())
    mask     = np.array([c in keep_ids for c in tri_clusters])
    mesh.remove_triangles_by_mask(~mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    print(f"  Triangles after clean: {len(np.asarray(mesh.triangles)):,}")
    return mesh


# ---------------------------------------------------------------------------
# Boundary propagation (BFS)
# ---------------------------------------------------------------------------

def propagate_boundary_scores(mesh, is_boundary_seed, max_hops=5):
    from collections import deque
    triangles = np.asarray(mesh.triangles)
    N         = len(np.asarray(mesh.vertices))
    adjacency = [[] for _ in range(N)]
    for tri in triangles:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        adjacency[v0].append(v1); adjacency[v1].append(v0)
        adjacency[v1].append(v2); adjacency[v2].append(v1)
        adjacency[v2].append(v0); adjacency[v0].append(v2)

    scores   = np.zeros(N, dtype=np.float32)
    distance = np.full(N, -1, dtype=np.int32)
    seeds    = np.where(is_boundary_seed)[0]
    scores[seeds]   = 1.0
    distance[seeds] = 0
    queue    = deque(seeds.tolist())
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
    n_exp = int((scores > 0).sum())
    print(f"  Boundary propagation: {len(seeds):,} seeds → {n_exp:,} vertices "
          f"(max_hops={max_hops})")
    return scores


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_boundary_score_mesh(mesh, boundary_scores):
    interior = np.array([0.15, 0.35, 0.85])
    seam     = np.array([0.90, 0.15, 0.10])
    s        = np.clip(boundary_scores, 0.0, 1.0)[:, None]
    colors   = (1 - s) * interior + s * seam
    debug    = o3d.geometry.TriangleMesh(mesh)
    debug.vertex_colors = o3d.utility.Vector3dVector(colors)
    return debug


def make_segment_color_mesh(mesh, vertex_labels):
    n_vertices = len(np.asarray(mesh.vertices))
    colors     = np.full((n_vertices, 3), 0.55)
    unique     = np.unique(vertex_labels)
    unique     = unique[unique >= 0]
    golden     = 0.6180339887
    for i, label in enumerate(unique):
        hue = (i * golden) % 1.0
        h6  = hue * 6.0
        ki  = int(h6);  f = h6 - ki
        s, v = 0.75, 0.92
        p, q, t = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
        r, g, b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][ki % 6]
        colors[vertex_labels == label] = [r, g, b]
    debug = o3d.geometry.TriangleMesh(mesh)
    debug.vertex_colors = o3d.utility.Vector3dVector(colors)
    return debug


# ---------------------------------------------------------------------------
# Culling
# ---------------------------------------------------------------------------

def cull_mesh_by_alpha(mesh, alpha_scores, alpha_threshold):
    triangles = np.asarray(mesh.triangles)
    keep      = np.all(alpha_scores[triangles] >= alpha_threshold, axis=1)
    culled    = o3d.geometry.TriangleMesh(mesh)
    culled.remove_triangles_by_mask(~keep)
    culled.remove_unreferenced_vertices()
    culled.compute_vertex_normals()
    n_kept = int(keep.sum())
    print(f"  Culling: {n_kept:,}/{len(triangles):,} triangles kept "
          f"({100*n_kept/len(triangles):.1f}%)")
    return culled


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_mesh_with_boundaries(mesh, is_boundary):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    print("\n  Segmenting…")
    vertices  = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    N = len(vertices)

    if not is_boundary.any():
        print("  ⚠ No boundary vertices — single component")
        return np.zeros(N, dtype=np.int32)

    print(f"  Boundary vertices: {is_boundary.sum():,}/{N:,} "
          f"({100*is_boundary.sum()/N:.1f}%)")

    non_boundary     = ~is_boundary
    non_boundary_idx = np.where(non_boundary)[0]
    M       = len(non_boundary_idx)
    mapping = np.full(N, -1, dtype=np.int32)
    mapping[non_boundary_idx] = np.arange(M, dtype=np.int32)

    v0, v1, v2 = triangles[:,0], triangles[:,1], triangles[:,2]
    e01 = non_boundary[v0] & non_boundary[v1]
    e12 = non_boundary[v1] & non_boundary[v2]
    e20 = non_boundary[v2] & non_boundary[v0]

    row  = np.concatenate([mapping[v0[e01]], mapping[v1[e01]],
                           mapping[v1[e12]], mapping[v2[e12]],
                           mapping[v2[e20]], mapping[v0[e20]]])
    col  = np.concatenate([mapping[v1[e01]], mapping[v0[e01]],
                           mapping[v2[e12]], mapping[v1[e12]],
                           mapping[v0[e20]], mapping[v2[e20]]])
    adj  = csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(M, M))
    n_comp, labels = connected_components(adj, directed=False, connection='weak')
    print(f"  Found {n_comp} raw components")

    vertex_labels = np.full(N, -1, dtype=np.int32)
    vertex_labels[non_boundary_idx] = labels
    return vertex_labels


# ---------------------------------------------------------------------------
# Per-threshold runner
# ---------------------------------------------------------------------------

def run_threshold(mesh, alpha_scores, threshold, hops, min_cluster_size,
                  thresh_dir, save_segments=True):
    os.makedirs(thresh_dir, exist_ok=True)
    is_boundary = alpha_scores < threshold
    n_seam = is_boundary.sum()
    print(f"\n── threshold={threshold:g}  seam: {n_seam:,}/{len(alpha_scores):,} "
          f"({100*n_seam/len(alpha_scores):.1f}%) ──")
    print(f"   {thresh_dir}")

    if hops > 0:
        boundary_scores   = propagate_boundary_scores(mesh, is_boundary, hops)
        is_boundary_final = boundary_scores > 0
    else:
        is_boundary_final = is_boundary

    culled = cull_mesh_by_alpha(mesh, alpha_scores, threshold)
    culled_path = os.path.join(thresh_dir, 'culled_mesh_rgb.ply')
    o3d.io.write_triangle_mesh(culled_path, culled)
    print(f"  ✓ culled_mesh_rgb.ply  ({os.path.getsize(culled_path)/1024**2:.1f} MB)")
    del culled

    if not save_segments:
        return

    vertex_labels = segment_mesh_with_boundaries(mesh, is_boundary_final)

    seg_color = make_segment_color_mesh(mesh, vertex_labels)
    o3d.io.write_triangle_mesh(os.path.join(thresh_dir, 'segments.ply'), seg_color)
    print(f"  ✓ segments.ply")
    del seg_color

    segments_dir  = os.path.join(thresh_dir, 'sam3_segments')
    os.makedirs(segments_dir, exist_ok=True)
    triangles     = np.asarray(mesh.triangles)
    unique_labels = np.unique(vertex_labels)
    unique_labels = unique_labels[unique_labels >= 0]
    saved = 0
    for label in tqdm(unique_labels, desc=f"  Segments (t={threshold:g})"):
        tri_mask = np.all(vertex_labels[triangles] == label, axis=1)
        if tri_mask.sum() < min_cluster_size:
            continue
        sub = o3d.geometry.TriangleMesh(mesh)
        sub.remove_triangles_by_mask(~tri_mask)
        sub.remove_unreferenced_vertices()
        sub.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            os.path.join(segments_dir, f"segment_{label:04d}.ply"), sub)
        saved += 1
        del sub
    print(f"  ✓ {saved} segments saved → {segments_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 06 — Cull seams and segment into stones")
    parser.add_argument('--raw_mesh',    required=True,
                        help='raw_mesh_rgb.ply (step 03 output)')
    parser.add_argument('--alpha_mesh',  required=True,
                        help='alpha_mesh.ply (step 05 output)')
    parser.add_argument('--output_dir',  required=True,
                        help='Directory for score_map_mesh.ply and thresh_*/')
    parser.add_argument('--alpha_thresholds', type=float, nargs='+', default=[0.3],
                        help='One or more thresholds to sweep (default: 0.3)')
    parser.add_argument('--mesh_keep_components', type=int, default=1,
                        help='Keep N largest mesh components before scoring (0=off)')
    parser.add_argument('--boundary_propagation_hops', type=int, default=0,
                        help='BFS hops to fatten seam network (0=off)')
    parser.add_argument('--min_cluster_size', type=int, default=1000,
                        help='Minimum triangles per saved segment')
    parser.add_argument('--skip_segments', action='store_true',
                        help='Skip per-stone segment export (threshold tuning mode)')
    args = parser.parse_args()

    print("=" * 60)
    print("Step 06 — Cull + Segment")
    print("=" * 60)
    print(f"  raw_mesh   : {args.raw_mesh}")
    print(f"  alpha_mesh : {args.alpha_mesh}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  thresholds : {args.alpha_thresholds}")

    for label, path in [("raw_mesh", args.raw_mesh), ("alpha_mesh", args.alpha_mesh)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            import sys; sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading meshes…")
    raw_mesh   = o3d.io.read_triangle_mesh(args.raw_mesh)
    alpha_mesh = o3d.io.read_triangle_mesh(args.alpha_mesh)
    raw_mesh.compute_vertex_normals()
    print(f"  raw_mesh   : {len(raw_mesh.vertices):,} vertices")
    print(f"  alpha_mesh : {len(alpha_mesh.vertices):,} vertices")

    if args.mesh_keep_components > 0:
        raw_mesh = clean_mesh_keep_largest(raw_mesh, args.mesh_keep_components)

    print("\nTransferring alpha scores…")
    alpha_scores = transfer_alpha_scores(raw_mesh, alpha_mesh)
    del alpha_mesh

    score_map = make_boundary_score_mesh(raw_mesh, 1.0 - alpha_scores)
    score_path = os.path.join(args.output_dir, 'score_map_mesh.ply')
    o3d.io.write_triangle_mesh(score_path, score_map)
    print(f"  ✓ score_map_mesh.ply → {score_path}")
    del score_map

    thresholds = sorted(set(args.alpha_thresholds))
    for t in thresholds:
        run_threshold(
            raw_mesh, alpha_scores,
            threshold=t,
            hops=args.boundary_propagation_hops,
            min_cluster_size=args.min_cluster_size,
            thresh_dir=os.path.join(args.output_dir, f"thresh_{t:g}"),
            save_segments=not args.skip_segments,
        )

    print("\n" + "=" * 60)
    print("Step 06 complete")
    print(f"  score_map_mesh.ply : {score_path}")
    for t in thresholds:
        print(f"  thresh_{t:g}/       : {os.path.join(args.output_dir, f'thresh_{t:g}')}/")


if __name__ == "__main__":
    main()
