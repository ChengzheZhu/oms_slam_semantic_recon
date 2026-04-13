#!/usr/bin/env python3
"""
Export reconstruction mesh + camera trajectory as a GLB for geometry inspection.

Mirrors the visual style of the MASt3R+SAM3 pipeline's save_raw_ply:
  - Mesh with original vertex colours from TSDF
  - Camera positions as small spheres, rainbow-coloured by frame index
  - Optionally subsample cameras so the file stays manageable

Usage:
    python scripts/export_debug_glb.py \
        --mesh   output/<dataset>/mesh.ply \
        --traj   output/<dataset>/sparse/trajectory_open3d.log \
        --output output/<dataset>/debug.glb \
        [--cam_stride 5]        # show every 5th camera (default: auto)
        [--cam_radius 0.02]     # sphere radius in metres (default: 0.02)
"""

import argparse
import os
import numpy as np
import open3d as o3d


# ── rainbow colour (matches components pipeline's _frame_rainbow) ─────────────

def _rainbow(i: int, n: int) -> np.ndarray:
    t = i / max(n - 1, 1)
    r = max(0.0, min(1.0, 1.5 - abs(3 * t - 0)))
    g = max(0.0, min(1.0, 1.5 - abs(3 * t - 1)))
    b = max(0.0, min(1.0, 1.5 - abs(3 * t - 2)))
    return np.array([r, g, b], dtype=np.float64)


# ── trajectory loader ─────────────────────────────────────────────────────────

def load_trajectory(log_file: str) -> list:
    """Load Open3D trajectory log → list of 4×4 camera-to-world matrices."""
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


# ── camera marker geometry ────────────────────────────────────────────────────

def _camera_frustum(pose: np.ndarray, colour: np.ndarray,
                    radius: float = 0.02) -> o3d.geometry.TriangleMesh:
    """
    Small sphere at camera origin + a short line in the look-direction,
    all merged into one TriangleMesh for GLB export.

    pose: 4×4 camera-to-world (translation column = camera position in world)
    """
    cam_pos = pose[:3, 3]
    look_dir = pose[:3, 2]          # +Z axis = principal ray in OpenCV convention

    # Sphere at camera centre
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
    sphere.translate(cam_pos)
    sphere.paint_uniform_color(colour)

    # Tiny cone pointing in look direction (depth = 3× radius)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=radius * 0.4,
                                                  height=radius * 3.0,
                                                  resolution=8)
    # Default cone points along +Y; rotate so it aligns with look_dir
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, look_dir)
    axis_len = np.linalg.norm(axis)
    if axis_len > 1e-6:
        axis /= axis_len
        angle = np.arccos(np.clip(np.dot(up, look_dir), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cone.rotate(R, center=np.zeros(3))
    cone.translate(cam_pos + look_dir * radius)
    cone.paint_uniform_color(colour * 0.7)   # slightly darker than sphere

    return sphere + cone


# ── main export ───────────────────────────────────────────────────────────────

def export_debug_glb(mesh_path: str, traj_path: str, output_path: str,
                     cam_stride: int = 0, cam_radius: float = 0.02) -> None:

    print("="*70)
    print("Debug GLB Export")
    print("="*70)

    # Load mesh
    print(f"\nLoading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print(f"  Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")

    # Load trajectory
    print(f"\nLoading trajectory: {traj_path}")
    poses = load_trajectory(traj_path)
    print(f"  Poses: {len(poses)}")

    # Auto cam_stride so we show ≤ 200 cameras in the file
    if cam_stride <= 0:
        cam_stride = max(1, len(poses) // 200)
    selected = list(range(0, len(poses), cam_stride))
    print(f"  Showing {len(selected)} cameras (stride={cam_stride})")

    # Build camera markers
    scene_mesh = mesh
    n = len(poses)
    for i, idx in enumerate(selected):
        colour = _rainbow(idx, n)
        marker = _camera_frustum(poses[idx], colour, radius=cam_radius)
        scene_mesh = scene_mesh + marker

    scene_mesh.compute_vertex_normals()

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"\nExporting GLB: {output_path}")
    ok = o3d.io.write_triangle_mesh(output_path, scene_mesh,
                                    write_vertex_normals=True,
                                    write_vertex_colors=True)
    if ok:
        size_mb = os.path.getsize(output_path) / 1024**2
        print(f"✓ Saved  ({size_mb:.1f} MB)")
        print(f"\nOpen with: https://gltf-viewer.donmccurdy.com  or  Blender / MeshLab")
    else:
        print("✗ Export failed — check that Open3D >= 0.16 is installed")

    print("="*70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export mesh + cameras as GLB")
    parser.add_argument('--mesh',   required=True, help='Input mesh PLY')
    parser.add_argument('--traj',   required=True, help='Trajectory log (Open3D format)')
    parser.add_argument('--output', required=True, help='Output .glb path')
    parser.add_argument('--cam_stride', type=int, default=0,
                        help='Show every Nth camera (0 = auto, max 200 shown)')
    parser.add_argument('--cam_radius', type=float, default=0.02,
                        help='Camera sphere radius in metres (default: 0.02)')
    args = parser.parse_args()

    export_debug_glb(
        mesh_path=args.mesh,
        traj_path=args.traj,
        output_path=args.output,
        cam_stride=args.cam_stride,
        cam_radius=args.cam_radius,
    )


if __name__ == '__main__':
    main()
