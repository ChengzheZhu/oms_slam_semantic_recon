#!/usr/bin/env python3
"""
07_export_rhino.py — Export TSDF surface points, camera cones and the camera
trajectory in formats Rhino imports.

Outputs (in --output_dir):
  cloud.ply     TSDF surface points with RGB (+ normals): the vertices of the
                step 03 mesh, or of the step 03b batch meshes. A PLY without
                faces imports into Rhino as a point cloud.
  cameras.obj   One cone per posed frame, object cam_NNNNNN (= color/NNNNNN.jpg),
                plus object "trajectory": the camera centres of every posed
                frame in time order, as a degree-1 curve (a polyline in Rhino).

Not glTF/GLB: Rhino's glTF importer converts triangle primitives only, so the
points and the polyline would be dropped. PLY holds one object per file, hence
the cones and trajectory go in a separate OBJ.

Both files use the SLAM world frame (the frame of trajectory_open3d.log and the
step 03 mesh), scaled to --units. Neither format stores units, so import into
a Rhino document with the same units.

Cones follow the OpenCV camera convention (x right, y down, z forward): the
apex is the camera centre and the base is the image rectangle at depth
--cone_depth, shaped by the intrinsics. A small triangle beyond the base's top
edge (image row 0, -y) marks image "up".

Usage:
  python scripts/07_export_rhino.py \\
      --frames_dir output/<run>/frames \\
      --trajectory output/<run>/sparse/trajectory_open3d.log \\
      --mesh       output/<run>/raw_mesh_rgb.ply \\
      --output_dir output/<run>/export
"""

import argparse
import os

import numpy as np
import open3d as o3d

from pipeline_common import get_rgbd_file_lists, load_frame_poses, load_intrinsic

UNIT_SCALE = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}


def load_surface_points(mesh_files, voxel):
    """Vertices (+ colours, normals) of one or more mesh PLYs, merged."""
    pcd = o3d.geometry.PointCloud()
    for path in mesh_files:
        if not os.path.isfile(path):           # Open3D only warns on a missing file
            raise SystemExit(f"ERROR: mesh not found: {path}")
        part = o3d.io.read_point_cloud(path)   # vertex element only; faces ignored
        print(f"  {os.path.basename(path)}: {len(part.points):,} points")
        pcd += part
    if voxel > 0:
        n = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel)
        print(f"  Voxel {voxel} m: {n:,} -> {len(pcd.points):,} points")
    return pcd


def write_cloud_ply(path, pcd, scale, with_normals):
    """Binary PLY point cloud: float32 xyz [+ nx ny nz], uchar rgb."""
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if with_normals:
        fields += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
    if pcd.has_colors():
        fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    data = np.empty(len(pcd.points), dtype=fields)
    xyz = np.asarray(pcd.points) * scale
    data['x'], data['y'], data['z'] = xyz.T
    if with_normals:
        nrm = np.asarray(pcd.normals)
        length = np.linalg.norm(nrm, axis=1, keepdims=True)
        # Voxel averaging cancels opposite normals (both sides of a thin
        # surface in one voxel); those points get a zero normal.
        degenerate = length[:, 0] < 1e-6
        nrm = np.where(degenerate[:, None], 0.0, nrm / np.maximum(length, 1e-6))
        if degenerate.any():
            print(f"  {degenerate.sum():,} points with no defined normal (written as 0 0 0)")
        data['nx'], data['ny'], data['nz'] = nrm.T
    if pcd.has_colors():
        rgb = np.clip(np.asarray(pcd.colors) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        data['red'], data['green'], data['blue'] = rgb.T

    ply_type = {'<f4': 'float', 'u1': 'uchar'}
    header = ['ply', 'format binary_little_endian 1.0',
              'comment 07_export_rhino.py: TSDF surface points, SLAM world frame',
              f'element vertex {len(data)}']
    header += [f'property {ply_type[t]} {name}' for name, t in fields]
    header += ['end_header']
    with open(path, 'wb') as f:
        f.write(('\n'.join(header) + '\n').encode('ascii'))
        f.write(data.tobytes())


def cone_template(intrinsic, depth):
    """Cone vertices in the camera frame (OpenCV) and its triangles.

    0 apex; 1-4 image corners TL, TR, BR, BL at z = depth; 5-7 the "up"
    marker triangle above the top edge (-y).
    """
    w, h = intrinsic.width, intrinsic.height
    K = intrinsic.intrinsic_matrix
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def ray(u, v):
        return [(u - cx) / fx * depth, (v - cy) / fy * depth, depth]

    verts = np.array([[0.0, 0.0, 0.0],
                      ray(0, 0), ray(w, 0), ray(w, h), ray(0, h),
                      ray(0.35 * w, 0), ray(0.65 * w, 0), ray(0.5 * w, -0.2 * h)])
    # Side faces wound for outward normals; base faces +z.
    faces = np.array([[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
                      [1, 2, 3], [1, 3, 4],
                      [5, 6, 7]])
    return verts, faces


def write_cameras_obj(path, names, poses, cone_every, intrinsic, cone_depth, scale):
    """Cones (one OBJ object per sampled pose) + trajectory curve (all poses)."""
    tverts, tfaces = cone_template(intrinsic, cone_depth)
    tverts_h = np.c_[tverts, np.ones(len(tverts))]
    centres = np.array([T[:3, 3] for T in poses]) * scale
    lines = ['# 07_export_rhino.py: camera cones (OpenCV: x right, y down, '
             'z forward; marker = image top) + trajectory',
             '# SLAM world frame; cam_NNNNNN = frames/color/NNNNNN.jpg']
    v_base = 1                      # OBJ vertex indices are global and 1-based
    n_cones = 0
    for k, (name, T) in enumerate(zip(names, poses)):
        if k % cone_every:
            continue
        world = (tverts_h @ T.T)[:, :3] * scale
        lines += [f'o {name}', f'g {name}']
        lines += [f'v {x:.6f} {y:.6f} {z:.6f}' for x, y, z in world]
        lines += [f'f {a + v_base} {b + v_base} {c + v_base}' for a, b, c in tfaces]
        v_base += len(world)
        n_cones += 1

    n = len(centres)
    if n < 2:
        print("  WARNING: fewer than 2 poses; no trajectory written")
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return n_cones
    lines += ['o trajectory', 'g trajectory']
    lines += [f'v {x:.6f} {y:.6f} {z:.6f}' for x, y, z in centres]
    # Degree-1 B-spline through the centres; knot vector clamped, length n + 2.
    knots = [0] + list(range(n)) + [n - 1]
    lines += ['cstype bspline', 'deg 1',
              f'curv 0 {n - 1} ' + ' '.join(str(v_base + i) for i in range(n)),
              'parm u ' + ' '.join(str(k) for k in knots),
              'end']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return n_cones


def main():
    p = argparse.ArgumentParser(
        description='Export TSDF points, camera cones and trajectory for Rhino.')
    p.add_argument('--frames_dir', required=True,
                   help='Step 01 frames (intrinsic.json, timestamps.txt, color/)')
    p.add_argument('--trajectory', required=True, help='trajectory_open3d.log (step 02)')
    p.add_argument('--mesh', nargs='+', required=True,
                   help='Mesh PLY(s) whose vertices form the point cloud '
                        '(step 03 raw mesh, or step 03b batch meshes)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--intrinsic', default=None,
                   help='Default: <frames_dir>/intrinsic.json')
    p.add_argument('--units', choices=sorted(UNIT_SCALE), default='m',
                   help='Units of the written coordinates (default: m)')
    p.add_argument('--voxel', type=float, default=0.0,
                   help='Voxel downsample of the point cloud, metres (0 = off)')
    p.add_argument('--normals', action='store_true', help='Write point normals')
    p.add_argument('--cone_depth', type=float, default=0.05,
                   help='Cone length (apex to image plane), metres')
    p.add_argument('--cone_every', type=int, default=1,
                   help='Cone for every Nth posed frame (trajectory keeps all)')
    args = p.parse_args()

    scale = UNIT_SCALE[args.units]
    os.makedirs(args.output_dir, exist_ok=True)
    intrinsic, _ = load_intrinsic(args.intrinsic or
                                  os.path.join(args.frames_dir, 'intrinsic.json'))

    print("Point cloud")
    pcd = load_surface_points(args.mesh, args.voxel)
    with_normals = args.normals and pcd.has_normals()
    if args.normals and not with_normals:
        print("  WARNING: input has no normals; writing points without them")
    cloud_path = os.path.join(args.output_dir, 'cloud.ply')
    write_cloud_ply(cloud_path, pcd, scale, with_normals)
    print(f"  -> {cloud_path} ({len(pcd.points):,} points, "
          f"{os.path.getsize(cloud_path) / 1e6:.0f} MB)")

    print("Cameras")
    poses_all = load_frame_poses(args.trajectory, args.frames_dir)
    color_files, _ = get_rgbd_file_lists(args.frames_dir)
    posed = [(f"cam_{os.path.splitext(os.path.basename(color_files[i]))[0]}", T)
             for i, T in enumerate(poses_all) if T is not None]
    names, poses = zip(*posed)
    cams_path = os.path.join(args.output_dir, 'cameras.obj')
    n_cones = write_cameras_obj(cams_path, names, poses, max(1, args.cone_every),
                                intrinsic, args.cone_depth, scale)
    print(f"  -> {cams_path} ({n_cones} cones, trajectory {len(poses)} poses)")
    print(f"Units: {args.units} — import into a Rhino document in {args.units}.")


if __name__ == '__main__':
    main()
