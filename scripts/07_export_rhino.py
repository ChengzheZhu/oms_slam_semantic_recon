#!/usr/bin/env python3
"""
07_export_rhino.py — Export camera cones and the camera trajectory as an OBJ
for Rhino.

Output (in --output_dir):
  cameras.obj   One cone per posed frame, object cam_NNNNNN (= color/NNNNNN.jpg),
                plus object "trajectory": the camera centres of every posed
                frame in time order, as a degree-1 curve (a polyline in Rhino).

Geometry is not re-exported: import the pipeline's PLYs directly
(raw_mesh_rgb.ply, batches/*.ply, segments/thresh_*/culled_mesh_rgb.ply, ...).
They and cameras.obj share the SLAM world frame (the frame of
trajectory_open3d.log) in metres; no step transforms a mesh after TSDF. Neither
format stores units, so import into a Rhino document in metres, and leave the
OBJ import option "Map OBJ Y to Rhino Z" off (PLY import has no such swap).

Cones follow the OpenCV camera convention (x right, y down, z forward): the
apex is the camera centre and the base is the image rectangle at depth
--cone_depth, shaped by the intrinsics. A small triangle beyond the base's top
edge (image row 0, -y) marks image "up".

Usage:
  python scripts/07_export_rhino.py \\
      --frames_dir output/<run>/frames \\
      --trajectory output/<run>/sparse/trajectory_open3d.log \\
      --output_dir output/<run>/export
"""

import argparse
import os

import numpy as np

from pipeline_common import get_rgbd_file_lists, load_frame_poses, load_intrinsic


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


def write_cameras_obj(path, names, poses, cone_every, intrinsic, cone_depth):
    """Cones (one OBJ object per sampled pose) + trajectory curve (all poses)."""
    tverts, tfaces = cone_template(intrinsic, cone_depth)
    tverts_h = np.c_[tverts, np.ones(len(tverts))]
    centres = np.array([T[:3, 3] for T in poses])
    lines = ['# 07_export_rhino.py: camera cones (OpenCV: x right, y down, '
             'z forward; marker = image top) + trajectory',
             '# SLAM world frame, metres; cam_NNNNNN = frames/color/NNNNNN.jpg']
    v_base = 1                      # OBJ vertex indices are global and 1-based
    n_cones = 0
    for k, (name, T) in enumerate(zip(names, poses)):
        if k % cone_every:
            continue
        world = (tverts_h @ T.T)[:, :3]
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
        description='Export camera cones and trajectory for Rhino.')
    p.add_argument('--frames_dir', required=True,
                   help='Step 01 frames (intrinsic.json, timestamps.txt, color/)')
    p.add_argument('--trajectory', required=True, help='trajectory_open3d.log (step 02)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--intrinsic', default=None,
                   help='Default: <frames_dir>/intrinsic.json')
    p.add_argument('--cone_depth', type=float, default=0.05,
                   help='Cone length (apex to image plane), metres')
    p.add_argument('--cone_every', type=int, default=1,
                   help='Cone for every Nth posed frame (trajectory keeps all)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    intrinsic, _ = load_intrinsic(args.intrinsic or
                                  os.path.join(args.frames_dir, 'intrinsic.json'))

    print("Cameras")
    poses_all = load_frame_poses(args.trajectory, args.frames_dir)
    color_files, _ = get_rgbd_file_lists(args.frames_dir)
    posed = [(f"cam_{os.path.splitext(os.path.basename(color_files[i]))[0]}", T)
             for i, T in enumerate(poses_all) if T is not None]
    names, poses = zip(*posed)
    cams_path = os.path.join(args.output_dir, 'cameras.obj')
    n_cones = write_cameras_obj(cams_path, names, poses, max(1, args.cone_every),
                                intrinsic, args.cone_depth)
    print(f"  -> {cams_path} ({n_cones} cones, trajectory {len(poses)} poses)")
    print("Units: m. Import with the run's PLYs into a Rhino document in metres.")


if __name__ == '__main__':
    main()
