#!/usr/bin/env python3
"""
02_slam.py — Run ORB-SLAM3 RGB-D tracking and convert the trajectory.

Steps performed:
  1. Generate associations.txt (TUM format) from extracted frames
  2. Patch camera YAML with optional atlas save/load paths
  3. Run the rgbd_tum binary
  4. Copy CameraTrajectory.txt + KeyFrameTrajectory.txt to output_dir
  5. Convert TUM trajectory → Open3D flat log + pose-graph JSON

Outputs in <output_dir>/:
  CameraTrajectory.txt         — raw TUM format from ORB-SLAM3
  KeyFrameTrajectory.txt
  trajectory_open3d.log        — flat 4×4 row-major, used by steps 03/05
  trajectory_pose_graph.json

Usage:
  python scripts/02_slam.py --frames_dir /path/to/frames --output_dir output/sparse
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# TUM associations
# ---------------------------------------------------------------------------

def create_associations(frames_dir, output_file, fps=30.0):
    """Write TUM-format associations.txt. Uses real timestamps if available."""
    color_dir = os.path.join(frames_dir, 'color')
    depth_dir = os.path.join(frames_dir, 'depth')

    timestamps_file = os.path.join(frames_dir, 'timestamps.txt')
    real_ts = None
    if os.path.exists(timestamps_file):
        with open(timestamps_file) as f:
            real_ts = [float(l.strip()) for l in f if l.strip()]
        print(f"  Using real timestamps from {timestamps_file}")

    color_files = sorted(f for f in os.listdir(color_dir)
                         if f.endswith(('.jpg', '.png')))
    depth_files = sorted(f for f in os.listdir(depth_dir)
                         if f.endswith('.png'))
    n = min(len(color_files), len(depth_files))
    print(f"  Associations: {n} frames at {fps:.1f} fps")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        for i in range(n):
            ts = real_ts[i] if (real_ts and i < len(real_ts)) else float(i) / fps
            f.write(f"{ts:.6f} color/{color_files[i]} {ts:.6f} depth/{depth_files[i]}\n")
    return output_file


# ---------------------------------------------------------------------------
# CLAHE preprocessing — normalized color frames for SLAM only
# ---------------------------------------------------------------------------

def apply_clahe_to_frames(frames_dir: str, out_dir: str) -> str:
    """
    Build a lightweight SLAM-only frames directory with CLAHE-normalized color.

    Structure of out_dir mirrors frames_dir:
      out_dir/color/  — CLAHE-equalized copies of the original color frames
      out_dir/depth   → symlink to frames_dir/depth  (originals, unmodified)
      out_dir/*       → symlinks for all other files/dirs (timestamps.txt, etc.)

    The original frames_dir/color/ is never touched — 03_tsdf_rgb.py reads from
    there and will see the unmodified images.

    Returns out_dir (for chaining).
    """
    color_src = os.path.join(frames_dir, 'color')
    color_dst = os.path.join(out_dir, 'color')
    os.makedirs(color_dst, exist_ok=True)

    # Symlink every entry except 'color' so depth/timestamps/etc. are accessible
    for entry in os.listdir(frames_dir):
        if entry == 'color':
            continue
        os.symlink(os.path.abspath(os.path.join(frames_dir, entry)),
                   os.path.join(out_dir, entry))

    color_files = sorted(
        f for f in os.listdir(color_src) if f.endswith(('.jpg', '.png')))
    print(f"  Applying CLAHE to {len(color_files)} color frames "
          f"(clipLimit=2.0, tile=8×8) …")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for fname in color_files:
        img = cv2.imread(os.path.join(color_src, fname))
        if img is None:
            # Fall back to copying the original so associations.txt stays valid
            shutil.copy2(os.path.join(color_src, fname),
                         os.path.join(color_dst, fname))
            continue
        # Normalize local contrast in L*a*b* luminance channel only
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        cv2.imwrite(os.path.join(color_dst, fname),
                    cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    print(f"  CLAHE done — TSDF will still use originals in {color_src}")
    return out_dir


# ---------------------------------------------------------------------------
# TUM → Open3D trajectory conversion
# ---------------------------------------------------------------------------

def _quat_to_rot(qx, qy, qz, qw):
    n  = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])


def load_tum_trajectory(path):
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            T = np.eye(4)
            T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
            T[:3,  3] = [tx, ty, tz]
            poses.append((ts, T))
    return poses


def save_open3d_log(poses, path):
    with open(path, 'w') as f:
        f.write(f"# Open3D trajectory log\n")
        f.write(f"# Number of poses: {len(poses)}\n")
        f.write(f"# Format: 4x4 transformation matrix (row-major)\n#\n")
        for _, T in poses:
            f.write(' '.join(f'{v:.12f}' for v in T.flatten()) + '\n')


def save_pose_graph_json(poses, path):
    pg = {"class_name": "PoseGraph", "version_major": 1, "version_minor": 0,
          "nodes": [], "edges": []}
    for _, T in poses:
        pg["nodes"].append({"class_name": "PoseGraphNode",
                            "version_major": 1, "version_minor": 0,
                            "pose": T.tolist()})
    for i in range(len(poses) - 1):
        T_rel = np.linalg.inv(poses[i][1]) @ poses[i+1][1]
        pg["edges"].append({"class_name": "PoseGraphEdge",
                            "version_major": 1, "version_minor": 0,
                            "source_node_id": i, "target_node_id": i+1,
                            "transformation": T_rel.tolist(),
                            "information": np.eye(6).tolist(),
                            "uncertain": False})
    with open(path, 'w') as f:
        json.dump(pg, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orbslam_dir = os.path.join(project_dir, 'external', 'orbslam3')

    parser = argparse.ArgumentParser(
        description="Run ORB-SLAM3 RGB-D tracking and convert trajectory")
    parser.add_argument('--frames_dir',  required=True,
                        help='Extracted frames directory (from step 01)')
    parser.add_argument('--output_dir',  required=True,
                        help='Output directory for trajectory + atlas')
    parser.add_argument('--config',
                        default=os.path.join(project_dir, 'config/camera/RealSense_D456.yaml'),
                        help='Camera YAML config')
    parser.add_argument('--vocab',
                        default=os.path.join(orbslam_dir, 'Vocabulary/ORBvoc.txt'),
                        help='ORB vocabulary file')
    parser.add_argument('--fps',         type=float, default=30.0,
                        help='Effective FPS of extracted frames (default: 30)')
    parser.add_argument('--save_atlas',  default=None,
                        help='Save Atlas to this path (no .osa extension)')
    parser.add_argument('--load_atlas',  default=None,
                        help='Load Atlas from this path (no .osa extension)')
    parser.add_argument('--localize',    action='store_true',
                        help='Localization-only mode (no new map points)')
    parser.add_argument('--headless',    action='store_true', default=True,
                        help='Disable ORB-SLAM3 viewer (default: True)')
    parser.add_argument('--viewer',      action='store_true',
                        help='Enable ORB-SLAM3 viewer window')
    parser.add_argument('--equalize',    action='store_true',
                        help='Apply CLAHE on luminance channel before SLAM '
                             '(normalises RealSense exposure; originals kept '
                             'for dense TSDF in step 03)')
    args = parser.parse_args()

    use_viewer = args.viewer and not args.headless

    print("=" * 60)
    print("Step 02 — ORB-SLAM3 SLAM")
    print("=" * 60)
    print(f"  frames_dir : {args.frames_dir}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  config     : {args.config}")
    print(f"  fps        : {args.fps}")
    if args.save_atlas:
        print(f"  save_atlas : {args.save_atlas}")
    if args.load_atlas:
        print(f"  load_atlas : {args.load_atlas}")
    if args.localize:
        print(f"  mode       : localization-only")

    os.makedirs(args.output_dir, exist_ok=True)

    for label, path in [("frames_dir", args.frames_dir),
                        ("config", args.config),
                        ("vocab", args.vocab)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    orbslam_exec = os.path.join(orbslam_dir, 'Examples/RGB-D/rgbd_tum')
    if not os.path.exists(orbslam_exec):
        print(f"ERROR: rgbd_tum binary not found: {orbslam_exec}")
        sys.exit(1)

    # ── Step 1: CLAHE preprocessing (optional) ───────────────────────────
    # slam_frames_dir is what rgbd_tum sees; args.frames_dir is kept for TSDF.
    clahe_tmpdir = None
    if args.equalize:
        print("\n[1/3] CLAHE preprocessing (SLAM only — originals kept for TSDF)…")
        clahe_tmpdir = tempfile.mkdtemp(prefix='slam_clahe_')
        slam_frames_dir = apply_clahe_to_frames(args.frames_dir, clahe_tmpdir)
    else:
        slam_frames_dir = args.frames_dir

    # ── Step 2: associations ─────────────────────────────────────────────
    # Always write into slam_frames_dir so rgbd_tum finds color/ and depth/
    # at the right relative paths.
    assoc_file = os.path.join(slam_frames_dir, 'associations.txt')
    step_label = "2" if args.equalize else "1"
    if not os.path.exists(assoc_file):
        print(f"\n[{step_label}/3] Generating associations.txt…")
        create_associations(slam_frames_dir, assoc_file, fps=args.fps)
    else:
        print(f"\n[{step_label}/3] associations.txt already exists — skipping")

    # ── Step 3: run ORB-SLAM3 ────────────────────────────────────────────
    print("\n[3/3] Running ORB-SLAM3…" if args.equalize else "\n[2/3] Running ORB-SLAM3…")

    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False,
                                     prefix='orbslam_') as tmp:
        tmp_yaml = tmp.name

    try:
        shutil.copy2(args.config, tmp_yaml)
        if args.save_atlas:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_atlas)), exist_ok=True)
            with open(tmp_yaml, 'a') as f:
                f.write(f'\nSystem.SaveAtlasToFile: "{args.save_atlas}"\n')
        if args.load_atlas:
            with open(tmp_yaml, 'a') as f:
                f.write(f'\nSystem.LoadAtlasFromFile: "{args.load_atlas}"\n')

        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = ':'.join(filter(None, [
            f"{orbslam_dir}/lib",
            f"{orbslam_dir}/Thirdparty/DBoW2/lib",
            f"{orbslam_dir}/Thirdparty/g2o/lib",
            "/usr/local/lib",
            env.get('LD_LIBRARY_PATH', ''),
        ]))

        cmd = [
            orbslam_exec,
            args.vocab,
            tmp_yaml,
            slam_frames_dir,   # CLAHE dir (or original if --equalize not set)
            assoc_file,
            '1' if use_viewer else '0',
        ]
        if args.localize:
            cmd.append('localize')

        # Remove any stale output files in output_dir before the run so a
        # crash can't be mistaken for success (ORB-SLAM3 writes to CWD).
        for fname in ['CameraTrajectory.txt', 'KeyFrameTrajectory.txt',
                      'LocalizedPoses.txt']:
            stale = os.path.join(args.output_dir, fname)
            if os.path.exists(stale):
                os.remove(stale)

        # Run with output_dir as CWD — trajectory files land there directly,
        # no hidden outputs in the source tree.
        result = subprocess.run(cmd, cwd=args.output_dir, env=env)
        if result.returncode != 0:
            print("WARNING: ORB-SLAM3 exited non-zero "
                  "(cleanup crash is normal — checking for trajectory)")
    finally:
        os.unlink(tmp_yaml)
        if clahe_tmpdir and os.path.exists(clahe_tmpdir):
            shutil.rmtree(clahe_tmpdir, ignore_errors=True)

    # ── Verify trajectories written to output_dir ─────────────────────────
    found = 0
    for fname in ['CameraTrajectory.txt', 'KeyFrameTrajectory.txt']:
        path = os.path.join(args.output_dir, fname)
        if os.path.exists(path):
            print(f"  ✓ {fname}")
            found += 1

    if found == 0:
        print("ERROR: No trajectory files produced — SLAM failed")
        sys.exit(1)

    # ── Step 3: convert trajectory ────────────────────────────────────────
    print("\n[3/3] Converting trajectory to Open3D format…")
    tum_file = os.path.join(args.output_dir, 'CameraTrajectory.txt')
    poses    = load_tum_trajectory(tum_file)
    print(f"  Loaded {len(poses)} poses")

    if not poses:
        print("ERROR: Trajectory is empty")
        sys.exit(1)

    ts      = [t for t, _ in poses]
    dur     = ts[-1] - ts[0]
    avg_fps = len(poses) / dur if dur > 0 else 0
    print(f"  Duration: {dur:.1f}s  avg {avg_fps:.1f} fps")

    log_path  = os.path.join(args.output_dir, 'trajectory_open3d.log')
    json_path = os.path.join(args.output_dir, 'trajectory_pose_graph.json')
    save_open3d_log(poses, log_path)
    save_pose_graph_json(poses, json_path)
    print(f"  ✓ trajectory_open3d.log")
    print(f"  ✓ trajectory_pose_graph.json")

    print(f"\nDone — outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
