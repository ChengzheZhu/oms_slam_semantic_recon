# ORB_SLAM3 RGB-D Dense Reconstruction — Claude Code Context

## Project overview
Sub-project 2 of 3 in the OMS stone wall reconstruction pipeline:
1. **OMS components** (`~/Projects/oms`) — MASt3R/VGGT + SAM3 → per-stone point clouds
2. **This repo** — ORB-SLAM3 RGB-D SLAM + SAM3 EDT alpha scoring → segmented assembly mesh
3. **Instance matching** (future) — geometry-based registration of individual stones into the assembly

Uses a RealSense D456 camera. ORB-SLAM3 tracks the camera; Open3D TSDF fuses depth into a
dense mesh; SAM3 EDT alpha scoring separates stone interiors from seams.

## Repo layout
```
ORB_SLAM3_RGBD_DenseSlamReconstrction/
  config/
    camera/RealSense_D456.yaml   — camera intrinsics for ORB-SLAM3
  scripts/
    00_extract_frames.py         — .bag → PNG frames + intrinsic.json + streams.json
    01_run_orbslam3.sh           — runs ORB-SLAM3; passes viewer flag (0|1) as argv[5]
    01b_run_sam3_alpha.py        — pre-compute SAM3 mask cache (L1) for all frames
    02_convert_trajectory.py     — TUM format → Open3D log + pose graph JSON
    03_dense_reconstruction.py   — TSDF mesh (geometry only, no SAM3)
    03d_sam3_boundary_reconstruction.py — dual-TSDF + SAM3 EDT alpha → segmented mesh
    export_debug_glb.py          — mesh + camera frustums → GLB for geometry inspection
    create_associations.py       — TUM-format RGB-D associations file
    trim_bag.py                  — trim RealSense bag to a time window
    view_sam3.py                 — interactive viewer for SAM3 alpha images (TkAgg)
  install/
    install_dependencies.sh      — system apt packages (Eigen, Boost, OpenCV, etc.)
    install_pangolin.sh          — builds Pangolin from source
    build_orbslam3.sh            — builds ORB-SLAM3 C++ from source
    setup_env.sh                 — creates slam_recon conda env + clones/installs SAM3
  external/
    orbslam3/                    — git submodule: ChengzheZhu/ORB_SLAM3.git
    sam3/                        — git submodule: ChengzheZhu/sam3.git (separate from OMS)
  environment.yml                — conda env spec (slam_recon, Python 3.11)
  run_full_pipeline.sh           — production: .bag → frames → SLAM → SAM3 mesh
  run_quick_test_pipeline.sh     — quick test: coarser params, same flow
  run_03d.sh                     — quick launcher for 03d with threshold sweep
  run_meshing_with_segmentation.sh — re-mesh from existing SLAM output dir
```

## Environment
- Conda env: **`slam_recon`** (Python 3.11, PyTorch 2.7+cu126, Open3D 0.19)
- Create with: `bash install/setup_env.sh` (handles env + SAM3 editable install)
- SAM3 lives at `external/sam3` — **separate** from OMS project's `deps/sam3`
- Key version pins:
  - `opencv-python<4.10` — numpy<2 compatibility (SAM3 requires numpy<2)
  - `setuptools<71` — setuptools≥72 drops `pkg_resources` as top-level module
  - `psutil` — SAM3 transitive dep (eagerly imported in sam3_video_predictor.py)

## ORB-SLAM3 binary
- Built at `external/orbslam3/Examples/RGB-D/rgbd_tum`
- Viewer is a **runtime flag** — 5th argv `0` = headless, `1` = Pangolin viewer
- No recompile needed to switch modes; `01_run_orbslam3.sh --headless` passes `0`
- Vocabulary: `external/orbslam3/Vocabulary/ORBvoc.txt.tar.gz` → extract before first run

## Launching
```bash
# Activate env
conda activate slam_recon

# Full pipeline (bag → frames → ORB-SLAM3 → SAM3 mesh → debug GLB)
bash run_full_pipeline.sh /path/to/recording.bag

# Quick test (stride=3, coarser TSDF)
bash run_quick_test_pipeline.sh /path/to/recording.bag

# Resume from existing frames (skip re-extraction)
bash run_quick_test_pipeline.sh /path/to/recording.bag /path/to/existing/frames_dir

# Tune alpha threshold (edit vars at top of run_03d.sh, then run)
bash run_03d.sh

# Step by step
python scripts/00_extract_frames.py --bag recording.bag --output frames/ --stride 1
bash scripts/01_run_orbslam3.sh frames/ output/sparse/ --fps 30 --headless
python scripts/02_convert_trajectory.py \
    --input output/sparse/CameraTrajectory.txt \
    --output_log output/sparse/trajectory_open3d.log \
    --output_json output/sparse/trajectory_pose_graph.json
# Pre-compute SAM3 mask cache (once per frames_dir)
python scripts/01b_run_sam3_alpha.py --frames_dir frames/
# Run dual-TSDF with threshold sweep
python scripts/03d_sam3_boundary_reconstruction.py \
    --frames_dir frames/ --intrinsic frames/intrinsic.json \
    --trajectory output/sparse/trajectory_open3d.log \
    --output output/gamma_0.5/raw_mesh_rgb.ply \
    --alpha_thresholds 0.2 0.3 0.5 \
    --edt_gamma 0.5 --skip_segments
```

## 03d pipeline (dual-TSDF)
```
Pass 1  integrate_tsdf()
    All frames → raw RGB TSDF → raw_mesh_rgb.ply

Pass 2a  precompute_alphas()          [parallel, cached]
    L1 mask cache → EDT → gamma → alpha_maps/alpha_NNNNNN.npz  (L2 cache)
    score = (dist / max_dist) ** edt_gamma
    0 = seam/background, 1 = stone interior

Pass 2b  integrate_semantic_tsdf()
    L2 alpha maps → semantic TSDF → alpha_mesh (grayscale R=G=B=score)

Scoring & culling  (once per run)
    cKDTree: alpha_mesh vertices → raw_mesh vertices → alpha_scores[]
    score_map_mesh.ply — blue=interior, red=seam

Threshold sweep  (fast — no TSDF or EDT recompute)
    For each --alpha_thresholds value t → thresh_<t>/
      culled_mesh_rgb.ply  — raw RGB mesh, seam triangles removed
      segments.ply         — pseudo-colour patches  (skipped with --skip_segments)
      sam3_segments/       — individual stone PLYs  (skipped with --skip_segments)
```

## Cache layers
| Layer | Location | Content | Keyed on |
|-------|----------|---------|----------|
| L1 | `frames_dir/sam3_mask_cache/masks_NNNNNN.npz` | raw SAM3 masks (N,H,W) uint8 + scores (N,) float32 | sam_prompt, sam_confidence |
| L2 | `output_dir/alpha_maps/alpha_NNNNNN.npz` | EDT alpha float32 (H,W) | edt_gamma, sam_max_size_ratio |

Re-running `03d` with the same output dir and same gamma reuses the L2 cache and skips EDT entirely.

## Key design decisions
- **ORB-SLAM3 submodule**: `external/orbslam3` — built in-place; vocabulary not in git
- **Runtime viewer toggle**: `rgbd_tum.cc` patched to accept argv[5] (0=headless, 1=viewer)
- **FRAME_STRIDE matters**: stride=3 (100ms/frame) can cause tracking loss on fast camera
  motion → prefer stride=1 (30fps, 33ms) for reliable single-map tracking
- **SAM3 mask cache (L1)**: `01b_run_sam3_alpha.py` saves raw masks+scores; EDT and alpha
  params (`edt_gamma`, `sam_max_size_ratio`, `alpha_threshold`) are applied on-the-fly so
  SAM3 inference never needs to re-run when tuning downstream params
- **EDT gamma**: non-linear score falloff `(dist/max_dist)**gamma`; gamma<1 makes interiors
  saturate quickly for sharper seams; gamma=1 is linear; gamma>1 is more conservative
- **Threshold sweep**: expensive passes (TSDF ×2, EDT) run once; culling + segmentation
  loop over `--alpha_thresholds`; each threshold writes to `thresh_<t>/`
- **CPU TSDF**: `ScalableTSDFVolume` (Open3D legacy API); GPU `VoxelBlockGrid` is
  implemented but disabled — Open3D v0.19 has device-placement inconsistencies
- **EDT thread pool**: `max(cpu_count - 4, 1)` threads; scipy EDT releases the GIL for
  true CPU parallelism; single-threaded when SAM3 inference is running (GPU contention)
- **Segmentation**: scipy `connected_components` + vectorised numpy COO edges (no Python
  loops); streaming segment export (one submesh at a time) to avoid OOM
- **Depth filter**: always-on zero/min-range filter; optional confidence threshold for
  future bags that include a confidence stream
- **Frames stored outside repo**: extracted frames go beside the bag file

## On a new machine
```bash
git clone --recurse-submodules https://github.com/ChengzheZhu/ORB_SLAM3_RGBD_DenseSlamReconstrction.git
cd ORB_SLAM3_RGBD_DenseSlamReconstrction
sudo bash install/install_dependencies.sh
bash install/install_pangolin.sh
bash install/build_orbslam3.sh
cd external/orbslam3/Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd -
bash install/setup_env.sh          # creates slam_recon env + installs SAM3
conda activate slam_recon
bash run_quick_test_pipeline.sh /path/to/recording.bag
```

## Bag files
- RealSense D456, RGB-D only (no confidence stream in current recordings)
- Location on server: `~/cloud/cheng-3dcv/OMS/2/rs_bags/`
- Bags are large (2–8 GB) on a network mount — run locally for stride=1 tests

## Data layout
Output written to `output/<run_name>/` (gitignored):
```
output/gamma_0.5/
  sparse/
    CameraTrajectory.txt          — ORB-SLAM3 TUM poses
    trajectory_open3d.log         — Open3D camera log
  raw_mesh_rgb.ply                — Pass 1 TSDF mesh (original RGB)
  score_map_mesh.ply              — alpha score visualisation (blue→red)
  alpha_maps/                     — L2 EDT cache (per gamma)
  thresh_0.3/
    culled_mesh_rgb.ply           — RGB mesh with seam triangles removed
    segments.ply                  — pseudo-colour segment patches (optional)
    sam3_segments/segment_*.ply   — individual stone submeshes (optional)
  thresh_0.5/
    ...
  debug.glb                       — mesh + camera frustums for geometry inspection
```

## Tunable parameters (run_03d.sh)
| Variable | Typical | Effect |
|----------|---------|--------|
| `EDT_GAMMA` | 0.3–0.7 | Score falloff shape; lower = sharper seams |
| `ALPHA_THRESHOLDS` | `"0.2 0.3 0.5"` | Seam cutoff sweep; each → thresh_<t>/ |
| `VOXEL_SIZE` | 0.005 m | TSDF resolution; smaller = finer but slower |
| `MESH_KEEP_COMPONENTS` | 1 | Drop floating fragments before scoring |
| `MIN_CLUSTER_SIZE` | 1000 | Minimum triangles to keep a segment |
| `SKIP_SEGMENTS` | true | Skip segmentation; only save culled mesh |

## Pending tasks
- [ ] Re-enable GPU TSDF when Open3D VoxelBlockGrid API stabilises
- [ ] Wire `04_export_point_clouds.py` into `run_full_pipeline.sh`
- [ ] Server config: FRAME_STRIDE=3, VOXEL_SIZE=0.002 (wall motion is slower)
