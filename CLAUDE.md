# OMS SLAM Semantic Reconstruction — Claude Code Context

## Project overview
Sub-project 2 of 3 in the OMS stone-wall pipeline:
1. **Components** (`oms_monocular_semantic_recon`) — MASt3R/VGGT + SAM3 → per-stone point clouds
2. **This repo** — ORB-SLAM3 RGB-D SLAM + SAM3 EDT alpha scoring → segmented assembly mesh
3. **Registration** (future) — geometry-based matching of stones into the assembly

RealSense D456 camera. ORB-SLAM3 tracks the camera; Open3D TSDF fuses depth into a dense
mesh; SAM3 EDT alpha scoring separates stone interiors from seams.

## Pipeline (6 numbered stages)
Each stage = a root wrapper `NN_*.sh` (edit its config block) → `scripts/NN_*.py`.
`run_pipeline.sh` runs the non-batched track (`bash run_pipeline.sh [first] [last]`).

| # | Wrapper → script | Does |
|---|------------------|------|
| 01 | `01_extract.sh` → `01_extract_frames.py` | .bag → color/ depth/ confidence/ intrinsic.json timestamps.txt streams.json |
| 02 | `02_slam.sh` → `02_slam.py` | ORB-SLAM3 RGB-D tracking **and** TUM→Open3D trajectory conversion → trajectory_open3d.log (+ pose-graph JSON) |
| 03 | `03_tsdf_rgb.sh` → `03_tsdf_rgb.py` | TSDF-fuse all frames → raw_mesh_rgb.ply (geometry only) |
| 04 | `04_sam3_mask.sh` → `04_sam3_mask.py` | SAM3 **L1** mask cache; one image-encode, two prompts (stone + QR); QR-hole recovery |
| 05 | `05_sam3_score.sh` → `05_sam3_score.py` | **L2**: per-frame EDT alpha maps → semantic TSDF → alpha_maps/*.npz + alpha_mesh.ply |
| 06 | `06_cull_segment.sh` → `06_cull_segment.py` | KD-tree transfer alpha→raw mesh, cull seam triangles, segment into stone submeshes |

**Two tracks:**
- **Non-batched** (`03 → 05 → 06`): single TSDF volume; orchestrated by `run_pipeline.sh`.
- **Batched** (`03b → 05b → 06b`): overlapping temporal batches kept as separate meshes for
  large / high-res (2 mm voxel) scenes; `batch_size`/`batch_overlap` **must match** across 03b & 05b.
  Stages 01/02/04 are shared. Rationale: `docs/drafts/0421_dev_notes.md`.

## Repo layout
```
oms_slam_semantic_recon/
  01_extract.sh 02_slam.sh 03_tsdf_rgb.sh 04_sam3_mask.sh
  05_sam3_score.sh 06_cull_segment.sh     — stage wrappers
  03b_tsdf_rgb_batched.sh 05b_sam3_score_batched.sh 06b_cull_segment_batched.sh — batched variants
  run_pipeline.sh                                — orchestrator (non-batched)
  scripts/NN_*.py                                — stage implementations
  config/
    camera/RealSense_D456.yaml   — camera intrinsics for ORB-SLAM3
    orbslam/                     — ORB-SLAM3 configs
    pipeline/default.yaml        — pipeline defaults
  install/
    install_dependencies.sh      — system apt packages
    install_pangolin.sh          — builds Pangolin from source
    build_orbslam3.sh            — builds ORB-SLAM3 C++
    setup_env.sh                 — creates slam_recon env + installs SAM3
  external/
    orbslam3/                    — git submodule: ChengzheZhu/ORB_SLAM3.git
    sam3/                        — git submodule: ChengzheZhu/sam3.git (separate from components repo)
  environment.yml                — conda env spec (slam_recon, Python 3.11)
  docs/                          — SETUP.md + feature guides; docs/drafts/ = working notes
```

## Environment
- Conda env: **`slam_recon`** (Python 3.11, PyTorch 2.7+cu126, Open3D 0.19)
- Create with `bash install/setup_env.sh` (env from `environment.yml` + SAM3 editable install)
- SAM3 lives at `external/sam3` — **separate** from the components repo's SAM3
- Version pins: `opencv-python<4.10` (numpy<2 compat), `setuptools<71` (keeps `pkg_resources`),
  `psutil` (SAM3 eagerly imports it)

## ORB-SLAM3 binary
- Built at `external/orbslam3/Examples/RGB-D/rgbd_tum`
- Viewer is a **runtime flag** — 5th argv `0` = headless, `1` = Pangolin viewer (no recompile)
- Vocabulary: `external/orbslam3/Vocabulary/ORBvoc.txt.tar.gz` → extract before first run

## Launching
```bash
conda activate slam_recon

# Full non-batched pipeline (edit dataset paths at the top of each NN_*.sh first)
bash run_pipeline.sh              # stages 01–06
bash run_pipeline.sh 3 6          # stages 03–06 only

# Single stage
bash 04_sam3_mask.sh

# Batched track (large / high-res) — run stages individually
bash 03b_tsdf_rgb_batched.sh && bash 05b_sam3_score_batched.sh && bash 06b_cull_segment_batched.sh
```

## Scoring model (stages 04–06)
```
04  SAM3 L1 mask cache
    per frame → stone mask + QR mask (shared image encoding)
    QR-hole recovery: QR-on-stone holes OR-merged back into the stone mask
    → <frames_dir>/sam3_mask_cache_conf_<c>[_qr_filled]/masks_NNNNNN.npz

05  L2 EDT alpha + semantic TSDF
    L1 mask → EDT → gamma → alpha score  =  (dist / max_dist) ** edt_gamma
      0 = seam/background, 1 = stone interior
    → alpha_maps/alpha_NNNNNN.npz (L2 cache) + alpha_mesh.ply (grey = score)

06  transfer + cull + segment
    cKDTree: alpha_mesh vertices → raw_mesh vertices → per-vertex alpha
    cull triangles below --alpha_threshold; segment remainder into stone patches
```

## Cache layers
| Layer | Location | Content | Keyed on |
|-------|----------|---------|----------|
| L1 | `frames_dir/sam3_mask_cache_conf_<c>[_qr_filled]/masks_NNNNNN.npz` | raw SAM3 masks + scores | sam_prompt, sam_confidence, QR filling |
| L2 | `output_dir/scoring/alpha_maps/alpha_NNNNNN.npz` | EDT alpha float32 | edt_gamma, sam_max_size_ratio |

Re-running step 05 with the same output dir + gamma reuses the L2 cache (skips EDT).

## Key design decisions
- **ORB-SLAM3 submodule**: built in-place; vocabulary not in git
- **Runtime viewer toggle**: `rgbd_tum.cc` accepts argv[5] (0=headless, 1=viewer)
- **FRAME_STRIDE matters**: stride=1 (30 fps) is more reliable than stride=3 for tracking on fast motion
- **EDT gamma**: `(dist/max_dist)**gamma`; gamma<1 sharpens seams, =1 linear, >1 conservative
- **CPU TSDF**: `ScalableTSDFVolume` (Open3D legacy API); GPU `VoxelBlockGrid` disabled (v0.19 device issues)
- **Batched track**: keeps TSDF batches as separate PLYs to hit 2 mm voxel within ~30 GB RAM;
  scoring + culling operate on one batch mesh at a time (see `docs/drafts/0421_dev_notes.md`)
- **QR-hole recovery** (step 04): QR markers on stones read as holes in the stone mask; a ring
  dilation check merges an enclosed QR's pixels back into the covering stone mask
- **Frames stored outside repo**: extracted frames go beside the bag file

## On a new machine
```bash
git clone --recurse-submodules https://github.com/ChengzheZhu/oms_slam_semantic_recon.git
cd oms_slam_semantic_recon
sudo bash install/install_dependencies.sh
bash install/install_pangolin.sh
bash install/build_orbslam3.sh
cd external/orbslam3/Vocabulary && tar -xf ORBvoc.txt.tar.gz && cd -
bash install/setup_env.sh          # creates slam_recon env + installs SAM3
conda activate slam_recon
bash run_pipeline.sh               # edit dataset paths in the NN_*.sh first
```

## Bag files
- RealSense D456, RGB-D (no confidence stream in current recordings)
- Bags are large (2–8 GB); run locally for stride=1 tests

## Data layout
Output written to `output/<run>/` (gitignored):
```
output/<run>/
  sparse/
    CameraTrajectory.txt          — ORB-SLAM3 TUM poses
    trajectory_open3d.log         — Open3D camera log (used by 03 / 05)
  raw_mesh_rgb.ply                — stage 03 geometry mesh
  scoring/
    alpha_maps/alpha_*.npz        — L2 EDT cache
    alpha_mesh.ply                — stage 05 semantic mesh
    alpha_batches/                — batched track (05b)
  batches/                        — batched track (03b)
  segments/                       — stage 06 culled mesh + per-stone submeshes
```

## Tunable parameters
Live in the stage wrapper config blocks (edit at the top of each `NN_*.sh`):
| Where | Variable | Effect |
|-------|----------|--------|
| 05 | `EDT_GAMMA` | Score falloff; lower = sharper seams |
| 05 | `SAM_MAX_SIZE_RATIO` | Max mask size fraction kept |
| 03/05 | `VOXEL_SIZE` | TSDF resolution (m); smaller = finer/slower |
| 06 | `ALPHA_THRESHOLDS` | Seam cutoff sweep |
| 06 | `MESH_KEEP_COMPONENTS` | Keep N largest components |
| 06 | `MIN_CLUSTER_SIZE` | Min triangles per saved segment |
| 03b/05b | `BATCH_SIZE`,`BATCH_OVERLAP` | Batched-track chunking (must match across 03b & 05b) |
