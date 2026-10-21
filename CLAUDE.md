# OMS SLAM Semantic Reconstruction — Claude Code Context

## Project overview
Sub-project 2 of 3 in the OMS stone-wall pipeline:
1. **Components** (`oms_monocular_semantic_recon`) — MASt3R/VGGT + SAM3 → per-stone point clouds
2. **This repo** — ORB-SLAM3 RGB-D SLAM + SAM3 EDT alpha scoring → segmented assembly mesh
3. **Registration** (future) — geometry-based matching of stones into the assembly

RealSense D456 camera. ORB-SLAM3 tracks the camera; Open3D TSDF fuses depth into a dense
mesh; SAM3 EDT alpha scoring separates stone interiors from seams.

## Pipeline (6 numbered stages)
Each stage = a root wrapper `NN_*.sh` → `scripts/NN_*.py`. All settings in root `pipeline.yaml`
(key `section.key` → shell var `SECTION_KEY`, loaded by `scripts/pipeline_env.sh`; vars already set
in a wrapper's OVERRIDES block or the environment win). No hard-coded paths in wrappers.
`run_pipeline.sh` runs the non-batched track (`bash run_pipeline.sh [first] [last]`).

| # | Wrapper → script | Does |
|---|------------------|------|
| 01 | `01_extract.sh` → `01_extract_frames.py` | .bag → color/ depth/ confidence/ (depth+conf aligned to colour) intrinsic.json timestamps.txt streams.json |
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
  pipeline.yaml                                  — global settings: bag, output base, all stage params
  scripts/pipeline_env.sh, config_to_env.py      — YAML → shell vars + derived run paths
  scripts/NN_*.py                                — stage implementations
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

# Full non-batched pipeline (set paths.* in pipeline.yaml first)
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
- **ORB-SLAM3 settings generated, not stored**: `02_slam.py` renders them from
  `frames/intrinsic.json` (camera K, depth scale) + `pipeline.yaml` `orbslam.*` and writes
  `sparse/orbslam_settings.yaml` beside the atlas. `oms_component_registration` 02a reads it to
  match the atlas camera fx + ORB pyramid + CLAHE (`Preprocess.*` keys, ignored by ORB-SLAM3).
  `Stereo.b` = 0.095 (D456 baseline) and `Camera.RGB` = 0 (imread gives BGR); before 2026-10-21
  these were 0.0387 and 1 (R/B grey weights swapped in the atlas)
- **Shutdown waits for mapping** (fork branch `shutdown-wait`): `System::Shutdown()` drains local
  mapping + loop closing and waits for a running global BA before saving the atlas, up to
  `slam.shutdown_wait_s` (settings key `System.ShutdownWaitSeconds`; absent/0 = old behaviour).
  Upstream saved immediately, so a GBA started near the end of a sequence was lost.
- **Playback is real-time**: `rgbd_tum` sleeps to each frame's real timestamp gap (headless or
  background alike); `slam.fps` does NOT change speed — it is ORB-SLAM3's max frames between KFs.
- **Fork branch `shutdown-wait`** (off `66911af`) also fixes `SaveTrajectoryTUM` segfaulting after
  tracking loss (null parent KF) and `LoadAtlas` failing on absolute paths. Fork `master` diverges:
  it lowers relocalisation thresholds for all modes (50→20 PnP inliers) — not merged here.
- **Poses matched to frames by timestamp**: `trajectory_open3d.log` is `timestamp + 4×4 c2w` per
  tracked frame; `pipeline_common.load_frame_poses` pairs each frame's `timestamps.txt` entry with
  the nearest pose (≤ 1 ms) and steps 03/03b/05/05b skip frames with none. ORB-SLAM3 omits lost
  frames and, while tracking is recently lost, repeats the last pose under the last timestamp;
  `02_slam.py` drops those repeats. Index pairing (pre-2026-10-21) shifted every pose after the
  first lost frame. Old 16-column logs: used only if pose count == frame count, else regenerate
  with `python scripts/02_slam.py --convert_only --frames_dir … --output_dir <sparse dir>`.
- **Runtime viewer toggle**: `rgbd_tum.cc` accepts argv[5] (0=headless, 1=viewer)
- **FRAME_STRIDE matters**: stride=1 (30 fps) is more reliable than stride=3 for tracking on fast motion
- **EDT gamma**: `(dist/max_dist)**gamma`; gamma<1 sharpens seams, =1 linear, >1 conservative
- **CPU TSDF**: `ScalableTSDFVolume` (Open3D legacy API); GPU `VoxelBlockGrid` disabled (v0.19 device issues)
- **Batched track**: keeps TSDF batches as separate PLYs to hit 2 mm voxel within ~30 GB RAM;
  scoring + culling operate on one batch mesh at a time (see `docs/drafts/0421_dev_notes.md`)
- **QR-hole recovery** (step 04): QR markers on stones read as holes in the stone mask; a ring
  dilation check merges an enclosed QR's pixels back into the covering stone mask
- **Run layout**: everything for one bag under `<paths.output_base>/<run_name>/` (frames/ included);
  set `paths.frames_dir` to reuse frames stored elsewhere

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
bash run_pipeline.sh               # set paths.* in pipeline.yaml first
```

## Bag files
- RealSense D456, RGB-D (no confidence stream in current recordings)
- Bags are large (2–8 GB); run locally for stride=1 tests

## Data layout
Output written to `<paths.output_base>/<run_name>/` (default `output/`, gitignored):
```
<output_base>/<run_name>/
  frames/                         — stage 01 frames + intrinsic.json (+ stage 04 SAM3 caches)
  sparse/
    CameraTrajectory.txt          — ORB-SLAM3 TUM poses
    trajectory_open3d.log         — timestamp + 4×4 c2w per tracked frame (03 / 05 match by timestamp)
  raw_mesh_rgb.ply                — stage 03 geometry mesh
  scoring/
    alpha_maps/alpha_*.npz        — L2 EDT cache
    alpha_mesh.ply                — stage 05 semantic mesh
    alpha_batches/                — batched track (05b)
  batches/                        — batched track (03b)
  segments/                       — stage 06 culled mesh + per-stone submeshes
```

## Tunable parameters
All live in `pipeline.yaml` (full list + comments there). Key ones:
| Where | YAML key (shell var) | Effect |
|-------|----------------------|--------|
| 05 | `score.edt_gamma` (`SCORE_EDT_GAMMA`) | Score falloff; lower = sharper seams |
| 05 | `score.sam_max_size_ratio` | Max mask size fraction kept |
| 03/05 | `tsdf.voxel_size` (`TSDF_VOXEL_SIZE`) | TSDF resolution (m); smaller = finer/slower |
| 06 | `cull.alpha_thresholds` | Seam cutoff sweep |
| 06 | `cull.mesh_keep_components` | Keep N largest components |
| 06 | `cull.min_cluster_size` | Min triangles per saved segment |
| 03b/05b | `batch.size`, `batch.overlap` | Batched-track chunking (one setting, shared by 03b & 05b) |
