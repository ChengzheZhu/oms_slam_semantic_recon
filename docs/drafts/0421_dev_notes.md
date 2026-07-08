# Dev Notes — 2026-04-21

## Overview

Redesigned the reconstruction pipeline for high-resolution (2 mm voxel) output
while staying within RAM limits (~30 GB machine).  The key architectural change
is **keeping TSDF batches as separate PLY files** rather than merging them, and
propagating that structure through scoring and culling so that all downstream
queries work on small batch meshes one at a time.

---

## 1. Batched pipeline (03b → 05b → 06b)

### Problem
`03_tsdf_rgb.py` at 2 mm voxel hits ~22+ GB RAM on a 225 m² scene.
The prior `03b` merge step caused surface artefacts at batch boundaries.

### Solution: no-merge batch architecture
Each script now saves individual `batch_NNNN.ply` files alongside an
`index.json` that records `{batch_id, frame_start, frame_end, ply}`.

| Script | Input | Output |
|--------|-------|--------|
| `03b_tsdf_rgb_batched` | frames + trajectory | `batches/batch_NNNN.ply` + `index.json` |
| `05b_sam3_score_batched` | frames + L1 mask cache | `alpha_batches/alpha_batch_NNNN.ply` + `index.json` |
| `06b_cull_segment_batched` | `batches/` + `alpha_batches/` | `culled_batches/culled_batch_NNNN.ply` + `index.json` |

**Critical**: `03b` and `05b` must be run with identical `--batch_size` and
`--batch_overlap` so that `batch_NNNN.ply` ↔ `alpha_batch_NNNN.ply` are 1-to-1.
`06b` validates this at startup.

Peak RAM per script ≈ 1 batch mesh in memory at a time.

### 06b chunked score transfer
`transfer_alpha_scores_chunked()` queries the KDTree in vertex chunks of
`--chunk_size` (default 500 k) to bound peak RAM during score transfer.

---

## 2. QR hole recovery in step 04

### Problem
QR codes attached to stones appear as holes in SAM3 stone masks, causing
incomplete stone surface reconstruction and missed ray-cast hits.

### Solution: two-pass SAM3 + morphological enclosure test
`04_sam3_mask.py` now runs two SAM3 passes per frame:
- **Pass 1** — `"individual stone"` prompt → `sam3_mask_cache_conf_<c>/`
- **Pass 2** — `"QR code"` prompt → `sam3_qr_cache_conf_<c>/`

Then `apply_qr_recovery()` (CPU, idempotent):
1. Dilate each QR mask outward by `--qr_ring_px` pixels.
2. Compute what fraction of the surrounding ring overlaps each stone mask.
3. If a stone mask covers ≥ `--qr_min_coverage` (default 0.5) of the ring,
   the QR is enclosed by that stone → `stone_mask |= qr_mask`.
4. Writes merged masks to `sam3_mask_cache_conf_<c>_qr_filled/`.

Steps 05/05b must point `--mask_cache_dir` to the `_qr_filled` directory.

Default params (tuned 2026-04-21): `SAM_CONFIDENCE=0.5`, `QR_RING_PX=10`,
`QR_MIN_COVERAGE=0.75`.

---

## 3. File changes

### New scripts
- `scripts/03b_tsdf_rgb_batched.py` + `03b_tsdf_rgb_batched.sh`
- `scripts/05b_sam3_score_batched.py` + `05b_sam3_score_batched.sh`
- `scripts/06b_cull_segment_batched.py` + `06b_cull_segment_batched.sh`

### Modified scripts
- `scripts/04_sam3_mask.py` — two-pass + QR recovery
- `04_sam3_mask.sh` — `SAM_CONFIDENCE=0.5`, `QR_MIN_COVERAGE=0.75`
- `05_sam3_score_fusion.sh` — renamed from `05_sam3_score.sh`;
  `MASK_CACHE_DIR` updated to `_qr_filled` variant
- `06_cull_segment.sh` — `ALPHA_THRESHOLDS=0.4`

---

## 4. Recommended run order (high-res wall)

```
bash 03b_tsdf_rgb_batched.sh      # can run in parallel with 04
bash 04_sam3_mask.sh              # two SAM3 passes + QR recovery
bash 05b_sam3_score_batched.sh    # same batch_size/overlap as 03b
bash 06b_cull_segment_batched.sh  # per-batch cull; no full-mesh load
```

Then in `oms_component_registration`:
```
bash 03_build_tri_seg_map.sh      # cluster each culled batch
bash 03_reproject_debug.sh        # ray-cast + ICP merge → wall patches
```
