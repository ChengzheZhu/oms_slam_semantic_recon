# Segmented (Batched) Reconstruction Guide

High-resolution mesh reconstruction with limited RAM, using the **batched track**
of the pipeline (`03b → 05b → 06b`).

## Overview

The batched track splits the frame sequence into overlapping temporal batches,
processes each independently, and keeps the batch meshes **separate** (no global
merge). This enables:

- **Higher resolution** meshes (down to 2 mm voxels)
- **Lower RAM** per batch (one batch in memory at a time)
- **Same global alignment** — ORB-SLAM3 gives globally consistent poses, so all
  batches already live in one coordinate frame; no re-registration.

Rationale and design detail: [`drafts/0421_dev_notes.md`](drafts/0421_dev_notes.md).

## Running it

The batched stages share `01/02/04` with the non-batched track, then use the `b`
variants. Edit the config block at the top of each wrapper, then:

```bash
bash 03b_tsdf_rgb_batched.sh      # RGB TSDF → output/<run>/batches/
bash 05b_sam3_score_batched.sh    # per-batch EDT alpha → scoring/alpha_batches/
bash 06b_cull_segment_batched.sh  # cull + segment per batch → segments/
```

**Critical:** `BATCH_SIZE` and `BATCH_OVERLAP` **must match** between
`03b_tsdf_rgb_batched.sh` and `05b_sam3_score_batched.sh`, so each alpha batch has a
1-to-1 correspondence with its RGB batch. `06b` then queries batch-by-batch.

## Tuning

### Batch size (frames per batch)
- **Larger batches** — fewer batches, more RAM per batch, faster overall.
- **Smaller batches** — less RAM per batch, more batches to process.

### Batch overlap
- **Small (5–15 frames)** — faster; small risk of seams at batch boundaries.
- **Large (20–50 frames)** — smoother transitions, more redundant integration.

### Voxel size (`VOXEL_SIZE`, metres)
| Voxel | Detail | Relative RAM |
|-------|--------|--------------|
| 0.01  | coarse | 1× |
| 0.005 | good balance | ~2× |
| 0.002 | very high (wall production) | ~5× |

Rule of thumb: pick the batch size so one batch at your target voxel fits in RAM —
e.g. ~30 GB machine handles 2 mm voxels at ~100-frame batches (the current defaults).

## Output

```
output/<run>/
  batches/                       03b — per-batch RGB meshes
  scoring/alpha_batches/         05b — per-batch alpha meshes + index.json
  segments/                      06b — culled + segmented per batch
```
