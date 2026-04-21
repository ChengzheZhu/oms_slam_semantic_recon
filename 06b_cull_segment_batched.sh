#!/usr/bin/env bash
# 06b_cull_segment_batched.sh — Per-batch score transfer + cull; no full-mesh load.
# Reads batch_NNNN.ply (03b) and alpha_batch_NNNN.ply (05b) one pair at a time.
# Peak RAM = 1 RGB batch + 1 alpha batch (far less than full mesh).
# Saves culled_batch_NNNN.ply + index.json to OUTPUT_DIR/culled_batches/.
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET=base-highres

# ── EDIT THESE ────────────────────────────────────────────────────────────────
RGB_BATCHES_DIR=$PROJECT_DIR/output/$DATASET/batches
ALPHA_BATCHES_DIR=$PROJECT_DIR/output/$DATASET/scoring/alpha_batches
OUTPUT_DIR=$PROJECT_DIR/output/$DATASET/segments

ALPHA_THRESHOLD=0.5
CHUNK_SIZE=500000   # vertices per KD-tree query chunk — reduce if transfer OOMs

# Uncomment to save score_map_batch_NNNN.ply for diagnostics:
# SAVE_SCORE_MAPS=--save_score_maps
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/06b_cull_segment_batched.py" \
    --rgb_batches_dir   "$RGB_BATCHES_DIR"    \
    --alpha_batches_dir "$ALPHA_BATCHES_DIR"  \
    --output_dir        "$OUTPUT_DIR"         \
    --alpha_threshold   "$ALPHA_THRESHOLD"    \
    --chunk_size        "$CHUNK_SIZE"         \
    ${SAVE_SCORE_MAPS:-}
