#!/usr/bin/env bash
# 06b_cull_segment_batched.sh — Per-batch score transfer + cull; no full-mesh load.
# Reads batch_NNNN.ply (03b) and alpha_batch_NNNN.ply (05b) one pair at a time.
# Peak RAM = 1 RGB batch + 1 alpha batch (far less than full mesh).
# Saves culled_batch_NNNN.ply + index.json to <run dir>/segments/culled_batches/.
# Settings: pipeline.yaml (paths.*, cull_batched.*). Run:  bash 06b_cull_segment_batched.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# SEGMENTS_DIR=/abs/path/to/segments
# CULL_BATCHED_ALPHA_THRESHOLD=0.4
# CULL_BATCHED_SAVE_SCORE_MAPS=true
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"

python "$PROJECT_DIR/scripts/06b_cull_segment_batched.py" \
    --rgb_batches_dir   "$RUN_DIR/batches"                \
    --alpha_batches_dir "$SCORING_DIR/alpha_batches"      \
    --output_dir        "$SEGMENTS_DIR"                   \
    --alpha_threshold   "$CULL_BATCHED_ALPHA_THRESHOLD"   \
    --chunk_size        "$CULL_BATCHED_CHUNK_SIZE"        \
    $(flag "$CULL_BATCHED_SAVE_SCORE_MAPS" --save_score_maps)
