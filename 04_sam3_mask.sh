#!/usr/bin/env bash
# 04_sam3_mask.sh — Pre-compute SAM3 instance masks + QR hole recovery (L1 cache).
#
# Two SAM3 passes per frame:
#   Pass 1: sam3.prompt     ("individual stone") → sam3_mask_cache_conf_<c>/
#   Pass 2: sam3.qr_prompt  ("QR code")          → sam3_qr_cache_conf_<c>/
#
# QR hole recovery (CPU post-process):
#   QR masks enclosed by a stone mask are OR-merged into that stone mask.
#   Output: sam3_mask_cache_conf_<c>_qr_filled/  ← step 05 picks this up automatically
#
# Resumable: already-cached frames are skipped automatically.
# Settings: pipeline.yaml (paths.*, sam3.*). Run:  bash 04_sam3_mask.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# FRAMES_DIR=/abs/path/to/frames
# SAM3_CONFIDENCE=0.3
# SAM3_SKIP_QR_RECOVERY=true
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

python "$PROJECT_DIR/scripts/04_sam3_mask.py" \
    --frames_dir       "$FRAMES_DIR"            \
    --sam_prompt       "$SAM3_PROMPT"           \
    --sam_confidence   "$SAM3_CONFIDENCE"       \
    --qr_prompt        "$SAM3_QR_PROMPT"        \
    --qr_ring_px       "$SAM3_QR_RING_PX"       \
    --qr_min_coverage  "$SAM3_QR_MIN_COVERAGE"  \
    --n_workers        "$SAM3_N_WORKERS"        \
    $(flag "$SAM3_SKIP_QR_RECOVERY" --skip_qr_recovery)
