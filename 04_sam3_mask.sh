#!/usr/bin/env bash
# 04_sam3_mask.sh — Pre-compute SAM3 instance masks + QR hole recovery (L1 cache).
#
# Two SAM3 passes per frame:
#   Pass 1: SAM_PROMPT   ("individual stone") → sam3_mask_cache_conf_<c>/
#   Pass 2: QR_PROMPT    ("QR code")          → sam3_qr_cache_conf_<c>/
#
# QR hole recovery (CPU post-process):
#   QR masks enclosed by a stone mask are OR-merged into that stone mask.
#   Output: sam3_mask_cache_conf_<c>_qr_filled/  ← set MASK_CACHE_DIR in 05.sh to this
#
# Resumable: already-cached frames are skipped automatically.
# Edit the variables below, then run:  bash 04_sam3_mask.sh
set -e

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate slam_recon

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
FRAMES_DIR=/home/chengzhe/Data/OMS_data3/rs_bags/base_20260127_015119

SAM_PROMPT="individual stone"
SAM_CONFIDENCE=0.5

QR_PROMPT="QR code marker"
QR_RING_PX=20        # dilation ring size for enclosure check (pixels)
QR_MIN_COVERAGE=0.6  # fraction of QR ring that must be stone to count as enclosed

# Uncomment to skip QR recovery (output = raw stone cache):
# SKIP_QR=--skip_qr_recovery
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/04_sam3_mask.py" \
    --frames_dir       "$FRAMES_DIR"       \
    --sam_prompt       "$SAM_PROMPT"       \
    --sam_confidence   "$SAM_CONFIDENCE"   \
    --qr_prompt        "$QR_PROMPT"        \
    --qr_ring_px       "$QR_RING_PX"       \
    --qr_min_coverage  "$QR_MIN_COVERAGE"  \
    ${SKIP_QR:-}
