#!/usr/bin/env bash
# 04_sam3_mask.sh — Pre-compute SAM3 instance masks for all frames (L1 cache).
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
SAM_CONFIDENCE=0.3
# ──────────────────────────────────────────────────────────────────────────────

python "$PROJECT_DIR/scripts/04_sam3_mask.py" \
    --frames_dir     "$FRAMES_DIR"     \
    --sam_prompt     "$SAM_PROMPT"     \
    --sam_confidence "$SAM_CONFIDENCE"
