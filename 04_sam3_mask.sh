#!/usr/bin/env bash
# 04_sam3_mask.sh — Pre-compute SAM3 instance masks + label recovery (L1 cache).
#
# Two SAM3 passes per frame (cache dirs named by prompt, see scripts/sam3_cache.py):
#   Pass 1: sam3.prompt        ("individual stone") → sam3_individual_stone_conf_<c>/
#   Pass 2: sam3.label_prompt  ("paper label")      → sam3_paper_label_conf_<c>/
#
# Label recovery (CPU post-process):
#   each label goes to the stone covering >= label_min_coverage of its grown perimeter.
#   Output: sam3_merged_<stone>+<label>_conf_<c>/  ← step 05 picks this up automatically
#
# Resumable: already-cached frames are skipped automatically.
# Settings: pipeline.yaml (paths.*, sam3.*). Run:  bash 04_sam3_mask.sh
set -e

# ── OVERRIDES — this run only; uncomment to replace pipeline.yaml values ──────
# FRAMES_DIR=/abs/path/to/frames
# SAM3_CONFIDENCE=0.3
# SAM3_SKIP_LABEL_RECOVERY=true
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/scripts/pipeline_env.sh"
export PYTHONPATH="$PROJECT_DIR/external/sam3:$PYTHONPATH"

python "$PROJECT_DIR/scripts/04_sam3_mask.py" \
    --frames_dir          "$FRAMES_DIR"               \
    --sam_prompt          "$SAM3_PROMPT"              \
    --sam_confidence      "$SAM3_CONFIDENCE"          \
    --label_prompt        "$SAM3_LABEL_PROMPT"        \
    --label_ring_px       "$SAM3_LABEL_RING_PX"       \
    --label_min_coverage  "$SAM3_LABEL_MIN_COVERAGE"  \
    --label_grow_px       "$SAM3_LABEL_GROW_PX"       \
    --n_workers           "$SAM3_N_WORKERS"           \
    $(flag "$SAM3_SKIP_LABEL_RECOVERY" --skip_label_recovery)
