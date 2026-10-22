# pipeline_env.sh — sourced by every step script.
#
# Activates conda, loads pipeline.yaml (variables already set by the caller
# win) and derives the per-run paths. Sets PROJECT_DIR.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-slam_recon}"

PIPELINE_CONFIG="${PIPELINE_CONFIG:-$PROJECT_DIR/pipeline.yaml}"
eval "$(python "$PROJECT_DIR/scripts/config_to_env.py" "$PIPELINE_CONFIG")"

abspath() { case "$1" in /*|"") echo "$1" ;; *) echo "$PROJECT_DIR/$1" ;; esac; }
# Prints the CLI flag when the boolean setting is true.
flag()    { if [ "$1" = true ]; then echo "$2"; fi; }

PATHS_BAG="$(abspath "$PATHS_BAG")"
SLAM_VOCAB="$(abspath "$SLAM_VOCAB")"

# ── Run layout (override any of these in a step's OVERRIDES block) ──────────
: "${RUN_NAME:=${PATHS_RUN_NAME:-$(basename "$PATHS_BAG" .bag)}}"
: "${RUN_DIR:=$(abspath "$PATHS_OUTPUT_BASE")/$RUN_NAME}"
: "${FRAMES_DIR:=$(abspath "$PATHS_FRAMES_DIR")}"
: "${FRAMES_DIR:=$RUN_DIR/frames}"
: "${SPARSE_DIR:=$RUN_DIR/sparse}"
: "${TRAJECTORY:=$SPARSE_DIR/trajectory_open3d.log}"
: "${RAW_MESH:=$RUN_DIR/raw_mesh_rgb.ply}"
: "${SCORING_DIR:=$RUN_DIR/scoring}"
: "${SEGMENTS_DIR:=$RUN_DIR/segments}"
: "${EXPORT_DIR:=$RUN_DIR/export}"

# Step 04's merged mask cache (step 05 input); named by scripts/sam3_cache.py.
if [ "$SAM3_SKIP_LABEL_RECOVERY" = true ]; then _label=""; else _label="$SAM3_LABEL_PROMPT"; fi
: "${MASK_CACHE_DIR:=$(python "$PROJECT_DIR/scripts/sam3_cache.py" \
    "$FRAMES_DIR" "$SAM3_CONFIDENCE" "$SAM3_PROMPT" "$_label")}"
unset _label

echo "config : $PIPELINE_CONFIG"
echo "run    : $RUN_DIR"
