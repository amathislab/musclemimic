#!/usr/bin/env bash
# Collect demo videos for the HuggingFace Space playground.
#
# Usage:
#   bash scripts/collect_demo_videos.sh [OUTPUT_DIR]
#
# Default output: ./demo_videos
# Structure: demo_videos/{model_short}/{motion_tag}/eval_MyoFullBody.mp4

set -euo pipefail

# Force EGL for headless rendering (must be set before MuJoCo loads)
export MUJOCO_GL=egl

OUTPUT_DIR="${1:-./demo_videos}"

# --- Models ---
declare -A MODELS=(
  ["mm-fullbody-base"]="hf://amathislab/mm-fullbody-base"
  ["mm-base-ll"]="hf://chengkunli/mm-base-ll"
  ["mm-base-e"]="hf://chengkunli/mm-base-e"
  ["mm-10m-1"]="hf://chengkunli/mm-10m-1"
  ["mm-100M-s1"]="hf://cheryl-tootty/mm-100M-s1"
)

# --- Motions (representative subset from KIT Kinesis training) ---
# Edit this list to add/remove motions for the demo.
MOTIONS=(
  # Walking forward
  "KIT/11/WalkingStraightForwards05_poses"
  "KIT/314/walking_slow03_poses"
  "KIT/424/walking_fast10_poses"
  "KIT/314/walking_run09_poses"
  "KIT/167/walking_medium03_poses"
  # Turning (left + right)
  "KIT/12/LeftTurn03_poses"
  "KIT/8/RightTurn10_poses"
  "KIT/3/turn_right04_poses"
  "KIT/359/turn_left05_poses"
  "KIT/11/LeftTurn05_poses"
  "KIT/10/RightTurn02_poses"
  "KIT/7/LeftTurn06_poses"
  "KIT/424/turn_right02_poses"
  "KIT/348/turn_right06_poses"
  # Circular walking (clockwise + counter-clockwise)
  "KIT/9/WalkInClockwiseCircle04_poses"
  "KIT/6/WalkInCounterClockwiseCircle06_1_poses"
  "KIT/4/WalkInClockwiseCircle04_poses"
  "KIT/12/WalkInClockwiseCircle11_poses"
  "KIT/9/WalkInClockwiseCircle06_poses"
  "KIT/11/WalkInCounterClockwiseCircle02_poses"
  "KIT/6/WalkInCounterClockwiseCircle04_1_poses"
  # Walking backwards
  "KIT/7/WalkingStraightBackwards04_poses"
  "KIT/11/WalkingStraightBackwards09_1_poses"
  "KIT/6/WalkingStraightBackwards04_poses"
  "KIT/10/WalkingStraightBackwards07_poses"
  "KIT/12/WalkingStraightBackwards05_poses"
  "KIT/4/WalkingStraightBackwards01_poses"
)

N_STEPS=1000
EVAL_SEED=0

total=${#MOTIONS[@]}
model_count=${#MODELS[@]}
echo "=== Collecting demo videos ==="
echo "Models: ${!MODELS[*]}"
echo "Motions: $total"
echo "Total videos: $((total * model_count))"
echo "Output: $OUTPUT_DIR"
echo ""

for model_short in "${!MODELS[@]}"; do
  model_path="${MODELS[$model_short]}"
  record_dir="${OUTPUT_DIR}/${model_short}"
  mkdir -p "$record_dir"

  i=0
  for motion in "${MOTIONS[@]}"; do
    i=$((i + 1))
    motion_tag="${motion//\//_}"
    video_path="${record_dir}/${motion_tag}/eval_MyoFullBody.mp4"

    if [[ -f "$video_path" ]]; then
      echo "[${model_short}] (${i}/${total}) SKIP (exists): ${motion}"
      continue
    fi

    echo "[${model_short}] (${i}/${total}) Recording: ${motion}"
    uv run python fullbody/eval.py \
      --path "$model_path" \
      --motion_path "$motion" \
      --use_mujoco \
      --stochastic \
      --eval_seed "$EVAL_SEED" \
      --record \
      --record_dir "$record_dir" \
      --n_steps "$N_STEPS" \
      --start_from_beginning
  done
done

echo ""
echo "=== Done! ==="
echo "Videos saved to: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
find "$OUTPUT_DIR" -name "*.mp4" | head -20
