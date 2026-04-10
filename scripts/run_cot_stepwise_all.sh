#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to run step-wise CoT scoring in one command.
# Usage:
#   bash scripts/run_cot_stepwise_all.sh \
#     data/wisa_rigidbody/wisa_rigidbody.csv \
#     outputs/scored/cot_260410_wisa_rigidbody.csv \
#     outputs/scored/cot_260410_wisa_rigidbody.jsonl

SUBSET_CSV="${1:-data/wisa_rigidbody/wisa_rigidbody.csv}"
OUTPUT_CSV="${2:-outputs/scored/cot_stepwise.csv}"
OUTPUT_JSONL="${3:-outputs/scored/cot_stepwise.jsonl}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
NUM_FRAMES="${NUM_FRAMES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DEVICE="${DEVICE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-}"
PRINT_STEP_SUMMARY="${PRINT_STEP_SUMMARY:-1}"

STEP_JSONL="${STEP_JSONL:-${OUTPUT_JSONL%.jsonl}.steps.jsonl}"
LOG_FILE="${LOG_FILE:-${OUTPUT_JSONL%.jsonl}.log}"
RUN_STDOUT_LOG="${RUN_STDOUT_LOG:-${OUTPUT_JSONL%.jsonl}.stdout.log}"

mkdir -p "$(dirname "$OUTPUT_CSV")" "$(dirname "$OUTPUT_JSONL")"

CMD=(
  python scripts/run_cot_scoring.py
  --subset_csv "$SUBSET_CSV"
  --output_csv "$OUTPUT_CSV"
  --output_jsonl "$OUTPUT_JSONL"
  --step_jsonl "$STEP_JSONL"
  --log_file "$LOG_FILE"
  --model_name "$MODEL_NAME"
  --num_frames "$NUM_FRAMES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
  --device_map "$DEVICE_MAP"
  --cuda_visible_devices "$CUDA_VISIBLE_DEVICES_VALUE"
)

if [[ -n "$MAX_MEMORY_PER_GPU" ]]; then
  CMD+=(--max_memory_per_gpu "$MAX_MEMORY_PER_GPU")
fi

if [[ "$PRINT_STEP_SUMMARY" == "1" ]]; then
  CMD+=(--print_step_summary)
fi

echo "[run_cot_stepwise_all] subset_csv=${SUBSET_CSV}"
echo "[run_cot_stepwise_all] output_csv=${OUTPUT_CSV}"
echo "[run_cot_stepwise_all] output_jsonl=${OUTPUT_JSONL}"
echo "[run_cot_stepwise_all] step_jsonl=${STEP_JSONL}"
echo "[run_cot_stepwise_all] log_file=${LOG_FILE}"
echo "[run_cot_stepwise_all] cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}"
echo "[run_cot_stepwise_all] stdout_log=${RUN_STDOUT_LOG}"

"${CMD[@]}" 2>&1 | tee "$RUN_STDOUT_LOG"
