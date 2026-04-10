#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_cot_stepwise_all.sh \
    --subset_csv data/subsets/local_subset.csv \
    --output_dir outputs/stepwise_all \
    --gpus 0,1,2,3 \
    [--steps 1,2,3,4,5] \
    [--model_name Qwen/Qwen2.5-VL-3B-Instruct] \
    [--num_frames 8] [--max_new_tokens 512] \
    [--prompt_template prompts/cot_filtering_prompt.txt] \
    [--device auto] [--device_map auto] [--max_memory_per_gpu 70GiB] \
    [--python_bin python]

Description:
  - Step 1~5를 순차 실행합니다.
  - 각 step에서 CSV를 GPU 개수만큼 분할하여 병렬 처리합니다.
  - step N 출력 JSONL(part별)을 step N+1 입력 JSONL로 연결합니다.
  - step별 merged JSONL과 최종 merged 결과를 저장합니다.
USAGE
}

SUBSET_CSV=""
OUTPUT_DIR=""
GPUS="0"
STEPS="1,2,3,4,5"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
NUM_FRAMES="8"
MAX_NEW_TOKENS="512"
PROMPT_TEMPLATE="prompts/cot_filtering_prompt.txt"
DEVICE="auto"
DEVICE_MAP="auto"
MAX_MEMORY_PER_GPU=""
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset_csv) SUBSET_CSV="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --num_frames) NUM_FRAMES="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --prompt_template) PROMPT_TEMPLATE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device_map) DEVICE_MAP="$2"; shift 2 ;;
    --max_memory_per_gpu) MAX_MEMORY_PER_GPU="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SUBSET_CSV" || -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] --subset_csv, --output_dir 는 필수입니다."
  usage
  exit 1
fi

if [[ ! -f "$SUBSET_CSV" ]]; then
  echo "[ERROR] subset csv not found: $SUBSET_CSV"
  exit 1
fi

mkdir -p "$OUTPUT_DIR/shards" "$OUTPUT_DIR/logs"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "[ERROR] --gpus 값이 비어있습니다."
  exit 1
fi

IFS=',' read -r -a STEP_ARR <<< "$STEPS"
if [[ "${#STEP_ARR[@]}" -lt 1 ]]; then
  echo "[ERROR] --steps 값이 비어있습니다."
  exit 1
fi

echo "[INFO] splitting CSV into $NUM_GPUS shards..."
"$PYTHON_BIN" - "$SUBSET_CSV" "$OUTPUT_DIR" "$NUM_GPUS" <<'PY'
import os
import sys
import pandas as pd

subset_csv, output_dir, num_gpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(subset_csv)
if df.empty:
    raise SystemExit("[ERROR] input csv has no rows")

shards_dir = os.path.join(output_dir, "shards")
os.makedirs(shards_dir, exist_ok=True)

for idx in range(num_gpus):
    part = df.iloc[idx::num_gpus].copy()
    out_path = os.path.join(shards_dir, f"subset.part{idx}.csv")
    part.to_csv(out_path, index=False)
print(f"[INFO] rows={len(df)} split_done")
PY

run_one_step() {
  local step="$1"
  local pids=()

  for idx in "${!GPU_ARR[@]}"; do
    local gpu_id="${GPU_ARR[$idx]}"
    local shard_csv="$OUTPUT_DIR/shards/subset.part${idx}.csv"
    local output_jsonl="$OUTPUT_DIR/shards/step${step}.part${idx}.jsonl"
    local log_file="$OUTPUT_DIR/logs/step${step}.part${idx}.log"

    if [[ ! -s "$shard_csv" ]]; then
      echo "[WARN] shard is empty, skip part${idx}"
      continue
    fi

    local cmd=(
      "$PYTHON_BIN" scripts/run_cot_stepwise.py
      --step "$step"
      --subset_csv "$shard_csv"
      --output_jsonl "$output_jsonl"
      --model_name "$MODEL_NAME"
      --num_frames "$NUM_FRAMES"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --prompt_template "$PROMPT_TEMPLATE"
      --device "$DEVICE"
      --device_map "$DEVICE_MAP"
      --cuda_visible_devices "$gpu_id"
    )

    if [[ "$step" -gt 1 ]]; then
      local prev_jsonl="$OUTPUT_DIR/shards/step$((step-1)).part${idx}.jsonl"
      cmd+=(--input_jsonl "$prev_jsonl")
    fi

    if [[ -n "$MAX_MEMORY_PER_GPU" ]]; then
      cmd+=(--max_memory_per_gpu "$MAX_MEMORY_PER_GPU")
    fi

    echo "[INFO] launch step${step} part${idx} on GPU ${gpu_id}"
    "${cmd[@]}" > "$log_file" 2>&1 &
    pids+=("$!")
  done

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "[ERROR] no worker started for step ${step}"
    exit 1
  fi

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done

  if [[ "$fail" -ne 0 ]]; then
    echo "[ERROR] one or more workers failed at step ${step}. check $OUTPUT_DIR/logs/step${step}.part*.log"
    exit 1
  fi

  "$PYTHON_BIN" - "$OUTPUT_DIR" "$step" <<'PY'
import glob
import os
import sys

output_dir, step = sys.argv[1], int(sys.argv[2])
paths = sorted(glob.glob(os.path.join(output_dir, "shards", f"step{step}.part*.jsonl")))
out = os.path.join(output_dir, f"step{step}.merged.jsonl")
if not paths:
    raise SystemExit(f"[ERROR] no outputs for step {step}")

with open(out, "w", encoding="utf-8") as fout:
    for p in paths:
        if os.path.getsize(p) == 0:
            continue
        with open(p, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if line:
                    fout.write(line + "\n")
print(f"[INFO] step{step} merged={out}")
PY
}

for step in "${STEP_ARR[@]}"; do
  case "$step" in
    1|2|3|4|5) ;;
    *) echo "[ERROR] invalid step in --steps: $step"; exit 1 ;;
  esac
  echo "[INFO] ===== RUN STEP $step ====="
  run_one_step "$step"
done

LAST_STEP="${STEP_ARR[-1]}"
cp "$OUTPUT_DIR/step${LAST_STEP}.merged.jsonl" "$OUTPUT_DIR/final_merged.jsonl"

echo "[DONE] final output: $OUTPUT_DIR/final_merged.jsonl"
