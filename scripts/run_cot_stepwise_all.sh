#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_cot_stepwise_all.sh \
    --subset_csv data/subsets/local_subset.csv \
    --output_dir outputs/scored_stepwise \
    --gpus 0,1 \
    [--mode all|step] [--step 1..5] \
    [--model_name Qwen/Qwen2.5-VL-3B-Instruct] \
    [--num_frames 8] [--max_new_tokens 512] \
    [--prompt_template prompts/cot_filtering_prompt.txt] \
    [--device auto] [--device_map auto] [--max_memory_per_gpu 70GiB] \
    [--torch_cuda_alloc_conf expandable_segments:True] \
    [--stagger_seconds 2] [--python_bin python]

Description:
  - mode=all: Step1~5를 순차 실행(각 step은 멀티GPU shard 병렬), 이전 step JSONL을 다음 step 입력으로 사용.
  - mode=step: 지정된 step만 실행. step>1이면 이전 step 결과 JSONL을 자동 참조.
  - 각 step 완료 후 shard 결과를 step 디렉터리에 merge합니다.
  - GPU 메모리 단편화 완화를 위해 기본 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True를 적용합니다.
USAGE
}

SUBSET_CSV=""
OUTPUT_DIR=""
GPUS="0"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
NUM_FRAMES="8"
MAX_NEW_TOKENS="512"
PROMPT_TEMPLATE="prompts/cot_filtering_prompt.txt"
DEVICE="auto"
DEVICE_MAP="auto"
MAX_MEMORY_PER_GPU=""
PYTHON_BIN="python"
MODE="all"
STEP=""
STAGGER_SECONDS="2"
TORCH_CUDA_ALLOC_CONF="expandable_segments:True"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset_csv) SUBSET_CSV="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --num_frames) NUM_FRAMES="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --prompt_template) PROMPT_TEMPLATE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device_map) DEVICE_MAP="$2"; shift 2 ;;
    --max_memory_per_gpu) MAX_MEMORY_PER_GPU="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --stagger_seconds) STAGGER_SECONDS="$2"; shift 2 ;;
    --torch_cuda_alloc_conf) TORCH_CUDA_ALLOC_CONF="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ "$MODE" != "all" && "$MODE" != "step" ]]; then
  echo "[ERROR] --mode must be one of: all, step"
  exit 1
fi
if [[ "$MODE" == "step" ]]; then
  if [[ -z "$STEP" ]]; then
    echo "[ERROR] --mode step requires --step (1~5)"
    exit 1
  fi
  if ! [[ "$STEP" =~ ^[1-5]$ ]]; then
    echo "[ERROR] --step must be integer 1~5"
    exit 1
  fi
fi

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

run_step() {
  local step="$1"
  local step_dir="$OUTPUT_DIR/step${step}"
  local step_shards_dir="$step_dir/shards"
  local step_logs_dir="$step_dir/logs"
  mkdir -p "$step_shards_dir" "$step_logs_dir"

  pids=()
  for idx in "${!GPU_ARR[@]}"; do
    local gpu_id="${GPU_ARR[$idx]}"
    local shard_csv="$OUTPUT_DIR/shards/subset.part${idx}.csv"
    local out_csv="$step_shards_dir/scored.part${idx}.csv"
    local out_jsonl="$step_shards_dir/scored.part${idx}.jsonl"
    local log_file="$step_logs_dir/part${idx}.log"
    local prev_jsonl=""
    if [[ "$step" -gt 1 ]]; then
      prev_jsonl="$OUTPUT_DIR/step$((step-1))/shards/scored.part${idx}.jsonl"
      if [[ ! -f "$prev_jsonl" ]]; then
        echo "[ERROR] previous step jsonl not found for step${step}, part${idx}: $prev_jsonl"
        exit 1
      fi
    fi

    if [[ ! -s "$shard_csv" ]]; then
      echo "[WARN] shard is empty, skip part${idx}"
      continue
    fi

    echo "[INFO] launch step${step} part${idx} on GPU ${gpu_id}"
    cmd=(
      "$PYTHON_BIN" scripts/run_cot_scoring.py
      --subset_csv "$shard_csv"
      --output_csv "$out_csv"
      --output_jsonl "$out_jsonl"
      --model_name "$MODEL_NAME"
      --num_frames "$NUM_FRAMES"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --prompt_template "$PROMPT_TEMPLATE"
      --device "$DEVICE"
      --device_map "$DEVICE_MAP"
      --cuda_visible_devices "$gpu_id"
      --start_step "$step"
      --end_step "$step"
      --print_step_summary
    )
    if [[ -n "$prev_jsonl" ]]; then
      cmd+=(--input_jsonl "$prev_jsonl")
    fi
    if [[ -n "$MAX_MEMORY_PER_GPU" ]]; then
      cmd+=(--max_memory_per_gpu "$MAX_MEMORY_PER_GPU")
    fi
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export PYTORCH_CUDA_ALLOC_CONF="$TORCH_CUDA_ALLOC_CONF"
      "${cmd[@]}"
    ) > "$log_file" 2>&1 &
    pids+=("$!")
    if [[ "$STAGGER_SECONDS" != "0" ]]; then
      sleep "$STAGGER_SECONDS"
    fi
    "${cmd[@]}" > "$log_file" 2>&1 &
    pids+=("$!")
  done

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "[ERROR] no worker started for step${step}. check shards or --gpus"
    exit 1
  fi

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[ERROR] one or more workers failed at step${step}. check $step_logs_dir/*.log"
    exit 1
  fi

  echo "[INFO] merging shard outputs for step${step}..."
  "$PYTHON_BIN" - "$step_dir" <<'PY'
import glob
import json
import os
import pandas as pd
import sys

output_dir = sys.argv[1]
shards_dir = os.path.join(output_dir, "shards")

csv_paths = sorted(glob.glob(os.path.join(shards_dir, "scored.part*.csv")))
jsonl_paths = sorted(glob.glob(os.path.join(shards_dir, "scored.part*.jsonl")))
steps_paths = sorted(glob.glob(os.path.join(shards_dir, "scored.part*.steps.jsonl")))

if not csv_paths:
    raise SystemExit("[ERROR] no shard csv outputs found")

merged_csv = os.path.join(output_dir, "scored_merged.csv")
merged_jsonl = os.path.join(output_dir, "scored_merged.jsonl")
merged_steps = os.path.join(output_dir, "scored_merged.steps.jsonl")

frames = [pd.read_csv(p) for p in csv_paths if os.path.getsize(p) > 0]
if frames:
    pd.concat(frames, ignore_index=True).to_csv(merged_csv, index=False)
else:
    pd.DataFrame().to_csv(merged_csv, index=False)

with open(merged_jsonl, "w", encoding="utf-8") as fout:
    for p in jsonl_paths:
        if os.path.getsize(p) == 0:
            continue
        with open(p, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if line:
                    fout.write(line + "\n")

with open(merged_steps, "w", encoding="utf-8") as fout:
    for p in steps_paths:
        if os.path.getsize(p) == 0:
            continue
        with open(p, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if line:
                    fout.write(line + "\n")

meta = {
    "merged_csv": merged_csv,
    "merged_jsonl": merged_jsonl,
    "merged_steps_jsonl": merged_steps,
    "num_csv_shards": len(csv_paths),
    "num_jsonl_shards": len(jsonl_paths),
    "num_steps_shards": len(steps_paths),
}
with open(os.path.join(output_dir, "merge_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("[INFO] merge done")
print(json.dumps(meta, ensure_ascii=False, indent=2))
PY
}

if [[ "$MODE" == "all" ]]; then
  for step in 1 2 3 4 5; do
    run_step "$step"
  done
  cp "$OUTPUT_DIR/step5/scored_merged.csv" "$OUTPUT_DIR/scored_merged.csv"
  cp "$OUTPUT_DIR/step5/scored_merged.jsonl" "$OUTPUT_DIR/scored_merged.jsonl"
  cp "$OUTPUT_DIR/step5/scored_merged.steps.jsonl" "$OUTPUT_DIR/scored_merged.steps.jsonl"
  echo "[DONE] all steps completed. final outputs in: $OUTPUT_DIR"
else
  run_step "$STEP"
  if [[ "$STEP" == "5" ]]; then
    cp "$OUTPUT_DIR/step5/scored_merged.csv" "$OUTPUT_DIR/scored_merged.csv"
    cp "$OUTPUT_DIR/step5/scored_merged.jsonl" "$OUTPUT_DIR/scored_merged.jsonl"
    cp "$OUTPUT_DIR/step5/scored_merged.steps.jsonl" "$OUTPUT_DIR/scored_merged.steps.jsonl"
  fi
  echo "[DONE] step${STEP} completed. outputs in: $OUTPUT_DIR/step${STEP}"
fi
