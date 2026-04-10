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
    [--model_name Qwen/Qwen2.5-VL-3B-Instruct] \
    [--num_frames 8] [--max_new_tokens 512] \
    [--prompt_template prompts/cot_filtering_prompt.txt] \
    [--device auto] [--device_map auto] [--max_memory_per_gpu 70GiB] \
    [--python_bin python]

Description:
  - 입력 CSV를 GPU 개수만큼 균등 분할합니다.
  - 각 GPU에서 scripts/run_cot_scoring.py를 병렬 실행합니다.
  - 완료 후 shard 결과를 merged 파일로 합칩니다.
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

pids=()
for idx in "${!GPU_ARR[@]}"; do
  gpu_id="${GPU_ARR[$idx]}"
  shard_csv="$OUTPUT_DIR/shards/subset.part${idx}.csv"
  out_csv="$OUTPUT_DIR/shards/scored.part${idx}.csv"
  out_jsonl="$OUTPUT_DIR/shards/scored.part${idx}.jsonl"
  log_file="$OUTPUT_DIR/logs/part${idx}.log"

  if [[ ! -s "$shard_csv" ]]; then
    echo "[WARN] shard is empty, skip part${idx}"
    continue
  fi

  echo "[INFO] launch part${idx} on GPU ${gpu_id}"
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
    --print_step_summary
  )

  if [[ -n "$MAX_MEMORY_PER_GPU" ]]; then
    cmd+=(--max_memory_per_gpu "$MAX_MEMORY_PER_GPU")
  fi

  "${cmd[@]}" > "$log_file" 2>&1 &
  pids+=("$!")
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "[ERROR] no worker started. check shards or --gpus"
  exit 1
fi

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[ERROR] one or more workers failed. check $OUTPUT_DIR/logs/*.log"
  exit 1
fi

echo "[INFO] merging shard outputs..."
"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
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

echo "[DONE] outputs in: $OUTPUT_DIR"
