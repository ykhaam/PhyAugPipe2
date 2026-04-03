# I/O Contracts

이 문서는 각 스크립트(단계)의 입력/출력 형식을 명시합니다.

## Common Canonical Pair Record

모든 CoT 단계에서 공통으로 사용하는 최소 레코드.

```json
{
  "sample_id": "string",
  "original_prompt": "string",
  "video_path": "/abs/or/rel/path/to/video.mp4"
}
```

---

## 1) `scripts/build_local_pairs_csv.py`

### Input A (manifest mode)
- `--manifest_csv`: CSV
- required columns (args로 변경 가능):
  - `sample_id_col` (default: `sample_id`, 없으면 row index 사용)
  - `prompt_col` (default: `original_prompt`)
  - `video_col` (default: `video_path`)

### Input B (directory scan mode)
- `--video_root`: 비디오 폴더 (`.mp4/.webm/.mov/.mkv/.avi`)
- `--text_root`: 텍스트 폴더 (`<video_basename>.txt`)

### Output
- `--output_csv`: canonical pair CSV

| column | type | description |
|---|---|---|
| sample_id | str | 샘플 식별자 |
| original_prompt | str | 원문 텍스트 프롬프트 |
| video_path | str | 실제 존재하는 로컬 비디오 경로 |

---

## 2) `scripts/build_subset_csv.py` (Panda70M-like)

### Input
- `--input_csv`: 원본 메타 CSV
- column args:
  - `--id_col`
  - `--prompt_col`
  - `--video_col` or (`--video_root` + `--video_ext`)
- optional filter/sampling:
  - `--keywords`
  - `--max_count`
  - `--sample_mode` (`top`/`random`)

### Output
- `--output_csv`: canonical pair CSV (`sample_id,original_prompt,video_path`)

---

## 3) `scripts/build_wisa80k_subset_csv.py`

### Input
- `--dataset_name` (default: `qihoo360/WISA-80K`)
- `--split` (default: `train`)
- `--video_root` (local video root)
- optional columns:
  - `--prompt_col` (empty면 auto-detect)
  - `--video_name_col` (empty면 auto-detect)
- optional:
  - `--keywords` / `--no_keyword_filter`
  - `--max_count`
  - `--sample_mode`
  - `--require_local_video`

### Output
- `--output_csv`: canonical pair CSV (`sample_id,original_prompt,video_path`)

---

## 4) `scripts/run_cot_stepwise.py`

## Step 1 (Element Parsing)
### Input
- `--subset_csv`: canonical pair CSV
- `--step 1`

### Output JSONL (`--output_jsonl`)
```json
{
  "sample_id": "string",
  "original_prompt": "string",
  "video_path": "string",
  "parse": {
    "entities": [],
    "actions": [],
    "forces": [],
    "outcomes": []
  }
}
```

## Step 2 (Vision Checking)
### Input
- `--subset_csv`: canonical pair CSV
- `--step 2`
- `--input_jsonl`: step1 output

### Output JSONL
- step1과 동일한 구조, `parse`가 vision-checked 버전으로 갱신됨.

## Step 3 (Physics Reasoning)
### Input
- `--subset_csv`
- `--step 3`
- `--input_jsonl`: step2 output

### Output JSONL
```json
{
  "sample_id": "string",
  "original_prompt": "string",
  "video_path": "string",
  "parse": {"entities": [], "actions": [], "forces": [], "outcomes": []},
  "reason": "string"
}
```

## Step 4 (Data Scoring)
### Input
- `--subset_csv`
- `--step 4`
- `--input_jsonl`: step3 output

### Output JSONL
```json
{
  "sample_id": "string",
  "original_prompt": "string",
  "video_path": "string",
  "parse": {...},
  "reason": "string",
  "physics_richness": 0.0
}
```

## Step 5 (Prompt Extending)
### Input
- `--subset_csv`
- `--step 5`
- `--input_jsonl`: step4 output

### Output JSONL
```json
{
  "sample_id": "string",
  "original_prompt": "string",
  "video_path": "string",
  "parse": {...},
  "reason": "string",
  "physics_richness": 0.0,
  "extended": "string"
}
```

### Error Record (all steps)
에러 발생 시 다음 형태로 기록됨.
```json
{
  "sample_id": "string",
  "original_prompt": "string",
  "video_path": "string",
  "error": "exception message"
}
```

---

## 5) `scripts/run_cot_scoring.py` (all-in-one)

### Input
- `--subset_csv`: canonical pair CSV
- model/runtime args: `--model_name`, `--num_frames`, `--max_new_tokens`

### Outputs
1. `--output_csv`
   - columns:
     - `sample_id`
     - `original_prompt`
     - `video_path`
     - `physics_richness` (float)
     - `reason` (str)
     - `extended` (str)
     - `parse_json` (json string)
2. `--output_jsonl`
   - raw model 결과 (또는 error record)
