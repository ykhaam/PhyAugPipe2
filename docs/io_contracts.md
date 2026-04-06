# I/O Contracts

이 문서는 각 스크립트(단계)의 입력/출력 형식을 명시합니다.

## Model Default

- default: `Qwen/Qwen2.5-VL-3B-Instruct` (3B급 VL 모델)
- note: video+frame 입력이 필요해 텍스트 전용 3B 모델은 기본 파이프라인에서 사용할 수 없음

---

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

## 3-B) `scripts/build_wisa_pairs_from_metadata_json.py`

### Input
- `--metadata_json`: WISA metadata JSON 파일
- `--video_root`: 로컬 비디오 루트
- `--output_csv`: 출력 CSV
- optional:
  - `--prompt_key` (default: `captions`)
  - `--video_name_key` (default: `video_name`)
  - `--require_local_video`
  - `--include_meta` (width/fps/label/physical_* 컬럼 확장)
  - `--dedup_by_video_name`
  - `--max_count`, `--sample_mode`, `--seed`

### Output
- canonical pair CSV
  - required columns: `sample_id, original_prompt, video_path`
  - optional columns: `width,height,fps,duration,motion_score,...,physical_phys_law,...`

### Robustness
- 정상 JSON list뿐 아니라 `[...] , [...]`처럼 이어붙은 malformed 형태도 순차 파싱으로 병합 시도

---

## 4) `scripts/run_cot_stepwise.py`

## Step 1 (Element Parsing)
### Input
- `--subset_csv`: canonical pair CSV
- `--step 1`
- `--device`: `auto|cpu|cuda` (CPU 점검 시 `cpu`)

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
- model/runtime args: `--model_name`, `--num_frames`, `--max_new_tokens`, `--device`

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
