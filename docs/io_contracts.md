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
  - `--prompt_key` (default: `captions`, 권장)
  - `--allow_non_caption_prompt` (기본 차단 해제용)
  - `--video_name_key` (default: `video_name`)
  - `--require_local_video`
  - `--include_meta` (width/fps/label/physical_* 컬럼 확장)
  - `--dedup_by_video_name`
  - `--sanitize_text_fields` (CRLF 정규화/trim)
  - `--drop_corrupted` (헤더 오염 문자열 포함 row 제거)
  - `--max_count`, `--sample_mode`, `--seed`

### Output
- canonical pair CSV
  - required columns: `sample_id, original_prompt, video_path`
  - optional columns: `width,height,fps,duration,motion_score,...,physical_phys_law,...,physical_quantify_n0,physical_quantify_n1,physical_quantify_n2,physical_time_min_s,physical_time_max_s,physical_temp_min_c,physical_temp_max_c,physical_density_range_count,physical_q3_bool`

### Robustness
- 정상 JSON list뿐 아니라 `[...] , [...]`처럼 이어붙은 malformed 형태도 순차 파싱으로 병합 시도
- 멀티라인 텍스트는 CSV 셀 내 줄바꿈으로 저장되며 정상 동작
- `--drop_corrupted` 사용 시 헤더 문자열 오염 row 제거

---

## 3-C) `scripts/filter_wisa_metadata_by_videos.py`

### Input
- `--metadata_json`: 원본 metadata JSON
- `--video_folder`: 실제 비디오 파일이 있는 폴더 (예: `0.zip` 해제 폴더)
- `--output_json`: 필터된 metadata JSON
- optional:
  - `--video_name_key` (default: `video_name`)
  - `--recursive` (하위 폴더 재귀 탐색)

### Matching Rule
- metadata의 `video_name`과 폴더 파일명을 둘 다 비교
- 확장자 유무 모두 허용
  - `abc.mp4` (metadata) ↔ `abc` (file) 매칭
  - `abc` (metadata) ↔ `abc.mp4` (file) 매칭

### Output
- 입력 metadata와 동일한 객체 구조를 유지한 JSON list
- 단, `video_folder`에 실제 존재하는 비디오에 해당하는 row만 포함

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
