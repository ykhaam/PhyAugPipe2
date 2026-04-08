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

- Note: Step1~5 모두 프레임 증거를 사용하며, Stepwise 경로도 CoT 템플릿 기반 규칙을 따릅니다.


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
  "positive_checklist": {
    "multiple_physical_entities_present": false,
    "explicit_entity_interaction_present": false,
    "chain_or_dependent_interaction_present": false,
    "explicit_force_present": false,
    "explicit_outcome_present": false,
    "force_outcome_causally_linked": false,
    "cause_effect_relation_present": false,
    "multi_step_causality_present": false,
    "reason_supported_by_visible_process": false,
    "interaction_keywords": [],
    "force_keywords": [],
    "outcome_keywords": [],
    "causal_keywords": []
  },
  "penalty_analysis": {
    "camera_motion_dominant": false,
    "stylized_rendering": false,
    "static_aftermath": false,
    "showcase_without_interaction": false,
    "penalty_keywords": []
  },
  "score_breakdown": {
    "entity_interaction_score": 0.0,
    "force_outcome_score": 0.0,
    "causal_clarity_score": 0.0,
    "penalty_score": 0.0
  },
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
  "positive_checklist": {...},
  "penalty_analysis": {...},
  "score_breakdown": {...},
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
     - `penalty_score` (float)
     - `reason` (str)
     - `extended` (str)
     - `parse_json` (json string)
     - `penalty_analysis_json` (json string)
     - `score_breakdown_json` (json string)
2. `--output_jsonl`
   - raw model 결과 (또는 error record)

### Step 4 deterministic scoring formulas
- `entity_interaction_score = clamp(0.25*multiple + 0.45*explicit_interaction + 0.30*chain_interaction, 0, 1)`
- `force_outcome_score = clamp(0.30*explicit_force + 0.30*explicit_outcome + 0.40*force_outcome_linked, 0, 1)`
- `causal_clarity_score = clamp(0.35*cause_effect + 0.30*multi_step + 0.35*visible_process_supported, 0, 1)`
- `physics_richness = clamp(0.30*entity_interaction + 0.30*force_outcome + 0.30*causal_clarity - 0.10*penalty_score, 0, 1)`

Backward compatibility:
- If old Step 4 JSON lacks `positive_checklist`, pipeline uses conservative defaults (`False` + empty keywords), so execution still succeeds.

---

## 6) `scripts/postcot_threshold_filter.py` (Stage B)

### Input
- `--input_jsonl`: scored JSONL (`step4_score.jsonl` 권장, `step5_extended.jsonl`도 가능)
- filtering mode (exactly one):
  - `--threshold <float>`: `physics_richness >= threshold`
  - `--top_quantile <float>`: top q fraction by `physics_richness`
- optional:
  - `--drop_errors`
  - `--output_csv`

### Output
- `--output_jsonl`: filtered rows (original fields preserved)
- optional `--output_csv`

---

## 7) `scripts/postcot_action_cluster.py` (Stage C)

### Input
- `--input_jsonl`: Stage B output
- optional:
  - `--categories_file`: newline-separated category list
  - `--prompt_field`: `original_prompt` or `extended` (default: `original_prompt`)
    - `step4_score.jsonl` 기반이면 `original_prompt` 사용 권장
    - `step5_extended.jsonl` 기반 실험 시 `extended` 선택 가능
  - `--model_name`: sentence-transformer model name
  - `--batch_size`

### Output
- `--output_jsonl`: input fields +
  - `action_category` (str)
  - `action_match_score` (float, cosine similarity)
- optional `--output_csv`

---

## 8) `scripts/postcot_physics_resample.py` (Stage D)

### Input
- `--input_jsonl`: Stage C output (must include `action_category`)
- `--budget`: final sample count
- optional:
  - `--difficulty_field` (default: `videocon_physics_score`)
  - `--fallback_difficulty`: `inverse_physics_richness` or `uniform`
  - `--representative_topk`
  - `--min_per_category`
  - `--seed`
  - `--output_csv`

### Method Summary
1. Category별로 `action_match_score` 상위 대표 샘플 선택
2. `difficulty_field` 평균(없으면 fallback)으로 category difficulty 추정
3. difficulty 비례로 budget 분배
4. 각 category에서 상위 매칭 샘플부터 선택

### Output
- `--output_jsonl`: 최종 sampled subset
- optional `--output_csv`
