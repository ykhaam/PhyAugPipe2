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

## 4) `scripts/run_cot_scoring.py` (all-in-one)

### Input
- `--subset_csv`: canonical pair CSV
- model/runtime args: `--model_name`, `--num_frames`, `--max_new_tokens`, `--device`
- multi-GPU optional args:
  - `--cuda_visible_devices` (예: `0,1`)
  - `--device_map` (default: `auto`)
  - `--max_memory_per_gpu` (예: `70GiB`)
- logging/intermediate optional args:
  - `--log_file` (run log file path)
  - `--output_steps_dir` (step1~5 intermediate JSONL output directory)

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
3. optional `--output_steps_dir`
   - `step1_parse.jsonl`
   - `step2_checked.jsonl`
   - `step3_reason.jsonl`
   - `step4_score.jsonl`
   - `step5_extended.jsonl`

### Step 4 deterministic scoring formulas
- `entity_interaction_score = clamp(0.25*multiple + 0.45*explicit_interaction + 0.30*chain_interaction, 0, 1)`
- `force_outcome_score = clamp(0.30*explicit_force + 0.30*explicit_outcome + 0.40*force_outcome_linked, 0, 1)`
- `causal_clarity_score = clamp(0.35*cause_effect + 0.30*multi_step + 0.35*visible_process_supported, 0, 1)`
- `physics_richness = clamp(0.30*entity_interaction + 0.30*force_outcome + 0.30*causal_clarity - 0.05*penalty_score, 0, 1)`

Backward compatibility:
- If old Step 4 JSON lacks `positive_checklist`, pipeline uses conservative defaults (`False` + empty keywords), so execution still succeeds.

---

## 5) `scripts/postcot_threshold_filter.py` (Stage B)

### Input
- `--input_jsonl`: scored JSONL (`run_cot_scoring.py`의 `--output_jsonl`, 예: `outputs/scored/scored.jsonl`)
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

## 6) `scripts/postcot_action_cluster.py` (Stage C)

### Input
- `--input_jsonl`: Stage B output
- optional:
  - `--categories_file`: newline-separated category list
  - `--prompt_field`: `original_prompt` or `extended` (default: `original_prompt`)
    - 기본 `scored.jsonl`에서는 `original_prompt` 사용 권장
    - 확장 프롬프트 실험 데이터가 있으면 `extended` 선택 가능
  - `--model_name`: sentence-transformer model name
  - `--batch_size`
  - `--low_margin_threshold` (default: `0.05`)
  - `--output_stats_json` (category별 margin 통계 JSON)

### Output
- `--output_jsonl`: input fields +
  - `action_category` (str)
  - `action_match_score` (float, cosine similarity)
  - `top1_score` (float)
  - `top2_score` (float, category가 1개면 `0.0`)
  - `margin` (float, `top1_score - top2_score`)
- optional `--output_csv`
- optional `--output_stats_json`:
  - key: `action_category`
  - value:
    - `count`
    - `mean_margin`
    - `low_margin_ratio` (`margin < low_margin_threshold`)
- optional `--output_hist_json` (H_f): Stage C→D histogram contract 파일

### Stage C→D Contract: H_f (Histogram JSON)
- Stage C가 `--output_hist_json`으로 생성하고 Stage D가 `--input_hist_json`으로 소비
- schema:
```json
{
  "counts": {
    "<action_category>": 123
  },
  "total_count": 1234,
  "ratios": {
    "<action_category>": 0.0997
  }
}
```
- constraints:
  - `counts`: `{category: count}` 매핑 (count는 정수)
  - `total_count = sum(counts.values())`
  - `ratios[category] = counts[category] / total_count`

---

## 8) `scripts/postcot_physics_resample.py` (Stage D)

### Input
- `--input_jsonl`: Stage C output (must include `action_category`)
- `--N` or `--budget`: final sample count (`N` = total sampling budget)
- optional:
  - `--difficulty_field` (default: `videocon_physics_score`)
  - `--fallback_difficulty`: `inverse_physics_richness` or `uniform`
  - `--difficulty_config` (default: `configs/action_difficulty.yaml`)
  - `--representative_topk`
  - `--min_per_category`
  - `--min_count` (coverage 하한)
  - `--ambiguity_threshold` (default: `0.05`)
  - `--low_priority_mode`: `exclude` or `bucket`
  - `--difficulty_weights` (e.g. `failure=0.5,prior=0.3,ambiguity=0.2`)
  - `--videophy2_mode` (`auto`|`command`|`off`, default `auto`)
  - `--videophy2_eval_command` (optional VideoPhy2 representative evaluation command template with `{input_jsonl}` and `{output_jsonl}`)
  - `--videophy2_repo` (default: `https://github.com/Hritikbansal/videophy/tree/main/VIDEOPHY2`)
  - `--videophy2_root` (default: `VIDEOPHY2`, local clone path used in `auto` mode)
  - `--videophy2_checkpoint` (required in `auto` mode)
  - `--video_path_field`, `--caption_field` (bridge input mapping, default `video_path`, `original_prompt`)
  - `--seed`
  - `--input_hist_json` (H_f 파일; 제공 시 category 누락/예상 count 불일치 검증 수행)
  - `--output_csv`

### Method Summary
1. Category별로 `action_match_score` 상위 대표 샘플(top-nc) 선택
   - 기본(`--videophy2_mode auto`)으로 대표 샘플을 VideoPhy2로 채점 후 `difficulty_field`로 사용
   - `command` 모드에서는 사용자 제공 커맨드 템플릿으로 채점
   - `off` 모드에서는 외부 채점 없이 기존 필드/ fallback 사용
2. 결합 난이도 계산
   - `failure = 1 - mean(difficulty_field)` (없으면 fallback 사용)
   - `prior_difficulty = difficulty_config[action_category]` (없으면 0.5)
   - `ambiguity = mean(margin < ambiguity_threshold)`
   - `difficulty = w_failure*failure + w_prior*prior_difficulty + w_ambiguity*ambiguity`
3. `min_count` 미달 category는 `low_priority_mode` 정책으로 처리
4. difficulty 비례로 budget 분배
5. 각 category에서 상위 매칭 샘플부터 선택

### Output
- `--output_jsonl`: 최종 sampled subset
- optional `--output_csv`
