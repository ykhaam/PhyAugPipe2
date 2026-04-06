# PhyAugPipe2 (minimal)

논문(PhyAug 관련) Data Filtering with CoT의 **실행 가능한 최소 파이프라인**입니다.
현재 구현은 요청대로 threshold 강제 분류보다 앞 단계 중심이며, 출력으로 physics_richness 점수를 제공합니다.

## 1) 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# scripts/* 는 내부적으로 repo/src를 자동 추가하므로 PYTHONPATH 없이 실행 가능
export PYTHONPATH=src  # (옵션)
```

## 2) 입력 데이터 가정

`sample_id, original_prompt, video_path` 컬럼을 갖는 CSV가 필요합니다.

## Troubleshooting

- `RuntimeError: operator torchvision::nms does not exist`
  - 이 오류는 보통 `torch`/`torchvision` 버전 불일치에서 발생합니다.
  - 우선 `export TRANSFORMERS_NO_TORCHVISION=1` 후 재실행해 보세요.
  - 코드에서 torchvision import 실패 시 최소 stub를 주입하도록 처리했습니다(메타 등록 오류 회피).
  - 메타데이터 스크립트는 lazy import + 경량 비전 입력 처리로 import 단계 크래시를 피하도록 수정했습니다.
  - 현재 파이프라인은 `qwen_vl_utils`를 직접 쓰지 않으며, `transformers` processor만으로 image/video payload를 구성합니다.

---

## 모델 설정 (중요)

기본 모델은 `Qwen/Qwen2.5-VL-3B-Instruct` 입니다.
- 논문 대비 **더 작은 3B급 모델**로 재현/디버깅을 쉽게 하기 위한 기본값입니다.
- 영상 프레임 입력이 필요하므로 일반 텍스트 모델(`Qwen2.5-3B-Instruct`)이 아니라 **VL(vision-language) 변형**을 사용합니다.
- 더 큰 모델을 쓰고 싶으면 `--model_name`으로 교체하세요.

---

## I/O 형식 명세

각 스크립트/단계별 입력·출력 포맷은 `docs/io_contracts.md`에 정리했습니다.


---

## video_folder 기준 metadata subset 만들기 (0.zip 전용 등)

압축 해제한 특정 폴더(예: `0.zip`만 풀어둔 폴더)에 실제 존재하는 비디오들만 남겨 metadata JSON subset을 만들 수 있습니다.

```bash
python scripts/filter_wisa_metadata_by_videos.py \
  --metadata_json /path/to/metadata.json \
  --video_folder /path/to/0_zip_videos \
  --output_json data/subsets/metadata_only_0zip.json
```

- `video_name`이 `xxx.mp4`여도, 폴더에 `xxx`(확장자 없음) 파일이 있으면 매칭됩니다.
- 반대로 폴더 파일이 `xxx.mp4`이고 metadata가 `xxx`여도 매칭됩니다.

---

## WISA metadata.json 파싱

네가 준 형태의 `metadata.json`(중첩 `physical_annotation` 포함)에서 바로 pair CSV를 만들 수 있습니다.
비정상 형태(예: `[...] , [...]`처럼 배열이 이어붙은 JSON)도 파서가 최대한 복구합니다.

```bash
python scripts/build_wisa_pairs_from_metadata_json.py \
  --metadata_json /path/to/metadata.json \
  --video_root /path/to/wisa/videos \
  --output_csv data/subsets/wisa_from_meta.csv \
  --require_local_video \
  --include_meta \
  --dedup_by_video_name \
  --sanitize_text_fields \
  --drop_corrupted
```

출력은 canonical 컬럼(`sample_id, original_prompt, video_path`) + 선택 메타컬럼입니다.
- **video-text는 항상 한 세트**로 유지하세요(`sample_id + original_prompt + video_path` 필수).
- 기본적으로 `captions`만 `original_prompt`로 사용합니다(물리 annotation은 별도 메타로 유지). 필요 시 `--allow_non_caption_prompt`로만 override 가능합니다.
- 권장 메타: `width,height,fps,duration,motion_score,motion_score_v2,visual_quality_score,text_bbox_num,text_bbox_ratio,label`
- 권장 물리: `physical_phys_law,physical_n0,physical_n1,physical_n2,physical_q0,physical_q1,physical_q2,physical_q3,physical_q4,physical_quantify_n0,physical_quantify_n1,physical_quantify_n2,physical_time_min_s/max_s,physical_temp_min_c/max_c,physical_density_range_count,physical_q3_bool`
- 줄바꿈이 많은 텍스트(`captions`, `physical_q4`)는 CSV에서 멀티라인 셀로 보이는 것이 정상입니다.
- 만약 셀 안에 `sample_id,original_prompt,video_path` 같은 헤더 문자열이 섞여 있으면 `--drop_corrupted`로 제거하세요.
- `--include_meta`를 주면 `physical_quantify_n0/n1/n2`와 파생 수치(`physical_time_min_s`, `physical_time_max_s`, `physical_temp_min_c`, `physical_temp_max_c`, `physical_density_range_count`, `physical_q3_bool`)도 함께 저장합니다.


---

## 3) 메타데이터 없이 로컬 video+text로 subset 생성 (빠른 디버깅용)

### 옵션 1: manifest CSV가 있는 경우

```bash
python scripts/build_local_pairs_csv.py \
  --manifest_csv /path/to/manifest.csv \
  --sample_id_col sample_id \
  --prompt_col original_prompt \
  --video_col video_path \
  --output_csv data/subsets/local_subset.csv \
  --sample_mode random \
  --max_count 200
```

### 옵션 2: `video_root` + `text_root`(동일 basename의 `.txt`)를 스캔

```bash
python scripts/build_local_pairs_csv.py \
  --video_root /path/to/videos \
  --text_root /path/to/texts \
  --output_csv data/subsets/local_subset.csv \
  --sample_mode top \
  --max_count 200
```

---

## 4) 단계별 실행 (강력 권장)

한 번에 돌리지 않고, 각 단계 결과를 확인하면서 디버깅합니다.

```bash
# Step 1: Element Parsing
python scripts/run_cot_stepwise.py \
  --step 1 \
  --subset_csv data/subsets/local_subset.csv \
  --output_jsonl outputs/steps/step1_parse.jsonl

# Step 2: Vision Checking
python scripts/run_cot_stepwise.py \
  --step 2 \
  --subset_csv data/subsets/local_subset.csv \
  --input_jsonl outputs/steps/step1_parse.jsonl \
  --output_jsonl outputs/steps/step2_checked.jsonl

# Step 3: Physics Reasoning
python scripts/run_cot_stepwise.py \
  --step 3 \
  --subset_csv data/subsets/local_subset.csv \
  --input_jsonl outputs/steps/step2_checked.jsonl \
  --output_jsonl outputs/steps/step3_reason.jsonl

# Step 4: Data Scoring
python scripts/run_cot_stepwise.py \
  --step 4 \
  --subset_csv data/subsets/local_subset.csv \
  --input_jsonl outputs/steps/step3_reason.jsonl \
  --output_jsonl outputs/steps/step4_score.jsonl

# Step 5: Prompt Extending
python scripts/run_cot_stepwise.py \
  --step 5 \
  --subset_csv data/subsets/local_subset.csv \
  --input_jsonl outputs/steps/step4_score.jsonl \
  --output_jsonl outputs/steps/step5_extended.jsonl
```

각 step 출력(JSONL)을 직접 열어보며 중간결과를 검증할 수 있습니다.

---


## CPU에서 테스트하기

가능합니다. 다만 속도가 매우 느릴 수 있어서 먼저 샘플/프레임 수를 줄여 확인하세요.

```bash
python scripts/run_cot_stepwise.py \
  --step 1 \
  --subset_csv data/subsets/local_subset.csv \
  --output_jsonl outputs/steps/step1_parse_cpu.jsonl \
  --device cpu \
  --max_samples 5 \
  --num_frames 4 \
  --max_new_tokens 256
```

일괄 실행도 동일하게 `--device cpu`를 주면 됩니다.

---

## 5) 참고: 기존 일괄 실행

```bash
python scripts/run_cot_scoring.py \
  --subset_csv data/subsets/local_subset.csv \
  --output_csv outputs/scored/scored.csv \
  --output_jsonl outputs/scored/scored.jsonl \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --num_frames 8 \
  --max_new_tokens 512
```

---

## 6) 구현 범위

- Step 1: Element Parsing
- Step 2: Vision Checking
- Step 3: Physics Reasoning
- Step 4: Data Scoring (0~1)
- Step 5: Prompt Extending

`physics_label` threshold는 강제하지 않고 모델 출력값을 그대로 유지합니다(미정이면 null).
