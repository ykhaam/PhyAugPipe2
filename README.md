# PhyAugPipe2 (minimal)

논문(PhyAug 관련) Data Filtering with CoT의 **실행 가능한 최소 파이프라인**입니다.
현재 구현은 요청대로 threshold 강제 분류보다 앞 단계 중심이며, 출력으로 physics_richness 점수를 제공합니다.

## 1) 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## 2) 입력 데이터 가정

`sample_id, original_prompt, video_path` 컬럼을 갖는 CSV가 필요합니다.

## 모델 설정 (중요)

기본 모델은 `Qwen/Qwen2.5-VL-3B-Instruct` 입니다.
- 논문 대비 **더 작은 3B급 모델**로 재현/디버깅을 쉽게 하기 위한 기본값입니다.
- 영상 프레임 입력이 필요하므로 일반 텍스트 모델(`Qwen2.5-3B-Instruct`)이 아니라 **VL(vision-language) 변형**을 사용합니다.
- 더 큰 모델을 쓰고 싶으면 `--model_name`으로 교체하세요.

---

## I/O 형식 명세

각 스크립트/단계별 입력·출력 포맷은 `docs/io_contracts.md`에 정리했습니다.


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
  --dedup_by_video_name
```

출력은 canonical 컬럼(`sample_id, original_prompt, video_path`) + 선택 메타컬럼입니다.

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
