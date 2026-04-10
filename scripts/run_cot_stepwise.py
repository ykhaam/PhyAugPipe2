#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
import os



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CoT pipeline step-by-step with previous-step JSONL chaining")
    p.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], required=True)
    p.add_argument("--subset_csv", required=True, help="CSV with sample_id, original_prompt, video_path")
    p.add_argument("--input_jsonl", default="", help="Required for step>=2; previous step output JSONL")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--prompt_template", default="prompts/cot_filtering_prompt.txt")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--device_map", default="auto", help="transformers device_map (used when --device auto)")
    p.add_argument("--max_memory_per_gpu", default="", help="e.g. 70GiB; used when --device auto")
    p.add_argument("--cuda_visible_devices", default="", help="e.g. 0,1,2,3")
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def _load_subset(path: str, max_samples: int):
    import pandas as pd

    df = pd.read_csv(path)
    if max_samples > 0:
        df = df.head(max_samples)
    return df


def _load_previous(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id", ""))
            if sid:
                out[sid] = row
    return out


def _record_base(row: dict) -> dict:
    return {
        "sample_id": str(row["sample_id"]),
        "original_prompt": str(row["original_prompt"]),
        "video_path": str(row["video_path"]),
    }


def main() -> None:
    args = parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from tqdm import tqdm
    from phyaugpipe.pipeline import CoTFilteringPipeline, PipelineConfig
    from phyaugpipe.schemas import SampleRecord

    df = _load_subset(args.subset_csv, args.max_samples)
    prev = None
    if args.step >= 2:
        if not args.input_jsonl:
            raise ValueError("--input_jsonl is required for step >= 2")
        prev = _load_previous(args.input_jsonl)

    pipe = CoTFilteringPipeline(
        PipelineConfig(
            model_name=args.model_name,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            prompt_template_path=args.prompt_template,
            device=args.device,
            device_map=args.device_map,
            max_memory_per_gpu=args.max_memory_per_gpu,
        )
    )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(df.to_dict(orient="records"), total=len(df)):
            base = _record_base(row)
            sample = SampleRecord(**base)
            sid = base["sample_id"]

            try:
                if args.step == 1:
                    parse_obj = pipe.run_step1_parse(sample)
                    payload = {**base, "parse": parse_obj}
                elif args.step == 2:
                    prev_row = prev.get(sid, {}) if prev is not None else {}
                    parse_in = prev_row.get("parse", {})
                    parse_obj = pipe.run_step2_vision_check(sample, parse_in)
                    payload = {**base, "parse": parse_obj}
                elif args.step == 3:
                    prev_row = prev.get(sid, {}) if prev is not None else {}
                    parse_obj = prev_row.get("parse", {})
                    reason = pipe.run_step3_reason(sample, parse_obj)
                    payload = {**base, "parse": parse_obj, "reason": reason}
                elif args.step == 4:
                    prev_row = prev.get(sid, {}) if prev is not None else {}
                    parse_obj = prev_row.get("parse", {})
                    reason = prev_row.get("reason", "")
                    step4 = pipe.run_step4_score(sample, parse_obj, reason)
                    payload = {
                        **base,
                        "parse": parse_obj,
                        "reason": reason,
                        "positive_checklist": step4.get("positive_checklist", {}),
                        "penalty_analysis": step4.get("penalty_analysis", {}),
                        "score_breakdown": step4.get("score_breakdown", {}),
                        "physics_richness": float(step4.get("physics_richness", 0.0)),
                    }
                else:
                    prev_row = prev.get(sid, {}) if prev is not None else {}
                    parse_obj = prev_row.get("parse", {})
                    reason = prev_row.get("reason", "")
                    score = float(prev_row.get("physics_richness", 0.0))
                    extended = pipe.run_step5_extend(sample, parse_obj, reason)
                    payload = {
                        **base,
                        "parse": parse_obj,
                        "reason": reason,
                        "positive_checklist": prev_row.get("positive_checklist", {}),
                        "penalty_analysis": prev_row.get("penalty_analysis", {}),
                        "score_breakdown": prev_row.get("score_breakdown", {}),
                        "physics_richness": score,
                        "extended": extended,
                    }

                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as e:
                f.write(json.dumps({**base, "error": str(e)}, ensure_ascii=False) + "\n")

    print(f"step={args.step} saved={out_path}")


if __name__ == "__main__":
    main()
