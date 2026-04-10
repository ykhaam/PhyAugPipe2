#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import csv
import json
import logging
import os
from contextlib import ExitStack
from typing import Any

import pandas as pd
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CoT filtering/scoring (Algorithm 1 steps 1~5)")
    p.add_argument("--subset_csv", required=True, help="CSV with sample_id, original_prompt, video_path")
    p.add_argument("--output_csv", required=True, help="Scored CSV output")
    p.add_argument("--output_jsonl", required=True, help="Raw JSONL output")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--prompt_template", default="prompts/cot_filtering_prompt.txt")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--device_map", default="auto", help="transformers device_map (used when --device auto)")
    p.add_argument("--max_memory_per_gpu", default="", help="e.g. 70GiB; used when --device auto")
    p.add_argument("--cuda_visible_devices", default="", help="e.g. 0,1,2,3")
    p.add_argument("--output_steps_dir", default="", help="Optional dir for step1~5 JSONL intermediates")
    p.add_argument("--log_file", default="", help="Optional log file path")
    p.add_argument("--max_samples", type=int, default=0, help="0 means all rows")
    return p.parse_args()


def _build_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("run_cot_scoring")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    return logger


def _base_payload(sample: Any) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "original_prompt": sample.original_prompt,
        "video_path": sample.video_path,
    }


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    logger = _build_logger(args.log_file)
    logger.info("Starting CoT scoring")
    logger.info("subset_csv=%s output_csv=%s output_jsonl=%s", args.subset_csv, args.output_csv, args.output_jsonl)
    if args.output_steps_dir:
        logger.info("step outputs enabled: %s", args.output_steps_dir)

    from phyaugpipe.pipeline import CoTFilteringPipeline, PipelineConfig
    from phyaugpipe.schemas import SampleRecord

    df = pd.read_csv(args.subset_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

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

    out_csv = Path(args.output_csv)
    out_jsonl = Path(args.output_jsonl)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "sample_id",
        "original_prompt",
        "video_path",
        "physics_richness",
        "penalty_score",
        "reason",
        "extended",
        "parse_json",
        "penalty_analysis_json",
        "score_breakdown_json",
    ]

    step_files: dict[int, Any] = {}
    with ExitStack() as stack:
        f_csv = stack.enter_context(out_csv.open("w", newline="", encoding="utf-8"))
        f_jsonl = stack.enter_context(out_jsonl.open("w", encoding="utf-8"))

        if args.output_steps_dir:
            steps_dir = Path(args.output_steps_dir)
            steps_dir.mkdir(parents=True, exist_ok=True)
            for step_idx, name in [
                (1, "step1_parse.jsonl"),
                (2, "step2_checked.jsonl"),
                (3, "step3_reason.jsonl"),
                (4, "step4_score.jsonl"),
                (5, "step5_extended.jsonl"),
            ]:
                step_files[step_idx] = stack.enter_context((steps_dir / name).open("w", encoding="utf-8"))

        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        writer.writeheader()

        for row in tqdm(df.to_dict(orient="records"), total=len(df)):
            sample = SampleRecord(
                sample_id=str(row["sample_id"]),
                original_prompt=str(row["original_prompt"]),
                video_path=str(row["video_path"]),
            )
            base = _base_payload(sample)
            try:
                if step_files:
                    parse1 = pipe.run_step1_parse(sample)
                    step_files[1].write(json.dumps({**base, "parse": parse1}, ensure_ascii=False) + "\n")

                    parse2 = pipe.run_step2_vision_check(sample, parse1)
                    step_files[2].write(json.dumps({**base, "parse": parse2}, ensure_ascii=False) + "\n")

                    reason = pipe.run_step3_reason(sample, parse2)
                    step_files[3].write(json.dumps({**base, "parse": parse2, "reason": reason}, ensure_ascii=False) + "\n")

                    step4 = pipe.run_step4_score(sample, parse2, reason)
                    step4_payload = {
                        **base,
                        "parse": parse2,
                        "reason": reason,
                        "positive_checklist": step4.get("positive_checklist", {}),
                        "penalty_analysis": step4.get("penalty_analysis", {}),
                        "score_breakdown": step4.get("score_breakdown", {}),
                        "physics_richness": float(step4.get("physics_richness", 0.0)),
                    }
                    step_files[4].write(json.dumps(step4_payload, ensure_ascii=False) + "\n")

                    extended = pipe.run_step5_extend(sample, parse2, reason)
                    step5_payload = {**step4_payload, "extended": extended}
                    step_files[5].write(json.dumps(step5_payload, ensure_ascii=False) + "\n")

                    parse_json = json.dumps(parse2, ensure_ascii=False)
                    physics_richness = float(step4_payload["physics_richness"])
                    penalty_analysis = step4_payload["penalty_analysis"]
                    score_breakdown = step4_payload["score_breakdown"]
                    reason_text = reason
                    extended_text = extended
                    final_payload = {
                        "sample_id": sample.sample_id,
                        "video_path": sample.video_path,
                        "original": sample.original_prompt,
                        "parse": parse2,
                        "reason": reason_text,
                        "extended": extended_text,
                        "physics_richness": physics_richness,
                        "positive_checklist": step4_payload["positive_checklist"],
                        "penalty_analysis": penalty_analysis,
                        "score_breakdown": score_breakdown,
                        "physics_label": None,
                    }
                else:
                    result = pipe.run_one(sample)
                    parse_json = result.parse.model_dump_json(ensure_ascii=False)
                    physics_richness = result.physics_richness
                    penalty_analysis = result.penalty_analysis
                    score_breakdown = result.score_breakdown
                    reason_text = result.reason
                    extended_text = result.extended
                    final_payload = {
                        "sample_id": sample.sample_id,
                        "video_path": sample.video_path,
                        **result.model_dump(),
                    }

                rec = {
                    "sample_id": sample.sample_id,
                    "original_prompt": sample.original_prompt,
                    "video_path": sample.video_path,
                    "physics_richness": physics_richness,
                    "penalty_score": score_breakdown.get("penalty_score", 0.0),
                    "reason": reason_text,
                    "extended": extended_text,
                    "parse_json": parse_json,
                    "penalty_analysis_json": json.dumps(penalty_analysis, ensure_ascii=False),
                    "score_breakdown_json": json.dumps(score_breakdown, ensure_ascii=False),
                }
                writer.writerow(rec)
                f_jsonl.write(json.dumps(final_payload, ensure_ascii=False) + "\n")
                logger.info("done sample_id=%s", sample.sample_id)
            except Exception as e:
                err_payload = {
                    "sample_id": sample.sample_id,
                    "video_path": sample.video_path,
                    "error": str(e),
                }
                f_jsonl.write(json.dumps(err_payload, ensure_ascii=False) + "\n")
                logger.exception("failed sample_id=%s", sample.sample_id)

    print(f"saved csv: {out_csv}")
    print(f"saved jsonl: {out_jsonl}")
    if args.output_steps_dir:
        print(f"saved steps dir: {args.output_steps_dir}")
    if args.log_file:
        print(f"saved log: {args.log_file}")


if __name__ == "__main__":
    main()
