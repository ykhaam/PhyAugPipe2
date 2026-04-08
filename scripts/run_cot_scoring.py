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

import pandas as pd
from tqdm import tqdm

from phyaugpipe.pipeline import CoTFilteringPipeline, PipelineConfig
from phyaugpipe.schemas import SampleRecord


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
    p.add_argument("--max_samples", type=int, default=0, help="0 means all rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
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

    with out_csv.open("w", newline="", encoding="utf-8") as f_csv, out_jsonl.open("w", encoding="utf-8") as f_jsonl:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        writer.writeheader()

        for row in tqdm(df.to_dict(orient="records"), total=len(df)):
            sample = SampleRecord(
                sample_id=str(row["sample_id"]),
                original_prompt=str(row["original_prompt"]),
                video_path=str(row["video_path"]),
                metadata={k: v for k, v in row.items() if k not in {"sample_id", "original_prompt", "video_path"}},
            )
            try:
                result = pipe.run_one(sample)
                rec = {
                    "sample_id": sample.sample_id,
                    "original_prompt": sample.original_prompt,
                    "video_path": sample.video_path,
                    "physics_richness": result.physics_richness,
                    "penalty_score": result.score_breakdown.get("penalty_score", 0.0),
                    "reason": result.reason,
                    "extended": result.extended,
                    "parse_json": result.parse.model_dump_json(ensure_ascii=False),
                    "penalty_analysis_json": json.dumps(result.penalty_analysis, ensure_ascii=False),
                    "score_breakdown_json": json.dumps(result.score_breakdown, ensure_ascii=False),
                }
                writer.writerow(rec)

                payload = {
                    "sample_id": sample.sample_id,
                    "video_path": sample.video_path,
                    **result.model_dump(),
                }
                f_jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as e:
                err_payload = {
                    "sample_id": sample.sample_id,
                    "video_path": sample.video_path,
                    "error": str(e),
                }
                f_jsonl.write(json.dumps(err_payload, ensure_ascii=False) + "\n")

    print(f"saved csv: {out_csv}")
    print(f"saved jsonl: {out_jsonl}")


if __name__ == "__main__":
    main()
