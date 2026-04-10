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
import os
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CoT filtering/scoring (Algorithm 1 steps 1~5)")
    p.add_argument("--subset_csv", required=True, help="CSV with sample_id, original_prompt, video_path")
    p.add_argument("--output_csv", required=True, help="Scored CSV output")
    p.add_argument("--output_jsonl", required=True, help="Raw JSONL output")
    p.add_argument(
        "--input_jsonl",
        default="",
        help="Optional previous-step JSONL state input (for stepwise chaining)",
    )
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--prompt_template", default="prompts/cot_filtering_prompt.txt")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--device_map", default="auto", help="transformers device_map (used when --device auto)")
    p.add_argument("--max_memory_per_gpu", default="", help="e.g. 70GiB; used when --device auto")
    p.add_argument("--cuda_visible_devices", default="", help="e.g. 0,1,2,3")
    p.add_argument("--max_samples", type=int, default=0, help="0 means all rows")
    p.add_argument(
        "--step_jsonl",
        default="",
        help="Step-level JSONL output. Default: <output_jsonl stem>.steps.jsonl",
    )
    p.add_argument(
        "--log_file",
        default="",
        help="Plain log file path. Default: <output_jsonl stem>.log",
    )
    p.add_argument(
        "--print_step_summary",
        action="store_true",
        help="Print per-step summary to stdout while running",
    )
    p.add_argument("--start_step", type=int, default=1, help="Start step index (1~5)")
    p.add_argument("--end_step", type=int, default=5, help="End step index (1~5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (1 <= args.start_step <= 5 and 1 <= args.end_step <= 5 and args.start_step <= args.end_step):
        raise ValueError("--start_step/--end_step must satisfy 1 <= start <= end <= 5")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from phyaugpipe.pipeline import CoTFilteringPipeline, PipelineConfig
    from phyaugpipe.schemas import CoTResult, ParsedElements, SampleRecord

    df = pd.read_csv(args.subset_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    prev_state_by_sample_id = {}
    if args.input_jsonl:
        in_path = Path(args.input_jsonl)
        if not in_path.exists():
            raise FileNotFoundError(f"input_jsonl not found: {args.input_jsonl}")
        with in_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = str(obj.get("sample_id", ""))
                if sid:
                    prev_state_by_sample_id[sid] = obj

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
    out_step_jsonl = (
        Path(args.step_jsonl)
        if args.step_jsonl
        else out_jsonl.with_name(f"{out_jsonl.stem}.steps{out_jsonl.suffix or '.jsonl'}")
    )
    out_log = Path(args.log_file) if args.log_file else out_jsonl.with_suffix(".log")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_step_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)

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

    def _append_log(f_log, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        line = f"[{ts}] {message}"
        f_log.write(line + "\n")
        f_log.flush()

    with (
        out_csv.open("w", newline="", encoding="utf-8") as f_csv,
        out_jsonl.open("w", encoding="utf-8") as f_jsonl,
        out_step_jsonl.open("w", encoding="utf-8") as f_step_jsonl,
        out_log.open("w", encoding="utf-8") as f_log,
    ):
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        writer.writeheader()
        _append_log(f_log, f"run started: rows={len(df)} model={args.model_name}")

        for row in tqdm(df.to_dict(orient="records"), total=len(df)):
            sample = SampleRecord(
                sample_id=str(row["sample_id"]),
                original_prompt=str(row["original_prompt"]),
                video_path=str(row["video_path"]),
            )
            _append_log(f_log, f"sample_start sample_id={sample.sample_id} video_path={sample.video_path}")
            try:
                prev_state = prev_state_by_sample_id.get(sample.sample_id, {})
                parse_step1 = prev_state.get("parse")
                parse_step2 = prev_state.get("parse")
                reason_step3 = str(prev_state.get("reason", ""))
                step4 = {
                    "positive_checklist": prev_state.get("positive_checklist", {}),
                    "penalty_analysis": prev_state.get("penalty_analysis", {}),
                    "score_breakdown": prev_state.get("score_breakdown", {}),
                    "physics_richness": float(prev_state.get("physics_richness", 0.0)),
                }
                extended_step5 = str(prev_state.get("extended", ""))

                if args.start_step <= 1 <= args.end_step:
                    parse_step1 = pipe.run_step1_parse(sample)
                    parse_step2 = parse_step1
                    f_step_jsonl.write(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "video_path": sample.video_path,
                                "step": 1,
                                "name": "parse_initial",
                                "result": {"parse": parse_step1},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if args.start_step <= 2 <= args.end_step:
                    if not isinstance(parse_step2, dict):
                        raise ValueError("Step2 requires parse object from step1/input_jsonl")
                    parse_step2 = pipe.run_step2_vision_check(sample, parse_step2)
                    f_step_jsonl.write(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "video_path": sample.video_path,
                                "step": 2,
                                "name": "parse_vision_checked",
                                "result": {"parse": parse_step2},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if args.start_step <= 3 <= args.end_step:
                    if not isinstance(parse_step2, dict):
                        raise ValueError("Step3 requires parse object from step2/input_jsonl")
                    reason_step3 = pipe.run_step3_reason(sample, parse_step2)
                    f_step_jsonl.write(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "video_path": sample.video_path,
                                "step": 3,
                                "name": "reason",
                                "result": {"reason": reason_step3},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if args.start_step <= 4 <= args.end_step:
                    if not isinstance(parse_step2, dict):
                        raise ValueError("Step4 requires parse object from step2/input_jsonl")
                    if not reason_step3:
                        raise ValueError("Step4 requires reason from step3/input_jsonl")
                    step4 = pipe.run_step4_score(sample, parse_step2, reason_step3)
                    f_step_jsonl.write(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "video_path": sample.video_path,
                                "step": 4,
                                "name": "score",
                                "result": step4,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if args.start_step <= 5 <= args.end_step:
                    if not isinstance(parse_step2, dict):
                        raise ValueError("Step5 requires parse object from step2/input_jsonl")
                    if not reason_step3:
                        raise ValueError("Step5 requires reason from step3/input_jsonl")
                    extended_step5 = pipe.run_step5_extend(sample, parse_step2, reason_step3)
                    f_step_jsonl.write(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "video_path": sample.video_path,
                                "step": 5,
                                "name": "extend",
                                "result": {"extended": extended_step5},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                result = CoTResult(
                    original=sample.original_prompt,
                    parse=ParsedElements(**pipe._normalize_parse(parse_step2)),
                    reason=reason_step3,
                    extended=extended_step5,
                    physics_richness=float(step4.get("physics_richness", 0.0)),
                    positive_checklist=step4.get("positive_checklist", {}),
                    penalty_analysis=step4.get("penalty_analysis", {}),
                    score_breakdown=step4.get("score_breakdown", {}),
                    physics_label=None,
                )
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
                    "original_prompt": sample.original_prompt,
                    "video_path": sample.video_path,
                    **result.model_dump(),
                }
                f_jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
                _append_log(
                    f_log,
                    (
                        f"sample_done sample_id={sample.sample_id} "
                        f"physics_richness={result.physics_richness:.4f} "
                        f"penalty={result.score_breakdown.get('penalty_score', 0.0):.4f}"
                    ),
                )
                if args.print_step_summary:
                    print(
                        f"[sample_id={sample.sample_id}] "
                        f"S1 parse={len(result.parse.actions)} actions / "
                        f"S3 reason_len={len(result.reason)} / "
                        f"S4 physics_richness={result.physics_richness:.4f} / "
                        f"S5 extended_len={len(result.extended)}"
                    )
            except Exception as e:
                err_payload = {
                    "sample_id": sample.sample_id,
                    "video_path": sample.video_path,
                    "error": str(e),
                }
                f_jsonl.write(json.dumps(err_payload, ensure_ascii=False) + "\n")
                f_step_jsonl.write(
                    json.dumps(
                        {
                            "sample_id": sample.sample_id,
                            "video_path": sample.video_path,
                            "step": "error",
                            "error": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                _append_log(f_log, f"sample_error sample_id={sample.sample_id} error={e}")

    print(f"saved csv: {out_csv}")
    print(f"saved jsonl: {out_jsonl}")
    print(f"saved step jsonl: {out_step_jsonl}")
    print(f"saved log: {out_log}")


if __name__ == "__main__":
    main()
