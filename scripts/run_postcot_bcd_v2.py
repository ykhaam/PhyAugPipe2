#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Post-CoT Stage B->C->D in one command (merge-safe wrapper)."
    )
    # Shared
    p.add_argument("--workdir", default=".", help="Working directory where stage scripts exist")
    # Stage B
    p.add_argument("--stageb_input_jsonl", required=True, help="Input score jsonl for Stage B")
    p.add_argument("--stageb_output_jsonl", required=True)
    p.add_argument("--stageb_output_csv", default="")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--top_quantile", type=float, default=None)
    # Stage C
    p.add_argument("--stagec_output_jsonl", required=True)
    p.add_argument("--stagec_output_csv", default="")
    p.add_argument("--categories_file", default="configs/action_categories.txt")
    p.add_argument("--prompt_field", default="original_prompt", choices=["original_prompt", "extended"])
    p.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--low_margin_threshold", type=float, default=0.05)
    p.add_argument("--stagec_output_stats_json", default="")
    p.add_argument("--stagec_output_hist_json", default="")
    # Stage D
    p.add_argument("--staged_output_jsonl", required=True)
    p.add_argument("--staged_output_csv", default="")
    p.add_argument("--N", type=int, required=True, help="Total sampling budget")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--difficulty_field", default="videocon_physics_score")
    p.add_argument("--fallback_difficulty", default="inverse_physics_richness", choices=["inverse_physics_richness", "uniform"])
    p.add_argument("--representative_topk", type=int, default=20)
    p.add_argument("--min_count", type=int, default=1)
    p.add_argument("--low_priority_mode", default="exclude", choices=["exclude", "bucket"])
    p.add_argument("--videophy2_mode", default="auto", choices=["auto", "command", "off"])
    p.add_argument("--videophy2_eval_command", default="")
    p.add_argument("--videophy2_root", default="VIDEOPHY2")
    p.add_argument("--videophy2_checkpoint", default="")
    p.add_argument("--video_path_field", default="video_path")
    p.add_argument("--caption_field", default="original_prompt")
    return p.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    args = parse_args()
    cwd = Path(args.workdir).resolve()

    stageb = [
        "python", "scripts/postcot_threshold_filter.py",
        "--input_jsonl", args.stageb_input_jsonl,
        "--output_jsonl", args.stageb_output_jsonl,
    ]
    if args.stageb_output_csv:
        stageb += ["--output_csv", args.stageb_output_csv]
    if args.threshold is not None:
        stageb += ["--threshold", str(args.threshold)]
    if args.top_quantile is not None:
        stageb += ["--top_quantile", str(args.top_quantile)]
    _run(stageb, cwd)

    stagec = [
        "python", "scripts/postcot_action_cluster.py",
        "--input_jsonl", args.stageb_output_jsonl,
        "--output_jsonl", args.stagec_output_jsonl,
        "--categories_file", args.categories_file,
        "--prompt_field", args.prompt_field,
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--low_margin_threshold", str(args.low_margin_threshold),
    ]
    if args.stagec_output_csv:
        stagec += ["--output_csv", args.stagec_output_csv]
    if args.stagec_output_stats_json:
        stagec += ["--output_stats_json", args.stagec_output_stats_json]
    if args.stagec_output_hist_json:
        stagec += ["--output_hist_json", args.stagec_output_hist_json]
    _run(stagec, cwd)

    staged = [
        "python", "scripts/postcot_physics_resample.py",
        "--input_jsonl", args.stagec_output_jsonl,
        "--output_jsonl", args.staged_output_jsonl,
        "--N", str(args.N),
        "--tau", str(args.tau),
        "--difficulty_field", args.difficulty_field,
        "--fallback_difficulty", args.fallback_difficulty,
        "--representative_topk", str(args.representative_topk),
        "--min_count", str(args.min_count),
        "--low_priority_mode", args.low_priority_mode,
        "--videophy2_mode", args.videophy2_mode,
        "--videophy2_eval_command", args.videophy2_eval_command,
        "--videophy2_root", args.videophy2_root,
        "--videophy2_checkpoint", args.videophy2_checkpoint,
        "--video_path_field", args.video_path_field,
        "--caption_field", args.caption_field,
    ]
    if args.staged_output_csv:
        staged += ["--output_csv", args.staged_output_csv]
    if args.stagec_output_hist_json:
        staged += ["--input_hist_json", args.stagec_output_hist_json]
    _run(staged, cwd)

    print("[DONE] Stage B->C->D completed")


if __name__ == "__main__":
    main()

