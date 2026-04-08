#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bridge runner for VideoPhy2 representative scoring")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--videophy2_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--score_field", default="videocon_physics_score")
    p.add_argument("--video_path_field", default="video_path")
    p.add_argument("--caption_field", default="original_prompt")
    return p.parse_args()


def _load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("input_jsonl is empty")
    return pd.DataFrame(rows)


def _pick_score_column(df: pd.DataFrame) -> str:
    for name in ["videocon_physics_score", "score", "physics_score", "prob", "prediction_score"]:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find score column in VideoPhy2 output columns={list(df.columns)}")


def main() -> None:
    args = parse_args()
    reps = _load_jsonl(args.input_jsonl)
    if "__rep_uid" not in reps.columns:
        raise ValueError("input_jsonl must include __rep_uid")
    if args.video_path_field not in reps.columns:
        raise ValueError(f"input_jsonl missing video path field: {args.video_path_field}")
    if args.caption_field not in reps.columns:
        raise ValueError(f"input_jsonl missing caption field: {args.caption_field}")

    with tempfile.TemporaryDirectory(prefix="videophy2_bridge_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_csv = tmp_dir_path / "videophy2_input.csv"
        output_csv = tmp_dir_path / "videophy2_output.csv"

        payload = pd.DataFrame({
            "__rep_uid": reps["__rep_uid"].astype(str),
            "video_path": reps[args.video_path_field].astype(str),
            "caption": reps[args.caption_field].astype(str),
        })
        payload.to_csv(input_csv, index=False)

        infer_script = Path(args.videophy2_root) / "videocon" / "training" / "pipeline_video" / "entailment_inference.py"
        if not infer_script.exists():
            raise FileNotFoundError(f"VideoPhy2 inference script not found: {infer_script}")

        cmd = [
            "python",
            str(infer_script),
            "--input_csv",
            str(input_csv),
            "--output_csv",
            str(output_csv),
            "--checkpoint",
            str(args.checkpoint),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"VideoPhy2 inference failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_csv.exists():
            raise FileNotFoundError(f"VideoPhy2 did not produce output csv: {output_csv}")

        scored = pd.read_csv(output_csv)
        score_col = _pick_score_column(scored)
        if "__rep_uid" not in scored.columns:
            scored["__rep_uid"] = payload["__rep_uid"]

        out = scored[["__rep_uid", score_col]].copy()
        out = out.rename(columns={score_col: args.score_field})
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in out.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

