#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse

import pandas as pd

from phyaugpipe.panda70m_io import sample_rows


VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pair CSV from local video/text data without metadata filtering")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--manifest_csv", default="", help="Optional CSV containing prompt+video columns")
    p.add_argument("--prompt_col", default="original_prompt")
    p.add_argument("--video_col", default="video_path")
    p.add_argument("--sample_id_col", default="sample_id")
    p.add_argument("--video_root", default="", help="If no manifest_csv, scan this directory for videos")
    p.add_argument("--text_root", default="", help="If no manifest_csv, read txt prompts with same basename")
    p.add_argument("--max_count", type=int, default=200)
    p.add_argument("--sample_mode", choices=["top", "random"], default="top")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _from_manifest(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.manifest_csv)
    out = pd.DataFrame()
    out["sample_id"] = df[args.sample_id_col].astype(str) if args.sample_id_col in df.columns else df.index.astype(str)
    out["original_prompt"] = df[args.prompt_col].astype(str)
    out["video_path"] = df[args.video_col].astype(str)
    return out


def _from_local_dirs(args: argparse.Namespace) -> pd.DataFrame:
    if not args.video_root or not args.text_root:
        raise ValueError("When manifest_csv is not set, both --video_root and --text_root are required.")

    video_root = Path(args.video_root)
    text_root = Path(args.text_root)
    rows = []

    for vp in sorted(video_root.iterdir()):
        if not vp.is_file() or vp.suffix.lower() not in VIDEO_EXTS:
            continue
        sid = vp.stem
        tp = text_root / f"{sid}.txt"
        if not tp.exists():
            continue
        prompt = tp.read_text(encoding="utf-8", errors="ignore").strip()
        if not prompt:
            continue
        rows.append({"sample_id": sid, "original_prompt": prompt, "video_path": str(vp)})

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    pairs = _from_manifest(args) if args.manifest_csv else _from_local_dirs(args)

    pairs = pairs[pairs["video_path"].map(lambda p: Path(p).exists())].copy()
    subset = sample_rows(pairs, max_count=args.max_count, mode=args.sample_mode, seed=args.seed)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)

    print(f"pairs={len(pairs)} subset={len(subset)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
