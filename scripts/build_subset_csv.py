#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phyaugpipe.panda70m_io import keyword_filter, make_local_pairs, parse_keywords, sample_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paired subset CSV from Panda70M-like metadata")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--id_col", default="id")
    p.add_argument("--prompt_col", default="caption")
    p.add_argument("--video_col", default=None, help="Local video-path column. If omitted, video_root/id+ext used")
    p.add_argument("--video_root", default=None)
    p.add_argument("--video_ext", default=".mp4")
    p.add_argument("--keywords", default="run,walk,sport,sports,soccer,basketball,tennis,baseball")
    p.add_argument("--max_count", type=int, default=1000)
    p.add_argument("--sample_mode", choices=["top", "random"], default="top")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    kw = parse_keywords(args.keywords)
    filtered = keyword_filter(df, args.prompt_col, kw)

    paired = make_local_pairs(
        filtered,
        id_col=args.id_col,
        prompt_col=args.prompt_col,
        video_col=args.video_col,
        video_root=args.video_root,
        video_ext=args.video_ext,
    )
    subset = sample_rows(paired, max_count=args.max_count, mode=args.sample_mode, seed=args.seed)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)

    print(f"input={len(df)} keyword_filtered={len(filtered)} paired_exists={len(paired)} subset={len(subset)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
