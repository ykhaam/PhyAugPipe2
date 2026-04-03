#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from phyaugpipe.panda70m_io import (
    DEFAULT_MOTION_KEYWORDS,
    keyword_filter,
    make_wisa80k_pairs,
    parse_keywords,
    sample_rows,
)


def _pick_existing_column(candidates: list[str], cols: list[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"None of candidate columns exist: {candidates}. available={cols}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build WISA-80K subset CSV with prompt+video pairing")
    p.add_argument("--dataset_name", default="qihoo360/WISA-80K")
    p.add_argument("--split", default="train")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--video_root", required=True, help="Local directory where WISA video files are stored")
    p.add_argument("--prompt_col", default="", help="If empty, auto-detect from common text columns")
    p.add_argument("--video_name_col", default="", help="If empty, auto-detect from common video-name columns")
    p.add_argument("--keywords", default=",".join(DEFAULT_MOTION_KEYWORDS))
    p.add_argument("--no_keyword_filter", action="store_true", default=False)
    p.add_argument("--max_count", type=int, default=1000)
    p.add_argument("--sample_mode", choices=["top", "random"], default="top")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require_local_video", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset_name, split=args.split)
    df = ds.to_pandas()

    cols = list(df.columns)
    prompt_col = args.prompt_col or _pick_existing_column(
        ["captions", "caption", "text", "prompt", "description"], cols
    )
    video_name_col = args.video_name_col or _pick_existing_column(
        ["video_name", "video", "filename", "file_name", "path"], cols
    )

    if args.no_keyword_filter:
        filtered = df
    else:
        kw = parse_keywords(args.keywords)
        filtered = keyword_filter(df, prompt_col, kw)

    paired = make_wisa80k_pairs(
        filtered,
        prompt_col=prompt_col,
        video_name_col=video_name_col,
        video_root=args.video_root,
        require_local_video=args.require_local_video,
    )

    subset = sample_rows(paired, max_count=args.max_count, mode=args.sample_mode, seed=args.seed)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)

    print(f"dataset_rows={len(df)}")
    print(f"prompt_col={prompt_col} video_name_col={video_name_col}")
    print(f"after_filter={len(filtered)}")
    print(f"paired_rows={len(paired)}")
    print(f"subset_rows={len(subset)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
