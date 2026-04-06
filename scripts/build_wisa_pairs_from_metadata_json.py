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
from typing import Any

import pandas as pd

from phyaugpipe.panda70m_io import sample_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical pair CSV from WISA-80K metadata.json")
    p.add_argument("--metadata_json", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--prompt_key", default="captions")
    p.add_argument("--allow_non_caption_prompt", action="store_true", default=False)
    p.add_argument("--video_name_key", default="video_name")
    p.add_argument("--dedup_by_video_name", action="store_true", default=False)
    p.add_argument("--require_local_video", action="store_true", default=False)
    p.add_argument("--include_meta", action="store_true", default=False)
    p.add_argument("--max_count", type=int, default=0)
    p.add_argument("--sanitize_text_fields", action="store_true", default=False)
    p.add_argument("--drop_corrupted", action="store_true", default=False)
    p.add_argument("--sample_mode", choices=["top", "random"], default="top")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read_json_records(path: str) -> list[dict[str, Any]]:
    """
    Robust loader for normal JSON list and concatenated JSON values, e.g.:
      [ ... ], [ ... ]
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Fast path: normal single JSON value.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []
    except json.JSONDecodeError:
        pass

    # Fallback: parse sequential JSON values and merge.
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    out: list[dict[str, Any]] = []
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        obj, j = decoder.raw_decode(text, i)
        if isinstance(obj, list):
            out.extend([x for x in obj if isinstance(x, dict)])
        elif isinstance(obj, dict):
            out.append(obj)
        i = j
    return out


def _flatten_row(row: dict[str, Any], include_meta: bool) -> dict[str, Any]:
    base = {}
    if include_meta:
        for k in [
            "width",
            "height",
            "fps",
            "duration",
            "motion_score",
            "motion_score_v2",
            "visual_quality_score",
            "text_bbox_num",
            "text_bbox_ratio",
            "label",
        ]:
            if k in row:
                base[k] = row[k]

        pa = row.get("physical_annotation") or {}
        if isinstance(pa, dict):
            for k in ["phys_law", "n0", "n1", "n2", "q0", "q1", "q2", "q3", "q4"]:
                if k in pa:
                    base[f"physical_{k}"] = pa[k]

            # Keep quantified fields (as JSON strings) and useful derived numeric ranges.
            for qk in ["quantify_n0", "quantify_n1", "quantify_n2"]:
                if qk in pa:
                    base[f"physical_{qk}"] = json.dumps(pa[qk], ensure_ascii=False)

            qn1 = pa.get("quantify_n1")
            if isinstance(qn1, list) and len(qn1) == 2:
                base["physical_time_min_s"] = qn1[0]
                base["physical_time_max_s"] = qn1[1]

            qn2 = pa.get("quantify_n2")
            if isinstance(qn2, list) and len(qn2) == 2:
                base["physical_temp_min_c"] = qn2[0]
                base["physical_temp_max_c"] = qn2[1]

            qn0 = pa.get("quantify_n0")
            if isinstance(qn0, list):
                base["physical_density_range_count"] = len(qn0)

            q3 = pa.get("q3")
            if isinstance(q3, str):
                q3_norm = q3.strip().lower()
                if q3_norm in {"yes", "y", "true", "1"}:
                    base["physical_q3_bool"] = 1
                elif q3_norm in {"no", "n", "false", "0"}:
                    base["physical_q3_bool"] = 0

    return base




def _normalize_text(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    return x.replace("\r\n", "\n").replace("\r", "\n").strip()


def _looks_corrupted(row: dict[str, Any]) -> bool:
    sentinel = "sample_id,original_prompt,video_path"
    for v in row.values():
        if isinstance(v, str) and sentinel in v:
            return True
    return False

def main() -> None:
    args = parse_args()
    if args.prompt_key != "captions" and not args.allow_non_caption_prompt:
        raise ValueError("Use captions as prompt_key by default. Pass --allow_non_caption_prompt to override.")
    rows = _read_json_records(args.metadata_json)
    if not rows:
        raise ValueError(f"No valid records found in metadata json: {args.metadata_json}")

    root = Path(args.video_root)
    out_rows = []

    for row in rows:
        if args.prompt_key not in row or args.video_name_key not in row:
            continue
        video_name = str(row[args.video_name_key])
        prompt = str(row[args.prompt_key])
        video_path = str(root / video_name)
        if args.require_local_video and not Path(video_path).exists():
            continue

        rec = {
            "sample_id": _normalize_text(video_name) if args.sanitize_text_fields else video_name,
            "original_prompt": _normalize_text(prompt) if args.sanitize_text_fields else prompt,
            "video_path": video_path,
        }
        rec.update(_flatten_row(row, include_meta=args.include_meta))

        if args.sanitize_text_fields:
            for k, v in list(rec.items()):
                rec[k] = _normalize_text(v)

        if args.drop_corrupted and _looks_corrupted(rec):
            continue

        out_rows.append(rec)

    df = pd.DataFrame(out_rows)
    if args.dedup_by_video_name and not df.empty:
        df = df.drop_duplicates(subset=["sample_id"], keep="first")

    df = sample_rows(df, max_count=args.max_count, mode=args.sample_mode, seed=args.seed)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"raw_records={len(rows)} usable_pairs={len(out_rows)} final_rows={len(df)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
