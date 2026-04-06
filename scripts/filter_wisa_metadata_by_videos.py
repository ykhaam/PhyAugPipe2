#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keep only metadata entries whose video exists in a local folder (e.g., extracted 0.zip)."
    )
    p.add_argument("--metadata_json", required=True)
    p.add_argument("--video_folder", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--video_name_key", default="video_name")
    p.add_argument("--recursive", action="store_true", default=False)
    return p.parse_args()


def _read_json_records(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []
    except json.JSONDecodeError:
        pass

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


def _collect_video_ids(video_folder: str, recursive: bool) -> set[str]:
    root = Path(video_folder)
    if not root.exists():
        raise FileNotFoundError(f"video_folder not found: {video_folder}")

    paths = root.rglob("*") if recursive else root.iterdir()
    ids: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        ids.add(p.name)  # with extension
        ids.add(p.stem)  # without extension
    return ids


def main() -> None:
    args = parse_args()
    records = _read_json_records(args.metadata_json)
    if not records:
        raise ValueError(f"No valid records in metadata json: {args.metadata_json}")

    video_ids = _collect_video_ids(args.video_folder, recursive=args.recursive)

    kept = []
    dropped = 0
    for row in records:
        vname = str(row.get(args.video_name_key, "")).strip()
        if not vname:
            dropped += 1
            continue

        key1 = vname
        key2 = Path(vname).stem
        if key1 in video_ids or key2 in video_ids:
            kept.append(row)
        else:
            dropped += 1

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input_records={len(records)} kept={len(kept)} dropped={dropped}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
