#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretty-print step-wise CoT outputs from .steps.jsonl")
    p.add_argument("--step_jsonl", required=True, help="Path to step-level JSONL")
    p.add_argument("--sample_id", default="", help="Optional sample_id filter")
    p.add_argument("--max_rows", type=int, default=20, help="Maximum rows to print")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    step_path = Path(args.step_jsonl)
    shown = 0
    for line in step_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if args.sample_id and str(row.get("sample_id")) != args.sample_id:
            continue
        step = row.get("step")
        sample_id = row.get("sample_id")
        name = row.get("name", "")
        if step == "error":
            print(f"[sample_id={sample_id}] ERROR: {row.get('error')}")
        else:
            result = row.get("result", {})
            short = json.dumps(result, ensure_ascii=False)
            if len(short) > 240:
                short = short[:240] + "...(truncated)"
            print(f"[sample_id={sample_id}] step={step} {name} => {short}")
        shown += 1
        if shown >= args.max_rows:
            break
    print(f"shown rows: {shown}")


if __name__ == "__main__":
    main()
