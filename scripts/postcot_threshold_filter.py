#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage B: threshold/quantile filtering on scored JSONL (typically step4_score.jsonl)")
    p.add_argument("--input_jsonl", required=True, help="Scored JSONL with physics_richness (recommended: step4_score.jsonl)")
    p.add_argument("--output_jsonl", required=True, help="Filtered JSONL")
    p.add_argument("--output_csv", default="", help="Optional CSV output")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--threshold", type=float, help="Keep rows where physics_richness >= threshold")
    mode.add_argument("--top_quantile", type=float, help="Keep top q fraction by physics_richness (e.g., 0.15)")
    p.add_argument("--drop_errors", action="store_true", help="Drop rows that contain error field")
    return p.parse_args()


def _load_jsonl(path: str, drop_errors: bool) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if drop_errors and row.get("error"):
                continue
            if "physics_richness" not in row:
                continue
            rows.append(row)
    if not rows:
        raise ValueError("No valid rows with physics_richness were found")
    df = pd.DataFrame(rows)
    df["physics_richness"] = pd.to_numeric(df["physics_richness"], errors="coerce").fillna(0.0)
    return df


def main() -> None:
    args = parse_args()
    df = _load_jsonl(args.input_jsonl, drop_errors=args.drop_errors)

    if args.threshold is not None:
        filtered = df[df["physics_richness"] >= args.threshold].copy()
        policy = {"mode": "threshold", "threshold": float(args.threshold)}
    else:
        if not (0.0 < args.top_quantile <= 1.0):
            raise ValueError("--top_quantile must be in (0, 1]")
        k = max(1, int(round(len(df) * args.top_quantile)))
        filtered = df.nlargest(k, "physics_richness").copy()
        policy = {"mode": "top_quantile", "top_quantile": float(args.top_quantile), "selected_count": int(k)}

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in filtered.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(out_csv, index=False)

    print(json.dumps({
        "input_count": int(len(df)),
        "output_count": int(len(filtered)),
        "retention_ratio": float(len(filtered) / len(df)),
        "policy": policy,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
