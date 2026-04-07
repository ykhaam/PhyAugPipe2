#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage D: physics-aware resampling with category difficulty budgeting")
    p.add_argument("--input_jsonl", required=True, help="Action-clustered JSONL")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_csv", default="")
    p.add_argument("--budget", type=int, required=True, help="Final number of samples to keep")
    p.add_argument("--difficulty_field", default="videocon_physics_score", help="Field for category difficulty estimation")
    p.add_argument("--fallback_difficulty", choices=["inverse_physics_richness", "uniform"], default="inverse_physics_richness")
    p.add_argument("--representative_topk", type=int, default=20, help="Top-k per category by action_match_score for representative pool")
    p.add_argument("--min_per_category", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_rows(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("Input JSONL is empty")
    df = pd.DataFrame(rows)
    if "action_category" not in df.columns:
        raise ValueError("action_category field is required")
    if "action_match_score" not in df.columns:
        df["action_match_score"] = 0.0
    df["action_match_score"] = pd.to_numeric(df["action_match_score"], errors="coerce").fillna(0.0)
    if "physics_richness" in df.columns:
        df["physics_richness"] = pd.to_numeric(df["physics_richness"], errors="coerce").fillna(0.0)
    return df


def _estimate_difficulty(rep_df: pd.DataFrame, difficulty_field: str, fallback: str) -> float:
    if difficulty_field in rep_df.columns:
        vals = pd.to_numeric(rep_df[difficulty_field], errors="coerce").dropna()
        if not vals.empty:
            return float(np.clip(vals.mean(), 1e-6, None))

    if fallback == "inverse_physics_richness" and "physics_richness" in rep_df.columns:
        inv = 1.0 - pd.to_numeric(rep_df["physics_richness"], errors="coerce").fillna(0.0)
        return float(np.clip(inv.mean(), 1e-6, None))

    return 1.0


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")

    df = _load_rows(args.input_jsonl)
    if args.budget > len(df):
        raise ValueError(f"budget({args.budget}) > input_count({len(df)})")

    grouped = {}
    difficulties = {}
    for cat, g in df.groupby("action_category"):
        ranked = g.sort_values("action_match_score", ascending=False)
        reps = ranked.head(max(1, args.representative_topk))
        grouped[cat] = ranked
        difficulties[cat] = _estimate_difficulty(reps, args.difficulty_field, args.fallback_difficulty)

    cats = sorted(grouped.keys())
    n_cat = len(cats)
    base_alloc = {cat: min(args.min_per_category, len(grouped[cat])) for cat in cats}
    allocated = sum(base_alloc.values())
    if allocated > args.budget:
        raise ValueError("min_per_category allocation exceeds budget")

    remaining = args.budget - allocated
    weights = np.array([difficulties[c] for c in cats], dtype=float)
    weights = np.clip(weights, 1e-9, None)
    weights = weights / weights.sum()
    raw_extra = weights * remaining
    extra = np.floor(raw_extra).astype(int)

    leftover = remaining - int(extra.sum())
    if leftover > 0:
        frac_order = np.argsort(-(raw_extra - extra))
        for idx in frac_order[:leftover]:
            extra[idx] += 1

    target_alloc = {cat: base_alloc[cat] + int(extra[i]) for i, cat in enumerate(cats)}

    # Cap by available counts; redistribute deficit greedily
    deficit = 0
    for cat in cats:
        cap = len(grouped[cat])
        if target_alloc[cat] > cap:
            deficit += target_alloc[cat] - cap
            target_alloc[cat] = cap

    if deficit > 0:
        for cat in sorted(cats, key=lambda c: difficulties[c], reverse=True):
            room = len(grouped[cat]) - target_alloc[cat]
            if room <= 0:
                continue
            add = min(room, deficit)
            target_alloc[cat] += add
            deficit -= add
            if deficit == 0:
                break

    selected_parts = []
    for cat in cats:
        k = target_alloc[cat]
        if k <= 0:
            continue
        selected_parts.append(grouped[cat].head(k))

    selected = pd.concat(selected_parts, axis=0).copy()
    selected = selected.sort_values(["action_category", "action_match_score"], ascending=[True, False])

    out_rows = selected.to_dict(orient="records")
    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(out_csv, index=False)

    print(json.dumps({
        "input_count": int(len(df)),
        "output_count": int(len(selected)),
        "num_categories": n_cat,
        "allocations": target_alloc,
        "difficulty_field": args.difficulty_field,
        "fallback": args.fallback_difficulty,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
