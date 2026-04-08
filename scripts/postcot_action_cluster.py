#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_ACTION_CATEGORIES = [
    "falling", "throwing", "bouncing", "collision", "sliding", "rolling", "pouring",
    "breaking", "explosion", "burning", "floating", "sinking", "lifting", "pushing",
    "pulling", "spinning", "stretching", "deformation", "fluid flow", "projectile motion",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage C: action clustering via sentence-transformer matching")
    p.add_argument("--input_jsonl", required=True, help="Filtered JSONL from Stage B")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_csv", default="")
    p.add_argument("--categories_file", default="", help="Optional newline-separated action categories")
    p.add_argument("--prompt_field", default="original_prompt", choices=["original_prompt", "extended"], help="Text field used for matching")
    p.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--low_margin_threshold", type=float, default=0.05, help="Threshold used for low_margin_ratio (margin < threshold)")
    p.add_argument("--output_stats_json", default="", help="Optional path to save category-level stats JSON")
    return p.parse_args()


def _load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("Input JSONL is empty")
    return rows


def _load_categories(path: str) -> list[str]:
    if not path:
        return DEFAULT_ACTION_CATEGORIES
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    cats = [ln for ln in lines if ln and not ln.startswith("#")]
    if not cats:
        raise ValueError("No action categories found in categories_file")
    return cats


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def _iter_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.input_jsonl)
    categories = _load_categories(args.categories_file)

    prompts = [str(row.get(args.prompt_field, "")) for row in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)

    category_emb = model.encode(categories, normalize_embeddings=True)
    prompt_emb_chunks = []
    for batch in _iter_batches(prompts, args.batch_size):
        prompt_emb_chunks.append(model.encode(batch, normalize_embeddings=True))
    prompt_emb = np.concatenate(prompt_emb_chunks, axis=0)

    sim = _cosine_sim_matrix(prompt_emb, category_emb)
    sorted_idx = np.argsort(-sim, axis=1)
    best_idx = sorted_idx[:, 0]
    best_score = sim[np.arange(len(rows)), best_idx]
    if len(categories) >= 2:
        second_idx = sorted_idx[:, 1]
        second_score = sim[np.arange(len(rows)), second_idx]
    else:
        second_score = np.zeros(len(rows), dtype=float)

    for i, row in enumerate(rows):
        row["action_category"] = categories[int(best_idx[i])]
        row["action_match_score"] = float(best_score[i])
        row["top1_score"] = float(best_score[i])
        row["top2_score"] = float(second_score[i])
        row["margin"] = float(best_score[i] - second_score[i])

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    grouped = df.groupby("action_category", dropna=False)
    stats = {}
    for cat, g in grouped:
        margins = pd.to_numeric(g["margin"], errors="coerce").fillna(0.0)
        stats[str(cat)] = {
            "count": int(len(g)),
            "mean_margin": float(margins.mean()),
            "low_margin_ratio": float((margins < args.low_margin_threshold).mean()),
        }

    if args.output_stats_json:
        out_stats = Path(args.output_stats_json)
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "num_samples": len(rows),
        "num_categories": len(categories),
        "low_margin_threshold": args.low_margin_threshold,
        "category_stats": stats,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
