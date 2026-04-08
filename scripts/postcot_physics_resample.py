#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage D: physics-aware resampling with category difficulty budgeting")
    p.add_argument("--input_jsonl", required=True, help="Action-clustered JSONL")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_csv", default="")
    p.add_argument("--budget", type=int, required=True, help="Final number of samples to keep")
    p.add_argument("--difficulty_field", default="videocon_physics_score", help="Field for category difficulty estimation")
    p.add_argument("--fallback_difficulty", choices=["inverse_physics_richness", "uniform"], default="inverse_physics_richness")
    p.add_argument("--difficulty_config", default="configs/action_difficulty.yaml", help="YAML file containing per-category prior difficulty")
    p.add_argument("--representative_topk", type=int, default=20, help="Top-k per category by action_match_score for representative pool")
    p.add_argument("--min_per_category", type=int, default=1)
    p.add_argument("--min_count", type=int, default=1, help="Minimum category count to be included in normal allocation")
    p.add_argument("--ambiguity_threshold", type=float, default=0.05, help="Margin threshold for ambiguity ratio, margin < threshold")
    p.add_argument("--low_priority_mode", choices=["exclude", "bucket"], default="exclude", help="How to handle categories below min_count")
    p.add_argument("--difficulty_weights", default="failure=0.5,prior=0.3,ambiguity=0.2", help="Comma-separated weights for combined difficulty")
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


def _load_prior_difficulty(path: str) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"difficulty_config must be a mapping, got: {type(raw)}")
    priors = raw.get("prior_difficulty", raw)
    if not isinstance(priors, dict):
        raise ValueError("prior_difficulty must be a mapping")
    out: dict[str, float] = {}
    for k, v in priors.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _parse_weights(text: str) -> dict[str, float]:
    out = {"failure": 0.5, "prior": 0.3, "ambiguity": 0.2}
    if not text:
        return out
    for token in text.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        if k in out:
            out[k] = float(v.strip())
    total = sum(out.values())
    if total <= 0:
        raise ValueError("difficulty_weights must sum to > 0")
    return {k: v / total for k, v in out.items()}


def _mean_failure(rep_df: pd.DataFrame, difficulty_field: str, fallback: str) -> float:
    if difficulty_field in rep_df.columns:
        vals = pd.to_numeric(rep_df[difficulty_field], errors="coerce").dropna()
        if not vals.empty:
            return float(np.clip(1.0 - vals.mean(), 0.0, 1.0))
    fallback_diff = _estimate_difficulty(rep_df, difficulty_field, fallback)
    return float(np.clip(fallback_diff, 0.0, 1.0))


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")

    df = _load_rows(args.input_jsonl)
    if args.budget > len(df):
        raise ValueError(f"budget({args.budget}) > input_count({len(df)})")
    if args.min_count <= 0:
        raise ValueError("--min_count must be > 0")

    priors = _load_prior_difficulty(args.difficulty_config)
    weights_cfg = _parse_weights(args.difficulty_weights)
    grouped = {}
    difficulties = {}
    components = {}
    low_priority_cats: list[str] = []
    for cat, g in df.groupby("action_category"):
        ranked = g.sort_values("action_match_score", ascending=False)
        reps = ranked.head(max(1, args.representative_topk))
        if len(ranked) < args.min_count:
            low_priority_cats.append(cat)
            if args.low_priority_mode == "exclude":
                continue
        grouped[cat] = ranked
        failure = _mean_failure(reps, args.difficulty_field, args.fallback_difficulty)
        prior = float(np.clip(priors.get(str(cat), 0.5), 0.0, 1.0))
        margins = pd.to_numeric(ranked.get("margin", pd.Series([], dtype=float)), errors="coerce").dropna()
        ambiguity = float((margins < args.ambiguity_threshold).mean()) if not margins.empty else 0.0
        combined = (
            weights_cfg["failure"] * failure
            + weights_cfg["prior"] * prior
            + weights_cfg["ambiguity"] * ambiguity
        )
        difficulties[cat] = float(np.clip(combined, 1e-6, None))
        components[cat] = {
            "failure": failure,
            "prior_difficulty": prior,
            "ambiguity": ambiguity,
            "combined": difficulties[cat],
            "count": int(len(ranked)),
        }

    cats = sorted(grouped.keys())
    n_cat = len(cats)
    if n_cat == 0:
        raise ValueError("No categories available for allocation after min_count/low_priority_mode filtering")
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
        "difficulty_config": args.difficulty_config,
        "ambiguity_threshold": args.ambiguity_threshold,
        "min_count": args.min_count,
        "low_priority_mode": args.low_priority_mode,
        "low_priority_categories": sorted([str(c) for c in low_priority_cats]),
        "difficulty_weights": weights_cfg,
        "difficulty_components": components,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
