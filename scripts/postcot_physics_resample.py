#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import shlex

import numpy as np
import pandas as pd

VIDEOPHY2_GITHUB_URL = "https://github.com/Hritikbansal/videophy/tree/main/VIDEOPHY2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage D: physics-aware resampling with category difficulty budgeting")
    p.add_argument("--input_jsonl", required=True, help="Action-clustered JSONL")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_csv", default="")
    p.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Final number of samples to keep (alias: --N).",
    )
    p.add_argument(
        "--N",
        dest="sampling_budget_n",
        type=int,
        default=None,
        help="Total sampling budget N from the paper. If set, this value is used as the final sample count.",
    )
    p.add_argument("--difficulty_field", default="videocon_physics_score", help="Field for category difficulty estimation")
    p.add_argument("--fallback_difficulty", choices=["inverse_physics_richness", "uniform"], default="inverse_physics_richness")
    p.add_argument("--difficulty_config", default="configs/action_difficulty.yaml", help="Legacy option (ignored in paper-formula allocation)")
    p.add_argument("--representative_topk", type=int, default=20, help="Top-k per category by action_match_score for representative pool")
    p.add_argument("--min_per_category", type=int, default=1)
    p.add_argument("--min_count", type=int, default=1, help="Minimum category count to be included in normal allocation")
    p.add_argument("--ambiguity_threshold", type=float, default=0.05, help="Margin threshold for ambiguity ratio, margin < threshold")
    p.add_argument("--low_priority_mode", choices=["exclude", "bucket"], default="exclude", help="How to handle categories below min_count")
    p.add_argument("--difficulty_weights", default="failure=0.5,prior=0.3,ambiguity=0.2", help="Legacy option (ignored in paper-formula allocation)")
    p.add_argument("--tau", type=float, default=1.0, help="Temperature τ for paper allocation: r_k = exp(τ * d_k)")
    p.add_argument(
        "--videophy2_eval_command",
        default="",
        help=(
            "Optional shell command template to score representative(top-nc) samples with VideoPhy2. "
            "Use placeholders {input_jsonl} and {output_jsonl}. "
            "Input contains __rep_uid and action_category; output must contain __rep_uid and difficulty_field score."
        ),
    )
    p.add_argument(
        "--videophy2_repo",
        default=VIDEOPHY2_GITHUB_URL,
        help="Reference VideoPhy2 repository URL (used for provenance in output summary).",
    )
    p.add_argument(
        "--videophy2_mode",
        choices=["auto", "command", "off"],
        default="auto",
        help="VideoPhy2 representative scoring mode. 'auto' runs the official VideoPhy2 inference bridge by default.",
    )
    p.add_argument("--videophy2_root", default="VIDEOPHY2", help="Local path to cloned VideoPhy2 repo")
    p.add_argument("--videophy2_checkpoint", default="", help="Path to VideoPhy2 checkpoint directory")
    p.add_argument("--video_path_field", default="video_path", help="Input column name for video path")
    p.add_argument("--caption_field", default="original_prompt", help="Input column name for caption/prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input_hist_json", default="", help="Optional Stage C histogram JSON (H_f) to validate category counts")
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


def _validate_input_hist_json(df: pd.DataFrame, hist_path: str) -> None:
    raw = json.loads(Path(hist_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("input_hist_json must be a JSON object")

    counts_raw = raw.get("counts")
    if not isinstance(counts_raw, dict):
        raise ValueError("input_hist_json must include object field 'counts'")

    expected_counts: dict[str, int] = {}
    for cat, count in counts_raw.items():
        try:
            expected_counts[str(cat)] = int(count)
        except (TypeError, ValueError):
            raise ValueError(f"invalid count in input_hist_json for category={cat!r}: {count!r}")

    actual_counts_series = df.groupby("action_category").size()
    actual_counts = {str(cat): int(count) for cat, count in actual_counts_series.items()}

    missing_categories = sorted([cat for cat in expected_counts if cat not in actual_counts])
    unexpected_categories = sorted([cat for cat in actual_counts if cat not in expected_counts])
    mismatched_counts = {
        cat: {"expected": expected_counts[cat], "actual": actual_counts[cat]}
        for cat in sorted(set(expected_counts) & set(actual_counts))
        if expected_counts[cat] != actual_counts[cat]
    }

    if missing_categories or unexpected_categories or mismatched_counts:
        raise ValueError(
            "input_hist_json validation failed: "
            f"missing_categories={missing_categories}, "
            f"unexpected_categories={unexpected_categories}, "
            f"count_mismatches={mismatched_counts}"
        )

    total_count = raw.get("total_count")
    if total_count is not None and int(total_count) != int(len(df)):
        raise ValueError(
            f"input_hist_json total_count mismatch: expected={int(total_count)} actual={int(len(df))}"
        )

def _estimate_difficulty(rep_df: pd.DataFrame, difficulty_field: str, fallback: str) -> float:
    if difficulty_field in rep_df.columns:
        vals = pd.to_numeric(rep_df[difficulty_field], errors="coerce").dropna()
        if not vals.empty:
            return float(np.clip(vals.mean(), 1e-6, None))

    if fallback == "inverse_physics_richness" and "physics_richness" in rep_df.columns:
        inv = 1.0 - pd.to_numeric(rep_df["physics_richness"], errors="coerce").fillna(0.0)
        return float(np.clip(inv.mean(), 1e-6, None))

    return 1.0


def _mean_success(rep_df: pd.DataFrame, difficulty_field: str, fallback: str) -> float:
    if difficulty_field in rep_df.columns:
        vals = pd.to_numeric(rep_df[difficulty_field], errors="coerce").dropna()
        if not vals.empty:
            return float(np.clip(vals.mean(), 0.0, 1.0))
    if fallback == "inverse_physics_richness" and "physics_richness" in rep_df.columns:
        vals = pd.to_numeric(rep_df["physics_richness"], errors="coerce").dropna()
        if not vals.empty:
            return float(np.clip(vals.mean(), 0.0, 1.0))
    return 0.5

def _allocate_with_paper_formula(
    counts: dict[str, int],
    difficulties: dict[str, float],
    budget: int,
    tau: float,
) -> tuple[dict[str, int], dict[str, float]]:
    cats = sorted(counts.keys())
    d = np.array([float(np.clip(difficulties[c], 0.0, 1.0)) for c in cats], dtype=float)
    r = np.exp(float(tau) * d)
    raw = budget * r / np.clip(r.sum(), 1e-12, None)
    caps = np.array([counts[c] for c in cats], dtype=float)
    capped = np.minimum(caps, raw)
    alloc = np.floor(capped).astype(int)

    remainder = budget - int(alloc.sum())
    while remainder > 0:
        room = caps - alloc
        if np.all(room <= 0):
            break
        frac = capped - alloc
        frac[room <= 0] = -1.0
        idx = int(np.argmax(frac))
        if frac[idx] < 0:
            break
        alloc[idx] += 1
        remainder -= 1

    out_alloc = {c: int(alloc[i]) for i, c in enumerate(cats)}
    diag = {
        "tau": float(tau),
        "sum_r": float(r.sum()),
        "raw_quota_sum": float(raw.sum()),
        "capped_quota_sum": float(capped.sum()),
        "final_alloc_sum": int(alloc.sum()),
    }
    return out_alloc, diag

def _run_videophy2_on_representatives(
    reps_by_cat: dict[str, pd.DataFrame],
    difficulty_field: str,
    eval_command_template: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    if not eval_command_template:
        return reps_by_cat, {"num_representatives_scored": 0, "num_representatives_missing_score": 0}

    flat_parts: list[pd.DataFrame] = []
    for cat, reps in reps_by_cat.items():
        tmp = reps.copy()
        tmp["action_category"] = cat
        tmp["__rep_uid"] = [f"{cat}::{i}" for i in range(len(tmp))]
        flat_parts.append(tmp)

    if not flat_parts:
        return reps_by_cat, {"num_representatives_scored": 0, "num_representatives_missing_score": 0}

    reps_flat = pd.concat(flat_parts, axis=0).copy()

    with tempfile.TemporaryDirectory(prefix="videophy2_eval_") as tmp_dir:
        input_path = Path(tmp_dir) / "representatives_input.jsonl"
        output_path = Path(tmp_dir) / "representatives_scored.jsonl"
        with input_path.open("w", encoding="utf-8") as f:
            for row in reps_flat.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        command = eval_command_template.format(
            input_jsonl=str(input_path),
            output_jsonl=str(output_path),
        )
        proc = subprocess.run(command, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "videophy2_eval_command failed with non-zero exit code "
                f"{proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_path.exists():
            raise FileNotFoundError(
                "videophy2_eval_command did not produce output_jsonl file at "
                f"{output_path}"
            )

        scored_rows = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    scored_rows.append(json.loads(line))
        if not scored_rows:
            raise ValueError("videophy2_eval_command output_jsonl is empty")
        scored_df = pd.DataFrame(scored_rows)

    if "__rep_uid" not in scored_df.columns:
        raise ValueError("videophy2_eval_command output must include '__rep_uid'")
    if difficulty_field not in scored_df.columns:
        raise ValueError(
            "videophy2_eval_command output must include difficulty field "
            f"'{difficulty_field}'"
        )

    score_map: dict[str, float] = {}
    score_vals = pd.to_numeric(scored_df[difficulty_field], errors="coerce")
    for uid, val in zip(scored_df["__rep_uid"].astype(str), score_vals):
        if pd.notna(val):
            score_map[uid] = float(val)

    updated: dict[str, pd.DataFrame] = {}
    scored = 0
    missing = 0
    for cat, reps in reps_by_cat.items():
        out = reps.copy()
        uids = [f"{cat}::{i}" for i in range(len(out))]
        vals = []
        for uid in uids:
            v = score_map.get(uid, np.nan)
            if np.isnan(v):
                missing += 1
            else:
                scored += 1
            vals.append(v)
        out[difficulty_field] = vals
        updated[cat] = out

    return updated, {
        "num_representatives_scored": int(scored),
        "num_representatives_missing_score": int(missing),
    }

def _resolve_videophy2_eval_command(args: argparse.Namespace) -> str:
    if args.videophy2_mode == "off":
        return ""
    if args.videophy2_mode == "command":
        if not args.videophy2_eval_command:
            raise ValueError("--videophy2_mode command requires --videophy2_eval_command")
        return args.videophy2_eval_command
    # auto mode
    if args.videophy2_eval_command:
        return args.videophy2_eval_command
    if not args.videophy2_checkpoint:
        raise ValueError("--videophy2_mode auto requires --videophy2_checkpoint")
    bridge_script = Path(__file__).with_name("videophy2_eval_bridge.py")
    return (
        f"python {shlex.quote(str(bridge_script))} "
        f"--input_jsonl {{input_jsonl}} --output_jsonl {{output_jsonl}} "
        f"--videophy2_root {shlex.quote(args.videophy2_root)} "
        f"--checkpoint {shlex.quote(args.videophy2_checkpoint)} "
        f"--score_field {shlex.quote(args.difficulty_field)} "
        f"--video_path_field {shlex.quote(args.video_path_field)} "
        f"--caption_field {shlex.quote(args.caption_field)}"
    )


def main() -> None:
    args = parse_args()
    budget = args.sampling_budget_n if args.sampling_budget_n is not None else args.budget
    if budget is None:
        raise ValueError("Either --budget or --N must be provided")
    if budget <= 0:
        raise ValueError("Sampling budget must be > 0 (--budget/--N)")

    df = _load_rows(args.input_jsonl)
    if args.input_hist_json:
        _validate_input_hist_json(df, args.input_hist_json)
    if budget > len(df):
        raise ValueError(f"budget({budget}) > input_count({len(df)})")
    if args.min_count <= 0:
        raise ValueError("--min_count must be > 0")

    grouped = {}
    reps_by_cat: dict[str, pd.DataFrame] = {}
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
        reps_by_cat[cat] = reps

    resolved_eval_command = _resolve_videophy2_eval_command(args)
    reps_by_cat, videophy2_eval_stats = _run_videophy2_on_representatives(
        reps_by_cat=reps_by_cat,
        difficulty_field=args.difficulty_field,
        eval_command_template=resolved_eval_command,
    )

    for cat in sorted(grouped.keys()):
        ranked = grouped[cat]
        reps = reps_by_cat[cat]
        s_f = _mean_success(reps, args.difficulty_field, args.fallback_difficulty)
        d_k = float(np.clip(1.0 - s_f, 0.0, 1.0))
        difficulties[cat] = d_k
        components[cat] = {
            "S_f": s_f,
            "d_k": d_k,
            "count": int(len(ranked)),
        }

    cats = sorted(grouped.keys())
    n_cat = len(cats)
    if n_cat == 0:
        raise ValueError("No categories available for allocation after min_count/low_priority_mode filtering")
    h_f = {cat: int(len(grouped[cat])) for cat in cats}
    target_alloc, alloc_diag = _allocate_with_paper_formula(
        counts=h_f,
        difficulties=difficulties,
        budget=budget,
        tau=args.tau,
    )

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
        "sampling_budget_n": int(budget),
        "num_categories": n_cat,
        "allocations": target_alloc,
        "difficulty_field": args.difficulty_field,
        "fallback": args.fallback_difficulty,
        "difficulty_config": args.difficulty_config,
        "ambiguity_threshold": args.ambiguity_threshold,
        "min_count": args.min_count,
        "low_priority_mode": args.low_priority_mode,
        "low_priority_categories": sorted([str(c) for c in low_priority_cats]),
        "tau": args.tau,
        "difficulty_components": components,
        "paper_allocation_diagnostics": alloc_diag,
        "videophy2_mode": args.videophy2_mode,
        "videophy2_eval_command_used": bool(resolved_eval_command),
        "videophy2_eval_stats": videophy2_eval_stats,
        "videophy2_repo": args.videophy2_repo,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
