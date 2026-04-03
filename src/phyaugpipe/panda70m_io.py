from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_MOTION_KEYWORDS = [
    "run",
    "running",
    "walk",
    "walking",
    "sport",
    "sports",
    "soccer",
    "basketball",
    "tennis",
    "baseball",
    "football",
    "volleyball",
    "skate",
    "ski",
]


def parse_keywords(keywords: str) -> list[str]:
    if not keywords.strip():
        return []
    return [k.strip().lower() for k in keywords.split(",") if k.strip()]


def keyword_filter(df: pd.DataFrame, text_col: str, keywords: Iterable[str]) -> pd.DataFrame:
    keys = [k.lower() for k in keywords if k]
    if not keys:
        return df
    escaped = [re.escape(k) for k in keys]
    pattern = "|".join(rf"\\b{x}\\b" for x in escaped)
    mask = df[text_col].fillna("").astype(str).str.lower().str.contains(pattern, regex=True)
    return df[mask].copy()


def make_local_pairs(
    df: pd.DataFrame,
    *,
    id_col: str,
    prompt_col: str,
    video_col: Optional[str],
    video_root: Optional[str],
    video_ext: str,
) -> pd.DataFrame:
    """
    Build a pair table with columns: sample_id, original_prompt, video_path.

    - If video_col is given, it is treated as a local file path column.
    - Else, video_root + sample_id + video_ext is used.
    """
    out = pd.DataFrame()
    out["sample_id"] = df[id_col].astype(str)
    out["original_prompt"] = df[prompt_col].astype(str)

    if video_col:
        out["video_path"] = df[video_col].astype(str)
    else:
        if not video_root:
            raise ValueError("video_root is required when video_col is not provided")
        root = Path(video_root)
        out["video_path"] = out["sample_id"].apply(lambda x: str(root / f"{x}{video_ext}"))

    exists_mask = out["video_path"].apply(lambda p: Path(p).exists())
    return out[exists_mask].copy()


def make_wisa80k_pairs(
    df: pd.DataFrame,
    *,
    prompt_col: str = "captions",
    video_name_col: str = "video_name",
    video_root: str,
    require_local_video: bool = True,
) -> pd.DataFrame:
    """Build pair table for WISA-80K metadata + local videos directory."""
    root = Path(video_root)
    out = pd.DataFrame()
    out["sample_id"] = df[video_name_col].astype(str)
    out["original_prompt"] = df[prompt_col].astype(str)
    out["video_path"] = out["sample_id"].apply(lambda x: str(root / x))

    if require_local_video:
        exists_mask = out["video_path"].apply(lambda p: Path(p).exists())
        out = out[exists_mask].copy()
    return out


def sample_rows(df: pd.DataFrame, max_count: int, mode: str, seed: int) -> pd.DataFrame:
    if max_count <= 0 or len(df) <= max_count:
        return df
    if mode == "top":
        return df.head(max_count).copy()
    if mode == "random":
        return df.sample(n=max_count, random_state=seed).copy()
    raise ValueError(f"Unsupported mode: {mode}")
