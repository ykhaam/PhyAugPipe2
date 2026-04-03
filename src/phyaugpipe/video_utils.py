from __future__ import annotations

from pathlib import Path
from typing import List

import cv2


def sample_video_frames(video_path: str, num_frames: int = 8) -> List["cv2.typing.MatLike"]:
    """Uniformly sample frames from a local video path."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")

    target_indices = sorted({int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)})
    frames = []
    current_target = 0
    frame_idx = 0

    while current_target < len(target_indices):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx == target_indices[current_target]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            current_target += 1
        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"Frame sampling failed: {video_path}")

    return frames
