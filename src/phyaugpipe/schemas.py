from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParsedElements(BaseModel):
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    forces: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)


class CoTResult(BaseModel):
    original: str
    parse: ParsedElements
    reason: str
    extended: str
    physics_richness: float
    penalty_analysis: Dict[str, Any] = Field(default_factory=dict)
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    physics_label: Optional[int] = None


class SampleRecord(BaseModel):
    sample_id: str
    original_prompt: str
    video_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
