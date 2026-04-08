from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import os
import sys
import types
from enum import Enum

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch

from .schemas import CoTResult, ParsedElements, SampleRecord
from .video_utils import sample_video_frames


@dataclass
class PipelineConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_new_tokens: int = 512
    num_frames: int = 8
    prompt_template_path: str = "prompts/cot_filtering_prompt.txt"
    device: str = "auto"  # auto | cpu | cuda




def _ensure_torchvision_stub() -> None:
    """Provide a minimal torchvision stub when installed torchvision is broken."""
    try:
        import torchvision  # noqa: F401
        return
    except Exception:
        pass

    if "torchvision" in sys.modules:
        return

    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")

    class InterpolationMode(Enum):
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 1
        HAMMING = 4
        BOX = 5

    transforms.InterpolationMode = InterpolationMode
    tv.transforms = transforms

    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms

def _process_vision_info(messages: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """
    Lightweight replacement for qwen_vl_utils.process_vision_info.
    Collects image payloads from chat messages and returns (images, videos).
    """
    images: list[Any] = []
    videos: list[Any] = []
    for msg in messages:
        for item in msg.get("content", []):
            if item.get("type") == "image" and "image" in item:
                images.append(item["image"])
            elif item.get("type") == "video" and "video" in item:
                videos.append(item["video"])
    return images, videos


class CoTFilteringPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        _ensure_torchvision_stub()

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as e:
            raise RuntimeError(
                "Failed to import transformers vision stack. "
                "Set TRANSFORMERS_NO_TORCHVISION=1 and ensure torch/torchvision are compatible, "
                "or reinstall requirements."
            ) from e

        self.processor = AutoProcessor.from_pretrained(self.config.model_name, trust_remote_code=True)

        if self.config.device == "cpu":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.model.to("cpu")
            self.input_device = torch.device("cpu")
        elif self.config.device == "cuda":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            self.model.to("cuda")
            self.input_device = torch.device("cuda")
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                device_map="auto",
            )
            self.input_device = self.model.device

        self.template = Path(self.config.prompt_template_path).read_text(encoding="utf-8")

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in model response: {text[:200]}")
        return json.loads(text[start : end + 1])

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = _process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        if self.config.device in {"cpu", "cuda"}:
            inputs = inputs.to(self.input_device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text

    def _messages_with_frames(
        self,
        instruction: str,
        sample: SampleRecord,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        frames = sample_video_frames(sample.video_path, num_frames=self.config.num_frames)
        content = [{"type": "text", "text": instruction}]
        content.append({"type": "text", "text": f"Original prompt: {sample.original_prompt}"})
        if extra_payload is not None:
            content.append({"type": "text", "text": json.dumps(extra_payload, ensure_ascii=False)})
        for frame in frames:
            content.append({"type": "image", "image": frame})
        return [{"role": "user", "content": content}]

    def _messages_text_only(self, instruction: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
        ]
        return [{"role": "user", "content": content}]

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _compute_penalty_score(self, penalty_analysis: Dict[str, Any]) -> float:
        camera_motion_dominant = bool(penalty_analysis.get("camera_motion_dominant", False))
        stylized_rendering = bool(penalty_analysis.get("stylized_rendering", False))
        static_aftermath = bool(penalty_analysis.get("static_aftermath", False))
        showcase_without_interaction = bool(penalty_analysis.get("showcase_without_interaction", False))
        return self._clamp01(
            0.35 * int(camera_motion_dominant)
            + 0.25 * int(stylized_rendering)
            + 0.25 * int(static_aftermath)
            + 0.15 * int(showcase_without_interaction)
        )

    def _compute_physics_richness(
        self,
        score_breakdown: Dict[str, Any],
        fallback_physics_richness: float = 0.0,
    ) -> float:
        try:
            entity_interaction_score = float(score_breakdown["entity_interaction_score"])
            force_outcome_score = float(score_breakdown["force_outcome_score"])
            causal_clarity_score = float(score_breakdown["causal_clarity_score"])
            penalty_score = float(score_breakdown["penalty_score"])
        except (KeyError, TypeError, ValueError):
            return self._clamp01(fallback_physics_richness)

        base_score = (entity_interaction_score + force_outcome_score + causal_clarity_score) / 3.0
        return self._clamp01(base_score - penalty_score)

    def _step_instruction(self, step_idx: int, output_desc: str) -> str:
        return (
            f"{self.template}\n\n"
            f"Execute ONLY Step {step_idx}. "
            f"Output JSON only. {output_desc}"
        )

    def run_step1_parse(self, sample: SampleRecord) -> Dict[str, Any]:
        instruction = self._step_instruction(
            1,
            "Return JSON with key 'parse' containing entities(list), actions(list), forces(list), outcomes(list). Do not speculate beyond prompt+frames.",
        )
        out = self._generate(self._messages_with_frames(instruction, sample))
        parsed = self._extract_json(out)
        return parsed.get("parse", {})

    def run_step2_vision_check(self, sample: SampleRecord, parse_obj: Dict[str, Any]) -> Dict[str, Any]:
        instruction = self._step_instruction(
            2,
            "Given current parse, remove hallucinations and add clearly visible missing items. Return JSON with key 'parse'.",
        )
        out = self._generate(
            self._messages_with_frames(instruction, sample, extra_payload={"current_parse": parse_obj})
        )
        parsed = self._extract_json(out)
        return parsed.get("parse", parse_obj)

    def run_step3_reason(self, sample: SampleRecord, parse_obj: Dict[str, Any]) -> str:
        instruction = self._step_instruction(
            3,
            "Using parse and frame evidence, explain concise causal physics interactions and outcomes. Return JSON with key 'reason'.",
        )
        payload = {"parse": parse_obj}
        out = self._generate(self._messages_with_frames(instruction, sample, extra_payload=payload))
        parsed = self._extract_json(out)
        return str(parsed.get("reason", ""))

    def run_step4_score(self, sample: SampleRecord, parse_obj: Dict[str, Any], reason: str) -> Dict[str, Any]:
        instruction = self._step_instruction(
            4,
            (
                "Use BOTH original prompt text and sampled video frames to score physics quality. "
                "Return JSON with keys: 'penalty_analysis' and 'score_breakdown'. "
                "'penalty_analysis' must include booleans: camera_motion_dominant, stylized_rendering, "
                "static_aftermath, showcase_without_interaction; plus penalty_keywords (list, <=5 short phrases). "
                "Judge camera-motion dominance from global viewpoint shifts vs localized physical interaction. "
                "Judge stylized rendering from prompt/style cues (cartoon/CGI/rendered) and frame realism. "
                "Judge static aftermath from frames showing mostly final state with little process visibility. "
                "Judge showcase_without_interaction when arrangement/display dominates over active interaction. "
                "'score_breakdown' must include entity_interaction_score, force_outcome_score, causal_clarity_score in [0,1]. "
                "Do not provide long explanations."
            ),
        )
        payload = {"parse": parse_obj, "reason": reason}
        out = self._generate(self._messages_with_frames(instruction, sample, extra_payload=payload))
        parsed = self._extract_json(out)
        penalty_analysis = parsed.get("penalty_analysis", {})
        if not isinstance(penalty_analysis, dict):
            penalty_analysis = {}

        penalty_keywords = penalty_analysis.get("penalty_keywords", [])
        if not isinstance(penalty_keywords, list):
            penalty_keywords = []
        penalty_analysis["penalty_keywords"] = [str(x) for x in penalty_keywords[:5]]

        score_breakdown = parsed.get("score_breakdown", {})
        if not isinstance(score_breakdown, dict):
            score_breakdown = {}

        penalty_score = self._compute_penalty_score(penalty_analysis)
        score_breakdown["penalty_score"] = penalty_score

        physics_richness = self._compute_physics_richness(
            score_breakdown=score_breakdown,
            fallback_physics_richness=float(parsed.get("physics_richness", 0.0)),
        )
        return {
            "penalty_analysis": penalty_analysis,
            "score_breakdown": score_breakdown,
            "physics_richness": physics_richness,
        }

    def run_step5_extend(self, sample: SampleRecord, parse_obj: Dict[str, Any], reason: str) -> str:
        instruction = self._step_instruction(
            5,
            "Extend original prompt with causal physical details based on reason and frame evidence. Do not add new entities/forces/sensory descriptions. <=100 words. Return JSON with key 'extended'.",
        )
        payload = {"parse": parse_obj, "reason": reason}
        out = self._generate(self._messages_with_frames(instruction, sample, extra_payload=payload))
        parsed = self._extract_json(out)
        return str(parsed.get("extended", ""))

    def run_one(self, sample: SampleRecord) -> CoTResult:
        messages = self._messages_with_frames(self.template + "\nReturn strict JSON now.", sample)
        output_text = self._generate(messages)
        parsed = self._extract_json(output_text)

        parse_obj = ParsedElements(**parsed.get("parse", {}))
        penalty_analysis = parsed.get("penalty_analysis", {})
        if not isinstance(penalty_analysis, dict):
            penalty_analysis = {}

        penalty_keywords = penalty_analysis.get("penalty_keywords", [])
        if not isinstance(penalty_keywords, list):
            penalty_keywords = []
        penalty_analysis["penalty_keywords"] = [str(x) for x in penalty_keywords[:5]]

        score_breakdown = parsed.get("score_breakdown", {})
        if not isinstance(score_breakdown, dict):
            score_breakdown = {}
        score_breakdown["penalty_score"] = self._compute_penalty_score(penalty_analysis)
        physics_richness = self._compute_physics_richness(
            score_breakdown=score_breakdown,
            fallback_physics_richness=float(parsed.get("physics_richness", 0.0)),
        )

        return CoTResult(
            original=parsed.get("original", sample.original_prompt),
            parse=parse_obj,
            reason=parsed.get("reason", ""),
            extended=parsed.get("extended", ""),
            physics_richness=physics_richness,
            penalty_analysis=penalty_analysis,
            score_breakdown=score_breakdown,
            physics_label=parsed.get("physics_label", None),
        )
