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

    def run_step4_score(self, sample: SampleRecord, parse_obj: Dict[str, Any], reason: str) -> float:
        instruction = self._step_instruction(
            4,
            "Score physics_richness in [0,1] using parse+reason and frame evidence, including camera motion/stylization/static-aftermath penalties. Return JSON with key 'physics_richness'.",
        )
        payload = {"parse": parse_obj, "reason": reason}
        out = self._generate(self._messages_with_frames(instruction, sample, extra_payload=payload))
        parsed = self._extract_json(out)
        return float(parsed.get("physics_richness", 0.0))

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
        return CoTResult(
            original=parsed.get("original", sample.original_prompt),
            parse=parse_obj,
            reason=parsed.get("reason", ""),
            extended=parsed.get("extended", ""),
            physics_richness=float(parsed.get("physics_richness", 0.0)),
            physics_label=parsed.get("physics_label", None),
        )
