from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import os

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

    def _messages_with_frames(self, instruction: str, sample: SampleRecord) -> list[dict[str, Any]]:
        frames = sample_video_frames(sample.video_path, num_frames=self.config.num_frames)
        content = [{"type": "text", "text": instruction}]
        content.append({"type": "text", "text": f"Original prompt: {sample.original_prompt}"})
        for frame in frames:
            content.append({"type": "image", "image": frame})
        return [{"role": "user", "content": content}]

    def _messages_text_only(self, instruction: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
        ]
        return [{"role": "user", "content": content}]

    def run_step1_parse(self, sample: SampleRecord) -> Dict[str, Any]:
        instruction = (
            "Step1 Element Parsing only. Return JSON with key 'parse' containing: "
            "entities(list), actions(list), forces(list), outcomes(list). "
            "Do not speculate beyond prompt+frames."
        )
        out = self._generate(self._messages_with_frames(instruction, sample))
        parsed = self._extract_json(out)
        return parsed.get("parse", {})

    def run_step2_vision_check(self, sample: SampleRecord, parse_obj: Dict[str, Any]) -> Dict[str, Any]:
        instruction = (
            "Step2 Vision Checking only. Given original prompt, frames, and current parse, "
            "remove hallucinations and add clearly visible missing items. "
            "Return JSON with key 'parse'."
        )
        base_messages = self._messages_with_frames(instruction, sample)
        base_messages[0]["content"].append(
            {"type": "text", "text": f"Current parse: {json.dumps(parse_obj, ensure_ascii=False)}"}
        )
        out = self._generate(base_messages)
        parsed = self._extract_json(out)
        return parsed.get("parse", parse_obj)

    def run_step3_reason(self, sample: SampleRecord, parse_obj: Dict[str, Any]) -> str:
        instruction = (
            "Step3 Physics Reasoning only. Using original prompt and parse, explain concise causal "
            "physics interactions and outcomes. Return JSON with key 'reason'."
        )
        payload = {"original": sample.original_prompt, "parse": parse_obj}
        out = self._generate(self._messages_text_only(instruction, payload))
        parsed = self._extract_json(out)
        return str(parsed.get("reason", ""))

    def run_step4_score(self, parse_obj: Dict[str, Any], reason: str) -> float:
        instruction = (
            "Step4 Data Scoring only. Score physics_richness in [0,1] using entity interactions, explicit "
            "forces/outcomes, causal clarity, and penalties (camera motion/stylization/static aftermath). "
            "Return JSON with key 'physics_richness'."
        )
        payload = {"parse": parse_obj, "reason": reason}
        out = self._generate(self._messages_text_only(instruction, payload))
        parsed = self._extract_json(out)
        return float(parsed.get("physics_richness", 0.0))

    def run_step5_extend(self, sample: SampleRecord, parse_obj: Dict[str, Any], reason: str) -> str:
        instruction = (
            "Step5 Prompt Extending only. Extend original prompt with causal physical details based on reason. "
            "Do not add new entities/forces/sensory descriptions. <=100 words. Return JSON with key 'extended'."
        )
        payload = {"original": sample.original_prompt, "parse": parse_obj, "reason": reason}
        out = self._generate(self._messages_text_only(instruction, payload))
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
