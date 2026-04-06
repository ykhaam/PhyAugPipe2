"""PhyAugPipe2 minimal implementation for CoT-based data filtering."""

from importlib import import_module

__all__ = ["CoTFilteringPipeline", "PipelineConfig"]


def __getattr__(name: str):
    # Lazy import to avoid pulling heavy vision stack (torch/torchvision/qwen_vl_utils)
    # for lightweight metadata utilities.
    if name in {"CoTFilteringPipeline", "PipelineConfig"}:
        mod = import_module(".pipeline", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
