from __future__ import annotations

from pathlib import Path

from .base import ImageGenerator


def build_generator(model: str, ckpt_dir: Path, device: str) -> ImageGenerator:
    if model == "flux":
        from .flux import FluxGenerator
        return FluxGenerator(ckpt_dir, device)
    if model == "sd3":
        from .sd3 import SD3Generator
        return SD3Generator(ckpt_dir, device)
    raise ValueError(f"Unknown model: {model}")
