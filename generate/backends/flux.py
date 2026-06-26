from __future__ import annotations

from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image

from .base import ImageGenerator

MODEL_ID = "black-forest-labs/FLUX.1-dev"


class FluxGenerator(ImageGenerator):
    def __init__(self, ckpt_dir: Path, device: str) -> None:
        self.pipeline = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=ckpt_dir / "flux",
        )
        self.pipeline.enable_model_cpu_offload()

    def generate(self, prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator().manual_seed(seed)
        result = self.pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=50,
            guidance_scale=3.5,
        )
        return result.images[0]
