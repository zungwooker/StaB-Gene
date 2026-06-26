from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from .base import ImageGenerator

MODEL_ID = "stabilityai/stable-diffusion-3.5-large"


class SD3Generator(ImageGenerator):
    def __init__(self, ckpt_dir: Path, device: str) -> None:
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=ckpt_dir / "sd3",
        )
        self.pipeline.enable_model_cpu_offload()

    def generate(self, prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator().manual_seed(seed)
        result = self.pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=40,
            guidance_scale=4.5,
        )
        return result.images[0]
