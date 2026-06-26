"""LLaVA attribute extraction backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from attr_extraction.backends.base import AttributeExtractor
from attr_extraction.config import (
    IMAGE_SIZE,
    LLAVA_MODEL_ID,
    LLAVA_PROMPT,
    MAX_NEW_TOKENS,
)


class LLaVAExtractor(AttributeExtractor):
    def __init__(
        self,
        checkpoint: Path,
        dtype_name: str,
        device_name: str,
    ) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import (
                LlavaNextForConditionalGeneration,
                LlavaNextProcessor,
            )
        except ImportError as error:
            raise RuntimeError(
                "LLaVA dependencies are missing. Install torch, transformers, "
                "huggingface_hub, and sentencepiece."
            ) from error

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"{device_name} was requested, but CUDA is not available"
            )

        checkpoint = checkpoint.resolve()
        self._download_if_needed(checkpoint, snapshot_download)

        dtype: Any = "auto"
        if dtype_name != "auto":
            dtype = getattr(torch, dtype_name)

        print(f"[LLaVA] Loading processor from {checkpoint}")
        self.processor = LlavaNextProcessor.from_pretrained(
            str(checkpoint),
            local_files_only=True,
        )
        print(f"[LLaVA] Loading model on {device_name} with dtype={dtype_name}")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval().to(device_name)

        self.torch = torch
        self.device = torch.device(device_name)
        self.prompt = self._build_prompt()

    @staticmethod
    def _download_if_needed(checkpoint: Path, snapshot_download: Any) -> None:
        marker = checkpoint / ".download_complete"
        if marker.is_file():
            return

        checkpoint.mkdir(parents=True, exist_ok=True)
        print(f"[LLaVA] Downloading {LLAVA_MODEL_ID} to {checkpoint}")
        snapshot_download(repo_id=LLAVA_MODEL_ID, local_dir=str(checkpoint))
        marker.touch()

    def _build_prompt(self) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LLAVA_PROMPT},
                    {"type": "image"},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

    def extract(self, image: Image.Image) -> str:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        generated = output[0, input_length:]
        return self.processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()
