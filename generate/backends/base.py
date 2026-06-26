from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, seed: int) -> Image.Image:
        ...
