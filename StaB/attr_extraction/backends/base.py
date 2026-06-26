"""Shared extractor interface."""

from abc import ABC, abstractmethod

from PIL import Image


class AttributeExtractor(ABC):
    @abstractmethod
    def extract(self, image: Image.Image) -> str:
        """Return raw tags for one image."""
