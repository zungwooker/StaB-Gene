"""Model backend factory."""

from attr_extraction.backends.base import AttributeExtractor
from attr_extraction.backends.llava import LLaVAExtractor
from attr_extraction.backends.tag2text import Tag2TextExtractor
from attr_extraction.config import Settings


def build_extractor(settings: Settings) -> AttributeExtractor:
    if settings.vlm == "llava":
        return LLaVAExtractor(
            checkpoint=settings.llava_checkpoint,
            dtype_name=settings.dtype,
            device_name=settings.device,
        )

    return Tag2TextExtractor(
        source_root=settings.tag2text_root,
        checkpoint=settings.tag2text_checkpoint,
        deps_root=settings.tag2text_deps,
        threshold=settings.tag2text_threshold,
        device_name=settings.device,
    )
