"""Configuration and command-line parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DTYPE_CHOICES = ("auto", "float16", "bfloat16", "float32")

IMAGE_SIZE = 384
MAX_NEW_TOKENS = 20
SAVE_EVERY = 5
MAX_ATTEMPTS = 3

LLAVA_MODEL_ID = "llava-hf/llava-v1.6-vicuna-13b-hf"
LLAVA_PROMPT = (
    "Describe all the main visual attributes (tags) of the image, using only "
    "single words, and separate them with commas"
)

TAG2TEXT_REPOSITORY = "https://github.com/xinyu1205/recognize-anything.git"
TAG2TEXT_WEIGHT_URL = (
    "https://huggingface.co/spaces/xinyu1205/"
    "Recognize_Anything-Tag2Text/resolve/main/"
    "tag2text_swin_14m.pth?download=true"
)
TAG2TEXT_WEIGHT_SIZE = 4_478_705_095
TAG2TEXT_DELETE_TAG_INDEX = [127, 2961, 3351, 3265, 3338, 3355, 3359]
TAG2TEXT_RUNTIME_PACKAGES = (
    "timm>=0.4.12",
    "transformers>=4.25.1",
    "fairscale==0.4.4",
    "scipy",
    "Pillow",
)


@dataclass(frozen=True)
class Settings:
    dataset: str
    dataset_path: Path
    vlm: str
    dtype: str
    device: str
    llava_checkpoint: Path
    tag2text_root: Path
    tag2text_checkpoint: Path
    tag2text_deps: Path
    tag2text_threshold: float

    @property
    def dataset_root(self) -> Path:
        return (self.dataset_path / self.dataset).resolve()

    @property
    def attribute_column(self) -> str:
        return f"attrs_{self.vlm}"


def parse_args() -> Settings:
    stab_root = Path(__file__).resolve().parents[1]
    project_root = stab_root.parent

    parser = argparse.ArgumentParser(
        description="Extract image attributes into an existing metadata.csv"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. age")
    parser.add_argument(
        "--dataset_path",
        "--dataset-path",
        dest="dataset_path",
        required=True,
        type=Path,
        help="Parent directory containing the dataset",
    )
    parser.add_argument("--vlm", required=True, choices=("tag2text", "llava"))
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=DTYPE_CHOICES,
        help="LLaVA torch dtype (default: model's original dtype)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device inside the visible CUDA devices",
    )
    parser.add_argument(
        "--llava_checkpoint",
        "--llava-checkpoint",
        dest="llava_checkpoint",
        type=Path,
        default=project_root / "ckpts" / "llava-v1.6-vicuna-13b-hf",
    )
    parser.add_argument(
        "--tag2text_root",
        "--tag2text-root",
        dest="tag2text_root",
        type=Path,
        default=stab_root / "Tag2Text",
    )
    parser.add_argument(
        "--tag2text_checkpoint",
        "--tag2text-checkpoint",
        dest="tag2text_checkpoint",
        type=Path,
        default=project_root / "ckpts" / "tag2text" / "tag2text_swin_14m.pth",
    )
    parser.add_argument(
        "--tag2text_deps",
        "--tag2text-deps",
        dest="tag2text_deps",
        type=Path,
        default=stab_root / ".deps",
    )
    parser.add_argument(
        "--tag2text_threshold",
        "--tag2text-threshold",
        dest="tag2text_threshold",
        type=float,
        default=0.68,
        help="Official Tag2Text tagging threshold",
    )

    args = parser.parse_args()
    return Settings(**vars(args))
