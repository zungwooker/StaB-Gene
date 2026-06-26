from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass(frozen=True)
class PromptsConfig:
    dataset: str
    classes: Dict[str, str]


def load_prompts(path: Path) -> PromptsConfig:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PromptsConfig(
        dataset=data["dataset"],
        classes=data["classes"],
    )
