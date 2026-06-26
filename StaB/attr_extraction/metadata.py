"""Metadata loading, selection, path lookup, and safe saving."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, List

import pandas as pd


def normalize_attributes(raw_output: Any) -> str:
    """Convert comma/pipe-delimited model output to a stable tag string."""
    if raw_output is None:
        return ""

    if isinstance(raw_output, (list, tuple)):
        raw_text = ",".join(str(item) for item in raw_output)
    else:
        raw_text = str(raw_output)

    tags: List[str] = []
    seen = set()
    for piece in re.split(r"[,|]", raw_text):
        tag = re.sub(r"\s+", " ", piece).strip()
        tag = tag.strip(" \t\r\n\"'`[](){}").rstrip(".;:")
        tag = tag.lower().strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return ", ".join(tags)


class MetadataStore:
    """Own the in-memory metadata table and its atomic persistence."""

    def __init__(
        self,
        dataset_root: Path,
        attribute_column: str,
    ) -> None:
        self.dataset_root = dataset_root
        self.attribute_column = attribute_column
        self.path = dataset_root / "metadata.csv"

        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {dataset_root}"
            )
        if not self.path.is_file():
            raise FileNotFoundError(f"metadata.csv does not exist: {self.path}")

        self.frame = pd.read_csv(self.path, keep_default_na=True)
        self._validate()
        if attribute_column not in self.frame.columns:
            self.frame[attribute_column] = ""

    def _validate(self) -> None:
        required = {"sample_id", "class"}
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise RuntimeError(
                f"{self.path} is missing required columns: {', '.join(missing)}"
            )

        duplicated = self.frame["sample_id"].duplicated(keep=False)
        if duplicated.any():
            examples = self.frame.loc[duplicated, "sample_id"].head(5).tolist()
            raise RuntimeError(
                "metadata.csv contains duplicate sample_id rows: "
                + ", ".join(map(str, examples))
            )

    def selected_indices(self) -> List[int]:
        return self.frame.index.tolist()

    def pending_indices(self) -> List[int]:
        return [
            index
            for index in self.selected_indices()
            if not self._has_attribute(index)
        ]

    def _has_attribute(self, index: int) -> bool:
        value = self.frame.at[index, self.attribute_column]
        if pd.isna(value):
            return False
        return bool(str(value).strip())

    def sample_id(self, index: int) -> str:
        return str(self.frame.at[index, "sample_id"])

    def image_path(self, index: int) -> Path:
        row = self.frame.loc[index]
        expected = (
            self.dataset_root
            / str(row["class"])
            / str(row["sample_id"])
        )
        if expected.is_file():
            return expected

        matches = list(self.dataset_root.glob(f"*/{row['sample_id']}"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"Image not found for {row['sample_id']}: {expected}"
            )
        raise RuntimeError(
            f"Multiple images found for {row['sample_id']}"
        )

    def set_attributes(self, index: int, attributes: str) -> None:
        self.frame.at[index, self.attribute_column] = attributes

    def clear_attributes(self, index: int) -> None:
        self.frame.at[index, self.attribute_column] = ""

    def save(self) -> None:
        temporary = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            self.frame.to_csv(temporary, index=False)
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
