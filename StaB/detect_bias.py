#!/usr/bin/env python3
"""Rank class-dependent visual attributes using mutual information.

For every class, this script computes:

    dependency(c, tag) = p(c | tag) - p(c)

and the mutual information between the class variable C and the binary
presence variable for each tag. Only positively dependent tags are written.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd


VLM_CHOICES = ("tag2text", "llava")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-class attribute dependency and mutual information"
        )
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
    parser.add_argument("--vlm", required=True, choices=VLM_CHOICES)
    return parser.parse_args()


def normalize_tag_list(value: Any) -> List[str]:
    """Parse comma/pipe-separated tags and remove duplicates per sample."""
    if pd.isna(value):
        return []

    tags: List[str] = []
    seen = set()
    for piece in re.split(r"[,|]", str(value)):
        tag = re.sub(r"\s+", " ", piece).strip().lower()
        tag = tag.strip(" \t\r\n\"'`[](){}").rstrip(".;:")
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def entropy(probabilities: pd.DataFrame) -> pd.Series:
    """Compute entropy for every row, treating 0 log(0) as zero."""
    safe = probabilities.where(probabilities > 0)
    return -(safe * np.log2(safe)).sum(axis=1, skipna=True)


def validate_metadata(
    frame: pd.DataFrame,
    metadata_path: Path,
    attribute_column: str,
) -> None:
    required = {"sample_id", "class", attribute_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{metadata_path} is missing required columns: {', '.join(missing)}"
        )


def select_complete_samples(
    frame: pd.DataFrame,
    attribute_column: str,
) -> pd.DataFrame:
    selected = frame[["sample_id", "class", attribute_column]].copy()

    if selected.empty:
        raise RuntimeError("No samples found in metadata")

    has_attributes = (
        selected[attribute_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    missing_count = int((~has_attributes).sum())
    if missing_count:
        raise RuntimeError(
            f"{missing_count}/{len(selected)} selected samples have empty "
            f"{attribute_column}. Complete attribute extraction before "
            "computing dependency."
        )

    selected["tags"] = selected[attribute_column].map(normalize_tag_list)
    empty_after_parsing = selected["tags"].map(len).eq(0)
    if empty_after_parsing.any():
        raise RuntimeError(
            f"{int(empty_after_parsing.sum())} selected samples contain no "
            "valid tags after parsing"
        )
    return selected


def compute_tag_statistics(samples: pd.DataFrame) -> pd.DataFrame:
    """Compute class-conditional counts, dependency, and tag MI."""
    total_samples = len(samples)
    class_counts = samples["class"].value_counts().sort_index()
    class_probability = class_counts / total_samples
    class_entropy = float(
        -(class_probability * np.log2(class_probability)).sum()
    )

    exploded = (
        samples[["class", "tags"]]
        .explode("tags")
        .rename(columns={"tags": "tag"})
    )
    tag_class_counts = (
        exploded.groupby(["tag", "class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=class_counts.index, fill_value=0)
    )

    tag_counts = tag_class_counts.sum(axis=1)
    not_tag_counts = class_counts - tag_class_counts
    not_tag_totals = total_samples - tag_counts

    probability_class_given_tag = tag_class_counts.div(tag_counts, axis=0)
    probability_class_given_not_tag = not_tag_counts.div(
        not_tag_totals.replace(0, np.nan),
        axis=0,
    )

    probability_tag = tag_counts / total_samples
    entropy_given_tag = entropy(probability_class_given_tag)
    entropy_given_not_tag = entropy(probability_class_given_not_tag)
    conditional_entropy = (
        probability_tag * entropy_given_tag
        + (1 - probability_tag) * entropy_given_not_tag.fillna(0)
    )
    mutual_information = (class_entropy - conditional_entropy).clip(lower=0)

    records = []
    for class_value in class_counts.index:
        dependency = (
            probability_class_given_tag[class_value]
            - class_probability[class_value]
        )
        class_result = pd.DataFrame(
            {
                "class": class_value,
                "tag": tag_class_counts.index,
                "n_c": tag_class_counts[class_value].astype(int).values,
                "p_c_given_tag": probability_class_given_tag[
                    class_value
                ].values,
                "p_c": float(class_probability[class_value]),
                "dependency": dependency.values,
                "MI": mutual_information.values,
            }
        )
        records.append(class_result)

    return pd.concat(records, ignore_index=True)


def class_filename(class_value: Any, vlm: str) -> str:
    class_text = str(class_value)
    if class_text.endswith(".0"):
        class_text = class_text[:-2]
    class_text = re.sub(r"[^A-Za-z0-9_.-]+", "_", class_text)
    return f"class_{class_text}_{vlm}_dependency.csv"


def atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_class_results(
    statistics: pd.DataFrame,
    output_directory: Path,
    vlm: str,
) -> List[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []

    for class_value in sorted(statistics["class"].unique()):
        class_result = statistics.loc[
            (statistics["class"] == class_value)
            & (statistics["dependency"] > 0),
            ["tag", "n_c", "p_c_given_tag", "p_c", "dependency", "MI"],
        ].copy()
        class_result = class_result.sort_values(
            ["MI", "dependency", "tag"],
            ascending=[False, False, True],
            kind="mergesort",
        )

        output_path = output_directory / class_filename(class_value, vlm)
        atomic_write_csv(class_result, output_path)
        outputs.append(output_path)

    return outputs


def main() -> int:
    args = parse_args()
    dataset_root = (args.dataset_path / args.dataset).resolve()
    metadata_path = dataset_root / "metadata.csv"
    attribute_column = f"attrs_{args.vlm}"

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_root}"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv does not exist: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    validate_metadata(metadata, metadata_path, attribute_column)
    samples = select_complete_samples(metadata, attribute_column)
    statistics = compute_tag_statistics(samples)
    output_directory = dataset_root / "bias_results"
    outputs = save_class_results(statistics, output_directory, args.vlm)

    print(
        f"[data] samples={len(samples)}, classes={samples['class'].nunique()}, "
        f"tags={statistics['tag'].nunique()}"
    )
    for output in outputs:
        rows = sum(1 for _ in output.open(encoding="utf-8")) - 1
        print(f"[saved] {output} ({rows} positively dependent tags)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
