"""Extraction workflow orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from attr_extraction.backends import AttributeExtractor, build_extractor
from attr_extraction.config import MAX_ATTEMPTS, SAVE_EVERY, parse_args
from attr_extraction.metadata import MetadataStore, normalize_attributes


def extract_with_retries(
    extractor: AttributeExtractor,
    image_path: Path,
    sample_id: str,
) -> Optional[str]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with Image.open(image_path) as image:
                raw_tags = extractor.extract(image)
            attributes = normalize_attributes(raw_tags)
            if not attributes:
                raise RuntimeError("model returned no tags")
            return attributes
        except Exception as error:
            print(
                f"\n[warning] {sample_id}: attempt {attempt}/{MAX_ATTEMPTS} "
                f"failed: {error}",
                file=sys.stderr,
            )
    return None


def main() -> int:
    settings = parse_args()
    metadata = MetadataStore(
        dataset_root=settings.dataset_root,
        attribute_column=settings.attribute_column,
    )

    selected = metadata.selected_indices()
    pending = metadata.pending_indices()
    print(f"[dataset] {settings.dataset_root}")
    print(f"[metadata] {metadata.path}")
    print(f"[selection] {settings.pct_column}: {len(selected)} samples")
    print(f"[resume] completed={len(selected) - len(pending)}, pending={len(pending)}")

    if not pending:
        metadata.save()
        print("[done] No pending samples")
        return 0

    extractor = build_extractor(settings)
    completed = 0
    failed = 0
    attempts_since_save = 0

    try:
        progress = tqdm(
            pending,
            desc=f"{settings.vlm}:{settings.pct_column}",
            unit="image",
            dynamic_ncols=True,
        )
        for index in progress:
            sample_id = metadata.sample_id(index)
            try:
                attributes = extract_with_retries(
                    extractor,
                    metadata.image_path(index),
                    sample_id,
                )
            except Exception as error:
                print(f"\n[warning] {sample_id}: {error}", file=sys.stderr)
                attributes = None

            if attributes:
                metadata.set_attributes(index, attributes)
                completed += 1
            else:
                metadata.clear_attributes(index)
                failed += 1

            attempts_since_save += 1
            progress.set_postfix(completed=completed, failed=failed)
            if attempts_since_save == SAVE_EVERY:
                metadata.save()
                attempts_since_save = 0
    except KeyboardInterrupt:
        print("\n[interrupt] Saving current progress before exit", file=sys.stderr)
        metadata.save()
        return 130
    finally:
        if attempts_since_save:
            metadata.save()

    print(
        f"[done] column={settings.attribute_column}, "
        f"newly_completed={completed}, failed={failed}, "
        f"metadata={metadata.path}"
    )
    return 0
