#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import load_prompts
from backends import build_generator

IMAGE_SIZE = 256


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate T2I images for bias detection")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument(
        "--output_path",
        "--output-path",
        dest="output_path",
        required=True,
        type=Path,
    )
    parser.add_argument("--model", required=True, choices=("flux", "sd3"))
    parser.add_argument("--n_images", "--n-images", dest="n_images", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ckpt_dir",
        "--ckpt-dir",
        dest="ckpt_dir",
        type=Path,
        default=project_root / "ckpts",
    )
    return parser.parse_args()


def save_image(image: Image.Image, path: Path) -> None:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        resized.save(tmp, format="JPEG", quality=95)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_metadata(dataset_root: Path, records: list) -> Path:
    metadata_path = dataset_root / "metadata.csv"
    tmp = dataset_root / f".metadata.{uuid.uuid4().hex}.csv"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "class"])
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, metadata_path)
    return metadata_path


def main() -> int:
    args = parse_args()
    prompts_config = load_prompts(args.prompts)
    dataset_root = (args.output_path / prompts_config.dataset).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    generator = build_generator(args.model, args.ckpt_dir, args.device)

    metadata_records = []
    for class_name, prompt in prompts_config.classes.items():
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{class_name}] prompt: {prompt}")

        for i in tqdm(range(args.n_images), desc=class_name, unit="image"):
            sample_id = f"{class_name}_{i:05d}.jpg"
            image_path = class_dir / sample_id
            if not image_path.exists():
                image = generator.generate(prompt, seed=args.seed + i)
                save_image(image, image_path)
            metadata_records.append({"sample_id": sample_id, "class": class_name})

    metadata_path = write_metadata(dataset_root, metadata_records)
    print(f"[done] {len(metadata_records)} images, metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
