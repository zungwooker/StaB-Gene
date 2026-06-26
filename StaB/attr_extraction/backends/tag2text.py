"""Official Tag2Text setup and tag-only inference backend."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from attr_extraction.backends.base import AttributeExtractor
from attr_extraction.config import (
    IMAGE_SIZE,
    TAG2TEXT_DELETE_TAG_INDEX,
    TAG2TEXT_REPOSITORY,
    TAG2TEXT_RUNTIME_PACKAGES,
    TAG2TEXT_WEIGHT_SIZE,
    TAG2TEXT_WEIGHT_URL,
)


class Tag2TextInstaller:
    """Prepare official source, local dependencies, and pretrained weights."""

    def __init__(
        self,
        source_root: Path,
        checkpoint: Path,
        deps_root: Path,
    ) -> None:
        self.source_root = source_root.resolve()
        self.checkpoint = checkpoint.resolve()
        self.deps_root = deps_root.resolve()

    def prepare(self) -> None:
        self._prepare_source()
        self._prepare_dependencies()
        self._prepare_checkpoint()

    def _prepare_source(self) -> None:
        if (self.source_root / "ram").is_dir():
            return

        self.source_root.parent.mkdir(parents=True, exist_ok=True)
        project_root = self.source_root.parents[1]
        if (project_root / ".git").exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(project_root),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--",
                    "StaB/Tag2Text",
                ],
                check=False,
            )

        if not (self.source_root / "ram").is_dir():
            print(f"[Tag2Text] Cloning official source to {self.source_root}")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    TAG2TEXT_REPOSITORY,
                    str(self.source_root),
                ],
                check=True,
            )

    def _prepare_dependencies(self) -> None:
        self._prepend_python_path(self.deps_root)
        self._prepend_python_path(self.source_root)
        if self._official_imports_work():
            return

        print(f"[Tag2Text] Installing runtime dependencies to {self.deps_root}")
        self.deps_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            "--target",
            str(self.deps_root),
            *TAG2TEXT_RUNTIME_PACKAGES,
        ]
        subprocess.run(command, check=True)
        importlib.invalidate_caches()
        if not self._official_imports_work():
            raise RuntimeError("Tag2Text dependencies were installed but import failed")

    @staticmethod
    def _prepend_python_path(path: Path) -> None:
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    @staticmethod
    def _official_imports_work() -> bool:
        try:
            importlib.import_module("ram")
            importlib.import_module("ram.models")
            return True
        except ImportError:
            return False

    def _prepare_checkpoint(self) -> None:
        if self._checkpoint_is_complete():
            return

        if self.checkpoint.exists():
            print("[Tag2Text] Removing incomplete checkpoint")
            self.checkpoint.unlink()

        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = self.checkpoint.with_suffix(self.checkpoint.suffix + ".part")
        if partial.exists():
            partial.unlink()

        for attempt in range(1, 4):
            try:
                self._download_checkpoint(partial)
                if partial.stat().st_size != TAG2TEXT_WEIGHT_SIZE:
                    raise RuntimeError(
                        "checkpoint size mismatch: "
                        f"expected {TAG2TEXT_WEIGHT_SIZE}, "
                        f"got {partial.stat().st_size}"
                    )
                os.replace(partial, self.checkpoint)
                return
            except Exception:
                if partial.exists():
                    partial.unlink()
                if attempt == 3:
                    raise
                print(f"[Tag2Text] Download failed; retrying ({attempt}/3)")

    def _checkpoint_is_complete(self) -> bool:
        return (
            self.checkpoint.is_file()
            and self.checkpoint.stat().st_size == TAG2TEXT_WEIGHT_SIZE
        )

    @staticmethod
    def _download_checkpoint(destination: Path) -> None:
        print(f"[Tag2Text] Downloading checkpoint to {destination.parent}")
        request = urllib.request.Request(
            TAG2TEXT_WEIGHT_URL,
            headers={"User-Agent": "StaB/Tag2Text"},
        )
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length", 0))
            with destination.open("wb") as output:
                with tqdm(
                    total=total or TAG2TEXT_WEIGHT_SIZE,
                    unit="B",
                    unit_scale=True,
                    desc="Tag2Text weight",
                    dynamic_ncols=True,
                ) as progress:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        progress.update(len(chunk))


class Tag2TextExtractor(AttributeExtractor):
    """Run only Tag2Text's image-tagging branch, never its caption decoder."""

    def __init__(
        self,
        source_root: Path,
        checkpoint: Path,
        deps_root: Path,
        threshold: float,
        device_name: str,
    ) -> None:
        installer = Tag2TextInstaller(source_root, checkpoint, deps_root)
        installer.prepare()

        import torch
        from ram import get_transform
        from ram.models import tag2text

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"{device_name} was requested, but CUDA is not available"
            )

        self.torch = torch
        self.device = torch.device(device_name)
        self.transform = get_transform(image_size=IMAGE_SIZE)

        print(f"[Tag2Text] Loading checkpoint from {checkpoint.resolve()}")
        self.model = tag2text(
            pretrained=str(checkpoint.resolve()),
            image_size=IMAGE_SIZE,
            vit="swin_b",
            threshold=threshold,
            delete_tag_index=TAG2TEXT_DELETE_TAG_INDEX,
        )
        self.model.eval().to(self.device)

    def extract(self, image: Image.Image) -> str:
        tensor = self.transform(
            image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        ).unsqueeze(0).to(self.device)

        with self.torch.inference_mode():
            selected = self._predict_selected_tags(tensor)

        indices = selected[0].nonzero(as_tuple=False).flatten().cpu().tolist()
        return " | ".join(str(self.model.tag_list[index]) for index in indices)

    def _predict_selected_tags(self, tensor: Any) -> Any:
        image_embeds = self.model.visual_encoder(tensor)
        image_atts = self.torch.ones(
            image_embeds.size()[:-1],
            dtype=self.torch.long,
            device=self.device,
        )
        label_embed = self.model.label_embed.weight.unsqueeze(0).repeat(
            image_embeds.shape[0], 1, 1
        )
        tagging_output = self.model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )
        logits = self.model.fc(tagging_output[0])
        selected = self.torch.sigmoid(logits) > self.model.class_threshold.to(
            self.device
        )
        selected[:, self.model.delete_tag_index] = False
        return selected
