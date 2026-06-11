from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from wildlife_reid.data.catalog import ImageRecord, group_by_identity
from wildlife_reid.transforms import apply_augmentation, apply_preprocess


class IdentityDataset(Dataset):
    """Image + integer identity label for ArcFace training."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        backbone: str,
        image_size: int | None = None,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.backbone = backbone
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.augment:
            image = apply_augmentation(image)
        tensor = apply_preprocess(image, self.backbone, self.image_size)
        return tensor, label


def build_identity_samples(
    records: list[ImageRecord],
    max_per_identity: int | None = None,
    seed: int = 71,
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """Map ImageRecords to (path, class_index) samples."""
    grouped = group_by_identity(records)
    identities = sorted(grouped)
    identity_to_idx = {identity: idx for idx, identity in enumerate(identities)}

    rng = random.Random(seed)
    samples: list[tuple[Path, int]] = []
    for identity in identities:
        images = grouped[identity]
        if max_per_identity is not None:
            images = images[:max_per_identity]
        label = identity_to_idx[identity]
        for record in images:
            samples.append((record.path, label))

    rng.shuffle(samples)
    return samples, identity_to_idx


def split_identity_samples(
    samples: list[tuple[Path, int]],
    val_ratio: float,
    seed: int = 71,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not samples:
        return [], []
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_size:], shuffled[:val_size]
